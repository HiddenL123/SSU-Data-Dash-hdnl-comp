"""Full-strength chronological spline / geo / Tensor GAM for list-aware CatBoost.

Preserve the fitted CatBoost-Huber ratio-target model, comparable-sale features,
loss, and tuned settings as the baseline. Each new feature model uses the same
CatBoost configuration with additional spline / geographic predictors.
Tensor GAM corrections use ONLY prior-month, truly out-of-time list-aware
CatBoost residuals from the corresponding model, never off-market residuals.
The original off-market or XGBoost experiment files are not reused or changed.
A separate fingerprinted, resumable cache stores the new experiment.
"""
from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import data
from models.direct_listing import DirectListingModel
from models.spacetime_screen_residuals import SpaceTimeResidual
from spacetime_full_feature_comparison import (
    _feature_spec, _from_cache, _write_frame,
)
from spacetime_full_validation import (
    _frame, _jsonable_spec, _validate_baseline_reference,
)
from spacetime_three_month import (
    _attach_type, _bootstrap_delta, _type_summary, _validate_result, summarize,
)

PIPELINE_VERSION = 2
FEATURE_NAME = "Xpt_spatiotemporal"
BASES = {
    "splines": "spacetime_splines",
    "splines_geo": "spacetime_splines+geo_context",
}
CORRECTIONS = {
    "splines": "splines_tensor_gam",
    "splines_geo": "splines_geo_tensor_gam",
}
COMPARE_KEYS = ["month", "ml_num"]


def _check_list_aware_model(model):
    """Require the original CatBoost Huber ratio model, not the XGBoost snapshot."""
    if not isinstance(model, DirectListingModel) or not model._fitted:
        raise ValueError("Expected a fitted list-aware DirectListingModel")
    if model.target != "ratio":
        raise ValueError("Saved list-aware model must predict the sale/list ratio")
    if (model.modelspecs is None or len(model.modelspecs) != 1 or
        len(model.components) != 1):
        raise ValueError("Expected one fitted CatBoost stage")
    klass, name, params = model.modelspecs[0]
    if (name != FEATURE_NAME or
        model.components[0][0].feature_backend != "catboost"):
        raise ValueError(
            "Expected CatBoost + Xpt_spatiotemporal, not XGBoost. "
            "Load saved_models/catboost_spatiotemporal_900."
        )
    if (params or {}).get("loss_function") != "Huber:delta=0.05":
        raise ValueError(
            "Expected CatBoost Huber:delta=0.05; "
            "do not compare to an RMSE or XGBoost baseline."
        )
    if not hasattr(model.components[0][0].estimator, "save_model"):
        raise ValueError("Fitted native CatBoost model is not serializable")
    return model.modelspecs


def _catboost_fingerprint(sales, reference, specs, models, config):
    """Fingerprint raw rows, baseline predictions and serialized CatBoost weights."""
    digest = hashlib.sha256()
    for frame in (sales, reference):
        digest.update("|".join(frame.columns).encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(frame, index=True)
            .to_numpy(dtype="uint64").tobytes()
        )
    with tempfile.TemporaryDirectory(prefix="list_aware_cat_fingerprint_") as temp:
        for i, model in enumerate(models):
            file = Path(temp) / f"baseline_{i}.cbm"
            model.components[0][0].estimator.save_model(str(file), format="cbm")
            digest.update(file.read_bytes())
    digest.update(json.dumps({
        "spec": _jsonable_spec(specs), **config
    }, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _parent_comparisons(pooled, monthly, predictions, repeats, seed):
    """Calculate the incremental GAM benefit over its own feature model."""
    pooled = pooled.copy()
    monthly = monthly.copy()
    for family, corrected in CORRECTIONS.items():
        a = predictions.loc[
            (predictions.arm == "feature") & (predictions.variant == family)
        ].set_index(COMPARE_KEYS).sort_index()
        b = predictions.loc[
            (predictions.arm == "residual") & (predictions.variant == corrected)
        ].set_index(COMPARE_KEYS).loc[a.index]
        sold = a.sold_price.to_numpy(dtype=float)
        abs_a = 100 * np.abs(a.predicted_price.to_numpy() - sold) / sold
        abs_b = 100 * np.abs(b.predicted_price.to_numpy() - sold) / sold
        row = (pooled.arm == "residual") & (pooled.variant == corrected)
        pooled.loc[row, "parent_model"] = family
        pooled.loc[row, "delta_parent_mdape_pp"] = (
            np.median(abs_b) - np.median(abs_a))
        pooled.loc[row, "delta_parent_mape_pp"] = np.mean(abs_b - abs_a)
        pooled.loc[row, "parent_homes_improved_fraction"] = np.mean(
            abs_b < abs_a)
        ci = _bootstrap_delta(abs_a, abs_b, repeats=repeats, seed=seed)
        for column, value in ci.items():
            pooled.loc[row, column.replace("delta_", "delta_parent_")] = value
        for month in sorted(predictions.month.unique()):
            one_a = a.xs(month, level="month")
            one_b = b.xs(month, level="month")
            true = one_a.sold_price.to_numpy(dtype=float)
            ea = 100 * np.abs(one_a.predicted_price.to_numpy() - true) / true
            eb = 100 * np.abs(one_b.predicted_price.to_numpy() - true) / true
            key = ((monthly.month == month) & (monthly.arm == "residual") &
                   (monthly.variant == corrected))
            monthly.loc[key, "delta_parent_mdape_pp"] = (
                np.median(eb) - np.median(ea))
            monthly.loc[key, "delta_parent_mape_pp"] = np.mean(eb - ea)
    return pooled, monthly


def compare_list_aware_spline_gam(
    baseline_predictions, fitted_models, *, months=None,
    calibration_months=3, calibration_cap=None, residual_strength=0.5,
    bootstrap_repeats=500, seed=42,
    output_dir="outputs/spacetime_list_aware_spline_gam",
    resume=True, listings=None,
):
    """Compare CatBoost Huber ratio model and spline/geo/Tensor GAM add-ons.

    Pass fitted_models=load_model("catboost_spatiotemporal_900"), and
    baseline_predictions=None to generate baseline predictions from the
    trusted saved model (without retraining). Model pairs must be ordered
    off_market/list_aware for the same validation months.
    """
    if not 0 <= residual_strength <= 1:
        raise ValueError("residual_strength must be between 0 and 1")
    if calibration_months < 1 or bootstrap_repeats < 0:
        raise ValueError("Invalid calibration_months or bootstrap_repeats")
    if calibration_cap is not None and calibration_cap < 1:
        raise ValueError("calibration_cap must be positive or None")

    if months is None:
        if baseline_predictions is None:
            raise ValueError("Provide months when baseline_predictions=None")
        months = sorted(baseline_predictions.loc[
            baseline_predictions.track == "list_aware", "month"
        ].astype(str).unique())
    periods = pd.PeriodIndex(months, freq="M")
    if (not len(periods) or periods.has_duplicates or
        not periods.is_monotonic_increasing):
        raise ValueError("Months must be nonempty, unique and chronological")
    if any(p >= pd.Period("2026-06", freq="M") for p in periods):
        raise ValueError("Competition test period has no observed sale labels")
    if len(fitted_models) != len(periods):
        raise ValueError("One saved list-aware model per validation month is required")
    if any(len(pair) != 2 for pair in fitted_models):
        raise ValueError("Fitted models must be pairs: off_market, list_aware")

    reference_models = [pair[1] for pair in fitted_models]
    spec = _check_list_aware_model(reference_models[0])
    if any(_jsonable_spec(_check_list_aware_model(model)) !=
           _jsonable_spec(spec) for model in reference_models[1:]):
        raise ValueError("Saved list-aware model specifications differ")

    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_listed"] = pd.to_datetime(sales.date_listed)
    sales["date_sold"] = pd.to_datetime(sales.date_sold)
    if sales.duplicated(["ml_num", "date_sold"]).any():
        raise ValueError("Repeated sale identifiers prevent paired comparison")
    if (not np.isfinite(sales.list_price.to_numpy(dtype=float)).all() or
        (sales.list_price.to_numpy(dtype=float) <= 0).any()):
        raise ValueError("List-aware requires positive finite listing prices")

    all_months = set(periods) | {
        p - lag for p in periods for lag in range(1, calibration_months + 1)
    }
    periods_by_row = sales.date_sold.dt.to_period("M")
    cohorts = {p: sales.loc[periods_by_row == p] for p in all_months}
    if any(cohorts[p].empty for p in all_months):
        raise ValueError("A validation or calibration sale month has no homes")

    # Replay the original saved CatBoost Huber model to make reference
    # predictions; the existing notebook's 'predictions' are XGBoost and
    # MUST NOT be passed as this benchmark's CatBoost reference.
    if baseline_predictions is None:
        reference_rows = []
        for month, fitted in zip(periods, reference_models):
            held = cohorts[month]
            hidden = data.hide_answers(held, "list_aware")
            reference_rows.append(pd.DataFrame({
                "month": str(month),
                "ml_num": held.ml_num.to_numpy(),
                "track": "list_aware",
                "sold_price": held.sold_price.to_numpy(dtype=float),
                "predicted_price": np.round(
                    np.asarray(fitted.predict(hidden), dtype=float)),
            }))
        baseline_predictions = pd.concat(reference_rows, ignore_index=True)

    settings = {
        "pipeline_version": PIPELINE_VERSION,
        "track": "list_aware", "target": "ratio",
        "months": [str(p) for p in periods],
        "bases": BASES, "corrections": CORRECTIONS,
        "calibration_months": calibration_months,
        "calibration_cap": calibration_cap,
        "residual_strength": residual_strength,
        "bootstrap_repeats": bootstrap_repeats, "seed": seed,
    }
    fingerprint = _catboost_fingerprint(
        sales, baseline_predictions, spec, reference_models, settings)
    manifest = {
        "fingerprint": fingerprint,
        "spec": _jsonable_spec(spec),
        **settings,
    }
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_file = folder / "setup.json"
    had_manifest = manifest_file.is_file()
    if resume and had_manifest:
        if json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                "Saved list-aware experiment has different settings or models; "
                "choose a new output_dir"
            )
    reuse = resume and had_manifest
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    base_cache = {}
    oot_cache = {}
    rows = []

    def base_prediction(family, month):
        key = (family, month)
        if key in base_cache:
            return base_cache[key]
        held = cohorts[month]
        path = folder / "by_month" / str(month) / f"feature__{family}.csv"
        cached = (_from_cache(path, held, month=month, arm="feature",
                              variant=family) if reuse else None)
        if cached is not None:
            print(f"    {month} {family}: cached CatBoost", flush=True)
            base_cache[key] = cached
            return cached
        training = sales.loc[sales.date_sold < month.start_time]
        if training.empty:
            raise ValueError(f"No earlier training sales for {month}")
        params = _feature_spec(spec, BASES[family])
        print(f"    {month} {family}: training Huber ratio-CatBoost on "
              f"{len(training):,} past sales", flush=True)
        estimator = DirectListingModel(
            target="ratio", modelspecs=params
        ).fit(training)
        hidden = data.hide_answers(held, "list_aware")
        pred = np.asarray(estimator.predict(hidden), dtype=float)
        result = _frame(held, month, "feature", family, pred)
        _write_frame(path, result)
        base_cache[key] = result
        return result

    def oot_residual(family, month):
        key = (family, month)
        if key not in oot_cache:
            held = cohorts[month]
            base = base_prediction(family, month)
            aligned = base.set_index("ml_num").loc[held.ml_num.astype(str)]
            observed = held.sold_price.to_numpy(dtype=float)
            pred = aligned.predicted_price.to_numpy(dtype=float)
            if (not np.isfinite(observed).all() or (observed <= 0).any() or
                not np.isfinite(pred).all() or (pred <= 0).any()):
                raise ValueError(f"Invalid OOT sale prices for {family}/{month}")
            hidden = data.hide_answers(held, "list_aware")
            result = pd.DataFrame({
                "date_listed": hidden.date_listed.to_numpy(),
                "latitude": hidden.latitude.to_numpy(dtype=float),
                "longitude": hidden.longitude.to_numpy(dtype=float),
                "log_residual": np.log(observed) - np.log(pred),
            })
            if calibration_cap is not None and len(result) > calibration_cap:
                ix = np.random.default_rng(seed).choice(
                    len(result), size=calibration_cap, replace=False)
                result = result.iloc[np.sort(ix)].reset_index(drop=True)
            oot_cache[key] = result
        return oot_cache[key]

    expected = {("feature", "baseline")} | {
        ("feature", family) for family in BASES
    } | {("residual", name) for name in CORRECTIONS.values()}
    for i, month in enumerate(periods):
        held = cohorts[month]
        hidden = data.hide_answers(held, "list_aware")
        # Original CatBoost Huber snapshot: no baseline refitting.
        original = np.asarray(reference_models[i].predict(hidden), dtype=float)
        original_frame = _frame(held, month, "feature", "baseline", original)
        _validate_baseline_reference(
            baseline_predictions, original_frame, month, track="list_aware"
        )
        rows.append(original_frame)
        print(f"\n{month}: evaluating {len(held):,} list-aware homes", flush=True)
        for family, correction_name in CORRECTIONS.items():
            base = base_prediction(family, month)
            rows.append(base)
            path = (folder / "by_month" / str(month) /
                    f"residual__{correction_name}.csv")
            cached = (_from_cache(path, held, month=month, arm="residual",
                                  variant=correction_name) if reuse else None)
            if cached is not None:
                rows.append(cached)
                print(f"    {month} {correction_name}: cached", flush=True)
                continue

            historical_errors = pd.concat(
                [oot_residual(family, month - lag)
                 for lag in range(calibration_months, 0, -1)],
                ignore_index=True,
            )
            correction = SpaceTimeResidual("tensor_gam", seed=seed).fit(
                historical_errors,
                historical_errors.log_residual.to_numpy(dtype=float),
            ).predict(hidden)
            pred = base.set_index("ml_num").loc[
                held.ml_num.astype(str), "predicted_price"
            ].to_numpy(dtype=float)
            corrected = pred * np.exp(
                residual_strength * np.clip(correction, -0.25, 0.25)
            )
            result = _frame(held, month, "residual", correction_name, corrected)
            _write_frame(path, result)
            rows.append(result)
            print(f"    {month} {correction_name}: "
                  f"{len(historical_errors):,} prior OOT residuals", flush=True)

    output = pd.concat(rows, ignore_index=True)
    _validate_result(output, expected, [str(p) for p in periods])
    pooled, monthly = summarize(
        output, repeats=bootstrap_repeats, seed=seed)
    pooled, monthly = _parent_comparisons(
        pooled, monthly, output, bootstrap_repeats, seed)
    by_type = _type_summary(_attach_type(output, sales))
    _write_frame(folder / "predictions.csv", output)
    _write_frame(folder / "pooled_summary.csv", pooled)
    _write_frame(folder / "monthly_summary.csv", monthly)
    _write_frame(folder / "house_type_summary.csv", by_type)

    print("\nFull-strength list-aware CatBoost Huber / Tensor GAM results:\n")
    print(pooled[[
        "arm", "variant", "homes", "mdape", "mape",
        "delta_mdape_pp", "delta_mape_pp",
        "months_better_mdape",
    ]].to_string(index=False), flush=True)
    return pooled, monthly, by_type, output
