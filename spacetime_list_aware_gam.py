"""Full-strength list-aware XGBoost spline/geo/Tensor GAM benchmark.

Primary reference: the original saved tuned XGBoost list-aware sale/list-price
ratio model. All new spline-feature models and their historical out-of-time
GAM calibration models remain XGBoost with that exact target/specification.

Optional secondary reference: a separately saved list-aware CatBoost Huber
model. It is evaluated on the same homes but NEVER used to fit GAM residuals.
The main pooled and monthly delta_* metrics remain relative to XGBoost.
CatBoost comparisons can be attached later without refitting any XGBoost.

No off-market fits or residuals are reused. All outputs are isolated and
per-month feature/GAM results can be resumed with a fingerprinted manifest.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import data
from models.direct_listing import DirectListingModel
from models.spacetime_screen_residuals import SpaceTimeResidual
from spacetime_full_feature_comparison import (
    _feature_spec, _from_cache, _signature, _write_frame,
)
from spacetime_full_validation import (
    _frame, _jsonable_spec, _validate_baseline_reference,
)
from spacetime_three_month import (
    _attach_type, _bootstrap_delta, _type_summary, _validate_result, summarize,
)

PIPELINE_VERSION = 3
FEATURE_NAME = "Xpt_spatiotemporal__selected_binary"
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
    """Require exact, already-trained ratio-target XGBoost architecture."""
    if not isinstance(model, DirectListingModel) or not model._fitted:
        raise ValueError("Expected a fitted list-aware DirectListingModel")
    if model.target != "ratio":
        raise ValueError("Saved list-aware model must predict the sale/list ratio")
    if (model.modelspecs is None or len(model.modelspecs) != 1 or
        len(model.components) != 1):
        raise ValueError("Expected exactly one list-aware XGBoost stage")
    if (model.modelspecs[0][1] != FEATURE_NAME or
        model.components[0][0].feature_backend != "xgboost"):
        raise ValueError(
            "List-aware model must use the tuned XGBoost selected-binary features"
        )
    return model.modelspecs


def _check_catboost_huber(model):
    """Require a fitted Huber list-aware CatBoost ratio model."""
    if not isinstance(model, DirectListingModel) or not model._fitted:
        raise ValueError("Expected a fitted CatBoost DirectListingModel")
    if model.target != "ratio" or not model.components or not model.modelspecs:
        raise ValueError("Expected a fitted CatBoost sale/list-price ratio model")
    if len(model.modelspecs) != 1 or len(model.components) != 1:
        raise ValueError("CatBoost Huber reference must have exactly one stage")
    component = model.components[0][0]
    params = model.modelspecs[0][2] or {}
    loss = params.get("loss_function")
    if loss is None:
        loss = component.estimator.get_param("loss_function")
    if component.feature_backend != "catboost" or not str(loss).startswith("Huber"):
        raise ValueError(
            "Reference model is not CatBoost with Huber loss. "
            "Check the saved model name and specifications."
        )
    return model.modelspecs


def _catboost_comparisons(pooled, monthly, predictions, repeats, seed):
    """Add direct paired results versus CatBoost without altering XGBoost deltas."""
    pooled, monthly = pooled.copy(), monthly.copy()
    ref = predictions.loc[
        (predictions.arm == "reference") &
        (predictions.variant == "catboost_huber")
    ].set_index(COMPARE_KEYS).sort_index()
    observed = ref.sold_price.to_numpy(dtype=float)
    ref_ape = 100 * np.abs(ref.predicted_price.to_numpy() - observed) / observed
    for (arm, variant), chunk in predictions.groupby(["arm", "variant"]):
        aligned = chunk.set_index(COMPARE_KEYS).loc[ref.index]
        true = aligned.sold_price.to_numpy(dtype=float)
        ape = 100 * np.abs(aligned.predicted_price.to_numpy() - true) / true
        row = (pooled.arm == arm) & (pooled.variant == variant)
        pooled.loc[row, "delta_catboost_mdape_pp"] = (
            np.median(ape) - np.median(ref_ape))
        pooled.loc[row, "delta_catboost_mape_pp"] = np.mean(ape - ref_ape)
        pooled.loc[row, "catboost_homes_improved_fraction"] = (
            np.mean(ape < ref_ape) if variant != "catboost_huber" else 0.0)
        ci = _bootstrap_delta(ref_ape, ape, repeats=repeats, seed=seed)
        for col, value in ci.items():
            pooled.loc[row, col.replace("delta_", "delta_catboost_")] = value
        for month in sorted(predictions.month.unique()):
            m_ref = ref.xs(month, level="month")
            m_one = aligned.xs(month, level="month")
            y = m_ref.sold_price.to_numpy(dtype=float)
            a = 100 * np.abs(m_ref.predicted_price.to_numpy() - y) / y
            b = 100 * np.abs(m_one.predicted_price.to_numpy() - y) / y
            mask = ((monthly.month == month) & (monthly.arm == arm) &
                    (monthly.variant == variant))
            monthly.loc[mask, "delta_catboost_mdape_pp"] = (
                np.median(b) - np.median(a))
            monthly.loc[mask, "delta_catboost_mape_pp"] = np.mean(b - a)
    return pooled, monthly


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
    output_dir="outputs/spacetime_list_aware_spline_gam_xgb",
    resume=True, listings=None,
    catboost_model_name=None,
    catboost_models=None,
):
    """Test spline, geo and Tensor GAM list-aware XGBoost models.

    baseline_predictions and fitted_models are the saved XGBoost validation
    results. Set catboost_model_name to load an OPTIONAL independent Huber
    reference; leave None to skip. Or pass catboost_models explicitly.
    The returned summaries always compare to XGBoost. If CatBoost is available,
    additional delta_catboost_* paired comparisons are attached.
    Every fitted GAM uses out-of-time residuals of its own XGBoost parent.
    """
    if not 0 <= residual_strength <= 1:
        raise ValueError("residual_strength must be between 0 and 1")
    if calibration_months < 1 or bootstrap_repeats < 0:
        raise ValueError("Invalid calibration_months or bootstrap_repeats")
    if calibration_cap is not None and calibration_cap < 1:
        raise ValueError("calibration_cap must be positive or None")

    if months is None:
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
        raise ValueError("Saved list-aware XGBoost specifications differ")

    # CatBoost is a reference only. Never block the XGBoost experiments
    # when that optional saved snapshot was not kept.
    if catboost_models is None and catboost_model_name is not None:
        from save_model import load_model, list_saved_models
        try:
            catboost_models = load_model(catboost_model_name)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Requested CatBoost Huber snapshot {catboost_model_name!r} "
                f"was not saved. Available snapshots: {list_saved_models()}. "
                "Pass catboost_model_name=None to run XGBoost only."
            ) from exc

    cat_models = None
    if catboost_models is not None:
        if len(catboost_models) != len(periods) or any(
            len(pair) != 2 for pair in catboost_models
        ):
            raise ValueError(
                "CatBoost snapshot must contain one off-market/list-aware pair "
                "for each validation month"
            )
        cat_models = [pair[1] for pair in catboost_models]
        cat_spec = _check_catboost_huber(cat_models[0])
        if any(_jsonable_spec(_check_catboost_huber(m)) !=
               _jsonable_spec(cat_spec) for m in cat_models[1:]):
            raise ValueError("Saved CatBoost Huber specs differ across months")

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

    cat_ref = None
    if cat_models is not None:
        cat_frames = []
        for month, cat_model in zip(periods, cat_models):
            held = cohorts[month]
            hidden = data.hide_answers(held, "list_aware")
            cat_frames.append(_frame(
                held, month, "reference", "catboost_huber",
                np.asarray(cat_model.predict(hidden), dtype=float),
            ))
        cat_ref = pd.concat(cat_frames, ignore_index=True)

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
    fingerprint = _signature(
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
            print(f"    {month} {family}: cached XGBoost", flush=True)
            base_cache[key] = cached
            return cached
        training = sales.loc[sales.date_sold < month.start_time]
        if training.empty:
            raise ValueError(f"No earlier training sales for {month}")
        params = _feature_spec(spec, BASES[family])
        print(f"    {month} {family}: training ratio-XGBoost on "
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
    if cat_ref is not None:
        expected.add(("reference", "catboost_huber"))
    for i, month in enumerate(periods):
        held = cohorts[month]
        hidden = data.hide_answers(held, "list_aware")
        # Original snapshot: no retraining or recalibration for the baseline.
        original = np.asarray(reference_models[i].predict(hidden), dtype=float)
        original_frame = _frame(held, month, "feature", "baseline", original)
        _validate_baseline_reference(
            baseline_predictions, original_frame, month, track="list_aware"
        )
        rows.append(original_frame)
        if cat_ref is not None:
            rows.append(cat_ref.loc[cat_ref.month == str(month)])
        print(f"\n{month}: evaluating {len(held):,} list-aware homes "
              "(XGBoost baseline" +
              (", CatBoost Huber reference)" if cat_ref is not None else ")"),
              flush=True)
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
    if cat_ref is not None:
        pooled, monthly = _catboost_comparisons(
            pooled, monthly, output, bootstrap_repeats, seed)
    by_type = _type_summary(_attach_type(output, sales))
    _write_frame(folder / "predictions.csv", output)
    _write_frame(folder / "pooled_summary.csv", pooled)
    _write_frame(folder / "monthly_summary.csv", monthly)
    _write_frame(folder / "house_type_summary.csv", by_type)

    print("\nList-aware XGBoost spline/geo/Tensor GAM results "
          "(delta columns vs original XGBoost):\n")
    if cat_ref is None:
        print("CatBoost Huber not provided; comparison can be added later "
              "without retraining XGBoost.", flush=True)
    print(pooled[[
        "arm", "variant", "homes", "mdape", "mape",
        "delta_mdape_pp", "delta_mape_pp",
        "months_better_mdape",
    ]].to_string(index=False), flush=True)
    return pooled, monthly, by_type, output
