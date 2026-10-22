"""Full-strength space-time shortlisted validation integrated with model_run.ipynb.

Takes the already-fitted XGBoost models and predictions from the notebook's
existing validate() call, so the BASELINE never needs to be retrained.

Compare:
  * baseline: unchanged 47 binary-term XGBoost;
  * age_first: same exact model/hyperparameters, corrected comp eligibility;
  * tensor_gam: original XGBoost + chronological tensor-spline residual;
  * rbf_ridge: original XGBoost + chronological RBF/Ridge residual.

Residual calibration has no in-sample targets: for every calibration month C,
refit the original full-strength XGBoost only on sales completed before C;
predict homes sold in C; then fit the residual models using those genuinely
out-of-time errors. Calibration folds are REUSED across evaluation origins,
and persist in an optional fingerprint-checked cache.

The same function can evaluate expanding-window validation or a four-month
fixed-origin horizon. During fixed-origin validation, all calibration
sales must predate the original frozen training cutoff.

Never use this for automatic hyperparameter search on the repeatedly consulted
Dec 2025-May 2026 period. Results are separate from reports/validation_report.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
from models.model import OffMarketModel
from models.spacetime_screen_residuals import SpaceTimeResidual
from spacetime_three_month import (
    _attach_type, _type_summary, _validate_result, summarize,
)

VARIANTS = (
    ("feature", "baseline"),
    ("feature", "comp_age_first"),
    ("residual", "tensor_gam"),
    ("residual", "rbf_ridge"),
)
FEATURE_NAME = "Xpt_spatiotemporal__selected_binary"
PIPELINE_VERSION = 1


def _jsonable_spec(spec):
    return [
        {"estimator": f"{klass.__module__}.{klass.__name__}",
         "features": name, "params": params or {}}
        for klass, name, params in spec
    ]


def _fingerprint(sales, spec, config):
    required = ["ml_num", "date_sold", "date_listed", "sold_price"]
    hashed = pd.util.hash_pandas_object(sales[required], index=False)
    digest = hashlib.sha256()
    digest.update(hashed.to_numpy(dtype="uint64").tobytes())
    digest.update(
        json.dumps({
            "version": PIPELINE_VERSION,
            "model": _jsonable_spec(spec), **config,
        }, sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()


def _check_model(model):
    if not isinstance(model, OffMarketModel) or not model._fitted:
        raise ValueError("Expected a fitted OffMarketModel from model_run.validate")
    if len(model.modelspecs) != 1 or len(model.components) != 1:
        raise ValueError("This experiment requires a single fitted XGBoost stage")
    cls, name, _ = model.modelspecs[0]
    if name != FEATURE_NAME or model.components[0][0].feature_backend != "xgboost":
        raise ValueError(
            "Baseline must be XGBoost with Xpt_spatiotemporal__selected_binary"
        )
    if not model.center_target:
        raise ValueError("Baseline must use center_target=True")
    return model.modelspecs


def _calibration_snapshot(sales, month, spec, *, cap, seed,
                          cache_dir, fingerprint, resume,
                          pretrained_model=None):
    """One full-strength, chronologically OOT XGBoost refit per unique month."""
    path = cache_dir / f"{month}.csv"
    signature_path = cache_dir / f"{month}.meta.json"
    cutoff = month.start_time
    train = sales.loc[sales.date_sold < cutoff]
    held = sales.loc[sales.date_sold.dt.to_period("M") == month]
    if train.empty or held.empty:
        raise ValueError(f"No historical training or OOT calibration homes for {month}")
    if cap is not None:
        if cap < 1:
            raise ValueError("calibration cap must be positive or None")
        if len(held) > cap:
            held = held.sample(n=cap, random_state=seed).sort_index()
    signature = {
        "fingerprint": fingerprint,
        "month": str(month),
        "calibration_cap": cap,
        "seed": seed,
        "training_rows": len(train),
        "calibration_rows": len(held),
    }
    if resume and path.exists() and signature_path.exists():
        old = json.loads(signature_path.read_text(encoding="utf-8"))
        if old == signature:
            snapshot = pd.read_csv(path, parse_dates=["date_listed"])
            if (len(snapshot) == len(held) and
                np.isfinite(snapshot["log_residual"]).all()):
                print(f"    {month}: using cached full-strength OOT residuals", flush=True)
                return snapshot

    if pretrained_model is None:
        print(f"    {month}: fitting tuned base model on "
              f"{len(train):,} earlier sales; predicting "
              f"{len(held):,} unseen homes", flush=True)
        base = OffMarketModel(spec, center_target=True).fit(train)
    else:
        _check_model(pretrained_model)
        if _jsonable_spec(pretrained_model.modelspecs) != _jsonable_spec(spec):
            raise ValueError("Incompatible pretrained OOT calibration model")
        print(f"    {month}: reusing existing fitted validation model "
              f"for {len(held):,} OOT predictions", flush=True)
        base = pretrained_model
    hidden = data.hide_answers(held, "off_market")
    predicted = np.asarray(base.predict(hidden), dtype=float)
    sold = held.sold_price.to_numpy(dtype=float)
    if not (np.isfinite(predicted).all() and (predicted > 0).all() and
            np.isfinite(sold).all() and (sold > 0).all()):
        raise ValueError(f"Invalid calibration predictions for {month}")
    snapshot = pd.DataFrame({
        "date_listed": hidden.date_listed.to_numpy(),
        "latitude": hidden.latitude.to_numpy(),
        "longitude": hidden.longitude.to_numpy(),
        "log_residual": np.log(sold) - np.log(predicted),
    })
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(path, index=False)
    signature_path.write_text(
        json.dumps(signature, indent=2), encoding="utf-8"
    )
    return snapshot


def _validate_baseline_reference(reference, predictions, month, *, track="off_market"):
    rows = reference.loc[
        (reference.track == track) &
        (reference.month.astype(str) == str(month))
    ]
    if len(rows) != len(predictions) or rows.ml_num.duplicated().any():
        raise ValueError(f"Saved baseline rows do not match {month}")
    ordered = rows.set_index("ml_num").loc[predictions.ml_num.to_numpy()]
    if not np.array_equal(
        ordered.sold_price.to_numpy(dtype=float),
        predictions.sold_price.to_numpy(dtype=float),
    ):
        raise ValueError(f"Baseline targets do not match {month}")
    # model_run.validate rounds predictions to whole dollars.
    if not np.allclose(
        ordered.predicted_price.to_numpy(dtype=float),
        predictions.predicted_price.to_numpy(dtype=float),
        rtol=0, atol=1.01,
    ):
        raise ValueError(
            f"Fitted baseline models do not reproduce the supplied "
            f"model_run validation for {month}; check the mode and models."
        )


def _frame(holdout, month, arm, variant, prediction):
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError(f"Nonpositive or nonfinite predictions for {variant}")
    return pd.DataFrame({
        "ml_num": holdout.ml_num.to_numpy(),
        "month": str(month), "arm": arm, "variant": variant,
        "sold_price": holdout.sold_price.to_numpy(dtype=float),
        "predicted_price": prediction,
    })


def validate_spacetime_shortlist(
    baseline_predictions, fitted_models, *, months=None,
    mode="expanding", train_cutoff=None, calibration_months=3,
    calibration_cap=None, residual_strength=0.5, seed=42,
    bootstrap_repeats=500,
    output_dir="outputs/spacetime_full_validation",
    resume=True, listings=None,
):
    """Use fitted model_run models to evaluate top add-ons without base refit.

    Parameters
    ----------
    baseline_predictions, fitted_models
        EXACT return values of model_run.validate(); no estimator retraining
        for baseline. For expanding, fitted_models has one pair per month.
        For fixed_origin, it has one frozen pair for all horizon months.
    months
        Must match the evaluated months in baseline_predictions.
    calibration_cap
        None uses ALL available OOT calibration homes in each month.
    """
    if mode not in ("expanding", "fixed_origin"):
        raise ValueError("mode must be expanding or fixed_origin")
    if calibration_months < 1 or bootstrap_repeats < 0:
        raise ValueError("Invalid calibration months or bootstrap repetitions")
    if not 0 <= residual_strength <= 1:
        raise ValueError("residual_strength must be in [0, 1]")
    if months is None:
        months = sorted(
            baseline_predictions.loc[
                baseline_predictions.track == "off_market", "month"
            ].astype(str).unique()
        )
    periods = pd.PeriodIndex(months, freq="M")
    if len(periods) == 0 or periods.has_duplicates or not periods.is_monotonic_increasing:
        raise ValueError("Prediction months must be unique and chronological")
    if any(p >= pd.Period("2026-06", freq="M") for p in periods):
        raise ValueError("No known sold prices for competition test months")
    if mode == "expanding":
        if len(fitted_models) != len(periods):
            raise ValueError("Expanding mode needs one fitted-model pair per month")
    else:
        if len(fitted_models) != 1 or train_cutoff is None:
            raise ValueError("Fixed origin needs one fitted-model pair and train_cutoff")
        cutoff = pd.Timestamp(train_cutoff)
        if cutoff != cutoff.to_period("M").start_time:
            raise ValueError("train_cutoff must be the first day of a month")
        if any(p < cutoff.to_period("M") for p in periods):
            raise ValueError("Validation months must follow the fixed training origin")

    models = [pair[0] for pair in fitted_models]
    # In expanding mode, a previously validated month's fitted baseline
    # is EXACTLY the model needed to generate that month's out-of-time
    # calibration predictions for subsequent origins.
    reusable_calibration_models = (
        {month: models[i] for i, month in enumerate(periods)}
        if mode == "expanding" else {}
    )
    spec = _check_model(models[0])
    config_spec = _jsonable_spec(spec)
    for model in models[1:]:
        _check_model(model)
        if _jsonable_spec(model.modelspecs) != config_spec:
            raise ValueError("Fitted baseline model specs differ across months")

    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales["date_sold"])
    sales["date_listed"] = pd.to_datetime(sales["date_listed"])
    sale_keys = pd.DataFrame({
        "month": sales.date_sold.dt.to_period("M").astype(str),
        "ml_num": sales.ml_num.astype(str),
    })
    if sale_keys.duplicated().any():
        raise ValueError("Cannot align results with repeated month/ml_num keys")
    root = Path(output_dir)
    configuration = {
        "mode": mode, "months": [str(p) for p in periods],
        "train_cutoff": str(train_cutoff) if train_cutoff is not None else None,
        "calibration_months": calibration_months,
        "calibration_cap": calibration_cap, "residual_strength": residual_strength,
        "seed": seed, "bootstrap_repeats": bootstrap_repeats,
        "variants": [list(x) for x in VARIANTS],
    }
    fingerprint = _fingerprint(sales, spec, configuration)
    manifest = {**configuration, "spec": config_spec, "fingerprint": fingerprint}
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "setup.json"
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                "Existing output directory has different settings or underlying "
                "data. Choose a new directory or set resume=False."
            )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cache_dir = root / "calibration"
    expected = set(VARIANTS)
    snapshots = {}
    all_rows = []
    frozen_age_model = None

    for index, month in enumerate(periods):
        monthly_path = root / f"{month}_predictions.csv"
        if resume and monthly_path.exists():
            try:
                cached = pd.read_csv(monthly_path, dtype={"ml_num": str})
                _validate_result(cached, expected, [str(month)])
                wanted = sales.loc[
                    sales.date_sold.dt.to_period("M") == month, "ml_num"
                ].astype(str)
                base = cached.loc[
                    (cached.arm == "feature") & (cached.variant == "baseline"), "ml_num"
                ].astype(str)
                if set(base) == set(wanted) and len(base) == len(wanted):
                    all_rows.append(cached)
                    print(f"{month}: reused completed comparison", flush=True)
                    continue
            except (ValueError, KeyError, TypeError):
                pass

        origin = month.start_time if mode == "expanding" else cutoff
        origin_period = origin.to_period("M")
        base = models[index if mode == "expanding" else 0]
        holdout = sales.loc[sales.date_sold.dt.to_period("M") == month]
        if holdout.empty:
            raise ValueError(f"No validation properties for {month}")
        hidden = data.hide_answers(holdout, "off_market")
        base_prediction = np.asarray(base.predict(hidden), dtype=float)
        baseline_frame = _frame(holdout, month, "feature", "baseline", base_prediction)
        _validate_baseline_reference(
            baseline_predictions,
            baseline_frame, month
        )

        print(f"\n{month}: full-history model, {len(hidden):,} homes, "
              f"training cutoff {origin.date()}", flush=True)
        rows = [baseline_frame]

        age_spec = [(klass, name + "__comp_age_first", params)
                    for klass, name, params in spec]
        if mode == "expanding" or frozen_age_model is None:
            training = sales.loc[sales.date_sold < origin]
            print(f"  Age-first: training with {len(training):,} earlier sales",
                  flush=True)
            age_model = OffMarketModel(age_spec, center_target=True).fit(training)
            if mode == "fixed_origin":
                frozen_age_model = age_model
        else:
            age_model = frozen_age_model
        age_pred = np.asarray(age_model.predict(hidden), dtype=float)
        rows.append(_frame(
            holdout, month, "feature", "comp_age_first", age_pred
        ))

        calibration_parts = []
        for lag in range(calibration_months, 0, -1):
            past_month = origin_period - lag
            if past_month not in snapshots:
                snapshots[past_month] = _calibration_snapshot(
                    sales, past_month, spec, cap=calibration_cap,
                    seed=seed, cache_dir=cache_dir,
                    fingerprint=fingerprint, resume=resume,
                    pretrained_model=reusable_calibration_models.get(past_month),
                )
            calibration_parts.append(snapshots[past_month])
        calibration = pd.concat(calibration_parts, ignore_index=True)
        for variant in ("tensor_gam", "rbf_ridge"):
            correction = SpaceTimeResidual(variant, seed=seed).fit(
                calibration, calibration.log_residual.to_numpy(dtype=float)
            ).predict(hidden)
            corrected = base_prediction * np.exp(
                residual_strength * np.clip(correction, -0.25, 0.25)
            )
            rows.append(_frame(
                holdout, month, "residual", variant, corrected
            ))

        monthly_predictions = pd.concat(rows, ignore_index=True)
        _validate_result(monthly_predictions, expected, [str(month)])
        monthly_predictions.to_csv(monthly_path, index=False)
        all_rows.append(monthly_predictions)
        for arm, variant in VARIANTS:
            sub = monthly_predictions.loc[
                (monthly_predictions.arm == arm) &
                (monthly_predictions.variant == variant)
            ]
            signed = (sub.predicted_price.to_numpy() -
                      sub.sold_price.to_numpy()) / sub.sold_price.to_numpy()
            print(f"  {variant:16s} MdAPE={100*np.median(np.abs(signed)):.3f}% "
                  f"MAPE={100*np.mean(np.abs(signed)):.3f}%", flush=True)

    predictions = pd.concat(all_rows, ignore_index=True)
    _validate_result(predictions, expected, [str(m) for m in periods])
    pooled, monthly = summarize(
        predictions, repeats=bootstrap_repeats, seed=seed
    )
    labeled = _attach_type(predictions, sales)
    by_type = _type_summary(labeled)
    predictions.to_csv(root / "predictions.csv", index=False)
    pooled.to_csv(root / "pooled_summary.csv", index=False)
    monthly.to_csv(root / "monthly_summary.csv", index=False)
    labeled.to_csv(root / "predictions_with_type.csv", index=False)
    by_type.to_csv(root / "house_type_summary.csv", index=False)
    print("\nPooled full-strength comparison:\n",
          pooled[["arm", "variant", "homes", "mdape", "mape",
                  "delta_mdape_pp", "delta_mape_pp",
                  "months_better_mdape",
                  "delta_mdape_ci_low", "delta_mdape_ci_high"]].to_string(
                      index=False
                  ), flush=True)
    return pooled, monthly, by_type, predictions
