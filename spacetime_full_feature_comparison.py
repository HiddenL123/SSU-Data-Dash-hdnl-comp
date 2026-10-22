"""Full-history, chronological off-market comparison in model_run.ipynb.

The already-fitted monthly XGBoost models are the reference. Feature candidates
are complete 700-tree XGBoost refits with identical parameters, input rows,
log-target centering, and validation months. The GAM/RBF arms are separately
calibrated residual corrections of the ORIGINAL XGBoost, not of spline XGBoost.

Stores each successful month/variant independently for interruption-safe resume.
The manifest fingerprints source data, reference predictions, fitted baseline
weights (where exposed), settings, and model specs. Never reuse a cache after
changing this experimental code: increment PIPELINE_VERSION or use a new folder.
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
from spacetime_full_validation import (
    _calibration_snapshot, _check_model, _frame, _jsonable_spec,
    _validate_baseline_reference,
)
from spacetime_three_month import _attach_type, _type_summary, _validate_result, summarize

PIPELINE_VERSION = 1
BASE_FEATURE = "Xpt_spatiotemporal__selected_binary"
FEATURE_VARIANTS = (
    "baseline", "spacetime_splines", "geo_context", "combined",
    "comp_type_quota", "comp_both",
)
RESIDUAL_VARIANTS = ("tensor_gam", "rbf_ridge")
COMPARABLE_SUFFIXES = {
    "comp_type_quota": "__comp_type_quota",
    "comp_both": "__comp_both",
    "comp_age_first": "__comp_age_first",
}


def _feature_spec(spec, variant):
    """Preserve every base estimator setting; alter ONLY the feature provider."""
    if variant == "baseline":
        return spec
    suffix = COMPARABLE_SUFFIXES.get(variant, "__st_" + variant)
    return [(klass, name + suffix, params) for klass, name, params in spec]


def _signature(sales, reference, spec, models, config):
    digest = hashlib.sha256()
    # All model-input data matter: hash the full input frame, not just labels.
    for frame in (sales, reference):
        digest.update("|".join(frame.columns).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64").tobytes())
    for model in models:
        estimator = model.components[0][0].estimator
        # XGBoost exposes the trained booster. Hashing it detects a replaced
        # saved-model snapshot with unchanged constructor parameters.
        if hasattr(estimator, "get_booster"):
            digest.update(estimator.get_booster().save_raw(raw_format="ubj"))
        else:
            raise ValueError("Cannot fingerprint fitted baseline weights")
    digest.update(json.dumps({
        "version": PIPELINE_VERSION, "model_spec": _jsonable_spec(spec),
        **config,
    }, sort_keys=True).encode())
    return digest.hexdigest()


def _from_cache(path, holdout, *, month, arm, variant):
    """Reject truncated, misaligned, stale or wrong-variant cached predictions."""
    if not path.is_file():
        return None
    try:
        cached = pd.read_csv(path, dtype={"ml_num": str})
        if not {"ml_num", "month", "arm", "variant", "sold_price", "predicted_price"} <= set(cached):
            return None
        if (len(cached) != len(holdout) or cached.ml_num.duplicated().any() or
            not cached.month.eq(str(month)).all() or
            not cached.arm.eq(arm).all() or
            not cached.variant.eq(variant).all()):
            return None
        expected = holdout.set_index(holdout.ml_num.astype(str))
        got = cached.set_index("ml_num")
        if set(got.index) != set(expected.index):
            return None
        aligned = got.loc[expected.index]
        if not np.array_equal(aligned.sold_price.to_numpy(dtype=float),
                              expected.sold_price.to_numpy(dtype=float)):
            return None
        p = aligned.predicted_price.to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p <= 0).any():
            return None
        return cached
    except (KeyError, ValueError, TypeError, pd.errors.ParserError):
        return None


def _write_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def compare_full_spacetime_models(
    baseline_predictions, fitted_models, *, months=None,
    feature_variants=FEATURE_VARIANTS, residual_variants=RESIDUAL_VARIANTS,
    mode="expanding", train_cutoff=None, calibration_months=3,
    calibration_cap=None, residual_strength=0.5, seed=42,
    bootstrap_repeats=500, output_dir="outputs/spacetime_full_feature_comparison",
    resume=True, listings=None,
):
    """Run separately fitted candidates against exact original fitted XGBoost.

    Returns (pooled, monthly, by_house_type, predictions). Results are for the
    off_market track ONLY. For fixed_origin, candidates train once at the
    frozen cutoff and are reused for every requested subsequent month.
    """
    features, residuals = tuple(feature_variants), tuple(residual_variants)
    # Validate against this runner's own supported models and the directly
    # registered feature providers.  Do not reuse spacetime_shotgun's imported
    # registry: Jupyter can retain a stale copy of that module after git pull,
    # rejecting e.g. geo_context even though this runner supports it.
    from models.spacetime_screen_features import FEATURE_VARIANTS as AVAILABLE_ADDONS
    from models.spacetime_screen_residuals import RESIDUAL_VARIANTS as AVAILABLE_RESIDUALS
    if not features or features[0] != "baseline" or len(features) != len(set(features)):
        raise ValueError("feature_variants must begin with baseline and be unique")
    available_features = set(FEATURE_VARIANTS) | set(AVAILABLE_ADDONS) | set(COMPARABLE_SUFFIXES)
    unknown = sorted(set(features) - available_features)
    if unknown:
        raise ValueError(
            f"Unrecognized feature variants: {unknown}; "
            f"available variants: {sorted(available_features)}"
        )
    if len(residuals) != len(set(residuals)) or any(x not in AVAILABLE_RESIDUALS for x in residuals):
        raise ValueError("Unrecognized or repeated residual variant")
    if mode not in ("expanding", "fixed_origin"):
        raise ValueError("Unknown validation mode")
    if not (0 <= residual_strength <= 1) or calibration_months < 1 or bootstrap_repeats < 0:
        raise ValueError("Invalid residual strength, calibration months, or bootstraps")
    if calibration_cap is not None and calibration_cap < 1:
        raise ValueError("calibration_cap must be positive or None")
    if months is None:
        months = sorted(baseline_predictions.loc[
            baseline_predictions.track == "off_market", "month"].astype(str).unique())
    periods = pd.PeriodIndex(months, freq="M")
    if len(periods) == 0 or periods.has_duplicates or not periods.is_monotonic_increasing:
        raise ValueError("Months must be nonempty, unique and chronological")
    if any(p >= pd.Period("2026-06", freq="M") for p in periods):
        raise ValueError("Competition test months have no reference labels")
    cutoff = None
    if mode == "expanding":
        if len(fitted_models) != len(periods):
            raise ValueError("Expanding needs one baseline model pair per month")
    else:
        if len(fitted_models) != 1 or train_cutoff is None:
            raise ValueError("Fixed origin needs one model pair and train_cutoff")
        cutoff = pd.Timestamp(train_cutoff)
        if cutoff != cutoff.to_period("M").start_time or any(p < cutoff.to_period("M") for p in periods):
            raise ValueError("Invalid fixed cutoff relative to holdouts")

    baseline_models = [pair[0] for pair in fitted_models]
    spec = _check_model(baseline_models[0])
    if any(_jsonable_spec(_check_model(m)) != _jsonable_spec(spec) for m in baseline_models[1:]):
        raise ValueError("Baseline model specifications differ")
    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales.date_sold)
    sales["date_listed"] = pd.to_datetime(sales.date_listed)
    if sales.duplicated(["ml_num", "date_sold"]).any():
        raise ValueError("Duplicate sale identifiers prevent paired evaluation")
    config = {
        "mode": mode, "months": [str(p) for p in periods],
        "train_cutoff": str(cutoff) if cutoff is not None else None,
        "features": list(features), "residuals": list(residuals),
        "calibration_months": calibration_months, "calibration_cap": calibration_cap,
        "residual_strength": residual_strength, "seed": seed,
        "bootstrap_repeats": bootstrap_repeats,
    }
    fingerprint = _signature(sales, baseline_predictions, spec, baseline_models, config)
    manifest = {"fingerprint": fingerprint, **config, "spec": _jsonable_spec(spec),
                "pipeline_version": PIPELINE_VERSION}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_file = root / "setup.json"
    if resume and manifest_file.exists():
        if json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Cached output settings/baseline differ; use a fresh folder or resume=False")
    # Never pick up orphaned candidate CSVs without a compatible manifest.
    reuse_results = resume and manifest_file.is_file()
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    reusable_calibrators = ({period: baseline_models[i]
                            for i, period in enumerate(periods)}
                           if mode == "expanding" else {})
    snapshots = {}
    frozen_candidates = {}
    rows = []
    expected = {("feature", x) for x in features} | {("residual", x) for x in residuals}
    for i, month in enumerate(periods):
        origin = month.start_time if mode == "expanding" else cutoff
        origin_month = origin.to_period("M")
        reference_model = baseline_models[i if mode == "expanding" else 0]
        holdout = sales.loc[sales.date_sold.dt.to_period("M") == month]
        if holdout.empty:
            raise ValueError(f"No sold homes in {month}")
        hidden = data.hide_answers(holdout, "off_market")
        baseline_predict = np.asarray(reference_model.predict(hidden), dtype=float)
        base_frame = _frame(holdout, month, "feature", "baseline", baseline_predict)
        _validate_baseline_reference(baseline_predictions, base_frame, month)
        month_dir = root / "by_month" / str(month)
        rows.append(base_frame)
        print(f"\n{month}: {len(holdout):,} holdouts; full history before {origin.date()}", flush=True)
        for variant in features[1:]:
            path = month_dir / f"feature__{variant}.csv"
            cached = _from_cache(path, holdout, month=month,
                                 arm="feature", variant=variant) if reuse_results else None
            if cached is not None:
                print(f"  {variant}: cached", flush=True)
                rows.append(cached)
                continue
            if mode == "fixed_origin" and variant in frozen_candidates:
                fitted = frozen_candidates[variant]
            else:
                training = sales.loc[sales.date_sold < origin]
                print(f"  {variant}: fitting 700-tree spec to {len(training):,} sales", flush=True)
                fitted = OffMarketModel(_feature_spec(spec, variant), center_target=True).fit(training)
                if mode == "fixed_origin":
                    frozen_candidates[variant] = fitted
            result = _frame(holdout, month, "feature", variant,
                            np.asarray(fitted.predict(hidden), dtype=float))
            _write_frame(path, result)
            rows.append(result)
        if residuals:
            need_residuals = [variant for variant in residuals if not (
                reuse_results and _from_cache(month_dir / f"residual__{variant}.csv", holdout,
                                               month=month, arm="residual", variant=variant) is not None)]
            calibration = None
            if need_residuals:
                batches = []
                for lag in range(calibration_months, 0, -1):
                    cal_month = origin_month - lag
                    if cal_month not in snapshots:
                        snapshots[cal_month] = _calibration_snapshot(
                            sales, cal_month, spec, cap=calibration_cap, seed=seed,
                            cache_dir=root / "calibration", fingerprint=fingerprint,
                            resume=resume, pretrained_model=reusable_calibrators.get(cal_month),
                        )
                    batches.append(snapshots[cal_month])
                calibration = pd.concat(batches, ignore_index=True)
            for variant in residuals:
                path = month_dir / f"residual__{variant}.csv"
                cached = _from_cache(path, holdout, month=month,
                                     arm="residual", variant=variant) if reuse_results else None
                if cached is not None:
                    rows.append(cached)
                    print(f"  {variant}: cached", flush=True)
                    continue
                correction = SpaceTimeResidual(variant, seed=seed).fit(
                    calibration, calibration.log_residual.to_numpy(dtype=float)
                ).predict(hidden)
                predicted = baseline_predict * np.exp(
                    residual_strength * np.clip(correction, -0.25, 0.25)
                )
                result = _frame(holdout, month, "residual", variant, predicted)
                _write_frame(path, result)
                rows.append(result)
        print(f"  finished {month}", flush=True)

    predictions = pd.concat(rows, ignore_index=True)
    _validate_result(predictions, expected, [str(m) for m in periods])
    pooled, monthly = summarize(predictions, repeats=bootstrap_repeats, seed=seed)
    by_type = _type_summary(_attach_type(predictions, sales))
    predictions.to_csv(root / "predictions.csv", index=False)
    pooled.to_csv(root / "pooled_summary.csv", index=False)
    monthly.to_csv(root / "monthly_summary.csv", index=False)
    by_type.to_csv(root / "house_type_summary.csv", index=False)
    print("\nFull-history comparisons:\n", pooled[[
        "arm", "variant", "homes", "mdape", "mape",
        "delta_mdape_pp", "delta_mape_pp", "months_better_mdape",
        "delta_mdape_ci_low", "delta_mdape_ci_high",
    ]].to_string(index=False), flush=True)
    return pooled, monthly, by_type, predictions