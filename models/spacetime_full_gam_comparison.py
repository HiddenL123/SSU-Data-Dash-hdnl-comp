"""Full-history validation of spline XGBoost with model-specific Tensor GAM residuals.

Uses the same six walk-forward months, complete pre-origin sales and fitted
XGBoost spec as model_run.ipynb. Every GAM training label is a genuinely
out-of-time error from the *same* full-strength feature model it corrects.

Two base families:
  splines      = XGBoost + spatiotemporal spline features
  splines_geo  = XGBoost + spatiotemporal splines + geographic context

For each forecast month, calibrate a Tensor GAM with residuals from the prior
three SALE months. Every historical month is predicted by a feature-identical
XGBoost trained only on completed sales *before* that historical month. Thus
earlier calibration homes cannot see future prices, and validation months
never enter their own calibration.

Reuses previous spline-model holdout predictions ONLY if the old runner's
source manifest can be fingerprint-verified against the current raw sales,
baseline predictions, models and settings. Geo+spline is new and is refitted.

Outputs and resume caches are isolated from the earlier model comparisons.
Do not interpret the repeatedly tested Dec 2025-May 2026 fold as untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
from models.model import OffMarketModel
from models.spacetime_screen_residuals import SpaceTimeResidual
from comparisons.spacetime_full_feature_comparison import (
    _feature_spec, _from_cache, _signature, _write_frame,
)
from experiments.spacetime_full_validation import (
    _check_model, _frame, _jsonable_spec, _validate_baseline_reference,
)
from experiments.spacetime_three_month import (
    _attach_type, _type_summary, _validate_result, _bootstrap_delta, summarize,
)

PIPELINE_VERSION = 1
FAMILIES = {
    "splines": "spacetime_splines",
    "splines_geo": "spacetime_splines+geo_context",
}
CORRECTIONS = {
    "splines": "splines_tensor_gam",
    "splines_geo": "splines_geo_tensor_gam",
}


def _trusted_spline_cache(path, sales, reference, spec, models, periods):
    """Verify the original full-comparison manifest before reusing its fits."""
    if path is None:
        return None
    folder = Path(path)
    manifest_path = folder / "setup.json"
    if not manifest_path.is_file():
        print("No old spline cache manifest; missing fits will be trained.", flush=True)
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("mode") != "expanding" or
            manifest.get("months") != [str(m) for m in periods] or
            "spacetime_splines" not in manifest.get("features", []) or
            manifest.get("pipeline_version") != 1 or
            manifest.get("spec") != _jsonable_spec(spec)):
            raise ValueError("Previous feature-run configuration differs")
        config = {k: v for k, v in manifest.items()
                  if k not in {"fingerprint", "pipeline_version", "spec"}}
        expected = _signature(sales, reference, spec, models, config)
        if manifest.get("fingerprint") != expected:
            raise ValueError("Previous feature-run data/model fingerprint differs")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"Old spline cache not reusable: {exc}. Will refit.", flush=True)
        return None
    print(f"Verified old spline cache: {folder}", flush=True)
    return folder


def compare_full_spline_gam(
    baseline_predictions, fitted_models, *, months=None,
    calibration_months=3, calibration_cap=None, residual_strength=0.5,
    seed=42, bootstrap_repeats=500, resume=True,
    output_dir="outputs/spacetime_full_spline_gam",
    previous_feature_dir="outputs/spacetime_full_feature_comparison",
    listings=None,
):
    """Evaluate splines, spline+geo, and their own historical GAM corrections.

    Off-market expanding-window only. Return (pooled, monthly, by_type, rows).
    Reuses saved baseline XGBoost models but NEVER trains a GAM on its in-sample
    errors. Original baseline and existing full comparison are not mutated.
    """
    if calibration_months < 1 or bootstrap_repeats < 0:
        raise ValueError("Invalid calibration_months or bootstrap_repeats")
    if calibration_cap is not None and calibration_cap < 1:
        raise ValueError("calibration_cap must be positive or None")
    if not 0 <= residual_strength <= 1:
        raise ValueError("residual_strength must be in [0, 1]")
    if months is None:
        months = sorted(baseline_predictions.loc[
            baseline_predictions.track == "off_market", "month"
        ].astype(str).unique())
    periods = pd.PeriodIndex(months, freq="M")
    if (len(periods) == 0 or periods.has_duplicates or
        not periods.is_monotonic_increasing):
        raise ValueError("Provide nonempty chronological, unique validation months")
    if any(m >= pd.Period("2026-06", freq="M") for m in periods):
        raise ValueError("No known reference sale prices after May 2026")
    if len(fitted_models) != len(periods):
        raise ValueError("One saved baseline model pair required per validation month")

    models = [pair[0] for pair in fitted_models]
    spec = _check_model(models[0])
    if any(_jsonable_spec(_check_model(m)) != _jsonable_spec(spec)
           for m in models[1:]):
        raise ValueError("Saved baseline XGBoost specifications differ by month")

    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales.date_sold)
    sales["date_listed"] = pd.to_datetime(sales.date_listed)
    if sales.duplicated(["ml_num", "date_sold"]).any():
        raise ValueError("Repeated sale/month keys prevent paired evaluation")
    period_by_row = sales.date_sold.dt.to_period("M")
    cohorts = {m: sales.loc[period_by_row == m] for m in set(periods) |
               {p - lag for p in periods for lag in range(1, calibration_months + 1)}}
    if any(cohorts[m].empty for m in cohorts):
        raise ValueError("A validation or calibration month has no sold homes")

    config = {
        "pipeline_version": PIPELINE_VERSION,
        "months": [str(m) for m in periods],
        "bases": FAMILIES, "corrections": CORRECTIONS,
        "calibration_months": calibration_months,
        "calibration_cap": calibration_cap,
        "residual_strength": residual_strength,
        "bootstrap_repeats": bootstrap_repeats, "seed": seed,
        "previous_feature_dir": str(previous_feature_dir),
    }
    fingerprint = _signature(sales, baseline_predictions, spec, models, config)
    manifest = {"fingerprint": fingerprint, "spec": _jsonable_spec(spec), **config}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "setup.json"
    was_present = manifest_path.is_file()
    if resume and was_present:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Previous spline-GAM run differs; choose new output_dir")
    # Do not load orphaned partial CSVs without an exact compatible manifest.
    reuse_local = bool(resume and was_present)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    old_cache = _trusted_spline_cache(
        previous_feature_dir, sales, baseline_predictions, spec, models, periods)
    bases = {}       # (family, month) -> full held-out _frame
    snapshots = {}   # (family, month) -> historic OOT residual observations
    expected = {("feature", "baseline")} | {
        ("feature", family) for family in FAMILIES
    } | {("residual", name) for name in CORRECTIONS.values()}

    def base_prediction(family, month):
        key = (family, month)
        if key in bases:
            return bases[key]
        held = cohorts[month]
        filename = root / "by_month" / str(month) / f"base__{family}.csv"
        cached = (_from_cache(filename, held, month=month, arm="feature",
                              variant=family) if reuse_local else None)
        if cached is not None:
            print(f"  {month} {family}: local cached XGBoost predictions", flush=True)
            bases[key] = cached
            return cached
        # The old runner used the same FULL training history, estimator and
        # feature names for the single spatiotemporal spline variant.
        if family == "splines" and old_cache is not None and month in periods:
            source = old_cache / "by_month" / str(month) / "feature__spacetime_splines.csv"
            prior = _from_cache(
                source, held, month=month, arm="feature", variant="spacetime_splines"
            )
            if prior is not None:
                ordered = prior.set_index("ml_num").loc[held.ml_num.astype(str)]
                cached = _frame(
                    held, month, "feature", family,
                    ordered.predicted_price.to_numpy(dtype=float),
                )
                _write_frame(filename, cached)
                bases[key] = cached
                print(f"  {month} {family}: reused verified earlier full model", flush=True)
                return cached

        training = sales.loc[sales.date_sold < month.start_time]
        if training.empty:
            raise ValueError(f"No training history for {month}")
        variant = FAMILIES[family]
        print(f"  {month} {family}: fitting full model on "
              f"{len(training):,} earlier sales", flush=True)
        estimator = OffMarketModel(
            _feature_spec(spec, variant), center_target=True
        ).fit(training)
        prediction = np.asarray(
            estimator.predict(data.hide_answers(held, "off_market")), dtype=float
        )
        frame = _frame(held, month, "feature", family, prediction)
        _write_frame(filename, frame)
        bases[key] = frame
        return frame

    def oot_snapshot(family, month):
        key = (family, month)
        if key in snapshots:
            return snapshots[key]
        held = cohorts[month]
        prediction = base_prediction(family, month)
        ordered = prediction.set_index("ml_num").loc[held.ml_num.astype(str)]
        observed = held.sold_price.to_numpy(dtype=float)
        predicted = ordered.predicted_price.to_numpy(dtype=float)
        if (not np.isfinite(predicted).all() or (predicted <= 0).any() or
            not np.isfinite(observed).all() or (observed <= 0).any()):
            raise ValueError(f"Invalid OOT residuals for {family}/{month}")
        homes = data.hide_answers(held, "off_market")
        frame = pd.DataFrame({
            "date_listed": homes.date_listed.to_numpy(),
            "latitude": homes.latitude.to_numpy(dtype=float),
            "longitude": homes.longitude.to_numpy(dtype=float),
            "log_residual": np.log(observed) - np.log(predicted),
        })
        if calibration_cap is not None and len(frame) > calibration_cap:
            # Fixed per-month sample, independent of the later evaluation origin.
            idx = np.random.default_rng(seed).choice(
                len(frame), size=calibration_cap, replace=False)
            frame = frame.iloc[np.sort(idx)].reset_index(drop=True)
        snapshots[key] = frame
        return frame

    records = []
    for i, month in enumerate(periods):
        held = cohorts[month]
        hidden = data.hide_answers(held, "off_market")
        baseline_predict = np.asarray(models[i].predict(hidden), dtype=float)
        baseline_frame = _frame(
            held, month, "feature", "baseline", baseline_predict
        )
        _validate_baseline_reference(
            baseline_predictions, baseline_frame, month
        )
        records.append(baseline_frame)
        print(f"\n{month}: {len(held):,} held-out homes", flush=True)
        for family, correction_name in CORRECTIONS.items():
            base = base_prediction(family, month)
            records.append(base)
            corrected_path = (root / "by_month" / str(month) /
                              f"residual__{correction_name}.csv")
            cached = (_from_cache(
                corrected_path, held, month=month, arm="residual",
                variant=correction_name) if reuse_local else None)
            if cached is not None:
                print(f"  {month} {correction_name}: cached", flush=True)
                records.append(cached)
                continue
            snapshots_for_origin = [
                oot_snapshot(family, month - lag)
                for lag in range(calibration_months, 0, -1)
            ]
            calibration = pd.concat(snapshots_for_origin, ignore_index=True)
            if calibration.empty:
                raise ValueError(f"No calibration examples for {family}/{month}")
            gam = SpaceTimeResidual("tensor_gam", seed=seed).fit(
                calibration, calibration.log_residual.to_numpy(dtype=float)
            )
            correction = np.asarray(gam.predict(hidden), dtype=float)
            ordered = base.set_index("ml_num").loc[held.ml_num.astype(str)]
            corrected = ordered.predicted_price.to_numpy(dtype=float) * np.exp(
                residual_strength * np.clip(correction, -0.25, 0.25)
            )
            result = _frame(
                held, month, "residual", correction_name, corrected
            )
            _write_frame(corrected_path, result)
            records.append(result)
            print(f"  {month} {correction_name}: fitted on "
                  f"{len(calibration):,} prior OOT errors", flush=True)

    predictions = pd.concat(records, ignore_index=True)
    _validate_result(predictions, expected, [str(m) for m in periods])
    pooled, monthly = summarize(
        predictions, repeats=bootstrap_repeats, seed=seed
    )

    # GAM performance must also be compared with its *own* fitted XGBoost
    # base, not just with the unchanged original 700-tree XGBoost reference.
    # Pair predictions by month and property to keep the comparison exact.
    for family, correction_name in CORRECTIONS.items():
        keys = ["month", "ml_num"]
        base_rows = predictions.loc[
            (predictions.arm == "feature") & (predictions.variant == family)
        ].set_index(keys).sort_index()
        corrected_rows = predictions.loc[
            (predictions.arm == "residual") &
            (predictions.variant == correction_name)
        ].set_index(keys).loc[base_rows.index]
        price = base_rows.sold_price.to_numpy(dtype=float)
        base_ape = 100 * np.abs(base_rows.predicted_price.to_numpy() - price) / price
        corrected_ape = (100 * np.abs(
            corrected_rows.predicted_price.to_numpy() - price) / price)
        pairwise = _bootstrap_delta(
            base_ape, corrected_ape, repeats=bootstrap_repeats, seed=seed
        )
        row = pooled.index[
            (pooled.arm == "residual") & (pooled.variant == correction_name)
        ]
        pooled.loc[row, "parent_model"] = family
        pooled.loc[row, "delta_parent_mdape_pp"] = (
            np.median(corrected_ape) - np.median(base_ape)
        )
        pooled.loc[row, "delta_parent_mape_pp"] = (
            np.mean(corrected_ape) - np.mean(base_ape)
        )
        pooled.loc[row, "parent_homes_improved_fraction"] = np.mean(
            corrected_ape < base_ape
        )
        for metric, value in pairwise.items():
            pooled.loc[row, metric.replace("delta_", "delta_parent_")] = value
        for month in periods:
            base_month = base_rows.loc[str(month)]
            corr_month = corrected_rows.loc[str(month)]
            true_month = base_month.sold_price.to_numpy(dtype=float)
            a = 100 * np.abs(
                base_month.predicted_price.to_numpy() - true_month
            ) / true_month
            b = 100 * np.abs(
                corr_month.predicted_price.to_numpy() - true_month
            ) / true_month
            mask = ((monthly.month == str(month)) &
                    (monthly.arm == "residual") &
                    (monthly.variant == correction_name))
            monthly.loc[mask, "delta_parent_mdape_pp"] = (
                np.median(b) - np.median(a)
            )
            monthly.loc[mask, "delta_parent_mape_pp"] = np.mean(b) - np.mean(a)

    by_type = _type_summary(_attach_type(predictions, sales))
    predictions.to_csv(root / "predictions.csv", index=False)
    pooled.to_csv(root / "pooled_summary.csv", index=False)
    monthly.to_csv(root / "monthly_summary.csv", index=False)
    by_type.to_csv(root / "house_type_summary.csv", index=False)
    print("\nFull spline / GAM comparison:\n", pooled[[
        "arm", "variant", "homes", "mdape", "mape", "delta_mdape_pp",
        "delta_mape_pp", "months_better_mdape",
        "delta_mdape_ci_low", "delta_mdape_ci_high",
    ]].to_string(index=False), flush=True)
    return pooled, monthly, by_type, predictions
