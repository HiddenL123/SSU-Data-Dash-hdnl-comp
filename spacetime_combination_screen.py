"""Cheap development-only feature and spline+GAM combination screen.

The GAM combination is calibrated on genuine historical out-of-time errors
from the SAME spline-base features (never baseline-XGBoost residuals).
Every held-out validation home remains outside fitting and calibration.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import data
from models.features import make_features
from models.spacetime_screen_residuals import SpaceTimeResidual
from spacetime_shotgun import FEATURE_SET, XGB_TUNED, _cap, _make_estimator, _metric

# Same base selected-binary BoW model; __st_ suffix adds families to its features.
COMBINATIONS = {
    "baseline": "",
    "splines": "__st_spacetime_splines",
    "geo_context": "__st_geo_context",
    "comp_type_quota": "__comp_type_quota",
    "splines_geo": "__st_spacetime_splines+geo_context",
    "splines_quota": "__comp_type_quota__st_spacetime_splines",
    "splines_market": "__st_spacetime_splines+market",
    "splines_geo_quota": "__comp_type_quota__st_spacetime_splines+geo_context",
}
# These correction variants are trained on earlier out-of-time predictions
# from their exact feature base, not on errors of the ordinary XGBoost.
CORRECTIONS = {
    "splines_tensor_gam": "splines",
    "splines_geo_tensor_gam": "splines_geo",
}


def _predict(train, homes, *, suffix, params, trees):
    log_prices = np.log(train.sold_price.to_numpy(dtype=float))
    if not np.isfinite(log_prices).all():
        raise ValueError("Training log-prices must be finite")
    offset = float(np.median(log_prices))
    provider = make_features(FEATURE_SET + suffix, model="xgboost", track="off_market")
    tr = provider.fit_transform(train)
    te = provider.transform(homes)
    if list(tr.columns) != list(te.columns):
        raise ValueError("Train/holdout feature names differ")
    estimator = _make_estimator(params, trees)
    estimator.fit(tr, log_prices - offset)
    return np.exp(offset + np.asarray(estimator.predict(te), dtype=float)), len(tr.columns)


def _oot_residuals(sales, *, month, train_start, cap, calibration_cap,
                   folds, seed, suffix, params, trees):
    """Refit feature-identical model before each calibration sale month."""
    from_frames, log_errors = [], []
    for lag in range(folds, 0, -1):
        past = month - lag
        training = _cap(sales.loc[(sales.date_sold < past.start_time) &
                                  (sales.date_sold >= pd.Timestamp(train_start))],
                        cap, seed)
        held = _cap(sales.loc[sales.date_sold.dt.to_period("M") == past],
                    calibration_cap, seed + lag)
        if training.empty or held.empty:
            raise ValueError(f"Missing earlier training/calibration homes: {past}")
        observed = held.sold_price.to_numpy(dtype=float)
        hidden = data.hide_answers(held, "off_market")
        p, _ = _predict(training, hidden, suffix=suffix, params=params, trees=trees)
        from_frames.append(hidden)
        log_errors.append(np.log(observed) - np.log(p))
    return pd.concat(from_frames, ignore_index=True), np.concatenate(log_errors)


def run_combination_screen(
    *, validation_month="2025-08", train_start="2022-01-01",
    train_cap=10000, validation_cap=1000, trees=220, seed=42,
    combinations=None, params=None, calibration_folds=2,
    calibration_cap=300, residual_strength=0.5,
    output_dir="outputs/spacetime_combination_screen", listings=None,
):
    month = pd.Period(validation_month, freq="M")
    if month >= pd.Period("2025-12", freq="M"):
        raise ValueError("Use earlier development months for combination screening")
    if min(train_cap, validation_cap, trees, calibration_folds, calibration_cap) < 1:
        raise ValueError("Training cap, validation cap and trees must be positive")
    variants = tuple((*COMBINATIONS, *CORRECTIONS) if combinations is None else combinations)
    if not variants or variants[0] != "baseline" or len(set(variants)) != len(variants):
        raise ValueError("Combinations must start with unique baseline")
    if any(name not in COMBINATIONS and name not in CORRECTIONS for name in variants):
        raise ValueError("Unknown combination")
    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales.date_sold)
    train = _cap(sales.loc[(sales.date_sold >= pd.Timestamp(train_start)) &
                           (sales.date_sold < month.start_time)], train_cap, seed)
    holdout = _cap(sales.loc[sales.date_sold.dt.to_period("M") == month],
                   validation_cap, seed + 11)
    if train.empty or holdout.empty:
        raise ValueError("No training or holdout sales")
    hidden = data.hide_answers(holdout, "off_market")
    scores, predicted_rows, failures = [], [], []
    base = None
    fitted_preds = {}
    for name in variants:
        print(f"{month}: {name} on {len(train):,} training homes", flush=True)
        try:
            started = perf_counter()
            fit_params = XGB_TUNED if params is None else params
            if name in CORRECTIONS:
                underlying = CORRECTIONS[name]
                if underlying not in fitted_preds:
                    raise ValueError(f"Run prerequisite feature variant {underlying} first")
                oot_homes, oot_error = _oot_residuals(
                    sales, month=month, train_start=train_start,
                    cap=train_cap, calibration_cap=calibration_cap,
                    folds=calibration_folds, seed=seed, suffix=COMBINATIONS[underlying],
                    params=fit_params, trees=trees,
                )
                corr = SpaceTimeResidual("tensor_gam", seed=seed).fit(
                    oot_homes, oot_error
                ).predict(hidden)
                estimate = fitted_preds[underlying] * np.exp(
                    residual_strength * np.clip(corr, -0.25, 0.25)
                )
                ncols = None
            else:
                estimate, ncols = _predict(train, hidden, suffix=COMBINATIONS[name],
                                           params=fit_params, trees=trees)
                fitted_preds[name] = estimate
            if not np.isfinite(estimate).all() or (estimate <= 0).any():
                raise ValueError("Invalid predictions")
            metric = _metric(holdout.sold_price.to_numpy(dtype=float), estimate)
            if name == "baseline":
                base = metric
            metric.update(variant=name, runtime_seconds=perf_counter()-started,
                          n_features=ncols,
                          delta_mdape_pp=metric["mdape"]-base["mdape"],
                          delta_mape_pp=metric["mape"]-base["mape"])
            scores.append(metric)
            predicted_rows.append(pd.DataFrame({
                "ml_num": holdout.ml_num.astype(str).to_numpy(),
                "month": str(month), "variant": name,
                "sold_price": holdout.sold_price.to_numpy(dtype=float),
                "predicted_price": estimate,
            }))
        except Exception as exc:
            if name == "baseline":
                raise
            failures.append({"variant": name, "error": str(exc)})
            print(f"  FAILED: {name}: {exc}", flush=True)
    summary = pd.DataFrame(scores).sort_values(["mdape", "mape"]).reset_index(drop=True)
    predictions = pd.concat(predicted_rows, ignore_index=True)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    summary.to_csv(folder / "summary.csv", index=False)
    predictions.to_csv(folder / "predictions.csv", index=False)
    (folder / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    (folder / "setup.json").write_text(json.dumps({
        "validation_month": str(month), "train_start": train_start,
        "train_cap": train_cap, "validation_cap": validation_cap,
        "trees": trees, "seed": seed, "variants": variants,
        "calibration_folds": calibration_folds, "calibration_cap": calibration_cap,
        "residual_strength": residual_strength,
        "params": XGB_TUNED if params is None else params,
        "scope": "development screening only, not full-history confirmation",
    }, indent=2), encoding="utf-8")
    print(summary[["variant", "mdape", "mape", "delta_mdape_pp", "delta_mape_pp"]]
          .to_string(index=False), flush=True)
    return summary, predictions, failures