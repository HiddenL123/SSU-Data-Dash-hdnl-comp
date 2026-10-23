"""Chronological Matérn GP residual hyperparameter screen on tuned XGBoost.

This is NOT a retraining scheme for XGBoost. One cached XGBoost baseline
(and earlier out-of-time calibration fits) is reused for every GP trial.

Tune exclusively on earlier sale-month cohorts (May–July 2025 by default).
Use a separate confirmation call for later development folds
(September–November 2025; previously examined, NOT untouched final testing).

Usage is notebook-first; see the final cells of spacetime_fast.ipynb.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import data
from models.spacetime_structural_residuals import matern_gp
from models.spacetime_screen_residuals import SpaceTimeResidual
from models.spacetime_structural_features import SPLINE_GEO_SUFFIX
from spacetime_fast import XGB_TUNED, _cap
from spacetime_combination_screen import _predict as predict_xgb
from spacetime_structural_fast import _cache, _frame, _save, _report

VERSION = 1
KINDS = ("spatial", "spacetime")
DEFAULT_KERNEL = {
    "spatial_km": 5.0,
    "temporal_months": 2.0,
    "amplitude_var": 0.003,
    "noise_var": 0.012,
    "strength": 0.5,
}


def _ape(y, p):
    return 100.0 * np.abs(np.asarray(p, float) - np.asarray(y, float)) / np.asarray(y, float)


def _months(months):
    periods = pd.PeriodIndex(months, freq="M")
    if (not len(periods) or periods.has_duplicates or
        not periods.is_monotonic_increasing or
        any(m >= pd.Period("2025-12", freq="M") for m in periods)):
        raise ValueError("Use sorted, distinct, pre-December-2025 development months")
    return periods


def _proposal(rng):
    def log_uniform(low, high):
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return {
        "spatial_km": log_uniform(0.8, 30.0),
        "temporal_months": log_uniform(0.5, 18.0),
        "amplitude_var": log_uniform(0.00015, 0.025),
        "noise_var": log_uniform(0.0005, 0.065),
        "strength": float(rng.uniform(0.15, 1.2)),
    }


def _predict_corrected(fold, kind, params, *, max_gp_points, seed):
    correction = matern_gp(
        fold["calibration"], fold["calibration"].log_error.to_numpy(dtype=float),
        fold["hidden"], spatial_only=(kind == "spatial"),
        max_points=max_gp_points, seed=seed,
        spatial_km=params["spatial_km"],
        temporal_months=params["temporal_months"],
        amplitude_var=params["amplitude_var"],
        noise_var=params["noise_var"],
    )
    return fold["baseline"] * np.exp(
        params["strength"] * np.clip(correction, -0.25, 0.25)
    )


class _ChronologicalXGB:
    """Shared verified caches for tuner and subsequent confirmation."""

    def __init__(self, *, sales, root, train_start, train_cap, validation_cap,
                 calibration_cap, calibration_folds, trees, max_gp_points, seed,
                 tune_months, resume):
        if min(train_cap, validation_cap, calibration_cap,
               calibration_folds, trees, max_gp_points) < 1:
            raise ValueError("Positive training, calibration, GP and tree caps required")
        if max_gp_points < 2:
            raise ValueError("max_gp_points must be >= 2")
        self.sales = sales.copy()
        self.sales["date_sold"] = pd.to_datetime(self.sales.date_sold)
        self.sales["date_listed"] = pd.to_datetime(self.sales.date_listed)
        if self.sales.ml_num.isna().any():
            raise ValueError("Missing ml_num in housing data")
        self.sale_months = self.sales.date_sold.dt.to_period("M")
        self.root, self.resume = Path(root), bool(resume)
        self.root.mkdir(parents=True, exist_ok=True)
        self.start = pd.Timestamp(train_start)
        self.train_cap, self.validation_cap = train_cap, validation_cap
        self.calibration_cap = calibration_cap
        self.calibration_folds = calibration_folds
        self.trees, self.max_gp_points, self.seed = trees, max_gp_points, seed
        periods = _months(tune_months)
        if self.start >= (periods[0] - calibration_folds).start_time:
            raise ValueError("Training data must precede earliest calibration month")
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(self.sales, index=True)
            .to_numpy(dtype="uint64").tobytes()
        ).hexdigest()
        manifest = {
            "version": VERSION, "input_sha256": fingerprint,
            "train_start": str(self.start.date()),
            "train_cap": train_cap, "validation_cap": validation_cap,
            "calibration_cap": calibration_cap,
            "calibration_folds": calibration_folds,
            "trees": trees, "max_gp_points": max_gp_points,
            "seed": seed, "tune_months": [str(m) for m in periods],
            "feature_suffix": SPLINE_GEO_SUFFIX, "xgb_params": XGB_TUNED,
            "note": "Matérn-only hyperparameter tuning; never tune XGBoost here",
        }
        setup_path = self.root / "setup.json"
        if self.resume and setup_path.exists():
            old = json.loads(setup_path.read_text(encoding="utf-8"))
            if old != manifest:
                raise ValueError(
                    "Output directory contains different data/settings; "
                    "use a new output_dir instead of reusing stale fits"
                )
        if setup_path.exists() and not self.resume:
            raise ValueError("Use a fresh output_dir when resume=False")
        setup_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.fold_cache = {}
        self.oot_cache = {}

    def _holdout(self, month, role):
        cap = self.calibration_cap if role == "calibration" else self.validation_cap
        sample_seed = (
            self.seed + month.ordinal % 97 if role == "calibration"
            else self.seed + 11
        )
        held = _cap(self.sales.loc[self.sale_months == month], cap, sample_seed)
        if held.empty:
            raise ValueError(f"No {role} sales in {month}")
        if held.ml_num.astype(str).duplicated().any():
            raise ValueError(f"Duplicate MLS IDs in {role} month {month}")
        return held

    def _baseline(self, month, held, role):
        path = self.root / "xgb" / str(month) / f"{role}.csv"
        cached = _cache(path, held, month, "baseline") if self.resume else None
        if cached is not None:
            return cached.set_index("ml_num").loc[
                held.ml_num.astype(str), "predicted_price"
            ].to_numpy(dtype=float)
        earlier = self.sales.loc[
            (self.sales.date_sold >= self.start) &
            (self.sales.date_sold < month.start_time)
        ]
        train = _cap(earlier, self.train_cap, self.seed)
        if train.empty:
            raise ValueError(f"No historical sales before {month}")
        print(f"  XGBoost {month} ({role}), {len(train):,} historical homes",
              flush=True)
        t0 = perf_counter()
        estimated, _ = predict_xgb(
            train, data.hide_answers(held, "off_market"),
            suffix=SPLINE_GEO_SUFFIX, params=XGB_TUNED, trees=self.trees,
        )
        _save(path, _frame(month, held, "baseline",
                           estimated, perf_counter() - t0))
        return np.asarray(estimated, dtype=float)

    def _oot(self, month):
        if month not in self.oot_cache:
            held = self._holdout(month, "calibration")
            predicted = self._baseline(month, held, "calibration")
            hidden = data.hide_answers(held, "off_market").reset_index(drop=True)
            hidden["log_error"] = (
                np.log(held.sold_price.to_numpy(dtype=float)) -
                np.log(predicted)
            )
            self.oot_cache[month] = hidden
        return self.oot_cache[month]

    def fold(self, month):
        if month not in self.fold_cache:
            held = self._holdout(month, "validation")
            hidden = data.hide_answers(held, "off_market").reset_index(drop=True)
            baseline = self._baseline(month, held, "validation")
            calibration = pd.concat([
                self._oot(month - lag)
                for lag in range(self.calibration_folds, 0, -1)
            ], ignore_index=True)
            self.fold_cache[month] = {
                "month": month, "held": held, "hidden": hidden,
                "baseline": baseline, "calibration": calibration,
                "actual": held.sold_price.to_numpy(dtype=float),
            }
        return self.fold_cache[month]


def _context(*, tune_months, train_start="2022-01-01",
             train_cap=8000, validation_cap=900, calibration_cap=450,
             calibration_folds=2, trees=500, max_gp_points=350,
             seed=42, output_dir="outputs/spacetime_matern_tuning",
             resume=True, listings=None):
    sales = data.load("listings") if listings is None else listings
    return _ChronologicalXGB(
        sales=sales, root=output_dir, tune_months=tune_months,
        train_start=train_start, train_cap=train_cap,
        validation_cap=validation_cap, calibration_cap=calibration_cap,
        calibration_folds=calibration_folds, trees=trees,
        max_gp_points=max_gp_points, seed=seed, resume=resume,
    )


def run_matern_tuning(*, tune_months=("2025-05", "2025-06", "2025-07"),
                      trials_per_family=20, **context_kwargs):
    """Select GP-only hyperparameters on earlier chronological folds.

    First trial of each family is the historical unoptimized kernel to
    ensure a tuned model never loses access to that reference configuration.
    Primary tuning objective is .65*MdAPE + .35*MAPE on validation homes
    strictly after each fold's model/calibration data. Every trial reuses
    exactly the same cached XGBoost predictions and GP calibration homes.
    """
    if not isinstance(trials_per_family, int) or trials_per_family < 1:
        raise ValueError("trials_per_family must be >= 1")
    periods = _months(tune_months)
    ctx = _context(tune_months=tune_months, **context_kwargs)
    folds = [ctx.fold(m) for m in periods]
    root = ctx.root
    baseline_apes = [
        _ape(f["actual"], f["baseline"]) for f in folds
    ]
    rows, selections = [], {}
    for kind in KINDS:
        print(f"\nTuning {kind} Matérn: {trials_per_family} trials "
              f"across {len(folds)} earlier folds", flush=True)
        rng = np.random.default_rng(ctx.seed + (0 if kind == "spatial" else 1000))
        candidates = [dict(DEFAULT_KERNEL)] + [
            _proposal(rng) for _ in range(trials_per_family - 1)
        ]
        for i, params in enumerate(candidates):
            t0 = perf_counter()
            try:
                new_ape = [
                    _ape(f["actual"], _predict_corrected(
                        f, kind, params, max_gp_points=ctx.max_gp_points,
                        seed=ctx.seed,
                    ))
                    for f in folds
                ]
                old = np.concatenate(baseline_apes)
                new = np.concatenate(new_ape)
                mdape = float(np.median(new))
                mape = float(np.mean(new))
                dmdape = mdape - float(np.median(old))
                dmape = mape - float(np.mean(old))
                score = 0.65 * mdape + 0.35 * mape
                row = {
                    "kind": kind, "trial": i, "objective": score,
                    "mdape": mdape, "mape": mape,
                    "delta_mdape_pp": dmdape, "delta_mape_pp": dmape,
                    "months_better_mdape": sum(
                        np.median(a) < np.median(b)
                        for a, b in zip(new_ape, baseline_apes)
                    ),
                    "runtime_seconds": perf_counter() - t0,
                    **params,
                }
                rows.append(row)
                if kind not in selections or score < selections[kind]["objective"]:
                    selections[kind] = dict(row)
                print(f"  {kind} trial {i+1}/{trials_per_family}: "
                      f"MdAPE {mdape:.4f}% MAPE {mape:.4f}% "
                      f"objective {score:.4f}", flush=True)
            except Exception as exc:
                rows.append({
                    "kind": kind, "trial": i, "objective": float("inf"),
                    "failure": f"{type(exc).__name__}: {exc}", **params,
                })
                print(f"  {kind} trial {i+1} failed: {exc}", flush=True)
            _save(root / "trials.csv", pd.DataFrame(rows))
        if kind not in selections:
            raise RuntimeError(f"All {kind} Matérn kernel trials failed")

    best = {
        kind: {name: selections[kind][name] for name in DEFAULT_KERNEL}
        for kind in KINDS
    }
    metadata = {
        "tune_months": [str(m) for m in periods],
        "selection_objective": "0.65*pooled_MdAPE + 0.35*pooled_MAPE",
        "trials_per_family": trials_per_family, "best": best,
        "selected_score": {kind: selections[kind]["objective"] for kind in KINDS},
        "note": "Only earlier 2025 development sales used for kernel selection",
    }
    (root / "best_matern.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    trials = pd.DataFrame(rows).sort_values(["kind", "objective"])
    print("\nBEST CHRONOLOGICALLY TUNED MATÉRN PARAMETERS")
    print(pd.DataFrame(selections).T.to_string(), flush=True)
    return trials, metadata


def run_matern_confirmation(*, confirm_months=("2025-09", "2025-10", "2025-11"),
                            tune_months=("2025-05", "2025-06", "2025-07"),
                            bootstrap_repeats=500, **context_kwargs):
    """Apply *frozen* earlier-tuned kernels to later development folds.

    This is a follow-up on already-inspected Sep–Nov months, NOT an
    untouched out-of-sample final evaluation. The ordinary tensor GAM
    remains matched to these exact XGBoost folds.
    """
    periods = _months(confirm_months)
    tuned = _months(tune_months)
    if periods[0] <= tuned[-1]:
        raise ValueError("Confirmation folds must be strictly later than tuning")
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats cannot be negative")
    ctx = _context(tune_months=tune_months, **context_kwargs)
    best_path = ctx.root / "best_matern.json"
    if not best_path.is_file():
        raise FileNotFoundError(
            "Run run_matern_tuning first; the selected GP parameters are missing"
        )
    meta = json.loads(best_path.read_text(encoding="utf-8"))
    if meta["tune_months"] != [str(m) for m in tuned]:
        raise ValueError("Matérn parameters came from different tuning months")
    saved, monthly_rows = [], []
    for month in periods:
        f = ctx.fold(month)
        variants = {"baseline": f["baseline"]}
        gam_corr = SpaceTimeResidual("tensor_gam", seed=ctx.seed).fit(
            f["calibration"], f["calibration"].log_error.to_numpy(dtype=float)
        ).predict(f["hidden"])
        variants["xgb_tensor_gam"] = f["baseline"] * np.exp(
            0.5 * np.clip(gam_corr, -.25, .25)
        )
        for kind in KINDS:
            variants[f"xgb_{kind}_matern_tuned"] = _predict_corrected(
                f, kind, meta["best"][kind],
                max_gp_points=ctx.max_gp_points, seed=ctx.seed,
            )
        for variant, pred in variants.items():
            saved.append(_frame(month, f["held"], variant, pred, 0.0))
        print(f"{month}: confirmed on {len(f['held']):,} held-out homes",
              flush=True)

    pooled, monthly = _report(
        [pd.concat(saved, ignore_index=True)], months=periods,
        repeats=bootstrap_repeats, seed=ctx.seed,
    )
    preds = pd.concat(saved, ignore_index=True)
    # Additional paired comparison with tensor GAM (same exact properties).
    ref = preds[preds.variant == "xgb_tensor_gam"].set_index(
        ["month", "ml_num"]
    ).sort_index()
    p = _ape(ref.sold_price.to_numpy(dtype=float),
             ref.predicted_price.to_numpy(dtype=float))
    group_ids = np.asarray(ref.index.get_level_values("month"), dtype=str)
    groups = [
        np.flatnonzero(group_ids == str(month)) for month in periods
    ]
    rng = np.random.default_rng(ctx.seed)
    # Use precisely the same stratified resampling indices for all models.
    bootstrap_indices = [
        np.concatenate([
            rng.choice(group, len(group), replace=True)
            for group in groups
        ])
        for _ in range(bootstrap_repeats)
    ]
    for i, row in pooled.iterrows():
        other = preds[preds.variant == row.variant].set_index(
            ["month", "ml_num"]
        ).loc[ref.index]
        q = _ape(ref.sold_price.to_numpy(dtype=float),
                 other.predicted_price.to_numpy(dtype=float))
        pooled.loc[i, "delta_vs_tensor_mdape_pp"] = (
            np.median(q) - np.median(p)
        )
        pooled.loc[i, "delta_vs_tensor_mape_pp"] = np.mean(q-p)
        if bootstrap_indices:
            dmd = [
                np.median(q[j]) - np.median(p[j])
                for j in bootstrap_indices
            ]
            dma = [np.mean(q[j] - p[j]) for j in bootstrap_indices]
            pooled.loc[i, "tensor_mdape_ci_low"] = np.quantile(dmd, .025)
            pooled.loc[i, "tensor_mdape_ci_high"] = np.quantile(dmd, .975)
            pooled.loc[i, "tensor_mape_ci_low"] = np.quantile(dma, .025)
            pooled.loc[i, "tensor_mape_ci_high"] = np.quantile(dma, .975)
    _save(ctx.root / "confirmation_predictions.csv", preds)
    _save(ctx.root / "confirmation_pooled.csv", pooled)
    _save(ctx.root / "confirmation_monthly.csv", monthly)
    print("\nFROZEN MATÉRN CONFIRMATION VS SAME XGBOOST AND TENSOR GAM")
    print(pooled[[
        "variant", "homes", "mdape", "mape",
        "delta_mdape_pp", "delta_mape_pp",
        "delta_vs_tensor_mdape_pp", "delta_vs_tensor_mape_pp",
    ]].to_string(index=False), flush=True)
    return pooled, monthly, preds
