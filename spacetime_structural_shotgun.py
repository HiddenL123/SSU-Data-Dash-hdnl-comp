"""Chronological shotgun: genuine spatial models vs matched tuned XGBoost.

Pilot on pre-Dec-2025 development months only. A cheap screen does NOT prove
a method cannot beat the 700-tree full-history incumbent; it prioritizes
candidates for expensive full-history evaluation.

Quick smoke:
  python spacetime_structural_shotgun.py --months 2025-10 --train-cap 5000 --trees 250

Three-month research screen:
  python spacetime_structural_shotgun.py --months 2025-09 2025-10 2025-11
"""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import data
from spacetime_fast import _cap, XGB_TUNED
from spacetime_combination_screen import _predict as predict_xgb
from models.spacetime_structural_residuals import local_kernel, matern_gp
from models.spacetime_gpboost import gpboost_joint

VERSION = 1
METHODS = (
    "xgb_st_kernel",
    "xgb_st_kernel_type",
    "xgb_spatial_matern",
    "xgb_st_matern",
    "gpboost_vecchia",
)
INDEX = ["month", "ml_num"]


def _frame(month, held, method, estimated, seconds):
    y = held.sold_price.to_numpy(dtype=float)
    p = np.asarray(estimated, dtype=float).reshape(-1)
    if (len(p) != len(y) or not np.isfinite(p).all() or
        (p <= 0).any() or not np.isfinite(y).all() or (y <= 0).any()):
        raise ValueError("Predictions and actual prices must be positive and aligned")
    return pd.DataFrame({
        "month": str(month), "ml_num": held.ml_num.astype(str).to_numpy(),
        "variant": method, "sold_price": y, "predicted_price": p,
        "seconds": float(seconds),
    })


def _save(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def _cache(path, held, month, variant):
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, dtype={"ml_num": str, "month": str})
        if (len(df) != len(held) or df.ml_num.duplicated().any() or
            not df.month.eq(str(month)).all() or
            not df.variant.eq(variant).all()):
            return None
        expected = held.ml_num.astype(str)
        aligned = df.set_index("ml_num").loc[expected]
        if not np.array_equal(aligned.sold_price.to_numpy(dtype=float),
                              held.sold_price.to_numpy(dtype=float)):
            return None
        p = aligned.predicted_price.to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p <= 0).any():
            return None
        return df
    except (KeyError, ValueError, TypeError, pd.errors.ParserError):
        return None


def _report(records, *, months, repeats, seed):
    raw = pd.concat(records, ignore_index=True)
    base = raw[raw.variant == "baseline"].set_index(INDEX).sort_index()
    if base.empty or base.index.duplicated().any():
        raise ValueError("Bad baseline keys")
    rows, monthly = [], []
    errors = {}
    for m in months:
        ref = base.xs(str(m), level="month")
        y = ref.sold_price.to_numpy(dtype=float)
        errors[str(m)] = 100 * np.abs(ref.predicted_price.to_numpy(dtype=float)-y)/y
    rng = np.random.default_rng(seed)
    for name, original in raw.groupby("variant", sort=False):
        chunk = original.set_index(INDEX).sort_index()
        if not chunk.index.equals(base.index):
            raise ValueError(f"Misaligned holdout homes for {name}")
        y = base.sold_price.to_numpy(dtype=float)
        a = 100 * np.abs(base.predicted_price.to_numpy(dtype=float)-y)/y
        b = 100 * np.abs(chunk.predicted_price.to_numpy(dtype=float)-y)/y
        md = float(np.median(b) - np.median(a))
        ma = float(np.mean(b-a))
        better = 0
        worse_both = 0
        for m in months:
            left = base.xs(str(m), level="month")
            right = chunk.xs(str(m), level="month")
            truth = left.sold_price.to_numpy(dtype=float)
            ea = 100*np.abs(left.predicted_price.to_numpy(dtype=float)-truth)/truth
            eb = 100*np.abs(right.predicted_price.to_numpy(dtype=float)-truth)/truth
            dmd = float(np.median(eb)-np.median(ea))
            dma = float(np.mean(eb-ea))
            better += dmd < 0
            worse_both += dmd > 0 and dma > 0
            monthly.append({
                "month": str(m), "variant": name, "homes": len(truth),
                "mdape": float(np.median(eb)), "mape": float(np.mean(eb)),
                "delta_mdape_pp": dmd, "delta_mape_pp": dma,
            })
        idx_groups = [
            np.flatnonzero(
                np.asarray(base.index.get_level_values("month")) == str(m)
            ) for m in months
        ]
        draws = np.empty((repeats, 2))
        for i in range(repeats):
            ind = np.concatenate([
                rng.choice(group, size=len(group), replace=True)
                for group in idx_groups
            ])
            draws[i] = (
                np.median(b[ind])-np.median(a[ind]),
                np.mean(b[ind]-a[ind]),
            )
        # Only deprioritize strongly inferior, repeatedly worse pilots.
        if name == "baseline":
            status = "reference"
        elif len(months) > 1 and better >= len(months)-1 and (
            md <= -0.08 or ma <= -0.10
        ):
            status = "promising_follow_up"
        elif len(months) >= 3 and worse_both >= len(months)-1 and (
            md >= 0.12 and ma >= 0.12
        ):
            status = "deprioritize_pilot"
        else:
            status = "inconclusive"
        rows.append({
            "variant": name, "homes": len(y), "mdape": np.median(b),
            "mape": np.mean(b), "delta_mdape_pp": md,
            "delta_mape_pp": ma,
            "homes_improved_pct": 100*np.mean(b<a),
            "months_better_mdape": better,
            "runtime_seconds": float(chunk.groupby(level="month").seconds.first().sum()),
            "mdape_ci_low": np.quantile(draws[:, 0], .025) if repeats else np.nan,
            "mdape_ci_high": np.quantile(draws[:, 0], .975) if repeats else np.nan,
            "mape_ci_low": np.quantile(draws[:, 1], .025) if repeats else np.nan,
            "mape_ci_high": np.quantile(draws[:, 1], .975) if repeats else np.nan,
            "triage": status,
        })
    return pd.DataFrame(rows).sort_values(["mdape", "mape"]), pd.DataFrame(monthly)


def run_structural_shotgun(
    *, months=("2025-09", "2025-10", "2025-11"),
    train_start="2022-01-01", train_cap=8000,
    validation_cap=900, calibration_cap=450,
    calibration_folds=2, trees=500, gpboost_rounds=220,
    max_gp_points=350, residual_strength=0.5,
    methods=METHODS, seed=42, bootstrap_repeats=300,
    output_dir="outputs/spacetime_structural_shotgun",
    resume=True, listings=None,
):
    periods = pd.PeriodIndex(months, freq="M")
    if (not len(periods) or periods.has_duplicates or
        not periods.is_monotonic_increasing or
        any(m >= pd.Period("2025-12", freq="M") for m in periods)):
        raise ValueError("Only distinct chronological pre-Dec-2025 months allowed")
    methods = tuple(methods)
    if len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Unknown or duplicate methods")
    if (min(train_cap, validation_cap, calibration_cap, calibration_folds,
            trees, gpboost_rounds, max_gp_points) < 1 or
        bootstrap_repeats < 0 or not 0 <= residual_strength <= 1):
        raise ValueError("Invalid training/screen budget")
    start = pd.Timestamp(train_start)
    if start >= (periods[0]-calibration_folds).start_time:
        raise ValueError("train_start must precede first calibration month")
    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales.date_sold)
    sales["date_listed"] = pd.to_datetime(sales.date_listed)
    if sales.ml_num.isna().any():
        raise ValueError("Missing property identifiers")

    setup = dict(
        version=VERSION, months=[str(m) for m in periods],
        train_start=str(start.date()), train_cap=train_cap,
        validation_cap=validation_cap, calibration_cap=calibration_cap,
        calibration_folds=calibration_folds, trees=trees,
        gpboost_rounds=gpboost_rounds, max_gp_points=max_gp_points,
        residual_strength=residual_strength, methods=list(methods),
        seed=seed, bootstrap_repeats=bootstrap_repeats, xgb_params=XGB_TUNED,
    )
    # Store an input fingerprint; do not resume experiments on changed rows.
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(sales, index=True)
        .to_numpy(dtype="uint64").tobytes()
    ).hexdigest()
    setup["input_sha256"] = digest
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "setup.json"
    if resume and manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != setup:
            raise ValueError("Output folder has different data/settings; choose a new path")
    manifest.write_text(json.dumps(setup, indent=2), encoding="utf-8")
    sale_month = sales.date_sold.dt.to_period("M")
    calibration_cache = {}
    records, failures = [], []

    def training(month):
        return _cap(sales.loc[(sales.date_sold >= start) &
                              (sales.date_sold < month.start_time)],
                    train_cap, seed)

    def xgb_at(month, held, role):
        # Separate calibration/validation caches: same calendar month can be
        # used with different sampled homes in successive origins.
        path = root / "by_month" / str(month) / f"xgb_{role}.csv"
        cached = _cache(path, held, month, "baseline") if resume else None
        if cached is not None:
            print(f"  {month} XGBoost {role}: cached", flush=True)
            return cached
        past = training(month)
        if past.empty:
            raise ValueError(f"No pre-{month} training sales")
        t0 = perf_counter()
        estimated, _ = predict_xgb(
            past, data.hide_answers(held, "off_market"),
            suffix="", params=XGB_TUNED, trees=trees,
        )
        result = _frame(month, held, "baseline", estimated, perf_counter()-t0)
        _save(path, result)
        return result

    def past_errors(month):
        if month not in calibration_cache:
            prior = _cap(sales.loc[sale_month == month], calibration_cap,
                         seed + month.ordinal % 97)
            if prior.empty:
                raise ValueError(f"No residual calibration sales for {month}")
            predicted = xgb_at(month, prior, "calibration").set_index("ml_num")
            hidden = data.hide_answers(prior, "off_market").reset_index(drop=True)
            p = predicted.loc[
                prior.ml_num.astype(str), "predicted_price"
            ].to_numpy(dtype=float)
            hidden["log_error"] = np.log(
                prior.sold_price.to_numpy(dtype=float)
            ) - np.log(p)
            calibration_cache[month] = hidden
        return calibration_cache[month]

    for month in periods:
        held = _cap(sales.loc[sale_month == month],
                    validation_cap, seed+11)
        if held.empty:
            raise ValueError(f"No validation homes for {month}")
        base = xgb_at(month, held, "validation")
        records.append(base)
        hidden = data.hide_answers(held, "off_market")
        print(f"\n{month}: {len(held):,} matched off-market held-out homes", flush=True)
        for name in methods:
            path = root / "by_month" / str(month) / f"{name}.csv"
            cached = _cache(path, held, month, name) if resume else None
            if cached is not None:
                print(f"  {name}: cached", flush=True)
                records.append(cached)
                continue
            t0 = perf_counter()
            try:
                if name == "gpboost_vecchia":
                    prediction = gpboost_joint(
                        training(month), hidden, rounds=gpboost_rounds, seed=seed,
                    )
                else:
                    calibration = pd.concat(
                        [past_errors(month-lag)
                         for lag in range(calibration_folds, 0, -1)],
                        ignore_index=True,
                    )
                    r = calibration.log_error.to_numpy(dtype=float)
                    if name in ("xgb_st_kernel", "xgb_st_kernel_type"):
                        correction = local_kernel(
                            calibration, r, hidden,
                            same_type=(name == "xgb_st_kernel_type"),
                        )
                    else:
                        correction = matern_gp(
                            calibration, r, hidden,
                            spatial_only=(name == "xgb_spatial_matern"),
                            max_points=max_gp_points, seed=seed,
                        )
                    base_prices = base.set_index("ml_num").loc[
                        held.ml_num.astype(str), "predicted_price"
                    ].to_numpy(dtype=float)
                    prediction = base_prices * np.exp(
                        residual_strength * np.clip(correction, -.25, .25)
                    )
                result = _frame(
                    month, held, name, prediction, perf_counter()-t0,
                )
                records.append(result)
                _save(path, result)
                print(f"  {name}: {result.seconds.iloc[0]:.1f}s", flush=True)
            except Exception as exc:
                failures.append({
                    "month": str(month), "variant": name,
                    "error_type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(limit=6),
                })
                print(f"  {name} UNAVAILABLE/FAILED: {exc}", flush=True)
        _save(root / "predictions_partial.csv",
              pd.concat(records, ignore_index=True))

    result = pd.concat(records, ignore_index=True)
    # Only rank candidates with results on ALL requested months.
    completed = ["baseline"]
    for name in methods:
        if set(result.loc[result.variant == name, "month"]) == set(
            str(m) for m in periods
        ):
            completed.append(name)
    matched = result[result.variant.isin(completed)]
    pooled, monthly = _report(
        [matched], months=periods, repeats=bootstrap_repeats, seed=seed,
    )
    _save(root / "predictions.csv", result)
    _save(root / "pooled_summary.csv", pooled)
    _save(root / "monthly_summary.csv", monthly)
    if failures:
        _save(root / "failures.csv", pd.DataFrame(failures))
    print("\nDEVELOPMENT SHOTGUN (vs matched XGBoost; negative = better):")
    print(pooled[[
        "variant", "homes", "mdape", "mape",
        "delta_mdape_pp", "delta_mape_pp",
        "months_better_mdape", "runtime_seconds", "triage",
    ]].to_string(index=False), flush=True)
    print(f"Saved in {root}", flush=True)
    return pooled, monthly, result, pd.DataFrame(failures)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--months", nargs="+",
                   default=["2025-09", "2025-10", "2025-11"])
    p.add_argument("--train-cap", type=int, default=8000)
    p.add_argument("--validation-cap", type=int, default=900)
    p.add_argument("--calibration-cap", type=int, default=450)
    p.add_argument("--calibration-folds", type=int, default=2)
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--gpboost-rounds", type=int, default=220)
    p.add_argument("--max-gp-points", type=int, default=350)
    p.add_argument("--output-dir", default="outputs/spacetime_structural_shotgun")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    args = p.parse_args()
    run_structural_shotgun(
        months=args.months, train_cap=args.train_cap,
        validation_cap=args.validation_cap, calibration_cap=args.calibration_cap,
        calibration_folds=args.calibration_folds, trees=args.trees,
        gpboost_rounds=args.gpboost_rounds, max_gp_points=args.max_gp_points,
        output_dir=args.output_dir, methods=args.methods,
    )
