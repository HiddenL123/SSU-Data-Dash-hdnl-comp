"""Chronological, resumable shotgun for the LIST-AWARE housing competition track.

Reuses proven CatBoost Huber / tuned XGBoost / tuned LightGBM and transfers
spline+geo and prior-only adjusted comparable features from the off-market
experiments. Candidate fitting never sees evaluation homes' sold prices.
All predeclared log-price blends use fixed weights, not validation optimization.

PowerShell:
  py -3.13 listaware_shotgun.py
  py -3.13 listaware_shotgun.py --full --candidates cat_huber_baseline cat_huber_spline_geo xgb_spline_geo

Pilot caps use 10k sales / 300 trees / 350 validation homes per month.
Full mode uses all prior eligible sales and original recorded iteration counts.
Historical 4.06% baseline was a different cohort and is NOT a pilot comparator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import data
from features.features import make_features
from models.listaware_shotgun_specs import (
    BLENDS, CANDIDATES, DEFAULT_CANDIDATES, parameters,
)
from experiments.spacetime_fast import _cap, _metric
from experiments.spacetime_three_month import _bootstrap_delta

VERSION = 1
KEY = ["month", "ml_num"]
BASELINE = "cat_huber_baseline"


def _dataset_signature(sales):
    h = hashlib.sha256()
    h.update("|".join(sales.columns).encode())
    h.update(pd.util.hash_pandas_object(sales, index=True).to_numpy(
        dtype="uint64").tobytes())
    return h.hexdigest()


def _check_setup(months, candidates, train_cap, validation_cap, trees, full):
    if not candidates or BASELINE not in candidates:
        raise ValueError("Include cat_huber_baseline for matched evaluation")
    if len(candidates) != len(set(candidates)) or set(candidates) - set(CANDIDATES):
        raise ValueError("Unknown or repeated candidate")
    periods = pd.PeriodIndex(months, freq="M")
    if (not len(periods) or periods.has_duplicates
        or not periods.is_monotonic_increasing
        or any(p >= pd.Period("2025-12") for p in periods)):
        raise ValueError("Development months must be unique, chronological, before 2025-12")
    if min(train_cap, validation_cap, trees) < 1:
        raise ValueError("Train, evaluation, iteration caps must be positive")
    if full and trees < 700:
        raise ValueError("Full-strength mode requires at least 700 tree rounds")
    return periods


def _fit_predict(candidate, train, homes, *, tree_cap, threads, features_cache):
    """Train a fresh estimator on legal past sales and reconstruct sale price."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    key = (candidate.backend, candidate.feature_name)
    if key not in features_cache:
        provider = make_features(candidate.feature_name,
                                 model=candidate.backend, track="list_aware")
        X = provider.fit_transform(train)
        H = provider.transform(homes)
        if not X.index.equals(train.index) or not H.index.equals(homes.index):
            raise ValueError("Feature matrix rows are not aligned")
        if not X.columns.equals(H.columns):
            raise ValueError("Feature schema changed between fit and predict")
        if "sold_price" in H or "date_sold" in H:
            raise ValueError("Post-sale label leakage in holdout features")
        features_cache[key] = (X, H)
    X, H = features_cache[key]

    sold = np.asarray(train.sold_price, dtype=float)
    listed = np.asarray(train.list_price, dtype=float)
    if (not np.isfinite(sold).all() or not np.isfinite(listed).all()
        or np.any(sold <= 0) or np.any(listed <= 0)):
        raise ValueError("Invalid historical sale/list prices")
    y = np.log(sold)
    if candidate.target == "ratio":
        y -= np.log(listed)
    # Huber/MAE on uncentered log-price can be unstable. Ratio is already
    # centred near zero and matches the E28 baseline; direct target is centred.
    center = float(np.median(y)) if candidate.target == "direct" else 0.0
    params = parameters(candidate, trees=tree_cap, threads=threads)
    constructor = {
        "catboost": CatBoostRegressor,
        "xgboost": XGBRegressor,
        "lightgbm": LGBMRegressor,
    }[candidate.backend]
    model = constructor(**params)
    if candidate.backend == "catboost":
        categoricals = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        model.fit(X, y - center, cat_features=categoricals)
    else:
        model.fit(X, y - center)
    pred_log = np.asarray(model.predict(H), dtype=float).reshape(-1) + center
    if candidate.target == "ratio":
        hprice = np.asarray(homes.list_price, dtype=float)
        if (not np.isfinite(hprice).all() or (hprice <= 0).any()):
            raise ValueError("List-aware homes need positive listing prices")
        pred_log += np.log(hprice)
    predictions = np.exp(pred_log)
    if (predictions.shape != (len(homes),)
        or not np.isfinite(predictions).all()
        or (predictions <= 0).any()):
        raise ValueError("Invalid predictions")
    return predictions


def _frame(held, period, arm, variant, prediction):
    prices = held.sold_price.to_numpy(float)
    p = np.asarray(prediction, dtype=float)
    if len(p) != len(prices) or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Wrong-sized or invalid predictions")
    return pd.DataFrame({
        "month": str(period),
        "ml_num": held.ml_num.astype(str).to_numpy(),
        "arm": arm, "variant": variant,
        "house_type_name": held.house_type_name.fillna("Unknown").astype(str).to_numpy(),
        "sold_price": prices,
        "predicted_price": p,
        "ape": 100 * np.abs(p-prices)/prices,
    })


def _verified_cache(path, held, period, arm, variant):
    if not path.is_file():
        return None
    df = pd.read_csv(path, dtype={"ml_num": str})
    if set(df.columns) != {
        "month", "ml_num", "arm", "variant", "house_type_name",
        "sold_price", "predicted_price", "ape",
    } or not (
        len(df) == len(held)
        and df.month.eq(str(period)).all()
        and df.arm.eq(arm).all()
        and df.variant.eq(variant).all()
        and np.array_equal(df.ml_num.to_numpy(), held.ml_num.astype(str).to_numpy())
        and np.allclose(df.sold_price.to_numpy(float),
                        held.sold_price.to_numpy(float), rtol=0, atol=.0001)
        and np.array_equal(df.house_type_name.to_numpy(),
                           held.house_type_name.fillna("Unknown").astype(str).to_numpy())
        and np.isfinite(df.predicted_price.to_numpy(float)).all()
        and df.predicted_price.gt(0).all()
    ):
        raise ValueError(f"Invalid or stale cached predictions: {path}")
    expected_ape = 100 * abs(df.predicted_price.to_numpy(float)
                              - df.sold_price.to_numpy(float)
                              ) / df.sold_price.to_numpy(float)
    if not np.allclose(df.ape.to_numpy(float), expected_ape, rtol=1e-9, atol=1e-7):
        raise ValueError(f"Cached APE changed: {path}")
    return df


def _atomic_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _validate_all(predictions, baseline=BASELINE):
    frame = predictions.loc[
        (predictions.arm == "model") & (predictions.variant == baseline)
    ].set_index(KEY).sort_index()
    if frame.empty or not frame.index.is_unique:
        raise ValueError("Missing or repeated baseline keys")
    truth = frame.sold_price.to_numpy(float)
    for key, chunk in predictions.groupby(["arm", "variant"]):
        if (len(chunk) != len(frame) or chunk.duplicated(KEY).any()
            or not chunk.set_index(KEY).index.symmetric_difference(frame.index).empty):
            raise ValueError(f"Misaligned evaluation cohort: {key}")
        same = chunk.set_index(KEY).loc[frame.index]
        if not np.allclose(same.sold_price.to_numpy(float), truth,
                           rtol=0, atol=.0001):
            raise ValueError(f"Different actual prices: {key}")
        if not np.isfinite(same.predicted_price.to_numpy(float)).all():
            raise ValueError(f"Invalid predicted prices: {key}")
    return frame


def summarize(predictions, *, repeats=300, seed=42):
    control = _validate_all(predictions)
    y = control.sold_price.to_numpy(float)
    err = 100 * abs(control.predicted_price.to_numpy(float)-y)/y
    base = _metric(y, control.predicted_price.to_numpy(float))
    rows, monthly, by_type = [], [], []
    for (arm, name), chunk in predictions.groupby(["arm", "variant"], sort=False):
        aligned = chunk.set_index(KEY).loc[control.index]
        p = aligned.predicted_price.to_numpy(float)
        score = _metric(y, p)
        ape = 100 * abs(p-y)/y
        score.update({
            "arm": arm, "variant": name,
            "delta_mdape_pp": score["mdape"] - base["mdape"],
            "delta_mape_pp": score["mape"] - base["mape"],
            "p90_ape": float(np.percentile(ape, 90)),
            "homes_improved_fraction": float(np.mean(ape < err)),
        })
        score.update(_bootstrap_delta(err, ape, repeats=repeats, seed=seed))
        for month in sorted(predictions.month.unique()):
            mask = aligned.index.get_level_values("month") == month
            a = _metric(y[mask], p[mask])
            b = _metric(y[mask], control.predicted_price.to_numpy(float)[mask])
            monthly.append({
                "month": month, "arm": arm, "variant": name,
                **a, "delta_mdape_pp": a["mdape"]-b["mdape"],
                "delta_mape_pp": a["mape"]-b["mape"],
            })
        types = aligned.house_type_name.to_numpy(str)
        for house_type in sorted(set(types)):
            mask = types == house_type
            if mask.sum() >= 20:
                m = _metric(y[mask], p[mask])
                b = _metric(y[mask], control.predicted_price.to_numpy(float)[mask])
                by_type.append({
                    "house_type_name": house_type,
                    "arm": arm, "variant": name, **m,
                    "delta_mdape_pp": m["mdape"]-b["mdape"],
                })
        rows.append(score)
    return (pd.DataFrame(rows).sort_values(["mdape", "mape"]),
            pd.DataFrame(monthly).sort_values(["month", "mdape"]),
            pd.DataFrame(by_type).sort_values(
                ["house_type_name", "mdape"]) if by_type else pd.DataFrame())


def run_listaware_shotgun(
    *, months=("2025-08", "2025-09", "2025-10"),
    candidates=DEFAULT_CANDIDATES, full=False, mode="expanding",
    train_cap=None, validation_cap=None, trees=None,
    seed=42, threads=4, with_blends=True, bootstrap_repeats=300,
    output_dir=None, listings=None,
):
    if mode not in ("expanding", "fixed_origin"):
        raise ValueError("mode must be expanding or fixed_origin")
    candidates = tuple(candidates)
    months = tuple(months)
    train_cap = int(train_cap if train_cap is not None else (200000 if full else 10000))
    validation_cap = int(
        validation_cap if validation_cap is not None else (100000 if full else 350)
    )
    tree_cap = None if full else int(trees if trees is not None else 300)
    _check_setup(
        months, candidates, train_cap, validation_cap,
        700 if tree_cap is None else tree_cap, full
    )
    if threads < 1 or bootstrap_repeats < 0:
        raise ValueError("Invalid thread or bootstrap count")
    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales.date_sold)
    sales["date_listed"] = pd.to_datetime(sales.date_listed)
    if sales[KEY[1]].astype(str).duplicated().any():
        raise ValueError("Duplicate ml_num in source listings")
    if not (np.isfinite(sales.sold_price.to_numpy(float)).all()
            and np.isfinite(sales.list_price.to_numpy(float)).all()
            and (sales.sold_price > 0).all()
            and (sales.list_price > 0).all()):
        raise ValueError("Training needs valid sold and listing prices")
    if not set(["date_listed", "date_sold", "sold_price", "list_price"]).issubset(sales):
        raise ValueError("Missing crucial chronology/target columns")
    out = Path(output_dir or (
        f"outputs/listaware_shotgun_{'full' if full else 'pilot'}_{mode}"
    ))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": VERSION, "months": months, "candidates": candidates,
        "full": full, "mode": mode, "train_cap": train_cap,
        "validation_cap": validation_cap, "pilot_tree_cap": tree_cap,
        "seed": seed, "threads": threads, "with_blends": with_blends,
        "bootstrap_repeats": bootstrap_repeats,
        "train_start": "1900-01-01" if full else "2024-01-01",
        "dataset_sha256": _dataset_signature(sales),
        "development_only": True,
    }
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(manifest)
        ):
            raise ValueError("Experiment changed; choose a NEW output folder")
    else:
        if any(out.iterdir()):
            raise ValueError("Output folder not empty without matching manifest")
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    frames = []
    periods = pd.PeriodIndex(months, freq="M")
    for period in periods:
        cutoff = periods[0].start_time if mode == "fixed_origin" else period.start_time
        history = sales.loc[
            (sales.date_sold >= manifest["train_start"])
            & (sales.date_sold < cutoff)
        ]
        train = _cap(history, train_cap, seed)
        held = _cap(
            sales.loc[sales.date_sold.dt.to_period("M") == period],
            validation_cap, seed + 11
        )
        if train.empty or held.empty:
            raise ValueError(f"Empty training or validation homes in {period}")
        homes = data.hide_answers(held, "list_aware")
        if "list_price" not in homes or "sold_price" in homes or "date_sold" in homes:
            raise ValueError("List-aware hiding fields failed")
        if (pd.to_datetime(train.date_sold) >= cutoff).any():
            raise ValueError("Future sale labels entered training fold")
        cache = {}
        predictions = {}
        for name in candidates:
            member = CANDIDATES[name]
            file = out / str(period) / "models" / f"{name}.csv"
            stored = _verified_cache(file, held, period, "model", name)
            if stored is None:
                print(f"{period} {name}: fitting on {len(train):,} past sales "
                      f"/ {len(held):,} unseen homes", flush=True)
                start = perf_counter()
                p = _fit_predict(member, train, homes, tree_cap=tree_cap,
                                 threads=threads, features_cache=cache)
                stored = _frame(held, period, "model", name, p)
                _atomic_csv(file, stored)
                print(f"  finished: MdAPE={_metric(held.sold_price, p)['mdape']:.4f}% "
                      f"in {perf_counter()-start:.1f}s", flush=True)
            else:
                print(f"{period} {name}: cached prediction", flush=True)
            predictions[name] = stored.predicted_price.to_numpy(float)
            frames.append(stored)
        for name, weights in BLENDS.items():
            if not with_blends or not set(weights).issubset(predictions):
                continue
            file = out / str(period) / "blends" / f"{name}.csv"
            stored = _verified_cache(file, held, period, "blend", name)
            if stored is None:
                wsum = sum(weights.values())
                logblend = sum(
                    weight * np.log(predictions[member])
                    for member, weight in weights.items()
                ) / wsum
                stored = _frame(held, period, "blend", name, np.exp(logblend))
                _atomic_csv(file, stored)
            frames.append(stored)
    all_predictions = pd.concat(frames, ignore_index=True)
    pooled, monthly, by_type = summarize(
        all_predictions, repeats=bootstrap_repeats, seed=seed
    )
    _atomic_csv(out / "predictions.csv", all_predictions)
    _atomic_csv(out / "pooled_scores.csv", pooled)
    _atomic_csv(out / "monthly_scores.csv", monthly)
    _atomic_csv(out / "house_type_scores.csv", by_type)
    print("\nList-aware matched-home leaderboard:", flush=True)
    print(pooled[["arm", "variant", "homes", "mdape", "mape",
                  "delta_mdape_pp", "delta_mape_pp", "p90_ape"]]
          .to_string(index=False), flush=True)
    print(f"Saved results to: {out.resolve()}", flush=True)
    print("Caution: these months were already studied; not a fresh holdout.",
          flush=True)
    return pooled, monthly, by_type, all_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--mode", choices=("expanding", "fixed_origin"),
                        default="expanding")
    parser.add_argument("--months", nargs="+",
                        default=["2025-08", "2025-09", "2025-10"])
    parser.add_argument("--candidates", nargs="+", choices=sorted(CANDIDATES),
                        default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--train-cap", type=int, default=None)
    parser.add_argument("--validation-cap", type=int, default=None)
    parser.add_argument("--trees", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--no-blends", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_listaware_shotgun(
        full=args.full, mode=args.mode, months=args.months,
        candidates=args.candidates, train_cap=args.train_cap,
        validation_cap=args.validation_cap, trees=args.trees,
        threads=args.threads, seed=args.seed,
        bootstrap_repeats=args.bootstrap_repeats,
        with_blends=not args.no_blends, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
