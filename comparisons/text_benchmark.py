"""Chronological, matched-configuration text representation benchmark.

Run from repository root:
  uv run python text_benchmark.py --backend lightgbm --track off_market
  uv run python text_benchmark.py --backend xgboost --track off_market --params tuned_xgb.json

Every representation uses the same structured and historical comparable-sale
features, the same estimator hyperparameters, and the same monthly train/test
split. Comparables are computed once per fold, not once per representation.
December 2025-May 2026 is NOT used by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
from models.boost_regressor import (
    CatBoostRegressor, LightGBMRegressor, XGBoostRegressor,
)
from features.spatiotemporal_features import ComparableSalesFeatures
from features.text_features import TEXT_VARIANTS, make_text_base, training_vocabulary

BACKENDS = {
    "lightgbm": LightGBMRegressor,
    "xgboost": XGBoostRegressor,
    "catboost": CatBoostRegressor,
}
DEFAULT_MONTHS = ("2025-09", "2025-10", "2025-11")


def prediction_error(sold, prediction):
    sold = np.asarray(sold, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not (len(sold) and (sold > 0).all() and np.isfinite(prediction).all()):
        raise ValueError("Expected positive sold prices and finite predictions")
    signed = 100.0 * (prediction - sold) / sold
    return {
        "homes": int(len(sold)),
        "mape": float(np.mean(np.abs(signed))),
        "mdape": float(np.median(np.abs(signed))),
        "median_signed": float(np.median(signed)),
    }


def _one_fit(train, homes, *, backend, track, variant, params, comps_train,
             comps_homes, full_vocabulary, center_target):
    base = make_text_base(
        variant, include_time=True, full_vocabulary=full_vocabulary
    )
    X_train = base.fit_transform(train, model=backend, track=track)
    X_homes = base.transform(homes, model=backend)
    X_train = pd.concat([X_train, comps_train], axis=1)
    X_homes = pd.concat([X_homes, comps_homes], axis=1)

    if track == "list_aware":
        # Match RatioComparableFeatures._with_list_discrepancy exactly.
        for X, rows in ((X_train, train), (X_homes, homes)):
            listed = pd.to_numeric(rows["list_price"], errors="coerce")
            if (~np.isfinite(listed) | (listed <= 0)).any():
                raise ValueError("list_price must be positive and finite")
            X["comp_vs_list_log"] = (
                X["comp_log_price"] - np.log(listed)
            ).astype("float32")

    sold = train["sold_price"].to_numpy(dtype=float)
    if not np.isfinite(sold).all() or (sold <= 0).any():
        raise ValueError("sold_price must be finite and positive")
    y = np.log(sold)
    if track == "off_market":
        offset = float(np.median(y)) if center_target else 0.0
        y = y - offset
    else:
        offset = 0.0
        y = y - np.log(train["list_price"].to_numpy(dtype=float))
    component = BACKENDS[backend](**params)
    component.fit(X_train, y)
    predicted_log = np.asarray(component.predict(X_homes), dtype=float).reshape(-1)
    if track == "off_market":
        predicted_log += offset
    else:
        predicted_log += np.log(homes["list_price"].to_numpy(dtype=float))
    return np.exp(predicted_log)


def run_benchmark(
    *, backend="lightgbm", track="off_market", months=DEFAULT_MONTHS,
    variants=TEXT_VARIANTS, params=None, output_dir="outputs/text_benchmark",
    center_target=True, listings=None,
):
    """Run controlled earlier-period experiments; return pooled and monthly scores.

    `listings` is injectable to facilitate reproducible synthetic-data tests.
    Each feature transformer and model is fitted afresh per fold and variant.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    if track not in ("off_market", "list_aware"):
        raise ValueError(f"Unsupported track: {track}")
    variants = tuple(variants)
    if len(set(variants)) != len(variants) or any(v not in TEXT_VARIANTS for v in variants):
        raise ValueError(f"Select unique variants from {TEXT_VARIANTS}")
    if not variants:
        raise ValueError("At least one variant is required")
    periods = pd.PeriodIndex(months, freq="M")
    if len(periods) != len(set(periods)) or not periods.is_monotonic_increasing:
        raise ValueError("Months must be unique and chronological")
    if any(m >= pd.Period("2025-12", freq="M") for m in periods):
        raise ValueError(
            "Development runner is restricted to months before December 2025; "
            "use model_run.ipynb for final December-May confirmation."
        )
    listings = data.load("listings") if listings is None else listings.copy()
    listings["date_sold"] = pd.to_datetime(listings["date_sold"])
    listings["date_listed"] = pd.to_datetime(listings["date_listed"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_params = dict(params or {})
    output_rows = []
    monthly_rows = []
    records = []

    for month in periods:
        train = listings.loc[listings["date_sold"] < month.start_time]
        holdout = listings.loc[listings["date_sold"].dt.to_period("M") == month]
        if train.empty or holdout.empty:
            raise ValueError(f"Empty training or validation set for {month}")
        homes = data.hide_answers(holdout, track)
        print(f"\n{month}: {len(train):,} earlier sales -> {len(homes):,} predictions", flush=True)

        # Common historical statistics computed once for all representations.
        comps = ComparableSalesFeatures(
            candidates=128, neighbors=32, radius_km=4.0, half_life_days=180
        )
        comps_train = comps.fit_transform(train)
        comps_homes = comps.transform(homes)

        # One UNSUPERVISED training-fitted vocabulary shared by all 1k variants.
        full_vocabulary = (
            training_vocabulary(train["remarks"].fillna("").astype(str))
            if any(v.startswith("full_") for v in variants)
            else None
        )
        records.append({
            "month": str(month), "train_size": len(train),
            "validation_size": len(holdout),
            "full_vocabulary": list(full_vocabulary) if full_vocabulary else [],
        })

        for variant in variants:
            predicted = _one_fit(
                train, homes, backend=backend, track=track,
                variant=variant, params=base_params,
                comps_train=comps_train, comps_homes=comps_homes,
                full_vocabulary=full_vocabulary,
                center_target=center_target,
            )
            row = pd.DataFrame({
                "ml_num": holdout["ml_num"].to_numpy(),
                "month": str(month),
                "backend": backend,
                "track": track,
                "variant": variant,
                "sold_price": holdout["sold_price"].to_numpy(dtype=float),
                "predicted_price": predicted,
            })
            output_rows.append(row)
            score = prediction_error(row["sold_price"], row["predicted_price"])
            monthly_rows.append({"month": str(month), "variant": variant, **score})
            print(f"  {variant:<18} MdAPE {score['mdape']:6.3f}%  "
                  f"MAPE {score['mape']:6.3f}%", flush=True)

        # Flush after every fold, preserving useful results if a later fold fails.
        prefix = f"{backend}_{track}"
        pd.concat(output_rows, ignore_index=True).to_csv(
            output_dir / f"{prefix}_predictions.csv", index=False
        )
        pd.DataFrame(monthly_rows).to_csv(
            output_dir / f"{prefix}_monthly.csv", index=False
        )

    predictions = pd.concat(output_rows, ignore_index=True)
    pooled = pd.DataFrame([
        {"variant": variant, **prediction_error(
            part["sold_price"].to_numpy(), part["predicted_price"].to_numpy()
        )}
        for variant, part in predictions.groupby("variant", sort=False)
    ]).sort_values(["mdape", "mape"]).reset_index(drop=True)
    prefix = f"{backend}_{track}"
    pooled.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    (output_dir / f"{prefix}_setup.json").write_text(
        json.dumps({
            "backend": backend,
            "track": track,
            "months": [str(m) for m in periods],
            "variants": list(variants),
            "params_overrides": base_params,
            "center_target": center_target,
            "fold_vocabulary": records,
            "note": "Earlier chronological development only, not final validation",
        }, indent=2), encoding="utf-8"
    )
    print("\nPooled results (lower is better):\n", pooled.to_string(index=False), flush=True)
    print(f"Saved in {output_dir.resolve()}", flush=True)
    return pooled, pd.DataFrame(monthly_rows), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="lightgbm")
    parser.add_argument("--track", choices=("off_market", "list_aware"),
                        default="off_market")
    parser.add_argument("--months", nargs="+", default=list(DEFAULT_MONTHS))
    parser.add_argument("--variants", nargs="+", choices=TEXT_VARIANTS,
                        default=list(TEXT_VARIANTS))
    parser.add_argument("--params", type=Path, help="JSON file of model parameter overrides")
    parser.add_argument("--output-dir", default="outputs/text_benchmark")
    parser.add_argument("--no-center-target", action="store_true")
    args = parser.parse_args()
    params = json.loads(args.params.read_text(encoding="utf-8")) if args.params else {}
    if not isinstance(params, dict):
        parser.error("--params file must contain a JSON object")
    if not args.params:
        print("NOTE: No --params JSON supplied. Using repository DEFAULT_PARAMS, "
              "not your Optuna-tuned settings.")
    run_benchmark(
        backend=args.backend, track=args.track, months=args.months,
        variants=args.variants, params=params, output_dir=args.output_dir,
        center_target=not args.no_center_target,
    )


if __name__ == "__main__":
    main()
