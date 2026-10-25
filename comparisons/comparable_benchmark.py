"""Controlled comparable-sale selection ablation, with no changes to model defaults.

Four variants use the SAME model/feature hyperparameters:
  baseline     - nearest 128 geographically; age filter; weighted top 32
  age_first    - eligible age window FIRST, then nearest 128
  type_quota   - baseline search + up to 8 guaranteed same-type neighbors
  both         - age-first + same-type quota

Runs on earlier (before December 2025) chronological development folds by
default. December 2025-May 2026 confirmation remains in model_run.ipynb.
The results are exploratory and are not automatically written to logs.md.
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
from models.model import OffMarketModel
from models.direct_listing import DirectListingModel

BACKENDS = {
    "xgboost": XGBoostRegressor,
    "lightgbm": LightGBMRegressor,
    "catboost": CatBoostRegressor,
}

COMP_VARIANTS = {
    "baseline": "",
    "age_first": "__comp_age_first",
    "type_quota": "__comp_type_quota",
    "both": "__comp_both",
}

COMP_DIAGNOSTICS = (
    "comp_log_price",
    "comp_count",
    "comp_effective_n",
    "comp_same_type_share",
    "comp_nearest_km",
    "comp_mean_age_days",
)


def _scores(df):
    signed = 100.0 * (df["predicted_price"].to_numpy(dtype=float) -
                      df["sold_price"].to_numpy(dtype=float)) / df["sold_price"].to_numpy(dtype=float)
    return {
        "homes": len(df),
        "mape": float(np.mean(np.abs(signed))),
        "mdape": float(np.median(np.abs(signed))),
        "median_signed": float(np.median(signed)),
    }


def run_comparable_benchmark(
    *, backend="xgboost", track="off_market",
    text_feature="Xpt_spatiotemporal__selected_binary",
    params=None, months=("2025-10",), variants=tuple(COMP_VARIANTS),
    output_dir="outputs/comparable_benchmark", listings=None,
):
    """Evaluate four comparable algorithms on identical homes and model settings.

    The same training cutoff, property/text features, and estimator parameters
    are used for every variation. Train/validation outcomes are never used
    to choose comparable candidates. Results are flushed after each month.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown boosting backend: {backend}")
    if track not in ("off_market", "list_aware"):
        raise ValueError(f"Unknown track: {track}")
    if not text_feature.startswith("Xpt_spatiotemporal"):
        raise ValueError("text_feature must be a spatiotemporal feature name")
    if any(text_feature.endswith("__" + s) for s in
           ("comp_age_first", "comp_type_quota", "comp_both")):
        raise ValueError("Pass a base text_feature without a comparable suffix")
    variants = tuple(variants)
    if len(set(variants)) != len(variants) or not variants or any(
        variant not in COMP_VARIANTS for variant in variants
    ):
        raise ValueError(f"Select unique variants from {tuple(COMP_VARIANTS)}")
    periods = pd.PeriodIndex(months, freq="M")
    if (not len(periods) or
        len(periods) != len(set(periods)) or
        not periods.is_monotonic_increasing or
        any(period >= pd.Period("2025-12", freq="M") for period in periods)):
        raise ValueError("Use unique chronological development months before 2025-12")

    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales["date_sold"])
    sales["date_listed"] = pd.to_datetime(sales["date_listed"])
    params = dict(params or {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{backend}_{track}"

    rows = []
    monthly = []
    for month in periods:
        train = sales.loc[sales["date_sold"] < month.start_time]
        holdout = sales.loc[
            sales["date_sold"].dt.to_period("M") == month
        ]
        if train.empty or holdout.empty:
            raise ValueError(f"Missing training or validation homes for {month}")

        homes = data.hide_answers(holdout, track)
        print(f"{month}: {len(train):,} prior sales -> {len(homes):,} homes",
              flush=True)
        for variant in variants:
            feature_name = text_feature + COMP_VARIANTS[variant]
            spec = [(BACKENDS[backend], feature_name, params)]
            if track == "off_market":
                model = OffMarketModel(spec, center_target=True)
            else:
                model = DirectListingModel(target="ratio", modelspecs=spec)
            model.fit(train)
            predicted = model.predict(homes)

            # The provider is fitted already. Transforming only the ~2k
            # validation homes retrieves comparable diagnostics without
            # refitting a second model or changing its predictions.
            provider = model.components[0][1]
            diagnostics = provider.transform(homes)
            result = pd.DataFrame({
                "ml_num": holdout["ml_num"].to_numpy(),
                "month": str(month), "track": track, "backend": backend,
                "variant": variant, "house_type": holdout["house_type_name"].to_numpy(),
                "sold_price": holdout["sold_price"].to_numpy(dtype=float),
                "predicted_price": predicted,
            })
            for col in COMP_DIAGNOSTICS:
                result[col] = diagnostics[col].to_numpy(dtype=float)
            result["abs_pct_error"] = (
                100 * np.abs(result["predicted_price"] - result["sold_price"]) /
                result["sold_price"]
            )
            rows.append(result)

            score = _scores(result)
            quality = {
                "zero_same_type_rate": float(
                    (result["comp_same_type_share"].fillna(0) <= 0).mean()
                ),
                "no_comp_rate": float(
                    (result["comp_count"].fillna(0) < 2).mean()
                ),
                "median_effective_n": float(
                    result["comp_effective_n"].median()
                ),
            }
            monthly.append({"month": str(month), "variant": variant, **score, **quality})
            print(
                f"  {variant:12s} MdAPE {score['mdape']:.3f}%  "
                f"MAPE {score['mape']:.3f}%  "
                f"zero-same-type {quality['zero_same_type_rate']:.1%}",
                flush=True,
            )

        # Flush partial results safely after every completed month.
        pd.concat(rows, ignore_index=True).to_csv(
            output_dir / f"{prefix}_predictions.csv", index=False
        )
        pd.DataFrame(monthly).to_csv(
            output_dir / f"{prefix}_monthly.csv", index=False
        )

    predictions = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame([
        {"variant": variant, **_scores(part),
         "zero_same_type_rate": float(
             (part["comp_same_type_share"].fillna(0) <= 0).mean()
         )}
        for variant, part in predictions.groupby("variant", sort=False)
    ]).sort_values(["mdape", "mape"]).reset_index(drop=True)

    base = summary.loc[summary.variant == "baseline"]
    if not base.empty:
        summary["delta_mdape_pp"] = summary["mdape"] - float(base.mdape.iloc[0])
        summary["delta_mape_pp"] = summary["mape"] - float(base.mape.iloc[0])
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    (output_dir / f"{prefix}_setup.json").write_text(
        json.dumps({
            "backend": backend, "track": track, "text_feature": text_feature,
            "params": params, "months": [str(m) for m in periods],
            "variants": list(variants),
            "candidate_k": 128, "neighbor_k": 32,
            "age_window_days": 730,
            "minimum_same_type_neighbors": 8,
            "note": "Development folds; no December-May selection",
        }, indent=2), encoding="utf-8",
    )
    print("\nPooled results:\n", summary.to_string(index=False), flush=True)
    print(f"Saved in {output_dir.resolve()}", flush=True)
    return summary, pd.DataFrame(monthly), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="xgboost")
    parser.add_argument("--track", choices=("off_market", "list_aware"),
                        default="off_market")
    parser.add_argument("--months", nargs="+", default=["2025-10"])
    parser.add_argument("--variants", nargs="+", choices=tuple(COMP_VARIANTS),
                        default=list(COMP_VARIANTS))
    parser.add_argument("--text-feature",
                        default="Xpt_spatiotemporal__selected_binary")
    parser.add_argument("--params", type=Path, required=True,
                        help="JSON file containing your Optuna-tuned model parameters")
    parser.add_argument("--output-dir", default="outputs/comparable_benchmark")
    args = parser.parse_args()
    params = json.loads(args.params.read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        parser.error("--params JSON must be an object")
    run_comparable_benchmark(
        backend=args.backend, track=args.track,
        text_feature=args.text_feature, months=args.months,
        variants=args.variants, params=params, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
