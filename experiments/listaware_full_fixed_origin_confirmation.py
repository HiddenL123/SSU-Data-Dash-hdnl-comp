"""Full-strength fixed-origin confirmation of the *existing* list-aware shortlist.

Run from the repository root with its normal environment and listings dataset:
    py -3.13 listaware_full_fixed_origin_confirmation.py

One fit per base estimator using all eligible sales before 2025-08-01, full
recorded tree counts, then predictions for EVERY Aug--Oct 2025 sold listing.
The existing follow-up module defines all estimator specs and blend weights;
this script neither optimizes weights nor introduces new model candidates.

The previous 5,951-home evaluation informed model selection, so this is a
matched-protocol robustness comparison, NOT an untouched holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = ("2025-08", "2025-09", "2025-10")
WINNER = "followup_cat_mae_xgb_lgb"
# Retain earlier best alternatives and the historical reference, unchanged.
CONTROLS = (
    WINNER,
    "cat_mae",
    "blend_cat_xgb_lgb",
    "followup_winner_plus_mae",
    "blend_cat_baseline_spatial_xgb",
    "cat_huber_baseline",
    "cat_huber_spline_geo",
    "cat_huber_advanced",
)
KEY = ["month", "ml_num"]


def _metric(actual, predicted):
    if (len(actual) == 0 or len(actual) != len(predicted)
        or not np.isfinite(actual).all() or not np.isfinite(predicted).all()
        or (actual <= 0).any() or (predicted <= 0).any()):
        raise ValueError("Positive, finite, aligned predictions required")
    e = 100 * np.abs(predicted - actual) / actual
    return {"homes": len(e), "mdape": float(np.median(e)),
            "mape": float(np.mean(e)), "p90_ape": float(np.percentile(e, 90)),
            "median_signed": float(np.median(100 * (predicted - actual) / actual))}


def _paired_ci(ref_ape, contender_ape, *, repeats, seed):
    if repeats < 1:
        raise ValueError("Use at least one bootstrap repetition")
    rng = np.random.default_rng(seed)
    n = len(ref_ape)
    med = np.empty(repeats)
    mean = np.empty(repeats)
    for i in range(repeats):
        idx = rng.integers(0, n, size=n)
        med[i] = np.median(contender_ape[idx]) - np.median(ref_ape[idx])
        mean[i] = np.mean(contender_ape[idx] - ref_ape[idx])
    return {
        "delta_winner_mdape_ci_low": float(np.percentile(med, 2.5)),
        "delta_winner_mdape_ci_high": float(np.percentile(med, 97.5)),
        "delta_winner_mape_ci_low": float(np.percentile(mean, 2.5)),
        "delta_winner_mape_ci_high": float(np.percentile(mean, 97.5)),
    }


def compare_against_winner(predictions, *, candidates=CONTROLS,
                           repeats=2000, seed=42):
    """Require exactly matched month/home IDs and true prices for all arms."""
    required = {*KEY, "arm", "variant", "sold_price", "predicted_price"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing prediction columns: {sorted(required-set(predictions))}")
    frame = predictions.copy()
    frame["month"] = frame.month.astype(str)
    frame["ml_num"] = frame.ml_num.astype(str)
    if frame.duplicated(KEY + ["arm", "variant"]).any():
        raise ValueError("Duplicate month/home/model predictions")
    if not candidates or len(candidates) != len(set(candidates)) or WINNER not in candidates:
        raise ValueError("Unique selected candidates must include the prior winner")
    index = frame.loc[frame.variant.eq(WINNER)].set_index(KEY).sort_index()
    if index.empty or not index.index.is_unique:
        raise ValueError("The prior winner must be present once for every home")
    if not set(index.index.get_level_values("month")) == set(MONTHS):
        raise ValueError("Wrong month cohort; expected August--October 2025")
    actual = index.sold_price.to_numpy(float)
    ref_pred = index.predicted_price.to_numpy(float)
    _metric(actual, ref_pred)
    ref_ape = 100 * abs(ref_pred - actual) / actual

    scored, monthly = [], []
    for name in candidates:
        group = frame.loc[frame.variant.eq(name)].set_index(KEY).sort_index()
        if (group.empty or not group.index.equals(index.index)
            or not np.allclose(group.sold_price.to_numpy(float), actual,
                               rtol=0, atol=.0001)):
            raise ValueError(f"Missing, extra, or misaligned home IDs/prices: {name}")
        if group.arm.nunique() != 1:
            raise ValueError(f"Multiple arm types for {name}")
        predicted = group.predicted_price.to_numpy(float)
        stats = _metric(actual, predicted)
        ape = 100 * abs(predicted - actual) / actual
        row = {"arm": group.arm.iloc[0], "variant": name, **stats,
               "delta_winner_mdape_pp": stats["mdape"] - np.median(ref_ape),
               "delta_winner_mape_pp": stats["mape"] - np.mean(ref_ape),
               "homes_improved_vs_winner": float(np.mean(ape < ref_ape))}
        if name == WINNER:
            row.update({k: 0.0 for k in (
                "delta_winner_mdape_ci_low", "delta_winner_mdape_ci_high",
                "delta_winner_mape_ci_low", "delta_winner_mape_ci_high")})
        else:
            row.update(_paired_ci(ref_ape, ape, repeats=repeats, seed=seed))
        scored.append(row)
        for month in MONTHS:
            mask = index.index.get_level_values("month") == month
            sub = _metric(actual[mask], predicted[mask])
            monthly.append({"month": month, "variant": name, **sub,
                            "delta_winner_mdape_pp": sub["mdape"] - np.median(ref_ape[mask]),
                            "delta_winner_mape_pp": sub["mape"] - np.mean(ref_ape[mask])})
    return (pd.DataFrame(scored).sort_values(["mdape", "mape"]).reset_index(drop=True),
            pd.DataFrame(monthly).sort_values(["month", "mdape"]).reset_index(drop=True))


def _verified_full_origin_manifest(root):
    """Prevent reuse of expanding-origin, pilot, or incompatible predictions."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No fixed-origin manifest at {manifest_path}; "
            "this is NOT verified full fixed-origin evidence"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from experiments.listaware_fixed_origin_followup import CANDIDATES
    if (manifest.get("full") is not True
        or manifest.get("mode") != "single-fit_fixed_origin"
        or tuple(manifest.get("months", ())) != MONTHS
        or tuple(manifest.get("candidates", ())) != CANDIDATES
        or manifest.get("train_cutoff_exclusive") != "2025-08-01"
        or manifest.get("train_start") != "1900-01-01"
        or manifest.get("trees") is not None
        or manifest.get("train_cap") != 200000
        or manifest.get("validation_cap") != 100000
        or not manifest.get("data_sha256")):
        raise ValueError("Refusing to score a pilot or expanding-origin run")
    return manifest


def run(*, output_dir="outputs/listaware_fixed_origin_full", threads=4,
        bootstrap_repeats=2000, seed=42, from_existing=False):
    root = Path(output_dir)
    if from_existing:
        # This path neither loads the listings dataset nor fits any models.
        # It is valid ONLY when the original full-strength runner manifest
        # identifies the exact single-fit fixed-origin configuration.
        _verified_full_origin_manifest(root)
        source = root / "fixed_weight_followup" / "followup_predictions.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Expected saved follow-up predictions: {source}")
        predictions = pd.read_csv(source, dtype={"ml_num": str})
    else:
        from experiments.listaware_fixed_origin_followup import run_followup
        _, _, _, predictions = run_followup(
            full=True, months=MONTHS, output_dir=root, threads=threads,
            seed=seed, bootstrap_repeats=bootstrap_repeats,
        )
        _verified_full_origin_manifest(root)
    pooled, monthly = compare_against_winner(
        predictions, repeats=bootstrap_repeats, seed=seed,
    )
    pooled.to_csv(root / "confirmation_vs_prior_winner.csv", index=False)
    monthly.to_csv(root / "confirmation_vs_prior_winner_monthly.csv", index=False)
    print("\nFull-strength fixed-origin confirmation vs previous winner:")
    print(pooled[["variant", "homes", "mdape", "mape",
                  "delta_winner_mdape_pp", "delta_winner_mape_pp",
                  "delta_winner_mdape_ci_low", "delta_winner_mdape_ci_high"]]
          .to_string(index=False))
    print(f"\nSaved results in {root.resolve()}")
    return pooled, monthly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/listaware_fixed_origin_full")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-existing", action="store_true",
                        help="Score verified saved full fixed-origin predictions, no training")
    args = parser.parse_args()
    run(output_dir=args.output_dir, threads=args.threads,
        bootstrap_repeats=args.bootstrap_repeats, seed=args.seed,
        from_existing=args.from_existing)


if __name__ == "__main__":
    main()
