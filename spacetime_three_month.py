"""Three-month chronological follow-up for leading space-time XGBoost candidates.

This runner REUSES spacetime_shotgun.run_shotgun without changing production
features or targets, runs three larger chronological folds, and pools exactly
matched properties. Every month has a separate output folder; successful,
parameter-matched months may be resumed without retraining.

Defaults: September, October, November 2025; all homes in each validation
month; 40k sampled historical rows per fold; 500 tuned XGBoost trees; 3 prior
OOT calibration months; 1,200 calibration homes per month.

This is DEVELOPMENT research, not untouched confirmation: the October 2025
fold was already inspected during the shotgun screen. Prefer stability across
months and paired differences rather than treating small pooled wins as proof.

From the repository root:
  uv run python spacetime_three_month.py
  uv run python spacetime_three_month.py --train-cap 60000 --trees 700

Outputs in outputs/spacetime_three_month, plus one subfolder per month.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
from spacetime_shotgun import XGB_TUNED, run_shotgun

MONTHS = ("2025-09", "2025-10", "2025-11")
FEATURES = ("baseline", "comp_type_quota", "comp_age_first", "comp_both")
RESIDUALS = ("knn", "rbf_ridge", "tensor_gam")
COMPARE_COLUMNS = ["month", "ml_num"]
METRICS = ("mdape", "mape")
IDENTITY = ["arm", "variant"]
KEY = ["month", "ml_num", "arm", "variant"]


def _score(frame):
    true = frame["sold_price"].to_numpy(dtype=float)
    predicted = frame["predicted_price"].to_numpy(dtype=float)
    if not (np.isfinite(true).all() and (true > 0).all() and
            np.isfinite(predicted).all() and (predicted > 0).all()):
        raise ValueError("Invalid observed or predicted house prices")
    signed = 100.0 * (predicted - true) / true
    return {
        "homes": len(frame),
        "mape": float(np.mean(np.abs(signed))),
        "mdape": float(np.median(np.abs(signed))),
        "median_signed": float(np.median(signed)),
    }


def _validate_result(predictions, expected, months):
    if predictions.duplicated(KEY).any():
        raise ValueError("Repeated month/home/arm/variant keys in predictions")
    actual = set(map(tuple, predictions[IDENTITY].drop_duplicates().to_numpy()))
    if actual != expected:
        raise ValueError(f"Missing/extra model candidates: {actual ^ expected}")
    if set(predictions.month) != set(months):
        raise ValueError("Prediction months do not match the requested folds")
    baseline = predictions[
        (predictions.arm == "feature") & (predictions.variant == "baseline")
    ]
    if baseline.duplicated(COMPARE_COLUMNS).any():
        raise ValueError("Baseline validation property keys are not unique")
    base_index = pd.MultiIndex.from_frame(baseline[COMPARE_COLUMNS])
    if len(base_index) == 0:
        raise ValueError("Baseline predictions are empty")
    for (arm, variant), chunk in predictions.groupby(IDENTITY):
        idx = pd.MultiIndex.from_frame(chunk[COMPARE_COLUMNS])
        if len(chunk) != len(baseline) or not idx.is_unique or not idx.difference(base_index).empty:
            raise ValueError(f"{arm}/{variant} does not match the baseline homes")
        aligned = chunk.set_index(COMPARE_COLUMNS).loc[
            baseline.set_index(COMPARE_COLUMNS).index
        ]
        if not np.allclose(
            aligned.sold_price.to_numpy(dtype=float),
            baseline.sold_price.to_numpy(dtype=float),
            rtol=0, atol=0,
        ):
            raise ValueError(f"{arm}/{variant} ground truths differ from baseline")


def _bootstrap_delta(control, candidate, *, repeats, seed):
    """Pair-resample properties, recomputing the DIFFERENCE of both statistics."""
    if len(control) != len(candidate):
        raise ValueError("Bootstrap arrays must align")
    if repeats < 0:
        raise ValueError("repeats must be nonnegative")
    if not repeats:
        return {
            "delta_mdape_ci_low": np.nan, "delta_mdape_ci_high": np.nan,
            "delta_mape_ci_low": np.nan, "delta_mape_ci_high": np.nan,
        }
    rng = np.random.default_rng(seed)
    n = len(control)
    median_deltas, mean_deltas = [], []
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        median_deltas.append(float(np.median(candidate[idx]) - np.median(control[idx])))
        mean_deltas.append(float(np.mean(candidate[idx] - control[idx])))
    a, b = np.percentile(median_deltas, [2.5, 97.5])
    c, d = np.percentile(mean_deltas, [2.5, 97.5])
    return {
        "delta_mdape_ci_low": float(a), "delta_mdape_ci_high": float(b),
        "delta_mape_ci_low": float(c), "delta_mape_ci_high": float(d),
    }


def summarize(predictions, *, repeats=500, seed=42):
    """Pooled, monthly, and by-type summaries, all paired to baseline."""
    month_names = sorted(predictions.month.unique())
    expected = set(map(tuple, predictions[IDENTITY].drop_duplicates().to_numpy()))
    _validate_result(predictions, expected, month_names)

    baseline = predictions[
        (predictions.arm == "feature") & (predictions.variant == "baseline")
    ].set_index(COMPARE_COLUMNS).sort_index()
    base_score = _score(baseline)
    base_ape = (100 * (baseline.predicted_price - baseline.sold_price).abs() /
                baseline.sold_price).to_numpy(dtype=float)
    rows = []
    monthly_rows = []
    for (arm, variant), chunk in predictions.groupby(IDENTITY, sort=False):
        compared = chunk.set_index(COMPARE_COLUMNS).loc[baseline.index]
        ape = (100 * (compared.predicted_price - compared.sold_price).abs() /
               compared.sold_price).to_numpy(dtype=float)
        score = _score(compared)
        score.update({
            "arm": arm, "variant": variant,
            "delta_mdape_pp": score["mdape"] - base_score["mdape"],
            "delta_mape_pp": score["mape"] - base_score["mape"],
            "mean_paired_ape_delta": float(np.mean(ape - base_ape)),
            "homes_improved_fraction": float(np.mean(ape < base_ape)),
        })
        score.update(_bootstrap_delta(
            base_ape, ape, repeats=repeats, seed=seed
        ))
        positive_months = 0
        for month in month_names:
            this = compared.xs(month, level="month")
            ref = baseline.xs(month, level="month")
            subscore = _score(this)
            refscore = _score(ref)
            subscore.update({
                "month": month, "arm": arm, "variant": variant,
                "delta_mdape_pp": subscore["mdape"] - refscore["mdape"],
                "delta_mape_pp": subscore["mape"] - refscore["mape"],
            })
            positive_months += subscore["delta_mdape_pp"] < -1e-10
            monthly_rows.append(subscore)
        score["months_better_mdape"] = positive_months
        rows.append(score)
    summary = pd.DataFrame(rows).sort_values(["mdape", "mape"]).reset_index(drop=True)
    monthly = pd.DataFrame(monthly_rows).sort_values(
        ["month", "mdape", "mape"]
    ).reset_index(drop=True)
    return summary, monthly


def _attach_type(predictions, listings):
    fields = listings[["ml_num", "date_sold", "house_type_name"]].copy()
    fields["month"] = pd.to_datetime(fields.date_sold).dt.to_period("M").astype(str)
    fields = fields[["month", "ml_num", "house_type_name"]]
    if fields.duplicated(COMPARE_COLUMNS).any():
        # Different listings for one id/month could invalidate paired analysis.
        raise ValueError("Source listings have nonunique month/ml_num identifiers")
    return predictions.merge(
        fields, on=COMPARE_COLUMNS, how="left", validate="many_to_one"
    )


def _type_summary(labeled, min_count=20):
    rows = []
    for (house_type, arm, variant), chunk in labeled.groupby(
        ["house_type_name", "arm", "variant"], dropna=False
    ):
        if len(chunk) < min_count:
            continue
        score = _score(chunk)
        score.update({
            "house_type_name": house_type,
            "arm": arm, "variant": variant,
        })
        rows.append(score)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    ref = result[(result.arm == "feature") & (result.variant == "baseline")]
    base = ref.set_index("house_type_name")[["mdape", "mape"]]
    result["delta_mdape_pp"] = result.mdape - result.house_type_name.map(base.mdape)
    result["delta_mape_pp"] = result.mape - result.house_type_name.map(base.mape)
    return result.sort_values(
        ["house_type_name", "mdape"], kind="stable"
    ).reset_index(drop=True)


def run_three_month(*, months=MONTHS, train_start="2022-01-01",
                    train_cap=40000, trees=500, validation_cap=1_000_000,
                    calibration_cap=1200, calibration_folds=3,
                    features=FEATURES, residuals=RESIDUALS,
                    residual_strength=0.5, seed=42, params=None,
                    bootstrap_repeats=500,
                    output_dir="outputs/spacetime_three_month",
                    resume=True, listings=None):
    periods = pd.PeriodIndex(months, freq="M")
    if len(periods) != 3 or periods.has_duplicates or not periods.is_monotonic_increasing:
        raise ValueError("Provide exactly three distinct chronological months")
    if any(p >= pd.Period("2025-12", freq="M") for p in periods):
        raise ValueError("Three-month development cannot use December 2025 or later")
    if not features or "baseline" not in features or len(features) != len(set(features)):
        raise ValueError("Feature candidates must include baseline once")
    if len(residuals) != len(set(residuals)):
        raise ValueError("Residual candidates must be unique")
    if min(train_cap, trees, validation_cap, calibration_cap, calibration_folds) < 1:
        raise ValueError("Budgets must be positive")
    if pd.Timestamp(train_start) >= (periods[0] - calibration_folds).start_time:
        raise ValueError("train_start too late for earlier out-of-time calibration folds")

    from spacetime_shotgun import SCREEN_FEATURE_VARIANTS
    from models.spacetime_screen_residuals import RESIDUAL_VARIANTS
    if any(v not in SCREEN_FEATURE_VARIANTS for v in features) or any(
        v not in RESIDUAL_VARIANTS for v in residuals
    ):
        raise ValueError("Unknown feature/residual candidate name")
    params = dict(XGB_TUNED if params is None else params)
    expected = {("feature", v) for v in features} | {("residual", v) for v in residuals}
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales["date_sold"])
    sales["date_listed"] = pd.to_datetime(sales["date_listed"])

    setup = {
        "months": [str(p) for p in periods],
        "train_start": train_start,
        "train_cap": train_cap,
        "validation_cap": validation_cap,
        "calibration_cap": calibration_cap,
        "calibration_folds": calibration_folds,
        "trees": trees, "seed": seed, "params": params,
        "features": list(features), "residuals": list(residuals),
        "residual_strength": residual_strength,
        "note": "Exploratory prior development folds; October already screened",
    }
    (folder / "setup.json").write_text(json.dumps(setup, indent=2))

    chunks, failures = [], []
    for period in periods:
        month = str(period)
        subdir = folder / month
        config = {
            "validation_month": month, "train_start": train_start,
            "train_cap": train_cap, "validation_cap": validation_cap,
            "calibration_cap": calibration_cap,
            "calibration_folds": calibration_folds,
            "screen_trees": trees, "seed": seed, "params": params,
            "residual_strength": residual_strength,
            "feature_variants": list(features),
            "residual_variants": list(residuals),
        }
        cached = subdir / "predictions.csv"
        config_file = subdir / "setup.json"
        failed_file = subdir / "failures.json"
        reusable = False
        if resume and cached.exists() and config_file.exists() and failed_file.exists():
            old = json.loads(config_file.read_text(encoding="utf-8"))
            old_failures = json.loads(failed_file.read_text(encoding="utf-8"))
            reusable = not old_failures and all(
                old.get(k) == v for k, v in config.items()
            )
            if reusable:
                try:
                    check = pd.read_csv(cached, dtype={"ml_num": "str"})
                    _validate_result(check, expected, [month])
                except (ValueError, KeyError, TypeError):
                    reusable = False
            if reusable:
                print(f"{month}: reusing verified saved results", flush=True)
                chunks.append(check)
        if reusable:
            continue

        print(f"\n=== {month}: enlarged chronological experiment ===", flush=True)
        _, predictions, problems = run_shotgun(
            validation_month=month,
            train_start=train_start,
            train_cap=train_cap, validation_cap=validation_cap,
            calibration_cap=calibration_cap,
            calibration_folds=calibration_folds,
            screen_trees=trees, seed=seed, params=params,
            features=features, residuals=residuals,
            residual_strength=residual_strength,
            output_dir=subdir, listings=sales,
        )
        if problems:
            failures.extend([{"month": month, **item} for item in problems])
            (folder / "failures.json").write_text(
                json.dumps(failures, indent=2), encoding="utf-8"
            )
            # Do NOT pool a partial set of candidates as a valid comparison.
            raise RuntimeError(
                f"{month} has {len(problems)} failed candidates; "
                f"see {subdir / 'failures.json'}. "
                "The completed earlier months remain saved."
            )
        _validate_result(predictions, expected, [month])
        chunks.append(predictions)

    all_predictions = pd.concat(chunks, ignore_index=True)
    _validate_result(all_predictions, expected, [str(p) for p in periods])
    summary, monthly = summarize(
        all_predictions, repeats=bootstrap_repeats, seed=seed
    )
    labeled = _attach_type(all_predictions, sales)
    by_type = _type_summary(labeled)

    all_predictions.to_csv(folder / "predictions.csv", index=False)
    labeled.to_csv(folder / "predictions_with_type.csv", index=False)
    summary.to_csv(folder / "pooled_summary.csv", index=False)
    monthly.to_csv(folder / "monthly_summary.csv", index=False)
    by_type.to_csv(folder / "house_type_summary.csv", index=False)
    (folder / "failures.json").write_text("[]\n", encoding="utf-8")

    print("\n=== Pooled three-month scores (lower is better) ===\n",
          summary[[
              "arm", "variant", "homes", "mdape", "mape",
              "delta_mdape_pp", "delta_mape_pp", "months_better_mdape",
              "delta_mdape_ci_low", "delta_mdape_ci_high",
          ]].to_string(index=False), flush=True)
    print(f"\nOutputs saved to {folder.resolve()}", flush=True)
    return summary, monthly, by_type, all_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs=3, default=list(MONTHS))
    parser.add_argument("--train-start", default="2022-01-01")
    parser.add_argument("--train-cap", type=int, default=40000)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--validation-cap", type=int, default=1000000)
    parser.add_argument("--calibration-cap", type=int, default=1200)
    parser.add_argument("--calibration-folds", type=int, default=3)
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument("--residuals", nargs="*", default=list(RESIDUALS))
    parser.add_argument("--residual-strength", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/spacetime_three_month")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    params = (
        json.loads(args.params.read_text(encoding="utf-8"))
        if args.params is not None else None
    )
    if params is not None and not isinstance(params, dict):
        parser.error("--params file must contain a JSON object")
    run_three_month(
        months=args.months, train_start=args.train_start,
        train_cap=args.train_cap, trees=args.trees,
        validation_cap=args.validation_cap,
        calibration_cap=args.calibration_cap,
        calibration_folds=args.calibration_folds,
        features=tuple(args.features), residuals=tuple(args.residuals),
        residual_strength=args.residual_strength, seed=args.seed,
        params=params, bootstrap_repeats=args.bootstrap_repeats,
        output_dir=args.output_dir, resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
