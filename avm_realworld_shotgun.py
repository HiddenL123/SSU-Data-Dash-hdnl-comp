"""Run non-neural, real-world-inspired AVM shotgun against the frozen ensemble.

The 2025-08 to 2025-10 audit is already inspected historical development data:
this screen may guide engineering, but cannot be reported as untouched
confirmation. The existing ensemble is READ ONLY and never refitted.

Examples (PowerShell, project root):
  py -3.13 avm_realworld_shotgun.py
  py -3.13 avm_realworld_shotgun.py --full --features baseline rw_adjusted_comps rw_combined

Low-cost screen defaults to 10k historical rows / 220 trees / 350 homes per
month. --full uses eligible full history / 700 trees / every holdout home.
Both compare on *identically aligned homes*, but the pilot candidate models
use a lower budget than the frozen ensemble: do not equate their capacities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
from models.spacetime_screen_features import FEATURE_VARIANTS
from spacetime_fast import XGB_TUNED, run_shotgun, _metric
from spacetime_three_month import _bootstrap_delta

DEFAULT_FEATURES = (
    "baseline", "rw_adjusted_comps", "rw_local_residuals",
    "rw_robust_comps", "rw_multiscale_tiles", "rw_combined",
    "adaptive_comps", "hedonic_index",
)
DEFAULT_REFERENCE = Path(
    "outputs/ensemble_release_candidate_full/shadow_audit_predictions.csv"
)
VERSION = 1
KEY = ["month", "ml_num"]


def _check_inputs(months, features, ref):
    if len(months) < 1 or len(set(months)) != len(months):
        raise ValueError("Need unique evaluation months")
    periods = pd.PeriodIndex(months, freq="M")
    if not periods.is_monotonic_increasing or any(
        m >= pd.Period("2025-12") for m in periods
    ):
        raise ValueError("Use earlier chronological development months only")
    if "baseline" not in features or len(set(features)) != len(features):
        raise ValueError("Feature arms must contain baseline exactly once")
    if not set(features).issubset(FEATURE_VARIANTS):
        raise ValueError(f"Unknown feature families: {set(features)-set(FEATURE_VARIANTS)}")
    required = set(KEY + ["sold_price", "predicted_price"])
    if not required.issubset(ref.columns):
        raise ValueError(f"Ensemble reference lacks {required-set(ref.columns)}")
    if ref.duplicated(KEY).any():
        raise ValueError("Duplicate month/property keys in frozen ensemble reference")
    if not np.isfinite(ref[["sold_price", "predicted_price"]].to_numpy(float)).all():
        raise ValueError("Nonfinite ensemble reference prices")


def _assemble_reference(predictions, ensemble):
    """Require an ensemble prediction for *every* candidate evaluation home."""
    base = predictions.loc[
        (predictions.arm == "feature") & (predictions.variant == "baseline")
    ].copy()
    keys = base[KEY].copy()
    if keys.duplicated().any():
        raise ValueError("Repeated baseline validation homes")
    ref = keys.merge(ensemble, on=KEY, how="left", validate="one_to_one")
    if ref["predicted_price"].isna().any():
        missing = ref.loc[ref.predicted_price.isna(), KEY].head().to_dict("records")
        raise ValueError(f"Frozen ensemble lacks matching audited homes: {missing}")
    if not np.allclose(
        ref.sold_price.to_numpy(float), base.sold_price.to_numpy(float),
        rtol=0, atol=0.0001,
    ):
        raise ValueError("Reference and research prediction actual prices differ")
    ref["arm"] = "frozen_reference"
    ref["variant"] = "ensemble"
    if "abs_pct_error" in predictions.columns:
        ref["abs_pct_error"] = (100 * abs(ref.predicted_price - ref.sold_price)
                                / ref.sold_price)
    return pd.concat([
        predictions, ref[predictions.columns]
    ], ignore_index=True)


def _score_all(predictions, repeats=300, seed=42):
    reference = predictions.loc[
        (predictions.arm == "frozen_reference")
        & (predictions.variant == "ensemble")
    ].set_index(KEY).sort_index()
    baseline = predictions.loc[
        (predictions.arm == "feature")
        & (predictions.variant == "baseline")
    ].set_index(KEY).loc[reference.index]
    y = reference.sold_price.to_numpy(float)
    ensemble_ape = 100 * abs(reference.predicted_price.to_numpy(float) - y) / y
    control_ape = 100 * abs(baseline.predicted_price.to_numpy(float) - y) / y
    ensemble_metric = _metric(y, reference.predicted_price.to_numpy(float))
    control_metric = _metric(y, baseline.predicted_price.to_numpy(float))
    rows, monthly = [], []
    for (arm, name), chunk in predictions.groupby(["arm", "variant"]):
        aligned = chunk.set_index(KEY).loc[reference.index]
        if len(chunk) != len(reference):
            raise ValueError(f"Inconsistent cohort length: {arm}/{name}")
        if not np.allclose(y, aligned.sold_price.to_numpy(float), rtol=0, atol=0.0001):
            raise ValueError(f"Mismatched ground truth: {arm}/{name}")
        p = aligned.predicted_price.to_numpy(float)
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError(f"Invalid predictions: {arm}/{name}")
        ape = 100 * abs(p - y) / y
        score = _metric(y, p)
        score.update({
            "arm": arm, "variant": name,
            "delta_ensemble_mdape_pp": score["mdape"] - ensemble_metric["mdape"],
            "delta_ensemble_mape_pp": score["mape"] - ensemble_metric["mape"],
            "delta_control_mdape_pp": score["mdape"] - control_metric["mdape"],
            "delta_control_mape_pp": score["mape"] - control_metric["mape"],
            "fraction_better_than_ensemble": float(np.mean(ape < ensemble_ape)),
            "fraction_better_than_control": float(np.mean(ape < control_ape)),
        })
        score.update({
            f"ensemble_{k}": v for k, v in _bootstrap_delta(
                ensemble_ape, ape, repeats=repeats, seed=seed
            ).items()
        })
        for month in sorted(predictions.month.unique()):
            ix = aligned.index.get_level_values("month") == month
            ref_p = reference.predicted_price.to_numpy(float)[ix]
            m = _metric(y[ix], p[ix])
            rm = _metric(y[ix], ref_p)
            m.update({
                "month": month, "arm": arm, "variant": name,
                "delta_ensemble_mdape_pp": m["mdape"] - rm["mdape"],
                "delta_ensemble_mape_pp": m["mape"] - rm["mape"],
            })
            monthly.append(m)
        rows.append(score)
    return (pd.DataFrame(rows).sort_values(["mdape", "mape"]),
            pd.DataFrame(monthly).sort_values(["month", "mdape"]))


def _fingerprint(dataframe):
    """Reuse the repo's full-data fingerprint convention (index included)."""
    h = hashlib.sha256()
    h.update("|".join(dataframe.columns).encode())
    h.update(pd.util.hash_pandas_object(dataframe, index=True)
             .to_numpy(dtype="uint64").tobytes())
    return h.hexdigest()


def run_realworld_shotgun(
    *, months=("2025-08", "2025-09", "2025-10"),
    features=DEFAULT_FEATURES, full=False,
    train_cap=None, validation_cap=None, trees=None,
    seed=42, reference_file=DEFAULT_REFERENCE,
    output_dir=None, bootstrap_repeats=300,
    sales=None,
):
    months = tuple(months)
    features = tuple(features)
    ref_path = Path(reference_file)
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen reference: {ref_path}. Keep this exact audit CSV "
            "or specify --reference-file; no re-training the ensemble is necessary."
        )
    ensemble = pd.read_csv(ref_path, dtype={"ml_num": str})
    ensemble["month"] = ensemble["month"].astype(str)
    _check_inputs(months, features, ensemble)
    if not set(months).issubset(set(ensemble.month)):
        raise ValueError("Reference CSV does not contain every requested month")
    train_cap = int(train_cap if train_cap is not None else (200000 if full else 10000))
    validation_cap = int(
        validation_cap if validation_cap is not None else (100000 if full else 350)
    )
    trees = int(trees if trees is not None else (700 if full else 220))
    if min(train_cap, validation_cap, trees) <= 0 or bootstrap_repeats < 0:
        raise ValueError("Invalid experiment budget")
    dataset = data.load("listings") if sales is None else sales.copy()
    dataset["date_sold"] = pd.to_datetime(dataset.date_sold)
    dataset["date_listed"] = pd.to_datetime(dataset.date_listed)
    folder = Path(output_dir or (
        "outputs/realworld_avm_full" if full else "outputs/realworld_avm_pilot"
    ))
    folder.mkdir(parents=True, exist_ok=True)
    experiment = {
        "version": VERSION, "months": months, "features": features,
        "train_start": "1900-01-01" if full else "2024-01-01",
        "train_cap": train_cap, "validation_cap": validation_cap,
        "trees": trees, "seed": seed,
        "data_sha256": _fingerprint(dataset),
        "reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
        "development_only": True,
        "evaluation_note": (
            "Audit dates previously inspected. The frozen reference uses "
            "700-tree full-history ensemble members. Pilot candidate fits "
            "use reduced data/trees and must not be treated as matched-budget "
            "comparisons. This screening does not update ensemble weights."
        ),
    }
    signature_path = folder / "research_manifest.json"
    if signature_path.is_file():
        prior = json.loads(signature_path.read_text(encoding="utf-8"))
        if prior != json.loads(json.dumps(experiment)):
            raise ValueError("Research configuration/data changed; use a NEW output folder")
    else:
        if any(folder.iterdir()):
            raise ValueError("Output folder is not empty and has no manifest")
        signature_path.write_text(json.dumps(experiment, indent=2) + "\n")
    predictions = []
    for month in months:
        sub = folder / month
        file = sub / "predictions.csv"
        if file.is_file():
            cached = pd.read_csv(file, dtype={"ml_num": str})
            expected = {("feature", name) for name in features}
            actual = set(map(tuple, cached[["arm", "variant"]].drop_duplicates().to_numpy()))
            if (expected != actual or cached.duplicated(KEY + ["arm", "variant"]).any()
                or not (cached.month.astype(str) == month).all()
                or not np.isfinite(cached.predicted_price.to_numpy(float)).all()):
                raise ValueError(f"Invalid existing monthly cache: {file}")
            # Full per-home alignment and label checks happen below.
            print(f"{month}: reused research predictions", flush=True)
        else:
            print(f"{month}: training {len(features)} arms", flush=True)
            _, cached, errors = run_shotgun(
                validation_month=month,
                train_start=experiment["train_start"],
                train_cap=train_cap, validation_cap=validation_cap,
                screen_trees=trees, seed=seed,
                params=dict(XGB_TUNED),
                features=features, residuals=(),
                output_dir=str(sub), listings=dataset,
            )
            if errors:
                raise RuntimeError(
                    f"{len(errors)} feature candidates failed in {month}; "
                    f"inspect {sub / 'failures.json'} before comparing"
                )
        predictions.append(cached)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions["ml_num"] = predictions["ml_num"].astype(str)
    predictions["month"] = predictions["month"].astype(str)
    aligned = _assemble_reference(predictions, ensemble)
    pooled, monthly = _score_all(
        aligned, repeats=bootstrap_repeats, seed=seed
    )
    aligned.to_csv(folder / "paired_predictions.csv", index=False)
    pooled.to_csv(folder / "pooled_vs_ensemble.csv", index=False)
    monthly.to_csv(folder / "monthly_vs_ensemble.csv", index=False)
    print("\nMatched-homes comparison vs frozen 7.4% ensemble:", flush=True)
    print(pooled[["arm", "variant", "homes", "mdape", "mape",
                  "delta_ensemble_mdape_pp", "delta_ensemble_mape_pp",
                  "delta_control_mdape_pp"]].to_string(index=False), flush=True)
    print(f"\nSaved paired predictions/metrics in {folder}", flush=True)
    return pooled, monthly, aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--months", nargs="+",
                        default=["2025-08", "2025-09", "2025-10"])
    parser.add_argument("--features", nargs="+", choices=FEATURE_VARIANTS,
                        default=list(DEFAULT_FEATURES))
    parser.add_argument("--reference-file", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-cap", type=int, default=None)
    parser.add_argument("--validation-cap", type=int, default=None)
    parser.add_argument("--trees", type=int, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_realworld_shotgun(
        months=args.months, features=args.features, full=args.full,
        train_cap=args.train_cap, validation_cap=args.validation_cap,
        trees=args.trees, bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed, reference_file=args.reference_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
