"""Final targeted LIST-AWARE MAE feature ablation against the frozen 50/35/15 winner.

This is a RESEARCH comparison on already-inspected Aug--Oct 2025 homes,
not an independent holdout or a new production model. Requires an EXISTING
full-strength, single-fit fixed-origin run under outputs/listaware_fixed_origin_full.

Train only three previously untested CatBoost MAE feature configurations on
the exact pre-Aug-2025 cohort; reuse all previous candidate predictions.
Then replace ONLY the 50% MAE member in the earlier geometric blend, keeping
the XGBoost/LightGBM members and blend weights unchanged.

From the repository root:
    uv run python listaware_final_mae_features.py

Output is resumable under outputs/listaware_final_mae_features.
Existing models, factories, snapshots and fixed-origin outputs are read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
from experiments.listaware_quick_experiment import (
    _atomic_csv, _dataset_signature, _fit_predict, _frame, _verified_cache,
)
from experiments.listaware_fixed_origin_followup import ADDITIONAL_BLENDS
from experiments.listaware_full_fixed_origin_confirmation import (
    CONTROLS, KEY, MONTHS, WINNER, _verified_full_origin_manifest,
    compare_against_winner,
)
from models.listaware_shotgun_specs import BASIC, BINARY, SPATIAL, Candidate, parameters

VERSION = 1
SOURCE_DEFAULT = Path("outputs/listaware_fixed_origin_full")
OUTPUT_DEFAULT = Path("outputs/listaware_final_mae_features")
# Causal ablation order: quota on old MAE features, then established
# selected-word + spline/geo stack, then quota on that same spatial stack.
CHALLENGERS = {
    "cat_mae_type_quota": Candidate(
        "cat_mae_type_quota", "catboost", BASIC + "__comp_type_quota", loss="MAE"
    ),
    "cat_mae_spatial_geo": Candidate(
        "cat_mae_spatial_geo", "catboost", SPATIAL, loss="MAE"
    ),
    "cat_mae_spatial_geo_type_quota": Candidate(
        "cat_mae_spatial_geo_type_quota", "catboost",
        BINARY + "__comp_type_quota__st_spacetime_splines+geo_context",
        loss="MAE",
    ),
}
# REPLACE rather than append the MAE member, keeping the 35/15 XGB/LGB
# weights exactly from the previously selected list-aware winner.
REPLACEMENTS = {
    "replace_mae_type_quota_50_35_15": "cat_mae_type_quota",
    "replace_mae_spatial_geo_50_35_15": "cat_mae_spatial_geo",
    "replace_mae_spatial_geo_type_quota_50_35_15": "cat_mae_spatial_geo_type_quota",
}
BASE_WEIGHTS = ADDITIONAL_BLENDS[WINNER]
assert BASE_WEIGHTS == {
    "cat_mae": 0.50, "xgb_spline_geo": 0.35, "lgb_spline_geo": 0.15,
}


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_manifest(root, spec):
    """Refuse stale caches or changed historical data, predictions, or settings."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text(encoding="utf-8"))
        if prior != spec:
            raise ValueError("Feature experiment manifest mismatch: use a NEW output_dir")
    else:
        if any(root.iterdir()):
            raise ValueError("Output directory not empty without feature experiment manifest")
        manifest.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return root


def _verified_original_predictions(original, held_by_month):
    """Verify exact training-run predictions against real held-out home IDs/prices."""
    original = original.copy()
    required = {*KEY, "arm", "variant", "house_type_name",
                "sold_price", "predicted_price", "ape"}
    if not required.issubset(original.columns):
        raise ValueError(f"Missing original prediction fields: {required - set(original)}")
    original["month"] = original.month.astype(str)
    original["ml_num"] = original.ml_num.astype(str)
    if original.duplicated(KEY + ["arm", "variant"]).any():
        raise ValueError("Duplicate prediction keys in prior fixed-origin results")
    present = set(original.variant)
    needed = {*CONTROLS, *BASE_WEIGHTS}
    if not needed.issubset(present):
        raise ValueError(f"Missing previous models/blends: {needed - present}")
    if set(original.month) != set(MONTHS):
        raise ValueError("Original prediction months differ from experiment months")
    for month, held in held_by_month.items():
        assert month in MONTHS
        expected = held.copy()
        expected["ml_num"] = expected.ml_num.astype(str)
        expected = expected.set_index("ml_num")
        if not expected.index.is_unique:
            raise ValueError(f"Duplicate source MLS IDs in {month}")
        sold = expected.sold_price.to_numpy(dtype=float)
        types = expected.house_type_name.fillna("Unknown").astype(str).to_numpy()
        for (arm, variant), group in original.loc[original.month.eq(month)].groupby(
            ["arm", "variant"]
        ):
            # Saved BLENDS are sorted by MLS key while raw MODEL predictions
            # preserve heldout row order. Validate by EXACT keys, not position.
            paired = group.set_index("ml_num")
            if (len(paired) != len(expected)
                or not paired.index.is_unique
                or not paired.index.sort_values().equals(expected.index.sort_values())):
                raise ValueError(f"Existing cohort mismatch: {month} {arm}/{variant}")
            paired = paired.loc[expected.index]
            if (not np.allclose(paired.sold_price.to_numpy(float), sold,
                                rtol=0, atol=.0001)
                or not np.array_equal(paired.house_type_name.to_numpy(str), types)):
                raise ValueError(f"Existing cohort mismatch: {month} {arm}/{variant}")
            p = paired.predicted_price.to_numpy(float)
            if not (np.isfinite(p).all() and (p > 0).all()):
                raise ValueError(f"Invalid previous prediction: {month} {variant}")
            expected_ape = 100 * np.abs(p - sold) / sold
            if not np.allclose(paired.ape.to_numpy(float), expected_ape,
                               rtol=1e-9, atol=1e-7):
                raise ValueError(f"Incorrect cached APE: {month} {variant}")
    return original


def replacement_blends(predictions, *, replacements=REPLACEMENTS):
    """Generate fixed 50/35/15 geometric replacements with strict key joins.

    Pass already-validated model predictions. Never merge by row position,
    select missing homes silently, or change XGB/LGB/base blend weights.
    """
    frame = predictions.copy()
    if frame.duplicated(KEY + ["arm", "variant"]).any():
        raise ValueError("Duplicate source predictions")
    winner = frame.loc[frame.variant.eq(WINNER)].set_index(KEY).sort_index()
    if winner.empty or not winner.index.is_unique:
        raise ValueError("Winner is missing or duplicated")
    y = winner.sold_price.to_numpy(dtype=float)
    prices = {}
    for name in {"xgb_spline_geo", "lgb_spline_geo", *replacements.values()}:
        part = frame.loc[frame.variant.eq(name)].set_index(KEY).sort_index()
        if (part.empty or not part.index.equals(winner.index)
            or not np.allclose(part.sold_price.to_numpy(float), y,
                               rtol=0, atol=.0001)
            or not np.array_equal(part.house_type_name.to_numpy(str),
                                  winner.house_type_name.to_numpy(str))):
            raise ValueError(f"Missing/misaligned member for replacement: {name}")
        p = part.predicted_price.to_numpy(dtype=float)
        if not (np.isfinite(p).all() and (p > 0).all()):
            raise ValueError(f"Invalid replacement member: {name}")
        prices[name] = p
    outputs = []
    for name, mae_member in replacements.items():
        if name in set(frame.variant):
            raise ValueError(f"Output variant already exists: {name}")
        weights = {mae_member: BASE_WEIGHTS["cat_mae"],
                   "xgb_spline_geo": BASE_WEIGHTS["xgb_spline_geo"],
                   "lgb_spline_geo": BASE_WEIGHTS["lgb_spline_geo"]}
        pred = np.exp(sum(weight * np.log(prices[member])
                          for member, weight in weights.items()))
        out = winner.reset_index()[KEY + ["house_type_name", "sold_price"]].copy()
        out["arm"] = "replacement_blend"
        out["variant"] = name
        out["predicted_price"] = pred
        out["ape"] = 100 * abs(pred - y) / y
        outputs.append(out[frame.columns])
    return pd.concat(outputs, ignore_index=True)


def residual_correlations(predictions, *, variants):
    """Diagnostic ONLY; no residual regressor is fit on evaluation labels."""
    winner = predictions.loc[predictions.variant.eq(WINNER)].set_index(KEY).sort_index()
    y = winner.sold_price.to_numpy(float)
    base = winner.predicted_price.to_numpy(float)
    base_log_error = np.log(y) - np.log(base)
    base_ape = 100 * abs(base-y)/y
    rows = []
    for name in variants:
        other = predictions.loc[predictions.variant.eq(name)].set_index(KEY).sort_index()
        if not other.index.equals(winner.index):
            raise ValueError(f"Residual diagnostic cohort mismatch for {name}")
        p = other.predicted_price.to_numpy(float)
        log_error = np.log(y) - np.log(p)
        correlation = (
            float(np.corrcoef(base_log_error, log_error)[0, 1])
            if np.std(base_log_error) > 0 and np.std(log_error) > 0
            else np.nan
        )
        rows.append({
            "variant": name,
            "log_residual_corr_with_winner": correlation,
            "homes_improved_fraction": float(np.mean(100 * abs(p-y)/y < base_ape)),
        })
    return pd.DataFrame(rows)


def run(*, source_dir=SOURCE_DEFAULT, output_dir=OUTPUT_DEFAULT,
        threads=4, seed=42, bootstrap_repeats=2000, listings=None):
    """Use the original full-strength cutoff and ONLY train the new Cat MAEs."""
    if threads < 1 or bootstrap_repeats < 1:
        raise ValueError("Require positive threads and bootstrap_repeats")
    source = Path(source_dir)
    source_manifest = _verified_full_origin_manifest(source)
    if source_manifest.get("seed") != seed:
        raise ValueError("Seed differs from original baseline manifest")
    predictions_file = source / "fixed_weight_followup" / "followup_predictions.csv"
    if not predictions_file.is_file():
        raise FileNotFoundError(f"Full-strength baseline predictions missing: {predictions_file}")

    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales.date_sold)
    sales["date_listed"] = pd.to_datetime(sales.date_listed)
    if sales.ml_num.astype(str).duplicated().any():
        raise ValueError("Duplicate historical ml_num")
    data_hash = _dataset_signature(sales)
    if data_hash != source_manifest["data_sha256"]:
        raise ValueError("Historical listings differ from frozen baseline dataset")
    if not np.isfinite(sales[["sold_price", "list_price"]].to_numpy(float)).all():
        raise ValueError("Invalid historical prices")
    if (sales[["sold_price", "list_price"]] <= 0).any().any():
        raise ValueError("Prices must be strictly positive")

    origin = pd.Period(MONTHS[0], freq="M").start_time
    train = sales.loc[(sales.date_sold < origin)
                      & (sales.date_sold >= pd.Timestamp("1900-01-01"))].copy()
    if len(train) == 0 or len(train) > source_manifest["train_cap"]:
        raise ValueError("Historical training cohort is empty or exceeds frozen cap")
    periods = tuple(pd.Period(month, freq="M") for month in MONTHS)
    held_by_month = {
        str(period): sales.loc[sales.date_sold.dt.to_period("M") == period].copy()
        for period in periods
    }
    if any(held.empty for held in held_by_month.values()):
        raise ValueError("Missing validation month(s)")
    if any(len(held) > source_manifest["validation_cap"]
           for held in held_by_month.values()):
        raise ValueError("Validation month exceeds frozen cap")
    originals = _verified_original_predictions(
        pd.read_csv(predictions_file, dtype={"ml_num": str}), held_by_month
    )

    manifest = {
        "version": VERSION,
        "track": "list_aware", "protocol": "single-fit_fixed_origin",
        "months": list(MONTHS), "cutoff_exclusive": str(origin.date()),
        "full_strength": True,
        "source_manifest_sha256": _sha256(source / "manifest.json"),
        "source_predictions_sha256": _sha256(predictions_file),
        "data_sha256": data_hash,
        "seed": seed, "threads": threads, "bootstrap_repeats": bootstrap_repeats,
        "candidates": {name: asdict(spec) for name, spec in CHALLENGERS.items()},
        "estimator_params": {
            name: parameters(spec, trees=None, threads=threads)
            for name, spec in CHALLENGERS.items()
        },
        "replacements": REPLACEMENTS,
        "weights": BASE_WEIGHTS,
    }
    out = _prepare_manifest(output_dir, manifest)
    future = pd.concat([
        data.hide_answers(held, "list_aware")
        for held in held_by_month.values()
    ], axis=0)
    if ("sold_price" in future.columns or "date_sold" in future.columns
        or "list_price" not in future.columns):
        raise ValueError("Incorrectly hidden holdout columns")

    saved = []
    for name, spec in CHALLENGERS.items():
        paths = [out / month / "models" / f"{name}.csv" for month in MONTHS]
        cached = [_verified_cache(path, held_by_month[month],
                                  pd.Period(month, freq="M"), "model", name)
                  for path, month in zip(paths, MONTHS)]
        if all(chunk is not None for chunk in cached):
            print(f"{name}: reused complete saved predictions", flush=True)
            frames = cached
        else:
            print(f"{name}: fitting once on {len(train):,} pre-August sales; "
                  f"predicting {len(future):,} later homes", flush=True)
            forecast = _fit_predict(spec, train, future,
                                    tree_cap=None, threads=threads, features_cache={})
            frames = []
            offset = 0
            for path, month, previous in zip(paths, MONTHS, cached):
                held = held_by_month[month]
                pred = forecast[offset:offset+len(held)]
                offset += len(held)
                current = _frame(held, pd.Period(month, freq="M"), "model", name, pred)
                if previous is not None:
                    if not np.allclose(previous.predicted_price.to_numpy(float), pred,
                                       rtol=1e-6, atol=.01):
                        raise ValueError("Recomputed predictions differ from previous cache")
                    current = previous
                else:
                    _atomic_csv(path, current)
                frames.append(current)
            if offset != len(forecast):
                raise ValueError("Incorrect forecast partition length")
        saved.extend(frames)

    joined = pd.concat([originals, *saved], ignore_index=True)
    new_blends = replacement_blends(joined)
    complete = pd.concat([joined, new_blends], ignore_index=True)
    names = (*CONTROLS, *CHALLENGERS, *REPLACEMENTS)
    pooled, monthly = compare_against_winner(
        complete, candidates=names, repeats=bootstrap_repeats, seed=seed
    )
    diagnostics = residual_correlations(
        complete, variants=(*CHALLENGERS, *REPLACEMENTS)
    )
    _atomic_csv(out / "predictions.csv", complete)
    _atomic_csv(out / "pooled_vs_frozen_winner.csv", pooled)
    _atomic_csv(out / "monthly_vs_frozen_winner.csv", monthly)
    _atomic_csv(out / "error_correlations.csv", diagnostics)
    print("\nFinal list-aware MAE feature comparison (negative delta improves winner):")
    print(pooled[["variant", "homes", "mdape", "mape", "p90_ape",
                  "delta_winner_mdape_pp", "delta_winner_mape_pp",
                  "delta_winner_mdape_ci_low", "delta_winner_mdape_ci_high"]]
          .to_string(index=False), flush=True)
    print(f"\nSaved files: {out.resolve()}", flush=True)
    print("Caution: Aug--Oct 2025 was already inspected; treat wins as exploratory.")
    return pooled, monthly, diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=SOURCE_DEFAULT)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = p.parse_args()
    run(source_dir=args.source_dir, output_dir=args.output_dir,
        threads=args.threads, seed=args.seed,
        bootstrap_repeats=args.bootstrap_repeats)


if __name__ == "__main__":
    main()
