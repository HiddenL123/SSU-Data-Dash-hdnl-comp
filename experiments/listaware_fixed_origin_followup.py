"""Fixed-origin list-aware follow-up + predeclared cross-family blends.

Default runs the previous full-strength CatBoost/XGB/LGB shortlist with ONE
fixed training cutoff before August 2025, then computes four additional
predeclared geometric blends from its out-of-time per-home predictions.

Also supports no-training evaluation of the user's already-saved EXPANDING
experiment via --from-predictions outputs/listaware_full_shortlist/predictions.csv.
That is a historical, already-examined development comparison, not confirmation.

Usage:
 python listaware_fixed_origin_followup.py --pilot
 python listaware_fixed_origin_followup.py --full
 python listaware_fixed_origin_followup.py --from-predictions outputs/listaware_full_shortlist/predictions.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import data
from experiments.listaware_quick_experiment import (
    summarize, _atomic_csv, _fit_predict, _frame, _verified_cache,
    _dataset_signature, _cap, _check_setup,
)
from models.listaware_shotgun_specs import CANDIDATES as CANDIDATE_REGISTRY, BLENDS

CANDIDATES = (
    "cat_huber_baseline", "cat_mae", "cat_huber_spline_geo",
    "cat_huber_advanced", "xgb_spline_geo", "lgb_spline_geo",
)
KEYS = ["month", "ml_num"]
# Every blend weight is registered BEFORE looking at evaluation results.
# Weights apply to log predicted prices, sum to 1, and include alternatives
# to the original winning 50/35/15 blend.
ADDITIONAL_BLENDS = {
    "followup_winner_plus_mae": {
        "cat_huber_spline_geo": .40,
        "xgb_spline_geo": .28,
        "lgb_spline_geo": .12,
        "cat_mae": .20,
    },
    "followup_two_cat_spatial_advanced": {
        "cat_huber_spline_geo": .70,
        "cat_huber_advanced": .30,
    },
    "followup_cat_mae_xgb_lgb": {
        "cat_mae": .50,
        "xgb_spline_geo": .35,
        "lgb_spline_geo": .15,
    },
    "followup_huber_mae": {
        "cat_huber_baseline": .70,
        "cat_mae": .30,
    },
    "followup_cat_spatial_mae": {
        "cat_huber_spline_geo": .70,
        "cat_mae": .30,
    },
}
for _key, _weights in ADDITIONAL_BLENDS.items():
    if not np.isclose(sum(_weights.values()), 1):
        raise ValueError(f"Invalid blend sum: {_key}")


def append_blends(predictions, *, blends=ADDITIONAL_BLENDS):
    """Never join on row positions. Match EXACT home keys and sold prices."""
    required = {*KEYS, "arm", "variant", "house_type_name",
                "sold_price", "predicted_price", "ape"}
    if not required.issubset(predictions):
        raise ValueError(f"Prediction table missing {required - set(predictions)}")
    original = predictions.copy()
    original["month"] = original.month.astype(str)
    original["ml_num"] = original.ml_num.astype(str)
    if original.duplicated(KEYS + ["arm", "variant"]).any():
        raise ValueError("Duplicate model/month/home identifiers")
    reference = original.loc[
        (original.arm == "model") & (original.variant == "cat_huber_baseline")
    ].set_index(KEYS).sort_index()
    if reference.empty or not reference.index.is_unique:
        raise ValueError("Missing unique CatBoost Huber reference homes")
    for weights in blends.values():
        if not np.isclose(sum(weights.values()), 1):
            raise ValueError("Blend weights must sum to one")
        if any(v < 0 for v in weights.values()):
            raise ValueError("Blend weights must be nonnegative")

    selected = {}
    for name in set().union(*(set(x) for x in blends.values())):
        part = original.loc[(original.arm == "model")
                            & (original.variant == name)].set_index(KEYS).sort_index()
        if len(part) == 0:
            continue
        if (len(part) != len(reference) or not part.index.equals(reference.index)
            or not np.allclose(part.sold_price.to_numpy(float),
                               reference.sold_price.to_numpy(float), rtol=0, atol=.0001)
            or not np.array_equal(part.house_type_name.astype(str).to_numpy(),
                                   reference.house_type_name.astype(str).to_numpy())):
            raise ValueError(f"Model {name} is not aligned to identical homes/labels")
        p = part.predicted_price.to_numpy(float)
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError(f"Invalid prices for {name}")
        selected[name] = p
    chunks = [original]
    for name, weights in blends.items():
        absent = set(weights) - selected.keys()
        if absent:
            # A slim shortlist can legitimately exclude candidate members.
            print(f"Skipping {name}: missing {sorted(absent)}", flush=True)
            continue
        if name in set(original.variant):
            raise ValueError(f"Blend name already exists: {name}")
        prediction = np.exp(sum(w * np.log(selected[key]) for key, w in weights.items()))
        new = reference.reset_index()[KEYS + ["house_type_name", "sold_price"]].copy()
        new["arm"], new["variant"] = "blend", name
        new["predicted_price"] = prediction
        new["ape"] = 100 * np.abs(
            new.predicted_price.to_numpy(float) - new.sold_price.to_numpy(float)
        ) / new.sold_price.to_numpy(float)
        chunks.append(new[original.columns])
    combined = pd.concat(chunks, ignore_index=True)
    # The common research summary validates exact cohort keys, label matching,
    # and produces paired bootstrap diagnostics for all arms and blends.
    return combined


def report_followup(predictions, output_dir, *, bootstrap_repeats=500, seed=42):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    combined = append_blends(predictions)
    pooled, monthly, by_type = summarize(
        combined, repeats=bootstrap_repeats, seed=seed
    )
    _atomic_csv(folder / "followup_predictions.csv", combined)
    _atomic_csv(folder / "followup_pooled.csv", pooled)
    _atomic_csv(folder / "followup_monthly.csv", monthly)
    _atomic_csv(folder / "followup_by_type.csv", by_type)
    print("\nList-aware fixed-origin / existing-OOT blend scoreboard:", flush=True)
    print(pooled[["arm", "variant", "homes", "mdape", "mape", "p90_ape",
                  "delta_mdape_pp", "delta_mape_pp"]].to_string(index=False),
          flush=True)
    return pooled, monthly, by_type, combined



def _run_single_fit_fixed_origin(
    *, months, full, output_dir, threads, seed, bootstrap_repeats,
    listings=None,
):
    """Fit each candidate ONCE at the cutoff, predict every future holdout.

    In contrast with the generic listaware_shotgun fixed-origin runner, which
    refits deterministically each month, this is a literal fixed fitted model:
    one estimator / vocabulary per family and common cutoff for all months.
    It saves candidate predictions on each month so interrupted work resumes.
    """
    months = tuple(months)
    training_cap = 200_000 if full else 10_000
    held_cap = 100_000 if full else 350
    iterations = None if full else 300
    periods = _check_setup(
        months, CANDIDATES, training_cap, held_cap,
        700 if full else iterations, full
    )
    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales.date_sold)
    sales["date_listed"] = pd.to_datetime(sales.date_listed)
    if sales.ml_num.astype(str).duplicated().any():
        raise ValueError("Duplicate sales/listing IDs")
    if not np.isfinite(sales[["list_price", "sold_price"]].to_numpy(float)).all():
        raise ValueError("Invalid sale/list prices")
    if (sales[["list_price", "sold_price"]] <= 0).any().any():
        raise ValueError("Nonpositive sale/list prices")
    cutoff = periods[0].start_time
    train_start = pd.Timestamp("1900-01-01" if full else "2024-01-01")
    training = sales.loc[(sales.date_sold < cutoff) & (sales.date_sold >= train_start)]
    train = _cap(training, training_cap, seed)
    held = []
    hidden = []
    for period in periods:
        actual = _cap(
            sales.loc[sales.date_sold.dt.to_period("M") == period],
            held_cap, seed+11
        )
        if actual.empty:
            raise ValueError(f"Empty holdout {period}")
        masked = data.hide_answers(actual, "list_aware")
        if "sold_price" in masked or "date_sold" in masked or "list_price" not in masked:
            raise ValueError("Incorrect label hiding for list-aware test")
        held.append(actual)
        hidden.append(masked)
    if train.empty:
        raise ValueError("No historical training data before fixed origin")
    future = pd.concat(hidden)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 2, "mode": "single-fit_fixed_origin", "full": full,
        "months": months, "candidates": CANDIDATES,
        "train_cutoff_exclusive": str(cutoff.date()),
        "train_start": str(train_start.date()),
        "train_cap": training_cap, "validation_cap": held_cap,
        "trees": iterations, "threads": threads, "seed": seed,
        "data_sha256": _dataset_signature(sales),
    }
    manifest = root / "manifest.json"
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8")) != json.loads(
            json.dumps(spec)
        ):
            raise ValueError("Fixed-origin source/config changed; use another output folder")
    else:
        if any(root.iterdir()):
            raise ValueError("Output directory is nonempty without a valid manifest")
        manifest.write_text(json.dumps(spec, indent=2)+"\n", encoding="utf-8")

    forecasts = {}
    frames = []
    for name in CANDIDATES:
        paths = [root / str(period) / "models" / f"{name}.csv" for period in periods]
        cached = [_verified_cache(path, actual, period, "model", name)
                  for path, actual, period in zip(paths, held, periods)]
        if all(c is not None for c in cached):
            print(f"Fixed-origin {name}: reused all saved months", flush=True)
            streams = cached
        else:
            print(f"Fixed-origin {name}: ONE fit on {len(train):,} earlier sales; "
                  f"predict {len(future):,} future homes", flush=True)
            predicted = _fit_predict(
                CANDIDATE_REGISTRY[name], train, future,
                tree_cap=iterations, threads=threads,
                features_cache={},
            )
            streams = []
            i = 0
            for path, actual, period, previous in zip(paths, held, periods, cached):
                p = predicted[i:i+len(actual)]
                i += len(actual)
                model_frame = _frame(actual, period, "model", name, p)
                if previous is not None:
                    if not np.allclose(
                        previous.predicted_price.to_numpy(float), p,
                        rtol=1e-6, atol=.01
                    ):
                        raise ValueError("Recomputed fixed-origin cache differs")
                    model_frame = previous
                else:
                    _atomic_csv(path, model_frame)
                streams.append(model_frame)
            if i != len(predicted):
                raise ValueError("Fixed-origin future prediction count changed")
        for period, frame in zip(periods, streams):
            forecasts[(str(period), name)] = frame.predicted_price.to_numpy(float)
            frames.append(frame)
    for name, weights in BLENDS.items():
        if not set(weights).issubset(CANDIDATES):
            continue
        for period, actual in zip(periods, held):
            geometric = np.exp(sum(
                weight * np.log(forecasts[str(period), member])
                for member, weight in weights.items()
            ) / sum(weights.values()))
            frames.append(_frame(actual, period, "blend", name, geometric))
    full_predictions = pd.concat(frames, ignore_index=True)
    pooled, monthly, by_type = summarize(
        full_predictions, repeats=bootstrap_repeats, seed=seed
    )
    _atomic_csv(root / "predictions.csv", full_predictions)
    _atomic_csv(root / "pooled_scores.csv", pooled)
    _atomic_csv(root / "monthly_scores.csv", monthly)
    _atomic_csv(root / "house_type_scores.csv", by_type)
    return full_predictions


def run_followup(*, full=False, months=("2025-08", "2025-09", "2025-10"),
                 output_dir=None, threads=4, seed=42, bootstrap_repeats=500,
                 from_predictions=None, listings=None):
    if from_predictions is not None:
        source = Path(from_predictions)
        if not source.is_file():
            raise FileNotFoundError(source)
        # Scores-only mode does not refit any models and does not imply fixed origin.
        saved = pd.read_csv(source, dtype={"ml_num": str})
        out = Path(output_dir) if output_dir else source.parent / "fixed_weight_followup"
        return report_followup(saved, out, bootstrap_repeats=bootstrap_repeats, seed=seed)

    output = Path(output_dir or (
        "outputs/listaware_fixed_origin_full" if full
        else "outputs/listaware_fixed_origin_pilot"
    ))
    predictions = _run_single_fit_fixed_origin(
        months=tuple(months), full=full, output_dir=output,
        threads=threads, seed=seed, bootstrap_repeats=bootstrap_repeats,
        listings=listings,
    )
    return report_followup(
        predictions, output / "fixed_weight_followup",
        bootstrap_repeats=bootstrap_repeats, seed=seed
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--from-predictions", type=Path)
    p.add_argument("--months", nargs="+",
                   default=["2025-08", "2025-09", "2025-10"])
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-repeats", type=int, default=500)
    args = p.parse_args()
    run_followup(
        full=args.full, months=args.months, output_dir=args.output_dir,
        threads=args.threads, seed=args.seed,
        bootstrap_repeats=args.bootstrap_repeats,
        from_predictions=args.from_predictions,
    )


if __name__ == "__main__":
    main()
