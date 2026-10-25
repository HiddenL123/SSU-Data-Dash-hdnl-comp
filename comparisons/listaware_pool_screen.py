"""Broader, chronological LIST-AWARE individual-model / feature-family screen.

Stage A of a future list-aware ensemble factory: compare INDIVIDUAL fitted
CatBoost/XGBoost/LightGBM models before any bagging/Optuna/ensemble admission.

Predeclared feature and loss catalog. Screen on Mar--May 2025 with a bounded
pilot budget. Optionally promote a capped, family-diverse subset to FULL-strength
Jun--Jul 2025 chronological confirmation. Neither stage adjusts blend weights.

Run from repository root, or use experiments/listaware_model_pool_screen.ipynb.
    python -m comparisons.listaware_pool_screen --stage screen --suite all
    python -m comparisons.listaware_pool_screen --stage confirm \
        --screen-dir outputs/listaware_pool_screen_all

This does NOT replace or mutate the existing 3.8996% Aug--Oct fixed-origin
benchmark. Previous historical months have been inspected by other experiments:
even this later-period confirmation is NOT an untouched final holdout.
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
from features.listaware_final_mae_features import CHALLENGERS
from experiments.listaware_quick_experiment import (
    _atomic_csv, _dataset_signature, _fit_predict, _frame,
    _verified_cache, summarize,
)
from models.listaware_shotgun_specs import (
    ADJUSTED, ADVANCED, BASIC, BINARY, CANDIDATES, SPATIAL, Candidate,
    parameters,
)
from models.listaware_candidate_pool import (
    EXTRA_CANDIDATES, CURATED, CONFIRMATION_ANCHORS,
)
from experiments.spacetime_fast import _cap

VERSION = 1
SCREEN_MONTHS = ("2025-03", "2025-04", "2025-05")
CONFIRM_MONTHS = ("2025-06", "2025-07")
SEED_REFERENCE = ("cat_huber_baseline", "cat_mae", "xgb_spline_geo", "lgb_spline_geo")
KEYS = ["month", "ml_num"]


def _candidate(name, backend, feature_name, *, loss="RMSE",
               tuned=False, target="ratio"):
    return Candidate(name=name, backend=backend, feature_name=feature_name,
                     loss=loss, tuned=tuned, target=target)


# Factorized, named experiments: do not run the entire Cartesian product.
# Each block isolates a key idea with matched model-family controls.
EXTRA = {
    # Text representation: the same base comparables, changed text only.
    "cat_mae_no_text": _candidate("cat_mae_no_text", "catboost",
                                   BASIC + "__no_text", loss="MAE"),
    "cat_mae_binary": _candidate("cat_mae_binary", "catboost", BINARY, loss="MAE"),
    "cat_mae_full_binary": _candidate("cat_mae_full_binary", "catboost",
                                      BASIC + "__full_binary", loss="MAE"),
    "cat_mae_full_count": _candidate("cat_mae_full_count", "catboost",
                                     BASIC + "__full_count", loss="MAE"),
    "cat_mae_full_tfidf": _candidate("cat_mae_full_tfidf", "catboost",
                                     BASIC + "__full_tfidf", loss="MAE"),
    "xgb_no_text": _candidate("xgb_no_text", "xgboost",
                              BASIC + "__no_text", tuned=True),
    "xgb_full_binary": _candidate("xgb_full_binary", "xgboost",
                                  BASIC + "__full_binary", tuned=True),
    "xgb_full_count": _candidate("xgb_full_count", "xgboost",
                                 BASIC + "__full_count", tuned=True),
    "lgb_no_text": _candidate("lgb_no_text", "lightgbm",
                              BASIC + "__no_text", tuned=True),
    "lgb_full_binary": _candidate("lgb_full_binary", "lightgbm",
                                  BASIC + "__full_binary", tuned=True),
    "lgb_full_count": _candidate("lgb_full_count", "lightgbm",
                                 BASIC + "__full_count", tuned=True),
    # Historical comparable-sale ablations on an otherwise fixed base.
    "cat_mae_age_first": _candidate("cat_mae_age_first", "catboost",
                                    BASIC + "__comp_age_first", loss="MAE"),
    "cat_mae_both_comp": _candidate("cat_mae_both_comp", "catboost",
                                    BASIC + "__comp_both", loss="MAE"),
    "xgb_age_first": _candidate("xgb_age_first", "xgboost",
                                BINARY + "__comp_age_first", tuned=True),
    "xgb_type_quota": _candidate("xgb_type_quota", "xgboost",
                                 BINARY + "__comp_type_quota", tuned=True),
    "lgb_age_first": _candidate("lgb_age_first", "lightgbm",
                                BINARY + "__comp_age_first", tuned=True),
    "lgb_type_quota": _candidate("lgb_type_quota", "lightgbm",
                                 BINARY + "__comp_type_quota", tuned=True),
    # Isolate splines vs geographic context, besides their combination.
    "cat_mae_splines": _candidate("cat_mae_splines", "catboost",
                                  BINARY + "__st_spacetime_splines", loss="MAE"),
    "cat_mae_geo": _candidate("cat_mae_geo", "catboost",
                              BINARY + "__st_geo_context", loss="MAE"),
    "xgb_splines": _candidate("xgb_splines", "xgboost",
                              BINARY + "__st_spacetime_splines", tuned=True),
    "xgb_geo": _candidate("xgb_geo", "xgboost",
                          BINARY + "__st_geo_context", tuned=True),
    "lgb_splines": _candidate("lgb_splines", "lightgbm",
                              BINARY + "__st_spacetime_splines", tuned=True),
    "lgb_geo": _candidate("lgb_geo", "lightgbm",
                          BINARY + "__st_geo_context", tuned=True),
    # Direct log(sold price) vs log(sold/list); keep the same feature stack.
    "cat_mae_direct": _candidate("cat_mae_direct", "catboost", BASIC,
                                 loss="MAE", target="direct"),
    "xgb_direct_spatial": _candidate("xgb_direct_spatial", "xgboost",
                                     SPATIAL, tuned=True, target="direct"),
    "lgb_direct_spatial": _candidate("lgb_direct_spatial", "lightgbm",
                                     SPATIAL, tuned=True, target="direct"),
    # Preexisting AVM feature families, with no new algorithms in this screen.
    "lgb_adjusted": _candidate("lgb_adjusted", "lightgbm", ADJUSTED, tuned=True),
    "lgb_advanced": _candidate("lgb_advanced", "lightgbm", ADVANCED, tuned=True),
}
CATALOG = {
    **{name: CANDIDATES[name] for name in (
        "cat_huber_baseline", "cat_mae", "cat_rmse", "cat_tuned_rmse",
        "cat_huber_direct", "xgb_tuned_binary", "xgb_spline_geo",
        "xgb_adjusted", "xgb_advanced", "lgb_tuned_binary", "lgb_spline_geo",
        "cat_huber_adjusted", "cat_huber_advanced",
    )},
    **CHALLENGERS,
    **EXTRA,
    **EXTRA_CANDIDATES,
}

GROUPS = {
    "core": (
        "cat_huber_baseline", "cat_mae", "cat_rmse", "cat_tuned_rmse",
        "xgb_tuned_binary", "xgb_spline_geo",
        "lgb_tuned_binary", "lgb_spline_geo",
    ),
    "text": (
        "cat_mae_no_text", "cat_mae_binary", "cat_mae_full_binary",
        "cat_mae_full_count", "cat_mae_full_tfidf",
        "xgb_no_text", "xgb_full_binary", "xgb_full_count",
        "lgb_no_text", "lgb_full_binary", "lgb_full_count",
    ),
    "comparables": (
        "cat_mae_age_first", "cat_mae_type_quota", "cat_mae_both_comp",
        "xgb_age_first", "xgb_type_quota",
        "lgb_age_first", "lgb_type_quota",
    ),
    "spatial": (
        "cat_mae_splines", "cat_mae_geo", "cat_mae_spatial_geo",
        "cat_mae_spatial_geo_type_quota", "xgb_splines", "xgb_geo",
        "lgb_splines", "lgb_geo",
    ),
    "objectives": (
        "cat_huber_direct", "cat_mae_direct", "xgb_direct_spatial",
        "lgb_direct_spatial",
    ),
    "avm": (
        "cat_huber_adjusted", "cat_huber_advanced", "xgb_adjusted",
        "xgb_advanced", "lgb_adjusted", "lgb_advanced",
    ),
}
SUITES = {
    "core": GROUPS["core"],
    "text": (*SEED_REFERENCE, *GROUPS["text"]),
    "comparables": (*SEED_REFERENCE, *GROUPS["comparables"]),
    "spatial": (*SEED_REFERENCE, *GROUPS["spatial"]),
    "objectives": (*SEED_REFERENCE, *GROUPS["objectives"]),
    "avm": (*SEED_REFERENCE, *GROUPS["avm"]),
    # Standalone/family-diverse recipes identified in prior pilot screens.
    # New suite avoids changing the historical "all" screen manifest/caches.
    "curated": (*SEED_REFERENCE, *CURATED),
    "all": (*GROUPS["core"], *(
        name for group in GROUPS.values() for name in group
    )),
}
SUITES = {
    suite: tuple(dict.fromkeys(("cat_huber_baseline", *names)))
    for suite, names in SUITES.items()
}

if (set(CATALOG) != set(SUITES["all"]).union(SUITES["curated"])
    or any(not set(names).issubset(CATALOG) for names in SUITES.values())):
    raise RuntimeError("The experiment suites and candidate catalog differ")


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def catalog_frame(names=None):
    names = tuple(CATALOG) if names is None else tuple(names)
    return pd.DataFrame([
        {"variant": name, "family": CATALOG[name].backend,
         "loss": CATALOG[name].loss, "target": CATALOG[name].target,
         "features": CATALOG[name].feature_name,
         "tuned_parameters": CATALOG[name].tuned}
        for name in names
    ])


def choose_confirmation(pooled, *, top_k=8, anchors=()):
    """Choose by screened individual accuracy, retaining model-family breadth.

    THIS procedure can use ONLY the initial chronological SCREEN score table,
    never the later confirmation cohort. The selection is deterministic.
    """
    if top_k < 3 or top_k > len(CATALOG):
        raise ValueError("top_k outside supported range")
    if pooled.variant.duplicated().any() or not set(SEED_REFERENCE).issubset(
        set(pooled.variant)
    ):
        raise ValueError("Screen missing the four required references")
    metrics = pooled.copy()
    if (not np.isfinite(metrics[["mdape", "mape"]].to_numpy(float)).all()
        or (metrics[["mdape", "mape"]] < 0).any().any()):
        raise ValueError("Invalid screen metrics")
    metrics["objective"] = .65 * metrics.mdape + .35 * metrics.mape
    metrics = metrics.sort_values(["objective", "mdape", "variant"])
    mandatory = tuple(dict.fromkeys((*SEED_REFERENCE, *anchors)))
    if any(name not in set(metrics.variant) for name in mandatory):
        raise ValueError("Screen is missing required curated confirmation anchors")
    if top_k < len(mandatory):
        raise ValueError(
            f"Confirmation needs top_k >= {len(mandatory)} for the "
            "predeclared multi-family anchors"
        )
    selected = list(mandatory)
    # Explicitly preserve opportunities for all THREE families, but never
    # force their admission to the eventual ensemble.
    for family in ("catboost", "xgboost", "lightgbm"):
        if len(selected) >= top_k:
            break
        for name in metrics.variant:
            if name not in selected and CATALOG[name].backend == family:
                selected.append(name)
                break
    for name in metrics.variant:
        if name not in selected and len(selected) < top_k:
            selected.append(name)
    return tuple(selected)


def _read_screen_selection(screen_dir, *, data_hash, top_k, expected_suite):
    folder = Path(screen_dir)
    manifest_path = folder / "manifest.json"
    scores_path = folder / "pooled_scores.csv"
    if not manifest_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError("Complete screen manifest and pooled_scores.csv required")
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (record.get("stage") != "screen" or record.get("version") != VERSION
        or tuple(record.get("months", ())) != SCREEN_MONTHS
        or record.get("suite") != expected_suite
        or record.get("data_sha256") != data_hash):
        raise ValueError("Screen dataset/protocol mismatch: cannot select candidates")
    scores = pd.read_csv(scores_path)
    if set(scores.variant) != set(record["candidates"]):
        raise ValueError("Screen scores do not cover exactly the recorded candidates")
    names = choose_confirmation(
        scores, top_k=top_k,
        anchors=CONFIRMATION_ANCHORS if expected_suite == "curated" else (),
    )
    return names, {
        "source_screen_manifest_sha256": _sha256(manifest_path),
        "source_screen_scores_sha256": _sha256(scores_path),
        "confirmation_selection": list(names),
        "selection_top_k": int(top_k),
    }


def error_correlations(predictions):
    """Pairwise signed log-error correlation on EXACT matched evaluation homes.

    An exploratory diagnostic for the future model pool, NOT a model
    admission score: lower correlation alone does not imply blend gain.
    """
    reference = predictions.loc[
        predictions.variant.eq("cat_huber_baseline")
    ].set_index(KEYS).sort_index()
    if reference.empty or not reference.index.is_unique:
        raise ValueError("Unique matched CatBoost Huber predictions required")
    actual = reference.sold_price.to_numpy(float)
    names, errors = [], []
    for name, group in predictions.groupby("variant", sort=True):
        aligned = group.set_index(KEYS).sort_index()
        if (not aligned.index.equals(reference.index)
            or not np.allclose(aligned.sold_price.to_numpy(float), actual,
                               rtol=0, atol=.0001)):
            raise ValueError(f"Misaligned pool correlation cohort: {name}")
        p = aligned.predicted_price.to_numpy(float)
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError(f"Invalid pool prediction: {name}")
        names.append(name)
        errors.append(np.log(actual) - np.log(p))
    rows = []
    for i, left in enumerate(names):
        for j in range(i+1, len(names)):
            right = names[j]
            a, b = errors[i], errors[j]
            corr = (float(np.corrcoef(a, b)[0, 1])
                    if np.std(a) > 0 and np.std(b) > 0 else np.nan)
            rows.append({
                "variant_a": left, "variant_b": right,
                "signed_log_error_correlation": corr,
            })
    return pd.DataFrame(rows, columns=[
        "variant_a", "variant_b", "signed_log_error_correlation"
    ])


def _ensure_manifest(output_dir, record):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("Incompatible screen/confirmation cache; use NEW output directory")
    else:
        if any(root.iterdir()):
            raise ValueError("Output directory is not empty without manifest")
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return root


def run(*, stage="screen", suite="all", screen_dir=None, output_dir=None,
        top_k=8, threads=4, seed=42, bootstrap_repeats=300,
        train_cap=None, validation_cap=None, tree_cap=None, listings=None):
    """Train independent candidate models with a strict month-by-month origin.

    Screen: Mar--May, capped pilot. Confirm: Jun--Jul, full historical fit.
    Every candidate predicts the SAME selected homes with legal list prices.
    If an interrupted run is retried, finished month/model predictions resume.
    """
    if stage not in ("screen", "confirm"):
        raise ValueError("Stage must be screen or confirm")
    if suite not in SUITES:
        raise ValueError(f"Unknown suite: {suite}")
    if min(threads, bootstrap_repeats) < 1:
        raise ValueError("Positive threads and bootstrap count required")
    months = SCREEN_MONTHS if stage == "screen" else CONFIRM_MONTHS
    full = stage == "confirm"
    default_train_cap = 200_000 if full else 10_000
    default_validation_cap = 100_000 if full else 350
    train_cap = default_train_cap if train_cap is None else int(train_cap)
    validation_cap = (
        default_validation_cap if validation_cap is None else int(validation_cap)
    )
    tree_cap = (None if full else 300) if tree_cap is None else int(tree_cap)
    if min(train_cap, validation_cap) < 1 or (tree_cap is not None and tree_cap < 1):
        raise ValueError("Invalid training, validation, or tree budget")
    if full and (train_cap != 200_000 or validation_cap != 100_000
                 or tree_cap is not None):
        raise ValueError("Confirmation requires full-strength budgets and tree counts")
    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_listed"] = pd.to_datetime(sales.date_listed)
    sales["date_sold"] = pd.to_datetime(sales.date_sold)
    if sales.ml_num.astype(str).duplicated().any():
        raise ValueError("Duplicate listings MLS IDs")
    if (not np.isfinite(sales[["sold_price", "list_price"]].to_numpy(float)).all()
        or (sales[["sold_price", "list_price"]] <= 0).any().any()):
        raise ValueError("Historical sale and listing prices must be positive/finite")
    digest = _dataset_signature(sales)
    if full:
        if screen_dir is None:
            raise ValueError("Full confirmation requires a completed screen_dir")
        candidates, provenance = _read_screen_selection(
            screen_dir, data_hash=digest, top_k=top_k, expected_suite=suite
        )
    else:
        candidates = SUITES[suite]
        provenance = {}
    if "cat_huber_baseline" not in candidates:
        raise RuntimeError("Paired reference missing")
    folder = Path(output_dir or (
        f"outputs/listaware_pool_{'confirmation' if full else 'screen'}_{suite}"
    ))
    manifest = {
        "version": VERSION, "track": "list_aware",
        "stage": stage, "suite": suite,
        "months": list(months), "full": full, "mode": "expanding",
        "train_start": "1900-01-01", "train_cap": train_cap,
        "validation_cap": validation_cap, "tree_cap": tree_cap,
        "threads": threads, "seed": seed, "bootstrap_repeats": bootstrap_repeats,
        "data_sha256": digest,
        "candidates": list(candidates),
        "candidate_specs": {name: asdict(CATALOG[name]) for name in candidates},
        "model_parameters": {
            name: parameters(CATALOG[name], trees=tree_cap, threads=threads)
            for name in candidates
        },
        **provenance,
    }
    root = _ensure_manifest(folder, manifest)
    months_of_sales = sales.date_sold.dt.to_period("M")
    predictions = []
    for month in months:
        period = pd.Period(month, freq="M")
        cutoff = period.start_time
        legal = sales.loc[(sales.date_sold >= pd.Timestamp("1900-01-01"))
                          & (sales.date_sold < cutoff)]
        train = _cap(legal, train_cap, seed)
        held = _cap(sales.loc[months_of_sales == period],
                    validation_cap, seed + 11)
        if train.empty or held.empty:
            raise ValueError(f"Empty history or holdout for {month}")
        if not (train.date_sold < cutoff).all():
            raise ValueError(f"Training leaked {month} sale labels")
        hidden = data.hide_answers(held, "list_aware")
        if ("sold_price" in hidden.columns or "date_sold" in hidden.columns
            or "list_price" not in hidden.columns):
            raise ValueError("Holdout answers were not hidden correctly")
        for j, name in enumerate(candidates, 1):
            path = root / month / "models" / f"{name}.csv"
            saved = _verified_cache(path, held, period, "model", name)
            if saved is None:
                print(f"[{stage} {month} {j}/{len(candidates)}] {name}: "
                      f"fit on {len(train):,}; predict {len(held):,}", flush=True)
                p = _fit_predict(
                    CATALOG[name], train, hidden, tree_cap=tree_cap,
                    threads=threads, features_cache={},
                )
                saved = _frame(held, period, "model", name, p)
                _atomic_csv(path, saved)
            else:
                print(f"[{stage} {month} {j}/{len(candidates)}] "
                      f"{name}: verified cached predictions", flush=True)
            predictions.append(saved)
    all_preds = pd.concat(predictions, ignore_index=True)
    pooled, monthly, by_type = summarize(
        all_preds, repeats=bootstrap_repeats, seed=seed
    )
    correlations = error_correlations(all_preds)
    _atomic_csv(root / "error_correlations.csv", correlations)
    _atomic_csv(root / "predictions.csv", all_preds)
    _atomic_csv(root / "pooled_scores.csv", pooled)
    _atomic_csv(root / "monthly_scores.csv", monthly)
    _atomic_csv(root / "house_type_scores.csv", by_type)
    _atomic_csv(root / "candidate_catalog.csv", catalog_frame(candidates))
    print("\nList-aware individual-model screen (matched homes):")
    print(pooled[["variant", "homes", "mdape", "mape", "p90_ape",
                  "delta_mdape_pp", "delta_mape_pp"]].to_string(index=False))
    print(f"Reports: {root.resolve()}")
    print("Exploratory historical development; not an untouched test.")
    return pooled, monthly, by_type, all_preds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    p.add_argument("--suite", choices=tuple(SUITES), default="all")
    p.add_argument("--screen-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-repeats", type=int, default=300)
    p.add_argument("--train-cap", type=int)
    p.add_argument("--validation-cap", type=int)
    p.add_argument("--tree-cap", type=int)
    args = p.parse_args(argv)
    return run(
        stage=args.stage, suite=args.suite, screen_dir=args.screen_dir,
        output_dir=args.output_dir, top_k=args.top_k, threads=args.threads,
        seed=args.seed, bootstrap_repeats=args.bootstrap_repeats,
        train_cap=args.train_cap, validation_cap=args.validation_cap,
        tree_cap=args.tree_cap,
    )


if __name__ == "__main__":
    main()
