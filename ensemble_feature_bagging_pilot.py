"""Notebook-driven first feature/row-bagging pilot for off-market XGBoost.

Build chronological comparable/structured/full binary BoW features ONCE per
holdout month, before drawing bootstrapped training rows. Holdout labels are
used strictly to score predictions. All text vocab selection is training-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import data
from models.ensemble_bagging import build_text_bank, select_text_terms, sample_training_rows, starter_specs
from models.features import make_features
from models.text_features import TextAugmentedFeatures
from spacetime_fast import XGB_TUNED, _cap, _make_estimator

FEATURE_NAME = "Xpt_spatiotemporal__full_binary__st_spacetime_splines+geo_context"
VERSION = 1


def _text_provider(provider):
    if isinstance(provider, TextAugmentedFeatures):
        return provider
    for field in ("base_provider", "provider"):
        child = getattr(provider, field, None)
        if child is not None:
            found = _text_provider(child)
            if found is not None:
                return found
    return None


def _score(y, p):
    error = 100 * np.abs(np.asarray(p, dtype=float)-np.asarray(y, dtype=float))/np.asarray(y, dtype=float)
    return float(np.median(error)), float(np.mean(error))


def run_feature_bagging_pilot(
    *, validation_month="2025-08", train_start="2022-01-01",
    train_cap=6000, validation_cap=350, trees=250,
    specs=None, seed=42, max_extra_terms=1000, min_df=10,
    output_dir="outputs/ensemble_feature_bagging_smoke",
    resume=True, listings=None,
):
    month = pd.Period(validation_month, freq="M")
    if month >= pd.Period("2025-12", freq="M"):
        raise ValueError("Use only earlier development folds, never final validation")
    if pd.Timestamp(train_start) >= month.start_time:
        raise ValueError("Historical train_start must precede the held-out month")
    if min(train_cap, validation_cap, trees, max_extra_terms, min_df) < 1:
        raise ValueError("Caps, tree budget, vocabulary size and min_df must be positive")
    specs = tuple(starter_specs(seed)[:5] if specs is None else specs)
    if not specs or len({s.name for s in specs}) != len(specs):
        raise ValueError("Need uniquely named feature-bagging members")
    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales.date_sold)
    sales["date_listed"] = pd.to_datetime(sales.date_listed)
    historical = sales.loc[
        (sales.date_sold >= pd.Timestamp(train_start)) &
        (sales.date_sold < month.start_time)
    ]
    validation = sales.loc[sales.date_sold.dt.to_period("M") == month]
    train = _cap(historical, train_cap, seed)
    held = _cap(validation, validation_cap, seed+11)
    if not len(train) or not len(held):
        raise ValueError("No eligible historical train / held-out sale homes")
    if train.ml_num.astype(str).duplicated().any() or held.ml_num.astype(str).duplicated().any():
        raise ValueError("MLS IDs must uniquely identify each cohort")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": VERSION, "validation_month": str(month), "train_start": train_start,
        "train_cap": train_cap, "validation_cap": validation_cap, "trees": trees,
        "seed": seed, "max_extra_terms": max_extra_terms, "min_df": min_df,
        "specs": [s.record() for s in specs], "xgb_params": XGB_TUNED,
        "feature_name": FEATURE_NAME,
        "data_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(sales, index=True).to_numpy(dtype="uint64").tobytes()
        ).hexdigest(),
    }
    setup = root / "setup.json"
    if setup.exists():
        previous = json.loads(setup.read_text(encoding="utf-8"))
        if previous != plan:
            raise ValueError("Different data/settings in output directory; choose a new path")
        if not resume:
            raise ValueError("Existing output directory; enable resume or use a new path")
    setup.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    bank = build_text_bank(train, max_extra_terms=max_extra_terms, min_df=min_df)
    # One common vocabulary and full-history feature fit shared by all bags.
    provider = make_features(FEATURE_NAME, model="xgboost", track="off_market")
    text = _text_provider(provider)
    if text is None:
        raise RuntimeError("Full-binary provider missing TextAugmentedFeatures")
    text.terms = tuple(bank)
    X_train = provider.fit_transform(train)
    X_test = provider.transform(data.hide_answers(held, "off_market"))
    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Full/holdout feature columns differ")
    col_to_term = text.text_feature_names
    if len(col_to_term) != len(bank):
        raise ValueError("Vectorized bank size differs from training-fitted terms")
    term_to_col = {term: col for col, term in col_to_term.items()}
    non_text = [col for col in X_train if col not in col_to_term]
    y = np.log(train.sold_price.to_numpy(dtype=float))
    offset = float(np.median(y))
    truths = held.sold_price.to_numpy(dtype=float)
    identifier = held.ml_num.astype(str).to_numpy()
    scores, predictions, manifest = [], [], []
    for spec in specs:
        terms = select_text_terms(bank, spec)
        selected_cols = [*non_text, *(term_to_col[t] for t in terms)]
        rows = sample_training_rows(len(train), spec)
        path = root / "members" / f"{spec.name}.csv"
        started = perf_counter()
        reused = False
        if resume and path.is_file():
            cached = pd.read_csv(path, dtype={"ml_num": str})
            if (len(cached) == len(held) and
                np.array_equal(cached.ml_num.to_numpy(), identifier) and
                np.array_equal(cached.sold_price.to_numpy(dtype=float), truths) and
                cached.predicted_price.gt(0).all() and
                np.isfinite(cached.predicted_price.to_numpy(dtype=float)).all()):
                pred = cached.predicted_price.to_numpy(dtype=float)
                reused = True
        if not reused:
            model = _make_estimator(XGB_TUNED, trees)
            model.set_params(random_state=spec.seed)
            model.fit(X_train.iloc[rows][selected_cols], y[rows] - offset)
            pred = np.exp(offset + np.asarray(model.predict(X_test[selected_cols]), dtype=float))
            if not np.isfinite(pred).all() or (pred <= 0).any():
                raise ValueError(f"Member {spec.name} produced invalid prices")
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "ml_num": identifier, "sold_price": truths,
                "predicted_price": pred,
            }).to_csv(path, index=False)
        mdape, mape = _score(truths, pred)
        seconds = perf_counter()-started
        scores.append({
            "member": spec.name, "homes": len(held),
            "mdape": mdape, "mape": mape, "n_text": len(terms),
            "n_rows": len(rows), "unique_rows": len(np.unique(rows)),
            "seconds": seconds, "cached": reused,
        })
        predictions.append(pd.DataFrame({
            "month": str(month), "ml_num": identifier,
            "member": spec.name, "sold_price": truths,
            "predicted_price": pred,
        }))
        manifest.append({**spec.record(), "n_terms": len(terms),
                         "terms": list(terms), "n_training_rows": len(rows),
                         "unique_training_rows": len(np.unique(rows))})
        print(f"{spec.name}: MdAPE={mdape:.4f}% MAPE={mape:.4f}%, "
              f"terms={len(terms)}, rows={len(rows)}, cached={reused}", flush=True)
    scores = pd.DataFrame(scores).sort_values(["mdape", "mape"])
    predictions = pd.concat(predictions, ignore_index=True)
    scores.to_csv(root / "scores.csv", index=False)
    predictions.to_csv(root / "predictions.csv", index=False)
    (root / "member_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return scores, predictions, pd.DataFrame([
        {k: v for k, v in rec.items() if k != "terms"} for rec in manifest
    ])
