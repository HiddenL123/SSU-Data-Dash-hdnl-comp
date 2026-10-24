"""Quick, leak-safe OFF-MARKET cross-family blending experiment.

The frozen XGBoost ensemble reference is NEVER retrained or overwritten.
Fit standalone tuned XGBoost, CatBoost and LightGBM off-market regressors
on COMPLETED past sales, evaluate the SAME IDs as the frozen 2025 shadow audit,
and test PREDECLARED geometric blends anchored separately on standalone XGB
and on the already-frozen ensemble.

For a fast proof of concept:
  python offmarket_crossfamily_blend.py --reuse-family-caches outputs/offmarket_crossfamily_pilot
For all eligible prior training sales with recorded tuned iteration counts:
  python offmarket_crossfamily_blend.py --full
No list_price, test labels, or in-sample predictions are model features.
Pilot challengers use less compute than the frozen ensemble; do NOT treat this
as a matched-training-budget test of independent model family superiority.
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
from models.features import make_features
from models.listaware_shotgun_specs import CAT_TUNED_RMSE, LGB_TUNED, SPATIAL
from models.automl_checkpoint import write_csv
from spacetime_fast import XGB_TUNED, _metric
from spacetime_three_month import _bootstrap_delta

REFERENCE = Path("outputs/ensemble_release_candidate_full/shadow_audit_predictions.csv")
SOURCE_KEYS = ["month", "ml_num"]
ARMS = (
    "frozen_xgb_ensemble", "xgboost_offmarket",
    "catboost_offmarket", "lightgbm_offmarket",
)
BLENDS = {
    "frozen_plus_cat_80_20": {"frozen_xgb_ensemble": .80, "catboost_offmarket": .20},
    "frozen_plus_lgb_80_20": {"frozen_xgb_ensemble": .80, "lightgbm_offmarket": .20},
    "frozen_cat_lgb_70_20_10": {
        "frozen_xgb_ensemble": .70, "catboost_offmarket": .20,
        "lightgbm_offmarket": .10,
    },
    "frozen_cat_lgb_60_25_15": {
        "frozen_xgb_ensemble": .60, "catboost_offmarket": .25,
        "lightgbm_offmarket": .15,
    },
    # New hypothesis: mixing base XGBoost with other model families rather
    # than diluting the already-diversified frozen XGBoost ensemble.
    "base_xgb_plus_cat_80_20": {
        "xgboost_offmarket": .80, "catboost_offmarket": .20,
    },
    "base_xgb_plus_lgb_80_20": {
        "xgboost_offmarket": .80, "lightgbm_offmarket": .20,
    },
    "base_xgb_cat_lgb_70_20_10": {
        "xgboost_offmarket": .70, "catboost_offmarket": .20,
        "lightgbm_offmarket": .10,
    },
    "base_xgb_cat_lgb_50_35_15": {
        "xgboost_offmarket": .50, "catboost_offmarket": .35,
        "lightgbm_offmarket": .15,
    },
}
for _blend_name, _weights in BLENDS.items():
    if any(v < 0 for v in _weights.values()) or not np.isclose(sum(_weights.values()), 1):
        raise ValueError(f"Invalid predeclared blend: {_blend_name}")
FEATURE_NAME = SPATIAL
VERSION = 2


def _digest(dataframe):
    h = hashlib.sha256()
    h.update("|".join(dataframe.columns).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(dataframe, index=True)
             .to_numpy(dtype="uint64").tobytes())
    return h.hexdigest()


def _validate_reference(frame, *, months):
    fields = {*SOURCE_KEYS, "sold_price", "predicted_price"}
    if fields - set(frame):
        raise ValueError(f"Frozen reference missing {fields - set(frame)}")
    if frame.duplicated(SOURCE_KEYS).any():
        raise ValueError("Frozen reference has repeated month/home IDs")
    if any(pd.Period(m) >= pd.Period("2025-12") for m in months):
        raise ValueError("Only historical pre-December 2025 audit months supported")
    if not set(months).issubset(set(frame.month)):
        raise ValueError("Reference does not contain every requested audit month")
    if not np.isfinite(frame[["sold_price", "predicted_price"]].to_numpy(float)).all():
        raise ValueError("Nonfinite reference prices")
    if (frame[["sold_price", "predicted_price"]] <= 0).any().any():
        raise ValueError("Nonpositive reference prices")


def _cohort(sales, reference, month, *, max_validation, seed):
    ids = reference.loc[reference.month == month, SOURCE_KEYS + [
        "sold_price", "predicted_price"
    ]].copy()
    if not len(ids):
        raise ValueError(f"No reference homes for {month}")
    if len(ids) > max_validation:
        ids = ids.sample(n=max_validation, random_state=seed).sort_values("ml_num")
    candidates = sales.copy()
    candidates["ml_num"] = candidates.ml_num.astype(str)
    if candidates.ml_num.duplicated().any():
        raise ValueError("Historical data has duplicate listing IDs")
    held = ids[["ml_num"]].merge(
        candidates, on="ml_num", how="left", validate="one_to_one",
        indicator=True,
    )
    if held["_merge"].ne("both").any():
        raise ValueError("Frozen reference contains homes missing from listings")
    held = held.drop(columns="_merge")
    actual = held.sold_price.to_numpy(float)
    if not np.allclose(actual, ids.sold_price.to_numpy(float), rtol=0, atol=.0001):
        raise ValueError("Reference and historical sale prices differ")
    actual_months = pd.to_datetime(held.date_sold).dt.to_period("M").astype(str)
    if not actual_months.eq(month).all():
        raise ValueError("Reference home/month mismatch")
    return held, ids.predicted_price.to_numpy(float)


def _model_prediction(train, homes, family, *, full, trees, threads):
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    backends = {
        "xgboost_offmarket": "xgboost",
        "catboost_offmarket": "catboost",
        "lightgbm_offmarket": "lightgbm",
    }
    if family not in backends:
        raise ValueError(f"Unknown new model {family}")
    backend = backends[family]
    provider = make_features(FEATURE_NAME, model=backend, track="off_market")
    X, H = provider.fit_transform(train), provider.transform(homes)
    if (not X.index.equals(train.index) or not H.index.equals(homes.index)
        or not X.columns.equals(H.columns)):
        raise ValueError("Historical and evaluation matrices are not aligned")
    if "list_price" in H or "sold_price" in H or "date_sold" in H:
        raise ValueError("Off-market feature pipeline includes unavailable sale/list price")
    target = np.log(train.sold_price.to_numpy(float))
    offset = float(np.median(target))
    if backend == "catboost":
        params = dict(CAT_TUNED_RMSE)
        params["thread_count"] = threads
        params["loss_function"] = "RMSE"
        if not full:
            params["iterations"] = min(int(trees), params["iterations"])
        regressor = CatBoostRegressor(**params)
        cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        regressor.fit(X, target - offset, cat_features=cols)
    elif backend == "lightgbm":
        params = dict(LGB_TUNED)
        params["n_jobs"] = threads
        if not full:
            params["n_estimators"] = min(int(trees), params["n_estimators"])
        regressor = LGBMRegressor(**params)
        regressor.fit(X, target - offset)
    else:
        params = dict(XGB_TUNED)
        params["n_jobs"] = threads
        params["device"] = "cpu"
        params["enable_categorical"] = True
        if not full:
            params["n_estimators"] = min(int(trees), params["n_estimators"])
        regressor = XGBRegressor(**params)
        regressor.fit(X, target - offset)
    prediction = np.exp(offset + np.asarray(regressor.predict(H), dtype=float).reshape(-1))
    if (prediction.shape != (len(homes),) or not np.isfinite(prediction).all()
        or (prediction <= 0).any()):
        raise ValueError(f"Nonfinite or nonpositive {family} predictions")
    return prediction


def _frame(held, month, name, prediction):
    truth = held.sold_price.to_numpy(float)
    p = np.asarray(prediction, dtype=float)
    if len(p) != len(truth) or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError(f"Invalid predictions from {name}")
    return pd.DataFrame({
        "month": month, "ml_num": held.ml_num.astype(str).to_numpy(),
        "arm": "blend" if name in BLENDS else "model",
        "variant": name, "sold_price": truth, "predicted_price": p,
    })


def _check_saved(path, held, month, name):
    if not path.is_file():
        return None
    stored = pd.read_csv(path, dtype={"ml_num": str})
    expected = _frame(held, month, name, stored.predicted_price.to_numpy(float))
    if (list(stored.columns) != list(expected.columns)
        or not stored[["month", "ml_num", "arm", "variant"]].equals(
            expected[["month", "ml_num", "arm", "variant"]])
        or not np.allclose(stored.sold_price.to_numpy(float),
                           expected.sold_price.to_numpy(float), rtol=0, atol=.0001)):
        raise ValueError(f"Stale cached prediction file {path}")
    return stored


def _compare(predictions, *, repeats=500, seed=42):
    base = predictions.loc[predictions.variant == "frozen_xgb_ensemble"].set_index(
        SOURCE_KEYS
    ).sort_index()
    if base.empty or not base.index.is_unique:
        raise ValueError("Missing frozen ensemble reference")
    y = base.sold_price.to_numpy(float)
    pb = base.predicted_price.to_numpy(float)
    control = _metric(y, pb)
    reference_ape = 100 * np.abs(pb-y)/y
    standalone = predictions.loc[
        predictions.variant == "xgboost_offmarket"
    ].set_index(SOURCE_KEYS).sort_index()
    if (standalone.empty or not standalone.index.is_unique
        or not standalone.index.equals(base.index)
        or not np.allclose(y, standalone.sold_price.to_numpy(float), rtol=0, atol=.0001)):
        raise ValueError("Missing or misaligned standalone XGBoost baseline")
    p_xgb = standalone.predicted_price.to_numpy(float)
    if not np.isfinite(p_xgb).all() or (p_xgb <= 0).any():
        raise ValueError("Invalid standalone XGBoost baseline")
    xgb_control = _metric(y, p_xgb)
    xgb_ape = 100 * np.abs(p_xgb-y)/y
    pooled, monthly = [], []
    for name, frame in predictions.groupby("variant", sort=False):
        if frame.duplicated(SOURCE_KEYS).any() or len(frame) != len(base):
            raise ValueError(f"Different homes for model {name}")
        aligned = frame.set_index(SOURCE_KEYS).sort_index()
        if (not aligned.index.equals(base.index)
            or not np.allclose(y, aligned.sold_price.to_numpy(float), rtol=0, atol=.0001)):
            raise ValueError(f"Different reference cohort or sold prices for {name}")
        p = aligned.predicted_price.to_numpy(float)
        ape = 100 * np.abs(p-y)/y
        score = _metric(y, p)
        score.update({
            "arm": frame.arm.iloc[0], "variant": name,
            "delta_mdape_pp": score["mdape"]-control["mdape"],
            "delta_mape_pp": score["mape"]-control["mape"],
            "p90_ape": float(np.percentile(ape, 90)),
            "homes_improved_fraction": float(np.mean(ape < reference_ape)),
            "delta_base_xgb_mdape_pp": score["mdape"]-xgb_control["mdape"],
            "delta_base_xgb_mape_pp": score["mape"]-xgb_control["mape"],
            "homes_improved_vs_base_xgb": float(np.mean(ape < xgb_ape)),
        })
        score.update(_bootstrap_delta(reference_ape, ape,
                                     repeats=repeats, seed=seed))
        xgb_intervals = _bootstrap_delta(xgb_ape, ape,
                                         repeats=repeats, seed=seed)
        score.update({f"base_xgb_{key}": val for key, val in xgb_intervals.items()})
        pooled.append(score)
        for month in sorted(predictions.month.unique()):
            ix = aligned.index.get_level_values("month") == month
            sub = _metric(y[ix], p[ix])
            base_sub = _metric(y[ix], pb[ix])
            monthly.append({
                "month": month, "arm": frame.arm.iloc[0], "variant": name,
                **sub, "delta_mdape_pp": sub["mdape"]-base_sub["mdape"],
                "delta_mape_pp": sub["mape"]-base_sub["mape"],
                "delta_base_xgb_mdape_pp": sub["mdape"]-_metric(
                    y[ix], p_xgb[ix]
                )["mdape"],
                "delta_base_xgb_mape_pp": sub["mape"]-_metric(
                    y[ix], p_xgb[ix]
                )["mape"],
            })
    return (
        pd.DataFrame(pooled).sort_values(["mdape", "mape"]),
        pd.DataFrame(monthly).sort_values(["month", "mdape"]),
    )


def run_offmarket_blend(*, full=False, months=("2025-08","2025-09","2025-10"),
                        train_cap=None, validation_cap=None, trees=350,
                        seed=42, threads=4, bootstrap_repeats=500,
                        reference_file=REFERENCE, output_dir=None,
                        reuse_family_caches=None, listings=None):
    months = tuple(months)
    if not months or len(set(months)) != len(months) or months != tuple(sorted(months)):
        raise ValueError("Require ordered distinct audit months")
    if threads < 1 or trees < 1 or bootstrap_repeats < 0:
        raise ValueError("Invalid run budgets")
    train_cap = int(train_cap if train_cap is not None else (1_000_000 if full else 25_000))
    validation_cap = int(validation_cap if validation_cap is not None else (1_000_000 if full else 350))
    if min(train_cap, validation_cap) < 1:
        raise ValueError("Nonpositive sample cap")
    reference_path = Path(reference_file)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Missing saved frozen ensemble audit: {reference_path}. "
            "Use the real historical shadow_audit_predictions.csv."
        )
    reference = pd.read_csv(reference_path, dtype={"ml_num": str})
    reference.month = reference.month.astype(str)
    _validate_reference(reference, months=months)
    sales = data.load("listings") if listings is None else listings.copy()
    sales.date_sold = pd.to_datetime(sales.date_sold)
    sales.date_listed = pd.to_datetime(sales.date_listed)
    if sales.ml_num.astype(str).duplicated().any():
        raise ValueError("Duplicated ml_num in historical sales")
    folder = Path(output_dir or ("outputs/offmarket_base_xgb_full"
                                 if full else "outputs/offmarket_base_xgb_pilot"))
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": VERSION, "full": full, "months": months, "train_cap": train_cap,
        "validation_cap": validation_cap, "trees": None if full else trees,
        "seed": seed, "threads": threads, "features": FEATURE_NAME,
        "data_sha256": _digest(sales),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "blends": BLENDS,
        "source": "Frozen historical off-market XGBoost ensemble plus new fitted CatBoost/LightGBM",
    }
    manifest = folder / "manifest.json"
    canonical_meta = json.loads(json.dumps(meta))
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != canonical_meta:
            raise ValueError("Experiment source/config changed; choose new output directory")
    else:
        if any(folder.iterdir()):
            raise ValueError("Refusing to reuse nonempty output folder without manifest")
        manifest.write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
    # Reuse CatBoost/LightGBM predictions from the PREVIOUS pilot, where
    # candidate training and the sampled validation homes are identical.
    # Verify source manifest provenance, not just file existence.
    previous_folder = Path(reuse_family_caches) if reuse_family_caches else None
    if previous_folder is not None:
        old_manifest_path = previous_folder / "manifest.json"
        if not old_manifest_path.is_file():
            raise FileNotFoundError(f"Missing previous experiment manifest: {old_manifest_path}")
        old = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        fields = (
            "full", "months", "train_cap", "validation_cap", "trees",
            "seed", "threads", "features", "data_sha256", "reference_sha256",
        )
        mismatched = [key for key in fields if old.get(key) != canonical_meta.get(key)]
        if mismatched:
            raise ValueError(
                "Previous family caches do not match the new source/config: "
                + ", ".join(mismatched)
            )
    all_frames = []
    periods = sales.date_sold.dt.to_period("M")
    for month in months:
        held, base_predictions = _cohort(
            sales, reference, month, max_validation=validation_cap,
            seed=seed + 11,
        )
        train = sales.loc[(sales.date_sold < pd.Period(month).start_time)
                          & (sales.date_sold >= "1900-01-01")]
        if len(train) > train_cap:
            train = train.sample(n=train_cap, random_state=seed).sort_index()
        if train.empty:
            raise ValueError(f"No historical training sales before {month}")
        hidden = data.hide_answers(held, "off_market")
        if any(key in hidden for key in ("list_price", "sold_price", "date_sold")):
            raise ValueError("Leaky off-market evaluation homes")
        pred = {"frozen_xgb_ensemble": base_predictions}
        base_frame = _frame(held, month, "frozen_xgb_ensemble", base_predictions)
        all_frames.append(base_frame)
        for family in ARMS[1:]:
            path = folder / month / f"{family}.csv"
            saved = _check_saved(path, held, month, family)
            if (saved is None and previous_folder is not None
                and family in ("catboost_offmarket", "lightgbm_offmarket")):
                previous = _check_saved(
                    previous_folder / month / f"{family}.csv",
                    held, month, family,
                )
                if previous is not None:
                    saved = previous
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_csv(path, saved)
                    print(f"{month} {family}: reused verified previous pilot predictions", flush=True)
            if saved is None:
                print(f"{month} {family}: {len(train):,} past training rows; "
                      f"{len(hidden):,} held-out homes", flush=True)
                start = perf_counter()
                p = _model_prediction(train, hidden, family,
                                      full=full, trees=trees, threads=threads)
                saved = _frame(held, month, family, p)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_csv(path, saved)
                print(f"  completed in {perf_counter()-start:.1f}s", flush=True)
            else:
                print(f"{month} {family}: reused completed prediction", flush=True)
            pred[family] = saved.predicted_price.to_numpy(float)
            all_frames.append(saved)
        for name, weights in BLENDS.items():
            logp = sum(weight * np.log(pred[family])
                       for family, weight in weights.items())
            all_frames.append(_frame(held, month, name, np.exp(logp)))
    combined = pd.concat(all_frames, ignore_index=True)
    pooled, monthly = _compare(combined, repeats=bootstrap_repeats, seed=seed)
    write_csv(folder / "predictions.csv", combined)
    write_csv(folder / "pooled_scores.csv", pooled)
    write_csv(folder / "monthly_scores.csv", monthly)
    print("\nOff-market blended models: changes versus frozen ensemble and standalone XGB:")
    print(pooled[["variant", "homes", "mdape", "mape", "p90_ape",
                  "delta_mdape_pp", "delta_mape_pp",
                  "delta_base_xgb_mdape_pp", "delta_base_xgb_mape_pp"]].to_string(index=False))
    print(f"Outputs: {folder}", flush=True)
    return pooled, monthly, combined


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    p.add_argument("--months", nargs="+",
                   default=["2025-08", "2025-09", "2025-10"])
    p.add_argument("--reference-file", type=Path, default=REFERENCE)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--train-cap", type=int)
    p.add_argument("--validation-cap", type=int)
    p.add_argument("--trees", type=int, default=350)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--reuse-family-caches", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-repeats", type=int, default=500)
    args = p.parse_args()
    run_offmarket_blend(
        full=args.full, months=args.months,
        reference_file=args.reference_file, output_dir=args.output_dir,
        train_cap=args.train_cap, validation_cap=args.validation_cap,
        trees=args.trees, threads=args.threads,
        bootstrap_repeats=args.bootstrap_repeats, seed=args.seed,
        reuse_family_caches=args.reuse_family_caches,
    )


if __name__ == "__main__":
    main()
