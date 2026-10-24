"""Fast, chronological shotgun screen for spatiotemporal XGBoost improvements.

The FEATURE arm retrains XGBoost on the same reduced training sample while
adding one feature family at a time. The RESIDUAL arm trains lightweight
space-time corrections on genuine previous-month out-of-time predictions of
the baseline model. Both arms score the SAME untouched development cohort.

This is deliberately exploratory. Screening performance is not directly
comparable to the 7.82% full-history, 700-tree benchmark. After selecting a
few contenders, rerun them with the complete training history and the
existing final/fixed-origin validation procedures.

Example:
  uv run python spacetime_shotgun.py --validation-month 2025-10
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import data
from models.defaults import DEFAULT_PARAMS
from models.features import make_features
from models.spacetime_screen_features import FEATURE_VARIANTS, SpaceTimeAddon
from models.spacetime_screen_residuals import RESIDUAL_VARIANTS, SpaceTimeResidual

# The current tuned XGBoost Binary BoW model_run.ipynb parameters, except that
# n_estimators is deliberately capped by screen_trees (default 220).
XGB_TUNED = {
    "n_estimators": 700,
    "learning_rate": 0.09522271657253033,
    "max_depth": 8,
    "min_child_weight": 16.684619386293466,
    "reg_lambda": 19.50331236061207,
    "reg_alpha": 0.0,
    "subsample": 0.9804685587557792,
    "colsample_bytree": 0.8994749947027713,
    "tree_method": "hist",
    "grow_policy": "depthwise",
    "device": "cpu",
    "enable_categorical": True,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

FEATURE_SET = "Xpt_spatiotemporal__selected_binary"
COMPARABLE_VARIANTS = {
    "comp_age_first": "__comp_age_first",
    "comp_type_quota": "__comp_type_quota",
    "comp_both": "__comp_both",
}
SCREEN_FEATURE_VARIANTS = (*FEATURE_VARIANTS, *COMPARABLE_VARIANTS)
# Keep earlier generic shotgun defaults unchanged; new AVM arms are opt-in.
DEFAULT_SCREEN_FEATURE_VARIANTS = tuple(
    name for name in SCREEN_FEATURE_VARIANTS if not name.startswith("rw_")
)



def _cap(df, n, seed):
    if n < 1:
        raise ValueError("Sample limits must be positive")
    if len(df) <= n:
        return df.copy()
    # Sorted original indices preserve deterministic alignments with features.
    return df.sample(n=n, random_state=seed).sort_index()


def _metric(true, prediction):
    true = np.asarray(true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    signed = 100 * (prediction - true) / true
    return {
        "homes": int(len(true)),
        "mape": float(np.mean(np.abs(signed))),
        "mdape": float(np.median(np.abs(signed))),
        "median_signed": float(np.median(signed)),
    }


def _make_estimator(params, screen_trees):
    kwargs = dict(DEFAULT_PARAMS["xgboost"])
    kwargs.update(params)
    kwargs["n_estimators"] = screen_trees
    kwargs["device"] = "cpu"
    kwargs["n_jobs"] = 4  # controlled screening cost, not all local cores
    kwargs["enable_categorical"] = True
    return XGBRegressor(**kwargs)


def _base_features(train, homes, *, feature_set=FEATURE_SET):
    provider = make_features(feature_set, model="xgboost", track="off_market")
    return provider.fit_transform(train), provider.transform(homes)


def _train_and_predict(train, homes, params, screen_trees, *, variant="baseline",
                       X_train=None, X_homes=None, feature_set=FEATURE_SET):
    if variant in COMPARABLE_VARIANTS:
        if "__st_" in feature_set:
            raise ValueError("Comparable ablations require base feature sets without __st_ add-ons")
        provider = make_features(
            feature_set + COMPARABLE_VARIANTS[variant],
            model="xgboost", track="off_market"
        )
        X_train, X_homes = provider.fit_transform(train), provider.transform(homes)
    elif X_train is None or X_homes is None:
        X_train, X_homes = _base_features(train, homes, feature_set=feature_set)
    if variant != "baseline" and variant not in COMPARABLE_VARIANTS:
        add = SpaceTimeAddon(variant).fit(train)
        tr = add.transform(train, X_train)
        te = add.transform(homes, X_homes)
        if not tr.index.equals(X_train.index) or not te.index.equals(X_homes.index):
            raise ValueError("Add-on features are not row-aligned")
        if list(tr.columns) != list(te.columns):
            raise ValueError("Training/prediction feature schemas differ")
        if set(tr.columns) & set(X_train.columns):
            raise ValueError("Add-on overlaps base feature names")
        X_train = pd.concat([X_train, tr], axis=1)
        X_homes = pd.concat([X_homes, te], axis=1)

    log_y = np.log(train["sold_price"].to_numpy(dtype=float))
    offset = float(np.median(log_y))
    estimator = _make_estimator(params, screen_trees)
    estimator.fit(X_train, log_y - offset)
    return np.exp(offset + np.asarray(estimator.predict(X_homes), dtype=float))


def _calibration_data(sales, *, origin, train_start, train_cap,
                      calibration_cap, calibration_folds, params,
                      screen_trees, seed, feature_set=FEATURE_SET):
    """Create prior-fold residuals; each label is out-of-time to its base fit."""
    dates = sales["date_sold"]
    collected, residuals = [], []
    for lag in range(calibration_folds, 0, -1):
        month = origin - lag
        training = _cap(
            sales.loc[(dates >= train_start) & (dates < month.start_time)],
            train_cap, seed,
        )
        held = _cap(
            sales.loc[dates.dt.to_period("M") == month],
            calibration_cap, seed + lag,
        )
        if training.empty or held.empty:
            raise ValueError(f"Insufficient data for calibration month {month}")
        hidden = data.hide_answers(held, "off_market")
        pred = _train_and_predict(training, hidden, params, screen_trees,
                                  feature_set=feature_set)
        log_error = np.log(held["sold_price"].to_numpy(dtype=float)) - np.log(pred)
        # Align rows positionally (indices are retained only for consistency).
        collected.append(hidden)
        residuals.append(log_error)
        print(f"  Calibration {month}: {len(training):,} earlier sales, "
              f"{len(held):,} OOT errors", flush=True)
    return pd.concat(collected, axis=0, ignore_index=True), np.concatenate(residuals)


def run_shotgun(*, validation_month="2025-10",
                train_start="2024-01-01",
                train_cap=10000, validation_cap=650,
                calibration_cap=650, calibration_folds=2,
                screen_trees=220, seed=42, params=None,
                base_feature_set=FEATURE_SET,
                features=DEFAULT_SCREEN_FEATURE_VARIANTS, residuals=RESIDUAL_VARIANTS,
                residual_strength=0.5, output_dir="outputs/spacetime_shotgun",
                listings=None):
    """Run independent feature and residual screens on one development fold."""
    origin = pd.Period(validation_month, freq="M")
    if origin >= pd.Period("2025-12", freq="M"):
        raise ValueError("Shotgun screen is limited to months before 2025-12")
    if calibration_folds < 1:
        raise ValueError("calibration_folds must be >= 1")
    if not 0 <= residual_strength <= 1:
        raise ValueError("residual_strength must lie in [0, 1]")
    features, residuals = tuple(features), tuple(residuals)
    if "baseline" not in features or len(set(features)) != len(features):
        raise ValueError("Feature variants must include exactly one baseline")
    if any(x not in SCREEN_FEATURE_VARIANTS for x in features):
        raise ValueError(f"Unknown feature variant: {features}")
    if len(set(residuals)) != len(residuals) or any(
        x not in RESIDUAL_VARIANTS for x in residuals
    ):
        raise ValueError(f"Unknown residual variants: {residuals}")
    for limit in (train_cap, validation_cap, calibration_cap, screen_trees):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("Samples and screen_trees must be positive integers")
    tstart = pd.Timestamp(train_start)
    sales = data.load("listings") if listings is None else listings.copy()
    sales["date_sold"] = pd.to_datetime(sales["date_sold"])
    sales["date_listed"] = pd.to_datetime(sales["date_listed"])
    if tstart >= (origin - calibration_folds).start_time:
        raise ValueError("train_start must precede every calibration fold")
    dates = sales["date_sold"]
    train = _cap(
        sales.loc[(dates >= tstart) & (dates < origin.start_time)], train_cap, seed
    )
    validation = _cap(
        sales.loc[dates.dt.to_period("M") == origin],
        validation_cap, seed + 11
    )
    if train.empty or validation.empty:
        raise ValueError("Empty training or validation cohort")
    homes = data.hide_answers(validation, "off_market")
    prices = validation["sold_price"].to_numpy(dtype=float)
    params = dict(XGB_TUNED if params is None else params)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, predictions, failures = [], [], []
    metadata = {
        "validation_month": str(origin),
        "train_start": str(tstart.date()),
        "train_cap": train_cap, "actual_train_rows": len(train),
        "validation_cap": validation_cap,
        "actual_validation_rows": len(validation),
        "calibration_cap": calibration_cap, "calibration_folds": calibration_folds,
        "screen_trees": screen_trees, "seed": seed, "params": params,
        "base_feature_set": base_feature_set,
        "residual_strength": residual_strength,
        "feature_variants": list(features),
        "residual_variants": list(residuals),
        "note": "Reduced-sample development SCREEN; not full-history confirmation",
    }
    (out / "setup.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def record(arm, name, prediction, baseline):
        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise ValueError(f"Nonfinite/nonpositive predicted prices: {name}")
        score = _metric(prices, prediction)
        score.update(arm=arm, variant=name,
                     delta_mdape_pp=score["mdape"] - baseline["mdape"],
                     delta_mape_pp=score["mape"] - baseline["mape"])
        summary_rows.append(score)
        predictions.append(pd.DataFrame({
            "ml_num": validation["ml_num"].to_numpy(),
            "month": str(origin), "arm": arm, "variant": name,
            "sold_price": prices, "predicted_price": prediction,
            "abs_pct_error": 100 * np.abs(prediction - prices) / prices,
        }))
        pd.DataFrame(summary_rows).to_csv(out / "summary_partial.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(
            out / "predictions_partial.csv", index=False
        )
        print(f"  {arm:8s} {name:21s} MdAPE {score['mdape']:.3f}% "
              f"MAPE {score['mape']:.3f}% (ΔMdAPE {score['delta_mdape_pp']:+.3f}pp)",
              flush=True)

    print(f"Shotgun: {len(train):,} historical sales; "
          f"{len(homes):,} unseen {origin} homes; "
          f"{screen_trees} trees per XGBoost", flush=True)
    X_train, X_homes = _base_features(train, homes,
                                       feature_set=base_feature_set)
    control_pred = _train_and_predict(
        train, homes, params, screen_trees,
        X_train=X_train, X_homes=X_homes,
        feature_set=base_feature_set,
    )
    baseline = _metric(prices, control_pred)
    record("feature", "baseline", control_pred, baseline)

    for variant in features:
        if variant == "baseline":
            continue
        try:
            estimate = _train_and_predict(
                train, homes, params, screen_trees, variant=variant,
                X_train=X_train, X_homes=X_homes,
                feature_set=base_feature_set,
            )
            record("feature", variant, estimate, baseline)
        except Exception as exc:
            failures.append({"arm": "feature", "variant": variant,
                             "error": str(exc), "traceback": traceback.format_exc()})
            print(f"  FAILED feature {variant}: {exc}", flush=True)

    if residuals:
        try:
            calib_homes, calib_errors = _calibration_data(
                sales, origin=origin, train_start=tstart,
                train_cap=train_cap, calibration_cap=calibration_cap,
                calibration_folds=calibration_folds, params=params,
                screen_trees=screen_trees, seed=seed,
                feature_set=base_feature_set,
            )
            for variant in residuals:
                try:
                    correction = SpaceTimeResidual(variant, seed=seed).fit(
                        calib_homes, calib_errors
                    ).predict(homes)
                    estimate = control_pred * np.exp(
                        residual_strength * np.clip(correction, -0.25, 0.25)
                    )
                    record("residual", variant, estimate, baseline)
                except Exception as exc:
                    failures.append({"arm": "residual", "variant": variant,
                                     "error": str(exc), "traceback": traceback.format_exc()})
                    print(f"  FAILED residual {variant}: {exc}", flush=True)
        except Exception as exc:
            failures.append({"arm": "calibration", "variant": "all",
                             "error": str(exc), "traceback": traceback.format_exc()})
            print(f"  FAILED residual calibration: {exc}", flush=True)

    ranked = pd.DataFrame(summary_rows).sort_values(["mdape", "mape"])
    ranked.to_csv(out / "summary.csv", index=False)
    result_predictions = pd.concat(predictions, ignore_index=True)
    result_predictions.to_csv(out / "predictions.csv", index=False)
    (out / "failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    print("\nRanked screen (lower is better):\n",
          ranked.to_string(index=False), flush=True)
    if failures:
        print(f"{len(failures)} candidates failed; see {out / 'failures.json'}")
    print(f"Saved results in {out.resolve()}")
    return ranked, result_predictions, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-month", default="2025-10")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--train-cap", type=int, default=10000)
    parser.add_argument("--validation-cap", type=int, default=650)
    parser.add_argument("--calibration-cap", type=int, default=650)
    parser.add_argument("--calibration-folds", type=int, default=2)
    parser.add_argument("--screen-trees", type=int, default=220)
    parser.add_argument("--features", nargs="+", choices=SCREEN_FEATURE_VARIANTS,
                        default=list(DEFAULT_SCREEN_FEATURE_VARIANTS))
    parser.add_argument("--residuals", nargs="*", choices=RESIDUAL_VARIANTS,
                        default=list(RESIDUAL_VARIANTS))
    parser.add_argument("--params", type=Path,
                        help="Optional full tuned XGBoost params JSON")
    parser.add_argument("--output-dir", default="outputs/spacetime_shotgun")
    args = parser.parse_args()
    overrides = (json.loads(args.params.read_text(encoding="utf-8"))
                 if args.params else None)
    if overrides is not None and not isinstance(overrides, dict):
        parser.error("--params must be a JSON object")
    run_shotgun(
        validation_month=args.validation_month,
        train_start=args.train_start, train_cap=args.train_cap,
        validation_cap=args.validation_cap,
        calibration_cap=args.calibration_cap,
        calibration_folds=args.calibration_folds,
        screen_trees=args.screen_trees,
        params=overrides, features=args.features,
        residuals=args.residuals, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
