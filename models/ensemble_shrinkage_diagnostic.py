"""Fixed-weight shrinkage-versus-blending diagnostic for spatial residuals.

All predictions are ALREADY held-out, and the method never fits XGBoost, GAM,
GP, blend weights, or shrinkage factors. Exact matched (month, ml_num) keys
and the shared baseline are mandatory.

The source variants are prices from XGB + log-residual correction, including
their historical correction strengths and clipping. Recover the *effective*
log correction as log(corrected_price) - log(base_price) and then ask:
does blending DIFFERENT corrections improve beyond merely halving ONE?

This is development-only; Sep-Nov 2025 have been consulted repeatedly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE = "baseline"
GAM = "xgb_tensor_gam"
MATERN_VARIANTS = (
    "xgb_spacetime_matern_tuned",
    "xgb_spatial_matern_tuned",
)
VERSION = 1
KEYS = ("month", "ml_num")


def _prepare(predictions: pd.DataFrame, required_variants):
    columns = {"month", "ml_num", "variant", "sold_price", "predicted_price"}
    missing = columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction input missing columns: {sorted(missing)}")
    df = predictions[list(columns)].copy()
    df["month"] = df.month.astype(str)
    df["ml_num"] = df.ml_num.astype(str)
    df["variant"] = df.variant.astype(str)
    if df.empty or df[list(KEYS) + ["variant"]].isna().any().any():
        raise ValueError("Missing prediction identity / no prediction rows")
    if df.duplicated([*KEYS, "variant"]).any():
        raise ValueError("Duplicate (month, ml_num, variant) predictions")
    for column in ("sold_price", "predicted_price"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(df[column].to_numpy()).all() or (df[column] <= 0).any():
            raise ValueError(f"{column} must contain only finite positive prices")
    missing_variants = set(required_variants) - set(df.variant)
    if missing_variants:
        raise ValueError(f"Missing required variants: {sorted(missing_variants)}")
    # A complete match is mandatory. A partial cohort could make any apparent
    # ensemble advantage a selection artifact.
    source = df[df.variant.isin(required_variants)].copy()
    identity_count = source.groupby(list(KEYS), sort=True).variant.nunique()
    if not identity_count.eq(len(required_variants)).all():
        raise ValueError("All compared variants must predict exactly the same homes")
    if not source.groupby(list(KEYS)).sold_price.nunique().eq(1).all():
        raise ValueError("Disagreement in actual prices for matched homes")

    wide = source.pivot(index=list(KEYS), columns="variant", values="predicted_price")
    wide = wide[list(required_variants)].sort_index()
    truth = source.groupby(list(KEYS)).sold_price.first().reindex(wide.index)
    if wide.isna().any().any() or truth.isna().any():
        raise ValueError("Misaligned prediction matrix")
    return wide, truth


def _pct_error(truth, prices):
    return 100.0 * np.abs(np.asarray(prices) - truth) / truth


def _metrics(y, p):
    ape = _pct_error(y, p)
    signed = 100.0 * (np.asarray(p) - y) / y
    return {
        "homes": len(y),
        "mdape": float(np.median(ape)),
        "mape": float(np.mean(ape)),
        "p90_ape": float(np.quantile(ape, .9)),
        "median_signed_pct": float(np.median(signed)),
    }


def _bootstrap_index(months, repeats, seed):
    rng = np.random.default_rng(seed)
    groups = [
        np.flatnonzero(months == month) for month in sorted(set(months))
    ]
    return [
        np.concatenate([
            rng.choice(ids, size=len(ids), replace=True) for ids in groups
        ]) for _ in range(repeats)
    ]


def _pairwise(y, p, q, *, months, bootstrap_indices):
    """p is challenger, q is reference; negative delta favors challenger."""
    a = _pct_error(y, p)
    b = _pct_error(y, q)
    delta_mdape = float(np.median(a) - np.median(b))
    delta_mape = float(np.mean(a - b))
    result = {
        "delta_mdape_pp": delta_mdape,
        "delta_mape_pp": delta_mape,
        "homes_improved_pct": float(100 * np.mean(a < b)),
        "months_better_mdape": int(sum(
            np.median(a[months == month]) < np.median(b[months == month])
            for month in sorted(set(months))
        )),
    }
    if bootstrap_indices:
        dmd = np.array([
            np.median(a[idx]) - np.median(b[idx])
            for idx in bootstrap_indices
        ])
        dma = np.array([np.mean(a[idx] - b[idx]) for idx in bootstrap_indices])
        result.update({
            "mdape_ci_low": float(np.quantile(dmd, .025)),
            "mdape_ci_high": float(np.quantile(dmd, .975)),
            "mape_ci_low": float(np.quantile(dma, .025)),
            "mape_ci_high": float(np.quantile(dma, .975)),
        })
    else:
        result.update({
            k: np.nan for k in (
                "mdape_ci_low", "mdape_ci_high", "mape_ci_low", "mape_ci_high"
            )
        })
    return result


def run_shrinkage_vs_blending(
    predictions: pd.DataFrame, *,
    gam: str = GAM,
    matern_variants=MATERN_VARIANTS,
    baseline: str = BASELINE,
    shrinkage: float = 0.5,
    bootstrap_repeats: int = 500,
    seed: int = 42,
    output_dir: str | Path | None = None,
):
    """Evaluate preset shrinkage controls against preset geometric blends.

    Returns (pooled, monthly, pairwise, correlations, per_home_predictions).

    A 50/50 price blend corresponds to 0.5*r_gam + 0.5*r_matern
    ON THE LOG-PRICE SCALE, where r is the EFFECTIVE already-shrunk/
    clipped correction implied by the source models.

    Per-member shrinkage uses baseline + shrinkage*r_member.
    'blend_half_strength' adds shrinkage*(0.5*r_gam + 0.5*r_matern)
    to separate magnitude shrinkage from mixing two different patterns.

    Bootstrap uncertainty is home-resampled and stratified by month. It does
    not account for within-month geographical dependence or retrospective
    model selection; interpret intervals descriptively.
    """
    if not np.isfinite(shrinkage) or not 0 < shrinkage < 1:
        raise ValueError("shrinkage must be finite and strictly between 0 and 1")
    if not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be a nonnegative integer")
    matern_variants = tuple(matern_variants)
    if not matern_variants or len(matern_variants) != len(set(matern_variants)):
        raise ValueError("Supply one or more distinct Matérn variants")
    required = (baseline, gam, *matern_variants)
    if len(set(required)) != len(required):
        raise ValueError("Baseline, GAM and Matérn variant names must be distinct")
    wide, truth = _prepare(predictions, required)
    index = wide.index
    months = index.get_level_values("month").to_numpy(dtype=str)
    y = truth.to_numpy(dtype=float)
    base_log = np.log(wide[baseline].to_numpy(dtype=float))
    gam_log = np.log(wide[gam].to_numpy(dtype=float))
    gam_effect = gam_log - base_log
    log_streams = {
        baseline: base_log,
        gam: gam_log,
        "gam_half_strength": base_log + shrinkage * gam_effect,
    }
    corr_rows = []
    for matern in matern_variants:
        mat_log = np.log(wide[matern].to_numpy(dtype=float))
        mat_effect = mat_log - base_log
        label = ("spacetime" if matern == "xgb_spacetime_matern_tuned"
                 else "spatial" if matern == "xgb_spatial_matern_tuned"
                 else matern)
        if label in {"gam", "baseline", "gam_half_strength"}:
            raise ValueError("Matérn label clashes with existing output name")
        mixed = 0.5 * (gam_effect + mat_effect)
        log_streams[matern] = mat_log
        log_streams[f"{label}_half_strength"] = base_log + shrinkage * mat_effect
        log_streams[f"gam_{label}_50_50"] = base_log + mixed
        log_streams[f"gam_{label}_blend_half_strength"] = (
            base_log + shrinkage * mixed
        )
        corr_rows.append({
            "matern": matern,
            "homes": len(y),
            "gam_matern_log_correction_pearson": (
                float(np.corrcoef(gam_effect, mat_effect)[0, 1])
                if np.std(gam_effect) > 0 and np.std(mat_effect) > 0
                else np.nan
            ),
            "gam_matern_log_correction_spearman": float(
                pd.Series(gam_effect).corr(pd.Series(mat_effect), method="spearman")
            ),
            "sign_agreement_pct": float(100 * np.mean(
                np.sign(gam_effect) == np.sign(mat_effect)
            )),
            "gam_median_abs_correction": float(np.median(np.abs(gam_effect))),
            "matern_median_abs_correction": float(np.median(np.abs(mat_effect))),
            "gam_p95_abs_correction": float(np.quantile(np.abs(gam_effect), .95)),
            "matern_p95_abs_correction": float(np.quantile(np.abs(mat_effect), .95)),
        })

    prices = {name: np.exp(z) for name, z in log_streams.items()}
    for name, p in prices.items():
        if not np.isfinite(p).all() or np.any(p <= 0):
            raise ValueError(f"Invalid corrected prices for {name}")

    baseline_metrics = _metrics(y, prices[baseline])
    pooled_rows = []
    for name, p in prices.items():
        metrics = _metrics(y, p)
        pooled_rows.append({
            "variant": name, **metrics,
            "delta_mdape_vs_baseline_pp":
                metrics["mdape"] - baseline_metrics["mdape"],
            "delta_mape_vs_baseline_pp":
                metrics["mape"] - baseline_metrics["mape"],
        })
    pooled = pd.DataFrame(pooled_rows).sort_values(["mdape", "mape"]).reset_index(drop=True)

    monthly_rows = []
    for name, p in prices.items():
        for month in sorted(set(months)):
            mask = months == month
            metrics = _metrics(y[mask], p[mask])
            ref = _metrics(y[mask], prices[baseline][mask])
            monthly_rows.append({
                "month": month, "variant": name, **metrics,
                "delta_mdape_vs_baseline_pp": metrics["mdape"] - ref["mdape"],
                "delta_mape_vs_baseline_pp": metrics["mape"] - ref["mape"],
            })
    monthly = pd.DataFrame(monthly_rows).sort_values(["month", "mdape"])

    boot = _bootstrap_index(months, bootstrap_repeats, seed)
    pairs = []
    for matern in matern_variants:
        label = ("spacetime" if matern == "xgb_spacetime_matern_tuned"
                 else "spatial" if matern == "xgb_spatial_matern_tuned"
                 else matern)
        blend_name = f"gam_{label}_50_50"
        for reference in (
            baseline, gam, matern,
            "gam_half_strength",
            f"{label}_half_strength",
            f"gam_{label}_blend_half_strength",
        ):
            pairs.append({
                "challenger": blend_name,
                "reference": reference,
                **_pairwise(
                    y, prices[blend_name], prices[reference],
                    months=months, bootstrap_indices=boot,
                ),
            })
    pairwise = pd.DataFrame(pairs)
    correlations = pd.DataFrame(corr_rows)

    key_frame = index.to_frame(index=False).reset_index(drop=True)
    prediction_rows = []
    for name, p in prices.items():
        df = key_frame.copy()
        df["variant"] = name
        df["sold_price"] = y
        df["predicted_price"] = p
        prediction_rows.append(df)
    made_predictions = pd.concat(prediction_rows, ignore_index=True)

    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        # Fingerprint the canonical aligned input to prevent accidentally
        # replacing results from different homes or updated predictions.
        canonical = wide.copy()
        canonical["sold_price"] = truth
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(canonical, index=True)
            .to_numpy(dtype="uint64").tobytes()
        ).hexdigest()
        manifest = {
            "version": VERSION,
            "source_fingerprint": fingerprint,
            "required_variants": list(required),
            "months": sorted(set(months)),
            "homes": len(y),
            "shrinkage": shrinkage,
            "blend_weights": [0.5, 0.5],
            "bootstrap_repeats": bootstrap_repeats,
            "seed": seed,
            "note": (
                "Exploratory already-inspected development months; "
                "no fitting/selection of weights or corrections"
            ),
        }
        config = root / "setup.json"
        if config.is_file():
            if json.loads(config.read_text(encoding="utf-8")) != manifest:
                raise ValueError(
                    "Output directory has different data/settings; use a new path"
                )
        config.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name, frame in (
            ("pooled", pooled), ("monthly", monthly),
            ("pairwise", pairwise), ("correlations", correlations),
            ("predictions", made_predictions),
        ):
            frame.to_csv(root / f"{name}.csv", index=False)

    return pooled, monthly, pairwise, correlations, made_predictions
