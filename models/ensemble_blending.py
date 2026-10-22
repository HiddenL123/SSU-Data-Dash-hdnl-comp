"""Simple auditable log-price blends of already-aligned prediction streams.

No weights are optimized here: use predeclared weights, or a future separate
chronological OOF weight tuner, not the same holdout's scoring labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def blend_predictions(predictions, weights, *, name="fixed_log_blend",
                      variant_col="member"):
    required = {"month", "ml_num", variant_col, "sold_price", "predicted_price"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    if not weights or any(v < 0 or not np.isfinite(v) for v in weights.values()):
        raise ValueError("Blend weights must be finite nonnegative numbers")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Positive total blending weight required")
    df = predictions.copy()
    df["month"] = df.month.astype(str)
    df["ml_num"] = df.ml_num.astype(str)
    keys = ["month", "ml_num"]
    if df.duplicated(keys+[variant_col]).any():
        raise ValueError("Duplicate per-home member prediction")
    if (not np.isfinite(df.predicted_price.to_numpy(dtype=float)).all() or
        df.predicted_price.le(0).any() or
        not np.isfinite(df.sold_price.to_numpy(dtype=float)).all() or
        df.sold_price.le(0).any()):
        raise ValueError("Predictions and sale prices must be finite positive")
    truth_count = df.groupby(keys).sold_price.nunique()
    if not (truth_count == 1).all():
        raise ValueError("Inconsistent actual prices for matched homes")

    members = list(weights)
    if not set(members).issubset(set(df[variant_col])):
        raise ValueError("Requested blend member is missing from prediction input")
    wide = df.pivot(index=keys, columns=variant_col, values="predicted_price")
    selected = wide[members]
    if selected.isna().any().any():
        raise ValueError("Members did not predict the exact same homes")
    output = np.exp(
        np.log(selected.to_numpy(dtype=float))
        @ (np.array([weights[n] for n in members], dtype=float) / total)
    )
    truth = df.groupby(keys).sold_price.first().loc[selected.index]
    result = selected.index.to_frame(index=False).reset_index(drop=True)
    result["member"] = name
    result["sold_price"] = truth.to_numpy(dtype=float)
    result["predicted_price"] = output
    return result


def score_blends(predictions, blends, *, variant_col="member"):
    """Return a paired metric table and separate blended prediction stream."""
    created, metrics = [], []
    for name, weights in blends.items():
        p = blend_predictions(
            predictions, weights, name=name, variant_col=variant_col
        )
        error = 100 * np.abs(p.predicted_price.to_numpy()-p.sold_price.to_numpy()) / p.sold_price.to_numpy()
        metrics.append({
            "member": name, "homes": len(p),
            "mdape": float(np.median(error)), "mape": float(np.mean(error)),
        })
        created.append(p)
    return pd.DataFrame(metrics).sort_values(["mdape", "mape"]), pd.concat(created, ignore_index=True)
