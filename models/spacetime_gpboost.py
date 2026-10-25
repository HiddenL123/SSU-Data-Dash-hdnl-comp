"""Optional joint GPBoost tree + space-time Matérn covariance model.

This is a model-class challenger to the tuned XGBoost, not a GAM/residual
correction and not an alias for XGBoost. It uses the selected-binary property
feature provider, but fits an independent GPBoost mean/GP model.
Install optional dependency: pip install gpboost
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.features import make_features
from features.spacetime_structural_features import FEATURE_SET, SPLINE_GEO_SUFFIX, coords


def gpboost_joint(train, homes, *, rounds=220, seed=42, return_components=False):
    """Fit GPBoost boosting+Matérn GP jointly, using the same spline+geo XGBoost feature family.

    GPBoost has its own LightGBM-style tree, so this is a model-class
    challenger, not an identical XGBoost with a different covariance option.

    When return_components=True, preserve combined, tree-only, and random
    effects on the centered log-price scale for notebook diagnostics.
    """
    try:
        import gpboost as gpb
    except ImportError as exc:
        raise ImportError("Optional dependency: python -m pip install gpboost") from exc

    feature_provider = make_features(
        FEATURE_SET + SPLINE_GEO_SUFFIX, model="xgboost", track="off_market"
    )
    X = feature_provider.fit_transform(train).copy()
    H = feature_provider.transform(homes).copy()
    if list(X.columns) != list(H.columns):
        raise ValueError("Different train/test GPBoost feature schemas")
    categorical = []
    for c in X.columns:
        left, right = X[c], H[c]
        if (pd.api.types.is_object_dtype(left.dtype) or
            isinstance(left.dtype, pd.CategoricalDtype) or
            pd.api.types.is_string_dtype(left.dtype)):
            codes = {v: i for i, v in enumerate(left.dropna().astype(str).unique())}
            X[c] = left.astype("string").map(codes).fillna(-1).astype("int32")
            H[c] = right.astype("string").map(codes).fillna(-1).astype("int32")
            categorical.append(c)
        else:
            X[c] = pd.to_numeric(left, errors="coerce").astype("float32")
            H[c] = pd.to_numeric(right, errors="coerce").astype("float32")
    y = np.log(train.sold_price.to_numpy(dtype=float))
    intercept = float(np.median(y))
    gp = gpb.GPModel(
        gp_coords=coords(train), likelihood="gaussian",
        cov_function="matern_ard", cov_fct_shape=1.5,
        gp_approx="vecchia", num_neighbors=15,
        seed=seed, num_parallel_threads=4,
    )
    dataset = gpb.Dataset(
        X, label=y-intercept, categorical_feature=categorical
    )
    model = gpb.train(
        params=dict(
            objective="regression_l2", learning_rate=0.055,
            num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
            verbose=-1, num_threads=4, seed=seed,
        ),
        train_set=dataset, gp_model=gp, num_boost_round=rounds,
        verbose_eval=False,
    )
    test_coordinates = coords(homes)
    result = model.predict(
        data=H, gp_coords_pred=test_coordinates, pred_latent=False,
        predict_var=False,
    )
    combined_log_centered = np.asarray(
        result["response_mean"], dtype=float
    ).reshape(-1)
    if (combined_log_centered.shape != (len(homes),) or
        not np.isfinite(combined_log_centered).all()):
        raise ValueError("GPBoost returned nonfinite/misaligned response_mean")
    combined_price = np.exp(intercept + combined_log_centered)
    if not return_components:
        return combined_price

    # The package documentation specifies these keys for a Gaussian
    # likelihood. Inspect BOTH latent components; do not infer them from
    # the final sale-price predictions after exponentiating.
    latent = model.predict(
        data=H, gp_coords_pred=test_coordinates, pred_latent=True,
        predict_var=False,
    )
    fixed_log = np.asarray(latent["fixed_effect"], dtype=float).reshape(-1)
    random_log = np.asarray(
        latent["random_effect_mean"], dtype=float
    ).reshape(-1)
    tree_log = np.asarray(
        model.predict(data=H, ignore_gp_model=True), dtype=float
    ).reshape(-1)
    for key, value in (
        ("fixed_effect", fixed_log),
        ("random_effect_mean", random_log),
        ("tree_only", tree_log),
    ):
        if value.shape != (len(homes),) or not np.isfinite(value).all():
            raise ValueError(f"GPBoost diagnostic {key} invalid")
    # If this identity fails for Gaussian likelihood, the outputs are
    # being misinterpreted or a package-level prediction option differs.
    log_decomposition_gap = float(
        np.max(np.abs(combined_log_centered - fixed_log - random_log))
    ) if len(homes) else 0.0
    return {
        "price_combined": combined_price,
        "price_tree_only": np.exp(intercept + tree_log),
        "log_combined": intercept + combined_log_centered,
        "log_fixed": intercept + fixed_log,
        "log_random_effect": random_log,
        "log_tree_only": intercept + tree_log,
        "log_decomposition_gap": log_decomposition_gap,
        "log_tree_vs_fixed_gap": float(
            np.max(np.abs(tree_log - fixed_log))
        ) if len(homes) else 0.0,
        "train_target_offset": intercept,
        "train_log_target_min": float(np.min(y)),
        "train_log_target_median": float(np.median(y)),
        "train_log_target_max": float(np.max(y)),
        "training_rows": int(len(train)),
        "feature_count": int(X.shape[1]),
        "categorical_feature_count": int(len(categorical)),
        "model_rounds": int(rounds),
    }
