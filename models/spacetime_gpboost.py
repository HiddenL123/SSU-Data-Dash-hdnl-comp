"""Optional joint GPBoost tree + space-time Matérn covariance model.

This is a model-class challenger to the tuned XGBoost, not a GAM/residual
correction and not an alias for XGBoost. It uses the selected-binary property
feature provider, but fits an independent GPBoost mean/GP model.
Install optional dependency: pip install gpboost
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.features import make_features
from models.spacetime_structural_features import FEATURE_SET, coords


def gpboost_joint(train, homes, *, rounds=220, seed=42):
    """Fit GPBoost boosting+Matérn GP jointly, using the same XGBoost features.

    GPBoost has its own LightGBM-style tree, so this is a model-class
    challenger, not an identical XGBoost with a different covariance option.
    """
    try:
        import gpboost as gpb
    except ImportError as exc:
        raise ImportError("Optional dependency: python -m pip install gpboost") from exc

    feature_provider = make_features(
        FEATURE_SET, model="xgboost", track="off_market"
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
    result = model.predict(
        data=H, gp_coords_pred=coords(homes), pred_latent=False,
        predict_var=False,
    )
    return np.exp(intercept + np.asarray(result["response_mean"], dtype=float))
