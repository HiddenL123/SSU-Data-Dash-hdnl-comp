"""Pilot spatial/space-time kernels and a joint GPBoost model.

All residual correctors REQUIRE prior-month out-of-time log-price errors.
The joint GPBoost model fits directly on the historical log-sale-price target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from models.features import make_features
from spacetime_fast import FEATURE_SET


def coords(homes):
    lat = pd.to_numeric(homes.latitude, errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(homes.longitude, errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(homes.date_listed, errors="coerce")
    t = dates.dt.to_period("M").astype("int64").to_numpy(dtype=float)
    if (not np.isfinite(np.column_stack((lat, lon, t))).all() or
        (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any()):
        raise ValueError("Valid lat/lon and listing date required")
    # Approximate local projection: 1 degree longitude at Toronto latitude,
    # 1 degree latitude, and months, in km-equivalent scaled units.
    return np.column_stack((lon * 80.0, lat * 111.0, t * 2.5))


def local_kernel(calibration, residual, homes, *, same_type=False):
    """Anisotropic spatiotemporal Nadaraya-Watson with effective-N shrinkage."""
    xy = coords(calibration)
    query = coords(homes)
    n = min(80, len(calibration))
    if not n:
        raise ValueError("Empty OOT residual calibration")
    d, ix = cKDTree(xy).query(query, k=n, workers=1)
    d = np.asarray(d).reshape(len(query), n)
    ix = np.asarray(ix).reshape(len(query), n)
    w = np.exp(-0.5 * (d / 5.0)**2)
    if same_type:
        types = calibration.house_type_name.fillna("Unknown").astype(str).to_numpy()
        qtype = homes.house_type_name.fillna("Unknown").astype(str).to_numpy()
        w *= np.where(types[ix] == qtype[:, None], 1.0, 0.15)
    total = w.sum(axis=1)
    neff = total**2 / np.maximum((w**2).sum(axis=1), 1e-12)
    pred = (w * residual[ix]).sum(axis=1) / np.maximum(total, 1e-12)
    return pred * np.where(total > 1e-4, neff / (neff + 20.0), 0.0)


def matern_gp(calibration, residual, homes, *, spatial_only=False,
              max_points=350, seed=42):
    """Exact Matérn-3/2 Gaussian-process posterior on an OOT subset.

    Fixed pilot length scale and nugget, so no validation-label optimization.
    Spatial-only and true three-dimensional space-time GP are separate arms.
    """
    x, q = coords(calibration), coords(homes)
    if spatial_only:
        x, q = x[:, :2], q[:, :2]
    residual = np.asarray(residual, dtype=float)
    if len(x) > max_points:
        rng = np.random.default_rng(seed)
        ix = rng.choice(len(x), max_points, replace=False)
        x, residual = x[ix], residual[ix]
    kernel = (ConstantKernel(0.003, constant_value_bounds="fixed") *
              Matern(length_scale=np.full(x.shape[1], 5.0),
                     length_scale_bounds="fixed", nu=1.5) +
              WhiteKernel(noise_level=0.012, noise_level_bounds="fixed"))
    model = GaussianProcessRegressor(
        kernel=kernel, optimizer=None, normalize_y=False, alpha=1e-6,
        random_state=seed,
    ).fit(x, residual)
    return model.predict(q)


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
