"""Structural spatial and space-time correction models (experimental).

Caller must fit targets using strictly earlier OUT-OF-TIME predictions,
never in-sample fitted residuals. These pilots use a local kernel method and
a small exact Matérn GP, separate from the root experiment runner.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from models.spacetime_structural_features import coords


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

