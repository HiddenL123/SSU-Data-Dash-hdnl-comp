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

from features.spacetime_structural_features import coords


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
              max_points=350, seed=42, spatial_km=5.0,
              temporal_months=2.0, amplitude_var=0.003,
              noise_var=0.012):
    """Matérn-3/2 GP correcting out-of-time log-price residuals.

    Parameters describe *covariance* rather than feature engineering:
      spatial_km: correlation length in projected kilometres (east and north)
      temporal_months: correlation length in months, space-time variant only
      amplitude_var: latent residual-process variance (log-price squared)
      noise_var: observation nugget variance (log-price squared)

    Existing no-argument call matches the original fixed screening kernel
    exactly (5 km, 2 months * 2.5 coordinate units, .003, .012).
    GP hyperparameters are fixed during the fit; the separate chronological
    tuning runner selects them on earlier OOT calibration validation months.
    """
    if (max_points < 2 or spatial_km <= 0 or temporal_months <= 0 or
        amplitude_var <= 0 or noise_var <= 0 or
        not np.isfinite([spatial_km, temporal_months,
                         amplitude_var, noise_var]).all()):
        raise ValueError("Matérn kernel scales, variances, and cap must be positive")
    x, q = coords(calibration), coords(homes)
    length_scale = [float(spatial_km), float(spatial_km)]
    if spatial_only:
        x, q = x[:, :2], q[:, :2]
    else:
        # The shared coordinate transform measures time in 2.5 units/month.
        length_scale.append(float(temporal_months) * 2.5)
    residual = np.asarray(residual, dtype=float)
    if not len(x) or len(residual) != len(x) or not np.isfinite(residual).all():
        raise ValueError("Matérn residual labels must be finite and aligned")
    if len(x) > max_points:
        rng = np.random.default_rng(seed)
        ix = rng.choice(len(x), max_points, replace=False)
        x, residual = x[ix], residual[ix]
    kernel = (ConstantKernel(amplitude_var, constant_value_bounds="fixed") *
              Matern(length_scale=np.asarray(length_scale),
                     length_scale_bounds="fixed", nu=1.5) +
              WhiteKernel(noise_level=noise_var,
                          noise_level_bounds="fixed"))
    model = GaussianProcessRegressor(
        kernel=kernel, optimizer=None, normalize_y=False, alpha=1e-6,
        random_state=seed,
    ).fit(x, residual)
    return model.predict(q)

