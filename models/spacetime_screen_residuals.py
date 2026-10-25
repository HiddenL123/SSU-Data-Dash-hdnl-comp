"""Nonlinear spatiotemporal residual corrections for the fast screening pipeline.

Input targets MUST be historical out-of-time residuals, never in-sample
training errors. All predictions are on the log-price scale. Residual
methods use only coordinates and listing month, so their outputs form a
low-compute correction to a property-level boosting model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

RESIDUAL_VARIANTS = (
    "knn",
    "local_kernel",
    "rbf_ridge",
    "spline_gam",
    "tensor_gam",
    "extra_trees",
    "hist_boost",
    "small_gp",
)


def residual_coords(homes):
    dates = pd.to_datetime(homes["date_listed"], errors="coerce")
    months = dates.dt.to_period("M").astype("int64").to_numpy(dtype=float)
    lat = pd.to_numeric(homes["latitude"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(homes["longitude"], errors="coerce").to_numpy(dtype=float)
    return np.column_stack([lat, lon, months])


class SpaceTimeResidual:
    """A spatial-temporal correction fitted on genuine prior predictions."""

    def __init__(self, variant, *, max_gp_points=350, seed=42):
        if variant not in RESIDUAL_VARIANTS:
            raise ValueError(f"Unknown residual method: {variant}")
        self.variant = variant
        self.max_gp_points = int(max_gp_points)
        self.seed = seed
        self.fill_ = None
        self.anchor_ = None
        self.estimator_ = None
        self.transformers_ = None
        self.training_months_ = None

    def _features(self, homes):
        xyz = residual_coords(homes)
        xyz = np.where(np.isfinite(xyz), xyz, self.fill_)
        # Mean-centre locations; units are approximately km and months.
        # Scale time to a useful nonzero distance for nearest-neighbour kernels.
        return np.column_stack([
            (xyz[:, 0] - self.anchor_[0]) * 111.0,
            (xyz[:, 1] - self.anchor_[1]) * 80.0,
            (xyz[:, 2] - self.anchor_[2]) * 2.0,
        ])

    def _spline_matrix(self, x, *, fit):
        if fit:
            self.transformers_ = [
                SplineTransformer(n_knots=5, degree=3, include_bias=False).fit(x[:, [j]])
                for j in range(3)
            ]
        parts = [tr.transform(x[:, [j]]) for j, tr in enumerate(self.transformers_)]
        columns = list(parts)
        if self.variant == "tensor_gam":
            for a, b in ((0, 1), (0, 2), (1, 2)):
                columns.append(
                    np.einsum("ij,ik->ijk", parts[a], parts[b])
                    .reshape(len(x), -1)
                )
        return np.column_stack(columns)

    def fit(self, homes, residuals):
        y = np.asarray(residuals, dtype=float).reshape(-1)
        if not len(y) or len(y) != len(homes) or not np.isfinite(y).all():
            raise ValueError("Residual training labels must be aligned and finite")
        coords = residual_coords(homes)
        self.fill_ = np.nanmedian(
            np.where(np.isfinite(coords), coords, np.nan), axis=0
        )
        self.fill_ = np.where(np.isfinite(self.fill_), self.fill_, 0.0)
        coords = np.where(np.isfinite(coords), coords, self.fill_)
        self.anchor_ = np.mean(coords, axis=0)
        self.training_months_ = coords[:, 2].copy()
        x = self._features(homes)

        if self.variant == "knn":
            self.estimator_ = KNeighborsRegressor(
                n_neighbors=min(45, len(x)), weights="distance", n_jobs=-1
            ).fit(x, y)
        elif self.variant == "local_kernel":
            # Gaussian-distance weighting from the 80 nearest calibration
            # residuals; local smoother, distinct from inverse-distance KNN.
            self.estimator_ = KNeighborsRegressor(
                n_neighbors=min(80, len(x)), weights=lambda d: np.exp(
                    -0.5 * (d / 6.0) ** 2
                ) + 1e-9, n_jobs=-1,
            ).fit(x, y)
        elif self.variant == "rbf_ridge":
            self.estimator_ = make_pipeline(
                StandardScaler(),
                RBFSampler(gamma=0.45, n_components=128, random_state=self.seed),
                Ridge(alpha=12.0),
            ).fit(x, y)
        elif self.variant in ("spline_gam", "tensor_gam"):
            basis = self._spline_matrix(x, fit=True)
            self.estimator_ = Ridge(alpha=25.0).fit(basis, y)
        elif self.variant == "extra_trees":
            self.estimator_ = ExtraTreesRegressor(
                n_estimators=75, max_depth=10, min_samples_leaf=18,
                n_jobs=-1, random_state=self.seed
            ).fit(x, y)
        elif self.variant == "hist_boost":
            self.estimator_ = HistGradientBoostingRegressor(
                max_iter=120, max_depth=3, learning_rate=0.04,
                min_samples_leaf=25, l2_regularization=12.0,
                random_state=self.seed,
            ).fit(x, y)
        elif self.variant == "small_gp":
            rng = np.random.default_rng(self.seed)
            ix = (rng.choice(len(x), size=self.max_gp_points, replace=False)
                  if len(x) > self.max_gp_points else np.arange(len(x)))
            self.estimator_ = GaussianProcessRegressor(
                kernel=ConstantKernel(0.01, constant_value_bounds="fixed")
                * RBF(length_scale=[8.0, 8.0, 4.0],
                      length_scale_bounds="fixed")
                + WhiteKernel(noise_level=0.04, noise_level_bounds="fixed"),
                optimizer=None, alpha=1e-5, normalize_y=False,
                random_state=self.seed,
            ).fit(x[ix], y[ix])
        return self

    def predict(self, homes):
        if self.estimator_ is None:
            raise RuntimeError("Call fit() on chronological calibration residuals")
        # Eligibility is governed by the SALE dates of calibration labels,
        # versus the prediction ORIGIN in the outer runner. We cannot enforce
        # it here with date_listed: a home sold in October may have been listed
        # in September, even though its prediction is made at October origin.
        x = self._features(homes)
        if self.variant in ("spline_gam", "tensor_gam"):
            x = self._spline_matrix(x, fit=False)
        return np.asarray(self.estimator_.predict(x), dtype=float).reshape(-1)
