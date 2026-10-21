"""Small, leakage-safe, swappable spatiotemporal feature add-ons for XGBoost.

This is a research SCREEN, not a production replacement for the existing
ComparableSalesFeatures or models.features registry. All label-derived feature
statistics are constructed from historical sales in .fit() and queried only
using months strictly before each home's listing month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import SplineTransformer, StandardScaler

from models.market_features import MarketLevelFeatures
from models.spatiotemporal_features import ComparableSalesFeatures

FEATURE_VARIANTS = (
    "baseline",
    "market",
    "multiscale",
    "time_adjusted_comp",
    "spacetime_rbf",
    "spacetime_splines",
    "combined",
)


def _coords(homes):
    dates = pd.to_datetime(homes["date_listed"], errors="coerce")
    periods = dates.dt.to_period("M").astype("int64").to_numpy(dtype=float)
    values = np.column_stack([
        pd.to_numeric(homes["latitude"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(homes["longitude"], errors="coerce").to_numpy(dtype=float),
        periods,
    ])
    return values


class SpaceTimeAddon:
    """Fit on one chronological training fold; preserve its column schema."""

    def __init__(self, variant):
        if variant not in FEATURE_VARIANTS:
            raise ValueError(f"Unknown spatiotemporal add-on: {variant}")
        self.variant = variant
        self.market = None
        self.comps = {}
        self.scaler = None
        self.rbf = None
        self.splines = None
        self.fill_coords = None
        self.column_names_ = None

    def fit(self, train):
        modes = self._modes()
        if "market" in modes or "time_adjusted_comp" in modes:
            self.market = MarketLevelFeatures(shrinkage=20).fit(train)
        if "multiscale" in modes:
            for name, params in {
                "local": dict(candidates=64, neighbors=16, radius_km=1.0,
                              half_life_days=90, max_age_days=365),
                "regional": dict(candidates=128, neighbors=32, radius_km=9.0,
                                 half_life_days=365, max_age_days=730),
            }.items():
                self.comps[name] = ComparableSalesFeatures(
                    age_filter_first=True, **params
                ).fit(train)
        if "spacetime_rbf" in modes or "spacetime_splines" in modes:
            coords = _coords(train)
            self.fill_coords = np.nanmedian(
                np.where(np.isfinite(coords), coords, np.nan), axis=0
            )
            self.fill_coords = np.where(
                np.isfinite(self.fill_coords), self.fill_coords, 0.0
            )
            coords = np.where(np.isfinite(coords), coords, self.fill_coords)
            if "spacetime_rbf" in modes:
                self.scaler = StandardScaler().fit(coords)
                self.rbf = RBFSampler(
                    gamma=0.45, n_components=48, random_state=42
                ).fit(self.scaler.transform(coords))
            if "spacetime_splines" in modes:
                self.splines = [
                    SplineTransformer(n_knots=5, degree=3, include_bias=False)
                    .fit(coords[:, [i]])
                    for i in range(3)
                ]
        return self

    def _modes(self):
        if self.variant == "combined":
            return ("market", "multiscale", "spacetime_splines",
                    "time_adjusted_comp")
        return () if self.variant == "baseline" else (self.variant,)

    def transform(self, homes, base_features):
        if self.variant == "baseline":
            return pd.DataFrame(index=homes.index)
        if self.column_names_ is None and self.market is None and not self.comps and (
            self.scaler is None and self.splines is None
        ):
            raise RuntimeError("Fit the add-on before transforming")
        modes = self._modes()
        frames = []
        market_frame = None
        if "market" in modes or "time_adjusted_comp" in modes:
            market_frame = self.market.transform(homes)
            if "market" in modes:
                frames.append(market_frame.add_prefix("extra_"))
        if "time_adjusted_comp" in modes:
            # A deliberately simple hypothesis, not a validated repeat-sales
            # index: recent 3-month local log-ppsf momentum applied to the
            # average age of the comps. Clipped to prevent huge extrapolation.
            momentum = market_frame["market_municipality_type_momentum_3m"]
            fallback = market_frame["market_city_momentum_3m"]
            momentum = momentum.where(momentum.notna(), fallback)
            age = pd.to_numeric(base_features["comp_mean_age_days"],
                                errors="coerce").clip(0, 730)
            adjustment = ((age / 90.0) * momentum).clip(-0.20, 0.20)
            frames.append(pd.DataFrame({
                "extra_comp_log_price_time_adjusted": (
                    base_features["comp_log_price"] + adjustment
                ),
                "extra_market_momentum_used": momentum,
                "extra_comp_time_adjustment": adjustment,
            }, index=homes.index))
        if "multiscale" in modes:
            for name, comp in self.comps.items():
                frames.append(comp.transform(homes).add_prefix(f"extra_{name}_"))
        if "spacetime_rbf" in modes or "spacetime_splines" in modes:
            xyz = _coords(homes)
            xyz = np.where(np.isfinite(xyz), xyz, self.fill_coords)
            if "spacetime_rbf" in modes:
                matrix = self.rbf.transform(self.scaler.transform(xyz))
                frames.append(pd.DataFrame(
                    matrix, index=homes.index,
                    columns=[f"extra_rbf_{i}" for i in range(matrix.shape[1])]
                ))
            if "spacetime_splines" in modes:
                pieces = [spl.transform(xyz[:, [i]]) for i, spl in enumerate(self.splines)]
                data = {}
                for k, part in enumerate(pieces):
                    for j in range(part.shape[1]):
                        data[f"extra_spline_{k}_{j}"] = part[:, j]
                # Explicit non-additive space-time interactions.
                for a, b, title in ((0, 1, "space"), (0, 2, "lat_time"),
                                    (1, 2, "lon_time")):
                    cross = np.einsum("ij,ik->ijk", pieces[a], pieces[b])
                    for j in range(cross.shape[1]):
                        for k in range(cross.shape[2]):
                            data[f"extra_tensor_{title}_{j}_{k}"] = cross[:, j, k]
                frames.append(pd.DataFrame(data, index=homes.index))
        result = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=homes.index)
        # Match XGBoost's data types, preserving missingness as NaN.
        return result.astype("float32")

    def fit_transform(self, train, base_features):
        return self.fit(train).transform(train, base_features)
