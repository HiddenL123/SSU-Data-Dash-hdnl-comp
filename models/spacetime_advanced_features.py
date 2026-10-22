"""Experimental feature families for historical off-market XGBoost screening.

Every sold-price-derived feature uses only completed sales from strictly before
its query listing month. Fit uses only the outer training fold. No list price,
post-sale attributes, or validation outcomes are needed by transform().

The implementations are deliberately independent of the existing production
ComparableSalesFeatures and are activated only through SpaceTimeAddon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .spatiotemporal_features import ComparableSalesFeatures, EARTH_KM, _area

ADVANCED_FEATURE_VARIANTS = (
    "adaptive_comps", "hedonic_index", "geo_context",
    "property_spacetime", "kalman_market", "advanced_combined",
)


def _number(frame, name, default=np.nan):
    if name not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _string(frame, name):
    if name not in frame:
        return np.full(len(frame), "Unknown", dtype=object)
    return frame[name].fillna("Unknown").astype(str).to_numpy()


def _month(frame, column):
    values = pd.to_datetime(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Expected valid {column} dates")
    return values.dt.to_period("M").astype("int64").to_numpy(dtype=int)


def _haversine(lat, lon, center_lat, center_lon):
    a, b, c, d = map(np.deg2rad, (lat, lon, center_lat, center_lon))
    h = np.sin((a - c) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((b - d) / 2) ** 2
    return EARTH_KM * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


class AdaptiveComparableFeatures:
    """Select a local, medium, or regional comp window using *past* density.

    Each window is independently age-eligible; fallback to wider windows is
    based on the local effective sample size and eligible comp count, not price.
    """

    def fit(self, sales):
        settings = (
            dict(candidates=64, neighbors=16, radius_km=0.75,
                 half_life_days=90, max_age_days=365),
            dict(candidates=128, neighbors=32, radius_km=3.0,
                 half_life_days=180, max_age_days=730),
            dict(candidates=128, neighbors=48, radius_km=9.0,
                 half_life_days=365, max_age_days=730),
        )
        self.windows_ = [ComparableSalesFeatures(age_filter_first=True, **cfg).fit(sales)
                         for cfg in settings]
        return self

    def transform(self, homes):
        frames = [provider.transform(homes) for provider in self.windows_]
        local, middle, broad = frames
        local_ok = (local.comp_count.fillna(0).to_numpy() >= 8) & (
            local.comp_effective_n.fillna(0).to_numpy() >= 4)
        middle_ok = (middle.comp_count.fillna(0).to_numpy() >= 8) & (
            middle.comp_effective_n.fillna(0).to_numpy() >= 4)
        choice = np.where(local_ok, 0, np.where(middle_ok, 1, 2))
        out = pd.DataFrame(index=homes.index)
        out["adaptive_window_km"] = np.array([0.75, 3.0, 9.0])[choice]
        for key in ("comp_log_price", "comp_log_ppsf", "comp_count",
                    "comp_effective_n", "comp_nearest_km", "comp_mean_age_days",
                    "comp_same_type_share", "comp_price_std"):
            values = np.stack([frame[key].to_numpy(dtype=float) for frame in frames])
            out["adaptive_" + key.removeprefix("comp_")] = values[choice, np.arange(len(homes))]
        out["adaptive_local_count"] = local.comp_count.to_numpy(dtype=float)
        out["adaptive_mid_count"] = middle.comp_count.to_numpy(dtype=float)
        return out.astype("float32")


class GeographicContextFeatures:
    """Historical sale density and target-independent geographical centroids.

    Centroids use training-fold locations only, not prices. The nearest *other*
    municipality centroid is a proximity proxy, NOT an actual legal boundary.
    """

    def fit(self, sales):
        lat, lon = _number(sales, "latitude"), _number(sales, "longitude")
        months = _month(sales, "date_sold")
        good = np.isfinite(lat) & np.isfinite(lon) & (abs(lat) <= 90) & (abs(lon) <= 180)
        if not good.any():
            raise ValueError("No geocoded training sales")
        municipalities = _string(sales, "municipality")
        self.city_ = (float(np.median(lat[good])), float(np.median(lon[good])))
        centers = pd.DataFrame({"municipality": municipalities[good],
                                "lat": lat[good], "lon": lon[good]})
        self.centers_ = centers.groupby("municipality")[["lat", "lon"]].median()
        self.history_ = pd.DataFrame({"month": months[good], "lat": lat[good], "lon": lon[good]})
        return self

    def transform(self, homes):
        lat, lon = _number(homes, "latitude"), _number(homes, "longitude")
        months = _month(homes, "date_listed")
        names = _string(homes, "municipality")
        output = pd.DataFrame(index=homes.index)
        output["geo_city_centroid_km"] = _haversine(lat, lon, *self.city_)
        centers = self.centers_.reindex(names)
        output["geo_municipality_centroid_km"] = _haversine(
            lat, lon, centers.lat.to_numpy(), centers.lon.to_numpy())
        # Distance to other municipality centroids; useful as a boundary PROXY.
        other = np.full(len(homes), np.nan)
        for name in np.unique(names):
            ids = np.flatnonzero(names == name)
            possible = self.centers_.drop(index=name, errors="ignore")
            if len(possible):
                km = np.column_stack([
                    _haversine(lat[ids], lon[ids], c.lat, c.lon)
                    for c in possible.itertuples()
                ])
                other[ids] = np.nanmin(km, axis=1)
        output["geo_nearest_other_municipality_centroid_km"] = other
        dense_1 = np.zeros(len(homes), dtype=float)
        dense_5 = np.zeros(len(homes), dtype=float)
        for month in np.unique(months):
            loc = np.flatnonzero(months == month)
            hist = self.history_.loc[(self.history_.month < month) &
                                     (self.history_.month >= month - 12)]
            valid = np.isfinite(lat[loc]) & np.isfinite(lon[loc]) & (abs(lat[loc]) <= 90) & (abs(lon[loc]) <= 180)
            if hist.empty or not valid.any():
                continue
            tree = BallTree(np.deg2rad(hist[["lat", "lon"]].to_numpy()), metric="haversine")
            coords = np.deg2rad(np.column_stack([lat[loc[valid]], lon[loc[valid]]]))
            dense_1[loc[valid]] = tree.query_radius(coords, r=1.0 / EARTH_KM, count_only=True)
            dense_5[loc[valid]] = tree.query_radius(coords, r=5.0 / EARTH_KM, count_only=True)
        output["geo_sales_count_1km_12m"] = dense_1
        output["geo_sales_count_5km_12m"] = dense_5
        return output.astype("float32")


class PropertySpaceTimeFeatures:
    """Small interpretable set of interactions not explicit in base features."""

    def fit(self, sales):
        lat, lon = _number(sales, "latitude"), _number(sales, "longitude")
        area = _area(sales["house_area"])
        months = _month(sales, "date_listed")
        self.anchor_ = (float(np.nanmedian(lat)), float(np.nanmedian(lon)),
                        float(np.nanmedian(np.log(np.where(area > 0, area, np.nan)))),
                        float(np.median(months)))
        return self

    def transform(self, homes):
        lat, lon = _number(homes, "latitude"), _number(homes, "longitude")
        area = _area(homes["house_area"])
        months = _month(homes, "date_listed")
        la, lo, ar, tm = self.anchor_
        north, east = (lat - la) * 111, (lon - lo) * 80
        time = (months - tm) / 12
        logarea = np.log(np.where(area > 0, area, np.nan)) - ar
        beds = _number(homes, "bedrooms")
        baths = _number(homes, "washrooms")
        types = pd.Series(_string(homes, "house_type_name"), index=homes.index)
        condo = types.str.contains("Condo|Co-Op", case=False, regex=True).to_numpy(dtype=float)
        detached = types.str.contains("Detached", case=False, regex=False).to_numpy(dtype=float)
        return pd.DataFrame({
            "interaction_area_north": logarea * north,
            "interaction_area_east": logarea * east,
            "interaction_area_time": logarea * time,
            "interaction_bedrooms_north": beds * north,
            "interaction_bathrooms_east": baths * east,
            "interaction_condo_north": condo * north,
            "interaction_condo_time": condo * time,
            "interaction_detached_east": detached * east,
            "interaction_detached_time": detached * time,
        }, index=homes.index).astype("float32")


class HedonicMarketFeatures:
    """Lagged local price index constructed from expanding OOT hedonic residuals.

    A small, regularized citywide hedonic model is fitted on completed sales
    *before* each sale month. Residuals for sales in that month are then added
    to historical 3/6-month aggregates. This prevents in-sample target encoding
    and leakage from future sales into older training rows.
    """

    def __init__(self, shrinkage=20, min_history=100, ridge_alpha=10):
        self.shrinkage = float(shrinkage)
        self.min_history = int(min_history)
        self.ridge_alpha = float(ridge_alpha)

    def fit(self, sales):
        month = _month(sales, "date_sold")
        area = _area(sales["house_area"])
        price = _number(sales, "sold_price")
        types = _string(sales, "house_type_name")
        muni = _string(sales, "municipality")
        okay = np.isfinite(area) & (area > 0) & np.isfinite(price) & (price > 0)
        if not okay.any():
            raise ValueError("No valid hedonic training sales")
        categories = np.unique(types[okay])
        labels = (types[:, None] == categories[None, :]).astype(float)
        design = np.column_stack([
            np.ones(len(sales)),
            np.log(np.where(area > 0, area, np.nan)) - 6.5,
            np.nan_to_num(_number(sales, "bedrooms"), nan=0) / 4,
            np.nan_to_num(_number(sales, "washrooms"), nan=0) / 3,
            np.nan_to_num(_number(sales, "parking_total"), nan=0) / 2,
            labels,
        ])
        okay &= np.isfinite(design).all(axis=1)
        resid = np.full(len(sales), np.nan)
        x, y = design[okay], np.log(price[okay])
        sale_months = month[okay]
        cum_xx = np.zeros((x.shape[1], x.shape[1]))
        cum_xy = np.zeros(x.shape[1])
        n_prior = 0
        valid_positions = np.flatnonzero(okay)
        penalty = np.eye(x.shape[1]) * self.ridge_alpha
        penalty[0, 0] = 0
        for m in np.unique(sale_months):
            mask = sale_months == m
            if n_prior >= self.min_history:
                coef = np.linalg.solve(cum_xx + penalty, cum_xy)
                resid[valid_positions[mask]] = y[mask] - x[mask] @ coef
            now = x[mask]
            cum_xx += now.T @ now
            cum_xy += now.T @ y[mask]
            n_prior += len(now)
        self.history_ = pd.DataFrame({"month": month, "municipality": muni,
                                      "type": types, "residual": resid}).dropna(subset=["residual"])
        return self

    def transform(self, homes):
        months = _month(homes, "date_listed")
        muni = _string(homes, "municipality")
        types = _string(homes, "house_type_name")
        out = np.full((len(homes), 7), np.nan)
        out[:, 6] = 0

        def aggregates(df, columns):
            if df.empty:
                return {}
            grouped = df.groupby(columns, dropna=False).residual.agg(["sum", "count"])
            return {key if isinstance(key, tuple) else (key,):
                    (float(row["sum"]), int(row["count"]))
                    for key, row in grouped.iterrows()}

        for month in np.unique(months):
            ids = np.flatnonzero(months == month)
            previous = self.history_[(self.history_.month < month) &
                                     (self.history_.month >= month - 6)]
            if previous.empty:
                continue
            recent = previous[previous.month >= month - 3]
            if recent.empty:
                continue
            city3 = float(recent.residual.mean())
            city6 = float(previous.residual.mean())
            m3 = aggregates(recent, ["municipality"])
            t3 = aggregates(recent, ["type"])
            mt3 = aggregates(recent, ["municipality", "type"])
            mt6 = aggregates(previous, ["municipality", "type"])

            def smooth(record, prior):
                total, count = record
                return (total + self.shrinkage * prior) / (count + self.shrinkage) if (count + self.shrinkage) else np.nan

            for idx in ids:
                a, b = muni[idx], types[idx]
                mval = smooth(m3.get((a,), (0, 0)), city3)
                tval = smooth(t3.get((b,), (0, 0)), city3)
                mtval = smooth(mt3.get((a, b), (0, 0)), city3)
                mtval6 = smooth(mt6.get((a, b), (0, 0)), city6)
                out[idx] = [city3, mval, tval, mtval, mtval6,
                            mtval - city3, mt3.get((a, b), (0, 0))[1]]
        return pd.DataFrame(out, columns=[
            "hedonic_city_residual_3m", "hedonic_municipality_residual_3m",
            "hedonic_type_residual_3m", "hedonic_segment_residual_3m",
            "hedonic_segment_residual_6m", "hedonic_segment_vs_city_3m",
            "hedonic_segment_sales_3m",
        ], index=homes.index).astype("float32")


class KalmanMarketFeatures:
    """Causal local-level/trend filter of monthly log-price-per-square-foot.

    States stored for month M are *prior* states, before ingesting M sales.
    Missing local histories fall back to the city; no backward smoothing.
    """
    F = np.array([[1., 1.], [0., 1.]])
    Q = np.diag([0.0025, 0.0001])

    def fit(self, sales):
        months = _month(sales, "date_sold")
        area = _area(sales["house_area"])
        price = _number(sales, "sold_price")
        valid = np.isfinite(area) & (area > 0) & np.isfinite(price) & (price > 0)
        if not valid.any():
            raise ValueError("No valid market-state sales")
        df = pd.DataFrame({"month": months[valid],
                           "municipality": _string(sales, "municipality")[valid],
                           "type": _string(sales, "house_type_name")[valid],
                           "value": np.log(price[valid] / area[valid])})
        observations = {}
        for name, fields in (("city", ["month"]),
                             ("muni", ["month", "municipality"]),
                             ("segment", ["month", "municipality", "type"])):
            for key, row in df.groupby(fields).value.agg(["mean", "count"]).iterrows():
                tup = key if isinstance(key, tuple) else (key,)
                observations[(name,) + tup] = (float(row["mean"]), int(row["count"]))
        states = {}
        self.snapshots_ = {}
        for month in range(int(months[valid].min()), int(months[valid].max()) + 2):
            snapshot = {}
            for key, (x, p) in list(states.items()):
                xp = self.F @ x
                pp = self.F @ p @ self.F.T + self.Q
                states[key] = (xp, pp)
                snapshot[key] = (float(xp[0]), float(xp[1]), float(np.sqrt(pp[0, 0])))
            self.snapshots_[month] = snapshot
            for keyval, (obs, count) in observations.items():
                kind, m, *ident = keyval
                if m != month:
                    continue
                key = (kind, *ident)
                if key not in states:
                    states[key] = (np.array([obs, 0.]), np.diag([0.1, 0.01]))
                else:
                    x, p = states[key]
                    noise = 0.12 / max(count, 1) + 0.0005
                    gain = p[:, 0] / (p[0, 0] + noise)
                    x = x + gain * (obs - x[0])
                    p = p - np.outer(gain, p[0, :])
                    states[key] = (x, p)
        self.latest_ = max(self.snapshots_)
        return self

    def transform(self, homes):
        months = _month(homes, "date_listed")
        muni = _string(homes, "municipality")
        types = _string(homes, "house_type_name")
        arr = np.full((len(homes), 9), np.nan)
        for i, m in enumerate(months):
            snapshot = self.snapshots_.get(min(int(m), self.latest_), {})
            city = snapshot.get(("city",))
            if city is None:
                continue
            local = snapshot.get(("muni", muni[i]), city)
            segment = snapshot.get(("segment", muni[i], types[i]), local)
            arr[i] = [*city, *local[:2], *segment,
                      segment[0] - city[0]]
        return pd.DataFrame(arr, index=homes.index, columns=[
            "kalman_city_level", "kalman_city_trend", "kalman_city_uncertainty",
            "kalman_muni_level", "kalman_muni_trend",
            "kalman_segment_level", "kalman_segment_trend",
            "kalman_segment_uncertainty", "kalman_segment_city_gap",
        ]).astype("float32")


class AdvancedSpaceTimeAddon:
    """Uniform fit/transform interface for independent shotgun feature families."""
    FAMILIES = {
        "adaptive_comps": AdaptiveComparableFeatures,
        "hedonic_index": HedonicMarketFeatures,
        "geo_context": GeographicContextFeatures,
        "property_spacetime": PropertySpaceTimeFeatures,
        "kalman_market": KalmanMarketFeatures,
    }

    def __init__(self, variant):
        if variant not in ADVANCED_FEATURE_VARIANTS:
            raise ValueError(f"Unsupported advanced variant: {variant}")
        self.variant = variant
        names = tuple(self.FAMILIES) if variant == "advanced_combined" else (variant,)
        self.providers = [(name, self.FAMILIES[name]()) for name in names]
        self.fitted_ = False

    def fit(self, sales):
        for _, provider in self.providers:
            provider.fit(sales)
        self.fitted_ = True
        return self

    def transform(self, homes, base_features=None):
        if not self.fitted_:
            raise RuntimeError("Call fit before transform")
        frames = []
        for name, provider in self.providers:
            frame = provider.transform(homes)
            if not frame.index.equals(homes.index):
                raise ValueError(f"{name}: feature index mismatch")
            frames.append(frame.add_prefix("extra_" + name + "_"))
        combined = pd.concat(frames, axis=1)
        if combined.columns.duplicated().any():
            raise ValueError("Duplicate feature columns")
        return combined.astype("float32")