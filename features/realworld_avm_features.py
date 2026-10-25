"""Past-only AVM features inspired by adjusted comparable sales and geo tiling.

These are *approximations inspired by public methods*, not reproductions of
Zillow or Redfin's proprietary models. No neural networks or external data.

Every query is evaluated at its LISTING month. Eligible sale prices must
have date_sold strictly before the beginning of that month. The hedonic
model and its sale residuals are themselves fit strictly out of time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from features.spatiotemporal_features import EARTH_KM, _area

VARIANTS = (
    "rw_adjusted_comps", "rw_local_residuals", "rw_robust_comps",
    "rw_multiscale_tiles", "rw_combined",
)
COMP_VARIANTS = set(VARIANTS) - {"rw_multiscale_tiles"}
COMP_COLUMNS = (
    "rw_adjusted_log_price", "rw_adjusted_vs_simple", "rw_adjusted_spread",
    "rw_local_residual_mean", "rw_local_residual_mad", "rw_residual_count",
    "rw_comp_log_q25", "rw_comp_log_q50", "rw_comp_log_q75",
    "rw_comp_log_iqr", "rw_comp_count",
)


def _numeric(frame, key, default=np.nan):
    if key not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)


def _month(frame, key):
    date = pd.to_datetime(frame[key], errors="coerce")
    if date.isna().any():
        raise ValueError(f"Invalid {key} date")
    return (date.dt.year.to_numpy() * 12 + date.dt.month.to_numpy()).astype(int)


def _design(frame, kind_to_column, *, month):
    area = _area(frame["house_area"])
    n = len(frame)
    kinds = frame["house_type_name"].fillna("Unknown").astype(str).to_numpy()
    cat = np.zeros((n, len(kind_to_column)), dtype=float)
    for index, kind in enumerate(kinds):
        if kind in kind_to_column:
            cat[index, kind_to_column[kind]] = 1.0
    # Scale the interpretable hedonic covariates before Ridge estimation.
    features = np.column_stack([
        np.ones(n),
        np.log(np.where(area > 0, area, np.nan)) - 6.5,
        np.nan_to_num(_numeric(frame, "bedrooms"), nan=0) / 4,
        np.nan_to_num(_numeric(frame, "washrooms"), nan=0) / 3,
        np.nan_to_num(_numeric(frame, "parking_total"), nan=0) / 2,
        (np.asarray(month, dtype=float) - (2021 * 12 + 1)) / 60,
        cat,
    ])
    return features, area, kinds


def _weighted_quantile(values, weights, q):
    ordering = np.argsort(values, kind="stable")
    v, w = values[ordering], weights[ordering]
    return float(np.interp(q, np.cumsum(w) - 0.5 * w, v))


class HedonicNeighborhoodFeatures:
    """Adjusted comps, local out-of-time hedonic residuals and robust price quantiles."""

    def __init__(self, ridge=10.0, min_history=80, k=32, candidates=128):
        self.ridge = float(ridge)
        self.min_history = int(min_history)
        self.k = int(k)
        self.candidates = int(candidates)
        self.history_ = None
        self._coef_cache = {}
        self._tree_cache = {}

    def fit(self, sales):
        required = {"date_sold", "sold_price", "latitude", "longitude",
                    "house_area", "house_type_name"}
        missing = required - set(sales)
        if missing:
            raise KeyError(f"Missing hedonic history fields: {sorted(missing)}")
        sold = _month(sales, "date_sold")
        lat, lon = _numeric(sales, "latitude"), _numeric(sales, "longitude")
        price = _numeric(sales, "sold_price")
        kinds = sales["house_type_name"].fillna("Unknown").astype(str)
        self.types_ = {name: i for i, name in enumerate(sorted(kinds.unique()))}
        X, area, kinds = _design(sales, self.types_, month=sold)
        valid = (np.isfinite(lat) & np.isfinite(lon) & (abs(lat) <= 90)
                 & (abs(lon) <= 180) & np.isfinite(area) & (area > 0)
                 & np.isfinite(price) & (price > 0) & np.isfinite(X).all(axis=1))
        if not np.any(valid):
            raise ValueError("No eligible historical comparable sales")
        table = pd.DataFrame({
            "month": sold[valid], "lat": np.deg2rad(lat[valid]),
            "lon": np.deg2rad(lon[valid]), "area": np.log(area[valid]),
            "kind": kinds[valid], "log_price": np.log(price[valid]),
            "_row": np.flatnonzero(valid),
        }).sort_values("month", kind="stable").reset_index(drop=True)
        self.X_ = X[table["_row"].to_numpy(dtype=int)]
        self.history_ = table.drop(columns=["_row"])
        self.sold_months_ = sorted(table.month.unique())
        self.acc_xx_ = [np.zeros((self.X_.shape[1], self.X_.shape[1]))]
        self.acc_xy_ = [np.zeros(self.X_.shape[1])]
        self.acc_n_ = [0]
        self.oot_residual_ = np.full(len(table), np.nan)
        y = table.log_price.to_numpy()
        # For sale-month m, estimate beta using sale months STRICTLY earlier
        # than m. Never fit same-month or future sale prices into residuals.
        for m in self.sold_months_:
            ix = np.flatnonzero(table.month.to_numpy() == m)
            beta = self._coeff(self.acc_xx_[-1], self.acc_xy_[-1], self.acc_n_[-1])
            if beta is not None:
                self.oot_residual_[ix] = y[ix] - self.X_[ix] @ beta
            xx = self.X_[ix].T @ self.X_[ix]
            xy = self.X_[ix].T @ y[ix]
            self.acc_xx_.append(self.acc_xx_[-1] + xx)
            self.acc_xy_.append(self.acc_xy_[-1] + xy)
            self.acc_n_.append(self.acc_n_[-1] + len(ix))
        self._tree_cache = {}
        self._coef_cache = {}
        return self

    def _coeff(self, xx, xy, n):
        if n < self.min_history:
            return None
        penalty = np.eye(xx.shape[0]) * self.ridge
        penalty[0, 0] = 0
        return np.linalg.solve(xx + penalty, xy)

    def _month_state(self, month):
        import bisect
        i = bisect.bisect_left(self.sold_months_, int(month))
        if month not in self._coef_cache:
            self._coef_cache[month] = self._coeff(
                self.acc_xx_[i], self.acc_xy_[i], self.acc_n_[i])
        if month not in self._tree_cache:
            sale_month = self.history_.month.to_numpy()
            ids = np.flatnonzero((sale_month < month) & (sale_month >= month - 24))
            frame = self.history_.iloc[ids]
            tree = (BallTree(frame[["lat", "lon"]].to_numpy(), metric="haversine")
                    if len(frame) else None)
            self._tree_cache[month] = (ids, tree)
        return self._coef_cache[month], *self._tree_cache[month]

    def transform(self, homes):
        if self.history_ is None:
            raise RuntimeError("Call fit() before transform()")
        if len(homes) == 0:
            return pd.DataFrame(index=homes.index, columns=COMP_COLUMNS, dtype="float32")
        months = _month(homes, "date_listed")
        lat, lon = _numeric(homes, "latitude"), _numeric(homes, "longitude")
        X, area, types = _design(homes, self.types_, month=months)
        out = np.full((len(homes), len(COMP_COLUMNS)), np.nan)
        hist = self.history_
        hist_lp, hist_type = hist.log_price.to_numpy(), hist.kind.to_numpy()
        for month in np.unique(months):
            beta, ids, tree = self._month_state(int(month))
            if tree is None:
                continue
            for pos in np.flatnonzero(months == month):
                if (not np.isfinite(X[pos]).all() or not np.isfinite(lat[pos])
                    or not np.isfinite(lon[pos]) or abs(lat[pos]) > 90
                    or abs(lon[pos]) > 180 or area[pos] <= 0):
                    continue
                distances, nearest = tree.query(
                    np.deg2rad([[lat[pos], lon[pos]]]),
                    k=min(self.candidates, len(ids)),
                )
                jj = ids[nearest[0]]
                km = distances[0] * EARTH_KM
                ages = month - hist.month.to_numpy()[jj]
                same = hist_type[jj] == types[pos]
                delta_area = abs(hist.area.to_numpy()[jj] - np.log(area[pos]))
                weights = (np.exp(-km / 4.0) * np.exp(-np.log(2) * ages / 6.0)
                           * np.exp(-delta_area / .45)
                           * np.where(same, 1.0, .25))
                keep = np.argsort(-weights, kind="stable")[:self.k]
                jj, weights = jj[keep], weights[keep]
                if len(jj) < 2 or weights.sum() < 1e-12:
                    continue
                weights = weights / weights.sum()
                raw = hist_lp[jj]
                estimate = raw + ((X[pos] - self.X_[jj]) @ beta if beta is not None else 0)
                if beta is not None:
                    out[pos, :3] = (
                        float(weights @ estimate),
                        float(weights @ (estimate - raw)),
                        float(np.sqrt(weights @ (estimate - weights @ estimate) ** 2)),
                    )
                residual = self.oot_residual_[jj]
                okay = np.isfinite(residual)
                if np.any(okay):
                    w = weights[okay] / weights[okay].sum()
                    r = residual[okay]
                    center = float(w @ r)
                    out[pos, 3:6] = [center, float(w @ abs(r - center)), len(r)]
                out[pos, 6:9] = [_weighted_quantile(raw, weights, q)
                                  for q in (.25, .50, .75)]
                out[pos, 9] = out[pos, 8] - out[pos, 6]
                out[pos, 10] = len(jj)
        return pd.DataFrame(out, index=homes.index, columns=COMP_COLUMNS).astype("float32")


class LaggedTileMarketFeatures:
    """Zillow-inspired *non-neural* multi-resolution grid market statistics.

    Pure historical log-price-per-area aggregates, shrunk toward a prior-only
    city mean. Explicitly not Zillow's learned neural embeddings.
    """
    SCALES_KM = (1.0, 3.0, 10.0)

    def fit(self, sales):
        months = _month(sales, "date_sold")
        lat, lon = _numeric(sales, "latitude"), _numeric(sales, "longitude")
        area, price = _area(sales["house_area"]), _numeric(sales, "sold_price")
        good = (np.isfinite(lat) & np.isfinite(lon) & np.isfinite(area)
                & (area > 0) & np.isfinite(price) & (price > 0)
                & (abs(lat) <= 90) & (abs(lon) <= 180))
        self.history_ = pd.DataFrame({
            "month": months[good], "lat": lat[good], "lon": lon[good],
            "log_ppsf": np.log(price[good]) - np.log(area[good]),
        })
        self._grid_cache = {}
        return self

    @classmethod
    def _cells(cls, lat, lon, scale):
        # Fixed anchor prevents tiles moving when new records arrive.
        return list(zip(np.floor((lat - 43.0) * 111.0 / scale).astype(int),
                        np.floor((lon + 79.0) * 80.0 / scale).astype(int)))

    def transform(self, homes):
        if not hasattr(self, "history_"):
            raise RuntimeError("Call fit() before transform()")
        months = _month(homes, "date_listed")
        lat, lon = _numeric(homes, "latitude"), _numeric(homes, "longitude")
        names = [f"rw_tile_{int(k)}km_log_ppsf" for k in self.SCALES_KM]
        names += [f"rw_tile_{int(k)}km_count" for k in self.SCALES_KM]
        out = np.full((len(homes), len(names)), np.nan)
        for month in np.unique(months):
            if month not in self._grid_cache:
                prev = self.history_.loc[
                    (self.history_.month < month)
                    & (self.history_.month >= month - 12)
                ]
                city = prev.log_ppsf.mean() if len(prev) else np.nan
                grids = []
                for scale in self.SCALES_KM:
                    if prev.empty:
                        grids.append({})
                        continue
                    cell = self._cells(prev.lat.to_numpy(), prev.lon.to_numpy(), scale)
                    frame = prev.assign(_cell=cell)
                    means = frame.groupby("_cell").log_ppsf.agg(["sum", "count"])
                    grids.append({key: (float(row["sum"]), int(row["count"]))
                                  for key, row in means.iterrows()})
                self._grid_cache[month] = (city, grids)
            city, grids = self._grid_cache[month]
            positions = np.flatnonzero(months == month)
            valid = (np.isfinite(lat[positions]) & np.isfinite(lon[positions])
                     & (abs(lat[positions]) <= 90) & (abs(lon[positions]) <= 180))
            for pos in positions[valid]:
                for j, scale in enumerate(self.SCALES_KM):
                    cell = self._cells(np.array([lat[pos]]), np.array([lon[pos]]), scale)[0]
                    total, count = grids[j].get(cell, (0.0, 0))
                    if np.isfinite(city):
                        out[pos, j] = (total + 25 * city) / (count + 25)
                    out[pos, j + len(self.SCALES_KM)] = count
        return pd.DataFrame(out, index=homes.index, columns=names).astype("float32")


class RealWorldAVMAddon:
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown real-world AVM feature: {variant}")
        self.variant = variant

    def fit(self, sales):
        if self.variant in COMP_VARIANTS:
            self.comps = HedonicNeighborhoodFeatures().fit(sales)
        if self.variant in ("rw_multiscale_tiles", "rw_combined"):
            self.tiles = LaggedTileMarketFeatures().fit(sales)
        return self

    def transform(self, homes, base_features=None):
        chunks = []
        if hasattr(self, "comps"):
            frame = self.comps.transform(homes)
            columns = {
                "rw_adjusted_comps": COMP_COLUMNS[:3] + (COMP_COLUMNS[10],),
                "rw_local_residuals": COMP_COLUMNS[3:6],
                "rw_robust_comps": COMP_COLUMNS[6:],
                "rw_combined": COMP_COLUMNS,
            }[self.variant]
            chunks.append(frame[list(columns)])
        if hasattr(self, "tiles"):
            chunks.append(self.tiles.transform(homes))
        return pd.concat(chunks, axis=1).astype("float32")
