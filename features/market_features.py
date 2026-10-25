"""Leakage-safe, historical market features for housing-price prediction.

Features describe completed sales from *months strictly before* a home's
``date_listed`` month.  The transformer is fitted only on a model's available
historical training sales.  For dates beyond its latest historical month,
features freeze at the latest available snapshot (no fabricated future sales).

No listing-price data is used; compatible with the off-market track.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketLevelFeatures:
    """Lagged price-per-square-foot, momentum, and volume by market segment.

    Hierarchy: citywide, municipality, house type, municipality x house type.
    Segment averages shrink toward the citywide average with ``shrinkage``
    pseudo-observations, reducing noise in sparse local markets.

    ``fit(sales)`` requires ``sold_price``, ``date_sold``, ``house_area``,
    ``municipality`` and ``house_type_name``. ``transform(homes)`` requires only
    ``date_listed``, ``municipality``, ``house_type_name``.
    """

    _LEVELS = (
        ("city", ()),
        ("municipality", ("municipality",)),
        ("property_type", ("house_type_name",)),
        ("municipality_type", ("municipality", "house_type_name")),
    )

    def __init__(self, shrinkage: float = 20.0):
        if shrinkage < 0:
            raise ValueError("shrinkage must be nonnegative")
        self.shrinkage = float(shrinkage)
        self._tables = {}
        self._earliest_month = None
        self._latest_snapshot = None
        self.columns_ = None

    @staticmethod
    def _square_feet(area: pd.Series) -> pd.Series:
        pieces = area.astype("string").str.strip().str.split("-", n=1, expand=True)
        low = pd.to_numeric(pieces[0], errors="coerce")
        high = (
            pd.to_numeric(pieces[1], errors="coerce").fillna(low)
            if 1 in pieces.columns else low
        )
        return (low + high) / 2.0

    @staticmethod
    def _month_index(dates: pd.Series) -> np.ndarray:
        parsed = pd.to_datetime(dates, errors="coerce")
        if parsed.isna().any():
            raise ValueError("MarketLevelFeatures requires valid dates")
        return parsed.dt.to_period("M").astype("int64").to_numpy()

    @staticmethod
    def _keys(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            name: df[name].astype("string").fillna("Unknown").astype(str).to_numpy()
            for name in ("municipality", "house_type_name")
        }, index=df.index)

    @staticmethod
    def _rolling_summaries(month_numbers, values, start, end):
        """Return prior 3m, prior 6m, and preceding 3m sums at every month.

        The snapshot for month m is constructed using month numbers < m only.
        """
        size = end - start + 1
        month_totals = np.bincount(month_numbers - start, weights=values, minlength=size)
        cumulative = np.r_[0.0, np.cumsum(month_totals)]
        i = np.arange(size)
        i3 = np.maximum(i - 3, 0)
        i6 = np.maximum(i - 6, 0)
        return (
            cumulative[i] - cumulative[i3],
            cumulative[i] - cumulative[i6],
            cumulative[i3] - cumulative[i6],
        )

    def fit(self, sales: pd.DataFrame):
        required = {"date_sold", "sold_price", "house_area", "municipality", "house_type_name"}
        missing = required.difference(sales.columns)
        if missing:
            raise KeyError(f"Historical sales missing columns: {sorted(missing)}")

        area = self._square_feet(sales["house_area"]).to_numpy(dtype=float)
        price = pd.to_numeric(sales["sold_price"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(area) & (area > 0) & np.isfinite(price) & (price > 0)
        if not valid.any():
            raise ValueError("No usable past sales with positive prices and floor areas")

        data = self._keys(sales).loc[valid].reset_index(drop=True)
        data["_month"] = self._month_index(sales.loc[valid, "date_sold"])
        data["_value"] = np.log(price[valid] / area[valid])
        start = int(data["_month"].min())
        # Add a snapshot after the final observed month for future predictions.
        end = int(data["_month"].max()) + 1
        self._earliest_month = start
        self._latest_snapshot = end
        self._tables = {}

        for name, group_cols in self._LEVELS:
            parts = []
            if group_cols:
                grouped = data.groupby(list(group_cols), sort=False, dropna=False)
            else:
                grouped = [((), data)]
            for key, sub in grouped:
                key = key if isinstance(key, tuple) else (key,)
                months = sub["_month"].to_numpy(dtype=int)
                v = sub["_value"].to_numpy(dtype=float)
                totals = self._rolling_summaries(months, v, start, end)
                counts = self._rolling_summaries(months, np.ones(len(v)), start, end)
                part = pd.DataFrame({
                    "_month": np.arange(start, end + 1),
                    "sum3": totals[0], "sum6": totals[1], "sum_prev3": totals[2],
                    "n3": counts[0], "n6": counts[1], "n_prev3": counts[2],
                })
                for field, value in zip(group_cols, key):
                    part[field] = value
                parts.append(part)
            self._tables[name] = pd.concat(parts, ignore_index=True)

        self.columns_ = []
        for name, _ in self._LEVELS:
            self.columns_.extend((
                f"market_{name}_log_ppsf_3m",
                f"market_{name}_log_ppsf_6m",
                f"market_{name}_momentum_3m",
                f"market_{name}_log1p_sales_3m",
            ))
        return self

    @staticmethod
    def _mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("Call fit() before transform()")
        if homes.empty:
            return pd.DataFrame(index=homes.index, columns=self.columns_, dtype=float)

        query = self._keys(homes).reset_index(drop=True)
        wanted_month = self._month_index(homes["date_listed"])
        # Beyond the historical window, use the last known snapshot; no future outcomes.
        query["_month"] = np.minimum(wanted_month, self._latest_snapshot)
        result = pd.DataFrame(index=np.arange(len(homes)))

        def lookup(level_name, cols):
            return query.merge(self._tables[level_name],
                               how="left", on=[*cols, "_month"], sort=False)

        global_history = lookup("city", ())
        global_3 = self._mean(global_history["sum3"].to_numpy(), global_history["n3"].to_numpy())
        global_6 = self._mean(global_history["sum6"].to_numpy(), global_history["n6"].to_numpy())
        global_prev = self._mean(global_history["sum_prev3"].to_numpy(), global_history["n_prev3"].to_numpy())

        for level_name, group_cols in self._LEVELS:
            h = global_history if not group_cols else lookup(level_name, group_cols)
            n3 = h["n3"].fillna(0).to_numpy(dtype=float)
            n6 = h["n6"].fillna(0).to_numpy(dtype=float)
            nprev = h["n_prev3"].fillna(0).to_numpy(dtype=float)
            s3 = h["sum3"].fillna(0).to_numpy(dtype=float)
            s6 = h["sum6"].fillna(0).to_numpy(dtype=float)
            sprev = h["sum_prev3"].fillna(0).to_numpy(dtype=float)

            if not group_cols:
                mean3, mean6, meanprev = global_3, global_6, global_prev
            else:
                k = self.shrinkage
                # For sparse segments, use global prices from the *same past window*.
                def smooth(sum_, n_, global_):
                    if k == 0:
                        return self._mean(sum_, n_)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        fallback = self._mean(sum_, n_)
                        smoothed = (sum_ + k * global_) / (n_ + k)
                        return np.where(np.isfinite(global_), smoothed, fallback)
                mean3 = smooth(s3, n3, global_3)
                mean6 = smooth(s6, n6, global_6)
                meanprev = smooth(sprev, nprev, global_prev)

            result[f"market_{level_name}_log_ppsf_3m"] = mean3
            result[f"market_{level_name}_log_ppsf_6m"] = mean6
            result[f"market_{level_name}_momentum_3m"] = mean3 - meanprev
            result[f"market_{level_name}_log1p_sales_3m"] = np.log1p(n3)

        result.index = homes.index
        return result[self.columns_]

    def fit_transform(self, sales: pd.DataFrame) -> pd.DataFrame:
        return self.fit(sales).transform(sales)


class MarketAugmentedFeatures:
    """Wrap any fitted feature provider with a fit_transform / transform API.

    The base provider should return a pandas DataFrame with the same index as
    the input homes. Market features are appended after the base features.
    """

    def __init__(self, base_provider, shrinkage: float = 20.0):
        self.base_provider = base_provider
        self.market = MarketLevelFeatures(shrinkage=shrinkage)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.market.fit(train)
        base_X = self.base_provider.fit_transform(train)
        return pd.concat([base_X, self.market.transform(train)], axis=1)

    def transform(self, homes: pd.DataFrame) -> pd.DataFrame:
        base_X = self.base_provider.transform(homes)
        return pd.concat([base_X, self.market.transform(homes)], axis=1)

    @property
    def categorical_features(self):
        return self.base_provider.categorical_features
