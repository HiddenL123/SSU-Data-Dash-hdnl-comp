"""Shared legacy Ridge correction and chronological fitting strategy."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .gb_features import square_feet


def _checked_log_prices(train):
    prices = train["sold_price"].to_numpy(dtype=float)
    if len(prices) == 0 or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Training needs positive finite sold_price values")
    return np.log(prices)



class ListingAdjustment:
    """Off-market boosting+ARIMA followed by a listing-price Ridge correction.

    The Ridge correction is trained from past-only, chronological predictions
    of stage 1. Listing price is never fed to the off-market stage.
    """

    def __init__(self, correction_alpha=50.0, n_folds=3):
        self.base_model = self._new_base_model()
        self.correction_model = Ridge(alpha=correction_alpha)
        self.correction_scaler = StandardScaler()
        self.n_folds = n_folds
        self.house_types = None
        self.correction_columns = None
        self.correction_medians = None
        self.oof_count_ = 0
        self._fitted = False

    def _new_base_model(self):
        raise NotImplementedError("Subclasses must construct a fresh off-market model")

    @staticmethod
    def _log_list_price(homes):
        values = homes["list_price"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("List-aware prediction needs positive list_price")
        return np.log(values)

    def _correction_features(self, homes, base_log):
        base_log = np.asarray(base_log, dtype=float)
        discrepancy = self._log_list_price(homes) - base_log
        area = square_feet(homes["house_area"])
        listed = pd.to_datetime(homes["date_listed"])
        months = (listed.dt.year - 2021) * 12 + listed.dt.month - 1

        Z = pd.DataFrame(index=homes.index)
        Z["list_discrepancy"] = discrepancy
        Z["off_market_log_price"] = base_log
        Z["log_area"] = np.log1p(area.to_numpy(dtype=float))
        for col, name in [
            ("bedrooms", "bedrooms"), ("washrooms", "washrooms"),
            ("parking_total", "parking"), ("longitude", "longitude"),
            ("latitude", "latitude"),
        ]:
            Z[name] = pd.to_numeric(homes[col], errors="coerce").to_numpy()
        Z["months_since_2021"] = months.to_numpy(dtype=float)
        for house_type in self.house_types:
            indicator = (homes["house_type_name"] == house_type).to_numpy(dtype=float)
            name = f"type_{house_type}"
            Z[name] = indicator
            Z[f"discrepancy_x_{name}"] = discrepancy * indicator
        return Z.replace([np.inf, -np.inf], np.nan)

    def fit(self, train):
        log_prices = _checked_log_prices(train)
        self._log_list_price(train)
        if self.n_folds < 1:
            raise ValueError("n_folds must be at least one")
        listed = pd.to_datetime(train["date_listed"])
        sold = pd.to_datetime(train["date_sold"])
        if listed.isna().any() or sold.isna().any():
            raise ValueError("Stage-2 training needs valid date_listed and date_sold")
        self.house_types = sorted(train["house_type_name"].dropna().unique())

        # Exactly as with the historical two-stage model: every held-out
        # listing is evaluated with a base model trained on sales completed
        # before the corresponding cutoff.
        sorted_listed = listed.sort_values().to_numpy()
        cutoffs = sorted({
            pd.Timestamp(sorted_listed[min(int(q * len(train)), len(train) - 1)]).normalize()
            for q in np.linspace(0.55, 0.80, self.n_folds)
        })
        parts, targets = [], []
        for i, cutoff in enumerate(cutoffs):
            end = cutoffs[i + 1] if i + 1 < len(cutoffs) else None
            past = train.loc[sold < cutoff]
            future_mask = listed >= cutoff
            if end is not None:
                future_mask &= listed < end
            held_out = train.loc[future_mask]
            if len(past) < 100 or held_out.empty:
                continue
            earlier = self._new_base_model().fit(past)
            earlier_log = earlier.predict_log(held_out)
            parts.append(self._correction_features(held_out, earlier_log))
            targets.append(np.log(held_out["sold_price"].to_numpy(dtype=float)) - earlier_log)
        if not parts:
            raise ValueError("Not enough history for chronological list-aware fitting")

        Z = pd.concat(parts, axis=0)
        residuals = np.concatenate(targets)
        self.oof_count_ = len(residuals)
        self.correction_columns = Z.columns.tolist()
        self.correction_medians = Z.median().fillna(0.0)
        Z = Z.fillna(self.correction_medians)
        self.correction_model.fit(self.correction_scaler.fit_transform(Z), residuals)
        self.base_model.fit(train)
        self._fitted = True
        return self

    def predict_log(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        if homes.empty:
            return np.array([], dtype=float)
        base_log = self.base_model.predict_log(homes)
        Z = self._correction_features(homes, base_log)
        Z = Z.reindex(columns=self.correction_columns).fillna(self.correction_medians)
        return base_log + self.correction_model.predict(self.correction_scaler.transform(Z))

    def predict(self, homes):
        return np.exp(self.predict_log(homes))

