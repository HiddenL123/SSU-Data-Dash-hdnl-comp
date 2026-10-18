"""CatBoost / LightGBM / XGBoost housing-price models with ARIMA adjustment.

Requires boosting_features.py on the Python import path.

Public interface (compatible with run.py):
    model = make_model('off_market', algorithm='catboost')
    model.fit(train)
    predicted_prices = model.predict(unseen_homes)

For list-aware predictions, stage 2 is a Ridge correction trained on
chronologically out-of-time stage-1 predictions, not directly on in-sample
predictions. Stage 1 never uses list_price.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from models.gb_features import BoostingFeatures, square_feet
from models.dummy_model import DummyModel


DEFAULT_PARAMS = {
    "catboost": {
        "iterations": 900,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    },
    "lightgbm": {
        "n_estimators": 900,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 40,
        "reg_lambda": 5,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 900,
        "learning_rate": 0.04,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 5,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42,
        "n_jobs": -1,
    },
}


def _checked_log_prices(train):
    prices = train["sold_price"].to_numpy(dtype=float)
    if len(prices) == 0 or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Training needs positive finite sold_price values")
    return np.log(prices)


class GBPropertyModel:
    """One of three boosting regressors, all predicting log(sold_price)."""

    def __init__(self, algorithm="catboost", model_params=None):
        if algorithm not in DEFAULT_PARAMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.algorithm = algorithm
        self.params = {**DEFAULT_PARAMS[algorithm], **(model_params or {})}
        self.model = self._new_estimator()

    def _new_estimator(self):
        if self.algorithm == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(**self.params)
        if self.algorithm == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**self.params)
        from xgboost import XGBRegressor
        return XGBRegressor(**self.params)

    def fit(self, X, log_prices, categorical_features):
        if self.algorithm == "catboost":
            self.model.fit(X, log_prices, cat_features=categorical_features)
        else:
            self.model.fit(X, log_prices)
        return self

    def predict_log(self, X):
        return np.asarray(self.model.predict(X), dtype=float).reshape(-1)

    def feature_importance(self, columns):
        return pd.Series(
            self.model.feature_importances_, index=columns, dtype=float
        ).sort_values(ascending=False)


class ArimaResidualModel:
    """ARIMA on monthly medians of remaining log-price residuals."""

    def __init__(self, order=(1, 1, 0)):
        self.order = order
        self.result = None
        self.first_month = None
        self.last_month = None

    @staticmethod
    def _months(dates):
        parsed = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
        if parsed.isna().any():
            raise ValueError("ARIMA needs valid date_listed values")
        return parsed.dt.to_period("M")

    def fit(self, dates, log_residuals):
        months = self._months(dates)
        residuals = np.asarray(log_residuals, dtype=float).reshape(-1)
        if len(months) != len(residuals) or not np.isfinite(residuals).all():
            raise ValueError("ARIMA dates and residuals must be aligned and finite")

        monthly = pd.Series(residuals).groupby(months).median().sort_index()
        if len(monthly) < 6:
            raise ValueError("ARIMA needs at least six historical months")

        # Build a regular monthly time series. Missing months (if any) are
        # interpolated using only the training-period residual series.
        month_index = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
        monthly = monthly.reindex(month_index).interpolate(limit_direction="both")
        self.first_month, self.last_month = month_index[0], month_index[-1]
        self.result = ARIMA(monthly, order=self.order).fit()
        print(
            "ARIMA converged:",
            self.result.mle_retvals.get("converged")
        )

        print(
            "Iterations:",
            self.result.mle_retvals.get("iterations")
        )

        print(
            "Parameters:",
            self.result.params
        )
        return self

    def predict(self, dates):
        if self.result is None:
            raise RuntimeError("Fit ARIMA before predicting")
        months = self._months(dates)
        if months.empty:
            return np.array([], dtype=float)
        offsets = months.astype("int64").to_numpy() - self.first_month.ordinal
        if (offsets < 0).any():
            raise ValueError("Cannot predict before the first ARIMA training month")
        # In-sample predictions are one-step-ahead fitted values; future
        # months use forecasts. One call handles both and preserves row order.
        values = np.asarray(
            self.result.predict(start=0, end=int(offsets.max())), dtype=float
        )
        return values[offsets]

class GBOffMarketModel:
    """Boosted property model (including geography) + residual ARIMA."""

    def __init__(self, algorithm="catboost", *, use_remarks=True,
                 max_text_features=1000, model_params=None, arima_order=(1, 1, 0)):
        self.algorithm = algorithm
        self.use_remarks = use_remarks
        self.max_text_features = max_text_features
        self.model_params = model_params
        self.arima_order = arima_order
        self.features = BoostingFeatures(
            use_remarks=use_remarks,
            max_text_features=max_text_features,
            include_time=True,
        )
        self.property_model = GBPropertyModel(algorithm, model_params)
        self.temporal_model = ArimaResidualModel(order=arima_order)
        #self.temporal_model = DummyModel()
        self._fitted = False

    def fit(self, train):
        log_prices = _checked_log_prices(train)
        X = self.features.fit_transform(train, model=self.algorithm, track="off_market")
        self.property_model.fit(X, log_prices, self.features.categorical_features)
        residuals = log_prices - self.property_model.predict_log(X)
        self.temporal_model.fit(train["date_listed"], residuals)
        self._fitted = True
        return self

    def predict_log(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        if homes.empty:
            return np.array([], dtype=float)
        X = self.features.transform(homes, model=self.algorithm)
        return (self.property_model.predict_log(X)
                + self.temporal_model.predict(homes["date_listed"]))

    def predict(self, homes):
        return np.exp(self.predict_log(homes))

    def feature_importance(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before inspecting importance")
        return self.property_model.feature_importance(self.features.columns_)


class GBListAwareModel:
    """Off-market boosting+ARIMA followed by a listing-price Ridge correction.

    The Ridge correction is trained from past-only, chronological predictions
    of stage 1. Listing price is never fed to the off-market stage.
    """

    def __init__(self, algorithm="catboost", *, use_remarks=True,
                 max_text_features=1000, model_params=None,
                 arima_order=(1, 1, 0), correction_alpha=50.0, n_folds=3):
        self.base_kwargs = dict(
            algorithm=algorithm, use_remarks=use_remarks,
            max_text_features=max_text_features, model_params=model_params,
            arima_order=arima_order,
        )
        self.base_model = GBOffMarketModel(**self.base_kwargs)
        self.correction_model = Ridge(alpha=correction_alpha)
        self.correction_scaler = StandardScaler()
        self.n_folds = n_folds
        self.house_types = None
        self.correction_columns = None
        self.correction_medians = None
        self.oof_count_ = 0
        self._fitted = False

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
            earlier = GBOffMarketModel(**self.base_kwargs).fit(past)
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


def make_model(track, algorithm="catboost", **kwargs):
    """Drop-in model factory for `run.py` with selectable boosting library."""
    if track == "off_market":
        return GBOffMarketModel(algorithm=algorithm, **kwargs)
    if track == "list_aware":
        return GBListAwareModel(algorithm=algorithm, **kwargs)
    raise ValueError(f"Unknown competition track: {track}")
