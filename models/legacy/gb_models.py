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


from models.defaults import DEFAULT_PARAMS


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


from models.arima import ArimaResidualModel

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


from models.listing import ListingAdjustment


class GBListAwareModel(ListingAdjustment):
    """Legacy constructor for the unchanged shared listing adjustment."""

    def __init__(self, algorithm="catboost", *, use_remarks=True,
                 max_text_features=1000, model_params=None,
                 arima_order=(1, 1, 0), correction_alpha=50.0, n_folds=3):
        self.base_kwargs = dict(
            algorithm=algorithm, use_remarks=use_remarks,
            max_text_features=max_text_features, model_params=model_params,
            arima_order=arima_order,
        )
        super().__init__(correction_alpha=correction_alpha, n_folds=n_folds)

    def _new_base_model(self):
        return GBOffMarketModel(**self.base_kwargs)


def make_model(track, algorithm="catboost", **kwargs):
    """Drop-in model factory for `run.py` with selectable boosting library."""
    if track == "off_market":
        return GBOffMarketModel(algorithm=algorithm, **kwargs)
    if track == "list_aware":
        return GBListAwareModel(algorithm=algorithm, **kwargs)
    raise ValueError(f"Unknown competition track: {track}")
