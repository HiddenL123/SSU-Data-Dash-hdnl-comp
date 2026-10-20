"""Ordered additive models on the log-price scale.

Each spec is (component class, feature name, constructor kwargs or None).
Components implement fit(X, residual) and predict(X), both on the log scale.
"""

from copy import deepcopy

import numpy as np

from .arima import ArimaResidualModel
from .boost_regressor import CatBoostRegressor, XGBoostRegressor, LightGBMRegressor
from .components import adapt_component
from .features import make_features
from .listing import ListingAdjustment
from .defaults import DEFAULT_PARAMS


DEFAULT_MODELSPECS = [
    (CatBoostRegressor, "Xpt", deepcopy(DEFAULT_PARAMS["catboost"])),
    (ArimaResidualModel, "date_listed", {"order": (1, 1, 0)}),
]
# Alias for the original draft's name.
DEFAULT_SPECS = DEFAULT_MODELSPECS


class OffMarketModel:
    """Fit fresh components sequentially to remaining log-price residuals."""

    def __init__(self, modelspecs, *, center_target=False):
        self.center_target = center_target
        self.target_offset_ = 0.0
        self.modelspecs = deepcopy(list(modelspecs))
        if not self.modelspecs:
            raise ValueError("modelspecs must contain at least one component")
        self.components = []
        for model_class, feature_name, params in self.modelspecs:
            component = adapt_component(model_class(**(params or {})))
            features = make_features(
                feature_name, model=getattr(component, "feature_backend", "catboost")
            )
            self.components.append((component, features))
        self._fitted = False

    @staticmethod
    def _prediction(component, X, n):
        values = np.asarray(component.predict(X), dtype=float).reshape(-1)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError("Each component must predict one finite log value per home")
        return values

    def fit(self, train):
        self._fitted = False
        prices = train["sold_price"].to_numpy(dtype=float)
        if not len(prices) or not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Training needs positive finite sold_price values")
        
        log_prices = np.log(prices)
        self.target_offset_ = (
            float(np.median(log_prices))
            if self.center_target
            else 0.0
        )

        residual = log_prices - self.target_offset_
        for component, features in self.components:
            X = features.fit_transform(train)
            component.fit(X, residual)
            residual = residual - self._prediction(component, X, len(train))
        self._fitted = True
        return self

    def predict_log(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        prediction = np.full(
            len(homes),
            self.target_offset_,
            dtype=float,
        )
        if homes.empty:
            return prediction
        for component, features in self.components:
            X = features.transform(homes)
            prediction += self._prediction(component, X, len(homes))
        return prediction

    def predict(self, homes):
        return np.exp(self.predict_log(homes))


class ListAwareModel(ListingAdjustment):
    """Existing chronological Ridge adjustment with configurable base models."""

    def __init__(self, modelspecs, correction_alpha=50.0, n_folds=3):
        self.modelspecs = deepcopy(list(modelspecs))
        super().__init__(correction_alpha=correction_alpha, n_folds=n_folds)

    def _new_base_model(self):
        return OffMarketModel(self.modelspecs)


def make_model(track, modelspecs=None, list_aware_implementation = ListAwareModel):
    """Construct an unfitted competition model; defaults match legacy boosting."""
    specs = DEFAULT_MODELSPECS if modelspecs is None else modelspecs
    if track == "off_market":
        return OffMarketModel(specs)
    if track == "list_aware":
        return list_aware_implementation(specs)
    raise ValueError(f"Unknown competition track: {track}")
