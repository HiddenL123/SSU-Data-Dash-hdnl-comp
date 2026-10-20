"""Ordered additive models on the log-price scale.

Each spec is (component class, feature name, constructor kwargs or None).
Components implement fit(X, residual) and predict(X), both on the log scale.
"""

from copy import deepcopy

import numpy as np

from .arima import ArimaResidualModel
from .catboost import CatBoostRegressor
from .components import adapt_component
from .features import make_features
from .listing import ListingAdjustment
from .defaults import DEFAULT_PARAMS
from models.ratio_comparable_features import RatioComparableFeatures
from models.features import BoostingFeatures


DEFAULT_MODELSPECS = [
    (CatBoostRegressor, "Xpt", deepcopy(DEFAULT_PARAMS["catboost"])),
    (ArimaResidualModel, "date_listed", {"order": (1, 1, 0)}),
]
# Alias for the original draft's name.
DEFAULT_SPECS = DEFAULT_MODELSPECS


class OffMarketModel:
    """Fit fresh components sequentially to remaining log-price residuals."""

    def __init__(self, modelspecs):
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
        residual = np.log(prices)
        for component, features in self.components:
            X = features.fit_transform(train)
            component.fit(X, residual)
            residual = residual - self._prediction(component, X, len(train))
        self._fitted = True
        return self

    def predict_log(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        prediction = np.zeros(len(homes), dtype=float)
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

class _ListAwareBoundFeatures:
    """Bind track and backend to a feature provider."""

    def __init__(self, provider, model):
        self.provider = provider
        self.model = model

    def fit_transform(self, homes):
        return self.provider.fit_transform(
            homes,
            track="list_aware",
            model=self.model,
        )

    def transform(self, homes):
        return self.provider.transform(
            homes,
            model=self.model,
        )

    @property
    def categorical_features(self):
        return self.provider.categorical_features


def _list_aware_xpt(model):
    return BoostingFeatures(
        use_remarks=True,
        include_time=True,
    )


def _list_aware_xpt_spatiotemporal(model):
    return RatioComparableFeatures(
        _list_aware_xpt(model),
        candidates=128,
        neighbors=32,
        radius_km=4.0,
        half_life_days=180,
    )


LIST_AWARE_FEATURE_SETS = {
    "Xpt": _list_aware_xpt,
    "Xpt_spatiotemporal": _list_aware_xpt_spatiotemporal,
}
