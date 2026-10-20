
    
  
"""List-aware regression with legacy CatBoost and multistage MODEL_SPECS modes."""
from copy import deepcopy

import numpy as np
from catboost import CatBoostRegressor

from models.features import BoostingFeatures, make_features
from models.ratio_comparable_features import RatioComparableFeatures
from .components import adapt_component


class DirectListingModel:
    """Predict log sale price directly, or log sale/list ratio with reconstruction.

    `modelspecs=None` preserves the original single-stage CatBoost pipeline.
    If provided, each stage is fitted to the residual from preceding stages.
    """

    def __init__(
        self,
        target="ratio",
        model_params=None,
        include_time=True,
        use_comparables=False,
        modelspecs=None,
    ):
        if target not in ("direct", "ratio"):
            raise ValueError("target must be 'direct' or 'ratio'")

        self.target = target
        self.modelspecs = deepcopy(list(modelspecs)) if modelspecs is not None else None
        self.components = []

        if self.modelspecs is None:
            # Original feature and estimator path. Existing calls retain their behavior.
            self.features = BoostingFeatures(
                use_remarks=True,
                include_time=include_time,
            )
            if use_comparables:
                self.features = RatioComparableFeatures(
                    self.features,
                    candidates=128,
                    neighbors=32,
                    radius_km=4.0,
                    half_life_days=180,
                )
            params = {
                "loss_function": "RMSE",
                "random_seed": 42,
                "verbose": False,
            }
            params.update(model_params or {})
            self.model = CatBoostRegressor(**params)
        else:
            if not self.modelspecs:
                raise ValueError("modelspecs must contain at least one component")
            if model_params is not None:
                raise ValueError(
                    "In multistage mode put hyperparameters in MODEL_SPECS, "
                    "not model_params."
                )
            # The model spec controls the feature set. In this mode,
            # include_time/use_comparables are legacy compatibility arguments
            # and do not override the registered feature provider.
            for model_class, feature_name, params in self.modelspecs:
                component = adapt_component(model_class(**(params or {})))
                features = make_features(
                    feature_name,
                    model=getattr(component, "feature_backend", "catboost"),
                    track="list_aware",
                )
                self.components.append((component, features))
            self.model = self.components[0][0]
            self.features = self.components[0][1]

        self._fitted = False

    @staticmethod
    def _log_price(homes, column):
        values = homes[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column} must be positive and finite")
        return np.log(values)

    @staticmethod
    def _prediction(component, X, n):
        values = np.asarray(component.predict(X), dtype=float).reshape(-1)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError("Each component must predict one finite log value per home")
        return values

    def fit(self, train):
        self._fitted = False
        log_sold = self._log_price(train, "sold_price")
        log_list = self._log_price(train, "list_price")
        if not len(train):
            raise ValueError("Training data must not be empty")
        target_log = log_sold if self.target == "direct" else log_sold - log_list

        if self.modelspecs is None:
            # Original CatBoost path, including list-aware track and categorical columns.
            X = self.features.fit_transform(train, track="list_aware", model="catboost")
            self.model.fit(X, target_log, cat_features=self.features.categorical_features)
        else:
            residual = target_log.copy()
            for component, features in self.components:
                X = features.fit_transform(train)
                component.fit(X, residual)
                residual = residual - self._prediction(component, X, len(train))

        self._fitted = True
        return self


    def predict_log(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        if homes.empty:
            return np.array([], dtype=float)

        if self.modelspecs is None:
            # Original single-stage CatBoost path.
            X = self.features.transform(
                homes,
                model="catboost",
            )
            prediction = np.asarray(
                self.model.predict(X),
                dtype=float,
            ).reshape(-1)

        else:
            # Multistage: sum all component predictions.
            prediction = np.zeros(len(homes), dtype=float)

            for component, features in self.components:
                X = features.transform(homes)
                prediction += self._prediction(
                    component, X, len(homes)
                )

        if self.target == "ratio":
            prediction += self._log_price(homes, "list_price")

        return prediction

    def predict(self, homes):
        return np.exp(self.predict_log(homes))

