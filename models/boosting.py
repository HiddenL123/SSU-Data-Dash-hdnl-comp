"""Legacy boosting constructor options routed through the modular containers."""
import pandas as pd

from .arima import ArimaResidualModel
from .catboost import CatBoostRegressor
from .lightgbm import LGBMRegressor
from .xgboost import XGBRegressor
from .model import OffMarketModel, ListAwareModel, make_model as modular_make_model


BOOSTERS = {
    "catboost": CatBoostRegressor,
    "lightgbm": LGBMRegressor,
    "xgboost": XGBRegressor,
}


def boosting_specs(algorithm="catboost", model_params=None, arima_order=(1, 1, 0)):
    if algorithm not in BOOSTERS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return [
        (BOOSTERS[algorithm], "Xpt_time", model_params),
        (ArimaResidualModel, "date_listed", {"order": arima_order}),
    ]


class GBOffMarketModel(OffMarketModel):
    """Modular equivalent of the former GBOffMarketModel constructor."""

    def __init__(self, algorithm="catboost", *, use_remarks=True,
                 max_text_features=1000, model_params=None, arima_order=(1, 1, 0)):
        super().__init__(boosting_specs(algorithm, model_params, arima_order))
        self.algorithm = algorithm
        self.use_remarks = use_remarks
        self.max_text_features = max_text_features
        self.model_params = model_params
        self.arima_order = arima_order
        self.features = self.components[0][1].state
        self.features.use_remarks = use_remarks
        self.features.max_text_features = max_text_features
        self.property_model = self.components[0][0]
        self.temporal_model = self.components[1][0]

    def feature_importance(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before inspecting importance")
        return pd.Series(
            self.property_model.estimator.feature_importances_,
            index=self.features.columns_, dtype=float,
        ).sort_values(ascending=False)


class GBListAwareModel(ListAwareModel):
    """Keep historical folds configurable while using fresh modular base models."""

    def __init__(self, algorithm="catboost", *, use_remarks=True,
                 max_text_features=1000, model_params=None,
                 arima_order=(1, 1, 0), correction_alpha=50.0, n_folds=3):
        self.base_kwargs = dict(
            algorithm=algorithm, use_remarks=use_remarks,
            max_text_features=max_text_features, model_params=model_params,
            arima_order=arima_order,
        )
        super().__init__(boosting_specs(algorithm, model_params, arima_order),
                         correction_alpha=correction_alpha, n_folds=n_folds)

    def _new_base_model(self):
        return GBOffMarketModel(**self.base_kwargs)


def make_model(track, algorithm="catboost", *, modelspecs=None, **kwargs):
    """Support old boosting options or an explicit new component specification."""
    if modelspecs is not None:
        if algorithm != "catboost" or kwargs:
            raise ValueError("Use modelspecs or legacy boosting options, not both")
        return modular_make_model(track, modelspecs=modelspecs)
    if track == "off_market":
        return GBOffMarketModel(algorithm=algorithm, **kwargs)
    if track == "list_aware":
        return GBListAwareModel(algorithm=algorithm, **kwargs)
    raise ValueError(f"Unknown competition track: {track}")
