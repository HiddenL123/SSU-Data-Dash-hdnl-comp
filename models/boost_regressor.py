"""CatBoost, LightGBM, and XGBoost component with the existing repository defaults."""
from .components import BoostingComponent

class CatBoostRegressor(BoostingComponent):
    backend = "catboost"
    estimator_name = "CatBoostRegressor"


class LightGBMRegressor(BoostingComponent):
    backend = "lightgbm"
    estimator_name = "LGBMRegressor"


class XGBoostRegressor(BoostingComponent):
    backend = "xgboost"
    estimator_name = "XGBRegressor"