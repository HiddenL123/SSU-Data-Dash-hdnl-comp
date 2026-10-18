"""CatBoost component with the existing repository defaults."""
from .components import BoostingComponent


class CatBoostRegressor(BoostingComponent):
    backend = "catboost"
    estimator_name = "CatBoostRegressor"
