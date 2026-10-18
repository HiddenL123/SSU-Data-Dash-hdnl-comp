"""LightGBM component using training-fitted pandas categorical columns."""
from .components import BoostingComponent


class LGBMRegressor(BoostingComponent):
    backend = "lightgbm"
    estimator_name = "LGBMRegressor"
