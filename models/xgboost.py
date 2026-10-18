"""XGBoost component with native categorical handling enabled by default."""
from .components import BoostingComponent


class XGBRegressor(BoostingComponent):
    backend = "xgboost"
    estimator_name = "XGBRegressor"
