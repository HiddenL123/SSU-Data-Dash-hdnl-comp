"""Training-fitted feature providers, independent of residual orchestration."""

from .gb_features import BoostingFeatures, square_feet


class PropertyFeatures:
    """Reuse legacy text vocabulary, geography and native categorical encodings."""

    def __init__(self, model="catboost", include_time=False):
        if model not in ("catboost", "lightgbm", "xgboost"):
            raise ValueError(f"Unknown feature backend: {model}")
        self.model = model
        self.state = BoostingFeatures(include_time=include_time)

    def fit_transform(self, train):
        return self.state.fit_transform(train, model=self.model, track="off_market")

    def transform(self, homes):
        return self.state.transform(homes, model=self.model)


class DateFeatures:
    """Pass listing dates to a temporal component without learning state."""

    def fit_transform(self, train):
        return self.transform(train)

    def transform(self, homes):
        return homes["date_listed"].copy()


FEATURE_PROVIDERS = {
    "Xpt": lambda model: PropertyFeatures(model, include_time=False),
    "Xpt_time": lambda model: PropertyFeatures(model, include_time=True),
    "date_listed": lambda model: DateFeatures(),
}


def make_features(name: str, model: str = "catboost"):
    """Create independent state; register extra factories in FEATURE_PROVIDERS."""
    if name not in FEATURE_PROVIDERS:
        raise ValueError(f"Unknown feature specification: {name}")
    return FEATURE_PROVIDERS[name](model)
