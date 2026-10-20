"""Training-fitted feature providers, independent of residual orchestration."""

from .gb_features import BoostingFeatures, square_feet
from models.market_features import MarketAugmentedFeatures
from models.spatiotemporal_features import ComparableAugmentedFeatures
from models.ratio_comparable_features import RatioComparableFeatures


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
    "Xp": lambda model: PropertyFeatures(model, include_time=False),
    "Xpt": lambda model: PropertyFeatures(model, include_time=True),
    "date_listed": lambda model: DateFeatures(),
    "Xpt_market": lambda model: MarketAugmentedFeatures(make_features("Xpt", model=model)),
    "Xpt_spatiotemporal": lambda model: ComparableAugmentedFeatures(make_features("Xpt", model=model),
                                                                   candidates=128,
                                                                   neighbors=32,
                                                                   radius_km=4.0,
                                                                   half_life_days=180,)

}

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

def make_features(
    feature_name,
    model="catboost",
    track="off_market",
):
    if (
        track == "list_aware"
        and feature_name in LIST_AWARE_FEATURE_SETS
    ):
        return _ListAwareBoundFeatures(
            LIST_AWARE_FEATURE_SETS[feature_name](model),
            model,
        )

    if feature_name not in FEATURE_PROVIDERS:
        raise ValueError(f"Unknown feature set: {feature_name}")
    return FEATURE_PROVIDERS[feature_name](model)