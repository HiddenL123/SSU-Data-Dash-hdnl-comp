"""Training-fitted feature providers, independent of residual orchestration."""

from .gb_features import BoostingFeatures, square_feet
from models.market_features import MarketAugmentedFeatures
from models.spatiotemporal_features import ComparableAugmentedFeatures
from models.ratio_comparable_features import RatioComparableFeatures
from .text_features import TEXT_VARIANTS, make_text_base

# Append these names to either "Xpt_spatiotemporal" or its text variants.
# The unmodified feature names continue to use the exact legacy comp code.
COMPARABLE_EXPERIMENTS = {
    "comp_age_first": {"age_filter_first": True},
    "comp_type_quota": {"min_same_type_neighbors": 8},
    "comp_both": {
        "age_filter_first": True,
        "min_same_type_neighbors": 8,
    },
}




class SpaceTimeExperimentFeatures:
    """Off-market experiment wrapper; preserves fitted base XGBoost columns.

    A feature name ending with __st_spacetime_splines, or multiple add-ons
    separated by '+', works inside the original OffMarketModel pipeline.
    These are opt-in experiments and do not affect baseline feature names.
    """

    def __init__(self, base_provider, variants):
        from models.spacetime_screen_features import FEATURE_VARIANTS, SpaceTimeAddon

        names = tuple(variants)
        if (not names or len(names) != len(set(names)) or
            any(name not in FEATURE_VARIANTS or name == "baseline" for name in names)):
            raise ValueError(f"Unknown or repeated spatiotemporal add-on: {names}")
        self.base_provider = base_provider
        self.variants = names
        self.addons = [SpaceTimeAddon(name) for name in names]
        self.columns_ = None

    def _append(self, homes, base_X):
        import pandas as pd

        frames = [base_X]
        for name, addon in zip(self.variants, self.addons):
            extra = addon.transform(homes, base_X)
            if not extra.index.equals(base_X.index):
                raise ValueError(f"{name}: feature rows are misaligned")
            if extra.columns.duplicated().any():
                raise ValueError(f"{name}: duplicated add-on columns")
            frames.append(extra)
        combined = pd.concat(frames, axis=1)
        if combined.columns.duplicated().any():
            raise ValueError("Spatiotemporal features overlap base/add-on columns")
        return combined

    def fit_transform(self, train):
        base_X = self.base_provider.fit_transform(train)
        for addon in self.addons:
            addon.fit(train)
        combined = self._append(train, base_X)
        self.columns_ = list(combined.columns)
        return combined

    def transform(self, homes):
        if self.columns_ is None:
            raise RuntimeError("Call fit_transform() before transform()")
        base_X = self.base_provider.transform(homes)
        combined = self._append(homes, base_X)
        if list(combined.columns) != self.columns_:
            raise ValueError("Changed spatiotemporal feature schema after training")
        return combined

    @property
    def categorical_features(self):
        return getattr(self.base_provider, "categorical_features", [])

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

class _TrackBoundFeatures:
    """Pass the correct track and estimator backend through wrapped providers.

    OffMarketModel and DirectListingModel call fit_transform(train) without
    keyword arguments, so wrappers must not silently default to CatBoost's
    object-string categorical encoding when the estimator uses XGBoost/LightGBM.
    """

    def __init__(self, provider, model, track):
        self.provider = provider
        self.model = model
        self.track = track

    def fit_transform(self, homes):
        return self.provider.fit_transform(
            homes,
            track=self.track,
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


class _ListAwareBoundFeatures(_TrackBoundFeatures):
    """Preserve the previous list-aware feature-provider interface."""

    def __init__(self, provider, model):
        super().__init__(provider, model, track="list_aware")


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
    # Opt-in full-model spatial feature arms. Allow mixtures such as
    # Xpt_spatiotemporal__selected_binary__comp_type_quota__st_spacetime_splines+geo_context.
    # Resolve the ordinary/comp provider first, then append spatial columns.
    if "__st_" in feature_name:
        if track != "off_market":
            raise ValueError("Spatiotemporal experiment add-ons are off_market only")
        ordinary_name, extra_names = feature_name.rsplit("__st_", 1)
        if not ordinary_name or not extra_names:
            raise ValueError("Invalid spatiotemporal feature experiment identifier")
        base = make_features(ordinary_name, model=model, track=track)
        return SpaceTimeExperimentFeatures(base, extra_names.split("+"))

    # Optional independent comparable-sale ablations, available for the
    # default selected TF-IDF as well as any registered text representation.
    # E.g. Xpt_spatiotemporal__selected_binary__comp_type_quota
    comp_options = {}
    for suffix, settings in COMPARABLE_EXPERIMENTS.items():
        tail = "__" + suffix
        if feature_name.endswith(tail):
            comp_options = settings
            feature_name = feature_name[:-len(tail)]
            break

    if comp_options and feature_name == "Xpt_spatiotemporal":
        base = (
            _list_aware_xpt(model)
            if track == "list_aware"
            else make_features("Xpt", model=model, track="off_market")
        )
        kwargs = dict(
            candidates=128, neighbors=32, radius_km=4.0, half_life_days=180,
            **comp_options
        )
        if track == "list_aware":
            return _ListAwareBoundFeatures(
                RatioComparableFeatures(base, **kwargs), model
            )
        if track == "off_market":
            return _TrackBoundFeatures(
                ComparableAugmentedFeatures(base, **kwargs), model, "off_market"
            )
        raise ValueError(f"Unknown competition track: {track}")

    # Controlled text experiments use ordinary MODEL_SPECS feature identifiers,
    # so no model-factory or estimator changes are needed.
    # Example: "Xpt_spatiotemporal__full_binary".
    if feature_name.startswith("Xpt_spatiotemporal__"):
        variant = feature_name.removeprefix("Xpt_spatiotemporal__")
        if variant not in TEXT_VARIANTS:
            raise ValueError(f"Unknown text variant: {variant}")
        base = make_text_base(variant, include_time=True)
        comparable_kwargs = dict(
            candidates=128, neighbors=32,
            radius_km=4.0, half_life_days=180,
            **comp_options,
        )
        if track == "list_aware":
            provider = RatioComparableFeatures(base, **comparable_kwargs)
            return _ListAwareBoundFeatures(provider, model)
        if track == "off_market":
            return _TrackBoundFeatures(
                ComparableAugmentedFeatures(base, **comparable_kwargs),
                model,
                track="off_market",
            )
        raise ValueError(f"Unknown competition track: {track}")

    if comp_options:
        raise ValueError(f"Comparable ablation requires Xpt_spatiotemporal: {feature_name}")

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