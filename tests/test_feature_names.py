"""Feature names describe their inputs without changing the legacy matrices."""
import pandas as pd
import pytest

from features.features import make_features
from features.gb_features import BoostingFeatures
from test_starter import make_listings


@pytest.mark.parametrize("backend", ["catboost", "lightgbm", "xgboost"])
@pytest.mark.parametrize("name,include_time", [("Xp", False), ("Xpt", True)])
def test_named_features_match_legacy_matrices(backend, name, include_time):
    train = make_listings(100)
    homes = make_listings(10, seed=1).assign(community="Unseen")
    provider = make_features(name, model=backend)
    legacy = BoostingFeatures(include_time=include_time)
    pd.testing.assert_frame_equal(
        provider.fit_transform(train),
        legacy.fit_transform(train, model=backend, track="off_market"),
    )
    pd.testing.assert_frame_equal(provider.transform(homes), legacy.transform(homes, model=backend))
    changed_dates = homes.assign(date_listed=homes.date_listed + pd.DateOffset(months=1))
    before, after = provider.transform(homes), provider.transform(changed_dates)
    # Compare each column directly so missing numeric values count as equal.
    changed_columns = {c for c in before if not before[c].equals(after[c])}
    assert changed_columns == ({"months_since_2021", "season_sin", "season_cos"} if include_time else set())


@pytest.mark.parametrize("backend", ["catboost", "lightgbm", "xgboost"])
def test_market_variant_forwards_backend(backend):
    provider = make_features("Xpt_market", model=backend)
    assert provider.base_provider.model == backend
    train = make_listings(100)
    X = provider.fit_transform(train)
    assert "months_since_2021" in X
    assert any(column.startswith("market_") for column in X)
    if backend != "catboost":
        assert isinstance(X.community.dtype, pd.CategoricalDtype)
