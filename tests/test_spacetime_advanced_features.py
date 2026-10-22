"""Run with: pytest -q tests/test_spacetime_advanced_features.py"""
import numpy as np
import pandas as pd
import pytest

from models.spacetime_advanced_features import (
    ADVANCED_FEATURE_VARIANTS, AdvancedSpaceTimeAddon,
)
from models.spacetime_screen_features import FEATURE_VARIANTS, SpaceTimeAddon


@pytest.fixture(scope="module")
def sales():
    rng = np.random.default_rng(123)
    n = 110 * 8
    m = np.repeat(np.arange(8), 110)
    sold = pd.to_datetime("2024-01-15") + pd.to_timedelta(m * 30, unit="D")
    municipality = np.where(np.arange(n) % 3 == 0, "Toronto", "Mississauga")
    kind = np.where(np.arange(n) % 4 == 0, "Condo Apt", "Detached")
    lat = np.where(municipality == "Toronto", 43.67, 43.60) + rng.normal(0, 0.009, n)
    lon = np.where(municipality == "Toronto", -79.38, -79.61) + rng.normal(0, 0.011, n)
    size = rng.integers(650, 2300, n)
    logprice = (np.log(size) + 6.1 + (kind == "Detached") * 0.12
                + m * 0.003 + rng.normal(0, 0.1, n))
    return pd.DataFrame({
        "ml_num": np.arange(n), "date_sold": sold,
        "date_listed": sold - pd.to_timedelta(21, unit="D"),
        "sold_price": np.exp(logprice), "house_area": size.astype(str),
        "municipality": municipality, "house_type_name": kind,
        "latitude": lat, "longitude": lon,
        "bedrooms": rng.integers(1, 5, n),
        "washrooms": rng.integers(1, 4, n),
        "parking_total": rng.integers(0, 3, n),
    })


def _hidden(homes):
    return homes.drop(columns=["date_sold", "sold_price", "ml_num"])


@pytest.mark.parametrize("variant", ADVANCED_FEATURE_VARIANTS)
def test_screen_interface_and_no_sale_fields(sales, variant):
    train = sales.iloc[:770].copy()
    query = _hidden(sales.iloc[500:510].copy())
    addon = SpaceTimeAddon(variant).fit(train)
    actual = addon.transform(query, pd.DataFrame(index=query.index))
    assert variant in FEATURE_VARIANTS
    assert actual.index.equals(query.index)
    assert len(actual) == len(query)
    assert len(actual.columns) > 0
    assert not actual.columns.duplicated().any()
    assert all(pd.api.types.is_float_dtype(actual[c]) for c in actual)
    assert not np.isinf(actual.to_numpy(dtype=float)).any()


@pytest.mark.parametrize("variant", ADVANCED_FEATURE_VARIANTS)
def test_future_sale_prices_cannot_change_past_features(sales, variant):
    # All changes are to properties sold on/after the held-out query month.
    # Features for query month M must depend only on sales completed before M.
    train = sales.copy()
    month = pd.Period("2024-05", freq="M")
    query = train.loc[train.date_listed.dt.to_period("M") == month].iloc[:8]
    assert len(query) > 0
    hidden = _hidden(query.copy())
    before = AdvancedSpaceTimeAddon(variant).fit(train).transform(hidden)
    changed = train.copy()
    changed.loc[changed.date_sold >= month.start_time, "sold_price"] *= 10
    after = AdvancedSpaceTimeAddon(variant).fit(changed).transform(hidden)
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy(), equal_nan=True, rtol=0, atol=1e-6)


def test_experiment_is_opt_in(sales):
    # Existing feature family remains unchanged, even when the advanced file is imported.
    assert "baseline" in FEATURE_VARIANTS
    assert "adaptive_comps" in FEATURE_VARIANTS