"""Regression and ablation tests for comparable-sales candidate selection."""
import numpy as np
import pandas as pd
import pytest

import data
from models.features import make_features
from models.spatiotemporal_features import ComparableSalesFeatures
from models.boost_regressor import XGBoostRegressor
from models.model import OffMarketModel
from models.direct_listing import DirectListingModel
from test_starter import make_listings
from comparable_benchmark import run_comparable_benchmark


def _sales(types, lats, sold_dates, prices):
    n = len(types)
    assert len(lats) == len(sold_dates) == len(prices) == n
    return make_listings(n).assign(
        house_type_name=types,
        latitude=lats,
        longitude=-79.4,
        date_sold=pd.to_datetime(sold_dates),
        date_listed=pd.Timestamp("2025-01-01"),
        house_area="1200",
        sold_price=prices,
        list_price=np.asarray(prices, dtype=float),
        remarks="renovated original kitchen",
    )


def _home(house_type="Co-op Apt"):
    return make_listings(1, first_id=10000).assign(
        house_type_name=house_type, latitude=43.7, longitude=-79.4,
        house_area="1200", date_listed=pd.Timestamp("2025-05-15"),
        list_price=700000.0,
    )


def test_age_first_finds_eligible_sales_beyond_old_nearest_neighbors():
    train = _sales(
        ["Condo Apt"] * 4,
        [43.70001, 43.70002, 43.71, 43.711],
        ["2020-01-15", "2020-02-15", "2025-02-15", "2025-02-16"],
        [100000, 100000, 500000, 500000],
    )
    base = ComparableSalesFeatures(candidates=2, neighbors=2, min_comps=1)
    corrected = ComparableSalesFeatures(
        candidates=2, neighbors=2, min_comps=1, age_filter_first=True
    )
    legacy = base.fit(train).transform(_home())
    fixed = corrected.fit(train).transform(_home())
    assert legacy.comp_count.iloc[0] == 0
    assert np.isnan(legacy.comp_log_price.iloc[0])
    assert fixed.comp_count.iloc[0] == 2
    np.testing.assert_allclose(fixed.comp_log_price.iloc[0], np.log(500000), atol=1e-4)


def test_type_quota_adds_underrepresented_type_to_final_neighbors():
    train = _sales(
        ["Condo Apt", "Condo Apt", "Condo Apt", "Co-op Apt", "Co-op Apt"],
        [43.70001, 43.70002, 43.70003, 43.71, 43.711],
        ["2025-02-15"] * 5,
        [900000, 900000, 900000, 400000, 400000],
    )
    base = ComparableSalesFeatures(
        candidates=3, neighbors=2, min_comps=1
    )
    quota = ComparableSalesFeatures(
        candidates=3, neighbors=2, min_comps=1,
        min_same_type_neighbors=1,
    )
    original = base.fit(train).transform(_home())
    improved = quota.fit(train).transform(_home())
    assert original.comp_same_type_share.iloc[0] == 0
    assert improved.comp_same_type_share.iloc[0] > 0
    assert improved.comp_count.iloc[0] == 2
    assert improved.comp_log_price.iloc[0] < original.comp_log_price.iloc[0]


def test_type_quota_handles_nearest_pool_entirely_too_old():
    train = _sales(
        ["Condo Apt", "Condo Apt", "Co-op Apt", "Co-op Apt"],
        [43.70001, 43.70002, 43.71, 43.711],
        ["2020-01-15", "2020-02-15", "2025-02-15", "2025-02-16"],
        [100000, 100000, 400000, 400000],
    )
    model = ComparableSalesFeatures(
        candidates=2, neighbors=2, min_comps=1,
        min_same_type_neighbors=1, age_filter_first=False
    )
    out = model.fit(train).transform(_home())
    assert out.comp_count.iloc[0] == 2
    assert out.comp_same_type_share.iloc[0] > 0


@pytest.mark.parametrize("age_first,quota", [
    (False, 0), (True, 0), (False, 1), (True, 1),
])
def test_comparable_cutoff_excludes_future_sales(age_first, quota):
    train = _sales(
        ["Co-op Apt", "Co-op Apt", "Condo Apt"],
        [43.71, 43.711, 43.70001],
        ["2025-02-15", "2025-02-16", "2025-06-15"],
        [400000, 420000, 100000000],
    )
    kwargs = dict(
        candidates=3, neighbors=2, min_comps=1,
        age_filter_first=age_first, min_same_type_neighbors=quota,
    )
    before = ComparableSalesFeatures(**kwargs).fit(train).transform(_home())
    altered = train.assign(
        sold_price=[400000, 420000, 500000000]
    )
    after = ComparableSalesFeatures(**kwargs).fit(altered).transform(_home())
    pd.testing.assert_frame_equal(before, after)


def test_default_flags_match_unmodified_legacy_call():
    train = make_listings(80)
    homes = make_listings(5, seed=10)
    a = ComparableSalesFeatures()
    b = ComparableSalesFeatures(
        age_filter_first=False, min_same_type_neighbors=0
    )
    pd.testing.assert_frame_equal(
        a.fit(train).transform(homes),
        b.fit(train).transform(homes),
        check_exact=True,
    )


@pytest.mark.parametrize("kwargs", [
    {"min_same_type_neighbors": -1},
    {"min_same_type_neighbors": 33},
    {"min_same_type_neighbors": 1.5},
])
def test_invalid_type_quota_raises(kwargs):
    with pytest.raises(ValueError, match="min_same_type_neighbors"):
        ComparableSalesFeatures(**kwargs)


@pytest.mark.parametrize("track", ["off_market", "list_aware"])
@pytest.mark.parametrize("suffix", [
    "", "__comp_age_first", "__comp_type_quota", "__comp_both"
])
def test_factory_preserves_model_features_and_xgboost_categories(track, suffix):
    train = make_listings(90).assign(remarks="renovated original kitchen")
    homes = data.hide_answers(make_listings(4, seed=4), track)
    variant = "Xpt_spatiotemporal__selected_binary" + suffix
    feat = make_features(variant, model="xgboost", track=track)
    X = feat.fit_transform(train)
    future = feat.transform(homes)
    assert list(X.columns) == list(future.columns)
    assert len(X.columns) == len(future.columns)
    assert "comp_log_price" in future
    assert ("comp_vs_list_log" in future) == (track == "list_aware")
    for name in ["house_type_name", "house_style", "municipality", "community", "garage"]:
        assert isinstance(X[name].dtype, pd.CategoricalDtype)
        assert isinstance(future[name].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_combined_mode_fits_xgboost_end_to_end(track):
    train = make_listings(100).assign(remarks="renovated family original")
    future = data.hide_answers(make_listings(4, seed=9), track)
    specs = [(
        XGBoostRegressor,
        "Xpt_spatiotemporal__selected_binary__comp_both",
        {"n_estimators": 5, "n_jobs": 1},
    )]
    model = (
        OffMarketModel(specs, center_target=True)
        if track == "off_market"
        else DirectListingModel(target="ratio", modelspecs=specs)
    )
    y = model.fit(train).predict(future)
    assert (y > 0).all() and np.isfinite(y).all()


def test_runner_rejects_confirmation_period(tmp_path):
    with pytest.raises(ValueError, match="before 2025-12"):
        run_comparable_benchmark(
            months=["2026-01"], output_dir=tmp_path,
            listings=make_listings(10),
        )

def test_legacy_joblib_saved_comparables_load_without_new_attributes(tmp_path):
    """Old pickles omit fields introduced by the age-first/type experiments."""
    import joblib

    train = make_listings(60)
    query = make_listings(3, seed=32).assign(
        date_listed=pd.Timestamp("2026-06-01")
    )
    original = ComparableSalesFeatures().fit(train)
    expected = original.transform(query)
    del original.age_filter_first
    del original.min_same_type_neighbors

    path = tmp_path / "legacy_comparables.joblib"
    joblib.dump(original, path)
    restored = joblib.load(path)
    assert restored.age_filter_first is False
    assert restored.min_same_type_neighbors == 0
    pd.testing.assert_frame_equal(restored.transform(query), expected, check_exact=True)


def test_modern_joblib_saved_comparable_settings_survive(tmp_path):
    import joblib

    modern = ComparableSalesFeatures(
        age_filter_first=True, min_same_type_neighbors=8
    )
    path = tmp_path / "modern_comparables.joblib"
    joblib.dump(modern, path)
    restored = joblib.load(path)
    assert restored.age_filter_first is True
    assert restored.min_same_type_neighbors == 8
