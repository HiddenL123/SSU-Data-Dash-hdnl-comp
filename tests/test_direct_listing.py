"""Regression checks for the preserved direct-listing and nested feature pipelines."""
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor as NativeCatBoost

import data
from models import make_model
from models.arima import ArimaResidualModel
from models.direct_listing import DirectListingModel
from features.features import BoostingFeatures, make_features
from features.ratio_comparable_features import RatioComparableFeatures
from features.spatiotemporal_features import ComparableAugmentedFeatures
from test_modular_models import FractionOfMean
from test_starter import make_listings


SHORT_PARAMS = dict(iterations=8, depth=3, thread_count=1, random_seed=42,
                    loss_function="RMSE", verbose=False, allow_writing_files=False)


@pytest.mark.parametrize("target", ["ratio", "direct"])
@pytest.mark.parametrize("comparables", [False, True])
def test_legacy_and_specs_have_identical_features_and_predictions(target, comparables):
    train = make_listings(80)
    homes = data.hide_answers(make_listings(12, seed=3), "list_aware")
    name = "Xpt_spatiotemporal" if comparables else "Xpt"
    legacy = DirectListingModel(target, SHORT_PARAMS, True, comparables).fit(train)
    modular = DirectListingModel(target=target, modelspecs=[
        (NativeCatBoost, name, SHORT_PARAMS),
    ]).fit(train)
    pd.testing.assert_frame_equal(
        legacy.features.transform(homes, model="catboost"), modular.features.transform(homes))
    pd.testing.assert_frame_equal(
        legacy.features.fit_transform(train, track="list_aware", model="catboost"),
        modular.features.fit_transform(train))
    for rows in (train, homes):
        np.testing.assert_allclose(modular.predict_log(rows), legacy.predict_log(rows), rtol=0, atol=1e-12)
        np.testing.assert_allclose(modular.predict(rows), legacy.predict(rows), rtol=1e-12)
        np.testing.assert_allclose(modular.predict(rows), np.exp(modular.predict_log(rows)))
        assert (modular.predict(rows) > 0).all()


def test_list_aware_provider_identity_and_ratio_columns():
    plain = make_features("Xpt", track="list_aware")
    spatial = make_features("Xpt_spatiotemporal", track="list_aware")
    assert type(plain.provider) is BoostingFeatures
    assert type(spatial.provider) is RatioComparableFeatures
    base = spatial.provider.base_provider
    assert base.use_remarks and base.include_time
    comps = spatial.provider.comparables
    assert (comps.candidates, comps.neighbors, comps.radius_km, comps.half_life_days) == (128, 32, 4.0, 180)
    train = make_listings(60)
    X = spatial.fit_transform(train)
    assert {"log_list_price", "comp_vs_list_log"} <= set(X)
    np.testing.assert_allclose(X.log_list_price, np.log(train.list_price), rtol=1e-6)
    np.testing.assert_allclose(X.comp_vs_list_log,
                               X.comp_log_price - np.log(train.list_price), rtol=1e-6)
    assert "comp_vs_list_log" not in plain.fit_transform(train)


@pytest.mark.parametrize("backend", ["catboost", "lightgbm", "xgboost"])
@pytest.mark.parametrize("track,name", [
    ("off_market", "Xpt_market"), ("off_market", "Xpt_spatiotemporal"),
    ("list_aware", "Xpt"), ("list_aware", "Xpt_spatiotemporal"),
])
def test_nested_backend_preprocessing_matches_base(backend, track, name):
    train = make_listings(45)
    homes = data.hide_answers(make_listings(5, seed=3), track).assign(community="Unseen")
    wrapped = make_features(name, model=backend, track=track)
    base = make_features("Xpt", model=backend, track=track)
    X = wrapped.fit_transform(train)
    expected = base.fit_transform(train)
    pd.testing.assert_frame_equal(X[expected.columns], expected)
    actual = wrapped.transform(homes)
    pd.testing.assert_frame_equal(actual[expected.columns], base.transform(homes))
    if backend == "catboost":
        assert not isinstance(X.community.dtype, pd.CategoricalDtype)
    else:
        assert isinstance(X.community.dtype, pd.CategoricalDtype)
        assert actual.community.dtype == X.community.dtype


@pytest.mark.parametrize("name", ["Xp", "Xpt", "Xpt_market", "Xpt_spatiotemporal"])
def test_off_market_listing_price_is_unused_during_fit_and_transform(name):
    train = make_listings(45)
    altered = train.assign(list_price=train.list_price * 100)
    first, second = make_features(name), make_features(name)
    X = first.fit_transform(train)
    if name == "Xpt_spatiotemporal":
        original = ComparableAugmentedFeatures(
            make_features("Xpt"), candidates=128, neighbors=32,
            radius_km=4.0, half_life_days=180)
        pd.testing.assert_frame_equal(X, original.fit_transform(train))
        pd.testing.assert_frame_equal(first.transform(train), original.transform(train))
    pd.testing.assert_frame_equal(X, second.fit_transform(altered))
    pd.testing.assert_frame_equal(first.transform(train), first.transform(altered.drop(columns="list_price")))
    assert not {"list_price", "log_list_price", "comp_vs_list_log"} & set(X)


@pytest.mark.parametrize("track,target", [("off_market", None), ("list_aware", "ratio"),
                                           ("list_aware", "direct")])
def test_all_stages_fit_remaining_residual_and_reconstruct_prices(track, target):
    config = [(FractionOfMean, "date_listed", {"fraction": f}) for f in (0.5, 0.25, 0.1)]
    factory = lambda specs: DirectListingModel(target=target, modelspecs=specs)
    model = make_model(track, modelspecs=config, list_aware_implementation=factory)
    train = make_listings(30)
    model.fit(train)
    residual = np.log(train.sold_price.to_numpy())
    if target == "ratio":
        residual -= np.log(train.list_price.to_numpy())
    total = 0
    for component, _ in model.components:
        np.testing.assert_allclose(component.target, residual)
        total += component.value
        residual -= component.value
    expected = np.full(len(train), total)
    if target == "ratio":
        expected += np.log(train.list_price.to_numpy())
    np.testing.assert_allclose(model.predict_log(train), expected)
    np.testing.assert_allclose(model.predict(train), np.exp(expected))
    assert (model.predict(train) > 0).all()
    assert model.predict(train.iloc[:0]).size == 0


def test_list_aware_catboost_then_arima_executes_both_stages():
    train = make_listings(100)
    model = DirectListingModel(modelspecs=[
        (NativeCatBoost, "Xpt_spatiotemporal", SHORT_PARAMS),
        (ArimaResidualModel, "date_listed", {"order": (1, 1, 0)}),
    ]).fit(train)
    homes = data.hide_answers(make_listings(8, seed=3), "list_aware")
    parts = [c.predict(f.transform(homes)) for c, f in model.components]
    np.testing.assert_allclose(model.predict_log(homes), sum(parts) + np.log(homes.list_price))
    assert np.isfinite(model.predict(homes)).all()


@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_comparables_exclude_sales_on_and_after_month_cutoff(track):
    train = make_listings(40).assign(date_sold=pd.Timestamp("2025-01-31"))
    train.loc[20:29, "date_sold"] = pd.Timestamp("2025-02-01")
    train.loc[30:, "date_sold"] = pd.Timestamp("2025-03-01")
    homes = data.hide_answers(make_listings(4, seed=2), track).assign(date_listed=pd.Timestamp("2025-02-15"))
    altered = train.copy()
    altered.loc[20:, "sold_price"] *= 1000
    first = make_features("Xpt_spatiotemporal", track=track)
    second = make_features("Xpt_spatiotemporal", track=track)
    first.fit_transform(train)
    second.fit_transform(altered)
    X = first.transform(homes)
    assert X.comp_log_price.notna().all()
    pd.testing.assert_frame_equal(X, second.transform(homes))
    # Ensure the assertion is sensitive to eligible historical prices.
    altered.loc[:19, "sold_price"] *= 2
    second.fit_transform(altered)
    assert not np.allclose(X.comp_log_price, second.transform(homes).comp_log_price)


def test_legacy_constructor_options_and_validation():
    model = DirectListingModel("ratio", SHORT_PARAMS, False, True)
    assert isinstance(model.features, RatioComparableFeatures)
    assert not model.features.base_provider.include_time
    model.fit(make_listings(40))
    assert "months_since_2021" not in model.features.transform(make_listings(2))
    with pytest.raises(ValueError, match="target"):
        DirectListingModel(target="invalid")
    with pytest.raises(ValueError, match="at least one"):
        DirectListingModel(modelspecs=[])
    with pytest.raises(ValueError, match="model_params"):
        DirectListingModel(model_params={}, modelspecs=[(FractionOfMean, "date_listed", None)])
    for kwargs in ({"model_params": SHORT_PARAMS},
                   {"modelspecs": [(FractionOfMean, "date_listed", None)]}):
        model = DirectListingModel(**kwargs)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(make_listings(2))
        with pytest.raises(ValueError, match="list_price"):
            model.fit(make_listings(5).assign(list_price=0))
        with pytest.raises(ValueError, match="sold_price"):
            model.fit(make_listings(5).assign(sold_price=np.nan))
