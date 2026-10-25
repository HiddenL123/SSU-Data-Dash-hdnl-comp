"""Text representation tests: legacy parity, shared vocabularies and model parity."""
import numpy as np
import pandas as pd
import pytest

import data
from features.features import make_features
from features.text_features import (
    TEXT_VARIANTS, make_text_base, training_vocabulary,
)
from models.boost_regressor import LightGBMRegressor
from models.model import OffMarketModel
from models.direct_listing import DirectListingModel
from comparisons.text_benchmark import run_benchmark
from test_starter import make_listings


def sample_sales(n=140):
    sales = make_listings(n)
    sales["remarks"] = [
        "Renovated modern kitchen with lovely finishes" if i % 3
        else "Original dated interior with renovation potential"
        for i in range(n)
    ]
    return sales


def test_training_vocabulary_is_label_free_and_has_frequency_filter():
    train = pd.Series(["renovated modern kitchen"] * 14 + ["original kitchen"] * 12)
    terms = training_vocabulary(train, max_features=12, min_df=10)
    assert "kitchen" in terms
    assert "unseenword" not in terms
    assert "renovated modern" in terms
    assert len(terms) <= 12


def test_new_variants_share_exact_training_vocabulary():
    train = sample_sales(80)
    homes = sample_sales(3).assign(remarks="neverbeforeseenword")
    terms_by_mode = []
    for mode in ("full_tfidf", "full_binary", "full_count"):
        base = make_text_base(mode, max_features=30, min_df=2)
        before = base.fit_transform(train, model="lightgbm")
        terms_by_mode.append(tuple(base.vectorizer_.get_feature_names_out()))
        after = base.transform(homes, model="lightgbm")
        assert list(before.columns) == list(after.columns)
        assert len([c for c in before if c.startswith("text_")]) <= 30
        assert (after.filter(like="text_").to_numpy() == 0).all()
    assert terms_by_mode[0] == terms_by_mode[1] == terms_by_mode[2]


@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_selected_tfidf_variant_preserves_existing_feature_matrix(track):
    train = sample_sales()
    homes = sample_sales(5)
    old = make_features("Xpt_spatiotemporal", model="lightgbm", track=track)
    new = make_features(
        "Xpt_spatiotemporal__selected_tfidf", model="lightgbm", track=track
    )
    pd.testing.assert_frame_equal(old.fit_transform(train), new.fit_transform(train))
    hidden = data.hide_answers(homes, track)
    pd.testing.assert_frame_equal(old.transform(hidden), new.transform(hidden))


@pytest.mark.parametrize("track", ["off_market", "list_aware"])
@pytest.mark.parametrize("variant", TEXT_VARIANTS)
def test_registered_variants_work_without_changing_factory(track, variant):
    train = sample_sales(100)
    homes = data.hide_answers(sample_sales(4), track)
    features = make_features(
        f"Xpt_spatiotemporal__{variant}", model="lightgbm", track=track
    )
    X = features.fit_transform(train)
    Y = features.transform(homes)
    assert list(X.columns) == list(Y.columns)
    assert "comp_log_price" in X
    assert ("log_list_price" in X) == (track == "list_aware")
    assert ("comp_vs_list_log" in X) == (track == "list_aware")
    assert not {"sold_price", "date_sold", "list_price"} & set(X.columns)
    assert not any(c.startswith("text_") for c in X) if variant in ("selected_tfidf", "no_text") else True


@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_cached_runner_matches_regular_model_factory(tmp_path, track):
    sales = sample_sales(260)
    months = ["2025-09"]
    train = sales.loc[sales.date_sold < pd.Timestamp("2025-09-01")]
    holdout = sales.loc[sales.date_sold.dt.to_period("M") == pd.Period("2025-09")]
    assert len(train) and len(holdout)
    params = {"n_estimators": 8, "n_jobs": 1}
    spec = [(LightGBMRegressor, "Xpt_spatiotemporal__full_binary", params)]
    if track == "off_market":
        model = OffMarketModel(spec, center_target=True)
    else:
        model = DirectListingModel(target="ratio", modelspecs=spec)
    model.fit(train)
    reference = model.predict(data.hide_answers(holdout, track))
    _, _, predictions = run_benchmark(
        backend="lightgbm", track=track, months=months,
        variants=["full_binary"], params=params, listings=sales,
        output_dir=tmp_path,
    )
    predicted = predictions.loc[
        predictions.variant == "full_binary", "predicted_price"
    ].to_numpy()
    np.testing.assert_allclose(predicted, reference, rtol=1e-6)
    assert (tmp_path / f"lightgbm_{track}_summary.csv").exists()
    assert (tmp_path / f"lightgbm_{track}_setup.json").exists()


def test_development_runner_blocks_final_holdout(tmp_path):
    with pytest.raises(ValueError, match="restricted"):
        run_benchmark(
            months=["2026-01"], listings=sample_sales(40),
            output_dir=tmp_path,
        )

@pytest.mark.parametrize("track", ["off_market", "list_aware"])
@pytest.mark.parametrize("backend", ["catboost", "lightgbm", "xgboost"])
@pytest.mark.parametrize("variant", TEXT_VARIANTS)
def test_text_variant_categorical_dtypes_match_backend(track, backend, variant):
    """Catch silent CatBoost defaults in off-market comparable/text wrappers."""
    train = sample_sales(95)
    homes = data.hide_answers(sample_sales(4), track)
    feature_name = f"Xpt_spatiotemporal__{variant}"
    features = make_features(feature_name, model=backend, track=track)
    fitted = features.fit_transform(train)
    future = features.transform(homes)
    for X in (fitted, future):
        assert list(X.columns) == list(fitted.columns)
        for name in (
            "house_type_name", "house_style", "municipality", "community", "garage"
        ):
            if backend == "catboost":
                assert pd.api.types.is_object_dtype(X[name].dtype)
            else:
                assert isinstance(X[name].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_xgboost_text_binary_runs_through_model_factory(track):
    """Exercise real XGBoost fit/predict, not only feature-frame construction."""
    from models.boost_regressor import XGBoostRegressor

    train = sample_sales(120)
    homes = data.hide_answers(sample_sales(5), track)
    spec = [(
        XGBoostRegressor,
        "Xpt_spatiotemporal__selected_binary",
        {"n_estimators": 5, "n_jobs": 1},
    )]
    model = (
        OffMarketModel(spec, center_target=True)
        if track == "off_market"
        else DirectListingModel(target="ratio", modelspecs=spec)
    )
    predictions = model.fit(train).predict(homes)
    assert predictions.shape == (len(homes),)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()
