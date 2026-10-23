"""Tests for the reduced chronological space-time screening workflow."""
import numpy as np
import pandas as pd
import pytest

import data
from models.features import make_features
from models.spacetime_screen_features import SpaceTimeAddon, FEATURE_VARIANTS
from models.spacetime_screen_residuals import SpaceTimeResidual, RESIDUAL_VARIANTS
from spacetime_fast import (
    run_shotgun, _calibration_data, _train_and_predict, SCREEN_FEATURE_VARIANTS
)
from test_starter import make_listings


def _sales(n=900):
    homes = make_listings(n, seed=13)
    homes["remarks"] = np.where(
        np.arange(len(homes)) % 2 == 0,
        "newly renovated beautiful unit",
        "as is original house needs work",
    )
    return homes


def test_feature_registry_has_multiple_non_linear_families():
    assert "baseline" in SCREEN_FEATURE_VARIANTS
    assert {"comp_age_first", "comp_type_quota", "comp_both"}.issubset(
        SCREEN_FEATURE_VARIANTS
    )
    assert {"multiscale", "market", "time_adjusted_comp",
            "spacetime_rbf", "spacetime_splines"}.issubset(
        FEATURE_VARIANTS
    )
    assert {"knn", "local_kernel", "rbf_ridge", "spline_gam",
            "tensor_gam", "extra_trees", "hist_boost", "small_gp"}.issubset(
        RESIDUAL_VARIANTS
    )


@pytest.mark.parametrize("variant", [
    "market", "multiscale", "time_adjusted_comp",
    "spacetime_rbf", "spacetime_splines", "combined"
])
def test_addons_match_feature_schema_and_do_not_use_future_prices(variant):
    sales = _sales(100)
    train = sales.loc[sales.date_sold < pd.Timestamp("2025-11-01")]
    target = sales.loc[sales.date_sold >= pd.Timestamp("2025-11-01")].iloc[:3]
    assert len(train) > 10 and len(target) > 0
    # Base comparable-sale features must be fitted separately per fold.
    f = make_features(
        "Xpt_spatiotemporal__selected_binary", model="xgboost"
    )
    X = f.fit_transform(train)
    hidden = data.hide_answers(target, "off_market")
    Z = f.transform(hidden)
    add = SpaceTimeAddon(variant).fit(train)
    tr = add.transform(train, X)
    val = add.transform(hidden, Z)
    changed_labels = target.assign(sold_price=1e12)
    changed = add.transform(data.hide_answers(changed_labels, "off_market"), Z)
    assert list(tr.columns) == list(val.columns)
    assert len(val.columns) > 0
    pd.testing.assert_frame_equal(val, changed)
    assert set(val.columns).isdisjoint(X.columns)


@pytest.mark.parametrize("variant", ["knn", "rbf_ridge", "tensor_gam", "small_gp"])
def test_residual_models_produce_finite_predictions(variant):
    sales = _sales(80)
    train = sales.iloc[:65].copy()
    future = sales.iloc[65:].copy()
    # Align the prediction month strictly after training-origin months.
    future["date_listed"] = pd.Timestamp("2025-12-01")
    y = np.sin(np.arange(len(train))) * 0.03
    m = SpaceTimeResidual(variant, max_gp_points=25).fit(train, y)
    pred = m.predict(future)
    assert pred.shape == (len(future),)
    assert np.isfinite(pred).all()


def test_calibration_fits_only_past_sales(monkeypatch):
    import spacetime_fast as module
    sales = _sales(900)
    seen = []

    def fake_base(train, homes, params, screen_trees, **kwargs):
        # No training sale may occur in or after the OOT calibration month.
        first_calibration_sale_month = sales.loc[homes.index, "date_sold"].dt.to_period("M").min()
        assert (train["date_sold"] < first_calibration_sale_month.start_time).all()
        seen.append(first_calibration_sale_month)
        return np.full(len(homes), 800000.0)

    monkeypatch.setattr(module, "_train_and_predict", fake_base)
    data_frames, labels = _calibration_data(
        sales,
        origin=pd.Period("2025-10", freq="M"),
        train_start=pd.Timestamp("2025-01-01"),
        train_cap=100, calibration_cap=20, calibration_folds=2,
        params={}, screen_trees=5, seed=1,
    )
    assert len(data_frames) == len(labels)
    assert set(seen) == {pd.Period("2025-08"), pd.Period("2025-09")}


def test_reduced_shotgun_runs_with_tiny_tree_budget(tmp_path):
    sales = _sales(900)
    summary, predictions, failures = run_shotgun(
        validation_month="2025-10", train_start="2025-01-01",
        train_cap=110, validation_cap=16, calibration_cap=18,
        calibration_folds=2, screen_trees=8,
        features=("baseline", "spacetime_rbf", "comp_age_first"),
        residuals=("knn", "rbf_ridge"),
        listings=sales, output_dir=tmp_path,
    )
    assert not failures
    assert set(summary.variant) == {
        "baseline", "spacetime_rbf", "comp_age_first", "knn", "rbf_ridge"
    }
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "predictions.csv").is_file()
    assert (tmp_path / "setup.json").is_file()
    assert (predictions.predicted_price > 0).all()
    assert predictions.groupby("variant").size().nunique() == 1


def test_screen_rejects_confirmatory_dates(tmp_path):
    with pytest.raises(ValueError, match="before 2025-12"):
        run_shotgun(
            validation_month="2026-02",
            features=("baseline",), residuals=(), listings=_sales(20),
            output_dir=tmp_path,
        )

def test_screen_control_matches_existing_centered_offmarket_model():
    """Guard against a different target/feature encoding in the fast runner."""
    from models.boost_regressor import XGBoostRegressor
    from models.model import OffMarketModel
    from spacetime_fast import _make_estimator

    sales = _sales(160)
    train = sales.loc[sales.date_sold < pd.Timestamp("2025-11-01")]
    future = data.hide_answers(
        sales.loc[sales.date_sold >= pd.Timestamp("2025-11-01")].iloc[:5],
        "off_market",
    )
    assert len(train) > 30 and not future.empty
    params = {"n_estimators": 10, "max_depth": 3, "n_jobs": 4}
    reference = OffMarketModel([
        (XGBoostRegressor, "Xpt_spatiotemporal__selected_binary", params),
    ], center_target=True).fit(train).predict(future)
    screen = _train_and_predict(
        train, future, params, screen_trees=10
    )
    np.testing.assert_allclose(screen, reference, rtol=3e-5, atol=1)
