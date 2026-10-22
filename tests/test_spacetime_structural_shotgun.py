"""Model organization, chronology and reproducibility tests for structural pilot.

Tests use synthetic housing sales and tiny budgets. GPBoost is optional.
"""
import numpy as np
import pandas as pd
import pytest

from models.spacetime_structural_features import (
    FEATURE_SET, SPLINE_GEO_SUFFIX, coords,
)
from models.spacetime_structural_residuals import local_kernel, matern_gp
from spacetime_structural_shotgun import run_structural_shotgun
from test_starter import make_listings


def test_structural_feature_coordinates_ignore_outcome():
    homes = make_listings(10)
    original = coords(homes)
    changed = homes.assign(sold_price=9.99e9, list_price=1.0)
    np.testing.assert_array_equal(original, coords(changed))
    assert original.shape == (10, 3)
    assert np.isfinite(original).all()
    assert SPLINE_GEO_SUFFIX.startswith("__st_")
    assert FEATURE_SET.endswith("selected_binary")


def test_spatial_kernel_and_matern_finite_predictions():
    sales = make_listings(150, seed=31)
    historical = sales.loc[:100].copy()
    future = sales.loc[101:114].copy()
    # Residual labels are provided by the calling benchmark; the methods
    # deliberately cannot read the observed sale prices themselves.
    y = np.linspace(-0.05, 0.05, len(historical))
    predictions = [
        local_kernel(historical, y, future),
        local_kernel(historical, y, future, same_type=True),
        matern_gp(historical, y, future, spatial_only=True, max_points=55),
        matern_gp(historical, y, future, spatial_only=False, max_points=55),
    ]
    for p in predictions:
        assert p.shape == (len(future),)
        assert np.isfinite(p).all()


def test_structural_shotgun_resume_without_catboost_or_gpboost(
    tmp_path, monkeypatch
):
    """One run is possible without gpboost; successful candidates resume."""
    import spacetime_structural_shotgun as shot
    sales = make_listings(1100, seed=101)
    months = ("2025-10", "2025-11")
    calls = []

    def fake_xgb(training, homes, *, suffix, params, trees):
        # Stand-in for the fitted tuned XGBoost; predict from earlier prices
        # so that the integration test stays cheap.
        assert suffix == SPLINE_GEO_SUFFIX
        assert "sold_price" not in homes
        calls.append(len(training))
        return np.full(len(homes), np.median(training.sold_price)), 1

    def missing_gpboost(*args, **kwargs):
        raise ImportError("Simulated optional gpboost absence")

    monkeypatch.setattr(shot, "predict_xgb", fake_xgb)
    monkeypatch.setattr(shot, "gpboost_joint", missing_gpboost)
    settings = dict(
        months=months, train_start="2025-01-01",
        train_cap=150, validation_cap=11, calibration_cap=12,
        calibration_folds=1, trees=6, max_gp_points=35,
        gpboost_rounds=6, bootstrap_repeats=15,
        output_dir=tmp_path, listings=sales, resume=True,
    )
    pooled, monthly, predictions, failed = run_structural_shotgun(**settings)
    assert len(calls) >= 3
    assert set(pooled.variant) == {
        "baseline", "xgb_tensor_gam", "xgb_st_kernel",
        "xgb_st_kernel_type", "xgb_spatial_matern", "xgb_st_matern",
    }
    assert set(failed.variant) == {"gpboost_vecchia"}
    assert len(monthly) == len(pooled) * len(months)
    assert "delta_vs_tensor_mdape_pp" in pooled
    assert pooled.loc[
        pooled.variant == "xgb_tensor_gam", "delta_vs_tensor_mdape_pp"
    ].iloc[0] == pytest.approx(0)

    def should_not_refit(*args, **kwargs):
        raise AssertionError("Resumed predictions should not refit models")

    monkeypatch.setattr(shot, "predict_xgb", should_not_refit)
    # Skip the deliberately unavailable optional candidate on rerun:
    # successful variants remain cached. Use the same settings/manifest
    # so no changes to the candidate list are permitted.
    pooled2, _, _, failed2 = run_structural_shotgun(**settings)
    np.testing.assert_allclose(
        pooled.mdape.to_numpy(), pooled2.mdape.to_numpy()
    )
    assert len(failed2) == 2
    with pytest.raises(ValueError, match="different data/settings"):
        run_structural_shotgun(
            **{**settings, "train_cap": 151},
        )
