"""Synthetic end-to-end tests for full spline+GAM combinations.

The real full-history benchmark needs the local housing data and saved baseline.
"""
import numpy as np
import pandas as pd
import pytest

import data
from models.boost_regressor import XGBoostRegressor
from models.model import OffMarketModel
from models.spacetime_full_gam_comparison import compare_full_spline_gam
from test_starter import make_listings


def _reference(sales, months):
    spec = [(
        XGBoostRegressor, "Xpt_spatiotemporal__selected_binary",
        {"n_estimators": 6, "max_depth": 3, "n_jobs": 1,
         "tree_method": "hist", "enable_categorical": True,
         "random_state": 42},
    )]
    models, records = [], []
    for month in pd.PeriodIndex(months, freq="M"):
        historical = sales.loc[sales.date_sold < month.start_time]
        validation = sales.loc[sales.date_sold.dt.to_period("M") == month]
        assert len(validation)
        fitted = OffMarketModel(spec, center_target=True).fit(historical)
        models.append([fitted, None])
        pred = np.round(
            fitted.predict(data.hide_answers(validation, "off_market"))
        )
        records.append(pd.DataFrame({
            "ml_num": validation.ml_num.to_numpy(),
            "track": "off_market",
            "month": str(month),
            "sold_price": validation.sold_price.to_numpy(),
            "predicted_price": pred,
        }))
    return pd.concat(records, ignore_index=True), models


def _dataset():
    return make_listings(1100, seed=41)


def test_two_full_spline_gam_combinations_and_resume(tmp_path, monkeypatch):
    import models.spacetime_full_gam_comparison as runner

    sales = _dataset()
    months = ("2025-12", "2026-01")
    reference, models = _reference(sales, months)
    config = dict(
        months=months,
        listings=sales,
        output_dir=tmp_path / "gam",
        previous_feature_dir=None,
        calibration_months=2,
        calibration_cap=40,
        bootstrap_repeats=12,
    )
    pooled, monthly, by_type, predictions = compare_full_spline_gam(
        reference, models, **config
    )
    assert set(pooled.variant) == {
        "baseline", "splines", "splines_geo",
        "splines_tensor_gam", "splines_geo_tensor_gam",
    }
    assert len(monthly) == 10
    assert set(predictions.arm) == {"feature", "residual"}
    for corrected_name, parent in (
        ("splines_tensor_gam", "splines"),
        ("splines_geo_tensor_gam", "splines_geo"),
    ):
        child = pooled.loc[pooled.variant == corrected_name].iloc[0]
        base = pooled.loc[pooled.variant == parent].iloc[0]
        assert child.parent_model == parent
        assert child.delta_parent_mdape_pp == pytest.approx(
            child.mdape - base.mdape
        )
        assert child.delta_parent_mape_pp == pytest.approx(
            child.mape - base.mape
        )
    assert predictions.groupby(["arm", "variant", "month"]).size().nunique() == 1
    assert (predictions.predicted_price > 0).all()
    assert (tmp_path / "gam" / "by_month" / "2026-01" /
            "residual__splines_geo_tensor_gam.csv").is_file()
    assert (tmp_path / "gam" / "by_month" / "2025-11" /
            "base__splines.csv").is_file()
    assert (tmp_path / "gam" / "pooled_summary.csv").is_file()
    assert (tmp_path / "gam" / "house_type_summary.csv").is_file()

    class MustNotRefit:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Previously fitted full-history bases must be cached")
    monkeypatch.setattr(runner, "OffMarketModel", MustNotRefit)
    replay = compare_full_spline_gam(reference, models, **config)
    np.testing.assert_allclose(
        pooled.mdape.to_numpy(), replay[0].mdape.to_numpy(), rtol=0, atol=1e-9
    )


def test_calibration_uses_only_prior_sale_months(tmp_path, monkeypatch):
    import models.spacetime_full_gam_comparison as runner
    sales = _dataset()
    months = ("2025-12",)
    reference, models = _reference(sales, months)

    history_cutoffs = []
    actual_fit = runner.OffMarketModel.fit

    def inspected_fit(self, historical):
        history_cutoffs.append(historical.date_sold.max())
        return actual_fit(self, historical)

    monkeypatch.setattr(runner.OffMarketModel, "fit", inspected_fit)
    # Do not patch residual estimator itself: this exercises its real
    # nonlinear spline basis on out-of-time residuals.
    pooled, monthly, _, pred = compare_full_spline_gam(
        reference, models, months=months, listings=sales,
        output_dir=tmp_path / "one_month",
        previous_feature_dir=None,
        calibration_months=2, calibration_cap=25,
        bootstrap_repeats=0,
    )
    assert len(history_cutoffs) == 6  # Oct, Nov, Dec for each feature base
    assert max(history_cutoffs) < pd.Timestamp("2025-12-01")
    assert len(pooled) == 5
    assert np.isfinite(pred.predicted_price).all()


def test_saved_run_cannot_resume_with_modified_settings(tmp_path):
    sales = _dataset()
    months = ("2025-12",)
    reference, models = _reference(sales, months)
    config = dict(
        months=months, listings=sales, output_dir=tmp_path,
        previous_feature_dir=None, calibration_months=1,
        calibration_cap=20, bootstrap_repeats=0,
    )
    compare_full_spline_gam(reference, models, **config)
    with pytest.raises(ValueError, match="Previous spline-GAM run differs"):
        compare_full_spline_gam(
            reference, models, **{**config, "residual_strength": 0.7}
        )
