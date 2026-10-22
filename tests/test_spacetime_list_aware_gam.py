"""Synthetic checks for list-aware ratio XGBoost + spline / Tensor GAM."""
import numpy as np
import pandas as pd
import pytest

import data
from models.boost_regressor import XGBoostRegressor
from models.direct_listing import DirectListingModel
from models.features import make_features
from spacetime_list_aware_gam import compare_list_aware_spline_gam
from test_starter import make_listings


def _sales():
    return make_listings(1100, seed=41)


def _reference(sales, months):
    specs = [(
        XGBoostRegressor, "Xpt_spatiotemporal__selected_binary",
        {"n_estimators": 6, "max_depth": 3, "n_jobs": 1,
         "tree_method": "hist", "random_state": 42,
         "enable_categorical": True},
    )]
    models, records = [], []
    for month in pd.PeriodIndex(months, freq="M"):
        earlier = sales.loc[sales.date_sold < month.start_time]
        held = sales.loc[sales.date_sold.dt.to_period("M") == month]
        assert len(held)
        model = DirectListingModel(
            target="ratio", modelspecs=specs
        ).fit(earlier)
        models.append([None, model])
        predicted = np.round(model.predict(
            data.hide_answers(held, "list_aware")
        ))
        records.append(pd.DataFrame({
            "ml_num": held.ml_num.to_numpy(),
            "month": str(month), "track": "list_aware",
            "sold_price": held.sold_price.to_numpy(),
            "predicted_price": predicted,
        }))
    return pd.concat(records, ignore_index=True), models


@pytest.mark.parametrize("suffix", [
    "__st_spacetime_splines",
    "__st_spacetime_splines+geo_context",
])
def test_spatial_feature_provider_for_list_aware(suffix):
    sales = _sales()
    train = sales.loc[sales.date_sold < "2025-08-01"]
    hold = sales.loc[sales.date_sold >= "2025-08-01"].head(10)
    hidden = data.hide_answers(hold, "list_aware")
    assert "list_price" in hidden and "sold_price" not in hidden
    provider = make_features(
        "Xpt_spatiotemporal__selected_binary" + suffix,
        model="xgboost", track="list_aware",
    )
    x_train = provider.fit_transform(train)
    x_held = provider.transform(hidden)
    assert x_train.columns.tolist() == x_held.columns.tolist()
    assert x_train.columns.is_unique
    assert len(x_train) == len(train) and len(x_held) == len(hold)
    assert "comp_vs_list_log" in x_held.columns
    assert not {"sold_price", "date_sold"} & set(x_held.columns)


def test_full_list_aware_experiment_and_resume(tmp_path, monkeypatch):
    import spacetime_list_aware_gam as runner
    sales = _sales()
    months = ("2025-12", "2026-01")
    baseline, models = _reference(sales, months)
    args = dict(
        months=months, listings=sales, output_dir=tmp_path,
        calibration_months=2, calibration_cap=25, bootstrap_repeats=15,
    )
    pooled, monthly, by_type, output = compare_list_aware_spline_gam(
        baseline, models, **args)
    assert len(pooled) == 5 and len(monthly) == 10
    assert set(pooled.variant) == {
        "baseline", "splines", "splines_geo",
        "splines_tensor_gam", "splines_geo_tensor_gam",
    }
    assert output.groupby(["month", "arm", "variant"]).size().nunique() == 1
    assert len(output) == 5 * len(baseline)
    assert np.isfinite(output.predicted_price).all()
    assert (tmp_path / "by_month" / "2025-11" /
            "feature__splines_geo.csv").is_file()
    assert (tmp_path / "pooled_summary.csv").is_file()
    for corrected, base in [
        ("splines_tensor_gam", "splines"),
        ("splines_geo_tensor_gam", "splines_geo"),
    ]:
        parent = pooled.loc[pooled.variant == base].iloc[0]
        child = pooled.loc[pooled.variant == corrected].iloc[0]
        assert child.parent_model == base
        assert child.delta_parent_mape_pp == pytest.approx(
            child.mape - parent.mape
        )
    # Re-run with no training allowed: all monthly feature models cached.
    def no_fit(self, train):
        raise AssertionError("A cached ratio XGBoost must not refit")
    monkeypatch.setattr(DirectListingModel, "fit", no_fit)
    cached = compare_list_aware_spline_gam(baseline, models, **args)
    np.testing.assert_allclose(pooled.mdape, cached[0].mdape)


def test_history_cutoffs_are_chronological(tmp_path, monkeypatch):
    sales = _sales()
    months = ("2025-12",)
    baseline, models = _reference(sales, months)
    origins = []
    actual = DirectListingModel.fit

    def checked_fit(self, train):
        origins.append(train.date_sold.max())
        return actual(self, train)

    monkeypatch.setattr(DirectListingModel, "fit", checked_fit)
    compare_list_aware_spline_gam(
        baseline, models, months=months, listings=sales,
        output_dir=tmp_path, calibration_months=2,
        calibration_cap=20, bootstrap_repeats=0,
    )
    assert len(origins) == 6   # two models × Oct/Nov calibration + Dec holdout
    assert max(origins) < pd.Timestamp("2025-12-01")


def test_different_settings_cannot_reuse_predictions(tmp_path):
    sales = _sales()
    months = ("2025-12",)
    baseline, models = _reference(sales, months)
    args = dict(months=months, listings=sales, output_dir=tmp_path,
                calibration_months=1, calibration_cap=25, bootstrap_repeats=0)
    compare_list_aware_spline_gam(baseline, models, **args)
    with pytest.raises(ValueError, match="different settings or models"):
        compare_list_aware_spline_gam(
            baseline, models, **{**args, "residual_strength": 0.75}
        )
