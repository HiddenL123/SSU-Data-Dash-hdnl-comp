"""Full-strength shortlist integration tests on small synthetic housing data.

These use tiny XGBoost fits only. The genuine six-month housing benchmark
remains a local notebook run; no external data is downloaded in tests.
"""
import numpy as np
import pandas as pd
import pytest

import data
from models.boost_regressor import XGBoostRegressor
from models.model import OffMarketModel
from experiments.spacetime_full_validation import validate_spacetime_shortlist
from test_starter import make_listings


def _data():
    sales = make_listings(1400, seed=33)
    sales["remarks"] = np.where(
        np.arange(len(sales)) % 2 == 0,
        "renovated turnkey home",
        "as is older original unit",
    )
    return sales


def _make_reference(sales, months, *, fixed_origin=None):
    spec = [(
        XGBoostRegressor,
        "Xpt_spatiotemporal__selected_binary",
        {"n_estimators": 8, "max_depth": 3, "n_jobs": 1,
         "tree_method": "hist", "random_state": 42,
         "enable_categorical": True},
    )]
    models = []
    records = []
    if fixed_origin is not None:
        train = sales.loc[sales.date_sold < fixed_origin]
        shared = OffMarketModel(spec, center_target=True).fit(train)
        models.append([shared, None])
    for month in pd.PeriodIndex(months, freq="M"):
        held = sales.loc[sales.date_sold.dt.to_period("M") == month]
        if fixed_origin is None:
            train = sales.loc[sales.date_sold < month.start_time]
            current = OffMarketModel(spec, center_target=True).fit(train)
            models.append([current, None])
        else:
            current = shared
        predicted = current.predict(data.hide_answers(held, "off_market"))
        records.append(pd.DataFrame({
            "ml_num": held.ml_num.to_numpy(),
            "month": str(month), "track": "off_market",
            "sold_price": held.sold_price.to_numpy(),
            "predicted_price": np.round(predicted),
        }))
    return pd.concat(records, ignore_index=True), models


def test_expand_shortlist_runs_and_resumes_without_rerunning(tmp_path, monkeypatch):
    import experiments.spacetime_full_validation as module

    sales = _data()
    months = ("2025-12", "2026-01")
    predictions, fitted = _make_reference(sales, months)
    pooled, monthly, by_type, out = validate_spacetime_shortlist(
        predictions, fitted, months=months,
        calibration_months=2, calibration_cap=40,
        bootstrap_repeats=20, listings=sales, output_dir=tmp_path,
    )
    assert set(pooled.variant) == {
        "baseline", "comp_age_first", "tensor_gam", "rbf_ridge"
    }
    assert len(monthly) == 8
    assert out.groupby(["month", "arm", "variant"]).size().nunique() == 1
    assert (out.predicted_price > 0).all()
    assert (tmp_path / "calibration" / "2025-10.csv").is_file()
    assert (tmp_path / "calibration" / "2025-11.csv").is_file()
    assert (tmp_path / "calibration" / "2025-12.csv").is_file()
    assert (tmp_path / "pooled_summary.csv").is_file()
    assert (tmp_path / "house_type_summary.csv").is_file()

    # All monthly results already exist and have matching signatures.
    def forbidden_calibration(*args, **kwargs):
        raise AssertionError("Cached rerun must not refit calibration models")

    monkeypatch.setattr(module, "_calibration_snapshot", forbidden_calibration)
    again = validate_spacetime_shortlist(
        predictions, fitted, months=months,
        calibration_months=2, calibration_cap=40,
        bootstrap_repeats=20, listings=sales, output_dir=tmp_path,
        resume=True,
    )
    np.testing.assert_allclose(pooled.mdape, again[0].mdape, atol=1e-7)


def test_fixed_origin_shortlist_reuses_frozen_model(tmp_path):
    sales = _data()
    months = ("2025-12", "2026-01")
    reference, models = _make_reference(
        sales, months, fixed_origin=pd.Timestamp("2025-12-01")
    )
    pooled, monthly, _, _ = validate_spacetime_shortlist(
        reference, models, months=months, mode="fixed_origin",
        train_cutoff="2025-12-01", calibration_months=2,
        calibration_cap=25, bootstrap_repeats=0,
        output_dir=tmp_path, listings=sales,
    )
    assert len(pooled) == 4
    assert len(monthly) == 8
    assert (tmp_path / "calibration" / "2025-11.csv").is_file()
    # Frozen origin must NEVER collect December residual targets.
    assert not (tmp_path / "calibration" / "2025-12.csv").exists()


def test_shortlist_rejects_wrong_reference_and_stale_settings(tmp_path):
    sales = _data()
    months = ("2025-12",)
    reference, models = _make_reference(sales, months)
    changed = reference.copy()
    changed.loc[changed.index[0], "predicted_price"] += 100000
    with pytest.raises(ValueError, match="do not reproduce"):
        validate_spacetime_shortlist(
            changed, models, months=months, calibration_months=1,
            listings=sales, output_dir=tmp_path,
        )
    validate_spacetime_shortlist(
        reference, models, months=months, calibration_months=1,
        calibration_cap=20, bootstrap_repeats=0,
        listings=sales, output_dir=tmp_path, resume=False,
    )
    with pytest.raises(ValueError, match="different settings"):
        validate_spacetime_shortlist(
            reference, models, months=months, calibration_months=1,
            calibration_cap=40, bootstrap_repeats=0,
            listings=sales, output_dir=tmp_path, resume=True,
        )
