"""Synthetic checks for deterministic, leakage-safe GP kernel selection."""
import numpy as np
import pandas as pd
import pytest

from models.spacetime_structural_residuals import matern_gp
from test_starter import make_listings
import spacetime_matern_tuning as tuning


def test_matern_screen_defaults_preserved_and_scale_parameters():
    sales = make_listings(120, seed=5)
    train, test = sales.iloc[:80], sales.iloc[80:92]
    residual = np.linspace(-0.10, 0.10, len(train))
    original = matern_gp(train, residual, test, max_points=50)
    explicit = matern_gp(
        train, residual, test, max_points=50,
        spatial_km=5.0, temporal_months=2.0,
        amplitude_var=0.003, noise_var=0.012,
    )
    np.testing.assert_allclose(original, explicit)
    for bad in (
        {"spatial_km": 0},
        {"temporal_months": -1},
        {"amplitude_var": 0},
        {"noise_var": 0},
    ):
        with pytest.raises(ValueError):
            matern_gp(train, residual, test, max_points=50, **bad)
    assert np.isfinite(matern_gp(
        train, residual, test, max_points=50,
        spatial_only=False, temporal_months=9,
    )).all()


def test_matern_tuner_reuses_oot_xgb_and_confirms_later(tmp_path, monkeypatch):
    sales = make_listings(1700, seed=111)
    fit_calls = []

    def fake_tuned_xgb(train, homes, *, suffix, params, trees):
        assert suffix == tuning.SPLINE_GEO_SUFFIX
        assert "sold_price" not in homes.columns
        assert pd.to_datetime(train.date_sold).max() < (
            pd.to_datetime(homes.date_sold).min()
            if "date_sold" in homes.columns
            else pd.Timestamp("2100-01-01")
        )
        fit_calls.append(len(train))
        return np.full(len(homes), np.median(train.sold_price)), 1

    monkeypatch.setattr(tuning, "predict_xgb", fake_tuned_xgb)
    settings = {
        "tune_months": ("2025-04", "2025-05"),
        "train_start": "2025-01-01", "train_cap": 100,
        "validation_cap": 17, "calibration_cap": 15,
        "calibration_folds": 2, "trees": 6, "max_gp_points": 20,
        "seed": 42, "output_dir": tmp_path, "resume": True,
        "listings": sales,
    }
    trials, meta = tuning.run_matern_tuning(
        **settings, trials_per_family=2,
    )
    assert set(trials.kind) == {"spatial", "spacetime"}
    assert len(trials) == 4
    assert set(meta["best"]) == {"spatial", "spacetime"}
    assert len(fit_calls) >= 4
    assert (tmp_path / "best_matern.json").is_file()

    def do_not_refit(*args, **kwargs):
        raise AssertionError("Unchanged XGBoost folds must load cached predictions")

    monkeypatch.setattr(tuning, "predict_xgb", do_not_refit)
    # A repeated tuning run must read cached XGBoost results.
    trial_again, _ = tuning.run_matern_tuning(
        **settings, trials_per_family=2,
    )
    np.testing.assert_allclose(
        trials.objective.to_numpy(), trial_again.objective.to_numpy()
    )

    # May confirmation would overlap the tuning selection months.
    with pytest.raises(ValueError, match="strictly later"):
        tuning.run_matern_confirmation(
            **settings, confirm_months=("2025-05",),
            bootstrap_repeats=10,
        )

    # Fresh confirmation month June is allowed; it needs its own new
    # validation XGBoost fit, so temporarily restore the fake.
    monkeypatch.setattr(tuning, "predict_xgb", fake_tuned_xgb)
    pooled, monthly, predictions = tuning.run_matern_confirmation(
        **settings, confirm_months=("2025-06",),
        bootstrap_repeats=10,
    )
    assert len(pooled) == 4
    assert set(pooled.variant) == {
        "baseline", "xgb_tensor_gam",
        "xgb_spatial_matern_tuned", "xgb_spacetime_matern_tuned",
    }
    assert len(monthly) == 4
    assert "delta_vs_tensor_mdape_pp" in pooled
    assert (tmp_path / "confirmation_predictions.csv").is_file()
    assert (tmp_path / "confirmation_pooled.csv").is_file()
    assert predictions.groupby("variant").size().nunique() == 1

    with pytest.raises(ValueError, match="different data/settings"):
        tuning.run_matern_tuning(
            **{**settings, "trees": 7}, trials_per_family=1,
        )
