"""Deterministic, no-network tests for the separate list-aware candidate shotgun."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.listaware_shotgun_specs import (
    BLENDS, CANDIDATES, DEFAULT_CANDIDATES, ADJUSTED, SPATIAL, parameters,
)
from experiments.listaware_quick_experiment import (
    _check_setup, _fit_predict, _frame, _verified_cache,
    _validate_all, run_listaware_shotgun, summarize,
)


@pytest.fixture
def sales():
    rng = np.random.default_rng(234)
    n = 80 * 12
    cohorts = np.repeat(np.arange(12), 80)
    date_sold = pd.to_datetime("2024-01-15") + pd.to_timedelta(
        30 * cohorts, unit="D"
    )
    square = rng.integers(700, 2000, n)
    typ = np.where(np.arange(n) % 3 == 0, "Condo Apt", "Detached")
    market = np.exp(6.1 + np.log(square) + .14 * (typ == "Detached")
                    + .004 * cohorts + rng.normal(0, .05, n))
    listed = market * np.exp(rng.normal(.012, .06, n))
    return pd.DataFrame({
        "ml_num": [f"S{i:05}" for i in range(n)],
        "date_sold": date_sold, "date_listed": date_sold - pd.Timedelta(days=4),
        "sold_price": market, "list_price": listed,
        "house_area": square.astype(str), "house_type_name": typ,
        "latitude": 43.65 + rng.normal(0, .01, n),
        "longitude": -79.39 + rng.normal(0, .01, n),
        "bedrooms": rng.integers(1, 5, n), "washrooms": rng.integers(1, 4, n),
        "parking_total": rng.integers(0, 3, n),
        "municipality": "Toronto", "community": "Synthetic",
        "house_style": "Two-Storey", "garage": "None",
        "remarks": ["renovated family house"] * n,
    })


def _month_with_homes(sales, n=40):
    dates = pd.to_datetime(sales.date_sold)
    period = sorted(dates.dt.to_period("M").unique())[-1]
    held = sales.loc[dates.dt.to_period("M") == period].iloc[:n]
    train = sales.loc[sales.date_sold < period.start_time]
    return period, train, held


def test_candidate_registry_contains_previous_winners_and_cross_family_transfers():
    assert "cat_huber_baseline" in DEFAULT_CANDIDATES
    assert CANDIDATES["cat_huber_baseline"].loss == "Huber:delta=0.05"
    assert CANDIDATES["cat_huber_spline_geo"].feature_name == SPATIAL
    assert CANDIDATES["cat_huber_adjusted"].feature_name == ADJUSTED
    assert CANDIDATES["xgb_spline_geo"].feature_name == SPATIAL
    assert CANDIDATES["xgb_adjusted"].feature_name == ADJUSTED
    assert CANDIDATES["lgb_spline_geo"].feature_name == SPATIAL
    assert CANDIDATES["cat_huber_direct"].target == "direct"
    assert CANDIDATES["cat_huber_baseline"].target == "ratio"
    assert {"catboost", "lightgbm", "xgboost"} <= {x.backend for x in CANDIDATES.values()}


def test_saved_best_hyperparameters_retained_and_pilot_caps_reversible():
    cat = parameters(CANDIDATES["cat_huber_baseline"], trees=None)
    xgb = parameters(CANDIDATES["xgb_tuned_binary"], trees=None)
    light = parameters(CANDIDATES["lgb_tuned_binary"], trees=None)
    assert cat["iterations"] == 900
    assert xgb["n_estimators"] == 700
    assert light["n_estimators"] == 1800
    assert parameters(CANDIDATES["cat_huber_baseline"], trees=20)["iterations"] == 20
    assert parameters(CANDIDATES["xgb_spline_geo"], trees=20)["n_estimators"] == 20


def test_dev_validation_rejects_missing_baseline_and_future_months():
    with pytest.raises(ValueError, match="Include cat_huber_baseline"):
        _check_setup(["2025-08"], ["xgb_spline_geo"], 100, 50, 50, False)
    with pytest.raises(ValueError, match="chronological"):
        _check_setup(["2025-12"], ["cat_huber_baseline"], 100, 50, 50, False)
    with pytest.raises(ValueError, match="700"):
        _check_setup(["2025-08"], ["cat_huber_baseline"], 100, 50, 100, True)


def test_log_blend_geometric_mean_is_well_defined_and_weights_frozen():
    assert BLENDS["blend_cat_xgb_spatial"] == {
        "cat_huber_spline_geo": .70, "xgb_spline_geo": .30
    }
    assert sum(BLENDS["blend_cat_xgb_lgb"].values()) == pytest.approx(1.0)


def test_previously_trained_models_never_receive_holdout_labels(monkeypatch, sales):
    from experiments.listaware_quick_experiment import _fit_predict
    from models.listaware_shotgun_specs import Candidate
    import experiments.listaware_quick_experiment as mod
    period, history, hold = _month_with_homes(sales, 5)
    hidden = hold.drop(columns=["sold_price", "date_sold"])
    observed = []

    class FakeEstimator:
        def __init__(self, **params):
            pass

        def fit(self, X, y, cat_features=None):
            observed.append(("fit", len(y)))
            return self

        def predict(self, X):
            observed.append(("predict", set(X.columns)))
            return np.zeros(len(X))

    class FakeProvider:
        def fit_transform(self, train):
            assert "sold_price" in train and "date_sold" in train
            return pd.DataFrame({"safe_feature": np.ones(len(train))}, index=train.index)

        def transform(self, homes):
            assert "sold_price" not in homes and "date_sold" not in homes
            return pd.DataFrame({"safe_feature": np.ones(len(homes))}, index=homes.index)

    monkeypatch.setattr(mod, "make_features", lambda *args, **kw: FakeProvider())
    # The actual CatBoost import is inside _fit_predict. Test execution with
    # its own native learner on a tiny train cap rather than bypassing fit.
    chosen = Candidate("tiny", "catboost", "Xpt", loss="RMSE")
    # Synthetic predictions should reconstruct sale price by multiplying ratio
    # prediction (zero => 1) by the listed price.
    monkeypatch.setattr(__import__("catboost"), "CatBoostRegressor", FakeEstimator)
    prediction = _fit_predict(
        chosen, history, hidden, tree_cap=3, threads=1, features_cache={}
    )
    np.testing.assert_allclose(prediction, hidden.list_price.to_numpy(float))
    assert any(entry[0] == "fit" for entry in observed)
    assert all("sold_price" not in entry[1] for entry in observed
               if entry[0] == "predict")


def test_paired_scores_require_exactly_matching_homes_and_labels(sales):
    period, _, held = _month_with_homes(sales, 15)
    base = _frame(held, period, "model", "cat_huber_baseline",
                  held.sold_price.to_numpy() * 1.04)
    alternative = _frame(held, period, "model", "xgb_spline_geo",
                         held.sold_price.to_numpy() * 1.025)
    blend = _frame(held, period, "blend", "blend_cat_xgb_spatial",
                   held.sold_price.to_numpy() * 1.027)
    predictions = pd.concat([base, alternative, blend])
    pooled, monthly, types = summarize(predictions, repeats=10)
    assert len(pooled) == 3
    assert len(monthly) == 3
    assert pooled.iloc[0].mdape < pooled.loc[
        pooled.variant == "cat_huber_baseline", "mdape"
    ].iloc[0]
    damaged = alternative.copy()
    damaged.loc[damaged.index[0], "sold_price"] *= 1.01
    with pytest.raises(ValueError, match="Different actual prices"):
        _validate_all(pd.concat([base, damaged]))


def test_cache_refuses_changed_row_alignment_or_prices(tmp_path, sales):
    period, _, held = _month_with_homes(sales, 12)
    stored = _frame(held, period, "model", "cat_huber_baseline",
                    held.sold_price.to_numpy() * 1.04)
    path = tmp_path / "member.csv"
    stored.to_csv(path, index=False)
    assert len(_verified_cache(path, held, period, "model", "cat_huber_baseline")) == 12
    with pytest.raises(ValueError, match="stale"):
        _verified_cache(path, held.iloc[::-1], period, "model", "cat_huber_baseline")
    tampered = pd.read_csv(path)
    tampered.loc[0, "sold_price"] *= 1.3
    tampered.to_csv(path, index=False)
    with pytest.raises(ValueError, match="stale"):
        _verified_cache(path, held, period, "model", "cat_huber_baseline")


def test_interrupted_listaware_run_can_resume_per_model(tmp_path, monkeypatch, sales):
    import experiments.listaware_quick_experiment as runner
    period, _, _ = _month_with_homes(sales)
    counter = []
    def fake_fit(candidate, train, homes, **kw):
        assert (train.date_sold < period.start_time).all()
        assert "sold_price" not in homes and "date_sold" not in homes
        counter.append(candidate.name)
        return homes.list_price.to_numpy(float) * (
            1.01 if candidate.name == "cat_huber_baseline" else 1.02
        )
    monkeypatch.setattr(runner, "_fit_predict", fake_fit)
    opts = dict(
        months=[str(period)], candidates=[
            "cat_huber_baseline", "xgb_tuned_binary"
        ], listings=sales, output_dir=tmp_path / "exp",
        train_cap=150, validation_cap=25, trees=10,
        bootstrap_repeats=10, with_blends=True,
    )
    p1, _, _, _ = run_listaware_shotgun(**opts)
    assert len(counter) == 2
    p2, _, _, _ = run_listaware_shotgun(**opts)
    assert len(counter) == 2
    pd.testing.assert_frame_equal(
        p1.reset_index(drop=True), p2.reset_index(drop=True),
        check_exact=False, atol=1e-9
    )
    assert "blend_cat_baseline_xgb" in p1.variant.tolist()
    with pytest.raises(ValueError, match="NEW output"):
        run_listaware_shotgun(**{**opts, "trees": 11})


@pytest.mark.parametrize("name", ["cat_huber_spline_geo", "xgb_spline_geo"])
def test_real_cross_family_spline_geo_fit_and_predict_with_hidden_homes(sales, name):
    """Small real fit to catch CatBoost/XGBoost categorical and feature mismatches."""
    import data
    period, historical, held = _month_with_homes(sales, 10)
    train = historical.iloc[-200:].copy()
    hidden = data.hide_answers(held, "list_aware")
    prediction = _fit_predict(
        CANDIDATES[name], train, hidden, tree_cap=8,
        threads=1, features_cache={}
    )
    assert prediction.shape == (len(held),)
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()


def test_catboost_and_xgboost_share_engineered_feature_names_but_correct_dtypes(sales):
    import data
    from features.features import make_features
    _, history, held = _month_with_homes(sales, 5)
    train = history.iloc[-140:].copy()
    hidden = data.hide_answers(held, "list_aware")
    for name in ("cat_huber_spline_geo", "xgb_spline_geo"):
        candidate = CANDIDATES[name]
        provider = make_features(candidate.feature_name,
                                 model=candidate.backend, track="list_aware")
        X = provider.fit_transform(train)
        H = provider.transform(hidden)
        assert X.columns.equals(H.columns)
        assert "log_list_price" in X and "comp_vs_list_log" in X
        assert any(c.startswith("extra_tensor_") for c in X.columns)
        assert "sold_price" not in H and "date_sold" not in H
        assert X.index.equals(train.index) and H.index.equals(hidden.index)
        if candidate.backend == "catboost":
            assert H["house_type_name"].dtype == object
        else:
            assert isinstance(H["house_type_name"].dtype, pd.CategoricalDtype)
