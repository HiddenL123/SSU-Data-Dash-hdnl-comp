"""Isolated tests of GPBoost prediction accounting without optional gpboost."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

import models.spacetime_gpboost as mod


def test_gpboost_predictions_can_be_decomposed(monkeypatch):
    # This mock is only an API contract test. Real GPBoost covariance fitting
    # requires the optional library and real housing data.
    def homes(n, priced=False):
        df = pd.DataFrame({
            "latitude": [43.7] * n,
            "longitude": [-79.4] * n,
            "date_listed": pd.to_datetime(["2025-07-01"] * n),
            "type": ["condo"] * n,
        })
        if priced:
            df["sold_price"] = np.linspace(600000, 700000, n)
        return df

    class Provider:
        def fit_transform(self, train):
            return pd.DataFrame({"a": np.arange(len(train), dtype=float)})
        def transform(self, test):
            return pd.DataFrame({"a": np.arange(len(test), dtype=float)})

    monkeypatch.setattr(mod, "make_features", lambda *a, **kw: Provider())

    def booster_predict(*, data, gp_coords_pred=None, pred_latent=False,
                        predict_var=False, ignore_gp_model=False):
        size = len(data)
        fixed = np.full(size, 0.12)
        random_effect = np.full(size, -0.03)
        if ignore_gp_model:
            return fixed
        if pred_latent:
            return {"fixed_effect": fixed, "random_effect_mean": random_effect}
        return {"response_mean": fixed + random_effect}

    class GPModel:
        def __init__(self, **kwargs):
            assert kwargs["gp_approx"] == "vecchia"
        def dummy(self):
            pass

    def make_booster(*, train_set, gp_model, **kw):
        return SimpleNamespace(predict=booster_predict)

    monkeypatch.setitem(sys.modules, "gpboost", SimpleNamespace(
        GPModel=GPModel,
        Dataset=lambda *args, **kw: object(),
        train=make_booster,
    ))
    train = homes(8, priced=True)
    test = homes(3)
    price = mod.gpboost_joint(train, test, rounds=5)
    diag = mod.gpboost_joint(train, test, rounds=5, return_components=True)
    np.testing.assert_allclose(price, diag["price_combined"])
    np.testing.assert_allclose(diag["log_random_effect"], -.03)
    np.testing.assert_allclose(diag["log_fixed"] - diag["log_tree_only"], 0)
    np.testing.assert_allclose(
        diag["log_combined"], diag["log_fixed"] + diag["log_random_effect"])
    assert diag["log_decomposition_gap"] < 1e-12
    assert diag["log_tree_vs_fixed_gap"] < 1e-12
    assert diag["training_rows"] == 8
    assert diag["feature_count"] == 1


def test_gpboost_diagnostic_detects_inconsistent_latent_terms(monkeypatch):
    # The diagnostic must show a nonzero decomposition gap rather than hide
    # a changed response scale / package behavior.
    def provider():
        class P:
            def fit_transform(self, x):
                return pd.DataFrame({"x": np.ones(len(x))})
            def transform(self, x):
                return pd.DataFrame({"x": np.ones(len(x))})
        return P()

    monkeypatch.setattr(mod, "make_features", lambda *a, **kw: provider())

    def predict(*, data, pred_latent=False, ignore_gp_model=False, **kw):
        n = len(data)
        if ignore_gp_model:
            return np.zeros(n)
        if pred_latent:
            return {"fixed_effect": np.zeros(n),
                    "random_effect_mean": np.zeros(n)}
        return {"response_mean": np.ones(n)}

    monkeypatch.setitem(sys.modules, "gpboost", SimpleNamespace(
        GPModel=lambda **kw: object(),
        Dataset=lambda *args, **kw: object(),
        train=lambda **kw: SimpleNamespace(predict=predict),
    ))
    train = pd.DataFrame({
        "date_listed": pd.to_datetime(["2025-01-01"] * 3),
        "latitude": [43.7] * 3, "longitude": [-79.4] * 3,
        "sold_price": [100000., 120000., 110000.],
    })
    out = mod.gpboost_joint(
        train, train.drop(columns="sold_price"), rounds=2,
        return_components=True,
    )
    assert out["log_decomposition_gap"] == 1.
