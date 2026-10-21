"""Focused tests for XGBoost TreeSHAP on the modular housing model.

The SHAP plotting dependency is optional until the user runs `uv add shap`.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")

import data
from models.boost_regressor import XGBoostRegressor, LightGBMRegressor
from models.model import OffMarketModel
from models.shap_analysis import explain_offmarket_xgboost, global_importance
from test_starter import make_listings


def _sales(n=105):
    homes = make_listings(n, seed=9)
    homes["remarks"] = [
        "renovated kitchen family house" if i % 2
        else "original apartment investor opportunity"
        for i in range(n)
    ]
    return homes


def test_xgboost_shap_reconstructs_centered_log_price_and_labels_words():
    train = _sales()
    homes = data.hide_answers(_sales(7), "off_market")
    spec = [(
        XGBoostRegressor,
        "Xpt_spatiotemporal__selected_binary",
        {"n_estimators": 8, "max_depth": 3, "n_jobs": 1},
    )]
    model = OffMarketModel(spec, center_target=True).fit(train)
    explained, features, prices = explain_offmarket_xgboost(model, homes)
    assert explained.values.shape == (len(homes), features.shape[1])
    assert np.isfinite(explained.values).all()
    np.testing.assert_allclose(
        np.log(prices), model.predict_log(homes),
        rtol=2e-6, atol=2e-4,
    )
    assert any(name.startswith("remarks: ") for name in explained.feature_names)
    ranking = global_importance(explained, features)
    assert len(ranking) == len(features.columns)
    assert ranking.is_text.sum() == 47
    assert ranking.mean_abs_shap_log.ge(0).all()


def test_shap_helper_rejects_other_estimator_types():
    train = _sales()
    model = OffMarketModel([
        (LightGBMRegressor, "Xpt_spatiotemporal__selected_binary",
         {"n_estimators": 6, "n_jobs": 1})
    ], center_target=True).fit(train)
    with pytest.raises(ValueError, match="XGBoost"):
        explain_offmarket_xgboost(
            model, data.hide_answers(_sales(4), "off_market"),
        )
