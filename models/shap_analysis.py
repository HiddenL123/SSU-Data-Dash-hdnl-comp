"""Explain one fitted off-market XGBoost model in log-price units.

Designed for the SSU Data Dash modular model, including pandas native
categoricals and text-feature names. Uses XGBoost's own exact TreeSHAP
contributions; SHAP's Explanation class is used for plotting.

No training, refitting, vocabulary selection, or validation-label access occurs
inside this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def readable_feature_names(provider, columns):
    """Resolve numbered text columns back to training-fitted words/ngrams."""
    mappings = {}
    queue = [provider]
    seen = set()
    while queue:
        node = queue.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        mapping = getattr(node, "text_feature_names", None)
        if isinstance(mapping, dict):
            mappings.update(mapping)
        for attribute in ("provider", "base_provider", "state"):
            child = getattr(node, attribute, None)
            if child is not None:
                queue.append(child)
    return [f"remarks: {mappings[col]}" if col in mappings else col
            for col in columns]


def _plot_data(frame):
    """Give SHAP plotting numeric colors without altering estimator features.

    Ordinal codes for categoricals are *display-only*. Their numeric magnitude
    has no model interpretation, and they never enter the DMatrix.
    """
    numeric = frame.copy()
    for column in numeric.columns:
        if isinstance(numeric[column].dtype, pd.CategoricalDtype):
            encoded = numeric[column].cat.codes.to_numpy(dtype=float)
            encoded[encoded < 0] = np.nan
            numeric[column] = encoded
    try:
        return numeric.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Expected numeric or pandas categorical XGBoost input features"
        ) from exc


def explain_offmarket_xgboost(model, homes, *, check_additivity=True):
    """Return (SHAP Explanation, true feature frame, predicted prices).

    Parameters
    ----------
    model : fitted models.model.OffMarketModel
        Currently supports one XGBoost stage and no additional ARIMA stage,
        so the explanation is complete, not merely a partial decomposition.
    homes : DataFrame
        Prediction rows with labels hidden (e.g. data.hide_answers(...)).
    check_additivity : bool
        Verify XGBoost contributions reconstruct model log-price predictions.

    SHAP values are in log(sold_price), not dollars, and the target offset is
    added to the base value (not to every feature's contribution).
    """
    import shap
    import xgboost as xgb

    if not getattr(model, "_fitted", False):
        raise ValueError("Load or fit a trained model before explaining it")
    if not hasattr(model, "target_offset_"):
        raise TypeError("This helper expects an OffMarketModel, not a ratio model")
    if len(model.components) != 1:
        raise ValueError(
            "Only one-stage off-market XGBoost models are supported. "
            "An ARIMA/residual stage needs its own additional explanation."
        )
    component, provider = model.components[0]
    if getattr(component, "feature_backend", None) != "xgboost":
        raise ValueError("Expected an XGBoost component")
    if not hasattr(component.estimator, "get_booster"):
        raise ValueError("Estimator does not expose an XGBoost Booster")
    if homes.empty:
        raise ValueError("Provide at least one validation home")

    X = provider.transform(homes)
    booster = component.estimator.get_booster()
    if booster.feature_names and list(booster.feature_names) != list(X.columns):
        raise ValueError("Transformed feature ordering differs from fitted XGBoost")
    matrix = xgb.DMatrix(X, enable_categorical=True)
    contribs = np.asarray(
        booster.predict(matrix, pred_contribs=True, validate_features=True),
        dtype=float,
    )
    if contribs.shape != (len(X), X.shape[1] + 1):
        raise ValueError(f"Unexpected TreeSHAP shape: {contribs.shape}")

    offset = float(model.target_offset_)
    values = contribs[:, :-1]
    base_values = contribs[:, -1] + offset
    predictions_log = base_values + values.sum(axis=1)

    if check_additivity:
        margin = np.asarray(
            booster.predict(matrix, output_margin=True, validate_features=True),
            dtype=float,
        ).reshape(-1) + offset
        np.testing.assert_allclose(
            predictions_log, margin, rtol=2e-6, atol=2e-4,
            err_msg="TreeSHAP values do not reconstruct XGBoost margins",
        )
        np.testing.assert_allclose(
            predictions_log, model.predict_log(homes),
            rtol=2e-6, atol=2e-4,
            err_msg="TreeSHAP decomposition differs from the saved model output",
        )

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=_plot_data(X),
        feature_names=readable_feature_names(provider, X.columns),
    )
    return explanation, X, np.exp(predictions_log)


def global_importance(explanation, X):
    """Mean absolute log-price SHAP values, including a flag for text columns."""
    values = np.asarray(explanation.values, dtype=float)
    if values.shape[1] != len(X.columns):
        raise ValueError("Explanation and feature columns do not match")
    result = pd.DataFrame({
        "feature": explanation.feature_names,
        "original_column": list(X.columns),
        "mean_abs_shap_log": np.mean(np.abs(values), axis=0),
        "mean_shap_log": np.mean(values, axis=0),
    })
    result["is_text"] = result["original_column"].str.startswith(
        ("text_", "tfidf_")
    )
    return result.sort_values(
        "mean_abs_shap_log", ascending=False, kind="stable"
    ).reset_index(drop=True)
