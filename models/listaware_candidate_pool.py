"""Curated LIST-AWARE candidate *feature recipes* for ensemble research.

Predeclared after individual-model screens (not new confirmed winners).
Each Candidate is exactly one estimator family, one loss and one target.
Feature representations compose only through supported features.make_features:
a single text variant + a single comparable selector + compatible spatial
add-ons. A residual correction is a separate, optional OOT stage; none are
implicitly added here or trained on the same holdout answers.

Models with near-identical errors are allowed to compete for marginal
ensemble admission; they should NOT all be accepted based on solo MdAPE.
Original candidate names are reused to avoid duplicate specifications.

Research input (historically inspected Mar--May 2025, 1,050-home pilot):
 - CatBoost MAE selected binary + spline/geo: MdAPE 4.6685, MAPE 6.1159
 - CatBoost MAE full binary + spline/geo: 4.7141, MAPE 6.1070
 - CatBoost MAE quota + spline/geo: 4.7131, P90 12.9986
 - CatBoost MAE legacy text + spline/geo: 4.7193
 - CatBoost MAE selected binary without spatial: 4.6941
Across CatBoost variants signed-log-error correlation was ~0.99 or more.
These scores are exploratory and are NOT full-budget, fixed-origin estimates.

XGB/LGB recipes below come from their own separate individual screens,
not from assuming CatBoost feature effects transfer to another family.
"""
from __future__ import annotations

from models.listaware_shotgun_specs import (
    BASIC, BINARY, Candidate,
)

# Only recipes absent from the existing historical catalog are added here.
EXTRA_CANDIDATES = {
    "cat_mae_full_binary_spatial_geo": Candidate(
        "cat_mae_full_binary_spatial_geo", "catboost",
        BASIC + "__full_binary__st_spacetime_splines+geo_context",
        loss="MAE",
    ),
    "cat_mae_legacy_spatial_geo": Candidate(
        "cat_mae_legacy_spatial_geo", "catboost",
        BASIC + "__st_spacetime_splines+geo_context",
        loss="MAE",
    ),
    "cat_mae_legacy_quota_splines": Candidate(
        "cat_mae_legacy_quota_splines", "catboost",
        BASIC + "__comp_type_quota__st_spacetime_splines",
        loss="MAE",
    ),
    # Transfer hypothesis: whether a full binary vocabulary is helpful to
    # XGB/LGB with the spatial stack requires matched per-family confirmation.
    # Separate name and output cache; never claim these were past winners.
    "xgb_full_binary_splines": Candidate(
        "xgb_full_binary_splines", "xgboost",
        BASIC + "__full_binary__st_spacetime_splines",
        tuned=True,
    ),
    "lgb_full_binary_splines": Candidate(
        "lgb_full_binary_splines", "lightgbm",
        BASIC + "__full_binary__st_spacetime_splines",
        tuned=True,
    ),
}

# Balanced by family, with multiple strong *feature* configurations. Names
# not in EXTRA_CANDIDATES are defined in feature.listaware_pool_screen's
# original CATALOG and preserve their existing parameters.
CURATED_BY_FAMILY = {
    "catboost": (
        "cat_mae",
        "cat_mae_spatial_geo",
        "cat_mae_binary",
        "cat_mae_splines",
        "cat_mae_full_binary_spatial_geo",
        "cat_mae_spatial_geo_type_quota",
        "cat_mae_legacy_spatial_geo",
        "cat_mae_legacy_quota_splines",
        "cat_huber_adjusted",
        "cat_huber_baseline",
    ),
    "xgboost": (
        "xgb_splines",
        "xgb_spline_geo",
        "xgb_adjusted",
        "xgb_tuned_binary",
        "xgb_full_binary_splines",
    ),
    "lightgbm": (
        "lgb_splines",
        "lgb_advanced",
        "lgb_spline_geo",
        "lgb_tuned_binary",
        "lgb_full_binary_splines",
    ),
}

CURATED = tuple(dict.fromkeys(
    name for family in CURATED_BY_FAMILY.values() for name in family
))

# Prioritized for full-strength confirmation, even if the cheap screening
# ranking changes. CatBoost solo anchor, and multiple XGB/LGB recipes remain
# eligible for later incremental ensemble selection. Does NOT force weights.
CONFIRMATION_ANCHORS = (
    "cat_mae_spatial_geo",
    "cat_mae_full_binary_spatial_geo",
    "xgb_splines",
    "lgb_splines",
    "lgb_advanced",
)

if len(CURATED) != len(set(CURATED)):
    raise RuntimeError("Duplicate curated candidate name")
if any(name != spec.name for name, spec in EXTRA_CANDIDATES.items()):
    raise RuntimeError("Candidate key/name mismatch")
