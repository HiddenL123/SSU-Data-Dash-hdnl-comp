"""No-data tests for curating list-aware CatBoost, XGBoost and LightGBM."""
import pandas as pd
import pytest

from features.features import make_features
from comparisons.listaware_pool_screen import (
    CATALOG, SEED_REFERENCE, SUITES, choose_confirmation,
)
from models.listaware_candidate_pool import (
    CURATED, CURATED_BY_FAMILY, CONFIRMATION_ANCHORS, EXTRA_CANDIDATES,
    CURATED_CATALOG,
)


def test_three_backend_curated_pool_and_registered_compositions():
    assert len(CURATED) == 20
    assert len(set(CURATED)) == len(CURATED)
    assert set(CURATED) <= set(SUITES["curated"])
    assert set(CONFIRMATION_ANCHORS) < set(CURATED)
    assert set(CURATED_BY_FAMILY) == {"catboost", "xgboost", "lightgbm"}
    assert all(len(options) >= 4 for options in CURATED_BY_FAMILY.values())
    assert set(CATALOG) == set(SUITES["all"]).union(SUITES["curated"])
    for backend, options in CURATED_BY_FAMILY.items():
        for name in options:
            c = CATALOG[name]
            assert c.name == name
            assert c.backend == backend
            assert c.target == "ratio"
            assert make_features(c.feature_name, model=backend,
                                 track="list_aware") is not None
    # Exactly one text representation and one comparable selector per spec,
    # with compatible spatial add-ons. No residual stage is silently attached.
    assert CATALOG["cat_mae_spatial_geo"].feature_name == (
        "Xpt_spatiotemporal__selected_binary__st_spacetime_splines+geo_context"
    )
    assert CATALOG["cat_mae_full_binary_spatial_geo"].feature_name == (
        "Xpt_spatiotemporal__full_binary__st_spacetime_splines+geo_context"
    )
    assert CATALOG["cat_mae_spatial_geo_type_quota"].feature_name.endswith(
        "__comp_type_quota__st_spacetime_splines+geo_context"
    )
    assert all(name == c.name for name, c in EXTRA_CANDIDATES.items())


def test_curated_confirmation_anchors_preserve_each_family_but_not_weights():
    # Hypothetical rankings: even very poorly ranked anchors remain available
    # to test ensemble complementarity, without forcing them into final blend.
    scores = pd.DataFrame({
        "variant": list(SUITES["curated"]),
        "mdape": [3.5 + .13*i for i in range(len(SUITES["curated"]))],
        "mape": [4.5 + .08*i for i in range(len(SUITES["curated"]))],
    })
    keep = choose_confirmation(
        scores.sample(frac=1, random_state=25),
        top_k=14, anchors=CONFIRMATION_ANCHORS,
    )
    assert len(keep) == 14
    assert set(SEED_REFERENCE).union(CONFIRMATION_ANCHORS) <= set(keep)
    assert all(any(CATALOG[name].backend == backend for name in keep)
               for backend in CURATED_BY_FAMILY)
    assert choose_confirmation(
        scores.iloc[::-1], top_k=14,
        anchors=CONFIRMATION_ANCHORS,
    ) == keep
    with pytest.raises(ValueError, match="top_k"):
        choose_confirmation(
            scores, top_k=8, anchors=CONFIRMATION_ANCHORS
        )
    with pytest.raises(ValueError, match="missing required"):
        choose_confirmation(
            scores.loc[scores.variant != "lgb_advanced"],
            top_k=14, anchors=CONFIRMATION_ANCHORS
        )
