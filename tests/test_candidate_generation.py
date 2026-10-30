"""No-data feature-search and chronological-protocol checks."""
import pytest

from models.automl_search import generate_candidates as offmarket_candidates
from models.listaware_factory_search import (
    Candidate, generate_candidates, MODEL_FAMILIES,
    ADDON_OPTIONS, COMPARABLE_OPTIONS, TEXT_MODES, ROW_MODES,
)
from models.listaware_ensemble_automl_factory import _feature_name, FactorySettings
from features.features import make_features


def test_offmarket_generator_is_deterministic():
    assert offmarket_candidates(0, 3, 42) == offmarket_candidates(0, 3, 42)
    assert offmarket_candidates(1, 3, 42) != offmarket_candidates(0, 3, 42)


def test_listaware_generator_not_restricted_to_curated_recipes():
    first = generate_candidates(0, 3, 42)
    assert {x.family for x in first} == set(MODEL_FAMILIES)
    assert first == generate_candidates(0, 3, 42)
    later = generate_candidates(1, 20, 42)
    assert len(later) == 20
    assert len({x.name for x in later}) == 20
    for candidate in (*first, *later):
        assert not hasattr(candidate, "recipe")
        assert candidate.addons in ADDON_OPTIONS
        assert candidate.comparable in COMPARABLE_OPTIONS
        assert candidate.text_mode in TEXT_MODES
        assert candidate.row_mode in ROW_MODES
        provider = make_features(_feature_name(candidate),
                                 model=candidate.family, track="list_aware")
        assert provider is not None


def test_bad_generator_configuration_rejected():
    with pytest.raises(ValueError, match="family"):
        Candidate("bad", 1, family="unrecognized")
    with pytest.raises(ValueError, match="RMSE"):
        Candidate("bad", 1, family="xgboost", objective="MAE")
    with pytest.raises(ValueError, match="Invalid row fraction"):
        Candidate("bad", 1, row_fraction=0)


def test_chronological_factory_folds_are_separated():
    cfg = FactorySettings()
    assert max(cfg.tuning_months) < min(cfg.selection_months)
    assert max(cfg.selection_months) < min(cfg.audit_months)
    with pytest.raises(ValueError, match="tuning months"):
        FactorySettings(tuning_months=("2025-07", "2025-08"))
