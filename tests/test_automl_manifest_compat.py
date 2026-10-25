"""Pure regression checks for Colab-to-Windows AutoML manifest integrity."""
import json
import pandas as pd
import pytest

from models.ensemble_automl_factory import FactorySettings
from models.automl_manifest import (
    compare_manifests, compute_manifest, saved_factory_settings,
)
from experiments.spacetime_fast import XGB_TUNED


def fake_sales():
    return pd.DataFrame({
        "ml_num": ["A01", "A02", "A03"],
        "date_listed": ["2025-02-01", "2025-02-02", "2025-02-04"],
        "date_sold": ["2025-03-02", "2025-03-07", "2025-03-13"],
        "sold_price": [101000.0, 202000.0, 303000.0],
        "remarks": ["one", "two", "three"],
    })


def test_manifest_compatibility_preserves_original_v1_hash(tmp_path):
    settings = FactorySettings(output_dir="outputs/ensemble_release_candidate_full")
    frame = fake_sales()
    setup = compute_manifest(settings, frame, XGB_TUNED)
    assert setup["version"] == 1
    assert "generations" not in setup["settings"]
    assert compare_manifests(setup, compute_manifest(settings, frame, XGB_TUNED)) == {}

    # Generations is the one supported budget extension, not a new study.
    raised = FactorySettings(**{**settings.record(), "generations": 4})
    assert not compare_manifests(
        setup, compute_manifest(raised, frame, XGB_TUNED)
    )
    altered = frame.copy()
    altered.loc[0, "sold_price"] += 100
    diffs = compare_manifests(
        setup, compute_manifest(settings, altered, XGB_TUNED)
    )
    assert "data_sha256" in diffs


def test_reports_exact_setting_field_without_overwriting_manifest(tmp_path):
    settings = FactorySettings(output_dir="outputs/ensemble_release_candidate_full")
    setup = compute_manifest(settings, fake_sales(), XGB_TUNED)
    root = tmp_path / "checkpoint"
    root.mkdir()
    path = root / "manifest.json"
    path.write_text(json.dumps(setup), encoding="utf-8")
    original_bytes = path.read_bytes()

    changed = FactorySettings(**{**settings.record(), "train_start": "1900-01-01"})
    diffs = compare_manifests(
        setup, compute_manifest(changed, fake_sales(), XGB_TUNED)
    )
    assert diffs == {
        "settings.train_start": {
            "checkpoint": settings.train_start,
            "local": "1900-01-01",
        }
    }

    # Explicitly select the saved protocol, never force an unexpected value.
    recovered = saved_factory_settings(root, generations=2)
    assert recovered.train_start == settings.train_start
    assert recovered.generations == 2
    assert path.read_bytes() == original_bytes
    assert not compare_manifests(
        setup, compute_manifest(recovered, fake_sales(), XGB_TUNED)
    )


def test_rejects_missing_manifest_when_loading_saved_settings(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        saved_factory_settings(tmp_path, generations=2)
