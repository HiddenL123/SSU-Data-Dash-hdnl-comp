"""DATA_DIR may pin training data without blocking submission prediction inputs.

These tests avoid accessing real housing data or the network.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import data


def _create_file(root: Path, filename: str):
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    target.write_bytes(b"fixture")
    return target


def test_prediction_file_falls_back_to_project_data_when_override_has_only_listings(
    tmp_path, monkeypatch
):
    override = tmp_path / "colab_parquet"
    local = tmp_path / "normal_data"
    saved = _create_file(override, "listings.parquet")
    test_csv = _create_file(local, "test_off_market.csv")
    list_aware = _create_file(local, "test_list_aware.parquet")
    monkeypatch.setattr(data, "LOCAL_DATA_DIR", local)
    monkeypatch.setenv("DATA_DIR", str(override))

    assert data.find_file("listings") == saved
    assert data.find_file("off_market") == test_csv
    assert data.find_file("list_aware") == list_aware


def test_prediction_override_takes_priority_over_project_folder(tmp_path, monkeypatch):
    override = tmp_path / "override"
    local = tmp_path / "normal_data"
    _create_file(local, "test_off_market.parquet")
    _create_file(override, "test_off_market.csv")
    preferred = _create_file(override, "test_off_market.parquet")
    monkeypatch.setattr(data, "LOCAL_DATA_DIR", local)
    monkeypatch.setenv("DATA_DIR", str(override))

    assert data.find_file("off_market") == preferred


def test_missing_historical_listings_never_falls_back_silently(tmp_path, monkeypatch):
    override = tmp_path / "override"
    local = tmp_path / "normal_data"
    _create_file(local, "listings.csv")
    override.mkdir()
    monkeypatch.setattr(data, "LOCAL_DATA_DIR", local)
    monkeypatch.setenv("DATA_DIR", str(override))

    with pytest.raises(FileNotFoundError, match="DATA_DIR"):
        data.find_file("listings")


def test_prediction_uses_hugging_face_when_both_local_sources_are_missing(
    tmp_path, monkeypatch
):
    override = tmp_path / "override"
    local = tmp_path / "normal_data"
    override.mkdir()
    local.mkdir()
    monkeypatch.setattr(data, "LOCAL_DATA_DIR", local)
    monkeypatch.setenv("DATA_DIR", str(override))
    calls = []
    cached = tmp_path / "hf" / "test_off_market.parquet"
    def fake_download(repo_id, filename, repo_type):
        calls.append((repo_id, filename, repo_type))
        return str(cached)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )

    assert data.find_file("off_market") == cached
    assert calls == [("Baliyang/SSU_data_dash", "test_off_market.parquet", "dataset")]


def test_no_override_still_prefers_project_data(tmp_path, monkeypatch):
    local = tmp_path / "normal_data"
    expected = _create_file(local, "listings.csv")
    monkeypatch.setattr(data, "LOCAL_DATA_DIR", local)
    monkeypatch.delenv("DATA_DIR", raising=False)
    assert data.find_file("listings") == expected
