"""Release candidate dual-execution, checkpoint and deployable-model tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
import pandas as pd
import pytest

from models.automl_profiles import (
    get_profile, release_settings, WORKER_PROFILES,
)
from models.automl_portable import export_checkpoint, import_checkpoint
from models.final_ensemble import WeightedOffMarketEnsemble
from ensemble_automl_factory import FactoryRun, FactorySettings
from ensemble_release_candidate import finalize_release, predict_release
import ensemble_release_candidate as release
import ensemble_rc
from models.automl_search import Candidate
from test_starter import make_listings


def test_profiles_share_full_fidelity_research_settings():
    cfg = release_settings()
    assert cfg["trees"] == 700
    assert cfg["train_cap"] >= 150_000
    assert cfg["validation_cap"] >= 10_000
    assert pd.Timestamp(cfg["train_start"]) <= pd.Timestamp("1900-01-01")
    assert get_profile("day_cloud").xgb_threads == 2
    assert get_profile("overnight_local").xgb_threads >= 4
    assert get_profile("day_local_quiet").xgb_threads <= 2
    assert set(WORKER_PROFILES) == {"day_cloud", "overnight_local", "day_local_quiet"}
    assert release_settings(generations=4)["trees"] == cfg["trees"]
    with pytest.raises(ValueError, match="Unknown execution"):
        get_profile("unsupported")


def test_sqlite_consistent_export_restore_and_no_overwrite(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    (root / "manifest.json").write_text('{"version": 1}', encoding="utf-8")
    (root / "roster.json").write_text('{"processed": ["a"]}', encoding="utf-8")
    (root / ".run.lock").write_bytes(b"0")
    sql = root / "optuna" / "studies.db"
    sql.parent.mkdir()
    conn = sqlite3.connect(sql)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE foo(id INTEGER PRIMARY KEY, score REAL)")
        conn.execute("INSERT INTO foo(score) VALUES(1.23)")
        conn.commit()
        archive = tmp_path / "drive" / "snapshot.zip"
        export_checkpoint(root, archive)
    finally:
        conn.close()
    assert archive.exists()
    assert (tmp_path / "drive" / "snapshot.zip.sha256").exists()
    with zipfile.ZipFile(archive) as z:
        assert ".run.lock" not in z.namelist()
        assert "optuna/studies.db" in z.namelist()
    copy = import_checkpoint(archive, tmp_path / "new_vm")
    assert not (copy / ".run.lock").exists()
    assert json.loads((copy / "roster.json").read_text())["processed"] == ["a"]
    with sqlite3.connect(copy / "optuna" / "studies.db") as restored:
        assert restored.execute("SELECT score FROM foo").fetchone()[0] == 1.23
    with pytest.raises(FileExistsError, match="nonempty"):
        import_checkpoint(archive, copy)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        archive.with_name(archive.name + ".sha256").read_text().strip()
    )


def test_archive_refuses_path_traversal(tmp_path):
    archive = tmp_path / "hostile.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("_automl_checkpoint_format.txt", "1\n")
        z.writestr("../outside.txt", "do not create")
        z.writestr("manifest.json", "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        import_checkpoint(archive, tmp_path / "recovered")
    assert not (tmp_path / "outside.txt").exists()


class StaticMember:
    def __init__(self, price):
        self.price = float(price)

    def predict_member(self, homes, name):
        return np.full(len(homes), self.price, dtype=float)


def test_weighted_ensemble_independent_providers_and_rounding():
    homes = pd.DataFrame({"ml_num": ["a", "b"], "latitude": [43.7, 43.8]})
    ensemble = WeightedOffMarketEnsemble(
        {"a": StaticMember(100.), "b": StaticMember(400.)},
        {"a": 0.5, "b": 0.5},
    )
    assert list(ensemble.predict_members(homes)) == ["a", "b"]
    np.testing.assert_allclose(ensemble.predict(homes), [200., 200.])
    with pytest.raises(ValueError, match="weight"):
        WeightedOffMarketEnsemble(
            {"a": StaticMember(100.)}, {"a": -1.}
        )


def _fake_ready_factory(tmp_path, monkeypatch, *, audit_good):
    homes = make_listings(180, seed=31)
    cfg = FactorySettings(
        tuning_months=("2025-03",), selection_months=("2025-04",),
        audit_months=("2025-05",), train_start="2025-01-01",
        train_cap=120, validation_cap=50, trees=700,
        generations=1, candidates_per_generation=1,
        output_dir=str(tmp_path / "experiment"),
    )
    experiment = FactoryRun(cfg, listings=homes)
    candidate = Candidate(
        name="selected_47_splines_geo_reference", seed=42,
        addons=("spacetime_splines", "geo_context"),
    )
    roster = {
        "version": 1, "processed": [candidate.name],
        "history": [], "members": [{
            "candidate": candidate.record(),
            "parameters": {"max_depth": 7, "learning_rate": .09},
            "admitted_alpha": 1.0,
        }],
    }
    from models.automl_checkpoint import write_json
    write_json(experiment.root / "roster.json", roster)
    checksum = hashlib.sha256(
        json.dumps(roster, sort_keys=True).encode()
    ).hexdigest()
    write_json(experiment.root / "shadow_audit_lock.json", {
        "status": "complete", "roster_sha256": checksum,
    })
    write_json(experiment.root / "summary.json", {
        "shadow_audit_status": "locked",
        "shadow_audit_metrics": {
            "mdape": 8.0 if audit_good else 11.0,
            "mape": 10.0, "p90_ape": 21.0,
        },
        "audit_first_reference_metrics": {
            "mdape": 9.0, "mape": 10.0, "p90_ape": 21.0,
        },
    })
    return cfg, homes


def test_release_gate_snapshot_resume_and_submission(tmp_path, monkeypatch):
    cfg, homes = _fake_ready_factory(tmp_path, monkeypatch, audit_good=True)
    archive = {}
    trained = []

    def saved_info(name, *, track):
        if name not in archive:
            raise FileNotFoundError(name)
        return {"metadata": archive[name][1]}

    def persist(model, name, *, track, metadata):
        assert track == "off_market"
        archive[name] = (model, metadata)

    monkeypatch.setattr(release, "model_info", saved_info)
    monkeypatch.setattr(release, "save_model", persist)
    monkeypatch.setattr(release, "load_model", lambda name, *, track: archive[name][0])
    def train_stub(train, candidate, params, cfg, *, threads):
        trained.append(candidate.name)
        return StaticMember(200_000.)
    monkeypatch.setattr(release, "_fit_member", train_stub)

    finalized = finalize_release(
        cfg, threads=2, model_name="rc_test", listings=homes
    )
    assert trained == ["selected_47_splines_geo_reference"]
    assert set(finalized.weights) == {"selected_47_splines_geo_reference"}
    np.testing.assert_allclose(finalized.predict(homes.iloc[:2]), [200_000., 200_000.])
    assert archive["rc_test"][1]["audit_non_regression_gate_passed"] is True
    finalize_release(cfg, threads=4, model_name="rc_test", listings=homes)
    assert len(trained) == 1  # fitted member cached; no second training
    assert (tmp_path / "experiment" / "release_manifest.json").is_file()

    candidate_output = tmp_path / "submission_off_market_rc.csv"
    result = predict_release(
        model_name="rc_test", homes=homes.iloc[:3],
        output_file=candidate_output,
    )
    assert result == candidate_output
    submission = pd.read_csv(candidate_output)
    assert list(submission.columns) == ["ml_num", "predicted_price"]
    assert submission.predicted_price.tolist() == [200_000] * 3
    with pytest.raises(FileExistsError):
        predict_release(
            model_name="rc_test", homes=homes.iloc[:3],
            output_file=candidate_output,
        )


def test_underperforming_audit_requires_explicit_override(tmp_path, monkeypatch):
    cfg, homes = _fake_ready_factory(tmp_path, monkeypatch, audit_good=False)
    with pytest.raises(ValueError, match="non-regression"):
        finalize_release(cfg, threads=2, model_name="bad_rc", listings=homes)
    with pytest.raises(SystemExit):
        ensemble_rc.main(["run", "--profile", "day_cloud",
                          "--output-dir", str(tmp_path / "empty")])


def test_release_cli_status_read_only(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "progress.json").write_text('{"stage":"screening"}')
    assert ensemble_rc.main(["status", "--output-dir", str(root)]) == 0
