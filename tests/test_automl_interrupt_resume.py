"""Injected interruption tests for the chronological ensemble factory."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import ensemble_automl_factory as factory
from ensemble_automl_factory import FactoryRun, FactorySettings
from models.automl_checkpoint import exclusive_factory_lock, write_json, write_csv
from test_starter import make_listings


def fake_factory(tmp_path, monkeypatch):
    sales = make_listings(280, seed=22)
    cfg = FactorySettings(
        tuning_months=("2025-03",), selection_months=("2025-04",),
        audit_months=("2025-05",), train_start="2025-01-01",
        train_cap=45, validation_cap=12, trees=5, min_df=1,
        candidates_per_generation=2, generations=1, optuna_trials=2,
        screen_max_mdape=50, output_dir=str(tmp_path / "factory"),
    )
    calls = []
    def predict(self, candidate, month, params, *, fold_name, cache=True):
        _, held = self._cohorts(month)
        y = held.sold_price.to_numpy(dtype=float)
        return y * (1.10 if candidate.name.endswith("_reference") else 1.02)
    def tune(self, candidate):
        calls.append(candidate.name)
        return {"max_depth": 7}, 12.0
    monkeypatch.setattr(FactoryRun, "_predict_fold", predict)
    monkeypatch.setattr(FactoryRun, "_tune", tune)
    return cfg, sales, calls


def test_atomic_checkpoint_preserves_old_file(tmp_path, monkeypatch):
    import models.automl_checkpoint as checkpoints
    dest = tmp_path / "roster.json"
    write_json(dest, {"valid": 1})
    actual = checkpoints.os.replace
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(checkpoints.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_json(dest, {"invalid": 2})
    assert json.loads(dest.read_text()) == {"valid": 1}
    monkeypatch.setattr(checkpoints.os, "replace", actual)
    write_csv(tmp_path / "preds.csv", pd.DataFrame({"p": [1., 2.]}))
    assert pd.read_csv(tmp_path / "preds.csv").p.tolist() == [1., 2.]


def test_single_writer_lock_can_be_reacquired(tmp_path):
    with exclusive_factory_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another factory process"):
            with exclusive_factory_lock(tmp_path):
                pass
    with exclusive_factory_lock(tmp_path):
        pass


def test_resume_at_durable_candidate_commit(tmp_path, monkeypatch):
    cfg, sales, calls = fake_factory(tmp_path, monkeypatch)
    actual = factory.write_json
    triggered = [False]
    def stopped_after_roster(path, data):
        actual(path, data)
        if (str(path).endswith("roster.json") and data.get("processed")
                and not triggered[0]):
            triggered[0] = True
            raise KeyboardInterrupt("forced interruption")
    monkeypatch.setattr(factory, "write_json", stopped_after_roster)
    with pytest.raises(KeyboardInterrupt):
        FactoryRun(cfg, listings=sales).run()
    checkpoint = tmp_path / "factory" / "roster.json"
    assert len(json.loads(checkpoint.read_text())["processed"]) == 1
    monkeypatch.setattr(factory, "write_json", actual)
    result, events, _ = FactoryRun(cfg, listings=sales).run()
    assert len(events) == 2
    assert len(calls) == 2  # no retuning the committed reference
    roster = json.loads(checkpoint.read_text())
    assert len(roster["processed"]) == len(roster["history"]) == 2
    assert json.loads((tmp_path / "factory" / "events.json").read_text()) == roster["history"]
    FactoryRun(cfg, listings=sales).run()
    assert len(calls) == 2


def test_audit_interruption_seals_roster_before_scoring(tmp_path, monkeypatch):
    cfg, sales, _ = fake_factory(tmp_path, monkeypatch)
    FactoryRun(cfg, listings=sales).run()
    old_predict = FactoryRun._predict_fold
    stop = [True]
    def interrupt_audit(self, candidate, month, params, *, fold_name, cache=True):
        if fold_name == "audit" and stop[0]:
            stop[0] = False
            raise KeyboardInterrupt("killed mid-audit")
        return old_predict(self, candidate, month, params, fold_name=fold_name, cache=cache)
    monkeypatch.setattr(FactoryRun, "_predict_fold", interrupt_audit)
    with pytest.raises(KeyboardInterrupt):
        FactoryRun(cfg, listings=sales).audit()
    lock = tmp_path / "factory" / "shadow_audit_lock.json"
    assert json.loads(lock.read_text())["status"] == "pending"
    with pytest.raises(RuntimeError, match="Shadow audit has already"):
        FactoryRun(replace(cfg, generations=2), listings=sales).run()
    result, predictions = FactoryRun(cfg, listings=sales).audit()
    assert "shadow_audit_metrics" in result
    assert predictions.month.unique().tolist() == ["2025-05"]
    assert json.loads(lock.read_text())["status"] == "complete"


def test_orphan_optuna_trial_is_retried(tmp_path, monkeypatch):
    optuna = pytest.importorskip("optuna")
    cfg, sales, _ = fake_factory(tmp_path, monkeypatch)
    run = FactoryRun(cfg, listings=sales)
    candidate = factory.generate_candidates(0, 2, 42)[0]
    folder = tmp_path / "factory" / "optuna"
    folder.mkdir()
    storage = f"sqlite:///{(folder / 'studies.db').as_posix()}"
    study_name = f"factory_v{factory.VERSION}_{candidate.name}"
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        direction="minimize", load_if_exists=True,
    )
    study.ask()  # Interrupted job remains RUNNING in SQLite
    class StubModel:
        def set_params(self, **kwargs):
            return self
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X))
    monkeypatch.setattr(factory, "_make_estimator", lambda *a, **k: StubModel())
    monkeypatch.setattr(FactoryRun, "_prepare_fold", lambda *a: {
        "X": np.zeros((8, 2)), "y": np.zeros(8),
        "H": np.zeros((4, 2)), "truth": np.full(4, 650000.),
        "offset": float(np.log(650000.)),
    })
    params, score = run._tune(candidate)
    trials = optuna.load_study(study_name=study_name, storage=storage).trials
    assert trials[0].state == optuna.trial.TrialState.FAIL
    assert sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials) == 2
    assert score == pytest.approx(0.)
    assert "max_depth" in params
    count = len(trials)
    run._tune(candidate)
    assert len(optuna.load_study(study_name=study_name, storage=storage).trials) == count
