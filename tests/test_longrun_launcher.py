"""Fast launcher checks; no actual ensemble fitting."""
from argparse import Namespace
import pytest

from train_longrun import main, parser_for_cli, settings_for, checkpoint_for


def test_defaults_are_longer_and_isolated():
    args = parser_for_cli().parse_args(["run", "--track", "both"])
    off = settings_for("off_market", args)
    listed = settings_for("list_aware", args)
    assert off.output_dir == "outputs/dual_track_longrun_v2/off_market"
    assert listed.output_dir == "outputs/dual_track_longrun_v2/list_aware"
    assert off.generations == listed.generations == 12
    assert off.candidates_per_generation == listed.candidates_per_generation == 3
    assert off.optuna_trials == listed.optuna_trials == 8
    assert off.trees == listed.trees == 700
    assert off.max_members == listed.max_members == 8
    assert off.train_cap >= 150000
    assert listed.train_cap >= 150000
    for config in (off, listed):
        assert max(config.tuning_months) < min(config.selection_months)
        assert max(config.selection_months) < min(config.audit_months)


def test_portable_checkpoint_names_are_track_separated():
    args = parser_for_cli().parse_args([
        "run", "--checkpoint-dir", "colab_checkpoints/test_longrun"
    ])
    off = checkpoint_for("off_market", args)
    listed = checkpoint_for("list_aware", args)
    assert off != listed
    assert off.name == "off_market_checkpoint.zip"
    assert listed.name == "list_aware_checkpoint.zip"


def test_status_does_not_train_or_initialize_data(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "new_search"
    monkeypatch.setattr("train_longrun.settings_for", lambda track, args: Namespace(
        output_dir=str(destination / track), generations=12,
        candidates_per_generation=3, optuna_trials=8, trees=700, max_members=8,
    ))
    assert main(["status", "--track", "both"]) == 0
    assert not destination.exists()
    assert "new experiment" in capsys.readouterr().out


def test_reject_nonpositive_threads():
    with pytest.raises(SystemExit, match="threads"):
        main(["run", "--threads", "0"])


def test_run_audits_and_exports_by_default(monkeypatch):
    from train_longrun import train_one
    from models import listaware_ensemble_automl_factory as factory_module
    events = []

    class StubFactory:
        def __init__(self, settings, runtime_threads, checkpoint_archive):
            self.settings = settings
            events.append("initialized")
        def run(self):
            events.append("selection")
            return {"selection_metrics": {"mdape": 4.0},
                    "effective_member_weights": {"reference_catboost": 1.0}}, None, None
        def audit(self):
            events.append("audit")
            return {"shadow_audit_metrics": {"mdape": 3.9}}, None

    monkeypatch.setattr(factory_module, "FactoryRun", StubFactory)
    monkeypatch.setattr(
        "train_longrun.export_completed_checkpoint",
        lambda track, args, root: events.append("export")
    )
    args = parser_for_cli().parse_args(["run", "--track", "list_aware"])
    train_one("list_aware", args)
    assert events == ["initialized", "selection", "audit", "export"]


def test_selection_only_skips_audit_but_exports(monkeypatch):
    from train_longrun import train_one
    from models import listaware_ensemble_automl_factory as factory_module
    events = []

    class StubFactory:
        def __init__(self, settings, runtime_threads, checkpoint_archive):
            pass
        def run(self):
            events.append("selection")
            return {"selection_metrics": {}, "effective_member_weights": {}}, None, None
        def audit(self):
            raise AssertionError("audit should remain unopened")

    monkeypatch.setattr(factory_module, "FactoryRun", StubFactory)
    monkeypatch.setattr(
        "train_longrun.export_completed_checkpoint",
        lambda track, args, root: events.append("export")
    )
    args = parser_for_cli().parse_args([
        "run", "--track", "list_aware", "--no-auto-audit"
    ])
    train_one("list_aware", args)
    assert events == ["selection", "export"]


def test_audit_failure_does_not_claim_success_or_export(monkeypatch):
    from train_longrun import train_one
    from models import listaware_ensemble_automl_factory as factory_module
    events = []

    class StubFactory:
        def __init__(self, settings, runtime_threads, checkpoint_archive):
            pass
        def run(self):
            events.append("selection")
            return {"selection_metrics": {}, "effective_member_weights": {}}, None, None
        def audit(self):
            events.append("audit_failed")
            raise RuntimeError("audit failed")

    monkeypatch.setattr(factory_module, "FactoryRun", StubFactory)
    monkeypatch.setattr(
        "train_longrun.export_completed_checkpoint",
        lambda track, args, root: events.append("export")
    )
    args = parser_for_cli().parse_args(["run", "--track", "list_aware"])
    with pytest.raises(RuntimeError, match="audit failed"):
        train_one("list_aware", args)
    assert events == ["selection", "audit_failed"]


def test_both_tracks_spawn_sequentially_with_export_settings(monkeypatch):
    import train_longrun
    calls = []
    def fake_subprocess_run(command, check):
        assert check is True
        calls.append(command)
    monkeypatch.setattr(train_longrun.subprocess, "run", fake_subprocess_run)

    assert main(["run", "--track", "both", "--generations", "20",
                 "--optuna-trials", "8", "--threads", "16"]) == 0
    assert len(calls) == 2
    assert calls[0][calls[0].index("--track") + 1] == "off_market"
    assert calls[1][calls[1].index("--track") + 1] == "list_aware"
    for command in calls:
        assert command[command.index("--generations") + 1] == "20"
        assert command[command.index("--threads") + 1] == "16"
        assert command[command.index("--export-dir") + 1] == "colab_checkpoints/dual_track_longrun_v2"
        assert "--no-auto-audit" not in command

    calls.clear()
    assert main(["run", "--track", "both", "--no-auto-audit"]) == 0
    assert all("--no-auto-audit" in command for command in calls)
