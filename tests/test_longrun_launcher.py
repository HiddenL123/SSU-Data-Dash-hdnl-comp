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
