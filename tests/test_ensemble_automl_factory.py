"""Fast synthetic checks for ensemble AutoML v1; no Optuna or XGB fits."""
import json
import numpy as np
import pandas as pd
import pytest

from models.automl_search import (
    Candidate, choose_admission, generate_candidates, log_blend,
    updated_weights, metrics,
)
from ensemble_automl_factory import FactorySettings, FactoryRun
from test_starter import make_listings


def test_candidates_reproducible_and_legal():
    a = generate_candidates(0, 12, 42)
    b = generate_candidates(0, 12, 42)
    assert a == b
    assert len(set(candidate.name for candidate in a)) == len(a)
    assert a[0].text_mode == "selected"
    assert a[0].addons == ("spacetime_splines", "geo_context")
    assert any(c.row_mode != "all" or c.text_mode != "selected" for c in a[1:])
    assert generate_candidates(1, 5, 42) != a[:5]
    with pytest.raises(ValueError):
        Candidate("bad", 42, addons=("unregistered",))


def test_admission_prefers_ensemble_gain_not_individual_accuracy():
    y = np.array([100, 100, 100, 100, 100, 100.], float)
    old = np.array([90, 90, 110, 110, 90, 110.], float)
    complement = np.array([110, 110, 90, 90, 110, 90.], float)
    # Both individuals are comparably inaccurate. Their log-price blend is
    # far closer to the labels than either one alone.
    accept = choose_admission(
        y, complement, old, min_gain=.01,
        max_mape_regression=0.05,
    )
    assert accept["accept"]
    assert accept["gain"] > 0
    assert accept["weight"] in (.05, .10, .20, .35, .50)
    assert accept["updated"]["mdape"] < accept["solo"]["mdape"]
    weights = updated_weights({"reference": 1.0}, "new", accept["weight"])
    assert np.isclose(sum(weights.values()), 1.0)
    reject = choose_admission(
        y, old, old, min_gain=.01,
    )
    assert not reject["accept"]
    assert choose_admission(y, old)["weight"] == 1.0
    with pytest.raises(ValueError):
        log_blend(old, complement, -1)
    with pytest.raises(ValueError):
        updated_weights({"ref": 1.}, "ref", 0.2)


def test_explicit_chronological_fold_boundaries():
    with pytest.raises(ValueError, match="tuning months"):
        FactorySettings(
            tuning_months=("2025-05",), selection_months=("2025-05",),
            audit_months=("2025-08",)
        )
    with pytest.raises(ValueError, match="tuning months"):
        FactorySettings(
            tuning_months=("2025-03",), selection_months=("2025-09",),
            audit_months=("2025-08",)
        )
    with pytest.raises(ValueError, match="precede Dec"):
        FactorySettings(
            audit_months=("2025-12",)
        )


def test_factory_resume_and_audit_after_frozen_selection(tmp_path, monkeypatch):
    sales = make_listings(220, seed=21)
    cfg = FactorySettings(
        tuning_months=("2025-03",),
        selection_months=("2025-04",),
        audit_months=("2025-05",),
        train_start="2025-01-01",
        train_cap=40, validation_cap=9, trees=5,
        min_df=1, candidates_per_generation=2,
        generations=1, optuna_trials=1,
        screen_max_mdape=30,
        minimum_ensemble_gain_pp=0.001,
        output_dir=str(tmp_path / "factory"),
    )
    # Synthetic stub tests *orchestration and chronology*, not XGBoost.
    visits = []
    def fake_prepare(self, candidate, month):
        _, homes = self._cohorts(month)
        ids = homes.ml_num.astype(str).to_numpy()
        return {
            "ids": ids, "truth": homes.sold_price.to_numpy(float),
            "month": str(month),
        }
    def fake_predict(self, candidate, month, params, *, fold_name, cache=True):
        visits.append((candidate.name, str(month), fold_name))
        f = fake_prepare(self, candidate, month)
        true = f["truth"]
        if candidate.name == "selected_47_splines_geo_reference":
            return true * (1 + .10 * np.where(np.arange(len(true)) % 2, 1, -1))
        # Second candidate has smaller errors, so it should improve the blend
        return true * (1 + .02 * np.where(np.arange(len(true)) % 2, 1, -1))
    def fake_tune(self, candidate):
        return dict(max_depth=7), 12.3

    monkeypatch.setattr(FactoryRun, "_prepare_fold", fake_prepare)
    monkeypatch.setattr(FactoryRun, "_predict_fold", fake_predict)
    monkeypatch.setattr(FactoryRun, "_tune", fake_tune)
    run = FactoryRun(cfg, listings=sales)
    outcome, events, audit = run.run()
    assert len(events) == 2
    assert outcome["effective_member_weights"]
    assert audit.month.nunique() == 1
    assert set(m for _, m, phase in visits if phase == "audit") == {"2025-05"}
    assert not any(phase == "audit" for _, _, phase in visits[:3])
    assert (tmp_path / "factory" / "roster.json").is_file()
    assert (tmp_path / "factory" / "summary.json").is_file()
    roster = json.loads(
        (tmp_path / "factory" / "roster.json").read_text(encoding="utf-8")
    )
    assert roster["processed"] == [
        "selected_47_splines_geo_reference", "g000_c001"
    ]
    original_visits = len(visits)
    repeated = FactoryRun(cfg, listings=sales).run()
    assert repeated[0]["members"] == outcome["members"]
    # Resume replays admitted members and recalculates the shadow audit but
    # must never resubmit already processed candidates to Optuna or screening.
    assert len(repeated[1]) == 2
    assert not any(phase == "screen" for _, _, phase in visits[original_visits:])
    with pytest.raises(ValueError, match="NEW output directory"):
        FactoryRun(
            FactorySettings(**{
                **cfg.record(), "trees": 7,
            }), listings=sales
        )
