"""Synthetic, CPU-light checks for the dedicated dual-track compute branch."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import listaware_ensemble_longrun as runner
from models.listaware_shotgun_specs import Candidate
from models.listaware_selected_ensemble import ListAwareSelectedEnsemble


def make_pool(folder: Path):
    folder.mkdir(parents=True)
    specs = {
        "cat_mae": Candidate("cat_mae", "catboost", "Xpt_spatiotemporal", loss="MAE"),
        "xgb_spline_geo": Candidate("xgb_spline_geo", "xgboost",
                                    "Xpt_spatiotemporal__selected_binary",
                                    tuned=True),
    }
    months = ["2025-06", "2025-07"]
    manifest = {
        "track": "list_aware", "stage": "confirm", "mode": "expanding",
        "full": True, "tree_cap": None,
        "train_cap": 200000, "validation_cap": 100000,
        "months": months,
        "data_sha256": "synthetic-listings-hash",
        "candidates": list(specs),
        "candidate_specs": {name: asdict(spec) for name, spec in specs.items()},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    prices = np.arange(100, 900, 100, dtype=float)
    base = prices * np.array([1.08, 1.06, 1.07, 1.08,
                              .92, .94, .93, .92])
    other = prices * np.array([.92, .94, .93, .92,
                               1.08, 1.06, 1.07, 1.08])
    records = []
    for j, month in enumerate(months):
        for name, pred in (("cat_mae", base), ("xgb_spline_geo", other)):
            for k in range(4):
                i = j * 4 + k
                records.append({
                    "month": month, "ml_num": f"MLS{i}",
                    "arm": "model", "variant": name,
                    "sold_price": prices[i], "predicted_price": pred[i],
                })
    pd.DataFrame(records).to_csv(folder / "predictions.csv", index=False)
    return specs


def make_sales():
    records = []
    for month in ("2025-06", "2025-07", *runner.AUDIT_MONTHS):
        for i in range(6):
            period = pd.Period(month, freq="M")
            date = period.start_time + pd.Timedelta(days=12)
            records.append({
                "ml_num": f"{month}-{i}", "date_sold": date,
                "date_listed": date - pd.Timedelta(days=2),
                "sold_price": 200000. + i*10000.,
                "list_price": 200100. + i*10000.,
                "house_type_name": "Detached",
            })
    return pd.DataFrame(records)


def test_greedy_selection_is_frozen_and_checkpoints_are_strict(tmp_path):
    pool = tmp_path / "pool"
    make_pool(pool)
    root = tmp_path / "selected"
    selection = runner.select(
        pool_dir=pool, output_dir=root, min_gain=.001, max_members=4
    )
    assert selection["weights"]["cat_mae"] < 1.0
    assert selection["weights"]["xgb_spline_geo"] > 0.
    assert sum(selection["weights"].values()) == pytest.approx(1.)
    assert selection["selection_metrics"]["homes"] == 8
    assert runner.select(
        pool_dir=pool, output_dir=root, min_gain=.001, max_members=4
    ) == selection
    with pytest.raises(ValueError, match="settings"):
        runner.select(pool_dir=pool, output_dir=root,
                      min_gain=3., max_members=4)
    original = pd.read_csv(pool / "predictions.csv")
    original.loc[0, "predicted_price"] *= 1.003
    original.to_csv(pool / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="changed"):
        runner.select(pool_dir=pool, output_dir=root,
                      min_gain=.001, max_members=4)


def test_audit_never_reads_target_labels_and_cannot_change_roster(
        tmp_path, monkeypatch
):
    pool = tmp_path / "pool"
    make_pool(pool)
    root = tmp_path / "selected"
    chosen = runner.select(
        pool_dir=pool, output_dir=root, min_gain=.001, max_members=4
    )
    sale = make_sales()
    monkeypatch.setattr(runner, "_historical_sales", lambda expected_hash: sale)
    calls = []

    def fake_fit(spec, train, hidden, *, tree_cap, threads, features_cache):
        assert tree_cap is None
        assert "date_sold" not in hidden and "sold_price" not in hidden
        assert "list_price" in hidden
        cutoff = hidden.date_listed.dt.to_period("M").min().start_time
        assert (train.date_sold < cutoff).all()
        calls.append((spec.name, len(train), len(hidden)))
        factor = 1.01 if spec.name == "cat_mae" else .99
        return hidden.list_price.to_numpy(float) * factor

    monkeypatch.setattr(runner, "_fit_predict", fake_fit)
    summary = runner.audit(output_dir=root, threads=1)
    assert summary["weights"] == chosen["weights"]
    assert summary["ensemble"]["homes"] == 18
    assert len(calls) == len(runner.AUDIT_MONTHS) * len(chosen["weights"])
    assert runner.audit(output_dir=root, threads=1) == summary
    assert len(calls) == len(runner.AUDIT_MONTHS) * len(chosen["weights"])
    changed = json.loads((root/"roster.json").read_text())
    changed["weights"]["cat_mae"] += .01
    (root/"roster.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Audit lock differs"):
        runner.audit(output_dir=root, threads=1)


def test_listaware_selected_bundle_requires_positive_weights_and_fit():
    spec = Candidate("cat_mae", "catboost", "Xpt_spatiotemporal", loss="MAE")
    with pytest.raises(ValueError, match="strictly positive"):
        ListAwareSelectedEnsemble({"cat_mae": spec}, {"cat_mae":0.0})
    model = ListAwareSelectedEnsemble({"cat_mae":spec},{"cat_mae":2.0})
    assert model.weights == {"cat_mae":1.0}
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_members(pd.DataFrame({"list_price":[100]}))
