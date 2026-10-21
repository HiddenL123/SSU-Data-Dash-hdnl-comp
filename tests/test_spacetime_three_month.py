"""Small, fast regression tests for the three-month shortlisted follow-up."""
import json

import numpy as np
import pandas as pd
import pytest

import spacetime_three_month as target


def _data():
    rows = []
    for month in ("2025-09", "2025-10", "2025-11"):
        for i in range(12):
            rows.append({
                "ml_num": f"{month}-property-{i:02}",
                "date_sold": pd.Timestamp(month + "-15"),
                "date_listed": pd.Timestamp(month + "-03"),
                "house_type_name": "Condo Apt" if i < 6 else "Detached",
                "sold_price": 500_000 + 20_000 * i,
            })
    return pd.DataFrame(rows)


def test_pairwise_pooled_scores_and_reuse(tmp_path, monkeypatch):
    sales = _data()
    calls = []
    variants = {
        ("feature", "baseline"): 1.10,
        ("feature", "comp_type_quota"): 1.04,
        ("residual", "knn"): 1.07,
    }

    def fake_shotgun(*, validation_month, output_dir, **kwargs):
        calls.append(validation_month)
        path = pd.io.common.stringify_path(output_dir)
        from pathlib import Path
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        subset = sales[
            sales.date_sold.dt.to_period("M") == pd.Period(validation_month)
        ]
        frames = []
        for (arm, variant), scale in variants.items():
            frames.append(pd.DataFrame({
                "ml_num": subset.ml_num.to_numpy(),
                "month": validation_month,
                "arm": arm,
                "variant": variant,
                "sold_price": subset.sold_price.to_numpy(dtype=float),
                "predicted_price": subset.sold_price.to_numpy(dtype=float) * scale,
            }))
        predicted = pd.concat(frames, ignore_index=True)
        predicted.to_csv(folder / "predictions.csv", index=False)
        (folder / "setup.json").write_text(json.dumps({
            "validation_month": validation_month,
            "train_start": kwargs["train_start"],
            "train_cap": kwargs["train_cap"],
            "validation_cap": kwargs["validation_cap"],
            "calibration_cap": kwargs["calibration_cap"],
            "calibration_folds": kwargs["calibration_folds"],
            "screen_trees": kwargs["screen_trees"],
            "seed": kwargs["seed"],
            "params": kwargs["params"],
            "residual_strength": kwargs["residual_strength"],
            "feature_variants": list(kwargs["features"]),
            "residual_variants": list(kwargs["residuals"]),
        }))
        (folder / "failures.json").write_text("[]")
        return pd.DataFrame(), predicted, []

    monkeypatch.setattr(target, "run_shotgun", fake_shotgun)

    def go():
        return target.run_three_month(
            listings=sales, output_dir=tmp_path,
            train_cap=100, trees=8, validation_cap=100,
            calibration_cap=20, calibration_folds=2,
            features=("baseline", "comp_type_quota"),
            residuals=("knn",),
            bootstrap_repeats=20,
        )

    pooled, monthly, by_type, pred = go()
    assert calls == list(target.MONTHS)
    assert len(pred) == len(sales) * 3
    assert len(monthly) == 9
    assert len(pooled) == 3
    best = pooled.loc[pooled.variant == "comp_type_quota"].iloc[0]
    assert best["homes"] == len(sales)
    assert best["delta_mdape_pp"] < -5.0
    assert best["months_better_mdape"] == 3
    assert best["delta_mdape_ci_high"] < 0
    assert (tmp_path / "pooled_summary.csv").is_file()
    assert (tmp_path / "monthly_summary.csv").is_file()
    assert (tmp_path / "predictions_with_type.csv").is_file()
    # Ensure successful months are reusable, without additional model calls.
    again = go()
    assert len(calls) == 3
    pd.testing.assert_frame_equal(
        pooled.reset_index(drop=True),
        again[0].reset_index(drop=True),
    )


def test_rejects_non_development_and_wrong_month_counts(tmp_path):
    with pytest.raises(ValueError, match="exactly three"):
        target.run_three_month(months=("2025-09", "2025-10"), output_dir=tmp_path)
    with pytest.raises(ValueError, match="December 2025"):
        target.run_three_month(
            months=("2025-10", "2025-11", "2025-12"), output_dir=tmp_path
        )


def test_unpaired_or_missing_candidates_rejected():
    base = pd.DataFrame({
        "month": ["2025-09", "2025-09"],
        "ml_num": ["a", "b"],
        "arm": ["feature", "feature"],
        "variant": ["baseline", "baseline"],
        "sold_price": [100, 100],
        "predicted_price": [95, 105],
    })
    missing = pd.concat([base, base.iloc[[0]].assign(variant="comp_type_quota")],
                        ignore_index=True)
    with pytest.raises(ValueError, match="does not match"):
        target._validate_result(
            missing,
            {("feature", "baseline"), ("feature", "comp_type_quota")},
            ["2025-09"],
        )
    repeated = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Repeated"):
        target._validate_result(repeated, {("feature", "baseline")}, ["2025-09"])
