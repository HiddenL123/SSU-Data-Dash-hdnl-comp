"""Statistical and alignment guards for the full fixed-origin confirmation."""
import pandas as pd
import pytest

from experiments.listaware_full_fixed_origin_confirmation import (
    CONTROLS, MONTHS, WINNER, compare_against_winner,
)


def _predictions():
    rows = []
    for variant in CONTROLS:
        for month in MONTHS:
            for k in range(12):
                y = 100_000 + 1000 * k
                ratio = 1.02 if variant == WINNER else 1.05
                rows.append(dict(
                    month=month, ml_num=f"{month}-{k}",
                    arm="blend" if variant.startswith(("blend", "followup")) else "model",
                    variant=variant, sold_price=y, predicted_price=y * ratio,
                ))
    return pd.DataFrame(rows)


def test_strict_matched_cohort_and_paired_scoring():
    predictions = _predictions().sample(frac=1.0, random_state=7)
    summary, by_month = compare_against_winner(predictions, repeats=50)
    assert set(summary.variant) == set(CONTROLS)
    assert summary.homes.eq(36).all()
    assert len(by_month) == len(CONTROLS) * len(MONTHS)
    baseline = summary.set_index("variant").loc[WINNER]
    assert baseline.mdape == pytest.approx(2.0)
    assert baseline.delta_winner_mdape_pp == 0
    assert summary.set_index("variant").loc["cat_mae"].delta_winner_mdape_pp == pytest.approx(3.0)
    assert summary.set_index("variant").loc["cat_mae"].delta_winner_mdape_ci_low == pytest.approx(3.0)


def test_missing_or_mislabelled_homes_rejected():
    preds = _predictions()
    sample = preds.loc[preds.variant.eq("cat_mae")].index[0]
    with pytest.raises(ValueError, match="misaligned"):
        compare_against_winner(preds.drop(index=sample), repeats=5)
    altered = preds.copy()
    altered.loc[sample, "sold_price"] += 100
    with pytest.raises(ValueError, match="misaligned"):
        compare_against_winner(altered, repeats=5)


def test_duplicate_rejected():
    preds = _predictions()
    doubled = pd.concat([preds, preds.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_against_winner(doubled, repeats=5)


def test_wrong_cohort_rejected():
    preds = _predictions()
    preds["month"] = preds.month.replace({"2025-10": "2025-11"})
    with pytest.raises(ValueError, match="Wrong month cohort"):
        compare_against_winner(preds, repeats=5)


def test_scores_only_from_verified_full_origin(tmp_path):
    import json
    from experiments.listaware_fixed_origin_followup import CANDIDATES
    from experiments.listaware_full_fixed_origin_confirmation import run

    manifest = {
        "full": True, "mode": "single-fit_fixed_origin",
        "months": list(MONTHS), "candidates": list(CANDIDATES),
        "train_cutoff_exclusive": "2025-08-01",
        "train_start": "1900-01-01", "trees": None,
        "train_cap": 200000, "validation_cap": 100000,
        "data_sha256": "synthetic-manifest-hash",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    folder = tmp_path / "fixed_weight_followup"
    folder.mkdir()
    _predictions().to_csv(folder / "followup_predictions.csv", index=False)
    pooled, monthly = run(output_dir=tmp_path, from_existing=True, bootstrap_repeats=10)
    assert len(pooled) == len(CONTROLS)
    assert len(monthly) == len(CONTROLS) * len(MONTHS)
    assert (tmp_path / "confirmation_vs_prior_winner.csv").is_file()

    manifest["full"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pilot or expanding"):
        run(output_dir=tmp_path, from_existing=True, bootstrap_repeats=10)
