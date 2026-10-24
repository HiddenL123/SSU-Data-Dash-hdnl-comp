"""Statistical and alignment guards for the full fixed-origin confirmation."""
import pandas as pd
import pytest

from listaware_full_fixed_origin_confirmation import (
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
