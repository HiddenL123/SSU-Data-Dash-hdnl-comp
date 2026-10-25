"""Synthetic safeguards for the final list-aware MAE feature ablation."""
import numpy as np
import pandas as pd
import pytest

from listaware_final_mae_features import (
    BASE_WEIGHTS, CHALLENGERS, MONTHS, REPLACEMENTS, WINNER,
    _prepare_manifest, _verified_original_predictions, replacement_blends,
)


def make_predictions():
    rows = []
    members = [WINNER, "xgb_spline_geo", "lgb_spline_geo", *CHALLENGERS]
    for variant in members:
        for month in MONTHS:
            for i in range(3):
                y = 300_000.0 + i * 15_000.0
                p = y * (1.02 if variant == WINNER else
                         1.03 if variant == "xgb_spline_geo" else
                         0.98 if variant == "lgb_spline_geo" else
                         1.01 if variant == "cat_mae_type_quota" else
                         0.99)
                rows.append({
                    "month": month, "ml_num": f"{month}-{i}",
                    "arm": "blend" if variant == WINNER else "model",
                    "variant": variant, "house_type_name": "Detached",
                    "sold_price": y, "predicted_price": p,
                    "ape": 100 * abs(p-y)/y,
                })
    return pd.DataFrame(rows)


def test_registered_feature_stacks_and_preserved_blend_weights():
    from models.features import make_features

    assert BASE_WEIGHTS == {
        "cat_mae": .50, "xgb_spline_geo": .35, "lgb_spline_geo": .15
    }
    for spec in CHALLENGERS.values():
        assert spec.target == "ratio"
        assert spec.loss == "MAE"
        assert spec.backend == "catboost"
        assert make_features(spec.feature_name, model="catboost",
                             track="list_aware") is not None


def test_fixed_replacement_blend_math_and_home_alignment():
    raw = make_predictions()
    result = replacement_blends(raw)
    assert set(result.variant) == set(REPLACEMENTS)
    assert result.groupby("variant").size().eq(len(MONTHS) * 3).all()
    src = raw.set_index(["month", "ml_num", "variant"])
    for label, model in REPLACEMENTS.items():
        aligned = result.loc[result.variant.eq(label)].set_index(["month", "ml_num"])
        idx = aligned.index
        p_mae = src.xs(model, level="variant").loc[idx, "predicted_price"].to_numpy(float)
        p_xgb = src.xs("xgb_spline_geo", level="variant").loc[idx, "predicted_price"].to_numpy(float)
        p_lgb = src.xs("lgb_spline_geo", level="variant").loc[idx, "predicted_price"].to_numpy(float)
        expected = np.exp(.50*np.log(p_mae) + .35*np.log(p_xgb) + .15*np.log(p_lgb))
        np.testing.assert_allclose(aligned.predicted_price.to_numpy(float), expected)


def test_missing_changed_or_duplicate_member_is_rejected():
    raw = make_predictions()
    chosen = raw.loc[raw.variant.eq("cat_mae_spatial_geo")].index[0]
    with pytest.raises(ValueError, match="Missing/misaligned"):
        replacement_blends(raw.drop(index=chosen))
    altered = raw.copy()
    altered.loc[chosen, "sold_price"] += 10
    with pytest.raises(ValueError, match="Missing/misaligned"):
        replacement_blends(altered)
    wrong_type = raw.copy()
    wrong_type.loc[chosen, "house_type_name"] = "Condo"
    with pytest.raises(ValueError, match="Missing/misaligned"):
        replacement_blends(wrong_type)
    repeated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        replacement_blends(repeated)


def test_frozen_original_cohort_matching():
    from listaware_full_fixed_origin_confirmation import CONTROLS
    original = make_predictions().loc[lambda x: ~x.variant.isin(CHALLENGERS)]
    # Original must also contain ALL controls, not just the three blend members.
    expanded = [original]
    template = original.loc[original.variant.eq(WINNER)].copy()
    for name in CONTROLS:
        if name not in set(original.variant):
            part = template.copy()
            part["variant"] = name
            part["arm"] = "model"
            expanded.append(part)
    original = pd.concat(expanded, ignore_index=True)
    held = {}
    for month in MONTHS:
        sample = original.loc[(original.month == month)
                              & original.variant.eq(WINNER)].copy()
        held[month] = sample[["ml_num", "house_type_name", "sold_price"]].copy()
    checked = _verified_original_predictions(original, held)
    assert len(checked) == len(original)
    altered = original.copy()
    ix = altered.loc[altered.variant.eq("lgb_spline_geo")].index[0]
    altered.loc[ix, "sold_price"] += 1000
    with pytest.raises(ValueError, match="Existing cohort mismatch"):
        _verified_original_predictions(altered, held)


def test_cache_manifest_rejects_changed_settings(tmp_path):
    original = {"experiment": 1, "data_sha256": "abc", "seed": 42}
    _prepare_manifest(tmp_path, original)
    _prepare_manifest(tmp_path, original)
    with pytest.raises(ValueError, match="manifest mismatch"):
        _prepare_manifest(tmp_path, {**original, "seed": 43})
