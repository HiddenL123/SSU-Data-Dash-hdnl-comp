"""Determinism, chronology and paired-log-blend tests for ensemble starter."""
import numpy as np
import pandas as pd
import pytest

import models.ensemble_bagging as bags
from models.ensemble_blending import blend_predictions, score_blends
import models.ensemble_feature_bagging_pilot as pilot
from features.text_features import TextAugmentedFeatures
from test_starter import make_listings


def test_selected_anchor_weighted_bags_and_row_bootstrap():
    selected = tuple(bags.SELECTED_TERMS)
    bank = tuple(dict.fromkeys((*selected, *[f"extra_{i}" for i in range(500)])))
    plan = bags.starter_specs(19)
    assert len(plan) >= 10
    assert set(bags.select_text_terms(bank, plan[0])) == set(selected)
    assert bags.select_text_terms(bank, plan[1]) == bank
    retained = bags.select_text_terms(bank, plan[2])
    assert set(selected).issubset(retained)
    assert len(retained) == len(selected) + 100
    for s in plan:
        assert bags.select_text_terms(bank, s) == bags.select_text_terms(bank, s)
    assert len(bags.select_text_terms(bank, plan[4])) == 150
    assert bags.select_text_terms(bank, plan[4]) != bags.select_text_terms(bank, plan[5])

    idx = bags.sample_training_rows(200, plan[6])
    assert len(idx) == 200 and len(np.unique(idx)) < 200
    assert np.array_equal(idx, bags.sample_training_rows(200, plan[6]))
    subbag = bags.sample_training_rows(200, plan[-1])
    assert len(subbag) == 160 and len(np.unique(subbag)) == 160


def test_training_only_vocabulary_keeps_selected_terms():
    train = pd.DataFrame({"remarks": [
        "Bright condo with renovated kitchen and balcony",
        "Renovated kitchen spacious detached home",
        "Bright home with balcony and garage",
    ]})
    bank = bags.build_text_bank(train, max_extra_terms=12, min_df=1)
    assert set(bags.SELECTED_TERMS).issubset(bank)
    assert "renovated" in bank
    with pytest.raises(ValueError):
        bags.build_text_bank(pd.DataFrame({"x": [1]}))


def test_geometric_blend_matching_and_missing_keys():
    p = pd.DataFrame([
        {"month": m, "ml_num": h, "member": v,
         "sold_price": y, "predicted_price": pred}
        for m,h,y,a,b in [
            ("2025-05","A",150.,100.,400.),
            ("2025-05","B",450.,400.,900.),
        ]
        for v,pred in (("a",a),("b",b))
    ])
    result = blend_predictions(p, {"a": 1, "b": 1})
    np.testing.assert_allclose(result.predicted_price, [200, 600])
    scores, streams = score_blends(p, {"half": {"a": 1, "b": 1}})
    assert len(scores) == 1 and len(streams) == 2
    with pytest.raises(ValueError, match="Duplicate"):
        blend_predictions(pd.concat([p,p.iloc[[0]]]), {"a": 1})
    with pytest.raises(ValueError, match="same homes"):
        blend_predictions(p.loc[~((p.member=="b")&(p.ml_num=="A"))],
                          {"a": 1, "b": 1})
    with pytest.raises(ValueError, match="Inconsistent actual"):
        broken = p.copy()
        broken.loc[broken.member=="b", "sold_price"] += 1
        blend_predictions(broken, {"a":1,"b":1})


def test_bagging_pilot_reuses_full_training_features_and_cached_predictions(
    tmp_path, monkeypatch
):
    sales = make_listings(900, seed=33)
    sales["remarks"] = np.resize(np.array([
        "Renovated spacious condo with parking balcony",
        "Detached family house with bright garden kitchen",
        "Modern loft beautiful windows central transit",
    ]), len(sales))

    class BaseProvider:
        def fit_transform(self, homes, **kwargs):
            self.columns_ = ["latitude", "longitude"]
            return self.transform(homes)
        def transform(self, homes, **kwargs):
            return homes[["latitude", "longitude"]].copy()

    def provider(*args, **kwargs):
        assert kwargs["model"] == "xgboost"
        assert kwargs["track"] == "off_market"
        return TextAugmentedFeatures(
            BaseProvider(), representation="binary", vocabulary_mode="full"
        )

    seen = []
    class DummyModel:
        def set_params(self, **kwargs):
            return self
        def fit(self, matrix, labels):
            seen.append((matrix.shape, len(labels)))
            self.mean = float(np.mean(labels))
            return self
        def predict(self, matrix):
            return np.full(len(matrix), self.mean)

    monkeypatch.setattr(pilot, "make_features", provider)
    monkeypatch.setattr(pilot, "_make_estimator", lambda *args: DummyModel())
    specs = (
        bags.BagSpec("selected", "selected", seed=11),
        bags.BagSpec("plus", "selected_plus", n_text=10, seed=12),
        bags.BagSpec("bootstrap", "weighted", n_text=60,
                     row_mode="bootstrap", seed=13),
    )
    args = dict(
        validation_month="2025-06", train_start="2025-01-01",
        train_cap=75, validation_cap=15, trees=4, specs=specs,
        max_extra_terms=30, min_df=1, listings=sales,
        output_dir=tmp_path, resume=True,
    )
    scores, preds, manifest = pilot.run_feature_bagging_pilot(**args)
    assert len(seen) == len(specs)
    assert len(scores) == len(specs)
    assert manifest.loc[manifest.name=="bootstrap", "unique_training_rows"].iat[0] < 75
    assert preds.groupby("member").size().nunique() == 1
    assert all(p.date() < pd.Timestamp("2025-06-01").date()
               for p in pd.to_datetime(sales.loc[
                   sales.date_sold < pd.Timestamp("2025-06-01"), "date_sold"
               ]))

    def fail_to_fit(*args):
        raise AssertionError("Saved per-member predictions must be reused")

    monkeypatch.setattr(pilot, "_make_estimator", fail_to_fit)
    scores2, _, _ = pilot.run_feature_bagging_pilot(**args)
    np.testing.assert_allclose(scores2.mdape, scores.mdape)
    with pytest.raises(ValueError, match="Different data/settings"):
        pilot.run_feature_bagging_pilot(**{**args, "trees": 5})
