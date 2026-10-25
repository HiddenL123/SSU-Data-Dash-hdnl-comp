"""Legacy compatibility and strict off-market/list-aware persistence tests.

Use tmp_path only; these tests never touch real saved model snapshots.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from models.save_model import (
    save_model, load_model, model_info,
    list_saved_models, list_model_catalog, save_validation_tracks,
)
from models.ensemble_snapshot import FeatureBaggingEnsemble, BaggedMember


class TinyProvider:
    """Picklable stand-in for a fitted historical feature provider."""

    def transform(self, homes):
        return homes[["latitude", "bedrooms"]].copy()


def test_track_specific_storage_does_not_modify_legacy(tmp_path):
    save_model({"old": 1}, "reference", save_dir=tmp_path)
    original = tmp_path / "reference" / "model.joblib"
    before = original.read_bytes()
    save_model({"new": "off"}, "reference", track="off_market",
               save_dir=tmp_path, metadata={"feature_set": "binary"})
    save_model({"new": "listed"}, "reference", track="list_aware",
               save_dir=tmp_path, metadata={"feature_set": "ratio"})
    assert original.read_bytes() == before
    assert load_model("reference", save_dir=tmp_path) == {"old": 1}
    assert load_model("reference", track="off_market",
                      save_dir=tmp_path) == {"new": "off"}
    assert load_model("reference", track="list_aware",
                      save_dir=tmp_path) == {"new": "listed"}
    assert model_info("reference", track="off_market",
                      save_dir=tmp_path)["storage_track"] == "off_market"
    assert model_info("reference", track="list_aware",
                      save_dir=tmp_path)["metadata"]["track"] == "list_aware"
    assert list_saved_models(save_dir=tmp_path) == ["reference"]
    assert list_saved_models(track="off_market", save_dir=tmp_path) == ["reference"]
    assert list_saved_models(track="list_aware", save_dir=tmp_path) == ["reference"]
    catalog = list_model_catalog(save_dir=tmp_path)
    assert {c["track"] for c in catalog} == {"legacy", "off_market", "list_aware"}
    assert len({c["relative_path"] for c in catalog}) == 3


def test_track_namespace_guards_and_no_implicit_fallback(tmp_path):
    save_model(123, "legacy_only", save_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_model("legacy_only", track="off_market", save_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        model_info("legacy_only", track="list_aware", save_dir=tmp_path)
    with pytest.raises(ValueError, match="track must"):
        save_model(1, "test", track="legacy", save_dir=tmp_path)
    with pytest.raises(ValueError, match="track must"):
        list_saved_models(track="../off_market", save_dir=tmp_path)
    with pytest.raises(ValueError, match="conflicts"):
        save_model(1, "test", track="off_market",
                   metadata={"track": "list_aware"}, save_dir=tmp_path)
    with pytest.raises(ValueError, match="only their own track"):
        save_model(1, "test", track="off_market",
                   metadata={"tracks": ["off_market", "list_aware"]},
                   save_dir=tmp_path)
    with pytest.raises(ValueError):
        save_model(1, "../test", track="off_market", save_dir=tmp_path)
    save_model(1, "x", track="off_market", save_dir=tmp_path)
    with pytest.raises(FileExistsError):
        save_model(2, "x", track="off_market", save_dir=tmp_path)
    save_model(2, "x", track="off_market", save_dir=tmp_path, overwrite=True)
    assert load_model("x", track="off_market", save_dir=tmp_path) == 2


def test_split_old_monthly_snapshots_in_order_without_mutation(tmp_path):
    original = [
        ["off_jan", "list_jan"],
        ["off_feb", "list_feb"],
    ]
    save_model(original, "mixed", save_dir=tmp_path,
               metadata={"tracks": ["off_market", "list_aware"]})
    paths = save_validation_tracks(
        original, "mixed", save_dir=tmp_path,
        metadata={
            "tracks": ["off_market", "list_aware"],
            "validation_months": ["2025-01", "2025-02"],
        },
    )
    assert paths["off_market"] == tmp_path / "off_market" / "mixed"
    assert paths["list_aware"] == tmp_path / "list_aware" / "mixed"
    assert load_model("mixed", save_dir=tmp_path) == original
    assert load_model("mixed", track="off_market", save_dir=tmp_path) == [
        "off_jan", "off_feb",
    ]
    assert load_model("mixed", track="list_aware", save_dir=tmp_path) == [
        "list_jan", "list_feb",
    ]
    assert model_info("mixed", track="list_aware", save_dir=tmp_path)[
        "metadata"]["validation_months"] == ["2025-01", "2025-02"]
    with pytest.raises(FileExistsError):
        save_validation_tracks(original, "mixed", save_dir=tmp_path)
    with pytest.raises(ValueError, match="exactly two"):
        save_validation_tracks([["only_one"]], "bad", save_dir=tmp_path)
    with pytest.raises(ValueError, match="count"):
        save_validation_tracks(original, "bad", save_dir=tmp_path,
                               metadata={"validation_months": ["2025-01"]})


def test_shared_fitted_ensemble_snapshot_round_trip(tmp_path):
    frame = pd.DataFrame(
        {"latitude": [43.6, 43.7, 43.8],
         "bedrooms": [1., 2., 3.]},
        index=[4, 8, 12],
    )
    provider = TinyProvider()
    fitted = {}
    for name, bias in (("small", -0.08), ("large", 0.04)):
        model = DummyRegressor(strategy="constant", constant=bias)
        model.fit(frame, np.full(len(frame), bias))
        fitted[name] = BaggedMember(
            name, model, ("latitude", "bedrooms"),
            float(np.log(750000.)), ("renovated",), {"seed": 42},
        )
    ensemble = FeatureBaggingEnsemble(
        provider, fitted, metadata={"feature_set": "binary"}
    )
    expected = ensemble.predict_members(frame)
    assert list(expected) == ["small", "large"]
    assert expected.index.tolist() == [4, 8, 12]
    a = ensemble.predict(frame, weights={"small": 1, "large": 1})
    np.testing.assert_allclose(a, np.sqrt(
        expected["small"].to_numpy() * expected["large"].to_numpy()
    ))
    save_model(ensemble, "ensemble_trial", track="off_market",
               save_dir=tmp_path, metadata=ensemble.metadata)
    recovered = load_model("ensemble_trial", track="off_market",
                           save_dir=tmp_path)
    np.testing.assert_allclose(
        recovered.predict_member(frame, "small"), expected["small"]
    )
    np.testing.assert_allclose(
        recovered.predict(frame), ensemble.predict(frame)
    )
    assert len(recovered.members) == 2
    with pytest.raises(ValueError, match="nonnegative"):
        recovered.predict(frame, weights={"small": -1})
    with pytest.raises(KeyError):
        recovered.predict_member(frame, "nonexistent")


class SmallTextBase:
    """Picklable lightweight replacement for production historical features."""
    def fit_transform(self, homes, **kwargs):
        return self.transform(homes)

    def transform(self, homes, **kwargs):
        return homes[["latitude", "longitude"]].copy()


class SmallXGB:
    """Picklable, deterministic estimator used in storage integration tests."""
    def set_params(self, **kwargs):
        return self

    def fit(self, matrix, y):
        self.center = float(np.mean(y))
        return self

    def predict(self, matrix):
        return np.full(len(matrix), self.center)


def test_feature_bagging_pilot_saves_reloadable_offmarket_models(
    tmp_path, monkeypatch,
):
    import data
    import models.ensemble_feature_bagging_pilot as pilot
    from models.ensemble_bagging import BagSpec
    from features.text_features import TextAugmentedFeatures
    from test_starter import make_listings

    listings = make_listings(650, seed=41)
    listings["remarks"] = np.resize([
        "sunny balcony newly renovated transit",
        "family detached backyard quiet garden",
        "spacious condo modern windows parking",
    ], len(listings))

    monkeypatch.setattr(
        pilot, "make_features",
        lambda *args, **kwargs: TextAugmentedFeatures(
            SmallTextBase(), representation="binary",
            vocabulary_mode="full",
        ),
    )
    calls = []
    def new_estimator(*args):
        calls.append(1)
        return SmallXGB()
    monkeypatch.setattr(pilot, "_make_estimator", new_estimator)

    specs = (
        BagSpec("reference", "selected", seed=42),
        BagSpec("bootstrap", "selected_plus", n_text=5,
                row_mode="bootstrap", seed=43),
    )
    args = dict(
        validation_month="2025-06", train_start="2025-01-01",
        train_cap=80, validation_cap=12,
        trees=5, specs=specs, listings=listings,
        max_extra_terms=20, min_df=1,
        output_dir=tmp_path / "outputs",
        model_save_dir=tmp_path / "snapshots",
        save_fitted_models=True, resume=True,
    )
    scores, preds, _ = pilot.run_feature_bagging_pilot(**args)
    assert len(calls) == len(specs)
    names = list_saved_models(
        track="off_market", save_dir=tmp_path / "snapshots"
    )
    assert len(names) == 1
    assert not list_saved_models(
        track="list_aware", save_dir=tmp_path / "snapshots"
    )
    saved = load_model(
        names[0], track="off_market", save_dir=tmp_path / "snapshots"
    )
    assert set(saved.members) == {"reference", "bootstrap"}
    held = listings.loc[
        listings.date_sold.dt.to_period("M") == pd.Period("2025-06")
    ]
    from experiments.spacetime_fast import _cap
    held = _cap(held, 12, 53)
    hidden = data.hide_answers(held, "off_market")
    for spec in specs:
        expected = preds.loc[
            preds.member == spec.name, "predicted_price"
        ].to_numpy()
        np.testing.assert_allclose(
            saved.predict_member(hidden, spec.name), expected
        )
    before = len(calls)
    scores2, _, _ = pilot.run_feature_bagging_pilot(**args)
    assert len(calls) == before
    np.testing.assert_allclose(scores.mdape, scores2.mdape)
    assert model_info(
        names[0], track="off_market",
        save_dir=tmp_path / "snapshots",
    )["metadata"]["track"] == "off_market"
