"""Inference contract tests using real saved bundles, no housing data or training."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from models import ensemble_serving as serving
from models.ensemble_snapshot import BaggedMember, FeatureBaggingEnsemble
from models.final_ensemble import WeightedOffMarketEnsemble
from models.listaware_factory_snapshot import ListAwareBaggedMember, ListAwareFactoryEnsemble
from models.save_model import load_model, save_model


class FittedProvider:
    def __init__(self, scale, *, reverse=False, mutate=False):
        self.scale = scale
        self.reverse = reverse
        self.mutate = mutate

    def fit(self, *args):
        raise AssertionError("Inference must never fit a feature provider")

    def transform(self, homes):
        result = pd.DataFrame({"signal": homes["signal"] * self.scale}, index=homes.index)
        if self.mutate:
            homes["signal"] = -999
        return result.iloc[::-1] if self.reverse else result


class FittedEstimator:
    def fit(self, *args):
        raise AssertionError("Inference must never train an estimator")

    def predict(self, features):
        return features["signal"].to_numpy(float)


class ArbitraryPredictor:
    def predict(self, homes):
        return np.ones(len(homes))


def bundle(track, **provider_options):
    members = {}
    for name, scale in (("a", 0.1), ("b", 0.3)):
        provider = FittedProvider(scale, **provider_options)
        common = dict(name=name, estimator=FittedEstimator(), selected_columns=("signal",),
                      target_offset=0.2 if track == "list_aware" else np.log(100_000),
                      terms=(name,), settings={"fitted": True})
        if track == "off_market":
            members[name] = FeatureBaggingEnsemble(provider, {name: BaggedMember(**common)})
        else:
            members[name] = ListAwareBaggedMember(feature_provider=provider, **common)
    cls = WeightedOffMarketEnsemble if track == "off_market" else ListAwareFactoryEnsemble
    # Non-unit weights and a different dict order detect accidental reweighting.
    return cls(members, {"b": 3.0, "a": 1.0})


@pytest.fixture(params=serving.TRACKS)
def track(request):
    return request.param


@pytest.fixture
def homes():
    return pd.DataFrame({"signal": [3.0, 1.0, 2.0], "list_price": [200_000., 100_000., 300_000.]},
                        index=pd.Index([9, 2, 9], name="original_row"))


def adapter(tmp_path, track, model=None):
    save_model(bundle(track) if model is None else model, "release", track=track,
               save_dir=tmp_path, metadata={"nested": {"value": 1}})
    return serving.TrainedEnsembleModel(track, "release", save_dir=tmp_path)


def test_roundtrip_predictions_weights_providers_and_cache(tmp_path, track, homes):
    model = adapter(tmp_path, track)
    direct = load_model("release", track=track, save_dir=tmp_path)
    before = homes.copy(deep=True)
    with patch.object(serving, "load_model", wraps=load_model) as loader:
        actual = model.predict(homes)
        loaded = model._model
        np.testing.assert_allclose(actual, direct.predict(homes))
        base = homes.list_price.to_numpy() * np.exp(0.2) if track == "list_aware" else 100_000
        np.testing.assert_allclose(actual, base * np.exp(homes.signal.to_numpy() * 0.25))
        np.testing.assert_allclose(model.predict(homes.iloc[::-1]), actual[::-1])
        model.load()
        assert model._model is loaded
        assert loader.call_count == 1
        assert loader.call_args.kwargs["track"] == track
    assert actual.shape == (3,)
    assert np.isfinite(actual).all() and (actual > 0).all()
    assert loaded.weights == {"b": 3.0, "a": 1.0}
    assert loaded.members["a"].feature_provider is not loaded.members["b"].feature_provider
    assert [m.feature_provider.scale for m in loaded.members.values()] == [0.1, 0.3]
    pd.testing.assert_frame_equal(homes, before)
    metadata = model.metadata
    metadata["nested"]["value"] = 2
    assert model.metadata["nested"]["value"] == 1


def test_empty_missing_columns_and_leakage(tmp_path, track, homes):
    model = adapter(tmp_path, track)
    assert model.predict(homes.iloc[:0]).shape == (0,)
    with pytest.raises(ValueError, match="Missing required"):
        model.predict(homes.drop(columns="signal"))
    for column in ("sold_price", "date_sold"):
        for frame in (homes, homes.iloc[:0]):
            with pytest.raises(ValueError, match="Hide actual"):
                model.predict(frame.assign(**{column: 1}))
    with pytest.raises(TypeError, match="DataFrame"):
        model.predict([])
    with pytest.raises(ValueError, match="unique"):
        model.predict(pd.concat([homes, homes[["signal"]]], axis=1))


@pytest.mark.parametrize("price", [np.nan, np.inf, -np.inf, 0, -1, "bad", None])
def test_invalid_list_prices(tmp_path, homes, price):
    model = adapter(tmp_path, "list_aware")
    with pytest.raises(ValueError, match="finite and positive"):
        model.predict(homes.assign(list_price=price))


def test_list_price_required_but_not_off_market(tmp_path, homes):
    for track in serving.TRACKS:
        model = adapter(tmp_path, track)
        frame = homes.drop(columns="list_price")
        if track == "list_aware":
            with pytest.raises(ValueError, match="requires list_price"):
                model.predict(frame)
        else:
            assert model.predict(frame).shape == (3,)


def test_missing_artifacts_and_no_fallback(tmp_path, track):
    other = next(t for t in serving.TRACKS if t != track)
    save_model(bundle(other), "release", track=other, save_dir=tmp_path)
    save_model(bundle(track), "release", save_dir=tmp_path)  # legacy is not a fallback
    model = serving.TrainedEnsembleModel(track, "release", save_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="metadata"):
        model.load()
    folder = save_model(bundle(track), "release", track=track, save_dir=tmp_path)
    (folder / "model.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="model.joblib"):
        model.load()
    assert model._model is None


@pytest.mark.parametrize("field,value", [("storage_track", "wrong"), ("name", "wrong"),
    ("format_version", 99), ("metadata", []), ("metadata", {"track": "wrong"}),
    ("metadata", {"tracks": ["off_market", "list_aware"]})])
def test_incompatible_metadata_rejected_before_deserialization(tmp_path, track, field, value):
    model = adapter(tmp_path, track)
    path = tmp_path / track / "release" / "metadata.json"
    info = json.loads(path.read_text())
    info[field] = value
    path.write_text(json.dumps(info))
    with patch.object(serving, "load_model") as loader:
        with pytest.raises(ValueError):
            model.load()
        loader.assert_not_called()


@pytest.mark.parametrize("kind", ["wrong_track", "arbitrary", "collection", "incomplete"])
def test_incompatible_objects(tmp_path, track, kind):
    invalid = {"wrong_track": bundle("list_aware" if track == "off_market" else "off_market"),
               "arbitrary": ArbitraryPredictor(), "collection": [bundle(track)]}
    if kind == "incomplete":
        obj = bundle(track)
        obj.members["a"].feature_provider = None
    else:
        obj = invalid[kind]
    with pytest.raises(TypeError):
        adapter(tmp_path, track, obj).load()


@pytest.mark.parametrize("weights", [{}, {"a": 1}, {"a": -1, "b": 2},
    {"a": 0, "b": 0}, {"a": np.nan, "b": 1}, {"a": np.inf, "b": 1}])
def test_corrupt_weights_are_rejected_not_repaired(tmp_path, track, weights):
    obj = bundle(track)
    obj.weights = weights
    with pytest.raises(ValueError, match="weight"):
        adapter(tmp_path, track, obj).load()
    assert obj.weights is weights


def test_provider_reordering_and_input_mutation(tmp_path, track, homes):
    model = adapter(tmp_path, track, bundle(track, reverse=True))
    # Unique indexes make the provider's reversal observable.
    with pytest.raises(ValueError, match="reordered"):
        model.predict(homes.reset_index(drop=True))
    mutate_dir = tmp_path / "mutating"
    model = adapter(mutate_dir, track, bundle(track, mutate=True))
    before = homes.copy(deep=True)
    model.predict(homes)
    pd.testing.assert_frame_equal(homes, before)


@pytest.mark.parametrize("bad", [np.array([1.]), np.ones((3, 1)), np.zeros(3),
                               np.array([1., np.nan, 2.]), np.array([1., np.inf, 2.])])
def test_invalid_final_predictions(tmp_path, homes, bad):
    model = adapter(tmp_path, "list_aware").load()
    with patch.object(model._model, "predict", return_value=bad):
        with pytest.raises(ValueError, match="invalid predicted prices"):
            model.predict(homes)


def test_fit_always_rejects_without_loading(tmp_path, track, homes):
    model = serving.TrainedEnsembleModel(track, "absent", save_dir=tmp_path)
    for train in (None, homes, homes.assign(sold_price=1, date_sold="2025-01-01")):
        with pytest.raises(RuntimeError, match="walk-forward"):
            model.fit(train)
    assert model._model is None


def test_missing_fitted_feature(tmp_path, track, homes):
    obj = bundle(track)
    member = obj.members["a"].members["a"] if track == "off_market" else obj.members["a"]
    member.selected_columns = ("absent_feature",)
    with pytest.raises(ValueError, match="missing.*feature"):
        adapter(tmp_path, track, obj).predict(homes)


def test_nonfinite_target_offset(tmp_path, track):
    obj = bundle(track)
    member = obj.members["a"].members["a"] if track == "off_market" else obj.members["a"]
    member.target_offset = np.nan
    with pytest.raises(ValueError, match="target offset"):
        adapter(tmp_path, track, obj).load()


def test_nonfinite_member_output(tmp_path, track, homes):
    model = adapter(tmp_path, track).load()
    member = (model._model.members["a"].members["a"] if track == "off_market"
              else model._model.members["a"])
    with patch.object(member.estimator, "predict", return_value=np.full(len(homes), np.nan)):
        with pytest.raises(ValueError, match="invalid"):
            model.predict(homes)


def test_two_track_interface(tmp_path, homes):
    for track in serving.TRACKS:
        adapter(tmp_path, track)
    interface = serving.EnsembleServingInterface(off_market_name="release", list_aware_name="release",
                                                  save_dir=tmp_path)
    assert interface.load() is interface
    for track in serving.TRACKS:
        np.testing.assert_allclose(interface.predict(track, homes), bundle(track).predict(homes))
        assert interface.for_track(track) is interface.models[track]
    with pytest.raises(ValueError, match="Unknown track"):
        interface.for_track("unknown")
    with pytest.raises(ValueError, match="Unsupported"):
        serving.TrainedEnsembleModel("unknown", "release")
