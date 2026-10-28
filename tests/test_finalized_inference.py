"""Synthetic, CPU-only integration checks for FINALIZED off-market and list-aware bundles."""
import numpy as np
import pandas as pd
import pytest

import run
from check_submission import check_all, check_file
from models.ensemble_snapshot import BaggedMember, FeatureBaggingEnsemble
from models.final_ensemble import WeightedOffMarketEnsemble
from models.listaware_factory_snapshot import ListAwareBaggedMember, ListAwareFactoryEnsemble
from models.ensemble_serving import EnsembleServingInterface, TrainedEnsembleModel
from models.save_model import load_model, save_model


class StaticFeatures:
    def transform(self, homes):
        return pd.DataFrame({"constant": np.ones(len(homes))}, index=homes.index)


class ConstantLogEstimator:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value)


@pytest.fixture
def stored_models(tmp_path):
    def off_member(name, price):
        member = BaggedMember(
            name=name, estimator=ConstantLogEstimator(np.log(price)),
            selected_columns=("constant",), target_offset=0.,
            terms=(), settings={},
        )
        return FeatureBaggingEnsemble(StaticFeatures(), {name: member})

    off = WeightedOffMarketEnsemble(
        {"a": off_member("a", 100.), "b": off_member("b", 400.)},
        {"a": 0.25, "b": 0.75},
    )
    aware = ListAwareFactoryEnsemble(
        {
            "c": ListAwareBaggedMember(
                name="c", estimator=ConstantLogEstimator(np.log(1.10)),
                feature_provider=StaticFeatures(), selected_columns=("constant",),
                target_offset=0., terms=(), settings={},
            ),
            "d": ListAwareBaggedMember(
                name="d", estimator=ConstantLogEstimator(np.log(0.90)),
                feature_provider=StaticFeatures(), selected_columns=("constant",),
                target_offset=0., terms=(), settings={},
            ),
        },
        {"c": 0.25, "d": 0.75},
    )
    save_model(off, "off_market_release_candidate_full",
               track="off_market", save_dir=tmp_path)
    save_model(aware, "list_aware_ensemble_v1",
               track="list_aware", save_dir=tmp_path)
    return tmp_path


def serving(root):
    return EnsembleServingInterface(
        off_market_name="off_market_release_candidate_full",
        list_aware_name="list_aware_ensemble_v1",
        save_dir=root,
    )


def test_real_joblib_roundtrip_and_member_blending(stored_models):
    facade = serving(stored_models)
    a = pd.DataFrame({"ml_num": ["a", "b"]}, index=[5, 1])
    b = pd.DataFrame({"ml_num": ["x", "y"], "list_price": [100., 200.]},
                     index=[4, 2])
    for track, data, name in (
        ("off_market", a, "off_market_release_candidate_full"),
        ("list_aware", b, "list_aware_ensemble_v1"),
    ):
        direct = load_model(name, track=track, save_dir=stored_models)
        np.testing.assert_allclose(facade.predict(track, data), direct.predict(data))
    np.testing.assert_allclose(
        facade.predict("off_market", a),
        [100 ** .25 * 400 ** .75] * 2,
    )
    np.testing.assert_allclose(
        facade.predict("list_aware", b),
        np.array([100., 200.]) * (1.10 ** .25 * .90 ** .75),
    )
    assert a.index.tolist() == [5, 1] and b.index.tolist() == [4, 2]
    assert facade.for_track("list_aware").metadata["track"] == "list_aware"


def test_cache_and_no_pretrained_refitting(stored_models, monkeypatch):
    import models.ensemble_serving as m
    original = m.load_model
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(m, "load_model", counted)
    model = TrainedEnsembleModel("off_market", "off_market_release_candidate_full",
                                 save_dir=stored_models)
    model.predict(pd.DataFrame({"ml_num": ["a"]}))
    model.predict(pd.DataFrame({"ml_num": ["b"]}))
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="Pretrained ensembles"):
        model.fit(pd.DataFrame({"sold_price": [100]}))


@pytest.mark.parametrize("price", [0., -1., np.nan, np.inf])
def test_invalid_list_price(stored_models, price):
    with pytest.raises(ValueError, match="List prices"):
        serving(stored_models).predict(
            "list_aware", pd.DataFrame({"list_price": [price]})
        )


def test_track_leakage_wrong_types_and_missing_files(stored_models):
    facade = serving(stored_models)
    with pytest.raises(ValueError, match="list_price"):
        facade.predict("list_aware", pd.DataFrame({"ml_num": ["x"]}))
    with pytest.raises(ValueError, match="list_price"):
        facade.predict("off_market", pd.DataFrame({"list_price": [123]}))
    with pytest.raises(ValueError, match="Hide actual sale"):
        facade.predict("off_market", pd.DataFrame({"sold_price": [123]}))
    with pytest.raises(ValueError, match="Unknown track"):
        facade.predict("wrong", pd.DataFrame({"ml_num": ["x"]}))
    with pytest.raises(FileNotFoundError):
        TrainedEnsembleModel("off_market", "absent", save_dir=stored_models).load()
    save_model({"not": "an ensemble"}, "wrong_type", track="off_market",
               save_dir=stored_models)
    with pytest.raises(TypeError, match="WeightedOffMarketEnsemble"):
        TrainedEnsembleModel("off_market", "wrong_type",
                            save_dir=stored_models).load()


def test_legacy_spacetime_snapshot_backwards_compatibility():
    from features.spacetime_screen_features import SpaceTimeAddon
    old = SpaceTimeAddon.__new__(SpaceTimeAddon)
    old.__setstate__({
        "variant": "baseline", "market": None, "comps": {},
        "scaler": None, "splines": None, "column_names_": None,
    })
    assert old.realworld is None and old.advanced is None
    homes = pd.DataFrame({"ml_num": ["x"]})
    assert old.transform(homes, pd.DataFrame(index=homes.index)).empty


def test_run_predict_generates_both_competition_files(stored_models, tmp_path, monkeypatch):
    datasets = {
        "off_market": pd.DataFrame({"ml_num": ["a", "b"]}),
        "list_aware": pd.DataFrame({
            "ml_num": ["x", "y"], "list_price": [100., 200.],
        }),
    }
    def data_load(track):
        assert track != "listings", "Saved-model prediction must not read training data"
        return datasets[track].copy()
    monkeypatch.setattr(run.data, "load", data_load)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setenv("SAVED_MODELS_DIR", str(stored_models))
    run.predict()
    for track in run.TRACKS:
        path = tmp_path / "submissions" / f"submission_{track}.csv"
        assert path.is_file()
        assert check_file(path, datasets[track].ml_num.astype(str)) == []
        assert pd.read_csv(path, dtype={"ml_num": str}).ml_num.tolist() == (
            datasets[track].ml_num.tolist()
        )
    assert check_all(tmp_path / "submissions")


def test_no_partial_submission_when_second_model_missing(stored_models, tmp_path, monkeypatch):
    def load(track):
        if track == "off_market":
            return pd.DataFrame({"ml_num": ["a"]})
        raise AssertionError("The second model should fail before test data is read")
    monkeypatch.setattr(run.data, "load", load)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setenv("SAVED_MODELS_DIR", str(stored_models))
    monkeypatch.setenv("LIST_AWARE_MODEL_NAME", "does_not_exist")
    with pytest.raises(FileNotFoundError):
        run.predict()
    assert not (tmp_path / "submissions" / "submission_off_market.csv").exists()


def test_invalid_listaware_submission_does_not_overwrite_other(stored_models, tmp_path, monkeypatch):
    def load(track):
        if track == "off_market":
            return pd.DataFrame({"ml_num": ["a"]})
        if track == "list_aware":
            return pd.DataFrame({"ml_num": ["x"], "list_price": [np.nan]})
        raise AssertionError("Training data must not be loaded")
    monkeypatch.setattr(run.data, "load", load)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setenv("SAVED_MODELS_DIR", str(stored_models))
    with pytest.raises(ValueError, match="List prices"):
        run.predict()
    assert not (tmp_path / "submissions" / "submission_off_market.csv").exists()


def test_validate_remains_chronological_baseline_only():
    import inspect
    source = inspect.getsource(run.validate)
    assert "models.make_model(track).fit(train)" in source
    assert "EnsembleServingInterface" not in source
