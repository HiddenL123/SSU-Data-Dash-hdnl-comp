"""Small integration and behavioural checks for the modular architecture."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")

import data
import models
import run
from models.arima import ArimaResidualModel
from models.catboost import CatBoostRegressor
from models.defaults import DEFAULT_PARAMS
from models.features import make_features
from models.legacy.gb_models import GBListAwareModel, GBOffMarketModel
from models.lightgbm import LGBMRegressor
from models.model import ListAwareModel, OffMarketModel
from models.xgboost import XGBRegressor
from test_starter import make_listings


CAT_PARAMS = {"iterations": 8, "depth": 3, "thread_count": 1}


def specs(arima=True):
    result = [(CatBoostRegressor, "Xpt_time", CAT_PARAMS)]
    if arima:
        result.append((ArimaResidualModel, "date_listed", {"order": (1, 1, 0)}))
    return result


class FractionOfMean:
    def __init__(self, fraction=0.5):
        self.fraction = fraction

    def fit(self, X, y):
        self.target = y.copy()
        self.value = self.fraction * np.mean(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def test_sequential_residuals_constructor_params_and_prediction_sum():
    train = make_listings(30)
    config = [(FractionOfMean, "date_listed", None),
              (FractionOfMean, "date_listed", {"fraction": 0.25})]
    model = OffMarketModel(config).fit(train)
    first, second = [c for c, _ in model.components]
    y = np.log(train.sold_price)
    np.testing.assert_allclose(first.target, y)
    np.testing.assert_allclose(second.target, y - 0.5 * y.mean())
    np.testing.assert_allclose(model.predict_log(train), 0.625 * y.mean())
    np.testing.assert_allclose(model.predict(train), np.exp(0.625 * y.mean()))
    assert model.components[0][1] is not model.components[1][1]


@pytest.mark.parametrize("backend", ["catboost", "lightgbm", "xgboost"])
@pytest.mark.parametrize("feature_name", ["Xpt", "Xpt_time"])
def test_features_reuse_vocabulary_categories_and_column_order(backend, feature_name):
    train = make_listings(50)
    train["remarks"] = ["renovated family" if i % 2 else "original home" for i in range(50)]
    provider = make_features(feature_name, model=backend)
    X = provider.fit_transform(train)
    state = provider.state
    vocabulary = deepcopy(state.vectorizer_.vocabulary_)
    idf = state.vectorizer_.idf_.copy()
    levels = deepcopy(state.category_levels_)
    homes = data.hide_answers(train.iloc[:3], "off_market").assign(
        community="Unseen community", remarks="brand new unseen words"
    )
    transformed = provider.transform(homes)
    assert list(transformed.columns) == list(X.columns)
    assert (transformed.community == "Unknown").all()
    assert state.vectorizer_.vocabulary_ == vocabulary
    np.testing.assert_array_equal(state.vectorizer_.idf_, idf)
    assert state.category_levels_ == levels
    if backend != "catboost":
        assert transformed.community.dtype == X.community.dtype
    assert {"longitude", "latitude"} <= set(X.columns)
    assert not {"sold_price", "date_sold", "log_list_price", "list_price"} & set(X.columns)
    assert ("months_since_2021" in X) == (feature_name == "Xpt_time")
    altered = train.assign(sold_price=1, list_price=1, date_sold=pd.Timestamp("2099-01-01"))
    pd.testing.assert_frame_equal(provider.transform(train), provider.transform(altered))


@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_predictions_match_legacy_for_equivalent_configurations(track):
    train = make_listings(450)
    homes = data.hide_answers(make_listings(20, seed=1), track)
    old_class = GBOffMarketModel if track == "off_market" else GBListAwareModel
    old = old_class(model_params=CAT_PARAMS).fit(train)
    new = models.make_model(track, specs()).fit(train)
    np.testing.assert_allclose(new.predict_log(homes), old.predict_log(homes), rtol=0, atol=1e-10)
    np.testing.assert_allclose(new.predict(homes), old.predict(homes), rtol=1e-10)
    if track == "list_aware":
        assert new.oof_count_ == old.oof_count_
        assert new.correction_columns == old.correction_columns
        np.testing.assert_allclose(new.correction_model.coef_, old.correction_model.coef_, atol=1e-10)


@pytest.mark.parametrize("arima", [False, True])
def test_catboost_with_optional_arima(arima):
    model = OffMarketModel(specs(arima)).fit(make_listings(200))
    homes = data.hide_answers(make_listings(10, seed=2), "off_market")
    assert np.isfinite(model.predict(homes)).all()
    assert len(model.components) == (2 if arima else 1)
    assert model.predict(homes.iloc[:0]).size == 0


@pytest.mark.parametrize("component,params", [
    (LGBMRegressor, {"n_estimators": 8, "n_jobs": 1}),
    (XGBRegressor, {"n_estimators": 8, "n_jobs": 1}),
])
def test_other_boosters_and_unknown_categories(component, params):
    model = OffMarketModel([(component, "Xpt_time", params)]).fit(make_listings(100))
    homes = data.hide_answers(make_listings(10, seed=2), "off_market").assign(community="Unseen")
    assert np.isfinite(model.predict(homes)).all()


@pytest.mark.parametrize("backend,component,extra", [
    ("catboost", CatBoostRegressor, CAT_PARAMS),
    ("lightgbm", LGBMRegressor, {"n_estimators": 8, "n_jobs": 1}),
    ("xgboost", XGBRegressor, {"n_estimators": 8, "n_jobs": 1}),
])
def test_native_classes_and_legacy_predictions(backend, component, extra):
    from importlib import import_module
    native_class = getattr(import_module(backend), component.estimator_name)
    params = {**DEFAULT_PARAMS[backend], **extra}
    train = make_listings(100)
    native = OffMarketModel([(native_class, "Xpt_time", params)]).fit(train)
    wrapped = OffMarketModel([(component, "Xpt_time", extra)]).fit(train)
    homes = data.hide_answers(make_listings(10, seed=2), "off_market").assign(community="Unseen")
    np.testing.assert_allclose(native.predict(homes), wrapped.predict(homes), rtol=1e-10)
    legacy = GBOffMarketModel(algorithm=backend, model_params=extra).fit(train)
    combined = OffMarketModel([
        (component, "Xpt_time", extra),
        (ArimaResidualModel, "date_listed", None),
    ]).fit(train)
    np.testing.assert_allclose(legacy.predict(homes), combined.predict(homes), rtol=1e-10)


def test_list_aware_folds_have_fresh_models_and_strict_chronology(monkeypatch):
    instances = []

    class RecordingBase(OffMarketModel):
        def fit(self, train):
            self.training_ids = set(train.ml_num)
            self.last_sale = train.date_sold.max()
            instances.append(self)
            return super().fit(train)

        def predict_log(self, homes):
            assert self.last_sale < homes.date_listed.min()
            assert self.training_ids.isdisjoint(homes.ml_num)
            return super().predict_log(homes)

    import models.model as module
    monkeypatch.setattr(module, "OffMarketModel", RecordingBase)
    train = make_listings(500)
    config = [(FractionOfMean, "Xpt_time", None)]
    model = ListAwareModel(config).fit(train)
    assert len(instances) == 4  # Three historical fits and the final base model.
    assert len({id(m.components[0][0]) for m in instances}) == 4
    assert len({id(m.components[0][1].state) for m in instances}) == 4
    assert instances[-1] is model.base_model
    assert len(model.base_model.training_ids) == len(train)


def test_defaults_match_existing_configuration_and_are_independent():
    first, second = models.make_model("off_market"), models.make_model("off_market")
    assert run.models.make_model is models.make_model
    assert len(first.components) == 2
    assert first.modelspecs[0][2] == DEFAULT_PARAMS["catboost"]
    assert first.modelspecs[0][1] == "Xpt_time"
    assert first.components[1][0].order == (1, 1, 0)
    first.modelspecs[0][2]["depth"] = 1
    assert second.modelspecs[0][2]["depth"] == 6


def test_original_cli_workflow_with_configured_factory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    listings = make_listings(700)
    test = make_listings(20, seed=1, first_id=5000)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "VALIDATION_MONTHS", pd.period_range("2026-04", "2026-05", freq="M"))
    monkeypatch.setattr(data, "load", lambda name: listings if name == "listings" else data.hide_answers(test, name))
    config = specs(False)
    monkeypatch.setattr(run, "models", SimpleNamespace(
        make_model=lambda track: models.make_model(track, modelspecs=config)
    ))
    assert run.validate() is None
    predictions = pd.read_csv(tmp_path / "outputs/validation_predictions.csv")
    assert isinstance(predictions, pd.DataFrame)
    assert set(predictions.track) == set(run.TRACKS)
    run.report()
    run.predict()
    assert run.check()
    assert (tmp_path / "figures/error_by_price_range.png").exists()


def test_invalid_inputs_and_unfitted_prediction():
    with pytest.raises(ValueError, match="Unknown competition track"):
        models.make_model("invalid")
    with pytest.raises(ValueError, match="at least one"):
        OffMarketModel([])
    with pytest.raises(ValueError, match="Unknown feature"):
        make_features("missing")
    model = OffMarketModel([(FractionOfMean, "date_listed", None)])
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(make_listings(1))
    with pytest.raises(ValueError, match="positive finite"):
        model.fit(make_listings(10).assign(sold_price=0))


def test_legacy_feature_module_can_still_be_loaded_by_notebook_file_path():
    import importlib.util
    from pathlib import Path
    from models.legacy.basic_features import tf_idf
    path = Path(__file__).resolve().parents[1] / "models/basic_features.py"
    spec = importlib.util.spec_from_file_location("project_basic_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.tf_idf is tf_idf


def test_notebook_validation_returns_dataframe_and_separate_fitted_models():
    import json
    from pathlib import Path
    from types import SimpleNamespace
    notebook = json.loads((Path(__file__).resolve().parents[1] / "model_run.ipynb").read_text(encoding="utf-8"))
    train = make_listings(700)
    namespace = {
        "pd": pd, "np": np,
        "data": SimpleNamespace(load=lambda _: train, hide_answers=data.hide_answers),
        "TRACKS": ["off_market", "list_aware"],
        "VALIDATION_MONTHS": pd.period_range("2026-04", "2026-05", freq="M"),
    }
    exec("".join(notebook["cells"][1]["source"]), namespace)
    predictions, fitted = namespace["validate"](
        modelspecs=[(FractionOfMean, "date_listed", None)]
    )
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.index.is_unique
    assert len(fitted) == 2 and all(len(pair) == 2 for pair in fitted)
    assert all(isinstance(pair[0], OffMarketModel) for pair in fitted)
