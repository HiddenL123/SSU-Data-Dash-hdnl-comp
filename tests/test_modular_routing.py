"""Routing checks for added modules and notebooks; upstream files stay untouched."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import data
import models.gb_models as compatibility
from models import make_model
from models.arima import ArimaResidualModel
from models.catboost import CatBoostRegressor
from models.legacy import gb_models as legacy
from models.model import OffMarketModel, ListAwareModel
from test_starter import make_listings


ROOT = Path(__file__).resolve().parents[1]


def notebook_source(name, cell):
    notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return "".join(notebook["cells"][cell]["source"])


@pytest.mark.parametrize("algorithm,params", [
    ("catboost", {"iterations": 8, "depth": 3, "thread_count": 1}),
    ("lightgbm", {"n_estimators": 8, "n_jobs": 1}),
    ("xgboost", {"n_estimators": 8, "n_jobs": 1}),
])
@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_old_boosting_entry_points_route_to_modular_with_equivalent_predictions(algorithm, params, track):
    kwargs = dict(algorithm=algorithm, model_params=params, use_remarks=False,
                  max_text_features=30, arima_order=(1, 1, 0))
    if track == "list_aware":
        kwargs.update(correction_alpha=7.0, n_folds=2)
    train = make_listings(450)
    homes = data.hide_answers(make_listings(10, seed=1), track)
    new = compatibility.make_model(track, **kwargs).fit(train)
    old = legacy.make_model(track, **kwargs).fit(train)
    expected = OffMarketModel if track == "off_market" else ListAwareModel
    assert isinstance(new, expected)
    np.testing.assert_allclose(new.predict(homes), old.predict(homes), rtol=1e-10)
    base = new if track == "off_market" else new.base_model
    assert not base.features.use_remarks
    assert not any(c.startswith("tfidf_") for c in base.features.columns_)
    assert len(base.feature_importance()) == len(base.features.columns_)
    if track == "list_aware":
        assert new.oof_count_ == old.oof_count_


def test_direct_boosting_classes_and_new_specs_are_routed():
    assert issubclass(compatibility.GBOffMarketModel, OffMarketModel)
    assert issubclass(compatibility.GBListAwareModel, ListAwareModel)
    config = [(CatBoostRegressor, "Xpt_time", {"iterations": 8})]
    assert len(compatibility.make_model("off_market", modelspecs=config).components) == 1
    assert not issubclass(legacy.GBOffMarketModel, OffMarketModel)
    with pytest.raises(ValueError, match="Unknown algorithm"):
        compatibility.make_model("off_market", algorithm="bad")


def test_interaction_discovery_and_text_diagnostics_use_modular_baseline():
    listings = make_listings(700)
    namespace = {
        "data": SimpleNamespace(load=lambda _: listings, hide_answers=data.hide_answers),
        "pd": pd, "np": np, "make_model": make_model,
        "MODEL_SPECS": [
            (CatBoostRegressor, "Xpt_time", {"iterations": 8, "thread_count": 1}),
            (ArimaResidualModel, "date_listed", None),
        ],
        "DISCOVERY_MONTHS": pd.period_range("2026-04", "2026-05", freq="M"),
        "display": lambda value: None,
    }
    exec(notebook_source("interaction.ipynb", 2), namespace)
    discovery = namespace["discovery"]
    model = namespace["model"]
    assert isinstance(model, OffMarketModel)
    assert set(discovery.month) == {"2026-04", "2026-05"}
    homes = listings.loc[listings.date_sold.dt.to_period("M") == pd.Period("2026-05")]
    expected = np.log(homes.sold_price.to_numpy()) - model.predict_log(data.hide_answers(homes, "off_market"))
    np.testing.assert_allclose(discovery.loc[discovery.month == "2026-05", "log_residual"], expected)
    exec(notebook_source("interaction.ipynb", 10), namespace)
    assert list(namespace["selected_words"].columns) == ["Word", "Importance"]
    assert len(namespace["selected_words"]) == len(model.components[0][1].state.text_feature_names)
    assert "basic_models" not in notebook_source("interaction.ipynb", 6)


def test_training_notebook_publishes_output_for_analysis_notebook(tmp_path, monkeypatch):
    listings = make_listings(20)
    predictions = listings[["ml_num", "sold_price", "house_type_name"]].assign(
        track="off_market", month="2026-04", predicted_price=listings.sold_price * 1.01
    )
    def validate(*, modelspecs):
        assert modelspecs[0][0] is CatBoostRegressor
        assert modelspecs[1][0] is ArimaResidualModel
        return predictions, []

    namespace = {"ROOT": tmp_path, "validate": validate}
    exec(notebook_source("model_run.ipynb", 2), namespace)
    assert (tmp_path / "outputs/validation_predictions.csv").exists()
    monkeypatch.chdir(tmp_path)
    namespace = {
        "Path": Path, "pd": pd, "np": np,
        "data": SimpleNamespace(load=lambda _: listings),
    }
    exec(notebook_source("notebook.ipynb", 2), namespace)
    exec(notebook_source("notebook.ipynb", 4), namespace)
    pd.testing.assert_frame_equal(namespace["predictions"], predictions, check_dtype=False)
    assert len(namespace["plot_df"]) == len(predictions)


@pytest.mark.parametrize("name,cell", [("model_run.ipynb", 2), ("interaction.ipynb", 0)])
def test_notebook_specs_exactly_match_legacy_configuration(name, cell):
    import ast
    # Evaluate configuration only, without running expensive validation.
    tree = ast.parse(notebook_source(name, cell))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "MODEL_SPECS" for t in node.targets))
    namespace = {"CatBoostRegressor": CatBoostRegressor, "ArimaResidualModel": ArimaResidualModel}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), name, "exec"), namespace)
    config = namespace["MODEL_SPECS"]
    assert config == [
        (CatBoostRegressor, "Xpt_time", legacy.DEFAULT_PARAMS["catboost"]),
        (ArimaResidualModel, "date_listed", {"order": (1, 1, 0)}),
    ]
    new = make_model("off_market", modelspecs=config)
    old = legacy.GBOffMarketModel()
    assert new.components[0][0].estimator.get_params() == old.property_model.model.get_params()
    provider = new.components[0][1].state
    assert provider.include_time == old.features.include_time
    assert provider.use_remarks == old.features.use_remarks
    assert provider.max_text_features == old.features.max_text_features
    assert new.components[1][0].order == old.temporal_model.order
