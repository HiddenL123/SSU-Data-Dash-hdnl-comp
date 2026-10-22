"""Integration tests for full-history comparisons and cheap combinations.

Synthetic data only; a real 700-tree / 10k-home benchmark requires user data.
"""
import numpy as np
import pandas as pd
import pytest

import data
from models.boost_regressor import XGBoostRegressor
from models.model import OffMarketModel
from models.features import make_features
from spacetime_full_feature_comparison import (
    compare_full_spacetime_models, _from_cache, _feature_spec,
)
from spacetime_combination_screen import run_combination_screen
from test_starter import make_listings


def _reference(sales, months=("2025-12", "2026-01")):
    spec = [(XGBoostRegressor, "Xpt_spatiotemporal__selected_binary", {
        "n_estimators": 6, "max_depth": 3, "n_jobs": 1,
        "tree_method": "hist", "random_state": 42,
        "enable_categorical": True,
    })]
    fitted, preds = [], []
    for month in pd.PeriodIndex(months, freq="M"):
        before = sales.loc[sales.date_sold < month.start_time]
        held = sales.loc[sales.date_sold.dt.to_period("M") == month]
        model = OffMarketModel(spec, center_target=True).fit(before)
        fitted.append([model, None])
        preds.append(pd.DataFrame({
            "month": str(month), "track": "off_market",
            "ml_num": held.ml_num.to_numpy(),
            "sold_price": held.sold_price.to_numpy(),
            "predicted_price": np.round(model.predict(data.hide_answers(held, "off_market"))),
        }))
    return pd.concat(preds, ignore_index=True), fitted


def _sales():
    return make_listings(1100, seed=41)


@pytest.mark.parametrize("suffix", [
    "__st_spacetime_splines", "__st_geo_context", "__st_combined",
    "__st_spacetime_splines+geo_context",
    "__comp_type_quota__st_spacetime_splines+geo_context",
])
def test_opt_in_features_keep_same_inference_schema(suffix):
    sales = _sales()
    train = sales.loc[sales.date_sold < "2025-08-01"]
    query = data.hide_answers(sales.loc[sales.date_sold >= "2025-08-01"].head(15), "off_market")
    provider = make_features("Xpt_spatiotemporal__selected_binary" + suffix, model="xgboost")
    X = provider.fit_transform(train)
    held = provider.transform(query)
    assert X.columns.tolist() == held.columns.tolist()
    assert len(X) == len(train) and len(held) == len(query)
    assert X.columns.is_unique
    assert not {"sold_price", "date_sold", "list_price"} & set(held.columns)


def test_full_history_compares_independently_and_resumes(tmp_path, monkeypatch):
    sales = _sales()
    months = ("2025-12", "2026-01")
    preds, models = _reference(sales, months)
    kwargs = dict(
        months=months, listings=sales, output_dir=tmp_path,
        feature_variants=("baseline", "spacetime_splines", "geo_context"),
        residual_variants=("tensor_gam",),
        calibration_months=1, calibration_cap=25,
        bootstrap_repeats=10,
    )
    pooled, monthly, by_type, rows = compare_full_spacetime_models(preds, models, **kwargs)
    assert len(pooled) == 4
    assert len(monthly) == 8
    assert set(pooled.variant) == {"baseline", "spacetime_splines", "geo_context", "tensor_gam"}
    assert rows.groupby(["month", "arm", "variant"]).size().nunique() == 1
    assert (tmp_path / "by_month/2026-01/feature__spacetime_splines.csv").is_file()

    # All candidates have been cached. Only baseline predictions need to be
    # recomputed; no feature model may be refitted on resume.
    import spacetime_full_feature_comparison as target
    class Forbidden:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Cached feature candidates must not retrain")
    monkeypatch.setattr(target, "OffMarketModel", Forbidden)
    monkeypatch.setattr(target, "_calibration_snapshot",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("cached residual")))
    replay = compare_full_spacetime_models(preds, models, **kwargs)
    np.testing.assert_allclose(pooled.mdape, replay[0].mdape)


def test_full_history_rejects_wrong_baseline_and_stale_cache(tmp_path):
    sales = _sales()
    preds, models = _reference(sales, ("2025-12",))
    kwargs = dict(months=("2025-12",), listings=sales, output_dir=tmp_path,
                  feature_variants=("baseline", "geo_context"),
                  residual_variants=(), bootstrap_repeats=0)
    changed = preds.copy()
    changed.loc[changed.index[0], "predicted_price"] += 200_000
    with pytest.raises(ValueError, match="do not reproduce"):
        compare_full_spacetime_models(changed, models, **kwargs)
    compare_full_spacetime_models(preds, models, **kwargs)
    with pytest.raises(ValueError, match="Cached output"):
        compare_full_spacetime_models(preds, models, **{**kwargs, "seed": 99})


def test_lightweight_feature_and_spline_gam_combinations(tmp_path):
    sales = _sales()
    scores, rows, failures = run_combination_screen(
        listings=sales, validation_month="2025-08", train_cap=180,
        validation_cap=25, trees=5, calibration_cap=18,
        calibration_folds=1, output_dir=tmp_path,
        combinations=("baseline", "splines", "splines_geo",
                      "splines_tensor_gam"),
    )
    assert not failures
    assert set(scores.variant) == {
        "baseline", "splines", "splines_geo", "splines_tensor_gam"
    }
    assert rows.groupby("variant").size().nunique() == 1
    assert (tmp_path / "summary.csv").is_file()


def test_full_cache_rejects_invalid_or_corrupted_candidate(tmp_path):
    from spacetime_full_feature_comparison import _frame
    sales = _sales().head(12)
    path = tmp_path / "prediction.csv"
    frame = _frame(sales, "2025-08", "feature", "spacetime_splines",
                   np.full(len(sales), 800000.))
    frame.to_csv(path, index=False)
    assert _from_cache(path, sales, month="2025-08",
                       arm="feature", variant="spacetime_splines") is not None
    frame.loc[0, "sold_price"] = 1
    frame.to_csv(path, index=False)
    assert _from_cache(path, sales, month="2025-08",
                       arm="feature", variant="spacetime_splines") is None