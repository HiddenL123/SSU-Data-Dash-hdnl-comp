"""No expensive fitting: validate list-aware model-pool screening and confirmation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import feature.listaware_pool_screen as pool


@pytest.fixture
def synthetic_sales():
    rows = []
    rng = np.random.default_rng(84)
    for month in pd.period_range("2024-10", "2025-07", freq="M"):
        for i in range(14):
            date = month.start_time + pd.Timedelta(days=12 + i % 9)
            price = float(400_000 + 1_100 * len(rows))
            rows.append({
                "ml_num": f"MLS_{str(month)}_{i}", "date_sold": date,
                "date_listed": date - pd.Timedelta(days=8),
                "sold_price": price, "list_price": price * 1.01,
                "house_type_name": "Detached" if i % 2 else "Condo Apt",
                "house_area": str(800 + i * 25),
                "latitude": 43.7 + float(rng.normal(0, .002)),
                "longitude": -79.4 + float(rng.normal(0, .002)),
                "bedrooms": 2, "washrooms": 2, "parking_total": 1,
                "municipality": "Toronto", "community": "Test",
                "house_style": "Two-Storey", "garage": "None",
                "remarks": "beautiful clean renovated home",
            })
    return pd.DataFrame(rows)


def test_candidate_catalog_is_factorized_and_complete():
    from features.features import make_features

    assert set(pool.CATALOG) == set(pool.SUITES["all"])
    assert len(pool.CATALOG) >= 35
    assert {"catboost", "xgboost", "lightgbm"} <= {
        x.backend for x in pool.CATALOG.values()
    }
    assert any(spec.target == "direct" for spec in pool.CATALOG.values())
    assert any(spec.feature_name.endswith("__full_binary")
               for spec in pool.CATALOG.values())
    assert any("comp_type_quota" in spec.feature_name
               for spec in pool.CATALOG.values())
    assert any("__st_geo_context" in spec.feature_name
               for spec in pool.CATALOG.values())
    for name, spec in pool.CATALOG.items():
        provider = make_features(spec.feature_name, model=spec.backend,
                                 track="list_aware")
        assert provider is not None, name
    for names in pool.SUITES.values():
        assert len(names) == len(set(names))
        assert names[0] == "cat_huber_baseline"


def test_confirmation_selection_uses_ranking_and_preserves_family_breadth():
    rows = []
    for i, name in enumerate(pool.SUITES["core"]):
        rows.append({
            "variant": name, "mdape": 5.0 + .08 * i,
            "mape": 6.0 + .04 * i,
        })
    scores = pd.DataFrame(rows).sample(frac=1, random_state=1)
    selected = pool.choose_confirmation(scores, top_k=8)
    assert set(pool.SEED_REFERENCE).issubset(selected)
    assert len(set(selected)) == len(selected)
    assert {"catboost", "xgboost", "lightgbm"} <= {
        pool.CATALOG[name].backend for name in selected
    }
    assert selected == pool.choose_confirmation(scores.iloc[::-1], top_k=8)
    with pytest.raises(ValueError, match="required references"):
        pool.choose_confirmation(scores.query("variant != 'cat_mae'"), top_k=8)


def test_screen_resumes_and_confirmation_only_sees_prior_months(
        tmp_path, monkeypatch, synthetic_sales
):
    observed = []

    def fake_predict(spec, train, hidden, *, tree_cap, threads, features_cache):
        assert "sold_price" not in hidden
        assert "date_sold" not in hidden
        assert "list_price" in hidden
        holdout_month = pd.to_datetime(hidden.date_listed).dt.to_period("M").min()
        assert (train.date_sold < holdout_month.start_time).all()
        observed.append((spec.name, len(train), len(hidden), tree_cap))
        return hidden.list_price.to_numpy(float) * (1.0 + .002 * len(spec.name))

    monkeypatch.setattr(pool, "_fit_predict", fake_predict)
    screen_dir = tmp_path / "screen"
    opts = {
        "stage": "screen", "suite": "core", "output_dir": screen_dir,
        "train_cap": 60, "validation_cap": 8, "tree_cap": 5,
        "seed": 42, "threads": 1, "bootstrap_repeats": 12,
        "listings": synthetic_sales,
    }
    scores, monthly, _, forecasts = pool.run(**opts)
    assert len(scores) == len(pool.SUITES["core"])
    assert set(monthly.month) == set(pool.SCREEN_MONTHS)
    assert len(forecasts) == len(pool.SCREEN_MONTHS) * 8 * len(pool.SUITES["core"])
    assert len(observed) == len(pool.SCREEN_MONTHS) * len(pool.SUITES["core"])
    first = len(observed)
    pool.run(**opts)
    assert len(observed) == first, "all model/month predictions must be reused"

    with pytest.raises(ValueError, match="Incompatible"):
        pool.run(**{**opts, "tree_cap": 6})

    confirm_dir = tmp_path / "confirmation"
    res = pool.run(
        stage="confirm", suite="core",
        screen_dir=screen_dir, output_dir=confirm_dir,
        seed=42, threads=1, bootstrap_repeats=12, top_k=8,
        listings=synthetic_sales,
    )
    assert set(res[1].month) == set(pool.CONFIRM_MONTHS)
    assert all(cap is None for _, _, _, cap in observed[first:])
    confirm_name = set(res[0].variant)
    assert confirm_name.issubset(set(scores.variant))
    assert set(pool.SEED_REFERENCE).issubset(confirm_name)
    with pytest.raises(ValueError, match="Screen dataset/protocol mismatch"):
        pool.run(
            stage="confirm", suite="core",
            screen_dir=screen_dir, output_dir=tmp_path / "bad",
            top_k=8, bootstrap_repeats=10,
            listings=synthetic_sales.assign(list_price=lambda d: d.list_price + 17),
        )


def test_cache_corruption_not_silently_reused(tmp_path, monkeypatch, synthetic_sales):
    def fake_predict(spec, train, hidden, **kw):
        return hidden.list_price.to_numpy(float) * 1.01

    monkeypatch.setattr(pool, "_fit_predict", fake_predict)
    args = {
        "suite": "core", "stage": "screen",
        "output_dir": tmp_path / "screen",
        "train_cap": 50, "validation_cap": 6, "tree_cap": 4,
        "threads": 1, "bootstrap_repeats": 5,
        "listings": synthetic_sales,
    }
    pool.run(**args)
    path = args["output_dir"] / "2025-03" / "models" / "cat_mae.csv"
    bad = pd.read_csv(path)
    bad.loc[0, "sold_price"] *= 2
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="stale|Invalid"):
        pool.run(**args)


def test_error_correlations_reject_mismatched_keys_and_preserve_interpretation():
    from listaware_shotgun import _frame

    held = pd.DataFrame({
        "ml_num": ["a", "b", "c", "d"],
        "sold_price": [100., 200., 300., 400.],
        "house_type_name": ["Detached"] * 4,
    })
    actual = held.sold_price.to_numpy(float)
    reference = _frame(held, pd.Period("2025-03"), "model",
                       "cat_huber_baseline", actual * np.array([.92, 1.06, .9, 1.05]))
    alternative = _frame(held, pd.Period("2025-03"), "model",
                         "cat_mae", actual * np.array([1.02, 1.01, 1.05, .98]))
    frame = pd.concat([reference, alternative], ignore_index=True)
    diag = pool.error_correlations(frame)
    assert len(diag) == 1
    assert set(diag.columns) == {
        "variant_a", "variant_b", "signed_log_error_correlation"
    }
    assert np.isfinite(diag.signed_log_error_correlation).all()

    broken = frame.copy()
    index = broken.loc[broken.variant.eq("cat_mae")].index[0]
    broken.loc[index, "sold_price"] += 10
    with pytest.raises(ValueError, match="Misaligned"):
        pool.error_correlations(broken)
