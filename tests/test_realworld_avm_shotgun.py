"""No-network synthetic regression checks for non-neural AVM research features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.realworld_avm_features import (
    HedonicNeighborhoodFeatures, LaggedTileMarketFeatures, VARIANTS,
)
from features.spacetime_screen_features import FEATURE_VARIANTS, SpaceTimeAddon
from feature.avm_realworld_shotgun import (
    _assemble_reference, _score_all, run_realworld_shotgun,
)


@pytest.fixture()
def housing():
    rng = np.random.default_rng(123)
    n = 50 * 12
    periods = np.repeat(np.arange(12), 50)
    sold = pd.to_datetime("2024-01-10") + pd.to_timedelta(periods * 31, unit="D")
    # Each listing occurs in the same month as its sale for the leakage test.
    listed = sold - pd.to_timedelta(4, unit="D")
    area = rng.integers(650, 2200, n)
    types = np.where(np.arange(n) % 3, "Detached", "Condo Apt")
    prices = np.exp(6.2 + np.log(area) + .11 * (types == "Detached")
                    + periods * .006 + rng.normal(0, .07, n))
    return pd.DataFrame({
        "ml_num": [f"T{i:04}" for i in range(n)],
        "date_listed": listed, "date_sold": sold,
        "sold_price": prices, "house_area": area.astype(str),
        "house_type_name": types,
        "latitude": 43.65 + rng.normal(0, .008, n),
        "longitude": -79.4 + rng.normal(0, .009, n),
        "bedrooms": rng.integers(1, 5, n),
        "washrooms": rng.integers(1, 4, n),
        "parking_total": rng.integers(0, 3, n),
        "remarks": "test property",
    })


def _query(housing, month=6):
    month_start = pd.Period("2024-01", freq="M") + month
    month_mask = housing.date_listed.dt.to_period("M") == month_start
    return housing.loc[month_mask].iloc[:6].drop(
        columns=["sold_price", "date_sold"], errors="ignore"
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_realworld_features_have_no_sale_columns_and_finite_schema(housing, variant):
    assert variant in FEATURE_VARIANTS
    query = _query(housing)
    addon = SpaceTimeAddon(variant).fit(housing)
    a = addon.transform(query, pd.DataFrame(index=query.index))
    assert len(a) == len(query)
    assert a.index.equals(query.index)
    assert a.shape[1] > 0
    assert a.columns.is_unique
    assert all(pd.api.types.is_float_dtype(a[name]) for name in a.columns)
    assert not np.isinf(a.to_numpy(float)).any()


@pytest.mark.parametrize("variant", VARIANTS)
def test_future_sale_price_perturbation_does_not_change_past_features(housing, variant):
    query = _query(housing)
    month = query.date_listed.dt.to_period("M").iloc[0].start_time
    old = SpaceTimeAddon(variant).fit(housing).transform(query, None)
    altered = housing.copy()
    altered.loc[altered.date_sold >= month, "sold_price"] *= 4
    changed = SpaceTimeAddon(variant).fit(altered).transform(query, None)
    np.testing.assert_allclose(old.to_numpy(), changed.to_numpy(), rtol=0,
                               atol=1e-5, equal_nan=True)


def test_local_hedonic_residuals_are_out_of_time(housing):
    fitted = HedonicNeighborhoodFeatures(min_history=50).fit(housing)
    # All OOT residuals associated with month m must have been generated from
    # coefficients trained on months < m. Altering same-month sold prices
    # cannot change the features for query homes listed in month m.
    query = _query(housing)
    month = query.date_listed.dt.to_period("M").iloc[0].start_time
    one = fitted.transform(query)
    changed_sales = housing.copy()
    changed_sales.loc[changed_sales.date_sold >= month, "sold_price"] *= 100
    two = HedonicNeighborhoodFeatures(min_history=50).fit(changed_sales).transform(query)
    np.testing.assert_allclose(one.to_numpy(), two.to_numpy(), rtol=0,
                               atol=1e-5, equal_nan=True)


def test_tiles_use_only_prior_sales(housing):
    query = _query(housing)
    model = LaggedTileMarketFeatures().fit(housing)
    tile = model.transform(query)
    assert tile.shape == (len(query), 6)
    assert (tile.filter(like="_count").to_numpy() >= 0).all()


def test_reference_must_cover_exact_homes_and_prices():
    candidate = pd.DataFrame({
        "month": ["2025-08", "2025-08"],
        "ml_num": ["A", "B"],
        "arm": ["feature", "feature"],
        "variant": ["baseline", "baseline"],
        "sold_price": [100., 200.],
        "predicted_price": [110., 180.],
    })
    ensemble = pd.DataFrame({
        "month": ["2025-08", "2025-08"],
        "ml_num": ["A", "B"],
        "sold_price": [100., 200.],
        "predicted_price": [105., 210.],
    })
    merged = _assemble_reference(candidate, ensemble)
    assert len(merged) == 4
    summary, monthly = _score_all(merged, repeats=10)
    assert len(summary) == 2
    assert len(monthly) == 2
    assert summary.loc[summary.arm == "frozen_reference", "delta_ensemble_mdape_pp"].iloc[0] == 0
    with pytest.raises(ValueError, match="lacks matching"):
        _assemble_reference(candidate, ensemble.iloc[:1])
    different = ensemble.copy()
    different.loc[0, "sold_price"] += 1
    with pytest.raises(ValueError, match="actual prices differ"):
        _assemble_reference(candidate, different)


def test_research_runner_resumes_completed_month_without_training(
    tmp_path, monkeypatch, housing
):
    import feature.avm_realworld_shotgun as runner

    # Build a valid 1-month audit reference matching the mocked screen cohort.
    chosen = housing.loc[housing.date_sold.dt.to_period("M") == pd.Period("2024-06")]
    assert len(chosen)
    ref = pd.DataFrame({
        "month": "2024-06", "ml_num": chosen.ml_num.astype(str),
        "sold_price": chosen.sold_price,
        "predicted_price": chosen.sold_price * 1.03,
    })
    path = tmp_path / "reference.csv"
    ref.to_csv(path, index=False)
    counter = [0]

    def fake_screen(*, validation_month, features, output_dir, **_):
        counter[0] += 1
        rows = []
        for variant in features:
            part = ref.copy()
            part["arm"], part["variant"] = "feature", variant
            part["predicted_price"] *= 1.01 if variant == "baseline" else .99
            rows.append(part)
        predictions = pd.concat(rows, ignore_index=True)
        root = __import__("pathlib").Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(root / "predictions.csv", index=False)
        return pd.DataFrame(), predictions, []

    monkeypatch.setattr(runner, "run_shotgun", fake_screen)
    opts = dict(months=["2024-06"], features=["baseline", "rw_robust_comps"],
                reference_file=path, output_dir=tmp_path / "results",
                sales=housing, bootstrap_repeats=10)
    runner.run_realworld_shotgun(**opts)
    runner.run_realworld_shotgun(**opts)
    assert counter[0] == 1
    with pytest.raises(ValueError, match="NEW output"):
        runner.run_realworld_shotgun(**{**opts, "trees": 221})
