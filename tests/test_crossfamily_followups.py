"""Unit tests for the two follow-up cross-family blending experiments."""
import numpy as np
import pandas as pd
import pytest

from listaware_fixed_origin_followup import (
    ADDITIONAL_BLENDS, append_blends, report_followup,
)
from offmarket_crossfamily_blend import (
    BLENDS, _compare, _cohort, _frame, _validate_reference, run_offmarket_blend,
)


def _synthetic_listaware_predictions():
    base = pd.DataFrame({
        "month": ["2025-08", "2025-08", "2025-09"],
        "ml_num": ["10", "20", "30"],
        "house_type_name": ["Detached", "Condo", "Detached"],
        "sold_price": [100., 200., 300.],
    })
    predictions = []
    value = {
        "cat_huber_baseline": [110., 210., 290.],
        "cat_mae": [105., 190., 320.],
        "cat_huber_spline_geo": [106., 190., 295.],
        "cat_huber_advanced": [108., 195., 308.],
        "xgb_spline_geo": [95., 230., 310.],
        "lgb_spline_geo": [98., 210., 305.],
    }
    for name, p in value.items():
        df = base.copy()
        df["arm"], df["variant"] = "model", name
        df["predicted_price"] = p
        df["ape"] = 100 * abs(df.predicted_price - df.sold_price)/df.sold_price
        predictions.append(df)
    source = pd.concat(predictions, ignore_index=True)
    result = append_blends(source)
    actual = result.loc[result.variant == "followup_winner_plus_mae"].sort_values(
        ["month", "ml_num"]
    )
    weights = ADDITIONAL_BLENDS["followup_winner_plus_mae"]
    expected = np.exp(sum(
        w * np.log(np.asarray(value[name], dtype=float)) for name,w in weights.items()
    ))
    np.testing.assert_allclose(actual.predicted_price.to_numpy(float), expected, rtol=1e-12)
    assert result.groupby(["arm","variant"]).size().eq(3).all()
    return source


def test_blend_math_and_exact_pairing():
    _synthetic_listaware_predictions()


def test_misaligned_listaware_source_fails():
    source = _synthetic_listaware_predictions()
    mismatched = source.copy()
    xgb_mask = mismatched.variant == "xgb_spline_geo"
    mismatched.loc[xgb_mask, "sold_price"] += 100.
    with pytest.raises(ValueError, match="not aligned"):
        append_blends(mismatched)
    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        append_blends(duplicate)


def test_scores_only_writes_separate_predictions(tmp_path):
    source = _synthetic_listaware_predictions()
    result, monthly, by_type, combined = report_followup(
        source, tmp_path, bootstrap_repeats=10
    )
    assert "followup_winner_plus_mae" in result.variant.to_list()
    assert (tmp_path / "followup_pooled.csv").is_file()
    assert (tmp_path / "followup_predictions.csv").is_file()
    assert len(monthly) > 0 and len(combined) > len(source)


@pytest.fixture
def sales():
    rng = np.random.default_rng(128)
    n = 65 * 10
    months = np.repeat(np.arange(10), 65)
    sold = pd.Timestamp("2024-01-10") + pd.to_timedelta(months * 30, unit="D")
    area = rng.integers(600, 2000, n)
    price = np.exp(np.log(area) + 6.0 + months * .004 + rng.normal(0,.08,n))
    return pd.DataFrame({
        "ml_num": [f"A{i:04}" for i in range(n)],
        "date_sold": sold,
        "date_listed": sold-pd.Timedelta(days=4),
        "sold_price": price,
        "list_price": price * 1.05,
        "house_area": area.astype(str),
        "house_type_name": np.where(np.arange(n)%2, "Detached","Condo Apt"),
        "latitude": 43.66 + rng.normal(0,.005,n),
        "longitude": -79.38 + rng.normal(0,.005,n),
        "municipality": "Toronto", "community": "Synthetic",
        "house_style": "Two-Storey", "garage": "None",
        "bedrooms": rng.integers(1,5,n),
        "washrooms": rng.integers(1,4,n),
        "parking_total": rng.integers(0,3,n),
        "remarks": "renovated detached family home",
    })


def test_offmarket_reference_requires_matching_ids_and_dates(sales):
    month = "2024-08"
    chosen = sales.loc[sales.date_sold.dt.to_period("M").astype(str) == month]
    audit = pd.DataFrame({
        "month": month, "ml_num": chosen.ml_num.astype(str),
        "sold_price": chosen.sold_price,
        "predicted_price": chosen.sold_price * .99,
    })
    _validate_reference(audit, months=(month,))
    cohort, prediction = _cohort(sales, audit, month,
                                 max_validation=1_000, seed=42)
    assert len(cohort) == len(audit)
    np.testing.assert_allclose(prediction, audit.predicted_price)
    changed = audit.copy()
    changed.loc[changed.index[0], "sold_price"] *= 2
    with pytest.raises(ValueError, match="sale prices differ"):
        _cohort(sales, changed, month, max_validation=1_000, seed=42)
    wrong = audit.copy()
    wrong["month"] = "2025-08"
    with pytest.raises(ValueError, match="home/month mismatch"):
        _cohort(sales, wrong, "2025-08", max_validation=1_000, seed=42)


def test_frozen_reference_blend_math():
    assert np.isclose(sum(BLENDS["frozen_cat_lgb_70_20_10"].values()), 1.0)
    truth = pd.DataFrame({
        "month": ["2025-08","2025-08"],
        "ml_num": ["a","b"],
        "sold_price": [100.,200.],
    })
    frames = []
    for name, p in [("frozen_xgb_ensemble", [101.,201.]),
                    ("xgboost_offmarket", [102.,198.]),
                    ("catboost_offmarket", [104.,190.]),
                    ("lightgbm_offmarket", [97.,202.])]:
        part = truth.copy()
        part["arm"], part["variant"] = "model", name
        part["predicted_price"] = p
        frames.append(part)
    prediction = {frame.variant.iloc[0]:frame.predicted_price.to_numpy(float)
                  for frame in frames}
    for name, weights in BLENDS.items():
        p=np.exp(sum(weight*np.log(prediction[k]) for k,weight in weights.items()))
        part=truth.copy()
        part["arm"], part["variant"] = "blend", name
        part["predicted_price"] = p
        frames.append(part)
    pooled, monthly = _compare(pd.concat(frames,ignore_index=True),repeats=10)
    assert len(pooled)==len(BLENDS)+4
    assert len(monthly)==len(BLENDS)+4
    assert "delta_base_xgb_mdape_pp" in pooled
    assert "base_xgb_delta_mdape_ci_low" in pooled
    assert pooled.loc[
        pooled.variant=="xgboost_offmarket", "delta_base_xgb_mdape_pp"
    ].iloc[0]==0
    assert pooled.loc[
        pooled.variant=="frozen_xgb_ensemble","delta_mdape_pp"
    ].iloc[0]==0


def test_offmarket_runner_resumes_without_touching_frozen_reference(tmp_path, monkeypatch, sales):
    import offmarket_crossfamily_blend as runner
    month="2024-08"
    chosen=sales.loc[sales.date_sold.dt.to_period("M").astype(str)==month]
    audit=pd.DataFrame({
        "month":month, "ml_num":chosen.ml_num, "sold_price":chosen.sold_price,
        "predicted_price": chosen.sold_price * 1.01,
    })
    ref=tmp_path/"audit.csv"
    audit.to_csv(ref,index=False)
    original_bytes=ref.read_bytes()
    counter=[]
    def fake_model(train, homes, family, **kwargs):
        assert (train.date_sold < pd.Period(month).start_time).all()
        assert "list_price" not in homes and "sold_price" not in homes
        counter.append(family)
        return np.full(len(homes), float(np.median(train.sold_price)))
    monkeypatch.setattr(runner,"_model_prediction",fake_model)
    opts=dict(months=[month], reference_file=ref, listings=sales,
              output_dir=tmp_path/"out", train_cap=100, validation_cap=8,
              bootstrap_repeats=5)
    a,_,_=run_offmarket_blend(**opts)
    b,_,_=run_offmarket_blend(**opts)
    assert len(counter)==3
    assert "xgboost_offmarket" in counter
    assert ref.read_bytes()==original_bytes
    pd.testing.assert_frame_equal(a.reset_index(drop=True),b.reset_index(drop=True))
    with pytest.raises(ValueError,match="config changed"):
        run_offmarket_blend(**{**opts,"train_cap":99})


def test_fixed_origin_fits_once_across_multiple_months(tmp_path, monkeypatch, sales):
    import listaware_fixed_origin_followup as follow
    periods=("2024-07","2024-08")
    calls=[]
    def fake_fit(candidate, train, homes, **kwargs):
        calls.append(candidate.name)
        assert (train.date_sold < pd.Period(periods[0]).start_time).all()
        assert "sold_price" not in homes and "date_sold" not in homes
        # Preserve chronological query order; give all months one fitted model.
        return homes.list_price.to_numpy(float)*1.02
    monkeypatch.setattr(follow, "_fit_predict", fake_fit)
    opts=dict(months=periods,full=False,output_dir=tmp_path/"fixed",
              threads=1,seed=42,bootstrap_repeats=5,listings=sales)
    first=follow._run_single_fit_fixed_origin(**opts)
    assert len(calls)==len(follow.CANDIDATES)
    assert first.month.nunique()==2
    second=follow._run_single_fit_fixed_origin(**opts)
    assert len(calls)==len(follow.CANDIDATES)  # no refits when cached
    pd.testing.assert_frame_equal(first,second)
    with pytest.raises(ValueError,match="config changed"):
        follow._run_single_fit_fixed_origin(**{**opts,"seed":43})


def test_base_xgb_results_are_compared_against_two_references():
    """Cross-family blends may improve base XGB while harming frozen ensemble."""
    truth = pd.DataFrame({
        "month": ["2025-08"]*4,
        "ml_num": ["1","2","3","4"],
        "sold_price": [100.,200.,300.,400.],
    })
    members = {
        "frozen_xgb_ensemble": [101.,198.,304.,394.],
        "xgboost_offmarket": [110.,190.,330.,360.],
        "catboost_offmarket": [90.,215.,285.,440.],
        "lightgbm_offmarket": [104.,196.,320.,380.],
    }
    frames = []
    for name, preds in members.items():
        part = truth.copy()
        part["variant"], part["arm"] = name, "model"
        part["predicted_price"] = preds
        frames.append(part)
    for name, weights in BLENDS.items():
        part = truth.copy()
        part["variant"], part["arm"] = name, "blend"
        part["predicted_price"] = np.exp(sum(
            weight*np.log(members[family])
            for family,weight in weights.items()
        ))
        frames.append(part)
    pooled, monthly = _compare(pd.concat(frames,ignore_index=True),repeats=10)
    assert len(pooled) == len(BLENDS) + 4
    base_row = pooled.loc[pooled.variant=="xgboost_offmarket"].iloc[0]
    frozen_row = pooled.loc[pooled.variant=="frozen_xgb_ensemble"].iloc[0]
    assert base_row.delta_base_xgb_mdape_pp == pytest.approx(0)
    assert frozen_row.delta_mdape_pp == pytest.approx(0)
    assert frozen_row.delta_base_xgb_mdape_pp < 0
    assert all(name in set(pooled.variant) for name in BLENDS)
    assert not monthly.empty


def test_previous_family_caches_reused_for_base_xgb_followup(tmp_path,monkeypatch,sales):
    """Old pilot Cat/LGB predictions are safe to reuse, leaving only 1 XGB fit."""
    import offmarket_crossfamily_blend as runner
    month="2024-08"
    chosen=sales.loc[sales.date_sold.dt.to_period("M").astype(str)==month]
    audit=pd.DataFrame({
        "month":month,"ml_num":chosen.ml_num,
        "sold_price":chosen.sold_price,
        "predicted_price":chosen.sold_price*1.02,
    })
    ref=tmp_path/"audit.csv"
    audit.to_csv(ref,index=False)
    calls=[]
    def fake_model(train, homes, family, **kwargs):
        calls.append(family)
        return np.full(len(homes),float(np.median(train.sold_price)))
    monkeypatch.setattr(runner,"_model_prediction",fake_model)
    opts=dict(months=[month],reference_file=ref,listings=sales,
              train_cap=100,validation_cap=8,trees=6,threads=1,
              bootstrap_repeats=5)
    previous_folder=tmp_path/"previous"
    runner.run_offmarket_blend(**opts,output_dir=previous_folder)
    assert len(calls)==3
    # Construct the previous v1 manifest, reflecting the original source
    # experiment's recipe while keeping the old cache and labels untouched.
    manifest=previous_folder/"manifest.json"
    import json
    source=json.loads(manifest.read_text())
    source["version"]=1
    manifest.write_text(json.dumps(source))
    calls.clear()
    new_folder=tmp_path/"new"
    pooled, _, _=runner.run_offmarket_blend(
        **opts,output_dir=new_folder,reuse_family_caches=previous_folder,
    )
    assert calls == ["xgboost_offmarket"]
    assert (new_folder / month / "catboost_offmarket.csv").is_file()
    assert "base_xgb_plus_cat_80_20" in pooled.variant.to_list()
    with pytest.raises(ValueError,match="do not match"):
        runner.run_offmarket_blend(
            **{**opts,"train_cap":101},
            output_dir=tmp_path/"other",reuse_family_caches=previous_folder,
        )
