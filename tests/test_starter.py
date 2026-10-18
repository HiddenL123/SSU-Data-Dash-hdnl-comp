"""Quick checks on tiny made-up data (nothing is downloaded). Run from the repo folder:

    pytest
"""
import numpy as np
import pandas as pd
import pytest

import check_submission
import data
import models.ex_models as ex_models
import models.basic_features as basic_features
import models.basic_models as basic_models
import run


def test_bag_of_words_handles_repeated_words_and_duplicate_index_labels():
    remarks = pd.DataFrame(
        {"remarks": [
            "home home condo",
            "home condo condo",
            "spacious garden",
            "modern kitchen",
            "bright room",
            "quiet street",
            "large yard",
            "updated bath",
            "corner lot",
            "sunny deck",
        ]},
        index=[7, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )

    result = basic_features.do_bag_of_words(remarks)

    assert result.index.tolist() == [7, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert result.loc[7, "bow_home"].tolist() == [1, 1]
    assert result.loc[7, "bow_condo"].tolist() == [1, 1]


def test_text_features_fit_vectorizers_before_transforming():
    remarks = pd.DataFrame({
        "remarks": [
            "spacious condo balcony",
            "large detached home garden",
            "updated condo parking",
            "quiet detached home",
        ]
    })

    bow = basic_features.do_bag_of_words(remarks)
    tfidf = basic_features.tf_idf(remarks)

    assert len(bow) == len(remarks)
    assert len(tfidf) == len(remarks)
    assert all(column.startswith("bow_") for column in bow.columns)
    assert all(column.startswith("tfidf_") for column in tfidf.columns)


def make_listings(n, seed=0, first_id=0):
    """A tiny made-up listings table with the same columns as the real one."""
    rng = np.random.default_rng(seed)
    sold = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 516, n), unit="D")
    price = rng.uniform(4e5, 2e6, n).round(-2)
    return pd.DataFrame({
        "ml_num": [f"SYN-{first_id + i:07d}" for i in range(n)],
        "sold_price": price,
        "list_price": (price * rng.uniform(0.9, 1.1, n)).round(-2),
        "date_listed": sold - pd.to_timedelta(rng.integers(1, 60, n), unit="D"),
        "date_sold": sold,
        "days_on_market": 10,
        "house_type_name": rng.choice(["Condo Apt", "Detached"], n),
        "house_style": "Apartment",
        "bedrooms": rng.integers(1, 5, n),
        "washrooms": rng.integers(1, 4, n),
        "parking_total": rng.integers(0, 3, n),
        "garage": 0.0,
        "house_area": rng.choice(["600-699", "1200"], n),
        "lot_front": np.nan,
        "lot_depth": np.nan,
        "municipality": "Toronto",
        "community": "Testville",
        "latitude": 43.7 + rng.normal(0, 0.05, n),
        "longitude": -79.4 + rng.normal(0, 0.05, n),
        "external_avm_estimate": price,
        "external_avm_estimate_date": sold,
        "remarks": "A home.",
    })


def test_load_prefers_local_csv_without_DATA_DIR(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    source = pd.DataFrame({
        "ml_num": ["SYN-0000001"],
        "date_sold": ["2025-01-15"],
    })
    path = folder / "listings.csv"
    source.to_csv(path, index=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr(data, "LOCAL_DATA_DIR", folder)

    loaded = data.load("listings")

    assert data.find_file("listings") == path
    assert len(loaded) == 1
    assert loaded.loc[0, "date_sold"] == pd.Timestamp("2025-01-15")


def test_load_sample_returns_n_repeatable_random_rows(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    source = pd.DataFrame({"ml_num": [f"SYN-{i:07d}" for i in range(20)]})
    source.to_csv(folder / "listings.csv", index=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setattr(data, "LOCAL_DATA_DIR", folder)

    first = data.load_sample("listings", n=5, random_state=42)
    second = data.load_sample("listings", n=5, random_state=42)

    assert len(first) == 5
    assert first.index.is_unique
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("track", ["off_market", "list_aware"])
def test_models_api_fits_and_predicts_for_run_pipeline(track):
    listings = make_listings(120)
    listings["remarks"] = [
        f"spacious home with feature{i % 12} near park"
        for i in range(len(listings))
    ]
    train = listings.iloc[:100]
    homes = data.hide_answers(listings.iloc[100:], track)

    model = basic_models.make_model(track).fit(train)
    predictions = model.predict(homes)

    assert len(predictions) == len(homes)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


# ---- the nearest-sales baseline never looks into the future ----

def test_nearest_sales_ignores_later_sales():
    listings = make_listings(400)
    homes = data.hide_answers(make_listings(50, seed=1, first_id=1000), "off_market")
    homes = homes[homes["date_listed"] > pd.Timestamp("2025-03-01")]   # so earlier sales exist
    with_later_sales = ex_models.NearestSalesBaseline(k=10).fit(listings).predict(homes)
    for i, (_, home) in enumerate(homes.iterrows()):
        earlier_only = listings[listings["date_sold"] < home["date_listed"]]
        alone = ex_models.NearestSalesBaseline(k=10).fit(earlier_only).predict(homes.loc[[home.name]])
        assert alone[0] == with_later_sales[i]


def test_a_later_sale_next_door_is_not_used():
    listings = make_listings(400)
    home = listings.iloc[[0]].assign(ml_num="SYN-8888888", latitude=44.0, date_listed=pd.Timestamp("2026-01-15"))
    next_door = home.assign(ml_num="SYN-9999999", date_sold=pd.Timestamp("2026-02-01"), sold_price=1e9)
    model = ex_models.NearestSalesBaseline(k=1).fit(pd.concat([listings, next_door]))
    assert model.predict(data.hide_answers(home, "off_market"))[0] < 1e9


def test_nearest_sales_uses_the_nearest_homes_of_the_same_type():
    def sales(n, kind, latitude, price, first_id):
        rows = make_listings(n, first_id=first_id)
        return rows.assign(house_type_name=kind, latitude=latitude, longitude=-79.4,
                           sold_price=price, date_sold=pd.Timestamp("2025-01-01"))
    listings = pd.concat([sales(10, "Detached", 43.70, 500_000, 0),     # near, same type
                          sales(10, "Condo Apt", 43.70, 2_000_000, 100),  # near, other type
                          sales(10, "Detached", 43.90, 100_000, 200)])    # far, same type
    home = listings.iloc[[0]].assign(date_listed=pd.Timestamp("2026-01-01"))
    model = ex_models.NearestSalesBaseline(k=10).fit(listings)
    assert model.predict(data.hide_answers(home, "off_market"))[0] == 500_000


# ---- check_submission rejects bad files ----

IDS = ["SYN-0000001", "SYN-0000002", "SYN-0000003"]
GOOD = "ml_num,predicted_price\nSYN-0000001,500000\nSYN-0000002,600000\nSYN-0000003,700000\n"
BAD_FILES = {
    "wrong column name": GOOD.replace("predicted_price", "price"),
    "extra column": GOOD.replace("predicted_price\n", "predicted_price,x\n"),
    "a home missing": "ml_num,predicted_price\nSYN-0000001,500000\nSYN-0000002,600000\n",
    "a home twice": GOOD + "SYN-0000003,700000\n",
    "an id from another file": GOOD + "SYN-0000004,700000\n",
    "the other track's homes": "ml_num,predicted_price\nSYN-0000007,500000\nSYN-0000008,600000\n",
    "empty price": GOOD.replace("700000", ""),
    "price is text": GOOD.replace("700000", "seven hundred"),
    "negative price": GOOD.replace("700000", "-700000"),
    "zero price": GOOD.replace("700000", "0"),
    "infinite price": GOOD.replace("700000", "inf"),
    "empty file": "",
}


def test_check_accepts_a_good_file(tmp_path):
    path = tmp_path / "submission.csv"
    path.write_text(GOOD)
    assert check_submission.check_file(path, IDS) == []


@pytest.mark.parametrize("problem", BAD_FILES)
def test_check_rejects_a_bad_file(tmp_path, problem):
    path = tmp_path / "submission.csv"
    path.write_text(BAD_FILES[problem])
    assert check_submission.check_file(path, IDS) != []


# ---- the report computes the metrics correctly ----

def test_metrics_on_known_errors():
    # Errors of -10%, +10%, +20% and 0%.
    predictions = pd.DataFrame({"sold_price": [100.0] * 4, "predicted_price": [90.0, 110.0, 120.0, 100.0]})
    scores = run.metrics(predictions)
    assert scores["n"] == 4
    assert scores["mape"] == 10.0                  # mean of 10, 10, 20, 0
    assert scores["mdape"] == 10.0                 # median of 0, 10, 10, 20
    assert scores["median_signed_error"] == 5.0    # median of -10, 0, 10, 20
    low, high = scores["mape_95ci"]
    assert low <= 10.0 <= high


# ---- every command runs end to end on tiny files ----

def test_whole_pipeline_on_tiny_data(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    listings = make_listings(1500)
    listings["remarks"] = [
        f"home with feature{i % 12} near green space"
        for i in range(len(listings))
    ]
    listings.to_parquet(folder / "listings.parquet")
    later = make_listings(40, seed=2, first_id=5000)
    data.hide_answers(later.iloc[:20], "off_market").to_parquet(folder / "test_off_market.parquet")
    data.hide_answers(later.iloc[20:], "list_aware").to_parquet(folder / "test_list_aware.parquet")
    monkeypatch.setenv("DATA_DIR", str(folder))
    monkeypatch.setattr(run, "ROOT", tmp_path)

    run.validate()
    run.report()
    run.predict()
    assert run.check()
    assert (tmp_path / "figures" / "error_by_price_range.png").exists()
