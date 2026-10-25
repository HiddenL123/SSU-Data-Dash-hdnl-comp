"""Feature extraction for CatBoost, LightGBM, and XGBoost.

Use one fresh BoostingFeatures instance per chronological training fold.
Fit only on the training rows, then transform future rows with the same state.
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.feature_extraction.text import TfidfVectorizer


CATEGORICAL_COLUMNS = (
    "house_type_name",
    "house_style",
    "municipality",
    "community",
    "garage",
)

tf_idf_features = ['tfidf_as is', 'tfidf_fully', 'tfidf_first', 'tfidf_downsizers', 'tfidf_young professionals', 'tfidf_investors', 'tfidf_starting', 'tfidf_discerning', 'tfidf_special', 'tfidf_family', 'tfidf_growing', 'tfidf_offered', 'tfidf_renovated', 'tfidf_move up', 'tfidf_potential', 'tfidf_first time', 'tfidf_vision', 'tfidf_entertaining', 'tfidf_redone', 'tfidf_entertain', 'tfidf_move', 'tfidf_time', 'tfidf_new', 'tfidf_original', 'tfidf_older', 'tfidf_custom', 'tfidf_growing family', 'tfidf_couple', 'tfidf_opportunity', 'tfidf_central air', 'tfidf_grow', 'tfidf_front', 'tfidf_discerning buyer', 'tfidf_turnkey', 'tfidf_investor', 'tfidf_project', 'tfidf_starting out', 'tfidf_unit', 'tfidf_cozy', 'tfidf_spa like', 'tfidf_wide plank', 'tfidf_top', 'tfidf_room to', 'tfidf_beautifully', 'tfidf_needs', 'tfidf_bones', 'tfidf_overhaul']


def square_feet(house_area):
    """Convert '622' or '900-1099' to approximate square footage."""
    values = house_area.astype("string").str.replace(",", "", regex=False)
    bounds = values.str.extract(
        r"^\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?"
    )
    low = pd.to_numeric(bounds[0], errors="coerce")
    high = pd.to_numeric(bounds[1], errors="coerce").fillna(low)
    return ((low + high) / 2).astype("float32")


class BoostingFeatures:
    """Shared training-fitted features with library-specific categorical dtypes.

    - model='catboost': string categoricals; pass cat_features to CatBoost.
    - model='lightgbm': pandas categorical dtypes (auto-detected).
    - model='xgboost': pandas categorical dtypes; use enable_categorical=True.

    `include_time=False` initially keeps date effects in the separate ARIMA model.
    Set it to True for a direct time-feature experiment.
    """

    def __init__(self, use_remarks=True, max_text_features=1000, include_time=False):
        self.use_remarks = use_remarks
        self.max_text_features = max_text_features
        self.include_time = include_time
        self.vectorizer_ = None
        self.category_levels_ = None
        self.columns_ = None
        self.track_ = None

    @staticmethod
    def _column(homes, name):
        if name not in homes:
            return pd.Series(np.nan, index=homes.index)
        return homes[name]

    def _numeric(self, homes):
        def number(name):
            return pd.to_numeric(
                self._column(homes, name), errors="coerce"
            ).astype("float32")

        X = pd.DataFrame(index=homes.index)
        X["square_feet"] = square_feet(self._column(homes, "house_area"))
        X["bedrooms"] = number("bedrooms")
        X["washrooms"] = number("washrooms")
        X["parking"] = number("parking_total")
        X["garage_count"] = number("garage")
        X["lot_front"] = number("lot_front")
        X["lot_depth"] = number("lot_depth")
        X["lot_area"] = number("lot_area").fillna(
            X["lot_front"] * X["lot_depth"]
        )
        X["longitude"] = number("longitude")
        X["latitude"] = number("latitude")

        if self.include_time:
            dates = pd.to_datetime(self._column(homes, "date_listed"), errors="coerce")
            month = dates.dt.month
            X["months_since_2021"] = (
                (dates.dt.year - 2021) * 12 + month - 1
            ).astype("float32")
            X["season_sin"] = np.sin(2 * np.pi * month / 12).astype("float32")
            X["season_cos"] = np.cos(2 * np.pi * month / 12).astype("float32")

        if self.track_ == "list_aware":
            prices = number("list_price")
            X["log_list_price"] = np.log(prices.where(prices > 0))

        return X.replace([np.inf, -np.inf], np.nan)

    def _categories(self, homes):
        X = pd.DataFrame(index=homes.index)
        for column in CATEGORICAL_COLUMNS:
            value = (
                self._column(homes, column)
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )
            X[column] = value.mask(value.eq(""), "Unknown")
        return X

    def fit(self, homes, track="off_market"):
        if track not in ("off_market", "list_aware"):
            raise ValueError("track must be 'off_market' or 'list_aware'")
        self.track_ = track
        categories = self._categories(homes)
        self.category_levels_ = {
            col: sorted(set(categories[col]).union({"Unknown"}))
            for col in CATEGORICAL_COLUMNS
        }

        if self.use_remarks:
            self.vectorizer_ = TfidfVectorizer(
                vocabulary=[
                    word.removeprefix("tfidf_")
                    for word in tf_idf_features
                ],
                ngram_range=(1, 2),
                dtype=np.float32,
            )

            self.vectorizer_.fit(
                self._column(homes, "remarks").fillna("").astype(str)
            )

        self.columns_ = list(self._numeric(homes).columns)
        self.columns_ += list(CATEGORICAL_COLUMNS)
        if self.use_remarks:
            self.columns_ += [
                f"tfidf_{i}" for i in range(len(self.vectorizer_.get_feature_names_out()))
            ]
        return self

    def transform(self, homes, model="catboost"):

        X = self._numeric(homes)
        categories = self._categories(homes)
        for col in CATEGORICAL_COLUMNS:
            known = self.category_levels_[col]
            values = categories[col].where(categories[col].isin(known), "Unknown")
            if model == "catboost":
                X[col] = values.astype(str)
            else:
                X[col] = values.astype(CategoricalDtype(categories=known))

        if self.use_remarks:
            matrix = self.vectorizer_.transform(
                self._column(homes, "remarks").fillna("").astype(str)
            )
            text = pd.DataFrame(
                matrix.toarray(),
                index=homes.index,
                columns=[f"tfidf_{i}" for i in range(matrix.shape[1])],
            )
            X = pd.concat([X, text], axis=1)

        return X[self.columns_]

    def fit_transform(self, homes, model="catboost", track="off_market"):
        return self.fit(homes, track=track).transform(homes, model=model)

    @property
    def categorical_features(self):
        return list(CATEGORICAL_COLUMNS)

    @property
    def text_feature_names(self):
        """Map numbered TF-IDF columns back to their tokens."""
        if self.vectorizer_ is None:
            return {}
        return {
            f"tfidf_{i}": word
            for i, word in enumerate(self.vectorizer_.get_feature_names_out())
        }
