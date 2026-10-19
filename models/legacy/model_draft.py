import numpy as np
import pandas as pd

from models.gb_models import CatBoostRegressor, ArimaResidualModel

DEFAULT_SPECS = [
    (
        CatBoostRegressor,
        "Xp",
        {
            "iterations": 1200,
            "depth": 6,
            "learning_rate": 0.04,
            "verbose": False,
            "cat_features": CATEGORICAL_COLUMNS,
        }
    ),
    (
        ArimaResidualModel,
        "date_listed",
        {"order": (1, 1, 0)}
    ),
]



class OffMarketModel:
    """Sequential additive model for log sale prices."""

    def __init__(self, modelspecs):
        self.components = []

        for model_class, feature_name, params in modelspecs:
            model = model_class(**(params or {}))
            features = make_features(feature_name)

            self.components.append((model, features))

    def fit(self, train):
        y = np.log(train["sold_price"].to_numpy(dtype=float))
        residual = y.copy()

        for model, features in self.components:
            X = features.fit_transform(train)

            model.fit(X, residual)
            residual -= np.asarray(model.predict(X), dtype=float)

        return self

    def predict_log(self, homes):
        prediction = np.zeros(len(homes), dtype=float)

        for model, features in self.components:
            X = features.transform(homes)
            prediction += np.asarray(model.predict(X), dtype=float)

        return prediction

    def predict(self, homes):
        return np.exp(self.predict_log(homes))


class ListAwareModel:
    """Listing-price adjustment on top of any OffMarketModel specification."""

    def __init__(self, modelspecs, correction_alpha=50.0, n_folds=3):
        self.modelspecs = modelspecs
        self.base_model = OffMarketModel(modelspecs)

        self.correction_model = Ridge(alpha=correction_alpha)
        self.scaler = StandardScaler()
        self.n_folds = n_folds

        self.columns_ = None
        self.medians_ = None

    def _features(self, homes, base_log):
        """Construct listing-price correction features."""

        base_log = np.asarray(base_log, dtype=float)
        list_log = np.log(
            homes["list_price"].to_numpy(dtype=float)
        )

        dates = pd.to_datetime(homes["date_listed"])

        X = pd.DataFrame(index=homes.index)

        # Listing price discrepancy
        X["list_discrepancy"] = list_log - base_log
        X["base_log_price"] = base_log

        # Property characteristics
        for col in ["bedrooms", "washrooms", "parking_total"]:
            X[col] = pd.to_numeric(
                homes[col], errors="coerce"
            )

        # Geography and time
        X["longitude"] = homes["longitude"]
        X["latitude"] = homes["latitude"]
        X["months_since_2021"] = (
            (dates.dt.year - 2021) * 12 + dates.dt.month - 1
        )

        # House type
        types = pd.get_dummies(
            homes["house_type_name"].fillna("Unknown"),
            prefix="type",
            dtype=float
        )

        return pd.concat([X, types], axis=1)

    def _prepare(self, homes, base_log, fit=False):
        X = self._features(homes, base_log)

        if fit:
            self.columns_ = X.columns.tolist()
            self.medians_ = X.median().fillna(0)

        X = X.reindex(columns=self.columns_, fill_value=0)
        X = X.fillna(self.medians_).fillna(0)

        if fit:
            return self.scaler.fit_transform(X)

        return self.scaler.transform(X)

    def fit(self, train):
        train = train.sort_values("date_sold").reset_index(drop=True)
        sale_months = train["date_sold"].dt.to_period("M")
        unique_months = sale_months.unique()

        X_oof = []
        y_oof = []

        # Generate out-of-time predictions for the correction model
        for _, val_idx in TimeSeriesSplit(
            n_splits=self.n_folds
        ).split(unique_months):

            val_months = unique_months[val_idx]
            cutoff = val_months[0].start_time

            past = train[train["date_sold"] < cutoff]
            held_out = train[sale_months.isin(val_months)]

            # Fresh off-market model with the SAME specifications
            model = OffMarketModel(self.modelspecs)
            model.fit(past)

            base_log = model.predict_log(held_out)

            X_oof.append(
                self._features(held_out, base_log)
            )
            y_oof.append(
                np.log(held_out["sold_price"].to_numpy())
                - base_log
            )

        # Fit Ridge on historical out-of-time errors
        X_oof = pd.concat(X_oof, ignore_index=True)
        y_oof = np.concatenate(y_oof)

        # _prepare constructs the same columns used for OOF
        self.columns_ = X_oof.columns.tolist()
        self.medians_ = X_oof.median().fillna(0)

        X = self.scaler.fit_transform(
            X_oof.fillna(self.medians_).fillna(0)
        )

        self.correction_model.fit(X, y_oof)

        # Final off-market model uses all available training data
        self.base_model.fit(train)

        return self

    def predict_log(self, homes):
        if homes.empty:
            return np.array([], dtype=float)

        base_log = self.base_model.predict_log(homes)

        X = self._prepare(homes, base_log)
        correction = self.correction_model.predict(X)

        return base_log + correction

    def predict(self, homes):
        return np.exp(self.predict_log(homes))

def make_model(track, modelspecs=None):
    if modelspecs is None:
        modelspecs = DEFAULT_SPECS

    if track == "off_market":
        return OffMarketModel(modelspecs)

    if track == "list_aware":
        return ListAwareModel(modelspecs)

    raise ValueError(f"Unknown competition track: {track}")
