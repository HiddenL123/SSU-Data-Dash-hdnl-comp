"""Interpretable housing-price models compatible with run.py.

Required interface:
    make_model(track).fit(training_listings).predict(unseen_homes)

The local features.sci_kit_learn_df() should return either
    (X, coordinates, months, fitted_state)
or
    (X, fitted_state).
The 2-return form derives coordinates/months from the original homes.
Only fitting the feature state on training rows prevents validation leakage.
"""

import numpy as np
import pandas as pd
from pygam import LinearGAM, te
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

import features


def make_model(track):
    """Return a fresh model for the competition track."""
    if track == "off_market":
        return OffMarketModel()
    if track == "list_aware":
        return ListingAdjustedModel()
    raise ValueError(f"Unknown track: {track}")


class PropertyModel:
    """HGB property model with Elastic Net TF-IDF selection."""

    def __init__(self, alpha=0.01, l1_ratio=0.8,
                 max_text_features=100):

        self.selector = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=20000,
            random_state=0
        )

        self.model = HistGradientBoostingRegressor(
            random_state=0
        )

        self.max_text_features = max_text_features

    def fit(self, X, y):
        log_y = np.log(np.asarray(y, dtype=float))

        # Fit Elastic Net for feature selection
        self.selector.fit(X, log_y)

        coefficients = pd.Series(
            self.selector.coef_,
            index=X.columns
        )

        # Keep ALL non-text features
        structured = [
            col for col in X.columns
            if not col.startswith("tfidf_")
        ]

        # Select text using nonzero Elastic Net coefficients
        text_importance = coefficients[
            coefficients.index.str.startswith("tfidf_")
        ].abs()

        selected_text = (
            text_importance[text_importance > 1e-8]
            .nlargest(self.max_text_features)
        )

        self.text_features_ = selected_text.index.tolist()

        # Final feature set
        self.feature_names = structured + self.text_features_

        # Fit HGB on reduced feature matrix
        self.model.fit(X[self.feature_names], log_y)

        return self

    def predict_log(self, X):
        return self.model.predict(X[self.feature_names])

    def predict(self, X):
        return np.exp(self.predict_log(X))

    def selected_text_features(self):
        return self.text_features_



class SpatialModel:
    """Two-dimensional tensor-product GAM for spatial log-price residuals."""

    def __init__(self, n_splines=10, lam=10):
        self.model = LinearGAM(
            te(0, 1, n_splines=[n_splines, n_splines], lam=lam),
            fit_intercept=False,
        )

    def fit(self, coordinates, log_residuals):
        self.model.fit(np.asarray(coordinates, dtype=float), log_residuals)
        return self

    def predict(self, coordinates):
        return np.asarray(
            self.model.predict(np.asarray(coordinates, dtype=float)),
            dtype=float,
        ).reshape(-1)


class TemporalModel:
    """Penalized spline trend plus annual seasonality, on log-price residuals."""

    def __init__(self, alpha=100.0, n_knots=10):
        self.spline = SplineTransformer(
            n_knots=n_knots,
            degree=3,
            include_bias=False,
            extrapolation="linear",
        )
        self.model = Ridge(alpha=alpha)

    @staticmethod
    def _months(t):
        values = np.asarray(t)
        if np.issubdtype(values.dtype, np.number):
            return values.astype(float).reshape(-1)

        dates = pd.to_datetime(pd.Series(t))
        if dates.isna().any():
            raise ValueError("Temporal model requires valid dates")
        return (
            (dates - pd.Timestamp("2021-01-01")).dt.total_seconds()
            / (86400 * 365.25 / 12)
        ).to_numpy(dtype=float)

    def _features(self, t, fit=False):
        months = self._months(t)
        time = months.reshape(-1, 1)
        smooth = (
            self.spline.fit_transform(time)
            if fit else self.spline.transform(time)
        )
        seasonal = np.column_stack([
            np.sin(2 * np.pi * months / 12),
            np.cos(2 * np.pi * months / 12),
        ])
        return np.column_stack([smooth, seasonal])

    def fit(self, months, log_residuals):
        self.model.fit(self._features(months, fit=True), log_residuals)
        return self

    def predict(self, months):
        return self.model.predict(self._features(months))


class InteractionModel:
    """Placeholder; interactions will be implemented separately later."""

    def fit(self, X, coordinates, months, log_residuals):
        return self

    def predict(self, X, coordinates, months):
        return np.zeros(len(X), dtype=float)


class _FeatureModel:
    """Shared preprocessing for both competition tracks.

    This calls features.sci_kit_learn_df just once per fit/predict batch.
    The property features X stay separate from spatial coordinates and time.
    """

    def __init__(self, *, include_list_price=False, alpha=0.01, l1_ratio=0.8):
        self.include_list_price = include_list_price
        self.model = PropertyModel(alpha=alpha, l1_ratio=l1_ratio)
        self.feature_state = None
        self.input_scaler = StandardScaler()
        self.input_columns = None
        self.input_medians = None
        self._fitted = False

    def _prepare(self, homes, *, fit):
        if fit:
            output = features.sci_kit_learn_df(homes)
        else:
            output = features.sci_kit_learn_df(homes, state=self.feature_state)

        if len(output) == 4:
            X, coordinates, months, state = output
        elif len(output) == 2:
            X, state = output
            coordinates = homes[["longitude", "latitude"]]
            months = (
                (pd.to_datetime(homes["date_listed"]) - pd.Timestamp("2021-01-01"))
                .dt.total_seconds() / (86400 * 365.25 / 12)
            )
        else:
            raise ValueError(
                "features.sci_kit_learn_df must return (X, state) "
                "or (X, coordinates, months, state)"
            )

        if fit:
            self.feature_state = state

        # Preserve the predictors in the recovered model: f(X) contains linear
        # location/time effects, while the GAM and spline correct residuals.
        # We can ablate these overlapping inputs later, but do not silently
        # change the original model while repairing it.
        coordinates = np.asarray(coordinates, dtype=float)
        months = TemporalModel._months(months)
        X = X.copy()
        X["longitude"] = homes["longitude"].to_numpy(dtype=float)
        X["latitude"] = homes["latitude"].to_numpy(dtype=float)
        X["months_since_2021"] = months

        if self.include_list_price:
            list_price = homes["list_price"].to_numpy(dtype=float)
            if not np.isfinite(list_price).all() or (list_price <= 0).any():
                raise ValueError("All list prices must be positive and finite")
            X["log_list_price"] = np.log(list_price)

        X = X.replace([np.inf, -np.inf], np.nan)
        if fit:
            self.input_columns = X.columns.tolist()
            self.input_medians = X.median().reindex(self.input_columns).fillna(0)
        else:
            X = X.reindex(columns=self.input_columns, fill_value=0)

        X = X.fillna(self.input_medians)
        scaled = (
            self.input_scaler.fit_transform(X)
            if fit else self.input_scaler.transform(X)
        )
        matrix = pd.DataFrame(scaled, index=X.index, columns=X.columns)

        if coordinates.shape != (len(matrix), 2):
            raise ValueError("Spatial coordinates must have shape (n_homes, 2)")
        if len(months) != len(matrix):
            raise ValueError("Temporal input must contain one value per home")
        return matrix, coordinates, months

    @staticmethod
    def _check_targets(train):
        if train.empty:
            raise ValueError("Cannot fit on an empty training set")
        y = train["sold_price"].to_numpy(dtype=float)
        if not np.isfinite(y).all() or (y <= 0).any():
            raise ValueError("Sold prices must be positive and finite")
        return y

    def fit(self, train):
        y = self._check_targets(train)
        X, _, _ = self._prepare(train, fit=True)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_log(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit(train) before predict(homes)")
        if homes.empty:
            return np.array([], dtype=float)
        X, _, _ = self._prepare(homes, fit=False)
        return self.model.predict_log(X)

    def predict(self, homes):
        return np.exp(self.predict_log(homes))

    def selected_features(self):
        return self.model.selected_features()


class OffMarketModel(_FeatureModel):
    """Full off-market model: Elastic Net + spatial GAM + temporal spline."""

    def __init__(self, alpha=0.01, l1_ratio=0.8):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio)
        self.spatial_model = SpatialModel()
        self.temporal_model = TemporalModel()
        self.interaction_model = InteractionModel()

    def fit(self, train):
        y = self._check_targets(train)
        X, coordinates, months = self._prepare(train, fit=True)

        # Sequential residual fitting on log-price; each component corrects
        # errors remaining after the earlier components.
        self.model.fit(X, y)
        residual = np.log(y) - self.model.predict_log(X)

        self.spatial_model.fit(coordinates, residual)
        residual -= self.spatial_model.predict(coordinates)

        self.temporal_model.fit(months, residual)
        residual -= self.temporal_model.predict(months)

        self.interaction_model.fit(X, coordinates, months, residual)
        self._fitted = True
        return self

    def predict_log(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit(train) before predict(homes)")
        if homes.empty:
            return np.array([], dtype=float)

        X, coordinates, months = self._prepare(homes, fit=False)
        return (
            self.model.predict_log(X)
            + self.spatial_model.predict(coordinates)
            + self.temporal_model.predict(months)
            + self.interaction_model.predict(X, coordinates, months)
        )


class ListingAdjustedModel:
    """Two-stage listing-aware model.

    1. Fit the *full* off-market model: log(base_price) = f(X) + s(v) + m(t).
    2. Train a small Ridge correction on genuinely out-of-time base predictions:

           d = log(list_price) - log(base_price)
           h = log(sold_price) - log(base_price)

       The correction uses d, property type, house attributes, location and time.
    3. Refit the off-market model on all available sales for final predictions.

    API matches run.py: make_model('list_aware').fit(train).predict(homes).
    """

    def __init__(self, alpha=0.01, l1_ratio=0.8,
                 correction_alpha=50.0, n_folds=3):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_folds = n_folds
        self.base_model = OffMarketModel(alpha=alpha, l1_ratio=l1_ratio)
        self.correction_model = Ridge(alpha=correction_alpha)
        self.correction_scaler = StandardScaler()
        self.correction_columns = None
        self.correction_medians = None
        self.house_types = None
        self.oof_count_ = 0
        self._fitted = False

    @staticmethod
    def _list_log(homes):
        prices = homes["list_price"].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or np.any(prices <= 0):
            raise ValueError("List-aware predictions require positive finite list prices")
        return np.log(prices)

    def _correction_features(self, homes, base_log):
        """A small interpretable h(d, X, v, t), not the large TF-IDF matrix."""
        base_log = np.asarray(base_log, dtype=float).reshape(-1)
        if len(base_log) != len(homes):
            raise ValueError("Expected one off-market log prediction per home")

        d = self._list_log(homes) - base_log
        area = features.square_feet(homes["house_area"])
        area = pd.to_numeric(area, errors="coerce").clip(lower=0)
        months = (
            (pd.to_datetime(homes["date_listed"]) - pd.Timestamp("2021-01-01"))
            .dt.total_seconds() / (86400 * 365.25 / 12)
        )

        Z = pd.DataFrame(index=homes.index)
        Z["list_discrepancy"] = d
        Z["off_market_log_price"] = base_log
        Z["log_area"] = np.log1p(area.to_numpy(dtype=float))
        Z["bedrooms"] = pd.to_numeric(homes["bedrooms"], errors="coerce")
        Z["washrooms"] = pd.to_numeric(homes["washrooms"], errors="coerce")
        Z["parking"] = pd.to_numeric(homes["parking_total"], errors="coerce")
        Z["longitude"] = pd.to_numeric(homes["longitude"], errors="coerce")
        Z["latitude"] = pd.to_numeric(homes["latitude"], errors="coerce")
        Z["months_since_2021"] = np.asarray(months, dtype=float)

        # Vary the degree of trust in list price across property types.
        for property_type in self.house_types:
            indicator = (homes["house_type_name"] == property_type).astype(float)
            key = f"type_{property_type}"
            Z[key] = indicator.to_numpy()
            Z[f"discrepancy_x_{key}"] = d * indicator.to_numpy()

        return Z.replace([np.inf, -np.inf], np.nan)

    def fit(self, train):
        """Build forward-in-time Stage-1 predictions, then fit Stage 2."""
        y = _FeatureModel._check_targets(train)
        self._list_log(train)
        if self.n_folds < 1:
            raise ValueError("n_folds must be at least 1")

        listed = pd.to_datetime(train["date_listed"])
        sold = pd.to_datetime(train["date_sold"])
        if listed.isna().any() or sold.isna().any():
            raise ValueError("Training requires valid date_listed and date_sold")

        self.house_types = sorted(train["house_type_name"].dropna().unique().tolist())

        # Recent chronological windows. The base model for each window is fitted
        # ONLY on properties whose sale was already completed before the window.
        sorted_listed = listed.sort_values().to_numpy()
        quantiles = np.linspace(0.55, 0.80, self.n_folds)
        cutoffs = sorted({
            pd.Timestamp(sorted_listed[min(int(q * len(train)), len(train) - 1)]).normalize()
            for q in quantiles
        })

        z_parts, r_parts = [], []
        for i, cutoff in enumerate(cutoffs):
            end = cutoffs[i + 1] if i + 1 < len(cutoffs) else None
            past = train.loc[sold < cutoff]
            mask = listed >= cutoff
            if end is not None:
                mask &= listed < end
            held_out = train.loc[mask]

            if len(past) < 100 or held_out.empty:
                continue

            earlier_model = OffMarketModel(alpha=self.alpha, l1_ratio=self.l1_ratio)
            earlier_model.fit(past)
            prior_log = earlier_model.predict_log(held_out)
            z_parts.append(self._correction_features(held_out, prior_log))
            r_parts.append(np.log(held_out["sold_price"].to_numpy(dtype=float)) - prior_log)

        if not z_parts:
            raise ValueError(
                "Not enough dated training history to build out-of-time Stage-1 "
                "predictions for the listing adjustment"
            )

        Z = pd.concat(z_parts, axis=0)
        residual = np.concatenate(r_parts)
        self.oof_count_ = len(residual)
        self.correction_columns = Z.columns.tolist()
        self.correction_medians = Z.median().fillna(0.0)
        Z = Z.fillna(self.correction_medians)
        Z_scaled = self.correction_scaler.fit_transform(Z)
        self.correction_model.fit(Z_scaled, residual)

        # Refit Stage 1 on ALL available earlier sales for final inference.
        self.base_model.fit(train)
        self._fitted = True
        return self

    def predict_log(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit(train) before predict(homes)")
        if homes.empty:
            return np.array([], dtype=float)

        base_log = self.base_model.predict_log(homes)
        Z = self._correction_features(homes, base_log)
        Z = Z.reindex(columns=self.correction_columns, fill_value=0)
        Z = Z.fillna(self.correction_medians)
        correction = self.correction_model.predict(self.correction_scaler.transform(Z))
        return base_log + correction

    def predict(self, homes):
        return np.exp(self.predict_log(homes))

    def correction_coefficients(self):
        """Ridge correction coefficients on their original, unscaled feature units."""
        if not self._fitted:
            raise RuntimeError("Call fit(train) before inspecting coefficients")
        weights = self.correction_model.coef_ / self.correction_scaler.scale_
        return pd.Series(weights, index=self.correction_columns).sort_values(
            key=lambda values: values.abs(), ascending=False
        )
