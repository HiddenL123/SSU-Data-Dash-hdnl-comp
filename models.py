"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

import features


def make_model(track):
    """Return a fresh model for one of the competition tracks."""

    if track == "off_market":
        print("using actual model")
        return OffMarketModel()

    if track == "list_aware":
        return ListingAdjustedModel()

    raise ValueError(f"Unknown track: {track}")

class PropertyModel:
    """Property-price regression component f(X)."""

    def __init__(self, alpha=0.01, l1_ratio=0.8):
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=20000,
            random_state=0
        )

        self.feature_names = None

    def fit(self, X, y):
        """X: processed features, y: actual sold prices."""

        self.feature_names = X.columns.copy()

        self.model.fit(X, np.log(y))

        return self

    def predict_log(self, X):
        """Return predicted log sold prices."""
        return self.model.predict(X)

    def predict(self, X):
        """Return predicted sold prices."""
        return np.exp(self.predict_log(X))

    def selected_features(self):
        """Return nonzero coefficients sorted by magnitude."""

        coefs = pd.Series(
            self.model.coef_,
            index=self.feature_names
        )

        selected = coefs[coefs.abs() > 1e-8]

        return selected.reindex(
            selected.abs().sort_values(
                ascending=False
            ).index
        )

class _FeatureModel:
    """Fit a regularized log-price model using the shared feature pipeline."""

    def __init__(self, include_list_price=False, alpha=0.01, l1_ratio=0.8):
        self.include_list_price = include_list_price
        self.model = PropertyModel(alpha=alpha, l1_ratio=l1_ratio)
        self.feature_state = None
        self.input_scaler = StandardScaler()
        self.input_columns = None
        self.input_medians = None

    def _features(self, homes, state=None):
        X, coordinates, months, state = features.sci_kit_learn_df(
            homes,
            state=state,
        )
        X = X.copy()
        X["longitude"] = coordinates["longitude"]
        X["latitude"] = coordinates["latitude"]
        X["months_since_2021"] = np.asarray(months)

        if self.include_list_price:
            list_prices = homes["list_price"].to_numpy(dtype=float)
            if np.any(list_prices <= 0):
                raise ValueError("List prices must be positive")
            X["log_list_price"] = np.log(list_prices)

        return X, state

    def fit(self, train):
        """Fit on sales, including their sold_price target."""
        if train.empty:
            raise ValueError("Cannot fit a model with no training rows")
        if (train["sold_price"] <= 0).any():
            raise ValueError("Sold prices must be positive")

        X, self.feature_state = self._features(train)
        self.input_columns = X.columns.tolist()
        self.input_medians = X.median().fillna(0)
        X = X.reindex(columns=self.input_columns).fillna(self.input_medians)
        values = self.input_scaler.fit_transform(X)
        X = pd.DataFrame(values, columns=self.input_columns, index=X.index)
        self.model.fit(X, train["sold_price"])
        return self

    def predict(self, homes):
        """Predict one positive sold price for each input home."""
        if self.feature_state is None:
            raise RuntimeError("Call fit(train) before predict(homes)")
        if homes.empty:
            return np.array([], dtype=float)

        X, _ = self._features(homes, state=self.feature_state)
        X = X.reindex(columns=self.input_columns, fill_value=0)
        X = X.fillna(self.input_medians)
        values = self.input_scaler.transform(X)
        X = pd.DataFrame(values, columns=self.input_columns, index=X.index)
        return self.model.predict(X)

    def selected_features(self):
        return self.model.selected_features()


class OffMarketModel(_FeatureModel):
    """Off-market model using property, location, and time features."""

    def __init__(self, alpha=0.01, l1_ratio=0.8):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio)


class ListingAdjustedModel(_FeatureModel):
    """List-aware model that additionally uses the home's list price."""

    def __init__(self, alpha=0.01, l1_ratio=0.8):
        super().__init__(
            include_list_price=True,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )

if __name__ == "__main__":
    import features
    import data
    homes = data.load("listings")
    homes = homes.sort_values("date_sold")

    # 80/20 split
    split = int(len(homes) * 0.8)

    train = homes.iloc[:split].copy()
    valid = homes.iloc[split:].copy()


    X_train, p_train, t_train, feature_state = features.sci_kit_learn_df(train)
    X_valid, p_valid, t_valid, _ = features.sci_kit_learn_df(
        valid,
        state=feature_state
    )
    model = PropertyModel()
    model.fit(X_train, train["sold_price"])

    #print(model.selected_features())
    #print("Selected:", len(model.selected_features()))
    row_number = 0
    selected = model.selected_features()
    row = X_valid.iloc[row_number]

    print(f"Row {row_number} (ml_num={valid.iloc[row_number]['ml_num']}):")
    print(pd.DataFrame({
        "value": row[selected.index],
        "coefficient": selected,
        "contribution": row[selected.index] * selected
    }).to_string())
