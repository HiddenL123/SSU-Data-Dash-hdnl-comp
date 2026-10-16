"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
from catboost import train
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet



def make_model(track):
    """Return a fresh, untrained model."""

    if track == "off_market":
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

class OffMarketModel:

    def __init__(self):
        self.property_model = PropertyModel()
        self.spatial_model = SpatialModel()
        self.temporal_model = TemporalModel()
        self.interaction_model = InteractionModel()

    def fit(self, X, p, t, y):
        # Joint or carefully staged training
        ...
        return self

    def predict_log(self, X, p, t):
        return (
            self.property_model.predict_log(X)
            + self.spatial_model.predict(p)
            + self.temporal_model.predict(t)
            + self.interaction_model.predict(X, p, t)
        )

    def predict(self, X, p, t):
        return np.exp(self.predict_log(X, p, t))


class OffMarketModel:
    """Stage 1: predict log sold price from property features."""

    def __init__(self, alpha=0.01, l1_ratio=0.8):
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=20000,
            random_state=0
        )
        self.feature_state = None
        self.feature_names = None

    def fit(self, train):
        X, self.feature_state = features.sci_kit_learn_df(train)

        y = np.log(train["sold_price"].to_numpy())

        self.model.fit(X, y)
        self.feature_names = X.columns

        return self

    def predict(self, homes):
        X, _ = features.sci_kit_learn_df(
            homes,
            state=self.feature_state
        )

        log_predictions = self.model.predict(X)

        return np.exp(log_predictions)

    def selected_features(self):
        """Return nonzero coefficients, sorted by magnitude."""
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


class ListingAdjustedModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))

if __name__ == "__main__":
    import features
    import data
    homes = data.load("listings")
    homes = homes.sort_values("date_sold")

    # 80/20 split
    split = int(len(homes) * 0.8)

    train = homes.iloc[:split].copy()
    valid = homes.iloc[split:].copy()


    X_train, feature_state = features.sci_kit_learn_df(train)
    X_valid, _ = features.sci_kit_learn_df(
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
