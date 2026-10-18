import numpy as np
import pandas as pd


class DummyModel:
    def fit(self, *features):
        return self

    def predict(self, *features):
        if not features:
            return np.zeros(0, dtype=float)

        length = len(features[0])
        if any(len(feature) != length for feature in features[1:]):
            raise ValueError("All feature arguments must have the same length")

        return np.zeros(length, dtype=float)

    def selected_interactions(self):
        return pd.Series(dtype=float)