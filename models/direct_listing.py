
import numpy as np
from catboost import CatBoostRegressor
from models.features import BoostingFeatures


class DirectListingModel:
    """
    List-aware CatBoost model.

    target="direct":
        Predict log(sold_price).

    target="ratio":
        Predict log(sold_price / list_price).

    No historical off-market fits or Ridge correction required.
    """

    def __init__(
        self,
        target="ratio",
        model_params=None,
        include_time=True,
    ):
        if target not in ("direct", "ratio"):
            raise ValueError("target must be 'direct' or 'ratio'")

        self.target = target

        self.features = BoostingFeatures(
            use_remarks=True,
            include_time=include_time,
        )

        params = {
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
        }
        params.update(model_params or {})

        self.model = CatBoostRegressor(**params)
        self._fitted = False

    @staticmethod
    def _log_price(homes, column):
        values = homes[column].to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column} must be positive and finite")

        return np.log(values)

    def fit(self, train):
        log_sold = self._log_price(train, "sold_price")
        log_list = self._log_price(train, "list_price")

        X = self.features.fit_transform(
            train,
            track="list_aware",
            model="catboost",
        )

        if self.target == "direct":
            y = log_sold
        else:
            y = log_sold - log_list

        self.model.fit(
            X,
            y,
            cat_features=self.features.categorical_features,
        )

        self._fitted = True
        return self

    def predict_log(self, homes):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        if homes.empty:
            return np.array([], dtype=float)

        X = self.features.transform(
            homes,
            model="catboost",
        )

        prediction = np.asarray(
            self.model.predict(X),
            dtype=float,
        ).reshape(-1)

        if self.target == "ratio":
            prediction += self._log_price(homes, "list_price")

        return prediction

    def predict(self, homes):
        return np.exp(self.predict_log(homes))
