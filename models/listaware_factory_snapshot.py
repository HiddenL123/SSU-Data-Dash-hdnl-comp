"""Persistable list-aware feature-bagged members of the AutoML factory.

Like off-market's ensemble_snapshot, but reconstructs sale prices by adding
log(list_price) to the predicted log(sold/list) target. Each independently
trained member owns its *fitted* training-only feature provider.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ListAwareBaggedMember:
    name: str
    estimator: object
    feature_provider: object
    selected_columns: tuple[str, ...]
    target_offset: float
    terms: tuple[str, ...]
    settings: dict

    def predict(self, homes):
        if "sold_price" in homes.columns or "date_sold" in homes.columns:
            raise ValueError("Hide sold price and sale date from list-aware inference")
        listed = pd.to_numeric(homes["list_price"], errors="coerce").to_numpy(float)
        if not np.isfinite(listed).all() or (listed <= 0).any():
            raise ValueError("List-aware member requires positive list_price")
        X = self.feature_provider.transform(homes)
        if not X.index.equals(homes.index):
            raise ValueError("Prediction rows were reordered")
        missing = set(self.selected_columns) - set(X.columns)
        if missing:
            raise ValueError(f"{self.name}: missing features {sorted(missing)}")
        delta = np.asarray(
            self.estimator.predict(X[list(self.selected_columns)]), float
        ).reshape(-1)
        result = listed * np.exp(self.target_offset + delta)
        if result.shape != (len(homes),) or not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError(f"{self.name}: invalid price estimates")
        return result


class ListAwareFactoryEnsemble:
    def __init__(self, members, weights, *, metadata=None):
        self.members = dict(members)
        self.weights = dict(weights)
        self.metadata = dict(metadata or {})
        if not self.members or set(self.members) != set(self.weights):
            raise ValueError("Each fitted member needs one ensemble weight")
        w = np.asarray(list(self.weights.values()), float)
        if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("Weights must be nonnegative with positive sum")

    def predict_members(self, homes):
        return pd.DataFrame({
            name: member.predict(homes)
            for name, member in self.members.items()
        }, index=homes.index)

    def predict(self, homes):
        if homes.empty:
            return np.empty(0, dtype=float)
        members = self.predict_members(homes)
        w = np.asarray([self.weights[n] for n in members.columns], float)
        return np.exp(np.log(members.to_numpy(float)) @ (w/w.sum()))
