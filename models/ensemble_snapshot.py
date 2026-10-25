"""Re-loadable feature-bagged XGBoost models sharing one fitted feature base.

The fitted provider includes historical comparable-sale context, vocabulary,
and space-time feature transforms. Saving *only* the estimator or its heldout
predictions cannot reproduce inference. The bundle stores the provider once
and independent tree models/column subsets beneath it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BaggedMember:
    """One independently fitted estimator plus its text/row-bagging manifest."""

    name: str
    estimator: Any
    selected_columns: tuple[str, ...]
    target_offset: float
    terms: tuple[str, ...]
    settings: dict

    def predict_from_features(self, features: pd.DataFrame) -> np.ndarray:
        missing = set(self.selected_columns).difference(features.columns)
        if missing:
            raise ValueError(f"{self.name}: missing fitted feature columns {sorted(missing)}")
        log_prices = np.asarray(
            self.estimator.predict(features[list(self.selected_columns)]),
            dtype=float,
        ).reshape(-1)
        if len(log_prices) != len(features) or not np.isfinite(log_prices).all():
            raise ValueError(f"{self.name}: invalid predictions")
        prices = np.exp(float(self.target_offset) + log_prices)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f"{self.name}: invalid predicted sale prices")
        return prices


class FeatureBaggingEnsemble:
    """Reusable off-market model bundle with shared fitted context features."""

    def __init__(self, feature_provider, members, *, metadata=None):
        self.feature_provider = feature_provider
        self.members = dict(members)
        if not self.members:
            raise ValueError("An ensemble needs at least one fitted member")
        if any(name != member.name for name, member in self.members.items()):
            raise ValueError("Member manifest keys and names differ")
        self.metadata = dict(metadata or {})

    def predict_members(self, homes) -> pd.DataFrame:
        """One price prediction per member, in the original input row order."""
        features = self.feature_provider.transform(homes)
        if len(features) != len(homes):
            raise ValueError("Feature provider changed the number of homes")
        return pd.DataFrame(
            {
                name: member.predict_from_features(features)
                for name, member in self.members.items()
            },
            index=homes.index,
        )

    def predict(self, homes, *, weights=None) -> np.ndarray:
        """Default equal-weight geometric blend, or explicit positive weights.

        This default is *not* a tuned production ensemble. Train blend weights
        using earlier out-of-time folds before treating an ensemble as final.
        """
        pred = self.predict_members(homes)
        if weights is None:
            weights = {name: 1.0 for name in pred.columns}
        if not weights or any(
            name not in pred.columns or not np.isfinite(value) or value < 0
            for name, value in weights.items()
        ):
            raise ValueError("Blend weights must reference fitted members and be nonnegative")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("At least one member must have positive weight")
        values = np.log(pred[list(weights)].to_numpy(dtype=float))
        w = np.array(list(weights.values()), dtype=float) / total
        return np.exp(values @ w)

    def predict_member(self, homes, name: str) -> np.ndarray:
        if name not in self.members:
            raise KeyError(f"Unknown saved ensemble member: {name}")
        return self.members[name].predict_from_features(
            self.feature_provider.transform(homes)
        )
