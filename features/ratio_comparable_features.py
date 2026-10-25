"""Comparable-sales features for the list-aware log sale/list ratio model.

Place at models/ratio_comparable_features.py.  Requires:
    models/spatiotemporal_features.py (with its stable-sort fix)

Wraps an existing BoostingFeatures instance, so the normal list-aware
CatBoost feature pipeline and target definition are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.spatiotemporal_features import ComparableAugmentedFeatures


class RatioComparableFeatures(ComparableAugmentedFeatures):
    """Augment list-aware features with historical comps and comp/list gap.

    comp_vs_list_log = historical size-adjusted comparable log value -
                       log(current listing price).

    All comparable prices are derived from fit(train) and are restricted
    by ComparableSalesFeatures to sales *before* the queried listing month.
    """

    @staticmethod
    def _with_list_discrepancy(X: pd.DataFrame, homes: pd.DataFrame) -> pd.DataFrame:
        if "list_price" not in homes.columns:
            raise KeyError("list-aware comparable features require list_price")
        if not X.index.equals(homes.index):
            raise ValueError("Features and home rows must have matching indices")

        listed = pd.to_numeric(homes["list_price"], errors="coerce")
        invalid = ~(np.isfinite(listed) & (listed > 0))
        if invalid.any():
            raise ValueError("list_price must be positive and finite")

        out = X.copy()
        out["comp_vs_list_log"] = (
            out["comp_log_price"] - np.log(listed)
        ).astype("float32")
        # NaN is expected when no eligible comparables exist; CatBoost supports it.
        return out

    def fit_transform(self, homes, model="catboost", track="list_aware"):
        if track != "list_aware":
            raise ValueError("RatioComparableFeatures is only for list_aware")
        X = super().fit_transform(homes, model=model, track=track)
        return self._with_list_discrepancy(X, homes)

    def transform(self, homes, model="catboost"):
        X = super().transform(homes, model=model)
        return self._with_list_discrepancy(X, homes)
