"""Deployable off-market geometric ensemble with independently fitted providers.

Members can have different bootstrap rows, text vocabularies and spatial
features, so they MUST NOT share a single provider unless proven identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class WeightedOffMarketEnsemble:
    def __init__(self, members: dict, weights: dict, *, provenance=None):
        self.members = dict(members)
        self.weights = {str(k): float(v) for k, v in weights.items()}
        self.provenance = dict(provenance or {})
        if not self.members or set(self.weights) != set(self.members):
            raise ValueError("Exactly one weight per fitted ensemble member required")
        w = np.array(list(self.weights.values()), dtype=float)
        if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("Weights must be nonnegative, finite and sum positive")

    def predict_members(self, homes):
        outputs = {}
        for name, bundle in self.members.items():
            # FeatureBaggingEnsemble has its own fitted historical provider.
            preds = np.asarray(bundle.predict_member(homes, name), dtype=float)
            if preds.shape != (len(homes),) or not np.isfinite(preds).all() or (preds <= 0).any():
                raise ValueError(f"Bad member predictions: {name}")
            outputs[name] = preds
        return pd.DataFrame(outputs, index=homes.index)

    def predict(self, homes):
        if len(homes) == 0:
            return np.empty(0, dtype=float)
        result = self.predict_members(homes)
        w = np.array([self.weights[n] for n in result.columns], dtype=float)
        return np.exp(np.log(result.to_numpy(dtype=float)) @ (w / w.sum()))
