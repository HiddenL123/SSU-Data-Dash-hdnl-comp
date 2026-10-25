"""Serializable full-history LIST-AWARE geometric ensemble of fitted candidates.

Every member has ONE estimator family, ONE objective, and its own training-
fitted feature provider. Ensemble weights combine separate member forecasts
in log space; test homes keep list_price, never post-sale labels.

The training scripts in this branch call fit() only AFTER chronological
member/weight selection has been frozen. This class is *not* wired into
run.py, which is deliberately unchanged in both release branches.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from features.features import make_features
from models.listaware_shotgun_specs import Candidate, parameters


@dataclass
class FittedRatioMember:
    specification: Candidate
    provider: object
    estimator: object
    columns: tuple[str, ...]
    center: float


class ListAwareSelectedEnsemble:
    """End-to-end fit/predict model, safe to persist with models.save_model."""

    def __init__(self, specs: dict[str, Candidate],
                 weights: dict[str, float], *, threads: int = 4):
        if not specs or set(specs) != set(weights):
            raise ValueError("Models and positive blend weights must match")
        if any(not np.isfinite(w) or w <= 0 for w in weights.values()):
            raise ValueError("Ensemble weights must be strictly positive")
        if threads < 1:
            raise ValueError("threads must be positive")
        total = sum(weights.values())
        self.specs = dict(specs)
        self.weights = {k: float(v/total) for k, v in weights.items()}
        self.threads = threads
        self.fitted: dict[str, FittedRatioMember] = {}

    def fit(self, train):
        import pandas as pd
        from catboost import CatBoostRegressor
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor

        if train.empty or not {"sold_price", "list_price"}.issubset(train):
            raise ValueError("Full historical labels and list prices required")
        sold = train.sold_price.to_numpy(float)
        listed = train.list_price.to_numpy(float)
        if not (np.isfinite(sold).all() and np.isfinite(listed).all()
                and (sold > 0).all() and (listed > 0).all()):
            raise ValueError("Invalid training prices")
        self.fitted = {}
        for name, spec in self.specs.items():
            print(f"Full-history list-aware fit: {name}", flush=True)
            provider = make_features(spec.feature_name,
                                     model=spec.backend, track="list_aware")
            X = provider.fit_transform(train)
            if not X.index.equals(train.index):
                raise ValueError(f"{name}: fitted feature row order changed")
            y = np.log(sold) - (np.log(listed) if spec.target == "ratio" else 0)
            center = float(np.median(y)) if spec.target == "direct" else 0.
            ctor = {"catboost": CatBoostRegressor,
                    "xgboost": XGBRegressor,
                    "lightgbm": LGBMRegressor}[spec.backend]
            model = ctor(**parameters(spec, trees=None, threads=self.threads))
            if spec.backend == "catboost":
                cat = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
                model.fit(X, y - center, cat_features=cat)
            else:
                model.fit(X, y - center)
            self.fitted[name] = FittedRatioMember(
                specification=spec, provider=provider, estimator=model,
                columns=tuple(X.columns), center=center
            )
        return self

    def predict_members(self, homes):
        import pandas as pd
        if set(self.fitted) != set(self.specs):
            raise RuntimeError("Complete full-history fit is required")
        if "sold_price" in homes or "date_sold" in homes:
            raise ValueError("Hide test labels before calling predict_members")
        if "list_price" not in homes:
            raise ValueError("List-aware prediction requires list_price")
        listed = homes.list_price.to_numpy(float)
        if not (np.isfinite(listed).all() and (listed > 0).all()):
            raise ValueError("Invalid test list prices")
        result = {}
        for name, entry in self.fitted.items():
            H = entry.provider.transform(homes)
            if not H.index.equals(homes.index) or tuple(H.columns) != entry.columns:
                raise ValueError(f"{name}: fitted/test feature schema mismatch")
            logp = (np.asarray(entry.estimator.predict(H), float).reshape(-1)
                    + entry.center)
            if entry.specification.target == "ratio":
                logp += np.log(listed)
            p = np.exp(logp)
            if p.shape != (len(homes),) or not np.isfinite(p).all() or (p <= 0).any():
                raise ValueError(f"{name}: invalid sale-price prediction")
            result[name] = p
        return pd.DataFrame(result, index=homes.index)

    def predict(self, homes):
        prices = self.predict_members(homes)
        return np.exp(sum(
            weight * np.log(prices[name].to_numpy(float))
            for name, weight in self.weights.items()
        ))
