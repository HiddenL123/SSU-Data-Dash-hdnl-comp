"""Inference-only bridge from saved track-specific ensembles to run.py's model API.

This module does NOT train, choose members, compute ensemble weights, or
modify run.py. Both track-specific bundles must already have been fitted and
saved through models.save_model.save_model(..., track=...).

Use separate adapters for each track, or one EnsembleServingInterface for both.
The .fit() method is deliberately a no-op to match run.py's prediction path;
DO NOT use pretrained bundles in run.py's walk-forward validate() (that method
requires freshly trained, pre-month models and would leak future labels).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from models.save_model import load_model, model_info

TRACKS = ("off_market", "list_aware")


@dataclass(frozen=True)
class SavedEnsembleSpec:
    """Name of a trusted, already fitted track-specific ensemble artifact."""
    track: str
    name: str

    def __post_init__(self):
        if self.track not in TRACKS:
            raise ValueError(f"Unsupported competition track: {self.track}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A saved ensemble model name is required")


class TrainedEnsembleModel:
    """Track-specific .predict(homes) facade using one frozen model bundle.

    Use the SAME object for repeated prediction batches to avoid reloading
    expensive tree ensembles for every request. Saving/importing the feature
    providers is handled by the underlying fitted ensemble bundle.
    """

    def __init__(self, track: str, model_name: str, *,
                 save_dir: str | Path | None = None):
        self.spec = SavedEnsembleSpec(track, model_name)
        self.save_dir = save_dir
        self._model = None
        self._info = None

    @property
    def track(self) -> str:
        return self.spec.track

    @property
    def model_name(self) -> str:
        return self.spec.name

    def load(self):
        """Load only trusted local joblib artifacts, and cache the fitted model."""
        if self._model is None:
            info = model_info(
                self.model_name, track=self.track, save_dir=self.save_dir
            )
            storage_track = info.get("storage_track")
            if storage_track != self.track:
                raise ValueError(
                    f"Saved model track mismatch: expected {self.track}, "
                    f"found {storage_track!r}"
                )
            fitted = load_model(
                self.model_name, track=self.track, save_dir=self.save_dir
            )
            if not callable(getattr(fitted, "predict", None)):
                raise TypeError("Saved ensemble must expose predict(homes)")
            self._model, self._info = fitted, info
        return self

    @property
    def metadata(self):
        self.load()
        return dict(self._info.get("metadata", {}))

    def fit(self, train=None):
        """Adapter for run.py predict(), NOT a retraining implementation.

        run.py.predict() calls .fit(all_listings), which can be satisfied by
        loading the fully pre-trained model. run.py.validate(), by contrast,
        requires refitting the model separately at each historical cutoff:
        this adapter cannot provide that, and MUST NOT be used for validation.
        """
        return self.load()

    def predict(self, homes: pd.DataFrame) -> np.ndarray:
        self.load()
        if not isinstance(homes, pd.DataFrame):
            raise TypeError("predict requires a pandas DataFrame")
        if "sold_price" in homes.columns or "date_sold" in homes.columns:
            raise ValueError("Hide actual sale prices/dates before prediction")
        if self.track == "list_aware":
            if "list_price" not in homes.columns:
                raise ValueError("List-aware prediction requires list_price")
            values = pd.to_numeric(homes["list_price"], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("List prices must be finite and positive")
        if len(homes) == 0:
            return np.empty(0, dtype=float)
        prices = np.asarray(self._model.predict(homes), dtype=float).reshape(-1)
        if (prices.shape != (len(homes),) or
            not np.isfinite(prices).all() or (prices <= 0).any()):
            raise ValueError("Saved ensemble returned invalid predicted prices")
        return prices


class EnsembleServingInterface:
    """Two frozen, independent ensembles with a consistent predict(track, homes).

    Example:
        serving = EnsembleServingInterface(
            off_market_name="off_market_release_candidate_full",
            list_aware_name="listaware_ensemble_release",
        )
        prices = serving.predict("list_aware", hidden_list_aware_homes)
    """

    def __init__(self, *, off_market_name: str, list_aware_name: str,
                 save_dir: str | Path | None = None):
        self.models = {
            "off_market": TrainedEnsembleModel(
                "off_market", off_market_name, save_dir=save_dir
            ),
            "list_aware": TrainedEnsembleModel(
                "list_aware", list_aware_name, save_dir=save_dir
            ),
        }

    def for_track(self, track: str) -> TrainedEnsembleModel:
        if track not in self.models:
            raise ValueError(f"Unknown track {track!r}; choose {TRACKS}")
        return self.models[track]

    def load(self):
        for model in self.models.values():
            model.load()
        return self

    def predict(self, track: str, homes: pd.DataFrame) -> np.ndarray:
        return self.for_track(track).predict(homes)
