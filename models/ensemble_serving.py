"""Inference-only bridge from saved track-specific ensembles to run.py's model API.

This module does NOT train, choose members, compute ensemble weights, or
modify run.py. Both track-specific bundles must already have been fitted and
saved through models.save_model.save_model(..., track=...).

Use separate adapters for each track, or one EnsembleServingInterface for both.
Call .load() or .predict() directly. .fit() always rejects use: run.py's
prediction and validation paths both call it, and there is no reliable way to
distinguish final inference from chronological fitting by inspecting train.
Unmodified run.py still constructs fresh models through models.model.make_model;
it does not select these saved ensembles.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from models.save_model import load_model, model_info
from models.final_ensemble import WeightedOffMarketEnsemble
from models.ensemble_snapshot import BaggedMember, FeatureBaggingEnsemble
from models.listaware_factory_snapshot import ListAwareBaggedMember, ListAwareFactoryEnsemble

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
            if not isinstance(info, dict) or info.get("format_version") != 1:
                raise ValueError("Unsupported saved ensemble metadata format")
            if info.get("name") != self.model_name:
                raise ValueError("Saved model metadata name mismatch")
            storage_track = info.get("storage_track")
            if storage_track != self.track:
                raise ValueError(
                    f"Saved model track mismatch: expected {self.track}, "
                    f"found {storage_track!r}"
                )
            metadata = info.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("Saved model metadata must be a dictionary")
            if (metadata.get("track", self.track) != self.track or
                    metadata.get("tracks", [self.track]) != [self.track]):
                raise ValueError("Saved model metadata track mismatch")
            fitted = load_model(
                self.model_name, track=self.track, save_dir=self.save_dir
            )
            expected = (WeightedOffMarketEnsemble if self.track == "off_market"
                        else ListAwareFactoryEnsemble)
            if not isinstance(fitted, expected):
                raise TypeError(f"Saved {self.track} ensemble must be {expected.__name__}")
            self._validate_bundle(fitted)
            self._model, self._info = fitted, info
        return self

    def _validate_bundle(self, fitted):
        # Unpickling bypasses constructors, so check persisted state as well.
        members = getattr(fitted, "members", None)
        weights = getattr(fitted, "weights", None)
        if (not isinstance(members, dict) or not members or
                not isinstance(weights, dict) or set(members) != set(weights)):
            raise ValueError("Saved ensemble needs exactly one weight per member")
        try:
            w = np.asarray(list(weights.values()), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid saved ensemble weights") from error
        if (w.shape != (len(members),) or not np.isfinite(w).all() or
                (w < 0).any() or not np.isfinite(w.sum()) or w.sum() <= 0):
            raise ValueError("Invalid saved ensemble weights")
        for name, bundle in members.items():
            if self.track == "off_market":
                if (not isinstance(bundle, FeatureBaggingEnsemble) or
                        not isinstance(getattr(bundle, "members", None), dict) or
                        name not in getattr(bundle, "members", {})):
                    raise TypeError(f"Invalid saved off-market member: {name}")
                member = bundle.members[name]
                expected_member = BaggedMember
            else:
                member = bundle
                expected_member = ListAwareBaggedMember
            if (not isinstance(member, expected_member) or
                    not callable(getattr(getattr(bundle, "feature_provider", None), "transform", None)) or
                    not callable(getattr(getattr(member, "estimator", None), "predict", None)) or
                    not hasattr(member, "selected_columns") or
                    not hasattr(member, "target_offset")):
                raise TypeError(f"Incomplete fitted member: {name}")
            try:
                offset = float(member.target_offset)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid fitted target offset: {name}") from error
            if not np.isfinite(offset):
                raise ValueError(f"Invalid fitted target offset: {name}")

    @property
    def metadata(self):
        self.load()
        return deepcopy(self._info.get("metadata", {}))

    def fit(self, train=None):
        """Reject chronological fitting rather than silently reuse future data."""
        raise RuntimeError(
            "Pretrained ensembles cannot fit or perform walk-forward validation. "
            "Use load() and predict(homes) explicitly for submission inference."
        )

    def predict(self, homes: pd.DataFrame) -> np.ndarray:
        if not isinstance(homes, pd.DataFrame):
            raise TypeError("predict requires a pandas DataFrame")
        if not homes.columns.is_unique:
            raise ValueError("Prediction columns must be unique")
        if "sold_price" in homes.columns or "date_sold" in homes.columns:
            raise ValueError("Hide actual sale prices/dates before prediction")
        if self.track == "list_aware":
            if "list_price" not in homes.columns:
                raise ValueError("List-aware prediction requires list_price")
            values = pd.to_numeric(homes["list_price"], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("List prices must be finite and positive")
        self.load()
        if len(homes) == 0:
            return np.empty(0, dtype=float)
        try:
            if self.track == "off_market":
                # The saved bundle's predict_member does not check row order.
                # Use each original provider and member, without refitting or
                # changing any persisted weights or feature selections.
                outputs = []
                for name, bundle in self._model.members.items():
                    features = bundle.feature_provider.transform(homes.copy(deep=True))
                    if not features.index.equals(homes.index):
                        raise ValueError("Prediction rows were reordered")
                    values = np.asarray(bundle.members[name].predict_from_features(features), float)
                    if values.shape != (len(homes),):
                        raise ValueError("Saved member returned invalid prediction shape")
                    outputs.append(values)
                w = np.asarray([self._model.weights[n] for n in self._model.members], float)
                prices = np.exp(np.log(np.column_stack(outputs)) @ (w / w.sum()))
            else:
                prices = np.asarray(self._model.predict(homes.copy(deep=True)), dtype=float)
        except KeyError as error:
            raise ValueError(f"Missing required prediction columns or features: {error}") from error
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
