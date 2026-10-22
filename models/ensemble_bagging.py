"""Deterministic diversity plans for real-estate XGBoost ensembles.

Sampling logic only: never reads validation labels, and does not redefine
geographic/comparable features. Build the full-vocabulary feature provider on
ALL chronologically eligible training rows BEFORE row resampling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from models.text_features import SELECTED_TERMS, training_vocabulary


@dataclass(frozen=True)
class BagSpec:
    name: str
    text_mode: str                  # selected, full, selected_plus, uniform, weighted
    n_text: int = 150
    selected_weight: float = 4.0
    row_mode: str = "all"           # all, bootstrap, subbag
    row_fraction: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if self.text_mode not in {"selected", "full", "selected_plus", "uniform", "weighted"}:
            raise ValueError("Unknown text feature bag mode")
        if self.row_mode not in {"all", "bootstrap", "subbag"}:
            raise ValueError("Unknown training-row bag mode")
        if self.n_text < 1 or self.selected_weight <= 0 or not 0 < self.row_fraction <= 1:
            raise ValueError("Invalid feature/row bag settings")

    def record(self):
        return asdict(self)


def build_text_bank(train, *, max_extra_terms=1000, min_df=10):
    """Label-free training-fitted 1000-term candidate bank + all selected terms.

    The original 47 selected terms are ALWAYS retained in the candidate
    bank, even if a particular term falls below this fold's document-frequency
    threshold. Additional terms are learned only on historic train remarks.
    """
    if "remarks" not in train:
        raise ValueError("Training homes must include remarks")
    extra = training_vocabulary(
        train["remarks"], max_features=max_extra_terms, min_df=min_df
    )
    return tuple(dict.fromkeys((*SELECTED_TERMS, *extra)))


def select_text_terms(bank, spec: BagSpec):
    """Sample a reproducible BoW column subset, WITHOUT validation labels."""
    pool = tuple(dict.fromkeys(str(term) for term in bank))
    if not pool:
        raise ValueError("Empty training-fitted text vocabulary")
    selected = frozenset(SELECTED_TERMS)
    rng = np.random.default_rng(spec.seed)
    anchors = [term for term in pool if term in selected]
    additional = [term for term in pool if term not in selected]

    if spec.text_mode == "full":
        return pool
    if spec.text_mode == "selected":
        if len(anchors) != len(selected):
            raise ValueError("The text bank is missing selected reference terms")
        return tuple(anchors)
    if spec.text_mode == "selected_plus":
        if len(anchors) != len(selected):
            raise ValueError("The text bank is missing selected reference terms")
        n = min(spec.n_text, len(additional))  # n_text means *extra* terms here
        other = rng.choice(additional, size=n, replace=False) if n else ()
        return tuple((*anchors, *other))
    n = min(spec.n_text, len(pool))
    p = None
    if spec.text_mode == "weighted":
        w = np.array([
            spec.selected_weight if term in selected else 1.0 for term in pool
        ], dtype=float)
        p = w / w.sum()
    return tuple(rng.choice(pool, size=n, replace=False, p=p).tolist())


def sample_training_rows(n_rows, spec: BagSpec):
    """Sample rows of the ALREADY FITTED full-history feature matrix."""
    if n_rows < 1:
        raise ValueError("Need at least one training row")
    rng = np.random.default_rng(spec.seed + 1_000_003)
    count = max(1, int(round(n_rows * spec.row_fraction)))
    if spec.row_mode == "all":
        return np.arange(n_rows)
    if spec.row_mode == "bootstrap":
        return rng.choice(n_rows, size=count, replace=True)
    return rng.choice(n_rows, size=count, replace=False)


def starter_specs(seed=42):
    """Small auditable roster. Expand only if candidate diversity proves useful."""
    return (
        BagSpec("selected_47_reference", "selected", seed=seed),
        BagSpec("full_vocab_binary", "full", seed=seed),
        BagSpec("selected_plus_100", "selected_plus", n_text=100, seed=seed+1),
        BagSpec("uniform_150_a", "uniform", n_text=150, seed=seed+2),
        BagSpec("weighted_150_a", "weighted", n_text=150, seed=seed+3),
        BagSpec("weighted_150_b", "weighted", n_text=150, seed=seed+4),
        BagSpec("weighted_150_bootstrap_a", "weighted", n_text=150,
                row_mode="bootstrap", seed=seed+5),
        BagSpec("weighted_150_bootstrap_b", "weighted", n_text=150,
                row_mode="bootstrap", seed=seed+6),
        BagSpec("selected_47_bootstrap_a", "selected",
                row_mode="bootstrap", seed=seed+7),
        BagSpec("selected_47_bootstrap_b", "selected",
                row_mode="bootstrap", seed=seed+8),
        BagSpec("weighted_150_subbag80", "weighted", n_text=150,
                row_mode="subbag", row_fraction=0.8, seed=seed+9),
    )
