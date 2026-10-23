"""Execution profiles and immutable full-strength release-candidate settings.

A worker profile controls *resources*, NOT data splits, model hyperparameters,
candidate seeds or predictions. Switching between Colab by day and a local PC
overnight must NEVER silently change the research experiment.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerProfile:
    name: str
    xgb_threads: int
    description: str
    kind: str


WORKER_PROFILES = {
    "day_cloud": WorkerProfile(
        "day_cloud", 2,
        "Colab CPU runtime; free the user's laptop; use local VM disk and archive checkpoints to Drive",
        "cloud",
    ),
    "overnight_local": WorkerProfile(
        "overnight_local", 8,
        "Local full-speed CPU training when the computer is otherwise idle",
        "local",
    ),
    "day_local_quiet": WorkerProfile(
        "day_local_quiet", 2,
        "Optional quiet local training while using the computer for other work",
        "local",
    ),
}


def get_profile(name):
    if name not in WORKER_PROFILES:
        raise ValueError(f"Unknown execution profile {name!r}: {tuple(WORKER_PROFILES)}")
    return WORKER_PROFILES[name]


# Full-strength selection and audit. Earlier pilot used just 220 trees and
# 5,000 sales; this release candidate intentionally trains on ALL eligible
# historical sales for each chronological fold and predicts full sale months.
# The validation periods have previously been examined by this project:
# audit isolation is per-factory, not a newly untouched holdout.
RELEASE_SETTINGS = {
    "tuning_months": ("2025-04", "2025-05"),
    "selection_months": ("2025-06", "2025-07"),
    "audit_months": ("2025-08", "2025-09", "2025-10"),
    "train_start": "2022-01-01",
    "train_cap": 1_000_000,
    "validation_cap": 1_000_000,
    "trees": 700,
    "max_extra_terms": 1000,
    "min_df": 10,
    "candidates_per_generation": 3,
    "generations": 2,
    "optuna_trials": 5,
    "minimum_ensemble_gain_pp": 0.02,
    "max_mape_regression_pp": 0.03,
    "max_members": 8,
    "seed": 42,
    "output_dir": "outputs/ensemble_release_candidate_full",
    "save_accepted_snapshots": False,
    "screen_max_mdape": 22.0,
}


def release_settings(*, output_dir=None, generations=None):
    settings = dict(RELEASE_SETTINGS)
    if output_dir is not None:
        settings["output_dir"] = str(output_dir)
    if generations is not None:
        settings["generations"] = int(generations)
    return settings
