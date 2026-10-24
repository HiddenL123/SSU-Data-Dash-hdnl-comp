"""Predeclared list-aware candidates from recorded winning experiments.

These specifications are research-only; no existing fitted models are edited.
E28: CatBoost Huber sale/list ratio, 4.06% full historical validation.
E32: tuned XGBoost, E31: tuned LightGBM, E33: tuned CatBoost RMSE.
The new arms transfer the strongest off-market spline + geographic context
features to CatBoost as well as XGBoost, including the AVM add-ons.
"""
from __future__ import annotations

from dataclasses import dataclass
from models.defaults import DEFAULT_PARAMS
from spacetime_fast import XGB_TUNED

BASIC = "Xpt_spatiotemporal"
BINARY = BASIC + "__selected_binary"
SPATIAL = BINARY + "__st_spacetime_splines+geo_context"
ADJUSTED = SPATIAL + "+rw_adjusted_comps"
ADVANCED = SPATIAL + "+rw_combined"
HEDONIC = SPATIAL + "+hedonic_index"
ADAPTIVE = SPATIAL + "+adaptive_comps"
TIME_ADJUSTED = SPATIAL + "+time_adjusted_comp"

# Literal parameter values from optuna.ipynb / model_run.ipynb.
LGB_TUNED = {
    "n_estimators": 1800,
    "learning_rate": 0.052995139011091236,
    "max_depth": 6,
    "num_leaves": 40,
    "min_child_samples": 30,
    "reg_lambda": 16.19372143031597,
    "reg_alpha": 0.0,
    "subsample": 0.8349167453037107,
    "subsample_freq": 1,
    "colsample_bytree": 0.953443325579757,
    "max_bin": 255,
    "objective": "regression",
    "random_state": 42,
    "n_jobs": 4,
    "verbosity": -1,
}
CAT_TUNED_RMSE = {
    "iterations": 1400, "depth": 8,
    "learning_rate": 0.07127609366218435,
    "l2_leaf_reg": 4.14053058084107,
    "loss_function": "RMSE",
    "random_seed": 42, "verbose": False,
    "allow_writing_files": False,
    "task_type": "CPU", "thread_count": 4,
    "random_strength": 0.580328206016201,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    backend: str
    feature_name: str
    target: str = "ratio"
    loss: str = "RMSE"
    tuned: bool = False

    def __post_init__(self):
        if self.backend not in {"catboost", "xgboost", "lightgbm"}:
            raise ValueError("Unsupported model family")
        if self.target not in {"ratio", "direct"}:
            raise ValueError("Unsupported target")


CANDIDATES = {
    # Historically strongest list-aware method, our matched baseline.
    "cat_huber_baseline": Candidate(
        "cat_huber_baseline", "catboost", BASIC, loss="Huber:delta=0.05"
    ),
    "cat_huber_binary": Candidate(
        "cat_huber_binary", "catboost", BINARY, loss="Huber:delta=0.05"
    ),
    # Explicit CatBoost + best XGBoost representation (the user's key request).
    "cat_huber_spline_geo": Candidate(
        "cat_huber_spline_geo", "catboost", SPATIAL, loss="Huber:delta=0.05"
    ),
    "cat_huber_adjusted": Candidate(
        "cat_huber_adjusted", "catboost", ADJUSTED, loss="Huber:delta=0.05"
    ),
    "cat_huber_advanced": Candidate(
        "cat_huber_advanced", "catboost", ADVANCED, loss="Huber:delta=0.05"
    ),
    "cat_huber_hedonic": Candidate(
        "cat_huber_hedonic", "catboost", HEDONIC, loss="Huber:delta=0.05"
    ),
    "cat_huber_adaptive": Candidate(
        "cat_huber_adaptive", "catboost", ADAPTIVE, loss="Huber:delta=0.05"
    ),
    "cat_huber_time_adjusted": Candidate(
        "cat_huber_time_adjusted", "catboost", TIME_ADJUSTED,
        loss="Huber:delta=0.05"
    ),
    "cat_mae": Candidate("cat_mae", "catboost", BASIC, loss="MAE"),
    "cat_rmse": Candidate("cat_rmse", "catboost", BASIC),
    "cat_tuned_rmse": Candidate(
        "cat_tuned_rmse", "catboost", BASIC, tuned=True
    ),
    "cat_huber_direct": Candidate(
        "cat_huber_direct", "catboost", BASIC,
        target="direct", loss="Huber:delta=0.05"
    ),
    "xgb_tuned_binary": Candidate("xgb_tuned_binary", "xgboost", BINARY, tuned=True),
    "xgb_spline_geo": Candidate("xgb_spline_geo", "xgboost", SPATIAL, tuned=True),
    "xgb_adjusted": Candidate("xgb_adjusted", "xgboost", ADJUSTED, tuned=True),
    "xgb_advanced": Candidate("xgb_advanced", "xgboost", ADVANCED, tuned=True),
    "lgb_tuned_binary": Candidate("lgb_tuned_binary", "lightgbm", BINARY, tuned=True),
    "lgb_spline_geo": Candidate("lgb_spline_geo", "lightgbm", SPATIAL, tuned=True),
}

# The first pass deliberately balances the breadth of methodologies and runtime.
DEFAULT_CANDIDATES = (
    "cat_huber_baseline", "cat_huber_binary", "cat_huber_spline_geo",
    "cat_huber_adjusted", "cat_huber_advanced", "cat_mae",
    "xgb_tuned_binary", "xgb_spline_geo", "xgb_adjusted",
    "lgb_tuned_binary", "lgb_spline_geo",
)

# All weights are FIXED BEFORE observing this screen's validation outcomes.
# Candidates must be present to activate a blend.
BLENDS = {
    "blend_cat_baseline_xgb": {
        "cat_huber_baseline": .70, "xgb_tuned_binary": .30,
    },
    "blend_cat_xgb_spatial": {
        "cat_huber_spline_geo": .70, "xgb_spline_geo": .30,
    },
    "blend_cat_xgb_adjusted": {
        "cat_huber_adjusted": .65, "xgb_adjusted": .35,
    },
    "blend_cat_xgb_lgb": {
        "cat_huber_spline_geo": .50, "xgb_spline_geo": .35,
        "lgb_spline_geo": .15,
    },
    "blend_cat_baseline_spatial_xgb": {
        "cat_huber_baseline": .40, "cat_huber_spline_geo": .35,
        "xgb_spline_geo": .25,
    },
}


def parameters(candidate: Candidate, *, trees=None, threads=4):
    """Use recorded full-strength params, with an optional PILOT tree cap."""
    if candidate.backend == "catboost":
        p = dict(CAT_TUNED_RMSE if candidate.tuned else DEFAULT_PARAMS["catboost"])
        p["loss_function"] = candidate.loss
        p["thread_count"] = threads
        p["allow_writing_files"] = False
        p["verbose"] = False
        p["random_seed"] = 42
        if trees is not None:
            p["iterations"] = min(int(p["iterations"]), int(trees))
    elif candidate.backend == "xgboost":
        p = dict(XGB_TUNED if candidate.tuned else DEFAULT_PARAMS["xgboost"])
        p["n_jobs"] = threads
        p["device"] = "cpu"
        p["enable_categorical"] = True
        if trees is not None:
            p["n_estimators"] = min(int(p["n_estimators"]), int(trees))
    else:
        p = dict(LGB_TUNED if candidate.tuned else DEFAULT_PARAMS["lightgbm"])
        p["n_jobs"] = threads
        if trees is not None:
            p["n_estimators"] = min(int(p["n_estimators"]), int(trees))
    return p
