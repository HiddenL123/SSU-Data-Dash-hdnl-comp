"""Unchanged boosting hyperparameters from the legacy implementation."""

DEFAULT_PARAMS = {
    "catboost": {
        "iterations": 900,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    },
    "lightgbm": {
        "n_estimators": 900,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 40,
        "reg_lambda": 5,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 900,
        "learning_rate": 0.04,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 5,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42,
        "n_jobs": -1,
    },
}

