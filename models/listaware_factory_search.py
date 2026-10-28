
"""Candidate generation for the list-aware AutoML ensemble.

Based on models.automl_search, with:
- CatBoost / XGBoost / LightGBM
- Optional comparable-sale adjustments
- Randomized spatial features
- Text feature bagging
- Row bootstrap/subbagging

No curated feature-recipe restriction.
"""

from dataclasses import dataclass, asdict
import numpy as np

from models.automl_search import (
    TEXT_MODES,
    ROW_MODES,
    ADDON_OPTIONS,
)

MODEL_FAMILIES = ("catboost", "xgboost", "lightgbm")

CATBOOST_OBJECTIVES = (
    "MAE",
    "RMSE",
    "Huber:delta=0.05",
)

COMPARABLE_OPTIONS = (
    "base",
    "comp_type_quota",
    "comp_age_first",
    "comp_both",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    seed: int

    family: str = "catboost"
    objective: str = "MAE"

    text_mode: str = "selected"
    n_text: int = 150
    selected_weight: float = 4.0

    row_mode: str = "all"
    row_fraction: float = 1.0

    addons: tuple[str, ...] = ()
    comparable: str = "base"

    def __post_init__(self):
        if self.family not in MODEL_FAMILIES:
            raise ValueError("Unsupported model family")

        if self.family == "catboost":
            if self.objective not in CATBOOST_OBJECTIVES:
                raise ValueError("Unsupported CatBoost objective")
        elif self.objective != "RMSE":
            raise ValueError("XGBoost and LightGBM use RMSE here")

        if self.text_mode not in TEXT_MODES:
            raise ValueError("Invalid text mode")

        if self.row_mode not in ROW_MODES:
            raise ValueError("Invalid row sampling mode")

        if tuple(self.addons) not in ADDON_OPTIONS:
            raise ValueError("Unsupported spatial combination")

        if self.comparable not in COMPARABLE_OPTIONS:
            raise ValueError("Unsupported comparable configuration")

        if self.n_text < 1:
            raise ValueError("Invalid text feature count")

        if not np.isfinite(self.selected_weight) or self.selected_weight <= 0:
            raise ValueError("Invalid text feature weight")

        if not np.isfinite(self.row_fraction) or not (
            0 < self.row_fraction <= 1
        ):
            raise ValueError("Invalid row fraction")

    def record(self):
        return asdict(self)


def generate_candidates(generation: int, count: int, seed: int):
    if generation < 0 or count < 1:
        raise ValueError("Invalid generation or candidate count")

    rng = np.random.default_rng(seed + 10007 * generation)

    result = []

    # Start from known strong model families.
    # These are reference candidates, not restrictions on later search.
    references = (
        Candidate(
            name="reference_catboost",
            seed=seed,
            family="catboost",
            objective="MAE",
            text_mode="selected",
            addons=("spacetime_splines", "geo_context"),
        ),
        Candidate(
            name="reference_xgboost",
            seed=seed + 1,
            family="xgboost",
            objective="RMSE",
            text_mode="selected",
            addons=("spacetime_splines", "geo_context"),
        ),
        Candidate(
            name="reference_lightgbm",
            seed=seed + 2,
            family="lightgbm",
            objective="RMSE",
            text_mode="selected",
            addons=("spacetime_splines", "geo_context"),
        ),
    )

    if generation == 0:
        result.extend(references[:count])

    while len(result) < count:
        i = len(result)

        family = MODEL_FAMILIES[
            int(rng.integers(len(MODEL_FAMILIES)))
        ]

        objective = (
            CATBOOST_OBJECTIVES[
                int(rng.integers(len(CATBOOST_OBJECTIVES)))
            ]
            if family == "catboost"
            else "RMSE"
        )

        text_mode = TEXT_MODES[
            int(rng.integers(len(TEXT_MODES)))
        ]

        row_mode = ROW_MODES[
            int(rng.integers(len(ROW_MODES)))
        ]

        addons = ADDON_OPTIONS[
            int(rng.integers(len(ADDON_OPTIONS)))
        ]

        comparable = COMPARABLE_OPTIONS[
            int(rng.integers(len(COMPARABLE_OPTIONS)))
        ]

        result.append(
            Candidate(
                name=f"g{generation:03d}_c{i:03d}_{family}",
                seed=int(rng.integers(1, 2**31 - 1)),
                family=family,
                objective=objective,
                text_mode=text_mode,
                n_text=int(rng.choice((75, 150, 250))),
                selected_weight=4.0,
                row_mode=row_mode,
                row_fraction=0.8 if row_mode == "subbag" else 1.0,
                addons=addons,
                comparable=comparable,
            )
        )

    return tuple(result)
