"""List-aware counterpart to models.automl_search candidate generation.

Same feature-bagging/row-bagging candidate structure as off-market, with
a frozen recipe from the existing curated pool and an estimator family.
Candidate identity and seed fully determine feature/row sampling.
"""
from dataclasses import dataclass, asdict
import numpy as np
from models.automl_search import TEXT_MODES, ROW_MODES
from models.listaware_candidate_pool import CURATED_BY_FAMILY

RECIPE_FAMILIES = {
    recipe: family for family, names in CURATED_BY_FAMILY.items()
    for recipe in names
}


@dataclass(frozen=True)
class Candidate:
    name: str
    seed: int
    recipe: str
    family: str
    text_mode: str = "selected"
    n_text: int = 150
    selected_weight: float = 4.0
    row_mode: str = "all"
    row_fraction: float = 1.0

    def __post_init__(self):
        if self.recipe not in RECIPE_FAMILIES:
            raise ValueError("Unknown list-aware feature recipe")
        if RECIPE_FAMILIES[self.recipe] != self.family:
            raise ValueError("Recipe's backend and estimator family disagree")
        if self.text_mode not in TEXT_MODES or self.row_mode not in ROW_MODES:
            raise ValueError("Invalid text or row bagging mode")
        if self.n_text < 1 or self.selected_weight <= 0 or not np.isfinite(self.selected_weight):
            raise ValueError("Invalid text bagging settings")
        if not 0 < self.row_fraction <= 1 or not np.isfinite(self.row_fraction):
            raise ValueError("Invalid row fraction")

    def record(self):
        return asdict(self)


def generate_candidates(generation: int, count: int, seed: int):
    """First generation seeds three distinct model families; later candidates
    draw from family-balanced recipes with different feature/row bags.
    """
    if generation < 0 or count < 1:
        raise ValueError("Invalid generation or candidate count")
    rng = np.random.default_rng(seed + 10007 * generation)
    refs = (("catboost", "cat_mae_spatial_geo"),
            ("xgboost", "xgb_splines"),
            ("lightgbm", "lgb_splines"))
    result = []
    if generation == 0:
        for i, (family, recipe) in enumerate(refs[:count]):
            result.append(Candidate(
                name=f"reference_{family}", seed=seed + i,
                recipe=recipe, family=family, text_mode="selected"
            ))
    families = ("catboost", "xgboost", "lightgbm")
    while len(result) < count:
        i = len(result)
        family = families[int(rng.integers(len(families)))]
        available = CURATED_BY_FAMILY[family]
        recipe = available[int(rng.integers(len(available)))]
        mode = TEXT_MODES[int(rng.integers(len(TEXT_MODES)))]
        row_mode = ROW_MODES[int(rng.integers(len(ROW_MODES)))]
        result.append(Candidate(
            name=f"g{generation:03d}_c{i:03d}_{family}",
            seed=int(rng.integers(1, 2**31 - 1)),
            family=family, recipe=recipe,
            text_mode=mode,
            n_text=int(rng.choice((75, 150, 250))),
            row_mode=row_mode, row_fraction=0.8 if row_mode == "subbag" else 1.,
        ))
    return tuple(result)
