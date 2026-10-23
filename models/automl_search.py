"""Pure candidate-generation and ensemble-admission logic for AutoML v1.

No validation targets enter candidate generation. The ensemble never admits a
member just for low standalone loss or low correlation: it must improve a
matched, later-fold log-price ensemble under bounded nonnegative weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

TEXT_MODES = ("selected", "selected_plus", "weighted", "uniform", "full")
ROW_MODES = ("all", "bootstrap", "subbag")
ADDON_OPTIONS = (
    (), ("spacetime_splines",), ("geo_context",),
    ("spacetime_splines", "geo_context"), ("market",),
)
WEIGHT_GRID = (0.05, 0.10, 0.20, 0.35, 0.50)


@dataclass(frozen=True)
class Candidate:
    name: str
    seed: int
    text_mode: str = "selected"
    n_text: int = 150
    selected_weight: float = 4.0
    row_mode: str = "all"
    row_fraction: float = 1.0
    addons: tuple[str, ...] = ()

    def __post_init__(self):
        if self.text_mode not in TEXT_MODES or self.row_mode not in ROW_MODES:
            raise ValueError("Unrecognized text or row bagging mode")
        if tuple(self.addons) not in ADDON_OPTIONS:
            raise ValueError("Unsupported spatial feature-addition combination")
        if self.n_text < 1 or not np.isfinite(self.selected_weight) or self.selected_weight <= 0:
            raise ValueError("Text bag settings must be positive")
        if not np.isfinite(self.row_fraction) or not 0 < self.row_fraction <= 1:
            raise ValueError("Row fraction must be in (0,1]")

    def record(self):
        return asdict(self)


def generate_candidates(generation: int, count: int, seed: int):
    """Include one reproducible strong reference, then random diverse proposals."""
    if generation < 0 or count < 1:
        raise ValueError("Invalid generation/count")
    rng = np.random.default_rng(seed + 10_007 * generation)
    result = []
    if generation == 0:
        result.append(Candidate(
            name="selected_47_splines_geo_reference",
            seed=seed, text_mode="selected",
            addons=("spacetime_splines", "geo_context"),
        ))
    while len(result) < count:
        ordinal = len(result)
        mode = TEXT_MODES[int(rng.integers(len(TEXT_MODES)))]
        row = ROW_MODES[int(rng.integers(len(ROW_MODES)))]
        additions = ADDON_OPTIONS[int(rng.integers(len(ADDON_OPTIONS)))]
        result.append(Candidate(
            name=f"g{generation:03d}_c{ordinal:03d}",
            seed=int(rng.integers(1, 2**31 - 1)),
            text_mode=mode, n_text=int(rng.choice([75, 150, 250])),
            row_mode=row,
            row_fraction=0.8 if row == "subbag" else 1.0,
            addons=additions,
        ))
    return tuple(result)


def metrics(y, predicted):
    y = np.asarray(y, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if not len(y) or y.shape != p.shape:
        raise ValueError("Actual and predicted prices must be aligned 1-D arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y <= 0).any() or (p <= 0).any():
        raise ValueError("Housing sale/predicted prices must be finite and positive")
    e = 100.0 * np.abs(p-y)/y
    return {"homes": int(len(y)), "mdape": float(np.median(e)),
            "mape": float(e.mean()), "p90_ape": float(np.quantile(e, .90))}


def objective(y, pred):
    m = metrics(y, pred)
    return 0.65*m["mdape"] + 0.35*m["mape"]


def log_blend(old, proposed, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Blending coefficient out of bounds")
    old, proposed = np.asarray(old, float), np.asarray(proposed, float)
    if old.shape != proposed.shape or not np.isfinite(old).all() or not np.isfinite(proposed).all() or (old <= 0).any() or (proposed <= 0).any():
        raise ValueError("Blend candidates must be finite, aligned, positive prices")
    return np.exp((1-alpha)*np.log(old) + alpha*np.log(proposed))


def choose_admission(
    y, candidate, current=None, *, min_gain=0.02,
    max_mape_regression=0.03, weights=WEIGHT_GRID
):
    """Select candidate's bounded log blending weight on selection folds only.

    A first valid reference model is seeded without comparison. Later candidates
    require a positive weighted-objective improvement AND a MAPE safeguard.
    This intentionally DOES NOT view shadow-audit prices.
    """
    solo = metrics(y, candidate)
    if current is None:
        return {"accept": True, "weight": 1.0, "gain": None,
                "solo": solo, "updated": solo}
    original = metrics(y, current)
    base_loss = objective(y, current)
    trials = []
    for weight in weights:
        combo = log_blend(current, candidate, weight)
        m = metrics(y, combo)
        gain = base_loss - objective(y, combo)
        if m["mape"] <= original["mape"] + max_mape_regression:
            trials.append((gain, weight, m))
    if not trials:
        return {"accept": False, "weight": 0., "gain": 0.,
                "solo": solo, "updated": original}
    gain, weight, updated = max(trials, key=lambda x: x[0])
    return {"accept": bool(gain >= min_gain), "weight": float(weight),
            "gain": float(gain), "solo": solo, "updated": updated}


def updated_weights(old_weights: dict[str, float], name: str, alpha: float):
    """Maintain an equivalent flat convex mixture of log-model predictions."""
    if name in old_weights or not 0 < alpha <= 1:
        raise ValueError("Duplicate model or invalid ensemble weight")
    if not old_weights:
        if alpha != 1.0:
            raise ValueError("First ensemble member must have weight 1")
        return {name: 1.0}
    result = {key: (1-alpha)*v for key, v in old_weights.items()}
    result[name] = alpha
    return result
