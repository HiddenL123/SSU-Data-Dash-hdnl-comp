# Dual-track long-running ensemble workspace

**Purpose:** a small code-only checkout to hand to someone with a stronger PC
for **off-market** and **list-aware** ensemble selection. Built from the current
`main` with unrelated experimental notebooks and shotgun scripts excluded.
The complete research history remains on `main`.

The two tracks are deliberately separate. Off-market has no listing price;
list-aware uses the sale/list ratio. **Never blend them or share targets.**

## Setup

```bash
git clone --branch ensemble/dual-track-longrun-clean --single-branch https://github.com/HiddenL123/SSU-Data-Dash-hdnl-comp.git
cd SSU-Data-Dash-hdnl-comp
uv sync
# The off-market factory imports Optuna, which is optional and not locked:
uv pip install optuna
```

Put the *same* historical `listings.parquet` / `listings.csv` in `data/`,
or set `DATA_DIR` to its directory. The manifests refuse to reuse checkpoints
with differing source files, models, and settings. Never run concurrent
writers to the same output directory.

**Data, checkpoints and fitted model bundles are intentionally NOT committed**
to this branch. Keep `outputs/`, `saved_models/` fitted artifacts and exported
checkpoints on local/persistent storage; transfer them securely between PCs.
A public GitHub branch does not make trained models appear automatically.

## A. Off-market full-strength long-run factory

The original off-market `models.ensemble_automl_factory` is unchanged.
This uses full eligible chronological history, 700 boosting rounds, Optuna
on earlier folds, later-fold marginal ensemble admission and locked audit.
A larger `--generations` increases candidates; it does **not** change the
five Optuna trials per proposal, fixed tree count, search space or baseline.

```bash
uv run python ensemble_rc.py run --profile overnight_local --generations 20 --output-dir outputs/offmarket_longrun_20
uv run python ensemble_rc.py status --output-dir outputs/offmarket_longrun_20
# Do NOT open the audit until the intended 20 generations are finished.
uv run python ensemble_rc.py audit --profile overnight_local --generations 20 --output-dir outputs/offmarket_longrun_20
uv run python ensemble_rc.py benchmark --profile overnight_local --generations 20 --output-dir outputs/offmarket_longrun_20
uv run python ensemble_rc.py finalize --profile overnight_local --generations 20 --output-dir outputs/offmarket_longrun_20 --model-name offmarket_longrun_20
```

The existing factory writes strict restartable checkpoints and uses an OS lock.
**Start at 20 generations in a new empty folder**; do not change generation
count or any frozen manifest values on an already-started or audited run.
The benchmark is not an untouched test.

Optional portable checkpoint export:

```bash
uv run python ensemble_rc.py export --output-dir outputs/offmarket_longrun_20 --checkpoint-archive offmarket_checkpoint.zip
```

## B. List-aware individual candidates, ensemble selection and audit

The list-aware branch still uses **existing historically tuned CatBoost/XGB/LGB
hyperparameter sets**, NOT yet a general multi-family Optuna factory or
text/row bagging search. It *does* fit a large, factorized candidate pool and
select the actual **ensemble members and marginal weights** using later
chronological folds.

```bash
# First explore a diverse candidate pool; restartable model/month caches.
uv run python -m feature.listaware_pool_screen --stage screen --suite all

# Shortlist using only Mar--May screen, then full-history fit on Jun--Jul.
# 12 is the requested shortlist size (at least 4 historical controls).
uv run python -m feature.listaware_pool_screen --stage confirm --suite all --top-k 12 --screen-dir outputs/listaware_pool_screen_all

# Greedy admission using Jun--Jul full-strength predictions ONLY.
uv run python listaware_ensemble_longrun.py select --max-members 8

# One-way locked Aug--Oct expanding-origin audit after weights frozen.
uv run python listaware_ensemble_longrun.py audit --threads 8

# Fit exactly the frozen members on all completed pre-Jun-2026 sales and
# save a portable inference model (not automatically used by run.py).
uv run python listaware_ensemble_longrun.py finalize --threads 8 --model-name listaware_longrun_8
```

The selection metric is `0.65*MdAPE + 0.35*MAPE`, with a minimum
0.01-percentage-point marginal improvement and a 0.03-pp MAPE regression
safeguard. The first seed is CatBoost MAE. Selection is **greedy**, not a
guarantee of globally optimal ensemble weights. The audit labels do not
feed back into member admission, and a saved audit lock prevents changing
the roster without an explicitly new experiment.

All historical evaluation months have already been consulted in the broader
project. Do not call their scores an untouched or final hidden-test estimate.

### Trained inference artifact

`models.listaware_selected_ensemble.ListAwareSelectedEnsemble` serializes
fitted feature providers, estimators, and member weights together under
`saved_models/list_aware/<name>/`. Transfer the model with its metadata and
dependencies; load **only trusted** joblib files.

The branch includes an unchanged `run.py`, but **`run.py` currently trains
its legacy default models and does NOT load either selected long-run
ensemble automatically**. Integrating those artifacts into a final submission
interface is a separate task; do not confuse this worker's research selection
with production integration.

## Review criteria

- Compare absolute scores only within the **same protocol and homes**.
- Preserve dataset fingerprints and complete checkpoints when resuming.
- Do not enlarge a searched candidate pool after seeing its audit result.
- Check separate list-aware / off-market prediction shapes, positive prices,
  serialized feature providers, and month-by-month stability.
- Submit results/checkpoint archives separately. Do not push data or fitted
  binary model files to this GitHub branch.
