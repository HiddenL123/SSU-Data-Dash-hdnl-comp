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
# Optuna is optional; invoke the factory using uv's --with overlay.
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
uv run --with optuna python ensemble_rc.py run --profile overnight_local --generations 20 --output-dir outputs/offmarket_longrun_20
uv run python ensemble_rc.py status --output-dir outputs/offmarket_longrun_20
# Do NOT open the audit until the intended 20 generations are finished.
uv run --with optuna python ensemble_rc.py audit --profile overnight_local --generations 20 --output-dir outputs/offmarket_longrun_20
uv run --with optuna python ensemble_rc.py benchmark --profile overnight_local --generations 20 --output-dir outputs/offmarket_longrun_20
uv run --with optuna python ensemble_rc.py finalize --profile overnight_local --generations 20 --output-dir outputs/offmarket_longrun_20 --model-name offmarket_longrun_20
```

The existing factory writes strict restartable checkpoints and uses an OS lock.
**Start at 20 generations in a new empty folder**; do not change generation
count or any frozen manifest values on an already-started or audited run.
The benchmark is not an untouched test.

Optional portable checkpoint export:

```bash
uv run python ensemble_rc.py export --output-dir outputs/offmarket_longrun_20 --checkpoint-archive offmarket_checkpoint.zip
```

## B. List-aware: same AutoML factory with three estimator families

**This uses the copied off-market factory architecture** in
`models/listaware_ensemble_automl_factory.py`, not the old handcrafted
`listaware_ensemble_longrun.py` selector (removed from this branch).

It includes the off-market workflow's training-only text vocabulary,
feature bagging (selected/weighted/uniform/full/additional terms), bootstrap
or 80% row subbagging, family-specific Optuna tuning, marginal-gain
ensemble admission, full cached/model manifest snapshots, checkpoint ZIP
export, and a separately locked chronological audit.

Differences from off-market:

- **Target:** `log(sold_price / list_price)`, with sale price reconstructed
  as `list_price * exp(predicted_log_ratio)`.
- **Model families:** CatBoost MAE/Huber, XGBoost, LightGBM with their own
  historical feature recipes and Optuna hyperparameter ranges.
- **Features:** the [curated pool](docs/listaware_curated_pool.md) of
  10 CatBoost / 5 XGB / 5 LGB historically screened feature recipes,
  with independently sampled text vocabulary and row bags.
- **Storage:** `outputs/listaware_ensemble_factory_v1/` and
  `saved_models/list_aware/`, separate from off-market.
- **Protocol:** Apr–May 2025 tuning, Jun–Jul selection, Aug–Oct sealed audit.
  These were previously inspected elsewhere, so audit is **not untouched**.

For a stronger machine, choose a **new empty output folder** and fix
the intended number of generations **before starting**:

```bash
# 20 generations * 3 candidate families/recipes per generation, with
# 5 Optuna trials per surviving candidate and full 700 boosting rounds.
uv run --with optuna python listaware_factory.py run --generations 20 --threads 16 --output-dir outputs/listaware_longrun_20

uv run --with optuna python listaware_factory.py status --output-dir outputs/listaware_longrun_20

# Do not examine audit scores before candidate selection is completely frozen.
uv run --with optuna python listaware_factory.py audit --generations 20 --threads 16 --output-dir outputs/listaware_longrun_20

uv run --with optuna python listaware_factory.py finalize --generations 20 --threads 16 --output-dir outputs/listaware_longrun_20 --model-name listaware_longrun_20
```

The first candidate generation seeds **all three families**. Later proposals
randomly sample family, predeclared recipe, text bag, and row bag. Like
off-market, every candidate must have positive marginal matched-home
ensemble gain to be admitted; mere error-correlation diversity is insufficient.

**Execution requirements:** the original off-market factory's manifest,
resume, and SQLite semantics apply. Use one writer per output folder.
`--threads` can change between machines, but data fingerprint, settings,
candidate seeds, and the Python/ML stack should be consistent. Store a
durable archive and share fitted model bundles separately, never in Git.

**Important:** `run.py` is unchanged and still uses the legacy default
models. The fitted list-aware ensemble is a separate joblib bundle until
a deliberate deployment integration is performed.

## Review criteria

- Compare absolute scores only within the **same protocol and homes**.
- Preserve dataset fingerprints and complete checkpoints when resuming.
- Do not enlarge a searched candidate pool after seeing its audit result.
- Check separate list-aware / off-market prediction shapes, positive prices,
  serialized feature providers, and month-by-month stability.
- Submit results/checkpoint archives separately. Do not push data or fitted
  binary model files to this GitHub branch.
