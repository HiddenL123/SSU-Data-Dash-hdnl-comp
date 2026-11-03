# Longer ensemble training — dedicated training branch

This is a research/training branch. The existing `release/runpy-clean` submission
runner and both saved production models remain unchanged.

## Train both tracks for longer

From your Windows project root:

```powershell
git fetch origin
git switch train/dual-track-longrun
git pull origin train/dual-track-longrun
uv run --with optuna python train_longrun.py run --track both --generations 12 --optuna-trials 8 --threads 8
```

The **run command now automatically audits and exports each completed track**.
No extra audit or export command is needed for the normal handoff. After
successful training, it writes four files for transfer:

```text
colab_checkpoints/dual_track_longrun_v2/off_market_checkpoint.zip
colab_checkpoints/dual_track_longrun_v2/off_market_checkpoint.zip.sha256
colab_checkpoints/dual_track_longrun_v2/list_aware_checkpoint.zip
colab_checkpoints/dual_track_longrun_v2/list_aware_checkpoint.zip.sha256
```

Send all four files. They preserve selections, Optuna studies, metrics,
and audit results, but **do not contain raw listings or finalized model.joblib**.
The full-history refit can be performed later, on a machine with the matching
historical dataset.

The default per-track search budget is **12 generations × 3 candidates**,
up to **8 Optuna trials per screened-in candidate**, **700 trees**, and at
most **8 accepted ensemble members**. This is a much larger search than the
earlier 2 generations × 5 trials. It may use substantial CPU time. Tracks
run sequentially in separate processes to free memory.

For a single track, replace `both` with `list_aware` or `off_market`.

## Resume, extend and inspect

```powershell
uv run python train_longrun.py status --track both
```

Each run writes independently to:

```text
outputs/dual_track_longrun_v2/off_market/
outputs/dual_track_longrun_v2/list_aware/
```

Atomic roster and progress files, cached fold predictions, and persistent
Optuna SQLite studies allow interrupted jobs to resume. Run the *same command*
to resume; already completed candidate decisions are not repeated.

Since a successful `run` now automatically opens the shadow audit, its
selection becomes locked. **Decide your generation budget in advance.**
To keep the option to extend the run, add `--no-auto-audit`:

```powershell
uv run --with optuna python train_longrun.py run --track list_aware --generations 12 --optuna-trials 8 --threads 8 --no-auto-audit
uv run --with optuna python train_longrun.py run --track list_aware --generations 20 --optuna-trials 8 --threads 8 --no-auto-audit
```

The selection-only command still exports its checkpoint ZIP. Once satisfied,
run the explicit `audit` command to lock selection and re-export. The default
command without `--no-auto-audit` is easiest for a friend operating the computer.

Keep `--optuna-trials`, `--candidates-per-generation`, `--trees`,
`--seed`, the output root, and `DATA_DIR` unchanged across resumes.
The factory checks the manifest and rejects mismatches. The number of CPU
threads may change. The audit locks ensemble selection: never extend
an already audited experiment such as `outputs/listaware_ensemble_release_full`.

To pin your known-matching local Parquet source (if applicable):

```powershell
$env:DATA_DIR = (Resolve-Path '.\data\colab_parquet').Path
```

The final ZIP export always runs once after each successful search/audit, under
`colab_checkpoints/dual_track_longrun_v2` (customizable with `--export-dir`).
Optional **additional** ZIP backups after every intermediate checkpoint can be
enabled with `--checkpoint-dir colab_checkpoints/intermediate_backups`.
Repeated exports increase disk I/O. Keep all archive directories outside
`outputs/dual_track_longrun_v2`.

## Audit (automatic) and finalization (separate)

The normal `run` command already completed the audit and exported both ZIPs.
Only if you used `--no-auto-audit` do you need an explicit audit:

```powershell
uv run --with optuna python train_longrun.py audit --track both --generations 12 --optuna-trials 8
```

Final fitting is **not** automatic; it can be done after transferring the
checkpoints and restoring them with their matching data. To finalize locally
on the same computer:

```powershell
uv run --with optuna python train_longrun.py finalize --track both --generations 12 --optuna-trials 8
```

For a search extended to 20 generations, use `--generations 20` above.

This final fitting step produces separately named artifacts:

```text
saved_models/off_market/off_market_longrun_v2/
saved_models/list_aware/list_aware_longrun_v2/
```

It does NOT overwrite the existing finalist models. The off-market
finalizer preserves its non-regression audit gate. The production runner will
not switch to these new models unless you explicitly configure its model-name
environment variables.

The CLI has been syntax-checked and code-reviewed, but a full training
run has not been executed here.
