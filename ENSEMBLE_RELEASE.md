# Ensemble release candidate — two execution pathways

The next experiment is **full-fidelity off-market** rather than the old
220-tree / 5,000-home pilot. This is an engineering release candidate, **not
an independently verified sub-7% MdAPE model**.

The **same** full-history/700-tree settings, random candidate seeds,
chronological cohorts and admission logic apply under either profile.
Profiles adjust CPU resources only. The list-aware competition track and
its legacy saved models are unchanged.

| | Daytime cloud | Overnight local |
| --- | --- | --- |
| Command profile | `day_cloud` | `overnight_local` |
| Where computation runs | Colab VM, **not your laptop** | Your laptop |
| XGBoost threads | 2 (adjust after benchmarking) | 8 |
| Trees per model | **700** | **700** |
| Historical sales per fold | **Full eligible history** | **Full eligible history** |
| Validation properties per month | **All available** | **All available** |
| Checkpoints | VM local disk + automatic portable backup to Drive | Persistent local disk; manual or optional portable ZIP |
| GPU | Not required; CPU `hist` initially | CPU `hist` initially |
| Free-runtime risks | VM reset, limited quota, disconnections | Laptop power/thermal and available disk |

An optional third profile `day_local_quiet` uses **2 local CPU threads**
for low-interference daytime training if Colab is unavailable. You can
switch profiles when **resuming the same checkpoint**; changing the
actual research experiment settings is not allowed.

## 1. Shared release-candidate specification

`models/automl_profiles.py` defines the frozen initial protocol:

- Tune in **April–May 2025**.
- Select ensemble members and fixed weights in **June–July 2025**.
- Open one-time shadow audit in **August–October 2025**, *only after*
  the desired generations finish.
- Use **700 trees**, **all sales before each month**, and **all validation
  homes**. Exact training row bootstraps and vocabulary masks are constructed
  on the eligible historical fold only.
- Initial search **2 generations × 3 candidates × 5 Optuna trials**,
  max 8 accepted members.
- Objective `0.65 * MdAPE + 0.35 * MAPE`, and strict per-candidate MAPE
  non-regression guard.

These 2025 months have been examined in prior experiments, so the audit is
shielded from THIS run's adaptation, **not a previously untouched holdout**.
Do not compare the new audit score with the earlier 7.4857% six-month result
as if they involved identical months/training protocols.

First verify the CLI and recovery tests locally:

```powershell
git pull origin main
pip install optuna
python -m pytest tests/test_ensemble_release.py tests/test_automl_interrupt_resume.py -q
python ensemble_rc.py status
```

## 2. Overnight local — full strength

Open PowerShell in the repository and run:

```powershell
python ensemble_rc.py run --profile overnight_local
```

Keep your laptop plugged in and disable automatic sleep if you want a long
uninterrupted overnight session. The OS can interrupt the factory and
its existing checkpoints will still be useful. To stop, use Ctrl+C;
**rerun the exact same command** to continue. Full-history training on
150,000 sale records can be compute/memory intensive. If heat or RAM is
an issue, run the same experiment in low-impact mode with:

```powershell
python ensemble_rc.py run --profile day_local_quiet
```

Both modes reuse `outputs/ensemble_release_candidate_full/`; no new
model-selection experiment is created. Profile change is a resource change,
not a hyperparameter or validation-split change.

## 3. Daytime Colab — leave your laptop free

Free Colab does **not** guarantee unlimited execution or background runs.
Do not rely on an unattended, indefinitely executing free Colab tab. The
default factory uses CPU histogram XGBoost: choosing a GPU runtime without
enabling GPU tree training does not help.

In a fresh Colab notebook:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then obtain the project checkout (Git clone if public/authorized; otherwise
upload a project archive), and put the housing data where `data.load`
can find the identical data as on your laptop. In a shell cell after
`%cd /content/SSU-Data-Dash-hdnl-comp`:

```bash
!pip install -r requirements.txt optuna
!python ensemble_rc.py run --profile day_cloud --checkpoint-archive /content/drive/MyDrive/automl/release_checkpoint.zip
```

The **live** Optuna SQLite database and training caches stay on the Colab
VM's fast local disk. After each saved validation/tuning fold, trial or
candidate decision, the factory exports a portable archive to Drive using
a consistent SQLite backup. The checkpoint ZIP and SHA-256 file live in
Drive; the VM may disappear at any time. A run interrupted during an
XGBoost `fit()` repeats only that unfinished fit. Checkpoints can fail if
Drive is out of quota or inaccessible; resolve the error before resuming.

### Handoff from Colab to overnight local

Make sure the Colab factory has **stopped** before transferring the archive.
Download `release_checkpoint.zip` and its `.sha256` companion from Drive,
then from your local repository:

```powershell
python ensemble_rc.py restore --checkpoint-archive D:/automl/release_checkpoint.zip
python ensemble_rc.py run --profile overnight_local
```

**Restore refuses a nonempty output directory.** Never merge two active
Optuna/roster folders or run local and Colab workers against the same
checkpoint simultaneously. If the local folder contains older work, move
it aside (rather than deleting blindly) before restoring the newer archive.
Both environments must use compatible package versions and the exact same
housing listings. The manifest hash checks for dataset/config drift.
The portable archive is the handoff mechanism, not a concurrent distributed
training service.

### Handoff from local to Colab

Stop the local factory and export:

```powershell
python ensemble_rc.py export --checkpoint-archive D:/automl/release_checkpoint.zip
```

Upload the archive (+ SHA-256) to Drive. In a new Colab VM use a **fresh,
empty project output folder**, then restore into the local VM disk:

```bash
!python ensemble_rc.py restore --checkpoint-archive /content/drive/MyDrive/automl/release_checkpoint.zip
!python ensemble_rc.py run --profile day_cloud --checkpoint-archive /content/drive/MyDrive/automl/release_checkpoint.zip
```

A restore to a new runtime is mandatory; do NOT point SQLite directly at
Google Drive, as mounted-Drive I/O and SQLite locks are unreliable.

## Troubleshooting: restored Colab checkpoint rejected on Windows

If `audit` or `benchmark` fails with
`Factory settings/data changed: use a NEW output directory`, **do not
delete the experiment folder**. The restored Optuna trials/roster may be
perfectly valid, but the local program computes a different manifest.

From the repo root in Windows PowerShell, run (using native Python 3.13):

```powershell
py -3.13 ensemble_rc.py diagnose
```

This **read-only** command prints every mismatched field, the local
housing data file selected by `data.load("listings")`, row count and
pandas version. It does not run or delete anything.

- **Only `settings.*` differs:** the checkpoint was created with
  slightly different release settings. Reuse the exact saved settings:
  `py -3.13 ensemble_rc.py diagnose --use-checkpoint-settings`.
  Once that reports no differences, use
  `py -3.13 ensemble_rc.py audit --profile overnight_local --use-checkpoint-settings`
  and then the same flag with `benchmark`. Do not copy tuning parameters
  into a new run or replace the manifest.
- **`data_sha256` differs:** the local loaded listings differ from Colab,
  including possible CSV-vs-Parquet dtype/index differences, row order, or
  pandas version. The historical cohorts must be identical to reuse
  validation caches. Compare `data.find_file("listings")` and
  `pd.__version__` in both environments. Use the exact Colab
  `listings.parquet` under a separate folder referenced through
  `DATA_DIR`, if needed, then rerun `diagnose`. Never suppress this
  protection; aligning only the row count is insufficient.
- **`xgb_parent.*` differs:** the baseline hyperparameters in
  `spacetime_fast.py` differ from the checkpoint. Restore the code
  revision that generated the checkpoint, or inspect exactly what changed
  before replaying predictions. Do not edit the frozen manifest.
- **`settings.output_dir` differs:** the saved experiment used a
  different relative output-path string. To reuse it exactly, give the
  original path when invoking the CLI; avoid modifying the checkpoint.

`--use-checkpoint-settings` is explicit and does **not** bypass the data
hash or baseline validation. The frozen checkpoint selection/audit protocol
remains authoritative; no new candidate generation is triggered by
`diagnose`.

### Prediction with a listings-only DATA_DIR override

For reproducible checkpoint matching, Windows may have `DATA_DIR` set to
`data/colab_parquet` (containing only `listings.parquet`). Prediction
inputs are different files: `test_off_market.parquet` or
`test_off_market.csv`.

The data loader now resolves prediction-only test files in this order:

1. The test file in `DATA_DIR`, if present.
2. The test file in the project's normal `data/` directory.
3. The competition dataset on Hugging Face (cached after download).

The *historical listings file* remains strictly loaded from `DATA_DIR`.
The experiment fingerprint is never bypassed. In a PowerShell terminal with
the original `DATA_DIR` still set, an already-finalized model can predict
without retraining:

```powershell
py -3.13 ensemble_rc.py predict --submission-file submissions/off_market_v1.csv
```

This uses the saved fitted ensemble and the prediction test file only;
no call to Optuna, audit, benchmark, or final refit is required. If
`off_market_v1.csv` already exists, choose another filename; the CLI
intentionally refuses overwriting existing submission candidates.

## 4. Freeze, audit, refit and export

Once generation search is finished, **stop all other workers** and run:

```powershell
python ensemble_rc.py audit --profile overnight_local
```

The one-time audit seals the roster, so don't open it while intending to
increase generations. Inspect `summary.json`, monthly audit metrics and
member weights.

For a **same-period comparison** with our earlier six-month benchmark, run
an optional **frozen-weight** evaluation of December 2025–May 2026. It
does not change model membership or weights, but these periods have already
been examined and therefore remain development evidence rather than a
fresh hidden holdout:

```powershell
python ensemble_rc.py benchmark --profile overnight_local
```

This writes `benchmark_full_6month_summary.json` and a cached matching
prediction table under the release experiment directory. Evaluate the
full-history audit and historical benchmark before finalizing. If the
frozen ensemble passes the predeclared MdAPE/MAPE/P90 non-regression gate
against its first reference, then:

```powershell
python ensemble_rc.py finalize --profile overnight_local
python ensemble_rc.py predict
```

`finalize` refits each accepted member on **all available labelled sales**
with 700 trees, preserving its fitted text vocabulary, spatial feature
provider and bootstrap seed. It saves each member separately under
`saved_models/off_market/` (recoverable after interruptions), and saves
the weighted `off_market_release_candidate_full` object. A prior
list-aware model remains untouched. `predict` writes
`submissions/submission_off_market_rc.csv` and refuses overwriting an
existing candidate file.

**No silent shipping:** if the audit underperforms the reference, finalization
refuses to mark it an approved winner. An explicit
`--allow-audit-regression` override may save an *experimental fallback*
but does not make it audit-approved. The model must still pass competition
submission validation before any manual replacement of the official
submission file.

## 5. Operational limits

- Colab free-tier runtimes are ephemeral and dynamically quota-limited.
  Google Drive I/O can also fail. Use a portable archive, NOT a database
  operating live on mounted Drive.
- The archive contains Optuna SQLite, term/row manifests, per-home
  predictions, roster and logs; it does **not** contain the original
  150k-property dataset or saved member joblib binaries. Restore the same
  input dataset separately; regenerate missing model binaries if needed.
- CPU thread settings can vary between daytime/overnight, but model
  predictions could differ slightly across different package/CPU versions.
  Confirm the final ensemble on the single frozen local refit.
- Factory candidates are XGBoost-only for now; Tensor GAM, spatial Matérn,
  CatBoost and LightGBM remain future member families, not falsely
  included in this initial release candidate.
