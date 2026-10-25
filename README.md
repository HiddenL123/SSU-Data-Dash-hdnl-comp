# SSU Data Dash: predict Toronto home prices

Start with the case handout, [`handout.md`](handout.md), and the judging rubric, [`rubric.md`](rubric.md).

Two tracks:
- **Off-market:** value a home with no list price.
- **List-aware:** predict a listed home's sale price, knowing its list price.

Train on 150,000 sales (January 2021 to May 2026). Predict homes that sold June to September 2026.

**Data:** [huggingface.co/datasets/Baliyang/SSU_data_dash](https://huggingface.co/datasets/Baliyang/SSU_data_dash).
All of it is synthetic.

## Setup

Python 3.10 or newer (3.11+ on macOS 27):

```
pip install -r requirements.txt
```

The data downloads itself on first use. No Python? Use the
[Colab notebook](https://colab.research.google.com/github/HiddenL123/SSU-Data-Dash-hdnl-comp/blob/main/notebooks/quickstart.ipynb).
Prefer R? See `r/baseline.R`.

## Commands

```
python run.py validate   # test on Dec 2025 to May 2026
python run.py report     # results table and a figure
python run.py predict    # write both submission files
python run.py check      # check them
```

On Windows, use `py` if `python` is not found.

## Repository layout

- [`features/`](features/): reusable feature providers, encoders, comparable sales, spatial, temporal and text transformations. Start with `from features import make_features`.
- [`models/`](models/): estimators, residual models, ensemble implementations and model storage (`models.save_model`).
- [`feature/`](feature/): feature research helpers. For example, `python -m comparisons.text_benchmark --help`.
- [`experiments/`](experiments/): research notebooks, including historical notebooks in `experiments/legacy/`.
- [`model_run.ipynb`](model_run.ipynb): training and validation entry point; [`ensemble_release.ipynb`](ensemble_release.ipynb): release/resume workflow.
- `data.py` and `run.py`: data access and the existing CLI.

Legacy Python imports and root CLI commands remain supported. Notebook setup
locates the checkout from the kernel's working directory or its ancestors and
keeps output paths relative to the repository root. For kernels started elsewhere,
set `DATA_DASH_ROOT` to the checkout. In the release notebook, select the local
execution profile and checkpoint archive path when running outside Colab.

See the [migration inventory and compatibility notes](docs/repository_organization.md)
for before/after trees, all moved files, retained aliases, and validation results.

## Follow-up: fixed-origin and cross-family blending

Two **separate, non-neural historical experiments** extend the existing
showdowns without changing either frozen release model:

- [List-aware fixed-origin and MAE blend notebook](experiments/listaware_fixed_origin_followup.ipynb)
  (`listaware_fixed_origin_followup.py`): train the six shortlisted
  CatBoost/XGBoost/LightGBM models **once** before August 2025, predict
  August–October without retraining, and compare the 50/35/15 winner with
  predeclared CatBoost-MAE blends. Existing full-strength expanding predictions
  can be analyzed with **zero retraining** using
  `--from-predictions outputs/listaware_full_shortlist/predictions.csv`.
- [Off-market frozen ensemble cross-family notebook](experiments/offmarket_crossfamily_blend.ipynb)
  (`offmarket_crossfamily_blend.py`): train tuned standalone XGBoost,
  CatBoost and LightGBM on the **same earlier sales**, then compare cross-family
  mixtures against both standalone XGBoost and the unchanged frozen ensemble.
  Existing CatBoost/LightGBM pilot predictions can be safely reused.

To run tests:

```powershell
py -3.13 -m pytest tests/test_crossfamily_followups.py -q
```

Pilot commands:

```powershell
py -3.13 listaware_fixed_origin_followup.py --pilot
py -3.13 offmarket_crossfamily_blend.py
```

Full historical runs are explicitly opt-in via `--full`. These experiments
use already-inspected 2025 development/audit months and do not constitute new
holdout tests. See
[follow-up methodology](docs/crossfamily_followups.md) for matched-cohort
requirements and result interpretation.

## Experiment notebooks

Two notebooks provide configurable, **non-neural** chronological research runs
using the existing project scripts, including cached predictions and matched
MdAPE comparisons:

- [Off-market AVM feature shotgun](experiments/realworld_avm_shotgun.ipynb):
  hedonic-adjusted comparables, out-of-time local residuals, robust comparable
  statistics and multiscale historical geographic features, compared with the
  frozen ensemble audit on identical homes.
- [List-aware CatBoost–XGBoost showdown](experiments/listaware_shotgun.ipynb):
  the CatBoost Huber baseline, tuned XGBoost/LightGBM, transplanted spatial
  features and predeclared cross-family blends.

Pilot screens are enabled; full-history training must be explicitly opted into.
The notebooks work from a checkout in VS Code/Jupyter or Google Colab. The
frozen off-market audit CSV is stored under `outputs/` (ignored by Git) and
must be available for the off-market comparison notebook.

## Validation

`validate` predicts each month from the sales before it, as in the real test. A random split lets
the model see the future and makes your score look better than it is. Judges compare your
validation error with your hidden-test error.

## Submission

One PDF report (template: `report/report.tex`, 4 pages max) that links to:
1. your video on YouTube, 5 minutes max (unlisted is fine, not private);
2. your GitHub repository (public, or shared with the organisers), holding your code and two files:

```
submissions/submission_off_market.csv    one row per home in test_off_market
submissions/submission_list_aware.csv    one row per home in test_list_aware

ml_num,predicted_price
SYN-0123456,815000
```

`predict` writes both; run `check` before you hand in. Deadline: see the case email.

## Rules

- Outside data only if it covers May 2026 or earlier. List every source with its URL.
- Neighbourhoods, coordinates and streets are synthetic, so map data will not line up.
- No real listing or sales records.
- Do not try to recover the hidden test prices.


## Fitted model snapshots

New reusable model artifacts are separated by prediction task:

- `saved_models/off_market/<run_name>/`: off-market XGBoost and feature-bagged ensembles
- `saved_models/list_aware/<run_name>/`: list-aware models and ensembles
- Existing `saved_models/<legacy_name>/`: **legacy** mixed-track snapshots; unchanged and still readable

Use `save_model(model, name, track="off_market", metadata=...)` or
`load_model(name, track="list_aware")` in new workflows. Existing legacy
`load_model(name)` calls continue working. See [saved_models/README.md](saved_models/README.md)
for the catalog API and optional migration of previously fitted monthly
validation models without retraining.
