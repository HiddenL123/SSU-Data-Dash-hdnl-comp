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
[Colab notebook](https://colab.research.google.com/github/erwinzhang7/SSU-Data-Dash/blob/main/notebooks/quickstart.ipynb).
Prefer R? See `r/baseline.R`.

## Commands

```
python run.py validate   # test on Dec 2025 to May 2026
python run.py report     # results table and a figure
python run.py predict    # write both submission files
python run.py check      # check them
```

On Windows, use `py` if `python` is not found.

## What to edit

- **`features.py`**: the features.
- **`models.py`**: the models. Two simple baselines to beat: list price (list-aware) and the
  10 nearest earlier sales (off-market). `FeatureModel` (gradient-boosted trees) is ready to switch on.

## Experiment notebooks

Two notebooks provide configurable, **non-neural** chronological research runs
using the existing project scripts, including cached predictions and matched
MdAPE comparisons:

- [Off-market AVM feature shotgun](realworld_avm_shotgun.ipynb):
  hedonic-adjusted comparables, out-of-time local residuals, robust comparable
  statistics and multiscale historical geographic features, compared with the
  frozen ensemble audit on identical homes.
- [List-aware CatBoost–XGBoost showdown](listaware_shotgun.ipynb):
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
