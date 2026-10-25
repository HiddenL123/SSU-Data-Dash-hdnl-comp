# Modular housing models

`make_model(track, modelspecs=None)` creates a fresh model for `off_market` or
`list_aware`. Both expose `fit(train)`, `predict_log(homes)` and `predict(homes)`.
The last method returns prices; components always predict on the log scale.

```python
from models import make_model, DEFAULT_MODELSPECS
from models.boost_regressor import CatBoostRegressor
from models.arima import ArimaResidualModel

catboost_only = [(CatBoostRegressor, "Xpt", None)]
catboost_arima = [
    (CatBoostRegressor, "Xpt", {"iterations": 900}),
    (ArimaResidualModel, "date_listed", {"order": (1, 1, 0)}),
]
model = make_model("off_market", modelspecs=catboost_arima).fit(train)
prices = model.predict(homes)
list_model = make_model("list_aware", modelspecs=catboost_arima)
```

The default is **900-iteration CatBoost with time features, then ARIMA(1,1,0)**,
matching `gb_models.py` at the time of this refactor. CatBoost depth 6, learning
rate 0.04, L2 regularization 5, RMSE loss and seed 42 are unchanged. ARIMA remains
optional through the specification; its optimization and warnings are unchanged.

Each tuple contains a class, a feature identifier, and constructor kwargs (or
`None`). Components are instantiated with `model_class(**(params or {}))` in
the specified order. Each fits the residual left by preceding components.
Each component and historical fold owns independent preprocessing state.

Available adapters:

| Import | Categorical handling |
| --- | --- |
| `models.boost_regressor.CatBoostRegressor` | String columns passed as `cat_features` |
| `models.boost_regressor.LightGBMRegressor` | Training-fitted pandas categories |
| `models.boost_regressor.XGBoostRegressor` | Training-fitted pandas categories; `enable_categorical=True` |
| `models.arima.ArimaResidualModel` | Monthly median residuals from listing dates |

Boosting adapters merge kwargs with the original library-specific defaults in
`defaults.py`. Native CatBoost, LightGBM and XGBoost estimator classes also work;
those use their own constructor defaults, so pass the full equivalent kwargs
when comparing them with the repository adapters. Native XGBoost requires
`enable_categorical=True`. The fitted native estimator is available as
`component.estimator`; ARIMA diagnostics remain at `component.result.mle_retvals`.
`off_market.components` contains `(component, provider)` pairs.

Feature providers in [`../features/`](../features/) implement `fit_transform(train)` and
`transform(homes)`:

| Name | Inputs |
| --- | --- |
| `Xp` | Existing property, geography, categorical and selected TF-IDF features |
| `Xpt` | Same features plus the existing trend and seasonal columns |
| `date_listed` | Listing-date Series for temporal models |

Here `X` means property variables, `p` means position, and `t` means time.
`Xp` has no engineered time columns; `Xpt` adds `months_since_2021`,
`season_sin`, and `season_cos`. `Xpt_market` adds historical market features
to `Xpt`. The previous names were `Xpt` for the time-independent set and
`Xpt_time` for the time-aware set; active configurations have been migrated
without changing their feature values. Feature identifiers in archived notebook
examples and the legacy draft have also been migrated; historical result filenames
are retained because their names alone cannot establish the inputs used.

The original fixed text vocabulary, fitted IDF weights, column ordering and
`Unknown` category handling are preserved. Off-market providers ignore listing
price and after-sale fields even when those fields are present during training.
Provider state is available at `provider.state` for property features.

To add a component, implement `fit(X, residual)` and `predict(X)` returning one
finite log-scale value per row. Register additional feature factories in
`FEATURE_PROVIDERS` (a factory receives the backend name). Custom components can
declare `feature_backend` for their provider. No container changes are required.

`ListAwareModel(specs, correction_alpha=50.0, n_folds=3)` reuses the original
listing-price discrepancy, property, geography, time and property-type interaction
features. It retains median imputation, StandardScaler, Ridge, the 55–80% listing
date cutoffs and minimum 100 earlier sales. Historical base fits use only sales
completed before the prediction window; the final base fits all training rows.

## CLI and notebooks

Existing commands remain unchanged:

```powershell
python run.py validate
python run.py report
python run.py predict
python run.py check
```

Full real-data validation is optional and can take time. The original
`run.validate()` and `run.predict()` take no arguments; they call
`models.make_model(track)`, which automatically supplies `DEFAULT_MODELSPECS`.
Validation writes the usual CSV and returns `None`, exactly as upstream.
For custom specifications use `make_model(track, modelspecs=...)` or the
`validate(modelspecs=...)` helper in `model_run.ipynb`. Reporting still uses
the same CSV, metrics and output paths. `model_run.ipynb` uses the modular factory
and retains fitted models separately; its reporting function also accepts the
list of DataFrames produced by older notebook runs.

Both `model_run.ipynb` and `experiments/interaction.ipynb` define editable, explicit
`MODEL_SPECS` lists rather than referencing `DEFAULT_MODELSPECS`. Their initial
values exactly match the legacy boosting pipeline: 900-iteration CatBoost with
the original parameters and time-aware features, then ARIMA(1,1,0).
The notebook validation helper requires specifications to be passed explicitly.
`experiments/notebook.ipynb` only analyzes the resulting CSV and does not fit models.

Previous implementations are in `legacy/`. `models.gb_models.make_model`,
`GBOffMarketModel` and `GBListAwareModel` now route through modular containers
using the constructor adapters in `boosting.py`. The existing `algorithm`,
`model_params`, `use_remarks`, `max_text_features`, `arima_order`, and list-aware
correction options are supported. For an explicit old implementation, import
`models.legacy.gb_models`. Other experimental families such as
`models.basic_models` retain their legacy semantics. ARIMA, boosting defaults
and the listing adjustment are shared rather
than duplicated. The unfinished original modular draft is preserved as
`legacy/model_draft.py` for reference; it was not a working implementation.
Basic/GAM models retain their old API rather than becoming new components here.

Install modular dependencies with `python -m pip install -r requirements-models.txt`.
This includes the unchanged upstream `requirements.txt` and the additional model libraries.
XGBoost 2.1 or later is needed with NumPy 2 (2.0.3 fails on categorical input).
The existing fixed TF-IDF vocabulary is retained; the legacy `max_text_features`
argument does not change that fixed vocabulary.

## Current project routing

`run.py` is kept as close as possible to
[erwinzhang7/SSU-Data-Dash](https://github.com/erwinzhang7/SSU-Data-Dash),
with only its model import changed. `check_submission.py` must remain an exact,
byte-for-byte upstream copy, with no import exceptions. The root `README.md`,
`rubric.md`, and `handout.md` are also do-not-touch files and match upstream
exactly. This `models/README.md` is separate implementation documentation.
The original files were restored from commit
`6c5831e12f523d617eccd1418d46a5cf2a0bb8b4`. The original baseline model and feature
implementations now live in `models/legacy/upstream_models.py` and
`features/upstream_features.py`, avoiding a dependency on removed root files.
The routing imports are: `run.py` uses
`import models.model as models`; the starter tests use
`import models.upstream as models` to explicitly access the preserved upstream
baselines. At the user's request,
`data.py` now retains the newer local-file-first implementation. Other files
may evolve; the protected exceptions are `run.py`, `check_submission.py`,
the root `README.md`, `rubric.md`, and `handout.md`.
Pre-restoration copies are under
`outputs/pre_upstream_restore/`.

The data loader automatically checks the project's `data/` folder before
downloading (local Parquet first, then CSV). `DATA_DIR` is optional and overrides
that folder when supplied:

```powershell
$env:DATA_DIR = (Resolve-Path ./data).Path
python run.py validate
```

| Entry point | Current implementation |
| --- | --- |
| `run.py` and `notebooks/quickstart.ipynb` | Original workflows; `run.py` imports `models.model` |
| `model_run.ipynb` | Modular validation, then writes `outputs/validation_predictions.csv` |
| `experiments/notebook.ipynb` | Reads that same current validation CSV throughout |
| `experiments/interaction.ipynb` | Fits configurable modular baselines for out-of-time residual discovery |
| `models.gb_models` factory and complete predictor classes | Modular containers through `models.boosting` |

The three root notebooks were archived before the routing update under
`experiments/legacy/`. Their current versions have cleared stored outputs so old
results cannot be mistaken for results from the new baseline. Run the training
notebook (or the existing validation CLI) before the analysis notebook.

Interaction discovery now uses the full specified modular pipeline as its
baseline, without accessing the old `_prepare` or `interaction_model` internals.
Candidate interaction evaluation remains a separate research experiment.
Its final text diagnostic reports the fitted booster's feature importance,
using the provider's token mapping, rather than fitting the old ElasticNet
selector. These new discovery results should not be interpreted as a rerun of
the archived HGB experiment. The diagnostic expects a boosting component first
in `MODEL_SPECS`, as in the defaults.

Run focused tests with:

```powershell
python -m pytest tests/test_modular_models.py -q -p no:cacheprovider
```

The tests use small datasets and short boosting runs; they check constructor
forwarding, residual order, preprocessing state, unseen categories, chronological
fold separation, CLI outputs and legacy prediction equivalence. No hyperparameter
search is performed.

## Verification notes

An independent comparison executed the committed pre-refactor `gb_models.py`
against the modular implementation using unchanged default hyperparameters.
Training used 1,500 real-data rows sampled with seed 42 from sales before
2026-04-01; prediction used 40 homes listed on or after that date. Both off-market
and list-aware predictions matched exactly: maximum absolute difference was
zero on both the log and dollar scales. This was a sampled parity check, not a
full real-data validation or a performance claim.

The modular tests cover the notebook validation API, all three boosting
libraries, both tracks and legacy file-path imports. The restored starter suite
tests the original baselines and the original CLI workflow. Custom additions to
the old starter test file were removed when that file was restored; its
pre-restoration version is retained in the backup directory above.

The local XGBoost installation was updated from 2.0.3 to 2.1.4 to run categorical
tests with NumPy 2. See the upstream
[2.1 release notes](https://xgboost.readthedocs.io/en/latest/changes/v2.1.0.html).
Tests use Matplotlib's `Agg` backend to avoid the local Tk installation problem.
ARIMA convergence warnings still occur in equivalent old and new fits, as
expected; this refactor does not change their treatment.

The subsequent routing update passes all 31 tests across
`test_modular_models.py` and `test_modular_routing.py`. The extra checks cover
all three redirected boosting libraries on both tracks (including constructor
options), notebook residual generation and the shared CSV path. The later
upstream restoration was verified byte-for-byte after reversing only the two
documented import substitutions.

After restoration, the full suite passes: **49 tests passed**. ARIMA warnings
remain unchanged.
