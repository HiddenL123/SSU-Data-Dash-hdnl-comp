# Repository migration validation

Validated on Windows with Python 3.13.11 and an isolated `uv sync --frozen`
environment. Base commit: `14527cb`. No production training or historical
experiment reruns were started; existing tests use small synthetic fits.

## Test results

| Check | Result |
| --- | --- |
| Existing suite before edits (`python -m pytest -q`) | 299 passed, 9 failed, 2 skipped |
| Full suite after migration, including the initial 86 migration checks | 385 passed, the same 9 failed, 2 skipped |
| Final targeted migration suite (`python -m pytest tests/test_repository_organization.py -q`) | 90 passed |
| Pre-refactor fitted-provider fixtures | 24 cases: exact training/prediction schema, dtype, column order and numerical equality |
| Original and canonical CLI help | All 12 original argparse entry points match captured help; moved CLIs also match when run with `python -m comparisons.<module>` |
| Notebook validation | All 20 notebooks validate and local import modules/attributes resolve; no notebook training cells rerun |
| Notebook preservation | Cell counts/order, execution counts, outputs, IDs and metadata match pre-migration hashes |
| Import order | `features` imports without loading `models`; subsequent legacy imports resolve to the same objects |
| Notebook setup | Root, experiments, nested legacy directories, explicit external-kernel root, and local execution of Colab-aware setup checked |
| Implementation audit | All 24 moved Python modules have identical non-import ASTs, excluding documentation and the required storage-root adjustment |
| Protected files | Saved model binaries/metadata, checkpoint archive, tuning artifacts, `data.py`, `pyproject.toml` and `uv.lock` unchanged |

The full suite includes synthetic checks of ensemble factory selection/resume,
release finalization/storage, list-aware predictions, feature ablations and
chronological validation. A Windows lock test and several existing assertions
remain failing, as listed below. Linux was not available for execution; notebook
root discovery uses platform-neutral `pathlib` and does not use `__file__`.

## Existing serialized models

Loaded each committed `saved_models/*/model.joblib` before and after migration,
then predicted the same three synthetic homes with each nested fitted model.
Seven snapshots (84 fitted models total) produce exactly equal predictions:

- `CatBoost_Optuna_v1`
- `LightGBM`
- `LightGBM_Optuna_v1`
- `LightGBM_text_binary`
- `XGBoost`
- `XGBoost_Optuna_v1`
- `XGBoost_text_binary`

`catboost_spatiotemporal_900` fails before and after migration with
`ModuleNotFoundError: No module named 'models.catboost'`. This missing historical
adapter predates this refactor. Its binary is untouched; no guessed replacement
class was introduced for potentially incompatible fitted state. Some snapshots
also emit their existing scikit-learn 1.8.0 versus locked 1.7.2 version warnings.

Independent synthetic fixtures were serialized before moving any code. They
exercise old `models.features`, `models.legacy.gb_features`, spatial, comparable
and text references through compatibility aliases. Fresh fitted providers
also survive joblib round trips with exact feature output equality.

## Baseline failures retained

These exact nine test IDs fail both before and after the refactor. Fixing their
behavior or experiment expectations is outside this structural change.

| Test | Existing failure |
| --- | --- |
| `test_automl_interrupt_resume.py::test_single_writer_lock_can_be_reacquired` | Windows lock read raises `PermissionError` rather than the expected `RuntimeError` |
| `test_ensemble_automl_factory.py::test_factory_resume_and_audit_after_frozen_selection` | Expected exception-message regex is stale |
| `test_modular_routing.py::test_training_notebook_publishes_output_for_analysis_notebook` | Test execution namespace lacks `display` |
| `test_modular_routing.py::test_notebook_specs_exactly_match_legacy_configuration[model_run.ipynb-2]` | Test execution namespace lacks `XGB_MODEL_SPECS` |
| `test_spacetime_full_feature_comparison.py::test_full_history_compares_independently_and_resumes` | Test expects equal cohort sizes across different months |
| `test_spacetime_full_feature_comparison.py::test_full_history_rejects_wrong_baseline_and_stale_cache` | Existing cached-output settings/baseline mismatch |
| `test_spacetime_full_gam_comparison.py::test_two_full_spline_gam_combinations_and_resume` | Test expects equal cohort sizes across different months |
| `test_spacetime_full_validation.py::test_expand_shortlist_runs_and_resumes_without_rerunning` | Test expects equal cohort sizes across different months |
| `test_spacetime_list_aware_gam.py::test_full_list_aware_experiment_and_resume` | Test expects equal cohort sizes across different months |

The initial system Python test attempt could not collect tests because its
Matplotlib installation was incomplete. Both reported baseline and post-migration
results above use the same clean locked environment instead. One existing
notebook lacks a cell ID; validation warns, and its metadata remains preserved.
