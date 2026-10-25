# Repository organization

Structural migration from `14527cb` (origin/main).

See [validation results and existing limitations](repository_validation.md).

## Before
```text
./
  *.ipynb (research and entry points)
  *_benchmark.py, ensemble_*.py, save_model.py
  models/ (features, estimators, ensembles, legacy code)
  notebooks/legacy/
  tests/
  docs/
  data.py, run.py
```

## After
```text
./
  experiments/ (research notebooks; legacy/ archives)
  feature/ (feature evaluation and research scripts)
  features/ (reusable implementations and make_features)
  models/ (estimators, residual models, ensemble and storage helpers)
  tests/
  docs/
  model_run.ipynb, ensemble_release.ipynb, notebooks/quickstart.ipynb
  data.py, run.py
  <legacy Python paths> (compatibility aliases)
```

## Moves
| Previous path | Canonical path |
| --- | --- |
| `models/features.py` | `features/features.py` |
| `models/text_features.py` | `features/text_features.py` |
| `models/market_features.py` | `features/market_features.py` |
| `models/spatiotemporal_features.py` | `features/spatiotemporal_features.py` |
| `models/ratio_comparable_features.py` | `features/ratio_comparable_features.py` |
| `models/spacetime_screen_features.py` | `features/spacetime_screen_features.py` |
| `models/spacetime_advanced_features.py` | `features/spacetime_advanced_features.py` |
| `models/realworld_avm_features.py` | `features/realworld_avm_features.py` |
| `models/spacetime_structural_features.py` | `features/spacetime_structural_features.py` |
| `models/legacy/gb_features.py` | `features/gb_features.py` |
| `models/legacy/basic_features.py` | `features/basic_features.py` |
| `models/legacy/ex_features.py` | `features/ex_features.py` |
| `models/legacy/upstream_features.py` | `features/upstream_features.py` |
| `text_benchmark.py` | `feature/text_benchmark.py` |
| `comparable_benchmark.py` | `feature/comparable_benchmark.py` |
| `avm_realworld_shotgun.py` | `feature/avm_realworld_shotgun.py` |
| `spacetime_combination_screen.py` | `feature/spacetime_combination_screen.py` |
| `spacetime_full_feature_comparison.py` | `feature/spacetime_full_feature_comparison.py` |
| `listaware_final_mae_features.py` | `feature/listaware_final_mae_features.py` |
| `save_model.py` | `models/save_model.py` |
| `ensemble_automl_factory.py` | `models/ensemble_automl_factory.py` |
| `ensemble_release_candidate.py` | `models/ensemble_release_candidate.py` |
| `ensemble_feature_bagging_pilot.py` | `models/ensemble_feature_bagging_pilot.py` |
| `ensemble_shrinkage_diagnostic.py` | `models/ensemble_shrinkage_diagnostic.py` |
| `comparable_experiment.ipynb` | `experiments/comparable_experiment.ipynb` |
| `ensemble_starter.ipynb` | `experiments/ensemble_starter.ipynb` |
| `interaction.ipynb` | `experiments/interaction.ipynb` |
| `listaware_fixed_origin_followup.ipynb` | `experiments/listaware_fixed_origin_followup.ipynb` |
| `listaware_shotgun.ipynb` | `experiments/listaware_shotgun.ipynb` |
| `notebook.ipynb` | `experiments/notebook.ipynb` |
| `notebooks/legacy/interaction.ipynb` | `experiments/legacy/interaction.ipynb` |
| `notebooks/legacy/model_run.ipynb` | `experiments/legacy/model_run.ipynb` |
| `notebooks/legacy/notebook.ipynb` | `experiments/legacy/notebook.ipynb` |
| `offmarket_crossfamily_blend.ipynb` | `experiments/offmarket_crossfamily_blend.ipynb` |
| `optuna.ipynb` | `experiments/optuna.ipynb` |
| `realworld_avm_shotgun.ipynb` | `experiments/realworld_avm_shotgun.ipynb` |
| `shap_analysis.ipynb` | `experiments/shap_analysis.ipynb` |
| `spacetime_combination_screen.ipynb` | `experiments/spacetime_combination_screen.ipynb` |
| `spacetime_fast.ipynb` | `experiments/spacetime_fast.ipynb` |
| `spacetime_three_month.ipynb` | `experiments/spacetime_three_month.ipynb` |
| `text_experiment.ipynb` | `experiments/text_experiment.ipynb` |

## Compatibility

Every moved Python module retains its old import path as a module alias, including private attributes and monkeypatches. `models.gb_features`, `models.basic_features`, and `models.ex_features` also forward directly to `features/`. Old pickle/joblib class references resolve through these aliases. Root research CLIs keep their original arguments; canonical execution is `python -m feature.<module>`. Saved-model paths, output defaults, manifests and data remain unchanged.

Feature implementations import only feature code and third-party libraries. Model code consumes features. Feature research helpers may depend on estimators and existing mixed model/feature experiment orchestration.

## Deliberately retained

`model_run.ipynb` remains the training/validation entry point; `ensemble_release.ipynb` remains the release/resume entry point; `notebooks/quickstart.ipynb` remains the onboarding entry point. Mixed model-family and residual experiments (including both list-aware scripts, `spacetime_fast.py`, and `offmarket_crossfamily_blend.py`) remain root CLIs because they orchestrate multiple model families, rather than reusable feature implementations. `ensemble_rc.py` stays the public command dispatcher.

Research notebooks discover the checkout from their working directory or ancestors and change to its root so historical relative output paths stay unchanged. If a kernel starts elsewhere, set `DATA_DASH_ROOT` to the checkout; the two Colab-aware shotgun notebooks also retain `PROJECT_ROOT`. Run cells in order. No notebook needs `__file__`. Outputs and metadata are retained; historical errors in archived outputs are not rerun or rewritten.

## Dependency inventory (before migration)

<details><summary>Python import graph</summary>

```text
avm_realworld_shotgun.py: __future__, argparse, data, hashlib, json, models.spacetime_screen_features, numpy, pandas, pathlib, spacetime_fast, spacetime_three_month
check_submission.py: data, numpy, pandas, pathlib, sys
comparable_benchmark.py: __future__, argparse, data, json, models.boost_regressor, models.direct_listing, models.model, numpy, pandas, pathlib
data.py: huggingface_hub, os, pandas, pathlib
ensemble_automl_factory.py: __future__, data, dataclasses, hashlib, json, models.automl_checkpoint, models.automl_manifest, models.automl_portable, models.automl_search, models.ensemble_bagging, models.ensemble_snapshot, models.features, models.text_features, numpy, optuna, pandas, pathlib, save_model, spacetime_fast, time
ensemble_feature_bagging_pilot.py: __future__, data, hashlib, json, models.ensemble_bagging, models.ensemble_snapshot, models.features, models.text_features, numpy, pandas, pathlib, save_model, spacetime_fast, time
ensemble_rc.py: __future__, argparse, ensemble_automl_factory, ensemble_release_candidate, json, models.automl_manifest, models.automl_portable, models.automl_profiles, pathlib
ensemble_release_candidate.py: __future__, check_submission, data, ensemble_automl_factory, hashlib, json, models.automl_checkpoint, models.automl_search, models.ensemble_bagging, models.ensemble_snapshot, models.features, models.final_ensemble, numpy, pandas, pathlib, save_model, spacetime_fast
ensemble_shrinkage_diagnostic.py: __future__, hashlib, json, numpy, pandas, pathlib
listaware_final_mae_features.py: __future__, argparse, data, dataclasses, hashlib, json, listaware_fixed_origin_followup, listaware_full_fixed_origin_confirmation, listaware_shotgun, models.listaware_shotgun_specs, numpy, pandas, pathlib
listaware_fixed_origin_followup.py: __future__, argparse, data, json, listaware_shotgun, models.listaware_shotgun_specs, numpy, pandas, pathlib
listaware_full_fixed_origin_confirmation.py: __future__, argparse, json, listaware_fixed_origin_followup, numpy, pandas, pathlib
listaware_shotgun.py: __future__, argparse, catboost, data, hashlib, json, lightgbm, models.features, models.listaware_shotgun_specs, numpy, pandas, pathlib, spacetime_fast, spacetime_three_month, time, xgboost
models/__init__.py: model
models/arima.py: numpy, pandas, statsmodels.tsa.arima.model
models/audit_information_cutoff.py: __future__, numpy, pandas, pandas.testing, spatiotemporal_features
models/automl_checkpoint.py: __future__, contextlib, fcntl, json, msvcrt, os, pandas, pathlib, tempfile
models/automl_manifest.py: __future__, data, dataclasses, ensemble_automl_factory, hashlib, json, models.automl_profiles, pandas, pathlib, spacetime_fast
models/automl_portable.py: __future__, hashlib, models.automl_checkpoint, os, pathlib, shutil, sqlite3, tempfile, zipfile
models/automl_profiles.py: __future__, dataclasses
models/automl_search.py: __future__, dataclasses, numpy
models/basic_features.py: models.legacy, models.legacy.basic_features
models/basic_models.py: models.legacy, models.legacy.basic_models
models/boost_regressor.py: components
models/boosting.py: arima, boost_regressor, model, pandas
models/components.py: defaults, importlib
models/defaults.py:
models/direct_listing.py: catboost, components, copy, models.features, models.ratio_comparable_features, numpy
models/dummy_model.py: models.legacy, models.legacy.dummy_model
models/ensemble_bagging.py: __future__, dataclasses, models.text_features, numpy
models/ensemble_blending.py: __future__, numpy, pandas
models/ensemble_snapshot.py: __future__, dataclasses, numpy, pandas, typing
models/ex_features.py: models.legacy, models.legacy.ex_features
models/ex_models.py: models.legacy, models.legacy.ex_models
models/features.py: gb_features, models.market_features, models.ratio_comparable_features, models.spacetime_screen_features, models.spatiotemporal_features, pandas, text_features
models/final_ensemble.py: __future__, numpy, pandas
models/gb_features.py: models.legacy, models.legacy.gb_features
models/gb_models.py: models.boosting, models.legacy, models.legacy.gb_models
models/legacy/__init__.py:
models/legacy/basic_features.py: data, numpy, os, pandas, sklearn.feature_extraction.text, sklearn.preprocessing
models/legacy/basic_models.py: models.basic_features, models.dummy_model, numpy, pandas, pygam, sklearn.ensemble, sklearn.linear_model, sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing, statsmodels.tsa.arima.model
models/legacy/dummy_model.py: numpy, pandas
models/legacy/ex_features.py: pandas
models/legacy/ex_models.py: models.ex_features, numpy, sklearn.ensemble
models/legacy/gb_features.py: numpy, pandas, pandas.api.types, sklearn.feature_extraction.text
models/legacy/gb_models.py: catboost, models.arima, models.boost_regressor, models.defaults, models.dummy_model, models.gb_features, models.listing, numpy, pandas, sklearn.linear_model, sklearn.preprocessing, statsmodels.tsa.arima.model
models/legacy/model_draft.py: models.gb_models, numpy, pandas
models/legacy/upstream_features.py: pandas
models/legacy/upstream_models.py: models.legacy, numpy, sklearn.ensemble
models/listaware_shotgun_specs.py: __future__, dataclasses, models.defaults, spacetime_fast
models/listing.py: gb_features, numpy, pandas, sklearn.linear_model, sklearn.preprocessing
models/market_features.py: __future__, numpy, pandas
models/model.py: arima, boost_regressor, components, copy, defaults, features, listing, numpy
models/ratio_comparable_features.py: __future__, models.spatiotemporal_features, numpy, pandas
models/realworld_avm_features.py: __future__, bisect, models.spatiotemporal_features, numpy, pandas, sklearn.neighbors
models/shap_analysis.py: __future__, numpy, pandas, shap, xgboost
models/spacetime_advanced_features.py: __future__, numpy, pandas, sklearn.neighbors, spatiotemporal_features
models/spacetime_gpboost.py: __future__, gpboost, models.features, models.spacetime_structural_features, numpy, pandas
models/spacetime_screen_features.py: __future__, models.market_features, models.realworld_avm_features, models.spacetime_advanced_features, models.spatiotemporal_features, numpy, pandas, sklearn.kernel_approximation, sklearn.preprocessing
models/spacetime_screen_residuals.py: __future__, numpy, pandas, sklearn.ensemble, sklearn.gaussian_process, sklearn.gaussian_process.kernels, sklearn.kernel_approximation, sklearn.linear_model, sklearn.neighbors, sklearn.pipeline, sklearn.preprocessing
models/spacetime_structural_features.py: __future__, numpy, pandas
models/spacetime_structural_residuals.py: __future__, models.spacetime_structural_features, numpy, scipy.spatial, sklearn.gaussian_process, sklearn.gaussian_process.kernels
models/spatiotemporal_features.py: __future__, numpy, pandas, sklearn.neighbors
models/text_features.py: __future__, legacy.gb_features, numpy, pandas, sklearn.feature_extraction.text
models/upstream.py: legacy.upstream_models
offmarket_crossfamily_blend.py: __future__, argparse, catboost, data, hashlib, json, lightgbm, models.automl_checkpoint, models.features, models.listaware_shotgun_specs, numpy, pandas, pathlib, spacetime_fast, spacetime_three_month, time, xgboost
run.py: check_submission, data, json, matplotlib.pyplot, models.model, numpy, pandas, pathlib, sys
save_model.py: __future__, datetime, importlib, joblib, json, os, pathlib, platform, re, shutil, tempfile, typing
spacetime_combination_screen.py: __future__, data, json, models.features, models.spacetime_screen_residuals, numpy, pandas, pathlib, spacetime_fast, time
spacetime_fast.py: __future__, argparse, data, json, models.defaults, models.features, models.spacetime_screen_features, models.spacetime_screen_residuals, numpy, pandas, pathlib, traceback, xgboost
spacetime_full_feature_comparison.py: __future__, data, hashlib, json, models.model, models.spacetime_screen_features, models.spacetime_screen_residuals, numpy, pandas, pathlib, spacetime_full_validation, spacetime_three_month
spacetime_full_gam_comparison.py: __future__, data, json, models.model, models.spacetime_screen_residuals, numpy, pandas, pathlib, spacetime_full_feature_comparison, spacetime_full_validation, spacetime_three_month
spacetime_full_validation.py: __future__, data, hashlib, json, models.model, models.spacetime_screen_residuals, numpy, pandas, pathlib, spacetime_three_month
spacetime_list_aware_gam.py: __future__, data, json, models.direct_listing, models.spacetime_screen_residuals, numpy, pandas, pathlib, save_model, spacetime_full_feature_comparison, spacetime_full_validation, spacetime_three_month
spacetime_matern_tuning.py: __future__, data, hashlib, json, models.spacetime_screen_residuals, models.spacetime_structural_features, models.spacetime_structural_residuals, numpy, pandas, pathlib, spacetime_combination_screen, spacetime_fast, spacetime_structural_fast, time
spacetime_structural_fast.py: __future__, argparse, data, hashlib, json, models.spacetime_gpboost, models.spacetime_screen_residuals, models.spacetime_structural_features, models.spacetime_structural_residuals, numpy, pandas, pathlib, spacetime_combination_screen, spacetime_fast, time, traceback
spacetime_three_month.py: __future__, argparse, data, json, models.spacetime_screen_residuals, numpy, pandas, pathlib, spacetime_fast
tests/test_automl_interrupt_resume.py: dataclasses, ensemble_automl_factory, json, models.automl_checkpoint, numpy, pandas, pytest, test_starter
tests/test_automl_manifest_compat.py: ensemble_automl_factory, json, models.automl_manifest, pandas, pytest, spacetime_fast
tests/test_comparable_ablation.py: comparable_benchmark, data, joblib, models.boost_regressor, models.direct_listing, models.features, models.model, models.spatiotemporal_features, numpy, pandas, pytest, test_starter
tests/test_crossfamily_followups.py: json, listaware_fixed_origin_followup, numpy, offmarket_crossfamily_blend, pandas, pytest
tests/test_data_prediction_fallback.py: data, pathlib, pytest, sys, types
tests/test_direct_listing.py: catboost, data, models, models.arima, models.direct_listing, models.features, models.ratio_comparable_features, models.spatiotemporal_features, numpy, pandas, pytest, test_modular_models, test_starter
tests/test_ensemble_automl_factory.py: dataclasses, ensemble_automl_factory, json, models.automl_search, numpy, pandas, pytest, test_starter
tests/test_ensemble_release.py: __future__, ensemble_automl_factory, ensemble_rc, ensemble_release_candidate, hashlib, json, models.automl_checkpoint, models.automl_portable, models.automl_profiles, models.automl_search, models.final_ensemble, numpy, pandas, pathlib, pytest, sqlite3, test_starter, zipfile
tests/test_ensemble_shrinkage_diagnostic.py: ensemble_shrinkage_diagnostic, json, numpy, pandas, pytest
tests/test_ensemble_starter.py: ensemble_feature_bagging_pilot, models.ensemble_bagging, models.ensemble_blending, models.text_features, numpy, pandas, pytest, test_starter
tests/test_feature_names.py: models.features, models.legacy.gb_features, pandas, pytest, test_starter
tests/test_gpboost_component_diagnostics.py: models.spacetime_gpboost, numpy, pandas, sys, types
tests/test_listaware_confirmation.py: json, listaware_fixed_origin_followup, listaware_full_fixed_origin_confirmation, pandas, pytest
tests/test_listaware_final_mae_features.py: listaware_final_mae_features, listaware_full_fixed_origin_confirmation, models.features, numpy, pandas, pytest
tests/test_listaware_shotgun.py: __future__, data, listaware_shotgun, models.features, models.listaware_shotgun_specs, numpy, pandas, pytest
tests/test_model_storage_tracks.py: data, ensemble_feature_bagging_pilot, models.ensemble_bagging, models.ensemble_snapshot, models.text_features, numpy, pandas, pytest, save_model, sklearn.dummy, spacetime_fast, test_starter
tests/test_modular_models.py: copy, data, importlib, importlib.util, json, matplotlib, models, models.arima, models.boost_regressor, models.defaults, models.features, models.legacy.basic_features, models.legacy.gb_models, models.model, numpy, pandas, pathlib, pytest, run, test_starter, types
tests/test_modular_routing.py: ast, data, json, models, models.arima, models.boost_regressor, models.gb_models, models.legacy, models.model, numpy, pandas, pathlib, pytest, test_starter, types
tests/test_realworld_avm_shotgun.py: __future__, avm_realworld_shotgun, models.realworld_avm_features, models.spacetime_screen_features, numpy, pandas, pytest
tests/test_shap_analysis.py: data, models.boost_regressor, models.model, models.shap_analysis, numpy, pandas, pytest, test_starter
tests/test_spacetime_advanced_features.py: models.spacetime_advanced_features, models.spacetime_screen_features, numpy, pandas, pytest
tests/test_spacetime_fast.py: data, models.boost_regressor, models.features, models.model, models.spacetime_screen_features, models.spacetime_screen_residuals, numpy, pandas, pytest, spacetime_fast, test_starter
tests/test_spacetime_full_feature_comparison.py: data, models.boost_regressor, models.features, models.model, numpy, pandas, pytest, spacetime_combination_screen, spacetime_fast, spacetime_full_feature_comparison, test_starter
tests/test_spacetime_full_gam_comparison.py: data, models.boost_regressor, models.model, numpy, pandas, pytest, spacetime_full_gam_comparison, test_starter
tests/test_spacetime_full_validation.py: data, models.boost_regressor, models.model, numpy, pandas, pytest, spacetime_full_validation, test_starter
tests/test_spacetime_list_aware_gam.py: data, models.boost_regressor, models.direct_listing, models.features, numpy, pandas, pytest, spacetime_list_aware_gam, test_starter
tests/test_spacetime_matern_tuning.py: models.spacetime_structural_residuals, numpy, pandas, pytest, spacetime_matern_tuning, test_starter
tests/test_spacetime_structural_fast.py: models.spacetime_structural_features, models.spacetime_structural_residuals, numpy, pandas, pytest, spacetime_structural_fast, test_starter
tests/test_spacetime_three_month.py: json, numpy, pandas, pathlib, pytest, spacetime_three_month
tests/test_starter.py: check_submission, data, models.upstream, numpy, pandas, pytest, run
tests/test_text_benchmark.py: data, models.boost_regressor, models.direct_listing, models.features, models.model, models.text_features, numpy, pandas, pytest, test_starter, text_benchmark
text_benchmark.py: __future__, argparse, data, json, models.boost_regressor, models.spatiotemporal_features, models.text_features, numpy, pandas, pathlib
```
</details>
