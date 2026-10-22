# Structural space-time experiment

The implementation follows the existing repository split between **models/**
(fitted estimators, feature transforms and residual correctors), root-level
research runners, and **tests/**.

| File | Responsibility |
| --- | --- |
| `models/spacetime_structural_features.py` | Shared Toronto-area projection, listing-time coordinate features and matched XGBoost feature identifier |
| `models/spacetime_structural_residuals.py` | Local spatiotemporal kernel, house-type-aware kernel, spatial Matérn GP, space-time Matérn GP |
| `models/spacetime_gpboost.py` | Independent joint GPBoost (boosted trees + Vecchia-approximated GP), optional package |
| `models/spacetime_screen_residuals.py` | Existing Tensor GAM used as an incumbent correction, not duplicated |
| `spacetime_structural_shotgun.py` | Reusable pilot-only sampling, chronological evaluation, caches, bootstrap intervals, model triage |
| `spacetime_fast.ipynb` | **Primary interactive entry point**: initial smoke experiment, plots, optional three-month test |
| `tests/test_spacetime_structural_shotgun.py` | Synthetic checks including no-target features, finite GP output, missing optional GPBoost and resumption |

## Run

**Preferred workflow:** Open `spacetime_fast.ipynb` and run only the appended
**Structural spatiotemporal shotgun** cells. They provide the initial one-month
experiment, display comparisons and failures, and expose an opt-in
`RUN_STRUCTURAL_THREE_MONTH` flag. Both previously completed runs default to off;
set the appropriate switch to True when needed. **GPBoost is archived and
excluded from the active notebook methods** after its tree-only and combined
models both produced 48.7828% MdAPE on October's smoke fold. Its code and
previous saved smoke predictions remain available for diagnostics. Previous
feature-screening cells do not need to be rerun.

Alternatively, from the repository root, run a smoke test without Jupyter.
The CLI defaults still contain GPBoost; pass `--methods` if you want the
same GPBoost-free active set as the notebook:

```bash
python spacetime_structural_shotgun.py --months 2025-10 --train-cap 5000 --trees 250 --validation-cap 250 --methods xgb_tensor_gam xgb_st_kernel xgb_st_kernel_type xgb_spatial_matern xgb_st_matern --output-dir outputs/spacetime_structural_cli_no_gpboost
```

Then run the three-month, 500-tree development pilot from the CLI if preferred:

```bash
python spacetime_structural_shotgun.py --months 2025-09 2025-10 2025-11
```

The GPBoost dependency is optional: `python -m pip install gpboost`.
If absent, its error is recorded and GPBoost is **not** ranked as a failed
predictive model. The other variants continue.

All methods use the same prior-sale training sample and validation homes as
the matched, tuned XGBoost spline+geography baseline. The `xgb_tensor_gam`
arm is also fitted on genuinely earlier *out-of-time* errors for an
additional incumbent comparison. A successful pilot must be repeated at
full scale before being considered a replacement for the full-history
XGBoost (7.4857% off-market MdAPE). The pilot is on earlier development
months and is not an untouched final validation set.

Results: `outputs/spacetime_structural_shotgun/`.

Caution: GPBoost changes the booster as well as adding a spatial covariance
term. Its default pilot tree settings are not comparable in optimization
effort to a fully tuned XGBoost; deprioritization at these defaults is not
proof that a tuned GPBoost could never win. Failed or missing-dependency
candidates are excluded from score rankings.

## Chronological Matérn covariance tuning

The Matérn pilot previously used hard-coded 5 km spatial correlation, 2-month
temporal correlation (space-time variant), latent variance 0.003, residual
noise variance 0.012, and correction strength 0.5. Its boosting features,
model and chronological splits are not altered by this experiment.

Open **`spacetime_fast.ipynb`** and run the final **Tune Matérn** cells:

1. `run_matern_tuning(...)` tunes spatial and space-time Matérn parameters
   on May, June and July 2025 only, with earlier out-of-time XGBoost errors.
   It uses a bounded seeded random search (one historical-default baseline
   plus 19 kernel settings per family). Each trial reuses the same cached
   XGBoost monthly validation and calibration predictions.
2. Turn on `RUN_MATERN_LATER_COMPARISON` to evaluate the *frozen* selected
   kernels against XGBoost splines+geo and its Tensor GAM on Sep–Nov 2025.

Outputs go to `outputs/spacetime_matern_chronological_tuning/`:
`setup.json`, `trials.csv`, `best_matern.json`, monthly XGBoost caches,
and on confirmation `confirmation_pooled.csv`,
`confirmation_monthly.csv`, `confirmation_predictions.csv`.

The objective is `0.65 * pooled MdAPE + 0.35 * pooled MAPE`, all in
percentage-error units. Earlier OOT calibration sales train the GP; tune-month
outcomes select the hyperparameters. Later folds reuse the winning parameters
without further optimization. Neither period is a genuinely untouched final
test set: the earlier development months may have informed prior work, and
Sep–Nov have already been inspected. Do not claim these scores match the
full-history 700-tree six-month XGBoost benchmark.
