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
**Structural spatiotemporal shotgun** cells. They run the initial one-month
experiment, display comparisons and failures, and expose an opt-in
`RUN_STRUCTURAL_THREE_MONTH` flag. Previous feature-screening cells do
not need to be rerun.

Alternatively, from the repository root, run a smoke test without Jupyter:

```bash
python spacetime_structural_shotgun.py --months 2025-10 --train-cap 5000 --trees 250 --validation-cap 250
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
