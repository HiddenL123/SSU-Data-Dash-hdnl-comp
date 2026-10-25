# Final list-aware CatBoost-MAE feature experiment

A **single targeted ablation**, designed before running and separate from
the **off-market-only** AutoML/ensemble factory.

## Why this test

The strongest full-strength list-aware fixed-origin result for 5,951
August--October 2025 homes was the existing geometric blend:

- 50% CatBoost MAE (`cat_mae`)
- 35% tuned XGBoost with splines/geo (`xgb_spline_geo`)
- 15% tuned LightGBM with splines/geo (`lgb_spline_geo`)

MdAPE 3.899564%, MAPE 4.853451%, P90 APE 10.040185%.
The benefit over standalone CatBoost MAE was just 0.024160 pp, with a
paired bootstrap interval spanning zero.

One small unanswered question is whether **the CatBoost MAE member itself**
benefits from the stronger comparable-sale selection and spatial context
features already developed elsewhere in the repository. Huber spatial
experiments are not sufficient to answer that MAE-specific question.

## What the new script trains

Exactly three new CatBoost **MAE loss / log sold-to-list ratio** models,
using the SAME full-strength settings and the SAME single fixed origin as
the original runner, without touching any historical model definitions:

| New model | Feature representation |
|---|---|
| `cat_mae_type_quota` | Current MAE comparable features + at least eight same-type neighbours where available |
| `cat_mae_spatial_geo` | Selected binary words + spatiotemporal splines + geographic context |
| `cat_mae_spatial_geo_type_quota` | Selected binary words, same-type quota, splines and geographic context |

The second comparison introduces selected-binary words AND the spatial
add-ons together, so it tests the established full stack, not a
single isolated spatial feature. The third model isolates the marginal
same-type quota on that stack.

For each, the script additionally constructs a predetermined replacement
50/35/15 geometric blend with the **original XGB/LGB prices unchanged**.
No blend weights are trained or searched on these evaluation labels.

## Run

From the repository root, after merging the feature-experiment branch:

```powershell
uv run pytest tests/test_listaware_final_mae_features.py tests/test_listaware_confirmation.py -q
uv run python listaware_final_mae_features.py
```

Requires the previously validated baseline outputs:

- `outputs/listaware_fixed_origin_full/manifest.json`
- `outputs/listaware_fixed_origin_full/fixed_weight_followup/followup_predictions.csv`

If saved elsewhere use `--source-dir` and `--output-dir`.

The source manifest must prove `full: true`, 
`mode: single-fit_fixed_origin`, and the same original candidate set,
date cutoff, training cap, and month set. The historical listings hash
must match. The test evaluates **every eligible** August, September and
October 2025 home and fits on all earlier completed sales as before.
Only **new** CatBoost candidates are fit. Existing model predictions
are read-only and have exact month+MLS ID, label and house-type checks.

Resume by rerunning the same command with unchanged source data/settings:
the runner caches the three new base-model predictions for each month
and reuses completed candidates. It deliberately rejects any change
to the source predictions, source manifest, feature specifications, seed,
runtime threading or experiment settings under the same output folder.

## Files produced

`outputs/listaware_final_mae_features/`:

- `pooled_vs_frozen_winner.csv`: unchanged previous baselines and all
  new models/replacement blends, with MdAPE/MAPE/P90 and paired bootstrap
  95% intervals for **candidate minus prior winner**.
- `monthly_vs_frozen_winner.csv`: same matched comparison month-by-month.
- `error_correlations.csv`: signed log-error correlation and fraction of
  homes improved against the winner, for **diagnostic use only**.
- `predictions.csv`: full home-level model/blend forecasts and labels.
- `manifest.json`, plus restartable `{month}/models/{candidate}.csv`.

This does **not** train any residual-correction model, since the previously
promising KNN/RBF/GAM corrections were off-market experiments, not validated
on list-aware out-of-time residuals. Those require additional independently
chronological calibration and are explicitly out of scope.

## Interpretation and stopping rule

August--October 2025 is a previously inspected validation period.
The reported bootstrap intervals are unadjusted exploratory diagnostics,
not proof of improvement on a fresh holdout.

Prefer the old 3.8996%-MdAPE model unless a replacement improves both
MdAPE and MAPE by a meaningful amount, shows reasonable month stability,
and passes an independent chronological check before final promotion.
Otherwise stop hand-engineering the list-aware track and proceed with
the **off-market** ensemble factory. Do not recycle these outcomes to
adaptively search new blend weights or change the model factory's pool.
