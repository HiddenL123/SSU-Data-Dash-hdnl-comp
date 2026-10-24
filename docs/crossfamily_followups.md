# Fixed-origin and cross-family blend follow-ups

## 1. List-aware fixed-origin comparison

Current leader (full-strength expanding, 5,951 historical homes): **50% CatBoost Huber spline/geo + 35% tuned XGBoost spline/geo + 15% tuned LightGBM spline/geo**, MdAPE 3.8970%, versus CatBoost Huber 3.9529%. The selected-fold paired MdAPE interval still includes zero. These numbers do **not** predict fixed-origin results.

`listaware_fixed_origin_followup.py` trains exactly six families **once** using sales completed before August 1, 2025, then predicts all August–October 2025 audit homes with the same fitted models. It includes CatBoost Huber baseline, CatBoost MAE, CatBoost spline/geo, CatBoost advanced, XGBoost spline/geo and LightGBM spline/geo. For a quick proof of concept, it caps historical training and rounds; `--full` uses all eligible historical sales and all recorded boosting iterations.

Predeclared additional geometric blends:

| Blend | Component weights |
| --- | --- |
| `followup_winner_plus_mae` | CatBoost spatial 40%, XGBoost spatial 28%, LightGBM spatial 12%, CatBoost MAE 20% |
| `followup_two_cat_spatial_advanced` | CatBoost spatial 70%, CatBoost advanced 30% |
| `followup_cat_mae_xgb_lgb` | CatBoost MAE 50%, XGBoost spatial 35%, LightGBM spatial 15% |
| `followup_huber_mae` | CatBoost Huber baseline 70%, CatBoost MAE 30% |
| `followup_cat_spatial_mae` | CatBoost spatial 70%, CatBoost MAE 30% |

The pre-existing 50/35/15 winner is also automatically compared, unchanged.

Commands:

```powershell
py -3.13 listaware_fixed_origin_followup.py --pilot
py -3.13 listaware_fixed_origin_followup.py --full
```

To **immediately analyze these additional blends without any retraining** against your completed expanding evaluation:

```powershell
py -3.13 listaware_fixed_origin_followup.py --from-predictions outputs/listaware_full_shortlist/predictions.csv
```

The latter produces `outputs/listaware_full_shortlist/fixed_weight_followup/followup_pooled.csv` and is **NOT fixed-origin evidence**. To save elsewhere, use `--output-dir`.

The real fixed-origin run saves model CSV caches under `outputs/listaware_fixed_origin_full/{month}/models/` and follow-up scores under `outputs/listaware_fixed_origin_full/fixed_weight_followup/`. Restarting with unchanged data and parameters reuses predictions.

## 2. Off-market family diversity screen

`offmarket_crossfamily_blend.py` reads the existing frozen XGBoost ensemble's historical `outputs/ensemble_release_candidate_full/shadow_audit_predictions.csv`, then fits *off-market* (no listing price) tuned standalone XGBoost, CatBoost RMSE and LightGBM regressors on the **same earlier sales**, using selected-binary/splines/geographic features. Predictions are scored on identical homes and months as the frozen ensemble. The existing ensemble is never refitted and its original audit CSV is never modified.

The default pilot uses 25,000 earlier sales, 350 boosting iterations per standalone model and 350 historical homes per evaluation month. The result is **not compute-matched** against the existing full-sized ensemble and should only decide whether full-strength challengers are worth fitting. The `--full` flag trains using all earlier eligible sales and the full prior tuned CatBoost/LightGBM iteration budgets.

Four **predeclared geometric blends** (weights on log predicted prices) are tested:

| Blend | Frozen XGB | CatBoost | LightGBM |
| --- | ---: | ---: | ---: |
| `frozen_plus_cat_80_20` | 80% | 20% | 0% |
| `frozen_plus_lgb_80_20` | 80% | 0% | 20% |
| `frozen_cat_lgb_70_20_10` | 70% | 20% | 10% |
| `frozen_cat_lgb_60_25_15` | 60% | 25% | 15% |

Commands:

```powershell
py -3.13 offmarket_crossfamily_blend.py
py -3.13 offmarket_crossfamily_blend.py --full
```

The new base-XGB pilot produces `outputs/offmarket_base_xgb_pilot/pooled_scores.csv`, `monthly_scores.csv`, `predictions.csv`, plus per-member/month restartable caches. The frozen ensemble's historical reference is **required**. Its May–October 2025 shadow audit may have been previously inspected and is not independent of earlier model development.

### Standalone XGBoost as the blending base (new follow-up)

The first 1,050-home pilot found the frozen ensemble superior to every blend
with CatBoost/LightGBM. That does **not** resolve whether separately trained
CatBoost/LightGBM add value to a **single** tuned XGBoost model.

The runner now fits a standalone tuned XGBoost using the **same historical
training rows, tree cap, feature stack, and evaluation cohort** as its
CatBoost/LightGBM alternatives. It reports two paired comparisons:

- `delta_base_xgb_mdape_pp` / `delta_base_xgb_mape_pp` and
  `base_xgb_delta_*_ci_*`: improvement vs **same-budget standalone XGBoost**;
- `delta_mdape_pp` / `delta_mape_pp` and `delta_*_ci_*`: improvement vs
  the **frozen full-budget ensemble**.

Additional *predeclared* blends:

| Blend | Standalone XGB | CatBoost | LightGBM |
| --- | ---: | ---: | ---: |
| `base_xgb_plus_cat_80_20` | 80% | 20% | 0% |
| `base_xgb_plus_lgb_80_20` | 80% | 0% | 20% |
| `base_xgb_cat_lgb_70_20_10` | 70% | 20% | 10% |
| `base_xgb_cat_lgb_50_35_15` | 50% | 35% | 15% |

To avoid unnecessary retraining, reuse the CatBoost and LightGBM
predictions from the old 25k-row/350-tree experiment, **only if their
manifest and exact evaluation homes match**:

```powershell
py -3.13 offmarket_crossfamily_blend.py --reuse-family-caches outputs/offmarket_crossfamily_pilot
```

If the older cache is unavailable, omit `--reuse-family-caches`: all three
single-family challengers will train. The new default output directory is
`outputs/offmarket_base_xgb_pilot/`, preserving the previous experiment.
For full-strength history and rounds use
`py -3.13 offmarket_crossfamily_blend.py --full`, which writes to
`outputs/offmarket_base_xgb_full/`.

The original `frozen_*` blends remain in the leaderboard as controls.
**A positive result relative to standalone XGBoost is not, by itself, a win
over the frozen ensemble.**

## 3. Guardrails

- Only previous completed sales enter training. Every query's post-sale fields are hidden; off-market additionally excludes list price.
- Compare on **identical month+MLS keys** and exact observed sale prices; corrupted or differently scoped outputs fail instead of silently inner-joining.
- Cache manifests hash the historical listings and/or read-only ensemble reference and refuse incompatible settings.
- Never tune blend weights using these same audit outcomes and describe them as out-of-sample confirmation. All blend weights above were predeclared.
- New experiments write new `outputs/` folders and do not update the existing off-market saved model or release roster.
- Run `py -3.13 -m pytest tests/test_crossfamily_followups.py -q` before training. All actual competition results require the user's local data files and saved ensemble audit; the code alone cannot establish improved accuracy.

Interactive root notebooks: [list-aware follow-up](../listaware_fixed_origin_followup.ipynb) and [off-market follow-up](../offmarket_crossfamily_blend.ipynb).
