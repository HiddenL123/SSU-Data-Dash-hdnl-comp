# Broader list-aware individual-model and feature-family screen

This is **Stage A**, not the final list-aware ensemble factory. It directly addresses
the gap between the earlier narrow model comparisons and the off-market
factory's wide feature-bank → bagging → Optuna → marginal ensemble-admission
pipeline.

**Notebook:** [../experiments/listaware_model_pool_screen.ipynb](../experiments/listaware_model_pool_screen.ipynb)

**Python runner:** `feature/listaware_pool_screen.py`

## Curated follow-up (from the latest composite feature experiment)

The `curated` suite is now the preferred **next** list-aware ensemble pool:
20 candidate model/feature recipes (10 CatBoost / 5 XGBoost / 5 LightGBM),
including selected-binary spline/geo, full-binary spline/geo, comparable
quota, XGB splines and LGB advanced alternatives. Refer to
[the curated pool rationale](listaware_curated_pool.md) for evidence,
hypotheses and non-duplication rules.

```powershell
uv run python -m feature.listaware_pool_screen --stage screen --suite curated
uv run python -m feature.listaware_pool_screen --stage confirm --suite curated --top-k 14 --screen-dir outputs/listaware_pool_screen_curated
```

Its output directories are separate from `all` to preserve existing cached
predictions/manifests. Three-family representation is guaranteed for
**candidate consideration**, not forced ensemble membership. The best
individual model may still have negligible marginal blending value.

## What the 44 model configurations explore

A factorized, documented catalog rather than a Cartesian-product search:

- CatBoost MAE, Huber and RMSE, including tuned RMSE and centered direct-target controls
- XGBoost and LightGBM tuned baselines and direct-target variants
- Original text features vs no text, selected 47-term binary,
  training-fitted full binary BoW, full count BoW, and CatBoost full TF-IDF
- Comparable-sale age-first filtering and same-property-type quota
- Spatial splines alone, geographical context alone, combined spline+geo
- Preexisting AVM feature add-ons in a small, controlled family

Each candidate belongs to at least one named suite: `core`, `text`,
`comparables`, `spatial`, `objectives`, `avm`, or `all`.

This experiment does **not** perform feature bagging or optimize individual
hyperparameters with Optuna; those are separate stages after stronger
standalone models and candidate feature recipes have been identified.

## Pipeline

| | Initial screen | Confirmation |
| --- | --- | --- |
| Months | Mar–May 2025 | Jun–Jul 2025 |
| Training protocol | expanding completed-sale history | expanding completed-sale history |
| Model fit | one model per candidate and month | one model per finalist and month |
| Train cap | 10,000 | 200,000 (dataset ~150,000) |
| Held-out cap | 350/month | 100,000/month (effectively uncapped) |
| Iterations | capped at 300 | full existing parameter counts |
| Candidate count | selected suite, up to 44 | capped top shortlist + prior controls |
| Shortlist | not applicable | selected from **screen** only |

A screen ranking criterion `0.65*MdAPE + 0.35*MAPE` chooses candidates for
confirmation. Four previously strong/necessary controls are always included:
`cat_huber_baseline`, `cat_mae`, `xgb_spline_geo`, `lgb_spline_geo`.
The selection also retains another screened model from each backend when
available; default `top_k=8` is a soft bound including these controls.

Every candidate predicts exactly matching `(month, ml_num)` homes with
`sold_price`, `date_sold`, and other post-sale columns hidden. All
historical labels used for fitting precede the corresponding prediction
month. Every per-candidate/month forecast is written immediately and
revalidated on restart. Manifests lock the source dataset fingerprint,
candidate specs, exact parameters, tree and sampling budgets, and settings.
The confirmation manifest additionally fingerprints the source screen
manifest and score file, recording the selected candidates.

## Notebook-first usage

Open `experiments/listaware_model_pool_screen.ipynb` in VS Code/Jupyter.

- Check the catalog. Set `SUITE="core"` to check the pipeline, then switch
  to `SUITE="all"` for the intended broader screen. **Use separate output
  folders** for differing suites or budgets.
- Set `RUN_SCREEN=True` to run or resume. This trains up to **44×3**
  independent models, so consider computer runtime; notebook defaults to
  disabled fitting.
- Examine screen predictions, MdAPE/MAPE/P90 and month stability. The
  shortlist is computed without seeing the later confirmation labels.
- Set `RUN_CONFIRM=True` only after completing the screen. The later run
  fits a small shortlist at full strength; rerunning with unchanged settings
  reuses verified caches.

CLI equivalents from repository root:

```powershell
uv run pytest tests/test_listaware_pool_screen.py tests/test_listaware_shotgun.py -q

# Small preflight screen
uv run python -m feature.listaware_pool_screen --stage screen --suite core

# Complete broad screen
uv run python -m feature.listaware_pool_screen --stage screen --suite all

# After its complete output exists; full-strength historical confirmation
uv run python -m feature.listaware_pool_screen --stage confirm --suite all \
    --screen-dir outputs/listaware_pool_screen_all
```

Candidate/model/split settings are pinned in separate output folders.
Default output names: `outputs/listaware_pool_screen_<suite>/` and
`outputs/listaware_pool_confirmation_<suite>/`.

Outputs:

- `manifest.json` — immutable experiment identity
- `candidate_catalog.csv` — exact family, target, objective, and feature recipe
- `{month}/models/{candidate}.csv` — restartable per-home predictions
- `predictions.csv` — all individual models on identical month/home IDs
- `error_correlations.csv` — matched-home signed-log-error correlations between every candidate pair, diagnostic only
- `pooled_scores.csv`, `monthly_scores.csv`, `house_type_scores.csv` —
  within-run paired error tables

## Interpretation

**Do not compare pilot raw MdAPE to the existing 3.8996% benchmark.**
The benchmark used an August–October 2025 full-strength **single-fit
fixed-origin** evaluation, while Stage A intentionally uses other months,
a pilot budget and **expanding** monthly fits.

The original 50/35/15 CatBoost-MAE/XGBoost/LightGBM ensemble is still a
strong target to beat in an eventual protocol-matched comparison. It is
**not** discarded, refitted, or overwritten in this model-pool experiment.

These historical months have been previously investigated. A June–July
confirmation later than the March–May screen is a *chronological* check
inside the new procedure, not an untouched dataset-wide held-out test.

## Next phases, not implemented here

1. Learn a **list-aware-specific** training-fold-only text-term ranking;
   compare it with the 47 off-market-selected terms, with no future labels.
2. Add reproducible selected+extra, weighted and uniform BoW feature
   bags, bootstrap and 80% row subbagging, and independent seeds.
3. Tune selected architectures with backend-specific Optuna spaces.
4. Build a separate list-aware ensemble factory with candidate
   admission/replacement based on held-out **marginal gain**, rather than
   single-model rankings or correlation alone.
5. Freeze candidate selection and evaluate a later chronological
   portfolio and then refit legally eligible full history for final inference.

Avoid expanding the suite or altering the shortlist because a previously
inspected confirmation period gives an attractive result.
