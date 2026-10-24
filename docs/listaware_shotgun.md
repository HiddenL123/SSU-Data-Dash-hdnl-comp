# List-aware model showdown — non-neural, matched chronological experiments

The list-aware housing model is **separate from** the frozen off-market
7.4059% MdAPE ensemble. We use the best recorded list-aware algorithms and
transfer the most successful off-market engineered features into them.

Entry point: `listaware_shotgun.py`; configurations in
`models/listaware_shotgun_specs.py`. For an interactive run with automatically
rendered leaderboards and diagnostics, open
[`listaware_shotgun.ipynb`](../listaware_shotgun.ipynb).

## Research candidates

- **CatBoost Huber sale-to-list ratio** — primary paired control, based on E28.
  Historical six-month MdAPE **4.06%**, but this old result **must not** be
  compared to a different-month/capped pilot result.
- CatBoost Huber + selected binary terms (from E34–E35 off-market).
- **CatBoost Huber + XGBoost's best spline/geography feature stack**.
  This is a CatBoost estimator trained directly on those XGBoost-developed
  features, not a copy of XGBoost predictions.
- CatBoost Huber + spline/geography + hedonic-adjusted comparables.
- CatBoost Huber + spline/geography + all four Redfin/Zillow-inspired
  prior-sale features (hedonic-adjusted comps, OOT local Ridge residuals,
  robust comp quantiles and smoothed multi-resolution geographic grids).
- CatBoost MAE, RMSE and optimized RMSE configurations from E28/E33.
- Tuned **XGBoost** ratio model (E32), both with binary terms alone and the
  spline/geographic stack (E37) and adjusted-comparable add-ons.
- Tuned **LightGBM** ratio model (E31) with binary and spline/geography features.
- Optional direct log-sale target with CatBoost Huber (centered to avoid the
  known log-target numerical instability), plus further previous feature
  families `hedonic_index`, `adaptive_comps`, `time_adjusted_comp`.

**Blends are additional candidates, not fitted on the evaluation fold:**
- 70% CatBoost Huber baseline + 30% tuned XGBoost binary;
- 70% CatBoost spline+geo + 30% XGBoost spline+geo;
- 65% CatBoost adjusted-comps + 35% XGBoost adjusted-comps;
- 50% CatBoost spline+geo + 35% XGBoost spline+geo + 15% LightGBM spline+geo;
- 40% CatBoost Huber baseline + 35% CatBoost spline+geo + 25% XGBoost spline+geo.

Blends use **geometric averaging of predicted sale prices**. Blend weights
were declared before this screen; no weights are optimized on validation
labels. If a blend requires an omitted candidate, the runner skips it.

## Quick pilot (first run)

Run from the repository root using Python 3.13 and the same `DATA_DIR` as
your previous Parquet experiments:

```powershell
git switch main
git pull origin main
py -3.13 -m pytest tests/test_realworld_avm_shotgun.py tests/test_listaware_shotgun.py -q
py -3.13 listaware_shotgun.py
```

Defaults:
- Chronologically train on sales completed **before** each holdout month.
- Development months: August, September, October 2025.
- Up to 10,000 earlier sales and 350 held-out sale homes per month.
- 300 boosting rounds per candidate (capped relative to recorded full models).
- 4 CPU threads per estimator (can use `--threads 2` in the daytime).
- Same homes for every candidate and blend.
- Supports **interruption-safe, per-candidate-per-month prediction caching**.
  If interrupted, rerun the *exact same command*.

Outputs under `outputs/listaware_shotgun_pilot_expanding/`:
- `pooled_scores.csv` — paired MdAPE, MAPE, P90 error, delta vs CatBoost
  Huber baseline, fraction of individual homes improved, paired bootstrap CI.
- `monthly_scores.csv` — month-level comparison.
- `house_type_scores.csv` — per-property-type diagnostic.
- `predictions.csv` — per-home historical sale/prediction across all arms.
- `{month}/models/{name}.csv` — individual candidate result checkpoints.
- `{month}/blends/{name}.csv` — predeclared blend results.
- `manifest.json` — source data hash and all experiment settings.

The pilot is not a 4.06% reproduction: it is a **small, matched-budget
screen** for identifying which ideas deserve full-strength training.

## Targeted experiments

For CatBoost + the best XGBoost feature representation, run just:

```powershell
py -3.13 listaware_shotgun.py --candidates cat_huber_baseline cat_huber_spline_geo xgb_spline_geo --output-dir outputs/listaware_cross_family_screen
```

For direct vs ratio and objective ablations:

```powershell
py -3.13 listaware_shotgun.py --candidates cat_huber_baseline cat_huber_direct cat_mae cat_rmse cat_tuned_rmse --output-dir outputs/listaware_objective_screen
```

These use distinct output folders because their candidate sets differ from
the default screen. A directory with an existing mismatched experiment
manifest is never silently reused.

## Full-strength shortlist confirmation

After ranking the pilot, select the most promising 2–4 models. For example:

```powershell
py -3.13 listaware_shotgun.py --full --candidates cat_huber_baseline cat_huber_spline_geo xgb_spline_geo lgb_spline_geo
```

`--full` refits on **all eligible prior sales** and retains the recorded
candidate boosting-round counts (900 CatBoost baseline, 700 XGBoost tuned,
1,800 LightGBM tuned). Every held-out home is evaluated. This may take
significantly longer than the pilot. Choose a shortlist rather than running
the entire combinatorial catalog at full strength.

Optional: `--mode fixed_origin` freezes training before the first month,
closer to the June–September 2026 hidden-test forecasting horizon.

After shortlist selection, separately evaluate December 2025–May 2026 with
frozen candidates and weights; those six months have already been repeatedly
consulted and are NOT an untouched test. If needed, use the existing
`spacetime_list_aware_gam.py` runner for a further XGBoost + out-of-time
Tensor GAM residual experiment. Its historical GAM calibrations must come
from the same candidate family, not from another model's in-sample residuals.

## No leakage or misleading comparisons

All supervised historical features use only sales with `date_sold` strictly
before the queried home's listing month, and query holdouts are passed
through `data.hide_answers(..., 'list_aware')`. Only listing price remains.
Model targets are either `log(sold_price/list_price)` or (optional)
centered `log(sold_price)`; full sale predictions are reconstructed before
scoring.

Do not use the full-history off-market release model's *in-sample predictions*
as features to fit list-aware training labels. The off-market-informed
CatBoost experiment is best undertaken with genuinely earlier-month,
out-of-time off-market valuations for **every** stage-two training row.
It is therefore a separate, more expensive experiment, not silently
included as a leakage-prone shortcut.

The test suite includes strict month/house cohort and target checks,
cat/xgb/lgb spec checks, reconstruction of ratio prices, cache validation,
predeclared blend weights and an interrupted-run resume check.

To verify implementation locally:

```powershell
py -3.13 -m pytest tests/test_realworld_avm_shotgun.py tests/test_listaware_shotgun.py -q
```

**No actual competition dataset results are claimed until the tests and
screening runs finish successfully on the local machine.**
