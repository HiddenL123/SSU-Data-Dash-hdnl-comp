# Curated list-aware ensemble candidate pool

**Status:** feature candidates selected from exploratory model experiments.
Not final ensemble members and not confirmed best settings. Use the
`curated` suite rather than expanding `all` (existing manifests unchanged).

| Family | Retained representations / adjustments | Why retain |
|---|---|---|
| CatBoost MAE | selected-binary + splines+geo | Best CatBoost pilot MdAPE, 4.6685% over 1,050 homes |
| CatBoost MAE | selected-binary only; splines only | Strong separate text/spatial signals and simpler controls |
| CatBoost MAE | full-binary + splines+geo | Slightly better MAPE (6.1070%) than selected-binary spatial (6.1159%) |
| CatBoost MAE | selected-binary + splines+geo + same-type quota | P90 APE 12.9986% vs 13.0367%, but slightly worse MdAPE |
| CatBoost MAE | legacy text + splines+geo | Competitive 4.7193% MdAPE; tests alternate text handling |
| CatBoost MAE | legacy text + same-type quota + splines | 12.9541% P90 APE (4.7804% MdAPE), tail-focused probe |
| CatBoost Huber | baseline and adjusted-comparable | Alternate objective; adjusted-comps Huber 4.7225% MdAPE |
| XGBoost tuned | splines alone, splines+geo, adjusted comps, selected binary | **Own** screen: splines 4.7444%, spatial 4.8925%; different tradeoffs |
| LightGBM tuned | splines alone, advanced, splines+geo, selected binary | **Own** screen: splines 4.8130%; advanced 6.1006% MAPE |
| XGB/LGB exploratory | full-binary + splines | Transfer hypothesis only; **not** a proven benefit |

The default `curated` pool registers 20 individual candidates:
10 CatBoost, 5 XGBoost and 5 LightGBM. Each candidate consists of exactly
one estimator family and one loss/objective. Spatial and comparable-sale
feature transformations are separate compatible modules. One optional
*out-of-time* residual corrector can be researched later for an individual
candidate; **no residual correction is presumed effective** by these screens.

## Key statistical cautions

- These 1,050-home results come from **Mar–May 2025** under a
  10,000-row / 300-tree / 350-home-per-month budget. They must NOT be
  compared directly with the previously observed 3.8996% full-strength
  single-fit fixed-origin Aug–Oct result.
- CatBoost signed-log prediction errors are **>0.99 correlated** across
  these nearby recipes. Standalone rankings do not establish independent
  incremental ensemble gains. Do not fill an ensemble with near-duplicates.
- The paired MdAPE confidence intervals of most near-leading alternatives
  include zero. The observed MAPE or P90 improvements are small and
  exploratory, not confirmations.
- No transfer of CatBoost effects to XGB or LGB is assumed. Their feature
  recipes were retained on independent family-specific evidence.
- XGBoost and LightGBM are **guaranteed opportunities in screening and
  confirmation**, not compulsory positive ensemble weights. Select a member
  only if it improves the matched held-out aggregate. This prevents a
  worse ensemble being forced by quotas.

## Run notebook-first

`experiments/listaware_model_pool_screen.ipynb` defaults to:

```python
SUITE = "curated"
TOP_K = 14
RUN_SCREEN = False
RUN_CONFIRM = False
```

Enable `RUN_SCREEN` for Mar–May 2025. Then select from those results
(with the predeclared Cat/XGB/LGB anchors), and enable `RUN_CONFIRM`
for full-budget Jun–Jul 2025 evaluation. Distinct, fingerprinted outputs:

- `outputs/listaware_pool_screen_curated/`
- `outputs/listaware_pool_confirmation_curated/`

CLI alternative:

```powershell
uv run python -m feature.listaware_pool_screen --stage screen --suite curated

uv run python -m feature.listaware_pool_screen --stage confirm --suite curated --top-k 14 --screen-dir outputs/listaware_pool_screen_curated
```

The compute branch's list-aware ensemble selector uses the latter directory
by default. It greedily admits **marginally helpful** candidates, with
one-time later chronological audit and a full-history refit. These later
months have also been inspected elsewhere in the research program.

The existing off-market ensemble factory stays independent and unchanged.
