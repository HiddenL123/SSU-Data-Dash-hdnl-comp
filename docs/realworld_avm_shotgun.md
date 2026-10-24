# Final non-neural AVM feature shotgun (October 2026)

**Purpose:** test genuinely new off-market feature representations inspired by
published real-estate AVM practices, before concentrating development on the
separate list-aware track. **No neural networks; no outside housing data.**

This is experimental feature research. It does not change the frozen AutoML
ensemble, overwrite submission files, retune weights or load hidden test labels.

## What is new (and what was already tested)

| Candidate | Intuition | What is implemented |
| --- | --- | --- |
| rw_adjusted_comps | Redfin Owner Estimate describes property-adjusted comparable prices | Prior-only Ridge hedonic adjustment of eligible local sold homes for subject size/type, beds/baths, parking and sale/listing time; weighted log-price and spread |
| rw_local_residuals | Local hedonic residuals/kriging-style correction | Distance/similarity-weighted **out-of-time Ridge** residuals of earlier sold homes. Not an XGBoost residual or a full kriging implementation |
| rw_robust_comps | Comparable sales can have skewed prices | Weighted 25th, 50th, 75th percentile and IQR of prior comparable log prices |
| rw_multiscale_tiles | Zillow explains hierarchical geographic tiles | Past-only, smoothed log-price-per-area statistics in fixed 1/3/10 km approximate grids. **Not** learned neural location embeddings |
| rw_combined | Complementarity | Combines all four new families |
| adaptive_comps / hedonic_index | Previous repo experiments | Reused as controls; not new discoveries |

The comparable adjustments use a *single past-trained* hedonic coefficient
vector at the subject's listing month. Historical sale residuals are estimated
with a coefficient vector fit only on sales from earlier months. The month of
each query listing is the information cutoff; any sale in or after that
month is excluded from the corresponding features, including during
training-time feature construction.

The geographic grid is a rough synthetic-coordinate proxy, not a real H3 or S2
map. Do **not** import external GTA house prices or real map labels into
synthetic coordinates.

Public sources and motivation:
- Redfin adjusted comparable methodology:
  https://support.redfin.com/hc/en-us/articles/360018505351-Owner-Estimate
- Zillow prior AVM workflow, geographic tiling and temporal modeling:
  https://www.zillow.com/news/building-the-neural-zestimate/
- Hedonic adjustments and localized residual smoothing are generic statistical
  modeling hypotheses tested here, not a claimed reconstruction of any
  proprietary production estimator.

This code is inspired by those disclosures, not a claim to implement their
proprietary AVMs or prove improved accuracy.

## Required preflight

Use the same environment and data source that validated your transferred
Colab checkpoint. In PowerShell, after git pull:

    py -3.13 -m pytest tests/test_realworld_avm_shotgun.py -q

Your previously completed off-market factory must contain:

    outputs/ensemble_release_candidate_full/shadow_audit_predictions.csv

It holds the frozen 7.401%-MdAPE reference predictions on 5,951 homes sold
August-October 2025. The new research runner READS that CSV; it NEVER
retrains the ensemble.

If DATA_DIR points to the Colab-compatible Parquet folder, leave it there for
stable historical-row and dtype selection.

## Stage A: quick shotgun (recommended first)

    py -3.13 avm_realworld_shotgun.py

Defaults:
- August, September, October 2025, **all months**;
- same chronological homes for each research arm within each month;
- 10,000 training homes per monthly fold, 220 XGBoost trees;
- 350 test homes per monthly fold;
- matched tuned XGBoost baseline using selected binary text, spatiotemporal splines and geographic context, plus five new variants and three previous controls;
- output: outputs/realworld_avm_pilot/;
- the reference is evaluated on **those same exact held-out homes**.

The frozen ensemble is full-history/700-tree, whereas these candidates are
reduced budget. Differences to the ensemble at this stage are for **context**,
not valid full-fidelity claims. The appropriate low-cost selection metric is
the **incremental delta to the matched 220-tree XGBoost baseline**, which
already includes splines and geographic context.

Outputs:
- paired_predictions.csv: both the candidates and the frozen ensemble's
  predictions on identical homes, with true sold prices;
- pooled_vs_ensemble.csv: MdAPE/MAPE, paired bootstrap CIs and delta from
  both the frozen ensemble and matched screen XGBoost;
- monthly_vs_ensemble.csv: per-month paired differences;
- per-month predictions.csv / summary.csv / failures.json;
- research_manifest.json: exact input-data fingerprint, reference file hash,
  feature list and compute settings for strict resumption.

If interrupted after a completed month, rerun the same command. Completed
months are reused; changing experiment settings/data/reference requires a new
output directory. An interrupted *individual month* may restart that month.

## Stage B: full-fidelity shortlisted confirmation

After inspecting pilot results, select 1-2 promising variants and use:

    py -3.13 avm_realworld_shotgun.py --full --features baseline rw_adjusted_comps rw_combined

This is still a **matched-homes comparison**, and it uses 700 trees / all
eligible historical sales / every holdout property, putting model compute
much closer to the ensemble reference. However, the independent challenger
remains one XGBoost model, while the reference is a selected weighted ensemble:
they are not identical architectures.

The runner reads the same frozen shadow-audit CSV. Do not select an ensemble
weight using August-October audit outcomes and report a win on those same
months. Treat them as examined historical development data. Only subsequently
run any stronger contender through a separately frozen, paired chronological
evaluation. December 2025-May 2026 has also already been inspected extensively
and is **not** untouched validation.

## Choosing whether to keep a new feature

Prioritize:
1. Consistent reduction in **MdAPE**, not just MAPE.
2. Same direction of improvement across most months and property types.
3. Statistically and practically meaningful paired deltas relative to a
   **matched-budget** XGBoost control.
4. Whether a feature adds error diversity to the existing ensemble: a strong
   individual does not automatically make a stronger ensemble.
5. Fitting/inference time and feature schema consistency.

No winner should be claimed before actual CSV results are available.

## Next track: quick list-aware replication (implemented)

The separate, resumable list-aware screen is now in `listaware_shotgun.py`,
with its full command guide in [`docs/listaware_shotgun.md`](listaware_shotgun.md).
Run it after inspecting the off-market feature screen, without adding listing
prices to the off-market model.

Proposed short screen (same chronological months, same homes per model):
- Current CatBoost Huber **log(sold/list)** baseline (4.06% previously reported).
- CatBoost Huber with adjusted-comps and lagged-tile features.
- Direct log-sold-price vs log-ratio targets, each with identical features.
- CatBoost RMSE/MAE/Huber, tuned LightGBM and XGBoost ratio candidates.
- Pricing-strategy features: log(list price / independently generated OOT
  off-market value) and comparable-estimate / list-price gap.
- Fixed low-complexity log-price blends from chronological out-of-time streams.
- Promote only winners to full-strength validation, then frozen list-aware
  ensemble selection. Never train a stage-2 feature on the same home's
  in-sample off-market prediction.

Do not reopen the previous off-market ensemble's already frozen roster.
