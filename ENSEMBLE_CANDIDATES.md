# Off-market ensemble candidate roster

**Objective:** attempt to lower full-history off-market MdAPE below **7.00%**
(current strongest observed MdAPE **7.4857%**; a reduction of at least
**0.4857 percentage points**, about **6.5% relative**). This is a stretch
goal, *not* a guaranteed score.

Start with **diverse XGBoost members**, not only spatial residual corrections.
Keep list-aware models in a **separate ensemble** because they can see
list price and are trained on a different target.

## What we already have

| Family | Variant or reference | Evidence / next step |
| --- | --- | --- |
| XGBoost | Tuned selected-binary reference (47 LASSO/Elastic-Net-selected terms) | Preserve the strong reference with 700 trees/full expanding history for confirmation |
| XGBoost spatial | Spline features, geographic context, splines + geo | Strong standalone feature representations; compare aligned predictions |
| XGBoost + GAM | Tensor GAM correction trained on model-specific OOT residuals | Three-month pilot MdAPE **10.7983%**, MAPE **13.4371%** |
| XGBoost + spatial GP | Tuned spatial Matérn correction | Three-month pilot MdAPE **10.9641%**, MAPE **13.3782%** |
| XGBoost + space-time GP | Tuned space-time Matérn correction | Three-month pilot MdAPE **10.7129%**, MAPE **13.6247%** |
| GPBoost | Independent joint boosting+GP | **Archive**: 48.78% MdAPE pilot, same tree-only and joint; not active |
| Other boosting | LightGBM, CatBoost | Future controlled model-family diversity; do not infer current performance from unverified scores |

The three-month pilots contain 2,700 homes on Sep–Nov 2025, and **are not
directly comparable** to the full-history 700-tree six-month score.

## Feature-bagging candidates

`models/text_features.py` provides **47 selected terms** and a **training-fitted
approximately 1,000-term binary BoW** representation. The candidate bank for
bagging is the union of the selected terms and up to 1,000 additional
training-discovered terms, so the union may be larger than 1,000.

| ID | Binary BoW terms | Rows | Why |
| --- | --- | --- | --- |
| `selected_47_reference` | All selected terms | All eligible rows | Fair baseline |
| `full_vocab_binary` | Complete training-fitted candidate bank | All | Tests whether the other terms help |
| `selected_plus_100` | All 47 + 100 sampled other terms | All | Retain established signal plus controlled exploration |
| `uniform_150_a` | 150 uniformly sampled terms | All | High text diversity, potentially lower individual strength |
| `weighted_150_a/b` | 150 terms weighted **4×** for the selected 47 | All | Preserve more selected signal while exploring; independent seeds |
| `weighted_150_bootstrap_a/b` | Weighted 150-term bag | Bootstrap eligible training rows | Joint text and row diversity |
| `selected_47_bootstrap_a/b` | Selected 47 | Bootstrap eligible training rows | Isolate benefit of row diversity |
| `weighted_150_subbag80` | Weighted 150-term bag | 80% without replacement | Alternative to duplicated bootstrap rows |

All candidates **retain structured, categorical, comparable-sale, spline,
and geographic features**. Only text columns and/or XGBoost-fitting rows
vary. The 4× prior is an *initial heuristic*, not a tuned or proven optimum.
Future **adaptive text-term weights** should be learned from earlier
out-of-time development performance, with a nonzero exploration floor;
do not update them based on Sep–Nov confirmation or Dec–May score.

**Crucial bootstrap ordering:** fit historical comparables and other
context on the FULL chronologically eligible training history first,
then sample rows of the resulting feature matrix for each ensemble
member. Do **not** rebuild the historical housing market from bootstrap
duplicates. Text vocabulary is also fitted only on eligible training
remarks, never on validation/test remarks.

## Ensemble experiments, prioritized

1. **Feature / row bagging** on one earlier development month: does a
   selected+extra or selected-weighted 150-term bag retain adequate accuracy
   while generating diverse residuals? Compare 47-only, full text,
   feature-only, row-only, and joint feature+row variants. Save member
   predictions keyed by `(month, ml_num)`.
2. **Small paired blends** of members on identical homes. First compare
   47-selected XGBoost versus a predeclared 50/50 blend with one feature bag,
   and a simple equal-weight blend of 2–3 bags. Evaluate incremental
   MdAPE/MAPE, *error correlation*, percentage of homes improved, subgroup
   consistency, and price-tail behavior. Don't use the scoring holdout to
   choose weights.
3. **Residual-complementarity check**: separately test fixed-weight
   Tensor GAM + spatial Matérn, then add space-time Matérn. These predictions
   should share exactly the same **underlying XGBoost and calibration
   provenance**; if not, treat each corrected model as an independent
   ensemble member rather than adding residuals across different bases.
4. **Controlled model-family / feature-base diversity**: splines-only,
   geo-only, splines+geo, text modes, different XGBoost seeds, LightGBM and
   CatBoost. Promote only candidates with reliable *marginal ensemble gain*.
5. **Full-strength expansion / final submission**: use frozen candidate
   definitions, full 700-tree expanding-history training, and a legitimate
   untouched chronological holdout when possible. Prior evaluations have
   already consulted Dec 2025–May 2026; that period cannot honestly be called
   untouched going forward.

**Practical ensemble admission criterion:** a candidate must have acceptable
standalone prediction quality **and** lower the current ensemble error on
separate chronological out-of-time folds. Low model correlation alone is
insufficient. Limit near-duplicates, avoid over-weighting one feature family,
and report both MdAPE and MAPE; optimizing only the median can worsen tail
errors.

## Code and artifacts

- `models/ensemble_bagging.py`: reproducible 47-prioritized feature-bank,
  uniform / 4× weighted text subsampling, row bootstrap and 80% subbag.
- `ensemble_feature_bagging_pilot.py`: opt-in one-month XGBoost pilot with
  train-only vocabulary, one full-context feature fit, member prediction
  caches, metrics, and a manifest of terms/rows for each bag.
- `models/ensemble_blending.py`: fixed-weight **log-price** blending with
  matched-key and positive-price checks; no leakage-prone weight fitting.
- `ensemble_starter.ipynb`: notebook entry point and opt-in pilot controls.

Pilot output path: `outputs/ensemble_feature_bagging_smoke/`. Prediction
tables use `month, ml_num, member, sold_price, predicted_price` and must
not be mixed with differently configured folds.

**No automated proof or performance claim** is implied by adding the roster.
Run the tests and pilot on your local housing data before promoting members.
