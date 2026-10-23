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


## Saving the growing ensemble model pool

Track-scoped storage is now active. New fitted member collections should use
`saved_models/off_market/<run_name>/` or
`saved_models/list_aware/<run_name>/`; the existing flat folder is **legacy**,
left untouched for the original 6-month XGBoost snapshots.

The opt-in `ensemble_starter.ipynb` pilot can set
`SAVE_FITTED_ENSEMBLE_MODELS=True` to persist a **reusable bundle** containing
one fitted feature provider (historical comparable index + training-fitted text
vocabulary), independently fitted XGBoost members, selected columns, log
target offsets and member seeds. Avoid saving only XGBoost trees or prediction
CSVs: those are insufficient to reconstruct full ensemble inference.
The bundle saves with a deterministic experiment-fingerprint name under
`saved_models/off_market/`, without overwriting older experiments.

Load with `load_model(name, track="off_market")`, call
`predict_member(homes, "selected_47_reference")` for one member or
`predict_members(homes)` for a prediction table. The bundled `predict()`
offers an equal-weight geometric blend by default; it is **not** a frozen,
validated production ensemble. The list-aware pool will be built separately
with its own target and historical validation.

## Shrinkage versus complementarity experiment (implemented)

Run the **shrinkage-versus-blending** cell in `ensemble_starter.ipynb`
after the Matérn confirmation CSV exists. This experiment fits **no new
models**, selects **no weights**, and compares exactly matched September–
November 2025 homes from `confirmation_predictions.csv`.

- Reference: same XGBoost splines+geography baseline, Tensor GAM, tuned
  spatial Matérn and tuned space-time Matérn.
- Controls: `gam_half_strength`, `spatial_half_strength`,
  `spacetime_half_strength` (halve the **effective** log correction).
- Blends: `gam_spatial_50_50`, `gam_spacetime_50_50`; and for each
  `gam_{kind}_blend_half_strength`, a further halving of the averaged
  correction.
- Evaluate paired MdAPE/MAPE delta against **both singly-shrunk controls**,
  the original GAM and Matérn, and the no-correction XGBoost. Include
  month-stratified paired bootstrap intervals, homes improved, monthly scores,
  and log-correction correlation.

Outputs: `outputs/ensemble_shrinkage_vs_spatial_blending/`: `setup.json`,
`pooled.csv`, `monthly.csv`, `pairwise.csv`,
`correlations.csv` and `predictions.csv`.

**Interpretation:** if a 50/50 GAM–Matérn blend beats only the unshrunk
models, not the individually-shrunk controls, its gain may largely be
regularization. If it consistently beats both shrunk controls on *separate*
chronological data, that is stronger evidence of complementary signals.
The earlier Sep–Nov folds have already been inspected; this diagnostic by
itself is not independent confirmation.

## Proposed iterative ensemble factory (design, not yet implemented)

The long-term factory can use a bounded iterative/evolutionary AutoML search
with candidate configurations like:

```yaml
track: off_market  # list_aware is an independent search
base_family: xgboost
text:
  representation: binary
  vocabulary_source: training_fold_only
  bag_strategy: selected_plus | selected_weighted | random_uniform | full
  selected_47_prior: 4.0
  term_count: 150
  feature_seed: 142
rows:
  sampling: bootstrap | subbag | full
  sample_fraction: 1.0
  bootstrap_seed: 258
spacetime_addons:
  - spacetime_splines
  - geo_context
model:
  tuning_engine: optuna
  hyperparameters: ... # only valid for the chosen model family
validation:
  fold_mode: chronological
  train_history: expanding
  track_specific: true
```

**Factory cycle:**
1. Sample feature-text representation, bag seed, row-bootstrap settings,
   eligible spatial features, model family and hyperparameter search space.
   Respect incompatible feature combinations and keep genuinely prospective
   comparable-sale context unchanged by row bootstrapping.
2. Quickly screen the model on earlier chronological folds and a fixed,
   identical property cohort; discard clearly broken or dominated models.
   Spend Optuna trials only on survivors, with search/pruning budgets.
3. Save fitted models + transformers, reproducible member configuration,
   historical cutoff, feature terms, row seed, and aligned out-of-time
   predictions (no future sales for feature fitting or calibration).
4. Score **marginal ensemble gain**, not only individual MdAPE. Optimize
   nonnegative log-price blend weights or a constrained forward-selection
   step using *separate* chronological selection folds.
5. If a new candidate is redundant, consider removing/replacing correlated
   old members and re-evaluating the **whole** ensemble. Admit it only
   for a stable, thresholded incremental gain across several folds.
6. Retain an elite set and a smaller exploration set, then sample a new
   generation. Bound model count, training work, disk size and Optuna
   budget; stop when independent improvement saturates.

**Critical safeguards:**

- With thousands of adaptive trials, overfitting a handful of validation
  months is the principal danger. Separate hyperparameter tuning,
  ensemble selection, and eventual untouched evaluation. The already
  repeatedly consulted Dec 2025–May 2026 months cannot be considered fresh
  holdout evidence.
- Keep identical chronological OOF home IDs for every candidate. Resampling
  training rows may duplicate sales **inside the estimator fit**, but
  comparable-sale context and vocabulary must be fitted on appropriate
  chronological *historical* records, not bootstrap-generated duplicates.
- Random feature masks can remove all useful predictors or violate spatial
  feature requirements; allow an always-on structured core and enforce
  valid dependencies. Use training-only vocabulary construction.
- A single top-N cutoff on standalone MdAPE tends to select correlated
  members. A weaker but complementary model may improve the **blend**
  more. Diversity (including residual correlation) is a screening
  diagnostic, not the objective itself.
- Treat MdAPE and MAPE as separate safeguards. MdAPE is not smooth and
  selecting on it repeatedly can harm extreme prices; use constrained
  objectives and report paired uncertainty.
- The model factory must ultimately **refit selected members** on the full
  eligible history for hidden-test predictions; save the fitted feature
  providers, not only model parameters. Off-market and list-aware have
  different feature availability/targets, so their ensemble pools never mix.

This outlines the eventual factory; the current notebook only implements
fixed-prediction shrinkage comparisons and the initial feature-bagging
pilot. Do not accidentally label the proposal as a running evolutionary
search engine.

## AutoML factory v1 (now implemented)

`ensemble_automl_factory.py` and `models/automl_search.py` implement
a **bounded, opt-in starter**, not the complete multi-family evolutionary
optimizer. Execute `RUN_AUTOML_FACTORY=True` in `ensemble_starter.ipynb`.
Install the optional dependency using `pip install optuna`.

- Candidate generator: deterministic random combinations of selected 47,
  selected+extra, weighted, uniform or full binary text; independent
  bootstrap/subbag/all-row fitting; valid combinations of spatial splines,
  geo context, market or no add-on.
- Historical comparable features and training-only BoW vocabulary are
  fitted on all eligible chronological training homes **before** fitting
  XGBoost on bootstrap/subbag rows. Model families beyond XGBoost are
  future work.
- Cheap gross-error check (`screen_max_mdape=22%`), then two Optuna trials
  per surviving candidate by default. This guard only rejects
  catastrophically poor models; it is not a general ranking threshold.
- Optuna tuning: Mar–Apr 2025. Ensemble selection and a fixed nonnegative
  geometric-blend weight grid: May–Jun 2025. The member must improve
  the weighted `0.65*MdAPE + 0.35*MAPE` objective by at least
  0.02 percentage points without more than 0.03 points of MAPE regression.
- **Separate one-time shadow audit:** Jul–Aug 2025, opened only by the
  explicit `RUN_AUTOML_SHADOW_AUDIT=True` notebook cell after the model
  roster and weights are frozen. It locks the roster against adding further
  generations in that directory. These historic months were consulted in
  other experiments; this is not a new untouched test.
- Persist SQLite Optuna studies, member predictions, roster, member weights,
  event history, cohort-scoped train settings, selected-ensemble predictions
  and shadow-audit predictions. Optional accepted-member snapshots contain
  one fitted feature provider and the model's terms, columns and log target
  offset, saved in `saved_models/off_market/`.
- Resume with the same settings and output directory without redoing
  already-processed candidates. **Increase `generations` before opening
  the shadow audit** to extend the same fixed-budget experiment; other
  configuration changes require a new output directory because selection
  comparisons would be incompatible. After audit, the roster is locked
  against further selection in that directory.

Initially run **1 generation × 2 candidates × 2 Optuna trials**.
Start only one factory process per output directory; the v1 roster writer is
not a transactional multi-worker scheduler.

Remaining before a full final factory:

1. Candidate-replacement/removal and adaptive pool size, beyond the current
   additive maximum of `max_members`.
2. Independent LightGBM/CatBoost members and integrated model-specific
   Matérn/Tensor GAM residual correctors.
3. Nested rolling evaluation that tracks adaptation of the selection set
   across many generations and a fresh final holdout if one becomes available.
4. Refit the final selected architectures on all legally available data with
   700-tree/full-strength parameters before submission.
5. Save a single weighted meta-ensemble of all final fitted members and
   reconcile shared versus separate historical feature transformers.

### Crash-safe pause and resume

The factory now treats its output directory as a **durable checkpoint**,
not a disposable report folder. Run **one writer process** per output
directory and stop it using Jupyter's Interrupt Kernel, Ctrl+C, or ordinary
process termination; rerun the same call with unchanged data/configuration.

Recovery boundaries:

| Interrupted operation | What survives | What may need repeating |
| --- | --- | --- |
| Completed candidate / ensemble admission | Atomic `roster.json` includes BOTH the membership decision and event | Nothing from that candidate's Optuna search or selection, except missing optional fitted-model snapshot serialization |
| Between roster write and event export | `roster.json` is authoritative | `events.json` regenerated from roster history |
| Completed Optuna trial | SQLite study retains its state and score | Nothing |
| Partly completed Optuna trial | Previously completed per-month objective values under `optuna/fold_scores/` and the trial's suggested parameters | Currently running fold's fit; an orphan `RUNNING` trial is failed/re-queued with the same known parameters |
| Completed candidate-month prediction | Atomic cache keyed by full parameter fingerprint | Nothing |
| During model binary saving | `save_model` stages binaries before atomic directory replacement | Only the interrupted snapshot fit/save; missing accepted snapshots are recovered on restart |
| During the shadow audit | Early, durable `shadow_audit_lock.json` with `pending` state and candidate prediction caches | Incomplete audit folds; final `complete` status is written last |

`progress.json` is the last recorded stage (not the source of truth).
`manifest.json` and `roster.json` are authoritative. Most JSON and
CSV outputs are written to temporary files in the same directory and
atomically replaced; an abrupt stop may leave a harmless `.tmp` file.
The OS-backed `.run.lock` **does not need to be deleted** after a crash:
the operating system releases it. Never run two sessions against the same
directory at once, and keep the output on a local disk.

**Limitation:** a single XGBoost `fit()` call interrupted mid-fit starts
again, since incremental boosting-round checkpoints are not enabled.
Likewise, unsaved RAM-only fitted feature transformers are recreated for
an interrupted fold. Progress is preserved at *completed training
folds/trials/candidate* boundaries, not every machine instruction.
Optuna's sampler may generate a different sequence of subsequent proposals
after a process restart; completed trials and their scores are not lost.

Once the audit is opened (even if it is interrupted in the middle), the
selected roster cannot be extended in that directory. Keep audit disabled
until the desired number of generations is complete. To change budgets
other than `generations` or the selection protocol, use a fresh output
directory to avoid incompatible cached scores.
