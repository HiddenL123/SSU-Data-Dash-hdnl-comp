
# Housing Price Model — Development Roadmap



Note: This is an AI generated TODO. We will replace this

## Objective

Predict residential property sale prices in the Toronto area using historical listing data.

**Competition tracks:**

- **Off-market:** Predict sold price without access to listing price.
- **List-aware:** Predict sold price with listing price available.

**Target:** `sold_price`, modelled on the logarithmic scale.

**Validation:** Expanding-window chronological validation, December 2025 – May 2026.

**Competition prediction period:** June – September 2026.

**Primary evaluation metric:** Median Absolute Percentage Error (MdAPE).

**Secondary metrics:** MAPE, signed prediction bias, and error distributions by property type, geography and month.

---

## Current Model Architecture

### Off-market

\[
\widehat{\log P}
= f_{\mathrm{HGB}}(X,\text{location},\text{text})

+ m_{\mathrm{ARIMA}}(t)
  \]

An alternative, previously tested architecture adds a residual spatial GAM.

### List-aware

Use the off-market prediction as a baseline and apply a second-stage Ridge adjustment informed by listing price and property attributes.

### Other components investigated

- Spatial GAM using a two-dimensional tensor-product spline.
- Temporal spline regression.
- ARIMA(1,1,0) and ARIMA(0,1,1).
- Automatic Elastic Net interaction selection.
- Fixed Ridge interactions using historical out-of-time residual discovery.

---

## Phase 1 — Data and Feature Engineering

- [X] Establish data loading and preprocessing pipeline.
- [X] Implement separate off-market and list-aware feature handling.
- [X] Engineer numerical and categorical property features.
- [X] Extract geographical coordinates.
- [X] Extract TF-IDF features from listing remarks.
- [X] Apply Elastic Net-based text feature selection.
- [X] Create chronological training and validation splits.
- [ ] Audit missing values and categorical encoding, particularly `garage`.
- [ ] Review additional useful property features and transformations.
- [ ] Evaluate feature importance and redundancy.

## Phase 2 — Baseline Modelling

- [X] Implement baseline regression models.
- [X] Implement HistGradientBoostingRegressor (HGB).
- [X] Implement spatial GAM residual correction.
- [X] Implement temporal spline correction.
- [X] Implement ARIMA temporal correction.
- [X] Implement two-stage list-aware adjustment.
- [X] Test HGB with longitude and latitude directly.
- [X] Compare HGB + GAM + ARIMA against geographic HGB + ARIMA.
- [ ] Test geographic HGB + residual GAM + ARIMA as a final spatial ablation.
- [ ] Investigate ARIMA convergence warnings and sensitivity.
- [ ] Consolidate the simplest well-performing baseline.

### Reference performance

| Architecture                           | Off-market MAPE | Off-market MdAPE | List-aware MdAPE |
| -------------------------------------- | --------------: | ---------------: | ---------------: |
| HGB + GAM + ARIMA                      |          14.02% | **11.23%** |            4.59% |
| Geographic HGB + ARIMA                 |          14.12% |           11.30% |  **4.58%** |
| HGB + GAM + ARIMA + fixed interactions |          14.09% |           11.32% |  **4.58%** |

These are historical walk-forward validation results, not hidden competition scores.

---

## Phase 3 — Stronger Gradient-Boosting Models

**Current priority: High**

- [ ] Implement CatBoost property model.
- [ ] Test CatBoost with native categorical features.
- [ ] Implement LightGBM property model.
- [ ] Implement XGBoost property model.
- [ ] Compare all three against sklearn HGB using identical chronological validation.
- [ ] Tune promising models using historical chronological folds.
- [ ] Compare geographic prediction errors and performance across property types.
- [ ] Select the strongest standalone property model.

Keep ARIMA and other residual corrections consistent during algorithm comparisons to isolate the effect of the underlying learner.

---

## Phase 4 — Interaction Modelling

**Current priority: Low to medium**

- [X] Implement automatic Elastic Net interaction discovery.
- [X] Investigate property × time and property × location effects.
- [X] Build separate historical interaction-discovery notebook.
- [X] Generate out-of-time residual predictions.
- [X] Evaluate interaction families using historical chronological validation.
- [X] Identify property attributes × time as a promising candidate.
- [X] Evaluate combinations with municipality, seasonality and lot area.
- [X] Implement fixed Ridge interactions.
- [X] Compare fixed interactions against the no-interaction baseline.
- [ ] Examine improvement separately across discovery folds.
- [ ] Train production interactions on out-of-time residuals rather than in-sample residuals.
- [ ] Retest fixed interactions using the final selected property model.

**Decision:** Do not enable interactions by default until they demonstrate a consistent improvement in the integrated model.

---

## Phase 5 — Validation and Diagnostics

**Current priority: High**

- [X] Implement expanding-window monthly validation.
- [X] Calculate MAPE, MdAPE and signed bias.
- [X] Visualize geographical prediction errors.
- [X] Analyze monthly prediction trends.
- [X] Analyze property type × month error patterns.
- [X] Generate historical interaction-discovery leaderboards.
- [ ] Implement paired comparisons between competing model predictions.
- [ ] Calculate confidence intervals for paired MdAPE improvements.
- [ ] Investigate the highest-error observations.
- [ ] Compare models separately by house type, municipality and month.
- [ ] Evaluate prediction stability across chronological folds.
- [ ] Ensure model selection and hyperparameter tuning do not leak future outcomes.

---

## Phase 6 — Ensemble Modelling

**Current priority: Medium, after model benchmarking**

- [ ] Save out-of-time predictions from HGB, CatBoost, LightGBM and XGBoost.
- [ ] Measure prediction-error correlation between models.
- [ ] Test simple equal-weight averaging on log-price predictions.
- [ ] Test weighted averaging using historical validation.
- [ ] Investigate stacking only if simpler averaging is insufficient.
- [ ] Evaluate whether ensemble gains persist across time periods.
- [ ] Keep the ensemble only if it outperforms the strongest standalone model.

Do not use a fixed prediction-correlation threshold such as 0.99 as the sole admission criterion. Model diversity is useful only when it translates into better out-of-time performance.

---

## Phase 7 — Finalization and Submission

- [ ] Select final off-market architecture.
- [ ] Select final list-aware architecture.
- [ ] Freeze feature definitions and hyperparameters.
- [ ] Refit final models on all eligible historical training data.
- [ ] Confirm no prohibited features or future sale outcomes enter predictions.
- [ ] Generate and validate submission format.
- [ ] Check all predicted prices are finite and positive.
- [ ] Document model design and feature engineering.
- [ ] Document validation methodology and final results.
- [ ] Prepare model interpretation and competition presentation.

---

## Immediate Next Steps

1. **Benchmark CatBoost and LightGBM against HGB** with identical features and validation periods.
2. Test whether retaining a residual spatial GAM benefits the geographical HGB architecture.
3. Examine paired prediction errors to determine whether the competing models complement each other.
4. Tune the strongest model using chronological validation.
5. Consider a lightweight ensemble of the best-performing models.
6. Revisit out-of-time-trained interactions only if there is time and evidence of further improvement.

## Modelling Principle

Prioritize improvements that generalize to future sales over increasingly complex model structures.

A simpler model with stable out-of-time performance is preferable to a more complex model with marginal, inconsistent validation gains.
