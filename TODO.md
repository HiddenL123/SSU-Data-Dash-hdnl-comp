
# Housing Price Model — Development Roadmap

**Last updated:** October 18, 2026

## Objective

Predict residential property sale prices in the Toronto area using historical real estate listing data.

**Competition tracks:**

- **Off-market:** Predict sold price without listing price.
- **List-aware:** Predict sold price with listing price available.

**Prediction target:** `sold_price`, generally modelled on the logarithmic scale.

**Validation:** Expanding-window chronological validation, December 2025 – May 2026, with 10,069 evaluated homes per track.

**Competition prediction period:** June – September 2026.

**Primary metric:** Median Absolute Percentage Error (MdAPE).

**Secondary metrics:** MAPE, signed prediction bias, month-by-month performance, error distributions, and paired model comparisons.

### Development targets

| Track      | Current best MdAPE | Goal | Stretch goal |
| ---------- | -----------------: | ---: | -----------: |
| Off-market |   **10.02%** | 9.5% |         9.0% |
| List-aware |    **4.54%** | 4.3% |         4.0% |

These are aspirational development targets, not externally established performance benchmarks. They are not strict prerequisites for ensembling.

---

## Current Model Architecture

### Off-market

The strongest tested configuration is:

\[
\widehat{\log P}
=f_{\mathrm{CatBoost}}(X,t)+m_{\mathrm{ARIMA}}(t).
\]

- CatBoost predicts log sale price using property, geographical, textual and temporal features.
- ARIMA(1,1,0) optionally models the remaining monthly median log-price residuals.
- The strongest configuration currently achieves **10.02% MdAPE**.
- Time-aware CatBoost without ARIMA achieves **10.05% MdAPE**, making it a competitive, simpler alternative.

### List-aware

The currently tested architecture uses:

\[
\widehat_
=========

\widehat{\log P}_{off}
+
h_{\mathrm{Ridge}}
(\log L-\widehat{\log P}_{off},X).
\]

The best recorded list-aware result is **4.54% MdAPE**, from an earlier HGB-based architecture.

Improving the off-market model has not consistently improved list-aware performance. The list-aware methodology therefore needs a dedicated audit and comparison against direct prediction alternatives.

### Planned modular architecture

The intended interface is:

```python
model = make_model(track, modelspecs=None)
model.fit(train)
predictions = model.predict(homes)
```

Each `modelspecs` entry contains:

```python
(ModelClass, "feature_set", {"parameter": value})
```

The architecture will support configurable sequential residual components and track-specific prediction strategies, while preserving existing implementations as legacy code.

---

## Phase 1 — Data and Feature Engineering

**Status: Core implementation complete; improvements ongoing**

- [X] Establish data loading and preprocessing.
- [X] Implement separate off-market and list-aware feature handling.
- [X] Engineer numerical and categorical property features.
- [X] Incorporate longitude and latitude.
- [X] Extract TF-IDF features from listing remarks.
- [X] Implement Elastic Net-based text feature selection.
- [X] Identify a fixed vocabulary of informative listing terms.
- [X] Implement optional temporal features for CatBoost, including elapsed months and seasonal encoding.
- [X] Establish chronological training and validation splits.
- [ ] Audit missing values, especially `garage` and other inconsistently encoded property fields.
- [ ] Audit categorical handling and unseen categories across chronological folds.
- [ ] Review additional property features, transformations and ratios.
- [ ] Evaluate CatBoost feature importance and SHAP explanations.
- [ ] Investigate feature redundancy and possible information leakage.
- [ ] Consider additional feature representations for property condition from listing remarks.
- [ ] Test whether different feature sets benefit the two competition tracks independently.

---

## Phase 2 — Modular Architecture Refactor

**Priority: Immediate**

**Goal:** Make experiments configurable without repeatedly modifying individual model classes.

### Core interface

- [X] Design the proposed `modelspecs` format.
- [X] Define the intended `OffMarketModel` and `ListAwareModel` interfaces.
- [X] Prepare a Codex implementation specification.
- [ ] Implement `models/model.py` with `OffMarketModel`, `ListAwareModel`, and `make_model()`.
- [ ] Implement configurable sequential residual fitting on log-price predictions.
- [ ] Implement a consistent model-component interface.
- [ ] Implement a feature-provider registry, such as `Xp`, `Xpt`, and `date_listed`.
- [ ] Ensure every fitted component retains its own preprocessing state.
- [ ] Support optional constructor parameters using keyword arguments.
- [ ] Define a stable default model specification.

### Model migration

- [ ] Adapt CatBoost to the new interface.
- [ ] Adapt LightGBM to the new interface.
- [ ] Adapt XGBoost to the new interface.
- [ ] Adapt ARIMA to the new interface.
- [ ] Adapt the list-aware Ridge correction.
- [ ] Preserve previous implementations in `models/legacy/`.
- [ ] Update package exports and imports.
- [ ] Ensure `run.py` works unchanged with the default model specifications.
- [ ] Allow notebooks and validation utilities to pass custom `modelspecs`.

### Verification

- [ ] Test CatBoost-only and CatBoost + ARIMA specifications.
- [ ] Test both competition tracks.
- [ ] Verify that residual components are fitted and added on the correct scale.
- [ ] Verify chronological separation in list-aware historical training.
- [ ] Compare predictions against equivalent legacy implementations.
- [ ] Test the submission and validation workflows after migration.

**Decision:** Complete and verify the core refactor before substantially expanding the model library. Do not alter statistical behaviour merely to simplify the code.

---

## Phase 3 — Baseline and Gradient-Boosting Models

**Priority: High**

### Completed

- [X] Implement initial regression baselines.
- [X] Implement HGB property modelling.
- [X] Implement spatial GAM correction.
- [X] Implement temporal spline correction.
- [X] Implement ARIMA residual correction.
- [X] Test geographical coordinates directly inside HGB.
- [X] Implement CatBoost property modelling.
- [X] Test native categorical features and TF-IDF with CatBoost.
- [X] Test CatBoost with and without explicit time features.
- [X] Test time-aware CatBoost with and without ARIMA.
- [X] Establish CatBoost as the strongest tested off-market property model.

### Experimental results

| Architecture                           |  Off-market MAPE | Off-market MdAPE | List-aware MdAPE |
| -------------------------------------- | ---------------: | ---------------: | ---------------: |
| HGB + GAM + ARIMA                      |           14.02% |           11.23% |            4.59% |
| Geographic HGB + ARIMA                 |           14.12% |           11.30% |            4.58% |
| HGB + GAM + ARIMA + fixed interactions |           14.09% |           11.32% |            4.58% |
| CatBoost + ARIMA, no explicit time     |           13.03% |           10.45% |            4.55% |
| CatBoost + time, no ARIMA              |           12.76% |           10.05% |            4.71% |
| **CatBoost + time + ARIMA**      | **12.67%** | **10.02%** |            4.73% |

The best recorded list-aware MdAPE remains **4.54%** from the earlier HGB + text + temporal spline configuration.

### Remaining experiments

- [ ] Verify the migrated LightGBM implementation and benchmark it.
- [ ] Verify the migrated XGBoost implementation and benchmark it.
- [ ] Compare CatBoost, LightGBM and XGBoost using identical chronological folds and comparable feature sets.
- [ ] Investigate Yggdrasil Decision Forests as a fourth tree-based competitor.
- [ ] Consider GPBoost for spatial and temporal dependence.
- [ ] Tune the strongest standalone model using historical chronological folds.
- [ ] Test MAE on log-price rather than RMSE as the training objective.
- [ ] Test recency-weighted training.
- [ ] Compare errors by property type, municipality, geography and prediction month.
- [ ] Save paired out-of-time predictions for later ensemble experiments.

**Decision:** CatBoost is the current benchmark. Do not assume another boosting library is better without direct validation.

---

## Phase 4 — List-Aware Model Redesign

**Priority: Very high**

**Goal:** Determine whether the current two-stage Ridge architecture is restricting performance.

### Existing observations

- [X] Implement the two-stage off-market + Ridge adjustment.
- [X] Train the adjustment using historical out-of-time off-market predictions.
- [X] Observe that substantial off-market improvements have produced limited list-aware gains.
- [X] Identify systematic negative signed bias in recent time-aware CatBoost configurations.
- [X] Formulate direct prediction and sale-to-list ratio alternatives.

### Required experiments

- [ ] Establish `sold_price = list_price` as a simple benchmark.
- [ ] Train direct CatBoost to predict `log(sold_price)` using listing price as an input.
- [ ] Train CatBoost to predict `log(sold_price / list_price)`.
- [ ] Compare both against the existing two-stage Ridge model.
- [ ] Audit Ridge feature construction, scaling and category alignment.
- [ ] Audit the historical out-of-time predictions used to fit Ridge.
- [ ] Check for differences between Ridge training inputs and final prediction inputs.
- [ ] Investigate systematic list-aware underprediction and calibration.
- [ ] Compare models by listing-price range, property type, municipality and month.
- [ ] Test whether temporal features help the direct or ratio-based list-aware models.
- [ ] Make the list-aware strategy configurable through the modular interface.
- [ ] Select the strongest independently validated list-aware architecture.

### Candidate architectures

**A. Existing two-stage adjustment**

\[
\widehat{\log P}=\widehat{\log P}_{off}+h_{\mathrm{Ridge}}(X,L).
\]

**B. Direct CatBoost**

\[
\widehat{\log P}=f_{\mathrm{CatBoost}}(X,\log L,t).
\]

**C. Sale-to-list ratio model**

\[
\widehat
========

\log L+g_{\mathrm{CatBoost}}(X,\log L,t).
\]

**Decision:** Do not assume the off-market winner should also be the foundation of the list-aware winner. Optimize the tracks separately where justified.

**Target:** Improve list-aware MdAPE from 4.54% toward 4.3%.

---

## Phase 5 — ARIMA and Temporal Modelling

**Priority: Medium**

### Completed

- [X] Implement ARIMA(1,1,0) on monthly median log residuals.
- [X] Compare ARIMA(1,1,0) and ARIMA(0,1,1).
- [X] Investigate maximum-likelihood convergence warnings.
- [X] Print optimizer convergence status, iteration counts and parameter estimates.
- [X] Observe that some state-space MLE fits fail to converge.
- [X] Observe relatively consistent AR(1) coefficients around −0.5 to −0.6.
- [X] Test the Burg estimator sufficiently to confirm its differencing warning is expected.
- [X] Compare time-aware CatBoost with and without the current ARIMA correction.

### Remaining experiments

- [ ] Complete Burg estimation validation and record its prediction metrics.
- [ ] Test Yule–Walker estimation.
- [ ] Compare numerical optimizers for state-space MLE.
- [ ] Test rescaling residuals before likelihood estimation.
- [ ] Test ARIMA(1,0,0) to investigate whether differencing is necessary.
- [ ] Compare ARIMA(1,1,0), (1,0,0), (0,1,0), and a constant-mean baseline.
- [ ] Inspect ACF/PACF and stationarity of monthly residuals.
- [ ] Compare monthly mean versus median residual aggregation.
- [ ] Check forecast corrections several months beyond the historical training period.
- [ ] Compare monthly validation performance with and without ARIMA.
- [ ] Measure paired prediction differences between ARIMA estimators.
- [ ] Determine whether ARIMA's small improvement justifies its complexity.

### Key question

Time-aware CatBoost without ARIMA achieves 10.05% MdAPE, while adding the current ARIMA improves this to 10.02%.

Is this improvement consistent and generalizable, or is it within validation variability?

**Decision:** Numerical convergence is useful but not the final objective. Keep ARIMA only if its forecast contribution is reliable and meaningful.

---

## Phase 6 — Market-Level Modelling

**Priority: High — New experiment**

**Goal:** Test whether historical housing-market conditions contain predictive information not fully captured by CatBoost's property and date features.

### Market-level features to investigate

- [ ] Historical rolling price trends and housing price indices.
- [ ] Recent sale-to-list ratios.
- [ ] Historical monthly sales volume.
- [ ] Municipality-level price momentum.
- [ ] Property-type-specific market trends.
- [ ] Recent market volatility or dispersion.
- [ ] Property-adjusted price indices or residual-based market signals.

Avoid relying solely on raw median sold prices, which may be distorted by changes in the mix of properties sold.

### Main experiments

- [ ] **M1 — CatBoost + market-level modelling, without ARIMA.**
- [ ] **M2 — CatBoost + market-level modelling + ARIMA.**
- [ ] Compare both against time-aware CatBoost and CatBoost + ARIMA.
- [ ] Compare market indicators used as CatBoost features versus a separate residual correction model.
- [ ] Determine whether market and ARIMA components provide complementary signals.
- [ ] Evaluate whether market effects differ across locations and property types.
- [ ] Test how market signals behave when forecasting multiple future months.

### Experimental matrix

| Configuration       | ARIMA | Market-level component | Status       |
| ------------------- | ----- | ---------------------- | ------------ |
| Time-aware CatBoost | No    | No                     | 10.05% MdAPE |
| Time-aware CatBoost | Yes   | No                     | 10.02% MdAPE |
| Time-aware CatBoost | No    | Yes                    | Pending      |
| Time-aware CatBoost | Yes   | Yes                    | Pending      |

### Leakage prevention

All market-level statistics must use information available before the prediction cutoff.

For example, March 2026 forecasts must not use sale outcomes from March 2026 or later.

Historical feature construction must follow the same chronological restrictions as validation.

**Decision:** Run CatBoost + market-level information first, followed by CatBoost + market-level information + ARIMA.

---

## Phase 7 — Interaction Modelling

**Priority: Low**

### Completed

- [X] Implement automatic Elastic Net interaction discovery.
- [X] Investigate property × time and property × location effects.
- [X] Build a separate historical interaction-discovery notebook.
- [X] Generate historical out-of-time residual predictions.
- [X] Evaluate interaction families through chronological validation.
- [X] Identify property attributes × time as a promising candidate.
- [X] Evaluate municipality, seasonality and lot-area combinations.
- [X] Implement fixed Ridge interactions.
- [X] Observe that the integrated fixed-interaction architecture did not improve MdAPE.

### Remaining

- [ ] Inspect improvements separately across historical discovery folds.
- [ ] Fit residual interactions using out-of-time rather than in-sample predictions.
- [ ] Reevaluate interactions after selecting the final property model.
- [ ] Determine whether CatBoost already captures the relevant nonlinear interactions.

**Decision:** Do not enable additional interaction models by default without demonstrable gains. Prioritize market-level modelling and list-aware improvements instead.

---

## Phase 8 — Validation and Model Diagnostics

**Priority: High**

### Completed

- [X] Implement expanding-window monthly validation.
- [X] Calculate MAPE, MdAPE and signed bias.
- [X] Generate bootstrap confidence intervals for individual model metrics.
- [X] Visualize geographical prediction errors.
- [X] Analyze monthly predicted versus actual prices.
- [X] Analyze house type × month signed-error patterns.
- [X] Build historical interaction-discovery comparisons.

### Remaining

- [ ] Store individual out-of-time predictions with stable observation identifiers.
- [ ] Implement paired model comparisons using the same homes.
- [ ] Calculate bootstrap confidence intervals for paired MdAPE differences.
- [ ] Compare performance month by month.
- [ ] Evaluate performance by property type, municipality, community and price range.
- [ ] Inspect extreme-error observations.
- [ ] Examine systematic overprediction and underprediction.
- [ ] Evaluate how accuracy changes with forecast horizon.
- [ ] Test model stability across chronological folds.
- [ ] Separate model-development folds from final model-selection evaluation where feasible.
- [ ] Prevent leakage during feature selection, model tuning, residual correction and market aggregation.

**Important:** Repeated experimentation against the same six validation months introduces model-selection bias. Small observed improvements should not automatically be treated as genuine improvements.

---

## Phase 9 — Ensemble Modelling

**Priority: Medium — After standalone benchmarking**

- [ ] Save out-of-time predictions from selected CatBoost, LightGBM, XGBoost and other candidate models.
- [ ] Measure prediction-error correlations.
- [ ] Compare simple averaging in log-price space.
- [ ] Test weighted averaging.
- [ ] Optimize ensemble weights on historical development folds.
- [ ] Investigate stacking if simpler averaging is insufficient.
- [ ] Compare ensemble gains across months and market segments.
- [ ] Investigate whether less accurate but diverse models improve the ensemble.
- [ ] Build a configurable `EnsembleModel` within the modular architecture.
- [ ] Keep the ensemble only if it improves the strongest standalone model on out-of-time data.

Do not select ensemble members using prediction correlation alone. Their actual contribution to validation accuracy matters.

Do not delay ensembling indefinitely while chasing arbitrary standalone accuracy targets.

---

## Phase 10 — Finalization and Submission

- [ ] Select final off-market architecture.
- [ ] Select final list-aware architecture independently.
- [ ] Resolve or document any remaining ARIMA numerical issues.
- [ ] Freeze feature definitions and hyperparameters.
- [ ] Refit final models on all eligible historical data.
- [ ] Verify that no prohibited features enter prediction.
- [ ] Validate market-feature availability at the hidden prediction dates.
- [ ] Confirm predictions are finite and positive.
- [ ] Generate the required submission files.
- [ ] Validate submission format and row alignment.
- [ ] Document final model architectures.
- [ ] Document experiments and statistical findings.
- [ ] Document validation methodology and limitations.
- [ ] Prepare model interpretation and competition presentation.

---

## Immediate Next Steps — Recommended Order

### 1. Complete modularization

Implement and verify the new `models/model.py` architecture, preserve legacy implementations and maintain compatibility with `run.py`.

### 2. Redesign list-aware modelling

Run the direct CatBoost and sale-to-list ratio experiments. Audit the Ridge correction and its negative prediction bias.

### 3. Run market-level experiments

Compare time-aware CatBoost + market-level information, both with and without ARIMA.

### 4. Finish ARIMA diagnostics

Record Burg results, compare against Yule–Walker and alternative ARIMA orders, and establish whether ARIMA is worth retaining.

### 5. Benchmark additional boosting libraries

Compare LightGBM and XGBoost against CatBoost using identical chronological validation.

### 6. Tune and refine feature engineering

Focus on the strongest model architectures, their objectives, time weighting and informative property features.

### 7. Evaluate model diversity and begin ensembling

Save paired predictions, measure complementarity and test simple blends before more complex ensemble optimization.

### 8. Final validation and submission

Freeze models, perform leakage checks and generate final competition outputs.

---

## Modelling Principles

1. **Chronological validity first.** Never allow future outcomes to influence historical predictions or features.
2. **Preserve controlled comparisons.** Change one modelling factor at a time where possible.
3. **Prioritize out-of-time performance.** Improvements must be useful for future sales, not merely the current validation period.
4. **Separate competition tracks.** The best off-market architecture need not be the best list-aware architecture.
5. **Prefer useful complexity.** New components should demonstrate measurable gains.
6. **Maintain reproducibility.** Record specifications, metrics, dates, model versions and predictions for every important experiment.
7. **Do not overfit the leaderboard or validation period.** Small gains require careful statistical interpretation.

**Current overall direction:** Modularize the code, improve list-aware modelling, test market-level information, then benchmark and ensemble the strongest property models.
