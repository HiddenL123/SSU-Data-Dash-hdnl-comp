**Initial regression models and feature engineering**

- E01 — Initial generalized linear regression baseline

  - Hypothesis: A relatively simple regression model can explain housing prices using structured property attributes.
  - Test: Train a baseline GLM and evaluate historical housing predictions.
  - Observation: Off-market MAPE was approximately 19.75%.
  - Conclusion: Linear relationships alone were insufficient. More flexible modelling and feature engineering were warranted.
- E02 — Elastic Net + Spatial GAM + Temporal Splines

  - Hypothesis: Separating property attributes, geographical price differences, and temporal market changes would improve prediction accuracy.
  - Test: Model log-price using Elastic Net, a two-dimensional spatial GAM, and a temporal spline.
  - Observation:
    - Off-market: MAPE 17.79%, MdAPE 14.01%.
    - List-aware: MAPE 6.33%, MdAPE 5.19%.
  - Conclusion: The combined model improved substantially over the initial regression baseline. Modelling nonlinear geographical and temporal effects appeared useful, although their individual contributions were not isolated.
- E03 — Two-stage list-aware correction

  - Hypothesis: Listing price provides valuable information that should be used to correct the off-market estimate, rather than simply included in the original regression.
  - Test: Generate historical out-of-time off-market predictions, then fit a Ridge model using the difference between listing price and estimated market price.
  - Observation:
    - Off-market: MAPE 17.73%, MdAPE 14.01%.
    - List-aware: MAPE 5.75%, MdAPE 4.61%.
  - Conclusion: A separate listing-price adjustment was effective. This became the basis for the current list-aware architecture.
- E04 — Replace the linear property model with HGB

  - Hypothesis: Gradient-boosted trees would capture nonlinear effects and interactions among property characteristics better than Elastic Net.
  - Test: Introduce HistGradientBoostingRegressor while retaining spatial GAM.
  - Observation:
    - Off-market: MAPE 14.97%, MdAPE 11.83%.
    - List-aware: MAPE 5.70%, MdAPE 4.60%.
  - Conclusion: HGB produced a major off-market improvement. The core property model appeared more influential than increasingly elaborate linear modelling.
- E05 — Add TF-IDF text features with Elastic Net selection

  - Hypothesis: Listing remarks contain additional information about property condition, renovations, intended buyers, and other characteristics not fully represented by structured variables.
  - Test: Extract up to 1,000 TF-IDF features, use Elastic Net to select informative terms, and train HGB on the reduced feature matrix.
  - Observation:

    - Off-market: MAPE 14.19%, MdAPE 11.33%.
    - List-aware: MAPE 5.63%, MdAPE 4.54%.
  - Conclusion: Text features improved prediction accuracy, particularly off-market. The TF-IDF extraction and Elastic Net selection became an important part of the pipeline.

**Temporal modelling**

- E06 — Replace temporal splines with ARIMA(1,1,0)

  - Hypothesis: A time-series model fitted to monthly pricing residuals could capture market movements more effectively than a smooth temporal spline.
  - Test: Fit ARIMA(1,1,0) to monthly median log-price residuals after HGB and spatial GAM.
  - Observation:
    - Off-market: MAPE 14.02%, MdAPE 11.23%.
    - List-aware: MAPE 5.66%, MdAPE 4.59%.
  - Conclusion: ARIMA slightly improved off-market accuracy over the temporal spline, but list-aware accuracy changed little. ARIMA(1,1,0) became the preferred temporal component.
- E07 — Test ARIMA(0,1,1)

  - Hypothesis: A moving-average model might describe monthly residual dynamics better than the autoregressive specification.
  - Test: Replace ARIMA(1,1,0) with ARIMA(0,1,1), keeping the broader architecture comparable.
  - Observation:
    - Off-market: MAPE 14.04%, MdAPE 11.24%.
    - List-aware: MAPE 5.68%, MdAPE 4.59%.
  - Conclusion: The two ARIMA orders performed almost identically. ARIMA(1,1,0) retained a marginal advantage.
- E08 — Investigate ARIMA convergence warnings

  - Hypothesis: Nonconvergent ARIMA optimization could lead to unstable temporal corrections and affect model comparisons.
  - Test: Inspect recurring ConvergenceWarning messages. Determine whether they appeared before or after the addition of interaction models.
  - Observation: The warnings were already present before interactions were added. The list-aware pipeline also performs multiple historical ARIMA fits, potentially producing warnings from intermediate models.
  - Conclusion: The warnings cannot currently be attributed to the interaction component. Their effect on prediction accuracy remains unresolved. We still need to identify failed fits, examine their forecast stability, and compare ARIMA against a no-temporal-correction baseline.

**Prediction diagnostics and interaction modelling**

- E09 — Geographical and property-type residual diagnostics
  - Hypothesis: Aggregate accuracy metrics might conceal systematic pricing errors associated with geography, property type, and time.
  - Test: Generate geographic error maps, monthly predicted-versus-actual price plots, and property-type × month signed-error heat maps.
  - Observation:
    - Geographic errors appeared relatively dispersed, without an obvious single dominant anomaly.
    - In May 2026, off-market mean signed error was approximately +4.75% for condo apartments and −5.66% for semi-detached homes.
    - Semi-detached signed errors changed substantially over time, from roughly +9.6% in December to −5.7% in May.
  - Conclusion: The model appeared to miss some property-type-dependent temporal effects. This motivated the interaction experiments.
- E10 — Automatic Elastic Net interaction selection
  - Hypothesis: Explicit interactions between property characteristics, time, and location could correct residual relationships that HGB, GAM, and ARIMA did not capture.
  - Test: Generate structured feature × time/location candidates, screen them by residual correlation, and fit sparse Elastic Net interactions.
  - Observation:
    - With interactions: off-market MAPE 13.89%, MdAPE 11.25%.
    - Without interactions: off-market MAPE 14.02%, MdAPE 11.23%.
    - List-aware MdAPE remained 4.59%.
  - Conclusion: Interactions slightly reduced mean percentage error but did not improve the primary off-market metric. They might help some larger errors while providing little benefit to the median observation.
- E11 — Inspect selected interactions directly
  - Hypothesis: The interaction model might be applying excessively large corrections, explaining unusual property-type bias patterns.
  - Test: Inspect selected Elastic Net interaction coefficients and calculate the interaction correction by property type for the December 2025 validation model.
  - Observation: The fitted interaction model selected zero terms, and all observed interaction corrections were 0%.
  - Conclusion: The interaction component did not cause the December bias pattern in that fitted model. We could not establish that the earlier heat-map differences were caused by interactions. This also suggested the automatic selection process was not consistently finding useful terms.
- E12 — Historical interaction discovery with fixed candidate families
  - Hypothesis: Automatically reselecting interactions during every training fold might be unstable. A separate historical discovery experiment could identify more reliable, interpretable interaction families.
  - Test: Generate out-of-time residual predictions from historical models, then evaluate predefined interactions using expanding-window temporal folds.
  - Observation: The discovery experiment evaluated 27,294 homes across five folds. Its no-correction baseline achieved MAPE 13.787% and MdAPE 10.961%.
  - Conclusion: A fixed interaction specification selected using earlier historical data was a promising alternative to repeated automatic discovery. However, the experiment needed to account for temporal validation and multiple candidate testing.
- E13 — Test linear and nonlinear interaction families
  - Hypothesis: More expressive interactions, particularly property attributes × time, could explain systematic out-of-time residual errors.
  - Test: Evaluate property type × time, area × time, municipality × time, seasonality, nonlinear location, quadratic trends, lot-area interactions, and other candidates.
  - Observation:
    - Property attributes × time: MAPE 13.554%, MdAPE 10.889%.
    - Lot area × time: MAPE 13.615%, MdAPE 10.906%.
    - Property type × time improved MAPE but worsened MdAPE.
    - Community × time and some nonlinear temporal interactions performed poorly.
  - Conclusion: Property attributes × time emerged as the strongest individual interaction family. More complicated interactions were not automatically better.
- E14 — Combine promising interaction families
  - Hypothesis: Combining property-attribute temporal effects with municipality-dependent seasonality could capture complementary residual patterns.
  - Test: Compare property attributes × time alone against combinations containing lot area, municipality × time, and municipality × seasonality.
  - Observation: In the initial comparison, attributes × time + municipality seasonality achieved MAPE 13.529% and MdAPE 10.877%, compared with 10.961% MdAPE for no correction.
  - Conclusion: The combination appeared promising, but its improvement over property attributes × time alone was small. Additional interaction families did not consistently improve performance.
- E15 — Correct interaction-discovery validation
  - Hypothesis: The apparent ranking of interaction families might depend on how Ridge regularization was selected.
  - Test: Replace fixed Ridge penalties or default cross-validation with chronological inner validation for penalty selection.
  - Observation:
    - Property attributes × time: MAPE 13.553%, MdAPE 10.883%.
    - Attributes × time + municipality seasonality: MAPE 13.535%, MdAPE 10.886%.
    - Attributes + lot area + municipality seasonality: MAPE 13.548%, MdAPE 10.885%.
  - Conclusion: Property attributes × time became the best candidate on MdAPE, although the leading models were extremely close. The simpler interaction family was preferred.
- E16 — Integrate fixed Ridge interactions into the complete model
- Hypothesis: The historically discovered property attributes × time effects would improve the actual HGB + GAM + ARIMA pipeline.
- Test: Replace automatic interaction selection with a fixed Ridge model using bedrooms, washrooms, parking, and garage × time.
- Observation:
  - Off-market: MAPE 14.09%, MdAPE 11.32%.
  - List-aware: MAPE 5.67%, MdAPE 4.58%.
- Conclusion: The fixed interactions did not improve integrated off-market accuracy. A likely reason is the training mismatch: discovery used historical out-of-time residuals, whereas production fitted interactions to residuals from models trained on the same houses. That explanation remains a hypothesis, not a confirmed cause.
- E17 — Replace the separate spatial GAM with HGB geographic features
  - Hypothesis: HGB could learn longitude, latitude, and their interactions with property characteristics directly, removing the need for a separate spatial model.
  - Test: Train HGB with coordinates included in its feature matrix, retaining ARIMA but removing GAM and explicit interactions.
  - Observation:
  - Off-market: MAPE 14.12%, MdAPE 11.30%.
  - List-aware: MAPE 5.66%, MdAPE 4.58%.
  - Original HGB + GAM + ARIMA: off-market MdAPE 11.23%, list-aware MdAPE 4.59%.
  - Conclusion: Direct geographical learning with HGB achieved almost identical accuracy. The separate GAM provided a slight off-market advantage, but a simpler single-model spatial architecture became viable.

**Special Gradient Boosting Libraries**

- E18 — CatBoost + ARIMA benchmark

  - Hypothesis: CatBoost's handling of categorical and nonlinear property relationships could outperform the HGB-based architectures.
  - Test: Train the new CatBoost property model with geographic features and selected TF-IDF terms, retaining ARIMA(1,1,0) for temporal residual adjustment.
  - Observation:
    - Off-market: MAPE 13.03%, MdAPE 10.45%, median signed error +0.04%.
    - List-aware: MAPE 5.65%, MdAPE 4.55%, median signed error −1.19%.
    - Compared with the previous best HGB + GAM + ARIMA, off-market MdAPE improved by 0.78 percentage points.
  - Conclusion: This is the strongest observed model so far. However, the CatBoost implementation also uses a redesigned preprocessing pipeline, so the full gain cannot yet be attributed solely to the learning algorithm.
- E19 — Time-aware CatBoost without ARIMA

  - Hypothesis: CatBoost can model geographical, property-specific, and temporal pricing relationships jointly, potentially replacing ARIMA residual correction.
  - Test: Include engineered listing-date and seasonal features in CatBoost and disable ARIMA.
  - Observation: Off-market MdAPE improved from 10.45% to 10.05%, with MAPE improving from 13.03% to 12.76%. However, list-aware MdAPE deteriorated from 4.55% to 4.71%.
  - Conclusion: Time-aware CatBoost is the new best off-market configuration. Its effect on list-aware predictions is less favorable, and the separate contributions of time features versus ARIMA removal remain to be isolated.
  - Next experiment: Time-aware CatBoost + ARIMA.
- E20 — Time-aware CatBoost + ARIMA

  - Hypothesis: ARIMA captures residual temporal patterns not explained by CatBoost's time features.
  - Test: Add ARIMA(1,1,0) to time-aware CatBoost.
  - Observation:
    - Off-market: MAPE 12.67%, MdAPE 10.02%, median signed error +0.38%.
    - List-aware: MAPE 5.82%, MdAPE 4.73%, median signed error −2.34%.
    - ARIMA improved off-market MdAPE by only 0.03 percentage points compared with CatBoost alone.
  - Conclusion: ARIMA provides marginal off-market improvement but no list-aware benefit.

**ARIMA Optimization**

- E21 — ARIMA convergence investigation

  - Hypothesis: Maximum-likelihood estimation (MLE) convergence failures may negatively affect prediction accuracy.
  - Test: Examine optimizer convergence, iterations, and estimated ARIMA parameters.
  - Observation: Some models failed convergence after 12–16 iterations. AR coefficients remained approximately −0.5 to −0.6.
  - Conclusion: Numerical optimization issues exist, but parameter estimates appear relatively consistent.
- E22 — Burg vs. MLE estimation

  - Hypothesis: Burg estimation can improve numerical reliability and predictive accuracy.
  - Test: Replace MLE with Burg while retaining ARIMA(1,1,0).
  - Observation:
    - Off-market: MAPE 12.67%, MdAPE 10.02% for both estimators.
    - List-aware: MAPE 5.82%, MdAPE 4.73% for both estimators.
  - Conclusion: No measurable accuracy improvement at the reported precision. Further ARIMA optimizer experiments are low priority.

**List-Aware Model Experiments**

- E23 — CatBoost sale-to-list ratio prediction

  - Hypothesis: Directly modelling the sale-to-list ratio with CatBoost outperforms the existing two-stage Ridge correction.
  - Test: Train CatBoost to predict `log(sold_price / list_price)` using property, geographical, textual, temporal, and listing-price features.
  - Observation:
    - List-aware: MAPE 5.36%, MdAPE 4.29%, median signed error +1.06%.
    - Previous Ridge model: MAPE 5.82%, MdAPE 4.73%.
    - MdAPE improved by 0.44 percentage points.
  - Conclusion: Ratio-based CatBoost substantially outperforms the previous Ridge architecture and becomes the strongest list-aware model.
- E24 — Direct log-price vs. log-ratio prediction

  - Hypothesis: Predicting the sale-to-list ratio provides a better learning target than predicting sold price directly.
  - Test: Compare CatBoost predicting `log(sold_price)` against CatBoost predicting `log(sold_price / list_price)`, keeping the features and hyperparameters identical.
  - Observation:
    - Direct: MAPE 5.52%, MdAPE 4.42%, median signed error +0.98%.
    - Ratio: MAPE 5.36%, MdAPE 4.29%, median signed error +1.06%.
    - Ratio prediction improved MdAPE by 0.13 percentage points.
  - Conclusion: The ratio formulation performs better, suggesting that using listing price as an explicit baseline is beneficial. Retain ratio CatBoost as the preferred list-aware architecture.
- E25: Market-Level Features

  - Hypothesis: Historical market statistics capture pricing trends beyond CatBoost's existing temporal features.
  - Test: Compare time-aware CatBoost with market features, both with and without ARIMA.
  - Observation:

    - CatBoost + Market: MAPE 12.82%, MdAPE 9.99%.
    - CatBoost + Market + ARIMA: MAPE 12.73%, MdAPE 10.04%.
    - CatBoost + ARIMA alone: MAPE 12.67%, MdAPE 10.02%.
  - Conclusion: Market features provide a small improvement in MdAPE without ARIMA, but combining them with ARIMA does not improve performance. Their incremental value is uncertain.

  **Spatiotemporal Modelling**

  * E26 — CatBoost with Historical Comparable-Sales Features
    * Hypothesis: Nearby historical sales, weighted by geographic distance, recency, and property similarity, contain predictive information beyond conventional property and temporal features.
    * Test: Add historical comparable-sales features to time-aware CatBoost without ARIMA or market-level features.
    * Observation:
      * Off-market MAPE: 11.84% (11.64–12.04%).
      * Off-market MdAPE: 9.20% (8.98–9.46%).
      * Median signed error: +0.90%.
      * Previous best MdAPE: 9.99% using CatBoost + Market.
      * List-aware ratio CatBoost remained unchanged at 5.36% MAPE and 4.29% MdAPE.
    * Conclusion: Spatiotemporal comparable-sales features produced a substantial improvement in both mean and median off-market prediction accuracy. Local historical transaction information appears considerably more valuable than the aggregate market features tested so far.


---

## Additional Experiments and Current Results — October 19–21, 2026

**Evaluation convention.** Unless stated otherwise, numerical results below refer to expanding-window, out-of-time validation on **December 2025–May 2026** (10,069 evaluated homes per track). Each month's model uses training sales from before that month. All metrics are percentages: MAPE = mean absolute percentage error, MdAPE = median absolute percentage error (the competition's primary accuracy metric), and median signed error = median of (prediction − actual) / actual. Results from other fold periods or fixed-origin evaluations must not be compared as if they were identical tests.

**Competition reference baselines**, as published in the handout for the same validation period:

| Track | Reference method | MAPE | MdAPE |
| --- | --- | ---: | ---: |
| Off-market | Median of the 10 nearest earlier sales | 27.40% | 20.80% |
| List-aware | Use list price as predicted sold price | 8.10% | 7.30% |
| Both | Prior-12-month neighbourhood × property-type median | 22.00% | 17.70% |

### E27 — Expand historical comparable-sales features and recheck CatBoost

- **Hypothesis:** Better similarity-weighted local sales information will improve property valuation more than additional generic temporal corrections.
- **Test:** Continue integrating historical comparable-sale price, price-per-square-foot, effective sample size, distance, recency, sale dispersion, same-type share, and local momentum into the time-aware CatBoost workflow. Support a corresponding list-aware ratio/comparable feature provider.
- **Observation:** The subsequently reported CatBoost RMSE configuration achieved off-market **11.86% MAPE / 9.16% MdAPE** and list-aware **5.25% MAPE / 4.19% MdAPE**. These supersede E26's earlier 11.84% / 9.20% snapshot but are from a later configuration; do not interpret differences as the isolated effect of one feature.
- **Conclusion:** Comparable sales remain a major source of useful information; rigorous ablation is still needed to attribute improvements to individual additions.

### E28 — Direct list-aware ratio model and alternative CatBoost objectives

- **Hypothesis:** CatBoost's loss function may affect median percentage error differently from mean-squared log error; modeling the sale-to-list ratio remains useful.
- **Test:** Compare RMSE, MAE, and Huber losses while using the list-aware target `log(sold_price / list_price)` and reconstructing sold price as `list_price * exp(predicted_log_ratio)`.
- **Observation:**

| CatBoost objective | Off-market MAPE | Off-market MdAPE | List-aware MAPE | List-aware MdAPE |
| --- | ---: | ---: | ---: | ---: |
| RMSE | 11.86% | 9.16% | 5.25% | 4.19% |
| MAE | 11.90% | 9.19% | 5.17% | 4.09% |
| Huber (`delta=0.05`), off-market centered | 11.92% | 9.17% | 5.17% | **4.06%** |

- **Implementation issue:** An earlier off-market Huber run without target centering generated absurdly large predictions. Centering the training-fold log-price target by its median, then restoring that offset at prediction time, resolved the pathological behavior. Do not use the uncentered Huber result as a meaningful performance benchmark.
- **Conclusion:** Huber produced the strongest reported list-aware MdAPE (**4.06%**); changing the off-market loss offered little improvement. Preserve the target-centering setting when reproducing Huber runs.

### E29 — Benchmark LightGBM and XGBoost against CatBoost

- **Hypothesis:** Alternative gradient-boosted tree implementations may capture the same nonlinear geographical, property, temporal, and text relationships more accurately or efficiently.
- **Test:** Use the common model-spec interface and comparable feature sets for CatBoost, LightGBM, and XGBoost. Off-market models predict log sale price; list-aware models predict the log sale-to-list ratio.
- **Observation (pre-Optuna configurations):**

| Library | Off-market MAPE | Off-market MdAPE | List-aware MAPE | List-aware MdAPE |
| --- | ---: | ---: | ---: | ---: |
| CatBoost RMSE | 11.86% | 9.16% | 5.25% | 4.19% |
| LightGBM | 10.86% | 8.28% | 5.16% | 4.13% |
| XGBoost | **10.73%** | **8.27%** | 5.16% | 4.15% |

- **Conclusion:** Both LightGBM and XGBoost substantially surpassed the comparable CatBoost RMSE configuration on off-market prediction. LightGBM and XGBoost also trained faster in the observed six-month experiments (approximately 20 minutes compared with approximately 30 minutes for CatBoost, dependent on environment). Neither unambiguously displaced the best list-aware CatBoost objective.

### E30 — Model factory and validation workflow extensions

- **Goal:** Ensure new experiments can share the same chronologically valid training and prediction pathway.
- **Implementation:** Consolidate backend wrappers in `models/boost_regressor.py`; maintain `MODEL_SPECS` integration with `OffMarketModel` and `DirectListingModel(target="ratio")`. Support optional off-market log-target centering. Extend `model_run.ipynb` to support both expanding-window and fixed-origin evaluations, and maintain a separate `optuna.ipynb` development-tuning notebook.
- **Important distinction:** Expanding-window validation retrains before each month. Fixed-origin validation trains once and predicts several following months, closer to the hidden June–September 2026 forecast horizon after May training ends.
- **Status:** These workflows were implemented/discussed; **no quantified fixed-origin horizon result is recorded here**. Recheck factory/tuning parity whenever factory or feature-provider behavior changes.

### E31 — Optuna optimization on earlier chronological folds

- **Hypothesis:** Library-specific tuning can improve both prediction accuracy and model regularization beyond the initial reasonable configurations.
- **Test:** Use separate persistent Optuna studies for each library × track on **September–November 2025** development folds. Tune within the development period, then evaluate the chosen candidate on the regular **December 2025–May 2026** six-month validation. Studies are resumable via SQLite. The development and confirmation results are **not interchangeable**.
- **Observation (post-Optuna, full six-month validation; 10,069 homes per track):**

| Library | Off-market MAPE | Off-market MdAPE | Off-market median signed | List-aware MAPE | List-aware MdAPE | List-aware median signed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM (Optuna) | **10.49%** | 8.07% | +0.35% | 5.17% | 4.18% | +0.53% |
| XGBoost (Optuna) | 10.51% | **7.87%** | +0.31% | 5.18% | 4.15% | +0.56% |
| CatBoost (Optuna) | 10.65% | 8.05% | **+0.13%** | **5.15%** | **4.10%** | +0.57% |

- **Reported uncertainty (off-market, bootstrap 95% intervals):**
  - LightGBM: MAPE 10.30–10.69%, MdAPE 7.85–8.24%.
  - XGBoost: MAPE 10.32–10.72%, MdAPE 7.69–8.06%.
  - CatBoost: MAPE 10.46–10.84%, MdAPE 7.88–8.29%.
- **Development-fold note:** CatBoost's reported development MdAPE was approximately 8.1%; this cannot be compared directly with the full-validation scores because the evaluation months differ.
- **Conclusion:** Optuna improved LightGBM off-market MdAPE from 8.28% to 8.07% and XGBoost from 8.27% to **7.87%**. All three tuned libraries converged near 8% off-market MdAPE. CatBoost remained slower and did not win standalone off-market MdAPE, but could contribute complementary ensemble errors. CatBoost Huber's **4.06%** remains the best *reported* list-aware MdAPE, rather than the tuned CatBoost's 4.10%.

### E32 — Current best validated results and practical interpretation

- **Best reported off-market MdAPE:** Optuna XGBoost, **7.87%** MdAPE and 10.51% MAPE (median signed +0.31%).
- **Best reported off-market MAPE:** Optuna LightGBM, **10.49%** MAPE and 8.07% MdAPE (median signed +0.35%).
- **Best reported list-aware MdAPE:** CatBoost Huber (`delta=0.05`), **4.06%** MdAPE and 5.17% MAPE.
- **Against the official starter off-market baseline:** XGBoost reduced MdAPE from 20.80% to 7.87% (**approximately 62.2% relative reduction**) and MAPE from 27.40% to 10.51% (**approximately 61.6% relative reduction**).
- **Against the official starter list-aware baseline:** CatBoost Huber reduced MdAPE from 7.30% to 4.06% (**approximately 44.4% relative reduction**).
- **Interpretation:** The three tuned libraries are close in aggregate accuracy. The 0.20-percentage-point XGBoost-versus-LightGBM MdAPE difference should be evaluated with *paired* property-level predictions and monthly diagnostics, not treated as proven superiority based on separate confidence intervals. Median signed error closer to zero does **not** establish that CatBoost is more stable over months.
- **Competition weighting:** Accuracy points favor off-market (20 points) over list-aware (10 points). MdAPE is primary, with MAPE as a tiebreaker. Off-market remains the main research priority.

### E33 — Overfitting controls and hidden-period robustness policy

- **Risk identified:** Many architecture changes and tuning choices have been compared on December–May. Repeatedly selecting small improvements there risks validation-set overfitting even when individual features are leakage-safe.
- **Agreed experiment policy:** Use earlier chronological development folds for dense searches (Optuna, text-feature selection, ensemble weights), reserving December–May for limited comparisons of well-motivated, frozen candidates. December–May has already been consulted often and is not an untouched holdout.
- **Important hidden-test mismatch:** The competition's hidden homes are June–September 2026, but labeled training sales end in May. Fixed-origin multi-month evaluation is needed alongside expanding-window validation to check horizon drift.
- **Needed checks:** Per-month MdAPE/MAPE, bias, high-error tails, price/property-type subgroups, paired bootstrap differences, correct cutoff of comparable sales, and whether selected vocabulary or transformations accidentally used validation labels.
- **Conclusion:** Small score gains require stronger out-of-time evidence; the next priority is improving information content and robustness, not automatically increasing trial counts.

### E34 — Planned text representation experiments (NOT YET VALIDATED)

- **Current baseline:** The active `BoostingFeatures` vectorizer uses a fixed selected set of **47 TF-IDF terms**, originally chosen with Elastic Net/Lasso-style selection; a constructor argument named `max_text_features=1000` does **not** itself activate 1,000 terms in the current code.
- **Hypothesis:** More of the approximately 1,000 candidate unigrams/bigrams in listing remarks may explain residual off-market valuation errors that are not captured by property, geographic, or comparable-sales information.
- **Planned controlled comparison:** Existing 47-term TF-IDF; identical 47-term binary BoW; up-to-1,000-term TF-IDF; same 1,000-term binary BoW; same 1,000-term count BoW. Use training-only vocabulary and identical vocabulary for full-width TF-IDF/BoW to isolate representation effects. LightGBM first for speed, XGBoost for follow-up.
- **Caveat:** Numerical word columns do not automatically gain CatBoost's native raw-text benefits. Full-vocabulary models and rare terms can overfit. Filter rare terms and compare out-of-time, rather than assuming more vocabulary means better accuracy.
- **Status:** Proposal only; **no measured BoW or full-vocabulary results yet**.

### E35 — Planned ensemble experiments: row bagging AND feature bagging (NOT YET VALIDATED)

- **Hypothesis:** Averaging models exposed to different training rows, text-feature subsets, and/or boosting libraries may reduce variance and capture more text information without making every individual model too complex.
- **Planned ablations:**
  1. Single tuned model with current 47 TF-IDF terms.
  2. Bootstrap ensemble of several models, using resampled eligible training **rows** with replacement.
  3. Subsampling ensemble using, for example, 80–90% of eligible rows without replacement.
  4. Random-subspace **feature bagging**: preserve structured/comp/temporal features, include 47 core terms if appropriate, and randomly sample additional text terms per member.
  5. Combine row sampling and text-feature bagging; separately test a simple XGBoost/LightGBM blend and potential CatBoost diversity.
- **Implementation safeguard:** Build comparable-sales and market-history features from the **full chronologically eligible historical data first**; row sampling for the estimator must not silently damage the historical comparison pool.
- **Evaluation:** Start with approximately five members, evaluate out-of-time performance versus single-model and simpler ensemble controls, and scale member count only if the gain justifies compute. Compare simple log-price averaging with other aggregations before tuning weights.
- **Correlation rule:** An `r=0.99` diversity rule was discussed, but high prediction correlation is not the same as high *error* correlation and does not guarantee ensemble generalization. Accept members based on actual out-of-time gain; tune combination weights on development folds only.
- **Status:** Design discussion only; **not yet tested** on the housing dataset.

### E36 — Other architecture/feature research queued (NOT YET VALIDATED)

- Try combined aggregate historical market and local comparable-sale features: current feature registry has `Xpt_market` and `Xpt_spatiotemporal` as separate alternatives.
- Evaluate time-filter-before-nearest-neighbor selection in `ComparableSalesFeatures` (the current implementation initially retrieves nearest geographic candidates, then applies age limits); test multi-scale neighbourhood and time-window summaries.
- Try residual learning around historical comparable-value estimates with a fallback for homes lacking comparables.
- Investigate GPBoost spatial residual effects; optionally test Yggdrasil Decision Forests on a compatible system.
- For list-aware, aggregate **historical local sold-to-list ratios** (including dispersion/trend/above-asking frequency) using strictly eligible past completed sales.
- Run fixed-origin evaluations across multiple months and inspect whether explicitly forecasting/adjusting market indices helps when recent sales cannot be observed.
- **Status:** All are hypotheses or queued experiments, not established score improvements.

**Next actions as of October 21, 2026:** Preserve tuned parameter sets and aligned per-home predictions for all three libraries; establish paired baseline/ensemble diagnostics; screen binary BoW vs. TF-IDF and vocabulary expansion on earlier chronological folds; implement independent row-bootstrap and text-feature-subspace studies before combining them; check fixed-origin robustness. Keep December–May as limited confirmation for selected candidates, and use hidden June–September only for the final competition evaluation.
