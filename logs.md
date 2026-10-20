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
