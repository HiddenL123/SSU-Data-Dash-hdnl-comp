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


**Further Comparable-Sales and Gradient Boosting Experiments**

- E27 — Refined CatBoost Comparable-Sales Model

  - Hypothesis: Historical comparable sales provide additional information about local property values that cannot be captured by property characteristics and geographic coordinates alone.
  - Test: Train CatBoost using historical comparable-sale features, including geographically and temporally weighted property-price estimates, price-per-square-foot information, and local transaction statistics.
  - Observation:
    - Off-market: MAPE 11.86%, MdAPE 9.16%.
    - List-aware: MAPE 5.25%, MdAPE 4.19%.
    - Compared with the earlier CatBoost results, this configuration substantially improved off-market accuracy.
  - Conclusion: Comparable-sales information is an important predictor of property values. The improved performance justified using these features in subsequent boosting-library comparisons.
- E28 — CatBoost Loss Function Comparison

  - Hypothesis: Alternative training objectives may improve median percentage error compared with the conventional RMSE objective.
  - Test: Compare CatBoost models trained using RMSE, MAE, and Huber loss. For the list-aware track, continue predicting the logarithm of the sale-to-list ratio.
  - Observation:
    - RMSE:
      - Off-market: MAPE 11.86%, MdAPE 9.16%.
      - List-aware: MAPE 5.25%, MdAPE 4.19%.
    - MAE:
      - Off-market: MAPE 11.90%, MdAPE 9.19%.
      - List-aware: MAPE 5.17%, MdAPE 4.09%.
    - Huber (delta = 0.05):
      - Off-market: MAPE 11.92%, MdAPE 9.17%.
      - List-aware: MAPE 5.17%, MdAPE 4.06%.
    - An initial off-market Huber implementation produced extreme predictions because the log-price target was not centered. Subtracting the training-fold median log price before fitting and restoring it during prediction resolved this numerical issue.
  - Conclusion: Alternative objectives offered little off-market improvement. However, Huber loss produced the strongest list-aware result, reducing MdAPE to 4.06%. This became the preferred list-aware CatBoost configuration.
- E29 — LightGBM and XGBoost Benchmark

  - Hypothesis: Alternative gradient-boosting implementations may produce more accurate property valuations than CatBoost using similar feature representations.
  - Test: Compare CatBoost, LightGBM, and XGBoost using the spatiotemporal comparable-sales feature set and chronological validation from December 2025 to May 2026.
  - Observation:
    - CatBoost (RMSE):
      - Off-market: MAPE 11.86%, MdAPE 9.16%.
      - List-aware: MAPE 5.25%, MdAPE 4.19%.
    - LightGBM:
      - Off-market: MAPE 10.86%, MdAPE 8.28%.
      - List-aware: MAPE 5.16%, MdAPE 4.13%.
    - XGBoost:
      - Off-market: MAPE 10.73%, MdAPE 8.27%.
      - List-aware: MAPE 5.16%, MdAPE 4.15%.
    - LightGBM and XGBoost completed the six-month evaluation in approximately 20 minutes, compared with approximately 30 minutes for CatBoost under the tested configurations.
  - Conclusion: LightGBM and XGBoost substantially outperformed the original CatBoost configuration for off-market prediction. Their similar results suggested that both libraries were viable alternatives. XGBoost achieved the strongest off-market MdAPE, while CatBoost remained competitive for list-aware prediction.

**Model Optimization**

- E30 — Optuna Hyperparameter Optimization Pipeline

  - Hypothesis: The initial boosting configurations were not necessarily optimal, and systematic hyperparameter optimization could improve predictive accuracy.
  - Test: Implement a dedicated Optuna notebook supporting CatBoost, LightGBM, and XGBoost. Use chronological development folds from September to November 2025 to select hyperparameters, with MdAPE as the primary optimization objective. Store trials in persistent SQLite studies to support resuming optimization.
  - Observation:
    - Hyperparameter optimization was completed for all three boosting libraries.
    - Each library was optimized separately.
    - Final configurations were evaluated independently on the December 2025–May 2026 validation period, containing 10,069 properties per track.
  - Conclusion: Separating Optuna development folds from the six-month validation period provided a more controlled way to compare tuned configurations without directly optimizing every trial against the main validation set.
- E31 — Optuna-Tuned LightGBM

  - Hypothesis: Optimizing LightGBM's tree complexity, regularization, and learning parameters would improve its performance beyond the initial configuration.
  - Test: Select LightGBM hyperparameters through Optuna and evaluate the resulting model on the complete six-month chronological validation period.
  - Observation:
    - Off-market:
      - MAPE 10.49% (95% CI: 10.30–10.69%).
      - MdAPE 8.07% (95% CI: 7.85–8.24%).
      - Median signed error +0.35%.
    - List-aware:
      - MAPE 5.17% (95% CI: 5.09–5.26%).
      - MdAPE 4.18% (95% CI: 4.09–4.27%).
      - Median signed error +0.53%.
    - Off-market MdAPE improved from 8.28% to 8.07%, while MAPE improved from 10.86% to 10.49%.
    - List-aware accuracy did not improve over the initial LightGBM configuration.
  - Conclusion: Hyperparameter optimization improved LightGBM's off-market predictions on both metrics. Its final off-market MAPE of 10.49% was the lowest observed among the three optimized boosting libraries.
- E32 — Optuna-Tuned XGBoost

  - Hypothesis: XGBoost could achieve further performance improvements through hyperparameter optimization.
  - Test: Select XGBoost hyperparameters using Optuna on the earlier development folds and evaluate the selected configuration on the December 2025–May 2026 validation period.
  - Observation:
    - Off-market:
      - MAPE 10.51% (95% CI: 10.32–10.72%).
      - MdAPE 7.87% (95% CI: 7.69–8.06%).
      - Median signed error +0.31%.
    - List-aware:
      - MAPE 5.18% (95% CI: 5.10–5.27%).
      - MdAPE 4.15% (95% CI: 4.04–4.24%).
      - Median signed error +0.56%.
    - Off-market MdAPE improved from 8.27% to 7.87%, a reduction of 0.40 percentage points.
    - Off-market MAPE improved from 10.73% to 10.51%.
    - List-aware performance remained approximately unchanged.
  - Conclusion: XGBoost achieved the strongest observed off-market MdAPE of 7.87%, becoming the first tested configuration to fall below 8% on the full six-month validation period. Its MAPE remained approximately equal to that of optimized LightGBM.
- E33 — Optuna-Tuned CatBoost

  - Hypothesis: CatBoost's weaker initial off-market performance might be explained partly by suboptimal hyperparameters rather than an inherent architectural disadvantage.
  - Test: Optimize CatBoost on the earlier chronological development folds and evaluate its selected configuration on the full December 2025–May 2026 period.
  - Observation:
    - Off-market:
      - MAPE 10.65% (95% CI: 10.46–10.84%).
      - MdAPE 8.05% (95% CI: 7.88–8.29%).
      - Median signed error +0.13%.
    - List-aware:
      - MAPE 5.15% (95% CI: 5.07–5.23%).
      - MdAPE 4.10% (95% CI: 4.00–4.19%).
      - Median signed error +0.57%.
    - Off-market MdAPE improved from 9.16% to 8.05%.
    - CatBoost's median signed off-market error was closest to zero among the three tuned libraries.
    - CatBoost required considerably more training time than LightGBM and XGBoost.
    - The earlier CatBoost Huber model retained the strongest list-aware MdAPE at 4.06%.
  - Conclusion: Hyperparameter optimization substantially improved CatBoost and narrowed its performance gap with the other libraries. However, its additional computational cost did not translate into superior standalone off-market MdAPE. The three tuned boosting libraries achieved similar aggregate performance, with XGBoost leading on off-market MdAPE and LightGBM leading on off-market MAPE.

* E34 — TF-IDF, Binary Bag-of-Words, and Count-Based Text Comparison
  * Hypothesis: The existing TF-IDF representation may not be optimal for gradient-boosted trees. Binary word-presence features could provide stronger predictive signals than term frequencies or TF-IDF weights.
  * Test: Compare LightGBM using the existing selected 47-term vocabulary, expanded text vocabularies, and no text features. Evaluate TF-IDF, binary Bag-of-Words, and count-based representations on the same chronological validation cohort.
  * Observation:
    * Validation cohort: 6,152 homes.
    * Selected 47-term binary: MdAPE 8.2996%, MAPE 10.8261%.
    * Selected 47-term TF-IDF: MdAPE 8.3677%.
    * Full binary vocabulary: MdAPE 8.4037%.
    * Full count vocabulary: MdAPE 8.4297%.
    * Full TF-IDF vocabulary: MdAPE 8.5161%.
    * No text features: MdAPE 9.2285%.
  * Conclusion: Textual descriptions provide substantial predictive information. Binary word-presence features outperformed TF-IDF under the tested configuration, while expanding the vocabulary did not improve MdAPE. The 47 selected terms were originally selected using TF-IDF, so additional vocabulary selection specifically for binary encoding may still be worthwhile.
* E35 — Full Validation of Binary Text Features with XGBoost
  * Hypothesis: The improvement from binary text encoding observed in LightGBM would also transfer to the tuned XGBoost model.
  * Test: Replace the existing selected TF-IDF features with binary word-presence features while preserving the tuned XGBoost architecture. Evaluate both prediction tracks using the full December 2025–May 2026 chronological validation period.
  * Observation:
    * Validation cohort: 10,069 homes per track.
    * Off-market: MAPE 10.38%, MdAPE 7.82%, median signed error +0.41%.
    * List-aware: MAPE 5.20%, MdAPE 4.17%, median signed error +0.51%.
    * Previous tuned XGBoost off-market MdAPE: 7.87%.
    * Binary encoding improved off-market MdAPE by approximately 0.05 percentage points.
  * Conclusion: Binary text representation produced a modest improvement in off-market XGBoost accuracy, making it the preferred representation for subsequent experiments. The gain was small relative to validation uncertainty, so it should not be interpreted as conclusive evidence of superiority in every evaluation period.

**Advanced Spatiotemporal Modelling**

* E36 — Spatiotemporal Splines and Tensor GAM Screening
  * Hypothesis: Explicit nonlinear interactions between geography and time could improve housing-price predictions beyond the existing comparable-sales and gradient-boosted features. A separate Tensor GAM residual model might also capture remaining spatial and temporal structure.
  * Test: Compare tuned XGBoost with spatiotemporal spline features against a Tensor GAM residual correction and the existing XGBoost baseline. Use three earlier development months, approximately 40,000 sampled training observations per fold, and 500 boosting trees.
  * Observation:
    * Original XGBoost baseline: MdAPE 9.1596%, MAPE 11.8614%.
    * XGBoost + spatiotemporal splines: MdAPE 8.6045%, MAPE 11.2658%.
    * XGBoost + Tensor GAM correction: MdAPE 9.1080%, MAPE 11.7474%.
    * Spatiotemporal splines produced the largest improvement of the tested configurations.
  * Conclusion: Explicit spatiotemporal spline features provided considerably more benefit than the standalone Tensor GAM residual correction in this development screen. These results justified advancing the spline architecture to full-history validation.
* E37 — Full-History Spatiotemporal Feature and Residual Comparison
  * Hypothesis: Combining spatiotemporal splines, geographic-context variables, and model-specific residual correction could improve the strongest XGBoost baseline when trained on the full historical dataset.
  * Test: Evaluate full-strength, 700-tree XGBoost models on all 10,069 validation homes from December 2025 to May 2026. Compare the baseline against spline features, geographic-context combinations, and Tensor GAM residual corrections calibrated using earlier out-of-time predictions.
  * Observation:
    * Original XGBoost baseline:
      * MdAPE 7.8182%, MAPE 10.3815%.
    * XGBoost + spatiotemporal splines:
      * MdAPE 7.5888%, MAPE 10.0193%.
    * XGBoost + splines + geographic context:
      * MdAPE 7.5316%, MAPE 10.0577%.
    * XGBoost + splines + Tensor GAM:
      * MdAPE 7.5894%, MAPE 9.9855%.
    * XGBoost + splines + geographic context + Tensor GAM:
      * MdAPE 7.4857%, MAPE 10.0339%.
    * The geographic-context and Tensor GAM combination achieved the lowest MdAPE.
    * Tensor GAM reduced MAPE in some configurations, but its incremental MdAPE gains were not consistently distinguishable from zero under the reported uncertainty intervals.
  * Conclusion: Spatiotemporal spline features provided a substantial and reproducible improvement over the existing XGBoost baseline. Additional geographic context further improved MdAPE. Tensor GAM provided smaller, less consistent gains, suggesting that much of the useful spatial-temporal structure was already captured by the expanded feature representation. The best observed full-history MdAPE was 7.4857%, making this architecture a strong reference for subsequent ensemble experiments.
* E38 — GPBoost Structural Spatiotemporal Pilot
  * Hypothesis: Combining gradient-boosted trees with a Gaussian-process spatial-temporal component could model residual geographical dependence more effectively than feature engineering alone.
  * Test: Implement a GPBoost pilot with spatial and temporal coordinates, using a Vecchia-approximated Gaussian process. Compare tree-only and tree-plus-Gaussian-process configurations against the tuned XGBoost baseline on a reduced October 2025 development cohort.
  * Observation:
    * Pilot cohort: 250 validation homes, approximately 5,000 training observations, and 250 boosting trees for the matched XGBoost reference.
    * XGBoost reference: MdAPE 10.9349%, MAPE 13.6510%.
    * GPBoost tree-only: MdAPE 48.7828%, MAPE 72.8239%.
    * GPBoost combined: MdAPE 48.7828%, MAPE 72.7422%.
    * The Gaussian-process adjustment did not materially improve the very poor tree-only predictions.
  * Conclusion: The GPBoost pilot failed to approach the accuracy of the existing XGBoost architecture. Since both GPBoost configurations performed similarly poorly, the result may reflect implementation, objective, target-scaling, or model-configuration problems rather than an inherent limitation of Gaussian processes. Further GPBoost development was deprioritized in favour of more reliable spatial residual methods.
* E39 — Spatial Matérn and Space-Time Matérn Residual Models
  * Hypothesis: A separate Matérn-kernel model fitted to historical out-of-time residuals could capture spatial and temporal pricing errors that the XGBoost feature model misses.
  * Test: Compare spatial Matérn, space-time Matérn, and Tensor GAM corrections against the same XGBoost baseline on an earlier three-month development cohort. Examine whether additional kernel tuning improves accuracy.
  * Observation:
    * Validation cohort: approximately 2,700 homes.
    * XGBoost baseline: MdAPE 10.8761%, MAPE 13.5692%.
    * Tensor GAM: MdAPE 10.7983%, MAPE 13.4371%.
    * Spatial Matérn: MdAPE 10.8921%, MAPE 13.4315%.
    * Space-time Matérn: MdAPE 10.8468%, MAPE 13.5404%.
    * Additional tuning of the space-time Matérn model produced MdAPE 10.8343%, MAPE 13.8816% in the reported tuning comparison.
  * Conclusion: Spatial residual correction offered relatively small improvements compared with feature engineering. Tensor GAM achieved the strongest MdAPE in the matched three-month screen, while spatial Matérn achieved slightly lower MAPE. Kernel complexity and additional tuning did not consistently outperform the simpler correction. These methods remained possible ensemble members but did not justify replacing the primary XGBoost model.

**Ensemble Learning and Optimization**

* E40 — Feature and Training-Row Bagging Architecture
  * Hypothesis: Training multiple XGBoost models on different feature representations and historical training subsets could produce partially independent prediction errors. An ensemble of such models might outperform a single optimized XGBoost configuration.
  * Test: Develop a feature-bagging and row-bagging system supporting different textual feature subsets, spatiotemporal feature configurations, randomized historical training samples, and reproducible candidate seeds. Introduce independent member predictions and weighted geometric averaging.
  * Observation:
    * The framework supports alternative feature selections, training-row samples, and vocabulary masks.
    * Features and text vocabularies are constructed from eligible historical information only.
    * Candidate predictions can be saved and reused without retraining successfully completed models.
    * The initial framework established the infrastructure needed for subsequent automated candidate selection.
  * Conclusion: Feature and row bagging provided a practical mechanism for generating heterogeneous XGBoost candidates. However, diversity alone does not guarantee improvement, so explicit candidate evaluation and ensemble-admission rules were needed.
* E41 — Diversity-Aware AutoML Ensemble Pilot
  * Hypothesis: Automatically selecting candidates based on both prediction accuracy and incremental ensemble improvement would outperform simply combining every available model.
  * Test: Implement an automated XGBoost ensemble factory using Optuna hyperparameter optimization, candidate generation, cached chronological predictions, and constrained forward ensemble selection. Evaluate an initial computationally reduced experiment before committing to the full training budget.
  * Observation:
    * Pilot budget: approximately 5,000 historical training observations and 220 XGBoost trees.
    * Selection cohort: 500 homes.
      * Ensemble MdAPE 11.0683%, MAPE 13.9111%.
    * Shadow-audit cohort: 500 homes.
      * Ensemble MdAPE 11.2745%, MAPE 14.8734%.
    * Selected ensemble weights:
      * `<span>selected_47_splines_geo_reference</span>`: 90%.
      * `<span>g000_c001</span>`: 10%.
    * Incremental ensemble-selection gain: approximately 0.0784 percentage points.
    * Candidate prediction-error correlation: approximately 0.8905.
  * Conclusion: The AutoML factory successfully identified a complementary candidate that improved the reference model rather than simply accepting every trained member. The relatively small incremental improvement and high error correlation demonstrated the importance of diversity-aware admission. The pilot validated the selection mechanism, but its reduced training budget made its absolute error values unsuitable for direct comparison with the full-history model.
* E42 — Resumable AutoML Training and Cross-Machine Checkpointing
  * Hypothesis: Full-history automated ensemble training would be too computationally expensive to require completion in a single uninterrupted session. Persistent checkpoints and interchangeable execution environments would make larger experiments practical.
  * Test: Add persistent Optuna trial storage, completed-prediction caches, atomic experiment-state updates, interruption recovery, and portable checkpoint archives. Separate the execution configuration into Colab daytime, full-speed local overnight, and reduced-resource local profiles.
  * Observation:
    * Completed trials, candidate decisions, ensemble membership, and historical validation predictions are persisted between sessions.
    * Portable checkpoints include consistent SQLite backups and SHA-256 verification.
    * Cloud training operates on local Colab runtime storage while writing checkpoint archives to Google Drive.
    * The cloud and local execution profiles preserve the same model settings, candidate seeds, and chronological evaluation protocol.
    * During the Colab-to-Windows transfer, the experiment manifest initially detected a different input-data fingerprint. Investigation identified that Windows was loading `<span>listings.csv</span>`, while the cloud configuration could use Parquet.
    * Additional diagnostics were introduced to identify data, parameter, and model-configuration mismatches without silently modifying the saved experiment.
  * Conclusion: Checkpointing made extended ensemble optimization and cross-machine recovery operationally feasible. Strict manifest checking is necessary to prevent cached predictions from being reused with incompatible source data or model settings. Computational resource allocation can change between machines without intentionally changing the underlying experiment.
* E43 — Full-Strength XGBoost AutoML Ensemble and Frozen Shadow Audit
  * Hypothesis: Applying the automated ensemble-selection process to full-history, 700-tree XGBoost models would yield meaningful improvements over the strongest selected reference model and generalize beyond the months used for candidate selection.
  * Test: Execute the full-strength ensemble factory with Optuna-tuned candidate configurations, feature and row bagging, and constrained ensemble admission. Use April–May 2025 for tuning, June–July 2025 for ensemble selection, and August–October 2025 for a frozen shadow audit. Fit each monthly model using only eligible earlier historical sales.
  * Observation:
    * Initial search configuration: 2 generations, 3 candidates per generation, and 5 Optuna trials per candidate.
    * Each model used up to 700 boosting trees and the full eligible historical training population.
    * Frozen audit cohort: 5,951 homes.
    * First-member reference:
      * MdAPE 7.5840%.
      * MAPE 10.0132%.
      * P90 absolute percentage error 21.3219%.
    * Selected ensemble:
      * MdAPE 7.4010%.
      * MAPE 9.8570%.
      * P90 absolute percentage error 21.0450%.
    * Ensemble improvements relative to the first member:
      * MdAPE decreased by 0.1830 percentage points.
      * MAPE decreased by 0.1562 percentage points.
      * P90 absolute percentage error decreased by 0.2768 percentage points.
    * Relative MdAPE improvement: approximately 2.41%.
    * Ensemble membership and weights were frozen before opening the shadow audit.
  * Conclusion: The full-strength AutoML ensemble outperformed its first-member reference on median, mean, and upper-tail percentage error. The consistent direction of improvement provides evidence that complementary candidates contribute useful predictive information beyond the reference model. This established 7.4010% MdAPE as the new observed shadow-audit result. However, the audit covers a different historical period from the previous six-month benchmark, and these development months have been examined during earlier research. A same-period December 2025–May 2026 evaluation remains necessary before claiming an improvement over the previous full-history benchmark.
