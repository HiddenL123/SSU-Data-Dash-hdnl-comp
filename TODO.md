# Housing Price Model: Roadmap

Target: `list_price` (modeled as `log(list_price)`), Toronto-area listings, Jul 2025 to Sep 2026.
Primary model family: gradient-boosted trees (LightGBM / sklearn HGB). Ridge is kept as a sanity-check baseline.

Legend: `[x]` done, `[ ]` to do. Items are ordered by priority within each phase.


Proposed equation

predicted_listing_price = f(X) + g(x, y) + m(t) + sum(interaction)
predicted_sold_price = listing_price + h(X)


where f(X) is linear model
g(x, y) a grid surface mesh that stores the locational pricing
m(t) is the market level for pricing changes over time

---

[ ] Test example model
[ ] Feature engineer rest of variable for f(X)
[ ] Train a baseline models for listing_price and sold_price
# Extra features
  [ ] Train model with language extraction
  - bag of words
  - TF-IDF
  [ ] Add in market_level(t) index to account for market over time
  Example:
    # 1) first-pass f without t, out-of-fold
    oof = cross_val_predict(f_model, train[feats_no_t], y, cv=GroupKFold(5), groups=train["ml_num"])
    resid = y - oof

    # 2) market index
    month = train["date_listed"].dt.to_period("M")
    m = resid.groupby(month).median()

    # 3) retrain f on the deflated target (optionally repeat 1-3 once more)
    y_adj = y - month.map(m).astype(float)
    f_model.fit(train[feats_no_t], y_adj)
  
  [ ] Add in spactial mesh


[ ] Compare the models and pick the better one
[ ] Train with LightGBM ensemble
[ ] Interpret the final model

Ideas
- Use Bag of words model to feature extract from the 'remarks' column
- LightGBM for tree model selector
- Treat time seperately from the other features log(price) = f(house attributes, location, text) + market_level(t) + noise


Ensemble -used for squeezing out last bit of performance/predictive power
  look for a threshold
  Set a maximum correlation 0.99 to other models before model is added to the ensemble

