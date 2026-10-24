# Full-strength list-aware fixed-origin confirmation

This experiment resolves the ranking reversal between the **350-homes-per-month,
300-tree pilot** and the previously examined **full-strength expanding-origin**
list-aware comparison.

## Reproduce

From the checkout root in the project environment:

```powershell
py -3.13 -m pytest tests/test_listaware_confirmation.py tests/test_crossfamily_followups.py -q
# If the 5,951-home results already came from --full single-fit fixed origin,
# DO NOT train again: reuse the existing output files instead.
py -3.13 listaware_full_fixed_origin_confirmation.py --from-existing

# Only if the full-strength fixed-origin predictions do not yet exist:
py -3.13 listaware_full_fixed_origin_confirmation.py
```

`--from-existing` reads only the saved predictions and validates the existing `manifest.json` (full=True, single-fit fixed origin, all original candidates, full training/validation caps). It fails rather than accidentally reusing an expanding or pilot result. The normal, training path requires the same listings source as previous experiments. The runner uses
`data.load("listings")`, and obeys `DATA_DIR` if set. It resumes already
verified candidate/month prediction CSVs under
`outputs/listaware_fixed_origin_full/`; it does not replace any frozen
production model. Do not switch historical source files during a run.

## Protocol

- Training uses every eligible historical sale completed **before
  August 1, 2025**, up to the existing 200,000-row safety cap (the dataset
  contains approximately 150,000 sales in total).
- **One fit per base estimator**, then forecast **all** August, September,
  and October 2025 homes without refitting.
- Existing configured CatBoost MAE and Huber, CatBoost spatial/advanced,
  tuned XGBoost spatial and LightGBM spatial candidates and **all existing
  predefined geometric blends** are evaluated with full recorded tree counts.
- No new features, residual corrections, blend weight optimization, or
  hyperparameter search. All model definitions remain in
  `models/listaware_shotgun_specs.py` and
  `listaware_fixed_origin_followup.py`.
- The comparison report fixes the previous top variant,
  `followup_cat_mae_xgb_lgb`, as the reference rather than changing the
  reference to whichever model wins the new evaluation.

## Outputs

`outputs/listaware_fixed_origin_full/` contains the ordinary follow-up
outputs and:

- `confirmation_vs_prior_winner.csv`: one row per retained leading
  alternative; MdAPE, MAPE, P90 APE, median signed error and paired
  bootstrap 95% intervals for **alternative minus prior winner**.
- `confirmation_vs_prior_winner_monthly.csv`: one row per
  month/candidate, compared against the same previous winner for that month.

A **negative** delta indicates that the alternative performed better than
the previous winner. The confidence intervals are exploratory, especially
because the earlier validation results influenced which candidates are being
compared; this is **not** an untouched holdout or a corrected
multiple-comparison inference.

Do not compare absolute numbers from this full-strength run to the pilot
without acknowledging the simultaneous differences in sample sizes,
training budgets, and validation protocol.

After reviewing these results, separately consider the earlier same-type
comparable-sale quota, KNN/RBF-Ridge residual corrections and other
feature experiments; they are intentionally excluded from this confirmation.

## Interpreting the newly supplied 5,951-home table

The same headline is printed by both `--full` and `--from-predictions` in the historical runner. A table alone cannot prove the fitting protocol. Confirm the originating output directory and `manifest.json`: a verified `outputs/listaware_fixed_origin_full/manifest.json` with `"full": true` and `"mode": "single-fit_fixed_origin"` establishes that the earlier result already is the full-strength fixed-origin comparison. In that case **do not retrain**; use `--from-existing` to compute head-to-head paired intervals directly versus CatBoost-MAE/XGB/LGB. The existing summary's confidence intervals are relative to CatBoost Huber, not the prior winner.
