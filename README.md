# Dual-track ensemble testing and run workspace

This branch retains the separate off-market and list-aware factories and now
includes the frozen-model submission runner from `release/runpy-clean`.
It has the latest compositional list-aware candidate generator from `main`.

## Run the synthetic tests (no expensive fitting)

```powershell
uv sync
uv run --with pytest python -m pytest tests/test_finalized_inference.py tests/test_candidate_generation.py -q
```

These tests serialize small synthetic instances of the actual saved ensemble
types. They verify member weights and log-space blending, independent fitted
feature providers, correctly loaded track-specific snapshots, model caching,
invalid/missing data handling, backward compatibility for older pickle
transformers, and the two-submission `run.py predict` path.

## Test the finalized models on your own computer

Copy trusted saved bundles into these track-specific paths:

- `saved_models/off_market/off_market_release_candidate_full/{model.joblib,metadata.json}`
- `saved_models/list_aware/list_aware_ensemble_v1/{model.joblib,metadata.json}`

Then run:

```powershell
uv run python run.py predict
uv run python run.py check
```

The generated files are `submissions/submission_off_market.csv` and
`submissions/submission_list_aware.csv`. The runner never reads training
labels, trains new models, or chooses ensemble weights. Missing models fail
loudly rather than falling back to baselines.

**Historical scores:** `run.py validate` and `report` still evaluate the
original models using true chronological fit splits. They are NOT valid
evaluations of pretrained full-history ensembles. Use the respective
historical factory benchmarks / audited predictions for those.

Actual model snapshots, local housing data and AutoML checkpoints should
remain off GitHub. Load only trusted joblib files.

This test branch is intentionally a smaller workspace rather than a wholesale
copy of the main research notebooks.
