# Frozen dual-track submission runner

A small inference-only branch. `run.py predict` loads finalized, trusted
snapshots via `models/ensemble_serving.py`; it never retrains or tunes them.

## Saved models needed (not included in Git)

- `saved_models/off_market/off_market_release_candidate_full/model.joblib`
- `saved_models/list_aware/list_aware_ensemble_v1/model.joblib`

Each folder also needs its `metadata.json`. Load only joblib files you trust.

## Usage

```powershell
uv sync
uv run python run.py predict
uv run python run.py check
```

Submission files are `submissions/submission_off_market.csv` and
`submissions/submission_list_aware.csv`. The command checks both temporary
CSVs before replacing final outputs, with no silent fallback. Override
`OFF_MARKET_MODEL_NAME`, `LIST_AWARE_MODEL_NAME`, or `SAVED_MODELS_DIR`
when the local model files use another name or location.

`run.py validate` / `report` remain the original chronological **baseline**
evaluation, NOT validation of the full-history fitted ensembles. Applying the
latter to historical sales would leak future outcomes. Historical frozen
ensemble results should be taken from the respective factory's audit/benchmark.
Legacy serialized space-time add-ons load with absent optional flags defaulted
to None, preserving their previously fitted features.
