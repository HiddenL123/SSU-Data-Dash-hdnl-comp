# Inference-only, run.py-first workspace

This intentionally small branch retains **run.py byte-for-byte unchanged**
from `main`. It includes `data.py`, `check_submission.py`, the original
`models/` and `features/` dependencies, pinned project environment metadata,
and essential import-path compatibility modules. Most experimental notebooks,
AutoML runners and research scripts have been excluded.

## Setup

```bash
git clone --branch release/runpy-clean --single-branch https://github.com/HiddenL123/SSU-Data-Dash-hdnl-comp.git
cd SSU-Data-Dash-hdnl-comp
uv sync
```

Data loading uses `data.py`; either place the required source files in
`data/` or set `DATA_DIR` (dataset provenance must remain consistent).

## Existing command line, unchanged

```bash
uv run python run.py validate
uv run python run.py report
uv run python run.py predict
uv run python run.py check
```

Outputs, reports, figures, and submissions are generated locally. Follow
the competition rubric for the two submission filenames.

**Important:** by design `run.py` still constructs the existing default
models through `models.model.make_model`. It does not automatically load
the optimized off-market factory or new list-aware selected-ensemble snapshot.
This branch is a clean execution baseline, **not** an integration of those
future fitted artifacts. Do not assume its scores match the ensemble factory's
benchmarks.

To deliver optimized ensemble predictions later, explicitly integrate and
test the saved models in a separate approved change, or run an inference
entry point compatible with them. Do not silently modify `run.py`.

Keep `saved_models/` binary artifacts and historic data off GitHub; share
them via a separate trusted transfer when necessary.
