# Saved model artifacts

Track-scoped model snapshots (new storage layout):

```text
saved_models/
├── off_market/
│   └── <run_name>/
│       ├── model.joblib
│       └── metadata.json
├── list_aware/
│   └── <run_name>/
│       ├── model.joblib
│       └── metadata.json
├── <legacy_run_name>/      # pre-existing mixed-track snapshots (unchanged)
│   ├── model.joblib
│   └── metadata.json
└── README.md
```

**The old flat snapshots are the legacy namespace.** They are not deleted,
moved, renamed or overwritten by this change; older notebooks using
`load_model("XGBoost_text_binary")` still load them. New snapshots should
specify `track="off_market"` or `track="list_aware"` explicitly; omitting it
is intentionally legacy behavior for backward compatibility.

```python
from models.save_model import (
    save_model, load_model, model_info,
    list_saved_models, list_model_catalog, save_validation_tracks,
)

save_model(model, "xgb_selected47_seed42", track="off_market",
           metadata={"features": "Xpt_spatiotemporal__selected_binary",
                     "seed": 42, "training_cutoff": "2025-08-01"})
model = load_model("xgb_selected47_seed42", track="off_market")
info = model_info("xgb_selected47_seed42", track="off_market")
print(list_saved_models(track="off_market"))
print(list_saved_models(track="list_aware"))
print(list_model_catalog())  # includes legacy, without unpickling models

# Optional: split a *trusted*, existing chronological dual-track snapshot.
legacy = load_model("XGBoost_text_binary")
save_validation_tracks(
    legacy, "XGBoost_text_binary",
    metadata={"tracks": ["off_market", "list_aware"],
              "validation_months": ["2025-12", "2026-01", "2026-02",
                                    "2026-03", "2026-04", "2026-05"]},
)
```

The two track namespaces can contain runs with the same name. Loading with
the wrong track raises `FileNotFoundError` rather than falling back silently
to the other track or to legacy. Snapshot names cannot contain paths or
directory traversal. Existing files cannot be overwritten unless the caller
explicitly sets `overwrite=True`.

### Recording ensemble members

Every fitted ensemble member or member bundle should record:
**track**, feature representation and vocabulary/term selection, target
definition (log sale price vs log sale/list ratio), seed, training period,
model hyperparameters, original baseline, chronological fold definition,
and provenance for the validation predictions. The fitted feature
transformer is part of the model: a naked XGBoost estimator and a CSV of
predictions are **not sufficient** to restore inference with text and
historical comparable-sale features.

A multi-member feature-bagging bundle can share one fitted feature transformer,
rather than duplicating a large historical comparable-sale index in every
serialized member. The starter notebook illustrates how to save that bundle
to the off-market namespace.

**Caution:** joblib uses pickle: only load snapshots you trust. Model
binaries can be large, depend on exact library versions, and should normally
remain local or in artifact storage rather than automatically committed to
Git.
