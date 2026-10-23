"""Exact, read-only diagnostics for the existing AutoML v1 manifest.

Keep the ORIGINAL v1 hash algorithm so previously saved Colab manifests remain
comparable. Never silently rewrite or waive an input-data fingerprint mismatch.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd


def compute_manifest(settings, sales, xgb_parent, *, version=1):
    """Reproduce the existing factory's v1 identity exactly."""
    df = sales.copy()
    df["date_listed"] = pd.to_datetime(df.date_listed)
    df["date_sold"] = pd.to_datetime(df.date_sold)
    if df.ml_num.astype(str).duplicated().any():
        raise ValueError("Monthly inputs require unique MLS identifiers")
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True)
        .to_numpy(dtype="uint64").tobytes()
    ).hexdigest()
    frozen = asdict(settings)
    frozen.pop("generations")
    return json.loads(json.dumps({
        "version": version,
        "settings": frozen,
        "data_sha256": digest,
        "xgb_parent": xgb_parent,
    }, sort_keys=True))


def compare_manifests(saved, local):
    """Return structured exact differences; no implicit value normalization."""
    diffs = {}
    for field in ("version", "settings", "data_sha256", "xgb_parent"):
        old, new = saved.get(field), local.get(field)
        if field in ("settings", "xgb_parent") and isinstance(old, dict) and isinstance(new, dict):
            for name in sorted(set(old) | set(new)):
                if old.get(name) != new.get(name):
                    diffs[f"{field}.{name}"] = {
                        "checkpoint": old.get(name),
                        "local": new.get(name),
                    }
        elif old != new:
            diffs[field] = {"checkpoint": old, "local": new}
    return diffs


def saved_factory_settings(root, *, generations):
    """Use checkpoint parameters explicitly, without skipping data checks."""
    from ensemble_automl_factory import FactorySettings

    file = Path(root) / "manifest.json"
    if not file.is_file():
        raise FileNotFoundError(f"No restored checkpoint manifest: {file}")
    frozen = json.loads(file.read_text(encoding="utf-8"))
    rec = dict(frozen["settings"])
    rec["generations"] = int(generations)
    # Intentionally do not rewrite rec['output_dir'] to avoid silently
    # bypassing an original experiment-path/config mismatch.
    return FactorySettings(**rec)


def diagnose(root, *, generations=2, use_saved_settings=False):
    """Report why this machine cannot use an existing checkpoint.

    No folders, roster, archive or Optuna state are changed.
    """
    import data
    from ensemble_automl_factory import FactorySettings, VERSION
    from models.automl_profiles import release_settings
    from spacetime_fast import XGB_TUNED

    root = Path(root)
    manifest_file = root / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint manifest absent at {manifest_file.resolve()}. "
            "Check that restore succeeded and you are in the project root."
        )
    saved = json.loads(manifest_file.read_text(encoding="utf-8"))
    if use_saved_settings:
        cfg = saved_factory_settings(root, generations=generations)
    else:
        cfg = FactorySettings(
            **release_settings(output_dir=root.as_posix(), generations=generations)
        )
    sales_path = data.find_file("listings")
    sales = data.load("listings")
    local = compute_manifest(cfg, sales, XGB_TUNED, version=VERSION)
    differences = compare_manifests(saved, local)
    result = {
        "manifest": str(manifest_file.resolve()),
        "data_source": str(sales_path),
        "row_count": int(len(sales)),
        "column_count": int(len(sales.columns)),
        "pandas_version": pd.__version__,
        "differences": differences,
        "settings_from_checkpoint": bool(use_saved_settings),
    }
    print(f"Restored checkpoint: {result['manifest']}")
    print(f"Local listings: {result['data_source']}")
    print(f"Rows={result['row_count']:,}, columns={result['column_count']}, pandas={pd.__version__}")
    if not differences:
        print("Manifest matches exactly. Audit/benchmark may proceed.")
    else:
        print("Frozen checkpoint versus local values:")
        print(json.dumps(differences, indent=2, default=str))
        if "data_sha256" in differences:
            print(
                "DATA MISMATCH: check whether Colab used a Parquet file "
                "while Windows used a CSV, or whether row order, dtypes, "
                "dataset content or pandas version changed. Reuse the exact "
                "Colab input file/version. Do NOT edit manifest.json."
            )
        if any(k.startswith("settings.") for k in differences):
            print(
                "SETTINGS MISMATCH: use --use-checkpoint-settings to load the "
                "original protocol. That still checks the data fingerprint."
            )
        if any(k.startswith("xgb_parent.") or k == "version" for k in differences):
            print(
                "CODE MISMATCH: align the model baseline and code revision "
                "with the version that created the checkpoint."
            )
    return result
