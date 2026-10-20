"""Save and reload fitted models (including nested validation-model lists).

A run is stored in ``saved_models/<name>/`` beside this module:
    model.joblib    - serialized fitted model(s)
    metadata.json  - human-readable provenance and configuration

Examples
--------
    from save_model import save_model, load_model, list_saved_models

    save_model(fitted_models, 'catboost_comps_900', metadata={
        'tracks': ['off_market', 'list_aware'],
        'validation_months': ['2025-12', '2026-01'],
        'iterations': 900,
    })
    fitted_models = load_model('catboost_comps_900')
    print(list_saved_models())

Only load model files you created or otherwise trust: joblib is pickle-based.
Keep the model's Python classes importable from their original modules when loading.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import tempfile
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import joblib

DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "saved_models"
_FORMAT_VERSION = 1
_ALLOWED_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not _ALLOWED_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(
            "Model name must contain only letters, numbers, underscores, "
            "hyphens or periods; it cannot include a path."
        )
    return name


def _directory(save_dir: str | Path | None) -> Path:
    return Path(save_dir) if save_dir is not None else DEFAULT_SAVE_DIR


def _describe(obj: Any, *, depth: int = 0) -> dict[str, Any]:
    """Record the shape/types of a model collection, without serializing it to JSON."""
    if isinstance(obj, (list, tuple)):
        result: dict[str, Any] = {"collection": type(obj).__name__, "length": len(obj)}
        if depth < 3:
            result["items"] = [_describe(item, depth=depth + 1) for item in obj]
        return result
    cls = type(obj)
    return {"type": f"{cls.__module__}.{cls.__qualname__}"}


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("joblib", "catboost", "scikit-learn", "pandas", "numpy", "statsmodels"):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            pass
    return versions


def save_model(
    model: Any,
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
    compress: int = 3,
    save_dir: str | Path | None = None,
) -> Path:
    """Persist one fitted model, a list, or a nested list of fitted models.

    Parameters
    ----------
    model: A fitted model or arbitrary (nested) collection of fitted models.
    name:  Unique experiment name (no path separators).
    metadata: Optional JSON-compatible run details (parameters, tracks, cutoffs).
    overwrite: Replace a previously saved run with the same name if True.
    compress: joblib compression level, 0 (fast/larger) through 9 (slow/smaller).
    save_dir: Directory containing named run folders; defaults to ./saved_models
              next to this file.

    Returns the folder containing the model and its metadata.
    """
    name = _check_name(name)
    if not isinstance(compress, int) or not 0 <= compress <= 9:
        raise ValueError("compress must be an integer from 0 to 9")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    root = _directory(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists() and not overwrite:
        raise FileExistsError(f"Already saved: {target} (choose a new name or overwrite=True)")

    info = {
        "format_version": _FORMAT_VERSION,
        "name": name,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "model_structure": _describe(model),
        "metadata": metadata or {},
    }
    # Validate metadata *before* beginning the potentially expensive model dump.
    json_text = json.dumps(info, indent=2, default=str)

    # Stage the complete run in a temporary folder. An unsuccessful dump does
    # not leave a partially saved model at its final location.
    with tempfile.TemporaryDirectory(prefix=".saving_", dir=root) as staging:
        staged = Path(staging)
        joblib.dump(model, staged / "model.joblib", compress=compress)
        (staged / "metadata.json").write_text(json_text + "\n", encoding="utf-8")

        if target.exists():
            if not target.is_dir():
                raise FileExistsError(f"A non-directory entry already exists: {target}")
            # Only remove an existing run when the user explicitly opts in.
            shutil.rmtree(target)
        os.replace(staged, target)

    return target


def load_model(name: str, *, save_dir: str | Path | None = None) -> Any:
    """Load the fitted model(s) exactly as saved (same list/nesting/order)."""
    folder = _directory(save_dir) / _check_name(name)
    path = folder / "model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"No saved model at {path}")
    # SECURITY: joblib.load can execute code. Never load untrusted files.
    return joblib.load(path)


def model_info(name: str, *, save_dir: str | Path | None = None) -> dict[str, Any]:
    """Read saved metadata without loading the potentially large model file."""
    path = _directory(save_dir) / _check_name(name) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"No model metadata at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_saved_models(*, save_dir: str | Path | None = None) -> list[str]:
    """List available saved runs by name, without loading their models."""
    root = _directory(save_dir)
    if not root.is_dir():
        return []
    return sorted(
        folder.name
        for folder in root.iterdir()
        if folder.is_dir() and (folder / "model.joblib").is_file()
    )


# Optional plural spellings; both functions also support single models.
save_models = save_model
load_models = load_model
