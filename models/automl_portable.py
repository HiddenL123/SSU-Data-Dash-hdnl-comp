"""Portable, consistent AutoML checkpoints for local <-> Colab handoff.

Never run a live Optuna SQLite database inside a mounted Google Drive folder.
Work on the runtime's local filesystem, and export an archive AFTER durable
checkpoints. Online sqlite3.backup() captures a transactionally consistent DB.

Archive does not include raw listings or saved_models; make the same listings
accessible on each worker and regenerate final fitted snapshots if necessary.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
import zipfile

from models.automl_checkpoint import exclusive_factory_lock

SKIP_FILES = {".run.lock"}
SKIP_SUFFIXES = {".tmp", ".db-wal", ".db-shm"}
ARCHIVE_VERSION_FILE = "_automl_checkpoint_format.txt"


def _sqlite_bytes(path):
    with tempfile.TemporaryDirectory() as temp:
        staging = Path(temp) / "copy.db"
        source = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        target = sqlite3.connect(staging)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        return staging.read_bytes()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive(root, destination):
    root = Path(root).resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"No initialized AutoML experiment in {root}")
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="automl_portable_") as tmp:
        draft = Path(tmp) / "checkpoint.zip"
        with zipfile.ZipFile(draft, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=3, allowZip64=True) as z:
            z.writestr(ARCHIVE_VERSION_FILE, "1\n")
            for file in sorted(root.rglob("*")):
                if not file.is_file() or file.is_symlink():
                    continue
                rel = file.relative_to(root)
                if file.name in SKIP_FILES or file.suffix in SKIP_SUFFIXES:
                    continue
                if file.resolve() == destination.resolve():
                    continue
                key = rel.as_posix()
                if file.suffix == ".db":
                    z.writestr(key, _sqlite_bytes(file))
                else:
                    z.write(file, arcname=key)
        # Copy to a temporary destination on the SAME filesystem as Drive
        # target so a remote quota/IO failure cannot corrupt last checkpoint.
        staging = destination.with_name(destination.name + ".partial")
        try:
            shutil.copyfile(draft, staging)
            os.replace(staging, destination)
            digest = _sha256(destination)
            digest_file = destination.with_name(destination.name + ".sha256")
            marker = digest_file.with_name(digest_file.name + ".partial")
            marker.write_text(digest + "\n", encoding="utf-8")
            os.replace(marker, digest_file)
        finally:
            staging.unlink(missing_ok=True)
        return destination


def export_checkpoint(root, archive_path, *, already_locked=False):
    root = Path(root)
    if already_locked:
        return _archive(root, archive_path)
    with exclusive_factory_lock(root):
        return _archive(root, archive_path)


def import_checkpoint(archive_path, output_dir):
    """Restore to a NEW / empty local folder, never merge two live studies."""
    archive = Path(archive_path).expanduser()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    verify = archive.with_name(archive.name + ".sha256")
    if verify.is_file():
        expected = verify.read_text(encoding="utf-8").strip()
        if _sha256(archive) != expected:
            raise ValueError("Checkpoint archive hash mismatch")
    output = Path(output_dir).expanduser()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite or merge nonempty experiment {output}. "
            "Keep only one authoritative active copy per handoff."
        )
    with zipfile.ZipFile(archive, "r") as z:
        if ARCHIVE_VERSION_FILE not in z.namelist():
            raise ValueError("Not an AutoML portable checkpoint archive")
        if z.read(ARCHIVE_VERSION_FILE) != b"1\n":
            raise ValueError("Unsupported portable checkpoint format")
        names = [n for n in z.namelist() if n != ARCHIVE_VERSION_FILE]
        for key in names:
            rel = PurePosixPath(key)
            if (rel.is_absolute() or not rel.parts or
                any(p in ("..", "") for p in rel.parts) or "\\" in key):
                raise ValueError(f"Unsafe archive entry: {key!r}")
            if z.getinfo(key).is_dir():
                continue
        output.mkdir(parents=True, exist_ok=True)
        for key in names:
            if z.getinfo(key).is_dir():
                continue
            target = output.joinpath(*PurePosixPath(key).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(key) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    if not (output / "manifest.json").is_file():
        raise ValueError("Restored checkpoint missing experiment manifest")
    return output
