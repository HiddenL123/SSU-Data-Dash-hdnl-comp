"""Crash-safe local file checkpoints and single-process factory lock.

Temporary files are written and flushed in their destination directory,
then atomically replaced. Interrupted writes leave the last complete version.
The persistent .run.lock file is deliberately *not* a PID sentinel: the OS
releases its advisory lock on process exit, Ctrl+C, or a terminated process.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import pandas as pd


@contextmanager
def exclusive_factory_lock(directory):
    """Only one factory writer or shadow auditor per output directory.

    Cross-platform OS advisory lock. Use a local filesystem, not cloud-sync
    folders or multi-machine network file shares.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+b") as file:
        if os.name == "nt":
            import msvcrt

            file.seek(0)
            if not file.read(1):
                file.write(b"0")
                file.flush()
            file.seek(0)
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"Another factory process is using {root}; stop it "
                    "before resuming this run."
                ) from exc
            try:
                yield
            finally:
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    f"Another factory process is using {root}; stop it "
                    "before resuming this run."
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _write_atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, scratch = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(scratch, path)
    finally:
        if os.path.exists(scratch):
            os.unlink(scratch)


def write_json(path, value):
    _write_atomic(
        path,
        lambda out: out.write(
            json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
        ),
    )


def write_csv(path, frame: pd.DataFrame):
    _write_atomic(path, lambda out: frame.to_csv(out, index=False))
