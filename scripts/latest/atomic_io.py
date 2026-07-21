#!/usr/bin/env python3
"""Crash-safe persistence for the memory store.

`open(path, "wb")` truncates the file the instant it is called, so a process
that dies between that truncation and the end of `pickle.dump` leaves a
half-written file behind, and the next `pickle.load` raises — the whole store is
gone. The window is not theoretical here: the store reaches ~56 MB / ~330 ms per
write at 1e5 patches, and these GPU boxes are reclaimed on a schedule with no
warning. Losing the store to a reclaim would also contradict the paper's central
property: a memory that never deletes.

The fix is the standard write-temp-then-rename dance. `os.replace` is atomic
within a filesystem, so a reader sees either the complete previous file or the
complete new one, never a torn one. The fsync before it forces the bytes out of
the page cache first — without it the rename can land while the data has not,
which is the classic way "atomic rename" still loses data on a hard crash.

Same-directory temp file: rename is only atomic within a filesystem, and the
store lives on ceph while /tmp does not.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write bytes so a crash can never leave a partial file at `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)          # atomic within the filesystem
    except BaseException:
        # Leave `path` untouched — the previous good copy survives.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_pickle_dump(obj: Any, path: str | Path, protocol: int | None = None) -> None:
    """Pickle `obj` to `path` atomically.

    Serializes to memory first so a pickling error (an unpicklable field slipped
    into the store) fails before anything on disk is touched, rather than
    half-way through writing.
    """
    blob = pickle.dumps(obj, protocol if protocol is not None else pickle.HIGHEST_PROTOCOL)
    atomic_write_bytes(path, blob)
