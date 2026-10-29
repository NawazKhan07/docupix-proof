"""Work done before the reviewer presses Compare, kept for when they do (inside the worker process).

The slow part of a comparison is getting the two files ready to be compared -- rendering them, reading a PDF's own
text and lines, lifting the artwork off each sheet, dividing it into panels, aligning the files and every panel -- and
none of it depends on anything the reviewer does after loading the files but choose the mode. So as soon as both files
are loaded the app asks the worker to PREPARE the comparison it would run right now (`worker_process`: the "prepare"
message), and Compare then finds the work done:

* The whole outcome of a prepared request is kept (`outcome_for` / `keep_outcome`): pressing Compare with nothing
  changed shows it at once.
* Each slow stage is kept by what it was computed from (`remembered`): a request that differs -- another mode, an
  outline drawn by hand -- still reuses every stage whose inputs are the same, and redoes only the rest.

Nothing here decides anything: a stage's result is exactly what computing it again would give, keyed by everything it
depends on (a file by its path, size and modification time; a picture by a digest of its pixels; settings by value).

A preparation gives way the moment the reviewer does something (`Superseded`): it is abandoned at its next stage, so it
never holds up the comparison actually asked for by more than one stage, and never runs beside it.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import pickle
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_MAX_PER_STAGE = 2  # results kept per stage: the pair being prepared and the one before it (a file swapped back)
_STAGE_LIMITS = {"render": 4, "pdf_extras": 4, "leaflet_align": 32}  # (a stage kept per FILE, or per page, keeps more)
_MIN_FREE_FOR_PICTURES = 2 * 1024**3  # pictures are kept only while this much memory is free (bytes)


class Superseded(Exception):
    """A preparation was overtaken by what the reviewer did next (a new request, or Compare itself)."""


_lock = threading.Lock()
_stages: dict[str, OrderedDict] = {}
_outcome: tuple[bytes, Any] | None = None


def file_key(path: Path | str) -> tuple:
    """What a file on disk is, for keeping work done on it: its path, size and when it was last written."""
    stat = os.stat(path)
    return (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)


def digest(array: np.ndarray | None) -> bytes | None:
    """A short fingerprint of a picture's pixels (and shape): two pictures with the same digest are the same picture."""
    if array is None:
        return None
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(repr((contiguous.shape, contiguous.dtype.str)).encode())
    hasher.update(memoryview(contiguous).cast("B"))
    return hasher.digest()


def settings_key(*values: Any) -> bytes:
    """Settings (dataclasses, geometries, numbers) as a key: by value."""
    return pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL)


def remembered(stage: str, key: Any, compute: Callable[[], Any]) -> Any:
    """`compute()`'s result for `key`, computed once and kept (a few per stage); every caller gets its own copy, so
    nothing one comparison does to it can reach another."""
    with _lock:
        kept = _stages.setdefault(stage, OrderedDict())
        if key in kept:
            kept.move_to_end(key)
            found = kept[key]
            logger.info("prepared stage reused", extra={"stage": stage})
            return _copy(found)
    result = compute()
    if _picture_bytes(result) > 0 and not _room_for_pictures():
        return result  # memory is short: the comparison asked for comes first, not keeping its work
    with _lock:
        kept = _stages.setdefault(stage, OrderedDict())
        kept[key] = result
        kept.move_to_end(key)
        while len(kept) > _STAGE_LIMITS.get(stage, _MAX_PER_STAGE):
            kept.popitem(last=False)
    return _copy(result)


def _picture_bytes(value: Any) -> int:
    """How many bytes of pictures a result holds (the only part of it that is big)."""
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (tuple, list)):
        return sum(_picture_bytes(item) for item in value)
    return 0


def _room_for_pictures() -> bool:
    from app.engine.file_loader import _memory_bytes

    measured = _memory_bytes()
    return measured is None or measured[1] >= _MIN_FREE_FOR_PICTURES


def outcome_for(key: bytes) -> Any | None:
    """The whole outcome of a request already prepared with exactly this key, if there is one."""
    with _lock:
        if _outcome is not None and _outcome[0] == key:
            logger.info("prepared comparison reused")
            return _outcome[1]
    return None


def keep_outcome(key: bytes, outcome: Any) -> None:
    global _outcome
    with _lock:
        _outcome = None  # (the last one goes first, whatever: it is not the one asked for now)
    if _room_for_pictures():
        with _lock:
            _outcome = (key, outcome)


def forget() -> None:
    """Drop everything kept (tests; and a worker that is told to start clean)."""
    global _outcome
    with _lock:
        _stages.clear()
        _outcome = None


def _copy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)
