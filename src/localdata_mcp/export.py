"""Writing query results out to a file.

Two decisions worth stating, because both are refusals:

* **An existing file is not replaced unless the caller says so.** SQLite's own
  ``VACUUM INTO`` behaves this way and it is the right default — a silently
  replaced file is unrecoverable, while a refusal costs one retry.
* **Exports are created ``0o600``.** A file written by SQLite lands ``0o644`` by
  default, world-readable on a shared host, containing the user's actual data.
  Rows leaving a database are exactly the payload that should not be readable by
  every account on the machine. Callers who want it shared can widen it.

Values are written as the query returned them. A temporal column holds integer
ticks, so it exports as integers; formatting it for human eyes is the caller's
choice, expressed in SQL, rather than a transformation applied silently here.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .paths import resolve_write_path

__all__ = ["ExportResult", "export_csv"]

#: Owner read/write only.
_EXPORT_MODE = 0o600


@dataclass(frozen=True)
class ExportResult:
    path: str
    row_count: int
    columns: list[str]
    replaced_existing: bool


def export_csv(
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    raw_path: str,
    *,
    overwrite: bool = False,
) -> ExportResult:
    """Write ``rows`` to ``raw_path`` as CSV.

    Refuses an existing target unless ``overwrite`` is set, and refuses any path
    outside the allowed root.
    """
    path = resolve_write_path(raw_path, overwrite=overwrite)
    replaced = path.exists()

    written = 0
    # Open through a file descriptor so the mode is set at creation rather than
    # after a window in which the file exists with the default mode.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _EXPORT_MODE)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(row)
                written += 1
    except Exception:
        # A partial file is worse than none: it looks like a complete export.
        _remove_quietly(path, existed_before=replaced)
        raise

    # O_CREAT honours the umask, so an inherited umask can widen the mode. Set it
    # explicitly once the file exists.
    os.chmod(path, _EXPORT_MODE)

    return ExportResult(
        path=str(path),
        row_count=written,
        columns=list(columns),
        replaced_existing=replaced,
    )


def _remove_quietly(path: Path, *, existed_before: bool) -> None:
    if existed_before:
        # We truncated a file the caller authorised us to replace. Deleting it
        # now would destroy more than we were asked to.
        return
    try:
        path.unlink()
    except OSError:
        pass
