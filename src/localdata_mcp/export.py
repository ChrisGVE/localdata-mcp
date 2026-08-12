"""Writing query results out to a file.

**The suffix chooses the format, and one it cannot write is refused by name.**
This mirrors ``loader.read_file``, which refuses a suffix it has no reader for,
and it replaces an earlier arrangement where every export was CSV whatever the
name said — ``out.parquet`` came back ``ok: true`` holding comma-separated text.
A file whose name lies about its contents is the worst of the three possible
answers, because nothing anywhere reports it.

Two further decisions worth stating, because both are refusals:

* **An existing file is replaced only when ``force`` says so**, and ``force``
  means the user was asked and answered — the destination is a name they chose
  and the agent relayed (``paths``). A file some live slot is sitting on is
  refused regardless.
* **Exports are created ``0o600``.** A file written by SQLite lands ``0o644`` by
  default, world-readable on a shared host, containing the user's actual data.
  Rows leaving a database are exactly the payload that should not be readable by
  every account on the machine. Callers who want it shared can widen it.

Values are written as the query returned them. A temporal column holds integer
ticks, so it exports as integers; formatting it for human eyes is the caller's
choice, expressed in SQL, rather than a transformation applied silently here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Defined in `errors` rather than here, for the same reason `LoadError` is: the
# format layer below raises it. Imported into this namespace deliberately — the
# tests and `server` both say `export.ExportError`.
from .errors import ExportError
from .loader import DELIMITED as _DELIMITED
from .paths import resolve_write_path
from .writers import (
    SPREADSHEET_ROW_LIMIT,
    Writer,
    _delimited_writer,
    _write_columnar,
    _write_csv,
    _write_json,
    _write_jsonl,
    _write_markdown,
    _write_tsv,
    _write_workbook,
    _write_xml,
    _write_yaml,
)

__all__ = [
    "DELIMITED",
    "SPREADSHEET_ROW_LIMIT",
    "ExportError",
    "ExportResult",
    "WRITERS",
    "export_rows",
]

#: Owner read/write only.
_EXPORT_MODE = 0o600


@dataclass(frozen=True)
class ExportResult:
    path: str
    row_count: int
    columns: list[str]


#: Extension to writer, the counterpart of ``loader.READERS``. A new output
#: format is one entry here; nothing upstream of it needs to know.
WRITERS: dict[str, Writer] = {
    ".xlsx": _write_workbook,
    ".ods": _write_workbook,
    ".csv": _write_csv,
    ".tsv": _write_tsv,
    ".xml": _write_xml,
    ".yaml": _write_yaml,
    ".yml": _write_yaml,
    ".md": _write_markdown,
    ".parquet": _write_columnar,
    ".feather": _write_columnar,
    ".orc": _write_columnar,
    # As on the read side, `.txt` is treated as comma-separated. The two
    # registries agree, so a file this server writes is a file it can read back.
    ".txt": _write_csv,
    ".json": _write_json,
    ".jsonl": _write_jsonl,
    ".ndjson": _write_jsonl,
}

#: The output formats a delimiter means anything for — the read side's set, not
#: a copy of it. The two sides describe one fact about a file, so a format this
#: server writes with a separator is one it can read back with it, and holding
#: the fact twice is how that stops being true. Re-exported rather than left to
#: callers to reach through `loader`, so `export.DELIMITED` keeps working.
DELIMITED = _DELIMITED


def export_rows(
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    raw_path: str,
    *,
    force: bool = False,
    claimed: Mapping[Path, str] | None = None,
    delimiter: str | None = None,
) -> ExportResult:
    """Write ``rows`` to ``raw_path`` in the format its suffix names.

    Refuses a suffix with no writer, a path outside the allowed root, a file a
    live slot is sitting on, and an existing file unless ``force``.

    ``delimiter`` replaces the separator the suffix implied, and is **ignored**
    for a format that has none. That is the opposite of the read side, and the
    asymmetry is real rather than an oversight: reading at the wrong separator
    changes what the data *is*, so a delimiter that cannot apply means the
    caller has misunderstood the file and is worth stopping. Writing Parquet
    produces correct Parquet whatever this says, so the parameter is merely
    inert — and refusing it would block a caller carrying one default delimiter
    across a mix of destinations.
    """
    # Before `resolve_write_path`, deliberately: that call deletes an existing
    # target under `force`, and a request this server was never going to be able
    # to satisfy must not cost the user a file on its way to failing.
    writer = _writer_for(raw_path)

    if delimiter is not None and Path(raw_path).suffix.lower() in DELIMITED:
        if len(delimiter) != 1:
            raise ExportError(
                f"delimiter must be a single character, not {delimiter!r}."
            )
        writer = _delimited_writer(delimiter)

    path = resolve_write_path(raw_path, force=force, claimed=claimed)

    # Create it here, ahead of the writer, so the file is private from the moment
    # it exists. A writer that opens the path itself — which any library-backed
    # format will — would otherwise create it 0o644 and leave the data readable
    # for the window before the chmod below.
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _EXPORT_MODE))

    try:
        written = writer(columns, rows, path)
    except Exception:
        # A partial file is worse than none: it looks like a complete export.
        # Safe to delete unconditionally — resolve_write_path returned a path
        # with nothing at it, so this file is one we just created.
        _remove_quietly(path)
        raise

    # O_CREAT is subject to the umask, and a writer may have reopened the path.
    # Set the mode explicitly now the file is complete.
    os.chmod(path, _EXPORT_MODE)

    return ExportResult(
        path=str(path),
        row_count=written,
        columns=list(columns),
    )


def _writer_for(raw_path: str) -> Writer:
    suffix = Path(raw_path).suffix.lower()
    writer = WRITERS.get(suffix)
    if writer is not None:
        return writer

    supported = ", ".join(sorted(WRITERS))
    named = repr(suffix) if suffix else "a name with no suffix"
    raise ExportError(
        f"No writer for {named}. The suffix chooses the format. Supported: {supported}"
    )


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
