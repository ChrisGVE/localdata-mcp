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

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape

from .paths import resolve_write_path

__all__ = ["ExportError", "ExportResult", "WRITERS", "export_rows"]

#: Owner read/write only.
_EXPORT_MODE = 0o600


class ExportError(ValueError):
    """A result this server will not write in the form it was asked for."""


@dataclass(frozen=True)
class ExportResult:
    path: str
    row_count: int
    columns: list[str]


#: A writer receives the header, the rows and a path that exists and is empty,
#: and returns how many rows it wrote. Opening the file is the writer's job,
#: because a format whose library owns the handle cannot be handed one.
Writer = Callable[[Sequence[str], Iterable[Sequence[object]], Path], int]


def _write_delimited(
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    path: Path,
    *,
    delimiter: str,
) -> int:
    written = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)
            written += 1
    return written


def _write_csv(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    return _write_delimited(columns, rows, path, delimiter=",")


def _write_tsv(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    return _write_delimited(columns, rows, path, delimiter="\t")


def _write_json(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    """An array of row objects, one per line.

    One object per line rather than indented: this is the format a large result
    is written in, and indentation costs bytes per value on a file whose whole
    reason for existing is that it was too big to return. It stays the shape
    ``_read_json`` takes back, which is what makes the round trip exact.
    """
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for row in rows:
            if written:
                handle.write(",\n")
            handle.write("  " + _as_json(columns, row))
            written += 1
        handle.write("\n]\n" if written else "]\n")
    return written


def _write_jsonl(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_as_json(columns, row) + "\n")
            written += 1
    return written


def _as_json(columns: Sequence[str], row: Sequence[object]) -> str:
    return json.dumps(dict(zip(columns, row)), default=_unserializable)


def _unserializable(value: object) -> str:
    """Refuse rather than invent a spelling for a value JSON has no type for.

    Reached for ``bytes`` out of a BLOB column, which has no JSON form that
    round-trips — base64 would come back a string and the loss would be silent.
    The partial file is removed by ``export_rows``, so the refusal costs nothing.
    """
    raise ExportError(
        f"JSON has no type for {type(value).__name__} ({value!r}). Convert the "
        f"column in SQL — hex() for binary, or a CAST — or export to CSV."
    )


#: What XML will accept as an element name. Conservative against the spec, which
#: also permits a large range of Unicode: a name outside this set is refused
#: rather than rewritten, so no column silently changes its name on the way out.
_XML_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*\Z")


def _write_xml(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    """``<rows><row><column>value</column></row></rows>``.

    **A NULL is an absent element, not an empty one.** ``<b></b>`` reads back as
    text and stops being a NULL, so the round trip would quietly turn every
    missing value into a present empty one.

    A column whose name XML cannot spell is refused by name. The alternative is
    inventing a spelling — ``first name`` becoming ``first_name`` — which the
    caller never asked for and would not be told about; the remedy is one ``AS``
    in their own SQL.
    """
    for column in columns:
        if not _XML_NAME.match(str(column)):
            raise ExportError(
                f"{column!r} cannot be an XML element name, so this result "
                f"cannot be written as XML without renaming a column. Alias it "
                f"in the SQL — SELECT {column!r} AS a_name — or export to CSV, "
                f"which puts no constraint on column names."
            )

    written = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n<rows>\n')
        for row in rows:
            handle.write("  <row>\n")
            for column, value in zip(columns, row):
                if value is None:
                    continue
                handle.write(f"    <{column}>{escape(str(value))}</{column}>\n")
            handle.write("  </row>\n")
            written += 1
        handle.write("</rows>\n")
    return written


#: Extension to writer, the counterpart of ``loader.READERS``. A new output
#: format is one entry here; nothing upstream of it needs to know.
WRITERS: dict[str, Writer] = {
    ".csv": _write_csv,
    ".tsv": _write_tsv,
    ".xml": _write_xml,
    # As on the read side, `.txt` is treated as comma-separated. The two
    # registries agree, so a file this server writes is a file it can read back.
    ".txt": _write_csv,
    ".json": _write_json,
    ".jsonl": _write_jsonl,
    ".ndjson": _write_jsonl,
}


def export_rows(
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    raw_path: str,
    *,
    force: bool = False,
    claimed: Mapping[Path, str] | None = None,
) -> ExportResult:
    """Write ``rows`` to ``raw_path`` in the format its suffix names.

    Refuses a suffix with no writer, a path outside the allowed root, a file a
    live slot is sitting on, and an existing file unless ``force``.
    """
    # Before `resolve_write_path`, deliberately: that call deletes an existing
    # target under `force`, and a request this server was never going to be able
    # to satisfy must not cost the user a file on its way to failing.
    writer = _writer_for(raw_path)

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
