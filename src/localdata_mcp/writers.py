"""The writers: one function per output format, and nothing above them.

Each takes the header, the rows and a path that already exists and is empty,
and returns how many rows it wrote. Opening the file is the writer's own job,
because a format whose library owns the handle cannot be handed one.

They live apart from ``export`` so that ``formats`` — the one table naming what
reads and what writes each suffix — can reach them without importing the module
that *calls* them. Nothing here resolves a path, decides whether a destination
may be overwritten, or knows what a slot is: a writer is handed a path that has
already been judged safe, and its only remaining question is how to lay the rows
out in the format it is named for.

A writer that fails part-way leaves a partial file behind. Deleting it is the
caller's job (``export.export_rows``), not the writer's, because only the caller
knows whether the file was one it had just created or one that was already
there.
"""

from __future__ import annotations

import csv
import importlib
import json
import re
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Sequence
from xml.sax.saxutils import escape

from .errors import ExportError, undeclared_separator

__all__ = ["SPREADSHEET_ROW_LIMIT", "Writer"]


#: A writer receives the header, the rows and a path that exists and is empty,
#: and returns how many rows it wrote. Opening the file is the writer's job,
#: because a format whose library owns the handle cannot be handed one.
Writer = Callable[[Sequence[str], Iterable[Sequence[object]], Path], int]


def write_delimited(
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


def delimited_writer(delimiter: str) -> Writer:
    """A writer for character-separated text, at the separator the caller gave.

    The exact mirror of ``readers.delimited``, and mirrored on purpose: the
    separator is declared on both sides or the round trip stops being one. A
    factory rather than a parameter so it does not have to be threaded through
    the signature of every other format's writer, none of which has one.
    """

    def write(
        columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
    ) -> int:
        return write_delimited(columns, rows, path, delimiter=delimiter)

    return write


def writing_needs_a_separator(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    """The writer every character-separated format has until one is declared.

    The mirror of ``readers.reading_needs_a_separator``, and the reason the read
    side's change could not be made alone. While the suffix supplied the
    separator it supplied it symmetrically — ``.tsv`` read at a tab and written
    at a tab — so a file this server wrote was one it could read back. Take the
    default off the read side only and that property breaks; take it off both
    and it holds, because the caller names the character in each direction.

    Leaving the write side defaulting would have been worse than asymmetric. It
    would write commas into a file called ``.tsv`` on request, which is the
    shape of the defect the format table exists to refuse: a file whose name
    disagrees with its contents, produced by a call that answered ``ok``.
    """
    raise ExportError(undeclared_separator(path.name, reading=False))


def write_json(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    """An array of row objects, one per line.

    One object per line rather than indented: this is the format a large result
    is written in, and indentation costs bytes per value on a file whose whole
    reason for existing is that it was too big to return. It stays the shape
    ``readers.read_json`` takes back, which is what makes the round trip exact.
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


def write_jsonl(
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


def write_xml(
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


def _require(module: str, extra: str, doing: str):
    """Import an optional format library, or say how to install it.

    The writing counterpart of ``loader._require``, and separate from it because
    the failure is a different kind: this one refuses a *destination* the caller
    named, so it raises ``ExportError`` and the partial file is cleaned up.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ExportError(
            f"{doing} needs {module}, which is not installed. Install it with: "
            f"pip install 'localdata-mcp[{extra}]' (or [all] for every format)."
        ) from exc


#: Rows per ``safe_dump`` call. Big enough that the per-call overhead is lost in
#: the serialising, small enough that a chunk is never the peak.
_YAML_CHUNK = 1_000


def write_yaml(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    """A sequence of mappings, dumped a chunk at a time.

    **The file is byte-for-byte what one ``safe_dump`` of the whole result would
    have written**, which is what makes chunking safe rather than merely
    cheaper. A top-level sequence dumped in pieces concatenates into one
    sequence: every item starts at column zero with ``- ``, so no piece is
    indented relative to another and no document marker separates them. Nothing
    here is aliased across a chunk boundary either — ``SafeDumper`` aliases
    collections, and a row holds only scalars.

    Dumping the whole result at once cost three copies of it: the list of rows
    that arrived, a dict per row collected into a second list, and the
    representation ``safe_dump`` builds of all of that before writing a byte.
    Measured at 200,000 rows of four columns, the peak was 540 MB and grew
    exactly with the row count.

    ``rows`` is made an iterator first, and that is load-bearing rather than
    tidiness: ``islice`` over a *list* restarts from the beginning every time,
    so a caller passing a materialised result would loop here forever.
    """
    yaml = _require("yaml", "yaml", "Writing YAML")

    # libyaml where PyYAML was built against it, the Python emitter where it was
    # not. Measured byte-identical on this writer's input and about 1.3x faster,
    # and the equality test above is what keeps that true: it compares the file
    # to a plain `safe_dump`, so a release where the two emitters diverged would
    # fail rather than quietly change the shape of everyone's YAML.
    dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)

    def dump(records: list[dict], handle) -> None:
        yaml.dump_all(
            [records], handle, Dumper=dumper, sort_keys=False, allow_unicode=True
        )

    remaining = iter(rows)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        while True:
            chunk = [dict(zip(columns, row)) for row in islice(remaining, _YAML_CHUNK)]
            if not chunk:
                break
            dump(chunk, handle)
            written += len(chunk)

        # An empty result is still a document. Left as a zero-length file it
        # would parse back as null rather than as no rows.
        if not written:
            dump([], handle)
    return written


def write_markdown(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    """A GitHub-flavoured table.

    Write-only, and it is the one format here that is deliberately lossy: a
    Markdown table has no types and no quoting, so there is no reader for it
    and none is offered. It exists because a result is sometimes wanted for a
    document rather than for another program.
    """
    tabulate = _require("tabulate", "markdown", "Writing Markdown")
    materialised = [list(row) for row in rows]
    path.write_text(
        tabulate.tabulate(materialised, headers=list(columns), tablefmt="github")
        + "\n",
        encoding="utf-8",
    )
    return len(materialised)


def write_columnar(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    """Parquet, Feather/Arrow or ORC, chosen by the suffix.

    The only writers here that cannot stream: a columnar file stores each column
    contiguously, so the whole result has to exist before any of it can be
    written. That is the format's shape, not an oversight.
    """
    _require("pyarrow", "parquet", f"Writing {path.suffix.lower()}")
    import pandas as pd

    frame = pd.DataFrame(list(rows), columns=list(columns))
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".orc":
        frame.to_orc(path, index=False)
    else:
        import pyarrow as pa

        table = pa.Table.from_pandas(frame, preserve_index=False)
        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
    return len(frame)


#: The most rows this server will write to a spreadsheet.
#:
#: Not the format's limit. A modern worksheet holds 1,048,576 rows and the older
#: one held 65,536, and this sits just under the older figure deliberately: a
#: spreadsheet is something a person opens and looks at, and a million-row
#: workbook is not that. It is a file that takes minutes to write, gigabytes to
#: build, and that no spreadsheet application opens comfortably.
#:
#: It bounds the memory too, which is the measured half. Both writers build the
#: whole document before writing any of it (§9.2), so the export peak scales
#: with the result: a million rows of eleven columns cost 12.9 GB as `.xlsx`,
#: and `.ods` crossed 16 GB without finishing. At this cap the same shape is
#: about a fifteenth of that.
#:
#: **Write side only.** Reading a large spreadsheet somebody else produced is
#: untouched — the limit is on what this server chooses to emit, not on what it
#: will accept.
SPREADSHEET_ROW_LIMIT = 65_535


def workbook_writer(module: str, extra: str) -> Writer:
    """A writer for one workbook format, bound to the library that writes it.

    The mirror of ``readers.workbook``, and a factory for the same reason: the
    engine is declared once per format in ``formats`` and reaches both sides
    from there. It used to be chosen here by a ternary on the suffix, with
    openpyxl as the fallback for anything unrecognised — which is how a
    silently-wrong file gets written (#99).
    """

    def write_workbook(
        columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
    ) -> int:
        """One sheet, named for the file.

        A result is one table, so it is one sheet — writing a workbook of several
        would need several results, which is not what this verb is handed. ``.xls``
        is absent from the writers on purpose: xlrd dropped writing and nothing
        maintained replaces it, so it is read-only here.

        ``engine`` is passed explicitly, and that is not belt-and-braces. Pandas
        infers the engine from the suffix of a ``str`` path but **not** of a
        ``Path`` — measured on pandas 3.0.2 — and this function is handed a
        ``Path``, so inference quietly fell back to openpyxl and every ``.ods``
        export was a workbook wearing an OpenDocument name. It was invisible for as
        long as it was because both formats are Zip archives that open ``PK\\x03\\x04``,
        and because pandas reads a workbook back by sniffing its contents rather
        than trusting the suffix, so the round trip returned the right rows out of
        the wrong file. We already know which engine we want here — the ``_require``
        check above names it — so saying so costs one argument and cannot regress.
        """
        suffix = path.suffix.lower()
        _require(module, extra, f"Writing {suffix}")
        import pandas as pd

        # Take one row more than the limit and no further. Both spreadsheet writers
        # build the entire document in memory before a byte reaches the disk, so the
        # count has to be settled *before* the frame is built or the refusal costs
        # the same memory as the export it is refusing. Measured at a million rows
        # of eleven columns: `.xlsx` reached 12.9 GB, and `.ods` crossed 16 GB
        # without producing a file at all.
        collected = list(islice(rows, SPREADSHEET_ROW_LIMIT + 1))
        if len(collected) > SPREADSHEET_ROW_LIMIT:
            raise ExportError(
                f"More than {SPREADSHEET_ROW_LIMIT:,} rows will not be written to "
                f"{suffix}. A spreadsheet is a format for reading, and both writers "
                f"build the whole document in memory before writing any of it, so a "
                f"result this size costs gigabytes and produces a file no "
                f"spreadsheet opens comfortably. Ask for .csv, .parquet or .jsonl, "
                f"none of which have a row limit — or narrow the result with LIMIT "
                f"if a spreadsheet is what you need."
            )

        frame = pd.DataFrame(collected, columns=list(columns))
        frame.to_excel(
            path, sheet_name=path.stem[:31] or "Sheet1", index=False, engine=module
        )
        return len(frame)

    return write_workbook
