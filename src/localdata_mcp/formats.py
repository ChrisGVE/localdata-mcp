"""One table of what this server knows about file formats.

Everything keyed by a file suffix is declared here, once. Before this module
the same knowledge sat in four tables in two modules — ``READERS`` and
``DELIMITED`` and ``STREAMED`` in ``loader``, ``WRITERS`` and a second
``DELIMITED`` in ``export`` — none of which knew about the others. Adding a
format meant four edits, and forgetting one was silent: the suffix would read
but not stream, or write but ignore the separator it was handed, and the answer
in each case was a plausible ``ok: true``.

The four tables still exist as names, at the bottom of this file, because the
tests and the documentation say ``READERS`` and ``WRITERS``. They are **views**
now — each one derived from ``FORMATS`` in a single expression — so they cannot
disagree with it or with each other.

**This table is closed-world, and that is the opposite of ``dialects.BACKENDS``
on purpose.** An unregistered dialect there gets a generic ``Backend`` that
works, because a database this server has never heard of still speaks SQL
through SQLAlchemy and guessing is a fair bet. A suffix absent from ``FORMATS``
has no such fallback: there is no generic way to read a file whose format
nobody declared, and the failure mode of guessing is the one this server exists
to prevent — data that loaded, looks fine, and is wrong. So absence here means
refusal, by name, with the supported list attached. The asymmetry between the
two registries is load-bearing rather than an inconsistency to iron out.

**Where a format is one this server cannot do both ways, ``None`` says so.**
Read-only and write-only are the absence of a callable rather than membership
of some other set: ``.md`` has no reader because a Markdown table has no types
and no quoting, and ``.xls`` has no writer because xlrd dropped writing. That is
also what makes "readable" and "writable" answerable from this one table instead
of by subtracting one dict from another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .errors import ExportError, LoadError, undeclared_separator
from .readers import (
    ReadResult,
    Reader,
    delimited as delimited_reader,
    read_columnar,
    read_json,
    read_jsonl,
    read_numbers,
    read_xml,
    read_yaml,
    workbook,
)
from .writers import (
    Writer,
    delimited_writer,
    workbook_writer,
    write_columnar,
    write_json,
    write_jsonl,
    write_markdown,
    write_xml,
    write_yaml,
)

__all__ = [
    "DELIMITED",
    "FORMATS",
    "READERS",
    "STREAMED",
    "WRITERS",
    "Format",
    "resolve_separator",
]


@dataclass(frozen=True)
class Format:
    """Everything keyed by one file suffix.

    ``Format`` is to a suffix what ``dialects.Backend`` is to a dialect name:
    the place per-format knowledge is *declared* rather than rediscovered by a
    branch at each site that needs it.

    ``reader`` and ``writer`` are ``None`` when the format cannot be read or
    cannot be written; at least one of them is always present, since a format
    this server can do neither way has no reason to be in the table.

    ``delimited`` says the file is character-separated text rather than
    something carrying its own structure — and therefore that a separator
    means something for it, in both directions. It belongs on the format rather
    than on either side because it is one fact governing two rules: a delimited
    format is read and written at a separator, and a format that is *not*
    delimited refuses a separator on the read side (the caller has misread the
    file) and ignores one on the write side (the parameter is merely inert, and
    refusing would block a caller carrying one default across a mix of
    destinations).

    ``standard_delimiter`` is the separator a published standard gives the
    format, used when the caller declares none: a comma for ``.csv`` (RFC 4180)
    and a tab for ``.tsv`` (the IANA ``text/tab-separated-values``
    registration). This is project first principle 7 — the standard is the
    default, and a value the caller passes always wins. ``None`` means no
    standard exists, as for ``.txt``, and then nothing is assumed: the format
    refuses to be read or written until a separator is declared. It is the
    only place a default separator is written down; ``resolve_separator``
    reads it for every path that reads or writes delimited text.

    ``streamed`` says the reader can produce the file in chunks instead of
    materialising it whole. It is a property of the format, not a preference: a
    columnar file stores each column contiguously and a workbook is a Zip
    archive, so neither can be handed out a row at a time.
    """

    suffix: str
    reader: Reader | None = None
    writer: Writer | None = None
    delimited: bool = False
    streamed: bool = False
    standard_delimiter: str | None = None

    def __post_init__(self) -> None:
        if self.reader is None and self.writer is None:
            raise ValueError(
                f"{self.suffix} declares neither a reader nor a writer, so "
                f"nothing about it can be acted on. Leave it out of the table."
            )
        if self.delimited and (self.reader is None or self.writer is None):
            raise ValueError(
                f"{self.suffix} is declared delimited but is missing a "
                f"{'reader' if self.reader is None else 'writer'}. A delimited "
                f"format has to go both ways: the separator is declared on each "
                f"side, and a file written at one this server could not read "
                f"back at is the property `delimited` exists to keep."
            )
        if self.standard_delimiter is not None and not self.delimited:
            raise ValueError(
                f"{self.suffix} declares a standard separator but is not "
                f"delimited. Only character-separated text has a separator, "
                f"so only a delimited format can have a standard one."
            )
        if self.standard_delimiter is not None and len(self.standard_delimiter) != 1:
            raise ValueError(
                f"{self.suffix} declares the standard separator "
                f"{self.standard_delimiter!r}, but a separator is one character."
            )
        if self.streamed and self.reader is None:
            raise ValueError(
                f"{self.suffix} is declared streamed but has no reader — "
                f"streaming here means reading in chunks."
            )


#: The workbook libraries, as ``(module, extra)``: what pandas dispatches on
#: and what the module is imported as, then this project's optional-dependency
#: group, which is what a refusal tells the user to install. They differ — the
#: `ods` extra installs `odfpy`, whose module is `odf` — so both are carried.
#: Each is written here once and reaches the reader and the writer alike.
_OPENPYXL = ("openpyxl", "excel")
_ODF = ("odf", "ods")
_XLRD = ("xlrd", "xls")


def _format(suffix: str, **known: object) -> tuple[str, Format]:
    """One entry, with the suffix said once rather than as key and field."""
    return suffix, Format(suffix=suffix, **known)  # type: ignore[arg-type]


def resolve_separator(name: str, declared: str | None, *, reading: bool) -> str:
    """The character a delimited file is split on, or the reason there is none.

    The one rule every path that reads or writes delimited text goes through —
    ``attach`` and ``create(source=…)``, read whole or in chunks, and
    ``query(path=…)`` — so none of them carries a default of its own:

    1. A separator the caller declared wins, on every suffix. A ``;`` declared
       for a ``.csv`` reads and writes at ``;``.
    2. Otherwise the format's ``standard_delimiter``: a comma for ``.csv``, a
       tab for ``.tsv``. Project first principle 7, in Chris's words: *"the
       standard is always the default, unless the LLM provides something
       else"*. A file named ``.csv`` that is really separated by ``;`` loads
       wrongly at the default, and ``readers.fat_column_note`` warns, naming
       the ``;``, in the two shapes its docstring describes.
    3. Otherwise refuse, because no standard exists to fall back on — ``.txt``.
       ``reading`` picks the error type: a ``LoadError`` for a source that is
       already there, an ``ExportError`` for a destination not yet written.

    Because both directions resolve the same way, a file this server writes
    with nothing declared is one it reads back with nothing declared.

    ``name`` is a file name or path; only its suffix is looked up, and it must
    be a delimited one. Whether a declared separator is one character is
    checked by the callers, before this, in their own words.
    """
    if declared is not None:
        return declared
    standard = FORMATS[Path(name).suffix.lower()].standard_delimiter
    if standard is not None:
        return standard
    error = LoadError if reading else ExportError
    raise error(undeclared_separator(Path(name).name, reading=reading))


def read_delimited_text(path: Path) -> ReadResult:
    """The table's reader for delimited text: at the standard, or refuse.

    This is what the format table binds, so ``READERS`` reads a ``.csv`` the
    way ``attach`` does when nothing is declared. A declared separator never
    reaches it — the loader builds a reader at that separator instead.
    """
    return delimited_reader(resolve_separator(path.name, None, reading=True))(path)


def write_delimited_text(
    columns: Sequence[str], rows: Iterable[Sequence[object]], path: Path
) -> int:
    """The table's writer for delimited text: at the standard, or refuse.

    The mirror of ``read_delimited_text``, so a file written through ``WRITERS``
    with nothing declared reads back through ``READERS`` with nothing declared.
    """
    separator = resolve_separator(path.name, None, reading=False)
    return delimited_writer(separator)(columns, rows, path)


#: Character-separated text: one format under three names. `.csv`, `.tsv` and
#: `.txt` share one reader, one writer and the same flags, so they are declared
#: from this one dictionary rather than three that have to be kept saying the
#: same thing. What differs is the separator a standard gives each, which is
#: why `standard_delimiter` is set per entry below and nowhere else.
_DELIMITED_TEXT: dict[str, object] = dict(
    reader=read_delimited_text,
    writer=write_delimited_text,
    delimited=True,
    streamed=True,
)


#: The whole catalogue. A new format is one entry here and nothing else.
FORMATS: dict[str, Format] = dict(
    (
        # RFC 4180 makes the comma the CSV separator; the IANA registration of
        # text/tab-separated-values makes it the tab for TSV. `.txt` has no
        # standard, so it has no default and a separator must be declared.
        _format(".csv", standard_delimiter=",", **_DELIMITED_TEXT),
        _format(".tsv", standard_delimiter="\t", **_DELIMITED_TEXT),
        _format(".txt", **_DELIMITED_TEXT),
        # Formats carrying their own structure. A delimiter means nothing for
        # any of them, which is why `delimited` is left false rather than
        # restated.
        _format(".json", reader=read_json, writer=write_json),
        _format(".jsonl", reader=read_jsonl, writer=write_jsonl),
        _format(".ndjson", reader=read_jsonl, writer=write_jsonl),
        _format(".xml", reader=read_xml, writer=write_xml),
        _format(".yaml", reader=read_yaml, writer=write_yaml),
        _format(".yml", reader=read_yaml, writer=write_yaml),
        # Columnar. Typed, so nothing is inferred on the way in — and not
        # streamable, because a columnar file stores each column contiguously
        # and the whole result has to exist before any of it can be written.
        _format(".parquet", reader=read_columnar, writer=write_columnar),
        _format(".feather", reader=read_columnar, writer=write_columnar),
        _format(".orc", reader=read_columnar, writer=write_columnar),
        # Workbooks. The engine each one needs is spelled once and reaches both
        # sides from here — it used to be a table on the read side and a ternary
        # on the write side, which is how `.ods` exports were silently openpyxl
        # workbooks for as long as they were (#99, docs/CONSTRAINTS.md).
        #
        # `.xls` and `.xlsm` are read-only: xlrd dropped writing, and `.xlsm` is
        # simply absent from the writers though openpyxl could write it.
        _format(
            ".xlsx", reader=workbook(*_OPENPYXL), writer=workbook_writer(*_OPENPYXL)
        ),
        _format(".ods", reader=workbook(*_ODF), writer=workbook_writer(*_ODF)),
        _format(".xlsm", reader=workbook(*_OPENPYXL)),
        _format(".xls", reader=workbook(*_XLRD)),
        # Apple Numbers: read through its own library, no writer.
        _format(".numbers", reader=read_numbers),
        # Write-only, deliberately: a Markdown table has no types and no
        # quoting, so no reader could return what went in.
        _format(".md", writer=write_markdown),
    )
)


#: Suffix to reader, for the formats that have one. A view of `FORMATS`, not a
#: second table — the same for the three below.
READERS: dict[str, Reader] = {
    suffix: fmt.reader for suffix, fmt in FORMATS.items() if fmt.reader is not None
}

#: Suffix to writer, for the formats that have one.
WRITERS: dict[str, Writer] = {
    suffix: fmt.writer for suffix, fmt in FORMATS.items() if fmt.writer is not None
}

#: The formats a delimiter means anything for.
DELIMITED: frozenset[str] = frozenset(
    suffix for suffix, fmt in FORMATS.items() if fmt.delimited
)

#: The formats that can be read in chunks rather than whole.
STREAMED: frozenset[str] = frozenset(
    suffix for suffix, fmt in FORMATS.items() if fmt.streamed
)
