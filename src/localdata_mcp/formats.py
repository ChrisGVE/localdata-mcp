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

from .readers import (
    Reader,
    delimited,
    read_columnar,
    read_fwf,
    read_json,
    read_jsonl,
    read_numbers,
    read_xml,
    read_yaml,
    workbook,
)
from .writers import (
    Writer,
    workbook_writer,
    write_columnar,
    write_csv,
    write_json,
    write_jsonl,
    write_markdown,
    write_tsv,
    write_xml,
    write_yaml,
)

__all__ = ["DELIMITED", "FORMATS", "READERS", "STREAMED", "WRITERS", "Format"]


@dataclass(frozen=True)
class Format:
    """Everything keyed by one file suffix.

    ``Format`` is to a suffix what ``dialects.Backend`` is to a dialect name:
    the place per-format knowledge is *declared* rather than rediscovered by a
    branch at each site that needs it.

    ``reader`` and ``writer`` are ``None`` when the format cannot be read or
    cannot be written; at least one of them is always present, since a format
    this server can do neither way has no reason to be in the table.

    ``delimited`` says a separator means something for this format — that the
    file is character-separated text rather than something carrying its own
    structure. It governs both directions and is the reason it belongs on the
    format rather than on either side: reading at the wrong separator changes
    what the data *is*, so the read side refuses a delimiter it cannot apply,
    while the write side merely ignores one. Two behaviours, one fact.

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

    def __post_init__(self) -> None:
        if self.reader is None and self.writer is None:
            raise ValueError(
                f"{self.suffix} declares neither a reader nor a writer, so "
                f"nothing about it can be acted on. Leave it out of the table."
            )
        if self.delimited and self.reader is None:
            raise ValueError(
                f"{self.suffix} is declared delimited but has no reader. A "
                f"separator is a fact about how a file is read; a write-only "
                f"format cannot be read back at the separator it was written "
                f"with, which is the property `delimited` exists to keep."
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


#: The whole catalogue. A new format is one entry here and nothing else.
FORMATS: dict[str, Format] = dict(
    (
        # Character-separated text. `.txt` is comma-separated by default, the
        # same as `.csv`, so a file this server writes it can read back.
        _format(
            ".csv",
            reader=delimited(","),
            writer=write_csv,
            delimited=True,
            streamed=True,
        ),
        _format(
            ".tsv",
            reader=delimited("\t"),
            writer=write_tsv,
            delimited=True,
            streamed=True,
        ),
        _format(
            ".txt",
            reader=delimited(","),
            writer=write_csv,
            delimited=True,
            streamed=True,
        ),
        # Fixed-width: read in chunks like the separated formats, but there is
        # no writer worth having — the column widths would have to be invented.
        _format(".fwf", reader=read_fwf, streamed=True),
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
