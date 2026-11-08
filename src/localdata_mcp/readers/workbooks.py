"""Spreadsheets: Excel and OpenDocument workbooks, and Apple Numbers documents.

Every sheet of a workbook is a table, and every table on a Numbers sheet is
one. A workbook written by a program can hold formulas with no stored result,
which load as NULL, so this module also says which columns those are. Numbers
is read by a library of its own, ``numbers_parser``, but raises the same
concerns: several named tables per file, and a grid around the data to trim.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Callable, Sequence
from xml.etree import ElementTree

import pandas as pd

from ..errors import LoadError
from ..names import column_names, sanitize
from .. import ooxml, opendocument
from ..ooxml import UncomputedColumn, column_letters
from .result import NamedFrame, ReadResult, Reader, inferred_types, require_library


def workbook(module: str, extra: str) -> Reader:
    """A reader for one workbook format, bound to the library that reads it.

    A factory rather than a lookup, for the same reason ``delimited`` is one:
    which library handles a format is a fact about the format, so it is declared
    once in ``formats`` and reaches both this reader and the matching writer
    from there. Held here as a table instead it was written down a second time
    on the write side, where the two could disagree without anything noticing
    (#99).

    ``module`` is what pandas dispatches on and what is imported to check the
    library is here; ``extra`` is this project's optional-dependency group, and
    the two differ, so both have to be carried.
    """

    def read_workbook(path: Path) -> ReadResult:
        """Every sheet of a workbook, each as a table under its own sheet name.

        The names are the file's, raw. Two that clean up to the same table name
        are told apart downstream, where every multi-table file is named.

        ``sheet_name=None`` rather than the default ``0``: the default reads the
        first sheet and says nothing about the others, which leaves data that is
        present in the file unreachable through the server. A workbook is a database
        and its sheets are its tables, so all of them land.
        """
        suffix = path.suffix.lower()
        require_library(module, extra, f"Reading {suffix}")

        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except LoadError:
            raise
        except Exception as exc:
            raise LoadError(f"Could not read {path.name}: {exc}") from exc

        # A workbook may carry a sheet that is entirely empty; it is a sheet with no
        # table in it, and dropping it is not loss. Refusing the whole file over one
        # would be, so only a workbook with nothing in any sheet is refused.
        tables = tuple(
            NamedFrame(frame, name)
            for name, frame in sheets.items()
            if not frame.empty or len(frame.columns)
        )
        if not tables:
            raise LoadError(f"Could not read {path.name}: every sheet in it is empty.")
        # The formats whose package says which formula cells have no stored
        # result. An `.xls` is not among them: xlrd hands back a formula's
        # result and not the formula, and the empty result a program like xlwt
        # writes is the same as a formula that really computed "", so there is
        # nothing to tell them apart by (CONSTRAINTS.md §38).
        scan = _FORMULA_SCANS.get(module)
        notes = _uncomputed_formula_notes(path, tables, scan) if scan else ()
        return ReadResult(tables, notes)

    return read_workbook


#: Which formula cells carry no stored result, per reading library: the
#: scan answers, for each named sheet, the columns where they sit.
FormulaScan = Callable[[Path, set[str]], dict[str, list[UncomputedColumn]]]

_FORMULA_SCANS: dict[str, FormulaScan] = {
    "openpyxl": ooxml.uncomputed_formulas,
    "odf": opendocument.uncomputed_formulas,
}


def _uncomputed_formula_notes(
    path: Path, tables: tuple[NamedFrame, ...], scan: FormulaScan
) -> tuple[str, ...]:
    """Say which formula columns loaded as NULL because the file has no result for them.

    Only sheets holding a NULL are scanned: a cell with no stored result is
    read as NULL, so a sheet without one cannot hold such a cell, and the scan
    is skipped where it can find nothing.
    """
    frames = {str(table.name): table.frame for table in tables}
    suspect = {name for name, frame in frames.items() if frame.isna().to_numpy().any()}
    if not suspect:
        return ()
    try:
        found = scan(path, suspect)
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError):
        # pandas has just read this package, so it is sound; a part this scan
        # cannot find is a layout it does not know, and the load is not the
        # place to fail over a warning.
        return ()
    return tuple(
        _uncomputed_formula_note(sheet, frames[sheet], columns)
        for sheet, columns in found.items()
    )


def _uncomputed_formula_note(
    sheet: str, frame: pd.DataFrame, columns: list[UncomputedColumn]
) -> str:
    """One sheet's uncomputed formula columns, as the sentence the caller reads."""
    landed = column_names(list(frame.columns))
    named = []
    for column in columns:
        letter = column_letters(column.index)
        where = f"spreadsheet column {letter}"
        name = (
            f"{landed[column.index]} ({where})" if column.index < len(landed) else where
        )
        named.append(
            f"{name}, {column.cells} {'cell' if column.cells == 1 else 'cells'}"
        )
    total = sum(column.cells for column in columns)
    counted = f", {total} cells in all" if len(columns) > 1 else ""
    first = columns[0].first_formula
    example = f" The first is ={first}." if first else ""
    return (
        f"Sheet {sheet!r} holds formulas whose results were never computed by a "
        f"spreadsheet application — {'; '.join(named)}{counted} — "
        f"so those cells loaded as NULL. A program wrote this workbook: it stored "
        f"each formula and no result, and this server reads stored results and "
        f"does not evaluate formulas.{example} Rebuild the column in SQL from the "
        f"columns it is computed from, where the letters are the sheet's columns "
        f"in order; or open the file in a spreadsheet application, save it, and "
        f"attach it again."
    )


def read_numbers(path: Path) -> ReadResult:
    """Apple Numbers, whose sheets each hold their own named tables.

    Two levels rather than one: a Numbers sheet is a canvas that may carry
    several tables, so the name here is the table's, qualified by its sheet only
    when another table's name would make the same table name in SQL. That is
    asked of the cleaned-up name, not the raw one: ``Q2 Prices`` on one sheet
    and ``Q2-Prices`` on another are different names and the same table, and
    are qualified as such. Two that still clean up alike after qualifying — both
    on one sheet — are told apart downstream, by the rule every multi-table file
    goes through.
    """
    parser = require_library("numbers_parser", "numbers", "Reading Apple Numbers")
    try:
        document = parser.Document(str(path))
        found = [
            (sheet.name, table.name, table.rows(values_only=True))
            for sheet in document.sheets
            for table in sheet.tables
        ]
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    counts: dict[str, int] = {}
    for _, table_name, _ in found:
        cleaned = sanitize(table_name, "table")
        counts[cleaned] = counts.get(cleaned, 0) + 1

    tables = []
    for sheet_name, table_name, rows in found:
        trimmed = _without_grid_padding(rows)
        if trimmed is None:
            continue
        header, body = trimmed
        frame = pd.DataFrame(body, columns=header)
        alone = counts[sanitize(table_name, "table")] == 1
        name = table_name if alone else f"{sheet_name}_{table_name}"
        tables.append(NamedFrame(inferred_types(frame), name))

    if not tables:
        raise LoadError(f"Could not read {path.name}: it holds no table with rows.")
    return ReadResult(tuple(tables))


def _without_grid_padding(rows: Sequence[Sequence[Any]]):
    """Strip a Numbers table's empty grid, and nothing that holds a value.

    A Numbers table is a fixed canvas — a new one is 8 columns by 12 rows —
    so the cells beyond the data come back as ``None`` and would otherwise
    become columns named ``None`` and a tail of all-null rows.

    Emptiness is tested on the *whole* column, not on its header: a column with
    values but no header is real data that happens to be unlabelled, and
    dropping it on the strength of a blank header would be exactly the silent
    loss this reader is meant to avoid. Returns ``None`` for a table that is
    entirely empty.
    """
    if not rows:
        return None

    header, *body = rows
    keep = [
        index
        for index in range(len(header))
        if header[index] is not None
        or any(index < len(row) and row[index] is not None for row in body)
    ]
    if not keep:
        return None

    kept_body = [[row[index] for index in keep] for row in body]
    kept_body = [row for row in kept_body if any(cell is not None for cell in row)]
    if not kept_body:
        return None

    return [header[index] for index in keep], kept_body
