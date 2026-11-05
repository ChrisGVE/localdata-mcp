"""Which formula cells in an ``.ods`` carry no result (task 20 follow-up).

An OpenDocument spreadsheet is a zip whose ``content.xml`` holds every sheet as
a ``table:table``. A formula cell holds two things: the formula, in its
``table:formula`` attribute, and the result the spreadsheet application last
computed for it — ``office:value`` for a number, percentage or currency,
``office:date-value``, ``office:time-value`` or ``office:boolean-value`` for the
others, and the cell's text under ``office:value-type="string"``. pandas reads
that stored result, so a sheet written by a program that stores the formula
alone reads every such cell as NULL and says nothing (``docs/CONSTRAINTS.md``
§38).

Nothing here evaluates a formula. It finds the cells that have no result, so
the reader can say so, with the rule the ``.xlsx`` scan uses: a string result
counts even when it is empty, and the header row is not data.

A sheet is compressed the way the format allows (ODF 1.3 Part 3, 9.1): a row
carries ``table:number-rows-repeated`` and a cell
``table:number-columns-repeated``, and both are expanded to positions here,
exactly as pandas expands them, so a column's position is its index in the
frame pandas builds. A merged-over cell (``table:covered-table-cell``) is a
position too. ``content.xml`` is read in one streaming pass, clearing each row
as it is passed, so the document is never held whole.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from .ooxml import UncomputedColumn

__all__ = ["uncomputed_formulas"]

_TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
_OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"

_SHEET = f"{_TABLE}table"
_ROW = f"{_TABLE}table-row"
_CELLS = {f"{_TABLE}table-cell", f"{_TABLE}covered-table-cell"}

#: The attributes that hold a stored result of a type other than text.
_RESULTS = tuple(
    f"{_OFFICE}{name}"
    for name in ("value", "date-value", "time-value", "boolean-value")
)

#: A formula is spelled with a namespace prefix naming its syntax — ``of:`` for
#: OpenFormula, ``oooc:`` for older OpenOffice files — before its ``=``.
_PREFIX = re.compile(r"^[A-Za-z][\w.-]*:(?==)")

#: The row pandas reads as the header: the sheet's first, counted from 0.
_HEADER_ROW = 0


def uncomputed_formulas(
    path: Path, sheets: set[str]
) -> dict[str, list[UncomputedColumn]]:
    """For each named sheet, the columns holding formulas with no stored result.

    Only data rows are counted; a formula in the header row is a header. A sheet
    with nothing to report is absent from the answer, as is any sheet not in
    ``sheets`` — the caller asks only about sheets where a NULL could hide one.
    """
    found: dict[str, list[UncomputedColumn]] = {}
    with zipfile.ZipFile(path) as package, package.open("content.xml") as stream:
        columns: dict[int, UncomputedColumn] = {}
        sheet: str | None = None
        # A table may sit inside a cell (a sub-table); only top-level tables
        # are sheets, so depth is tracked and the inner ones are not read.
        depth = 0
        row = 0
        row_repeat = 1
        column = 0
        for event, element in ElementTree.iterparse(stream, events=("start", "end")):
            tag = element.tag
            if tag == _SHEET:
                if event == "start":
                    depth += 1
                    if depth == 1:
                        name = element.get(f"{_TABLE}name", "")
                        sheet = name if name in sheets else None
                        columns, row = {}, 0
                    continue
                depth -= 1
                if depth == 0:
                    if sheet is not None and columns:
                        found[sheet] = sorted(columns.values(), key=lambda c: c.index)
                    element.clear()
                continue
            if depth != 1 or sheet is None:
                continue
            if tag == _ROW:
                if event == "start":
                    row_repeat = _repeat(element, "number-rows-repeated")
                    column = 0
                else:
                    row += row_repeat
                    element.clear()
                continue
            if tag not in _CELLS or event != "end":
                continue
            column_repeat = _repeat(element, "number-columns-repeated")
            formula = element.get(f"{_TABLE}formula")
            if formula is not None and not _has_result(element):
                # Positions row .. row + row_repeat - 1, less the header.
                cells = row_repeat - (1 if row <= _HEADER_ROW else 0)
                if cells > 0:
                    for index in range(column, column + column_repeat):
                        _count(columns, index, cells, formula)
            column += column_repeat
    return found


def _has_result(cell: ElementTree.Element) -> bool:
    """A stored result: a typed value, or text, which may be the empty string."""
    if cell.get(f"{_OFFICE}value-type") == "string":
        return True
    return any(cell.get(attribute) is not None for attribute in _RESULTS)


def _repeat(element: ElementTree.Element, attribute: str) -> int:
    return int(element.get(f"{_TABLE}{attribute}") or 1)


def _count(
    columns: dict[int, UncomputedColumn], index: int, cells: int, formula: str
) -> None:
    entry = columns.get(index)
    if entry is None:
        spelled = _PREFIX.sub("", formula).removeprefix("=")
        columns[index] = UncomputedColumn(index, cells, spelled)
    else:
        entry.cells += cells
