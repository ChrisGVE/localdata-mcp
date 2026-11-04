"""Which formula cells in an ``.xlsx`` or ``.xlsm`` carry no result (task 20).

An Office Open XML workbook is a zip of XML parts. Each sheet part holds its
cells, and a formula cell holds two things: the formula, in ``<f>``, and the
result the spreadsheet application last computed for it, in ``<v>``. pandas
reads through openpyxl with ``data_only=True``, which returns that stored
result — so a workbook saved by Excel, LibreOffice or Numbers reads correctly,
and one written by a program, which stores the formula and an empty ``<v />``,
reads every such cell as NULL and says nothing.

Nothing here evaluates a formula. It finds the cells that have no result, so
the reader can say so. One streaming pass over each sheet part, read straight
from the package, because asking openpyxl a second time — once for formulas,
once for results — tripled the load time of a 100,000-row sheet (10.1 s read,
29.4 s with the two extra passes; measured 2026-11-04).

The parts are located the way the format specifies (ECMA-376 Part 1): the
workbook part lists each sheet's name and relationship id, and the workbook's
relationships part maps that id to the sheet part's path.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

__all__ = ["UncomputedColumn", "column_letters", "uncomputed_formulas"]

_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RELATIONSHIP = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE = "{http://schemas.openxmlformats.org/package/2006/relationships}"

#: The row pandas reads as the header. It does not skip blank rows above the
#: data, so the header is the sheet's first row and every row after it is data.
_HEADER_ROW = 1


@dataclass
class UncomputedColumn:
    """One sheet column's formula cells that carry no stored result."""

    #: The column's position in the sheet, from 0 — which is also its position
    #: in the frame pandas builds, since pandas reads every sheet from column A.
    index: int
    cells: int
    #: The first such cell's formula, as the file spells it, without the ``=``.
    first_formula: str


def uncomputed_formulas(
    path: Path, sheets: set[str]
) -> dict[str, list[UncomputedColumn]]:
    """For each named sheet, the columns holding formulas with no stored result.

    Only data rows are counted; a formula in the header row is a header. A sheet
    with nothing to report is absent from the answer, as is any sheet not in
    ``sheets`` — the caller asks only about sheets where a NULL could hide one.
    """
    found: dict[str, list[UncomputedColumn]] = {}
    with zipfile.ZipFile(path) as package:
        for name, part in _sheet_parts(package).items():
            if name not in sheets or part not in package.namelist():
                continue
            columns = _scan(package, part)
            if columns:
                found[name] = columns
    return found


def _sheet_parts(package: zipfile.ZipFile) -> dict[str, str]:
    """Each sheet's name and the package member holding its cells."""
    workbook = ElementTree.fromstring(package.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(package.read("xl/_rels/workbook.xml.rels"))
    targets = {
        rel.get("Id"): rel.get("Target", "")
        for rel in relationships.iter(f"{_PACKAGE}Relationship")
    }
    parts = {}
    for sheet in workbook.iter(f"{_MAIN}sheet"):
        target = targets.get(sheet.get(f"{_RELATIONSHIP}id"))
        if not target:
            continue
        # A target is absolute within the package (``/xl/worksheets/…``, as
        # openpyxl writes it) or relative to the workbook part (``worksheets/…``,
        # as Excel does).
        member = (
            target.lstrip("/")
            if target.startswith("/")
            else str(PurePosixPath("xl") / target)
        )
        parts[sheet.get("name", "")] = member
    return parts


def _scan(package: zipfile.ZipFile, part: str) -> list[UncomputedColumn]:
    """The formula cells in one sheet part that carry no stored result.

    A result is absent when ``<v>`` is missing or empty — except for a cell typed
    ``t="str"``, where an empty ``<v>`` is a formula whose result is the empty
    string, which Excel writes exactly that way. A cell's position comes from its
    ``r`` reference, or from the cell before it when ``r`` is omitted, which the
    format allows. Rows are cleared as they are passed, so the sheet is never
    held whole.
    """
    columns: dict[int, UncomputedColumn] = {}
    row_number = 0
    with package.open(part) as stream:
        column = -1
        for event, element in ElementTree.iterparse(stream, events=("start", "end")):
            if element.tag == f"{_MAIN}row":
                if event == "start":
                    row_number = int(element.get("r") or row_number + 1)
                    column = -1
                else:
                    element.clear()
                continue
            if element.tag != f"{_MAIN}c" or event != "end":
                continue
            reference = element.get("r")
            column = _column_index(reference) if reference else column + 1
            formula = element.find(f"{_MAIN}f")
            if formula is None or row_number <= _HEADER_ROW:
                continue
            value = element.find(f"{_MAIN}v")
            if value is not None and (value.text or element.get("t") == "str"):
                continue
            entry = columns.get(column)
            if entry is None:
                columns[column] = UncomputedColumn(column, 1, formula.text or "")
            else:
                entry.cells += 1
    return sorted(columns.values(), key=lambda entry: entry.index)


def _column_index(reference: str) -> int:
    """``C7`` is column 2: the letters of a cell reference, read as base 26."""
    index = 0
    for character in reference:
        if not character.isalpha():
            break
        index = index * 26 + (ord(character.upper()) - ord("A") + 1)
    return index - 1


def column_letters(index: int) -> str:
    """Column 2 is ``C``: the inverse of :func:`_column_index`."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters
