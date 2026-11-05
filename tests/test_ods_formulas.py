"""A formula a program wrote into an ``.ods`` has no result, and loads as NULL.

An OpenDocument spreadsheet stores each formula in ``table:formula`` and the
result the spreadsheet application last computed beside it, in ``office:value``
(or ``office:date-value``, ``office:time-value``, ``office:boolean-value``, or a
string under ``office:value-type="string"``). pandas reads that stored result.
A sheet written by a program — odfpy here — can carry the formula alone, and
then every such cell reads as NULL. ``attach`` warns about it the way it does
for an ``.xlsx`` (``docs/CONSTRAINTS.md`` §38), and nothing is evaluated.

The fixtures are real OpenDocument files written by odfpy. Repeated rows and
columns are written with odfpy's own ``number-rows-repeated`` and
``number-columns-repeated``, which is how a spreadsheet application compresses
a sheet, so the positions the warning names are the ones pandas reads.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client
from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import CoveredTableCell, Table, TableCell, TableRow
from odf.text import P

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.loader import read_file


@pytest.fixture(autouse=True)
def root(tmp_path):
    shared = tmp_path / "root"
    shared.mkdir()
    config_module.use(Config(roots=(shared,)))
    return shared


def text(value: str, **attributes) -> TableCell:
    cell = TableCell(valuetype="string", **attributes)
    cell.addElement(P(text=value))
    return cell


def number(value: float, **attributes) -> TableCell:
    return TableCell(valuetype="float", value=value, **attributes)


def uncomputed(formula: str, **attributes) -> TableCell:
    """A formula cell as odfpy writes it when given only the formula."""
    return TableCell(formula=formula, **attributes)


def computed(formula: str, value: float, **attributes) -> TableCell:
    """The same cell with its result stored, as a spreadsheet application saves it."""
    cell = TableCell(formula=formula, valuetype="float", value=value, **attributes)
    cell.addElement(P(text=str(value)))
    return cell


def row(*cells, **attributes) -> TableRow:
    element = TableRow(**attributes)
    for cell in cells:
        element.addElement(cell)
    return element


def save(path: Path, sheets: dict[str, list[TableRow]]) -> Path:
    document = OpenDocumentSpreadsheet()
    for name, rows in sheets.items():
        table = Table(name=name)
        for element in rows:
            table.addElement(element)
        document.spreadsheet.addElement(table)
    document.save(str(path))
    return path


def header() -> TableRow:
    return row(text("qty"), text("Unit Price"), text("Line Total"), text("note"))


def program_written(path: Path, rows: int = 3, results: set[int] = frozenset()) -> Path:
    """A two-sheet workbook whose ``Sales`` sheet computes column C by formula.

    ``results`` names the sheet rows (from 2) whose formula carries its result.
    """
    data = []
    for sheet_row in range(2, rows + 2):
        formula = f"of:=[.A{sheet_row}]*[.B{sheet_row}]"
        line = (
            computed(formula, (sheet_row - 1) * 2.5)
            if sheet_row in results
            else uncomputed(formula)
        )
        data.append(row(number(sheet_row - 1), number(2.5), line, text("x")))
    plain = [row(text("a")), row(number(1))]
    return save(path, {"Sales": [header(), *data], "Plain": plain})


def formula_warnings(path: Path) -> list[str]:
    return [note for note in read_file(path).notes if "formula" in note]


def call(tool: str, **arguments):
    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# The fixture is what it says it is
# ---------------------------------------------------------------------------


def test_the_program_written_fixture_really_loads_the_column_as_null(root):
    """The condition under test, measured: the column is all NULL."""
    sales = read_file(program_written(root / "book.ods")).tables[0].frame

    assert sales["Line Total"].isna().all()
    assert sales["qty"].tolist() == [1, 2, 3]


def test_the_computed_fixture_really_holds_the_results(root):
    source = program_written(root / "book.ods", results={2, 3, 4})

    sales = read_file(source).tables[0].frame

    assert sales["Line Total"].tolist() == [2.5, 5.0, 7.5]


# ---------------------------------------------------------------------------
# The warning
# ---------------------------------------------------------------------------


def test_uncomputed_formulas_are_named_by_sheet_column_and_count(root):
    (warning,) = formula_warnings(program_written(root / "book.ods"))

    assert "Sales" in warning
    assert "line_total" in warning and "spreadsheet column C" in warning
    assert "3 cells" in warning
    assert "never computed" in warning and "NULL" in warning
    # The formula as the file spells it, without its namespace prefix.
    assert "=[.A2]*[.B2]" in warning and "of:" not in warning
    assert "Plain" not in warning


def test_a_sheet_saved_with_its_results_says_nothing(root):
    source = program_written(root / "book.ods", results={2, 3, 4})

    assert formula_warnings(source) == []


def test_only_the_cells_without_a_result_are_counted(root):
    source = program_written(root / "book.ods", rows=4, results={2, 3})

    (warning,) = formula_warnings(source)

    assert "2 cells" in warning


@pytest.mark.parametrize(
    ("result", "shown"),
    [
        ({"valuetype": "date", "datevalue": "2026-11-05"}, "2026-11-05"),
        # pandas reads a time from the text the cell displays.
        ({"valuetype": "time", "timevalue": "PT10H00M00S"}, "10:00:00"),
        ({"valuetype": "boolean", "booleanvalue": "true"}, "TRUE"),
        ({"valuetype": "string"}, "done"),
        # A formula whose result is the empty string, the ``t="str"`` case of
        # an ``.xlsx``: a result, which loads as NULL and is not flagged.
        ({"valuetype": "string"}, ""),
    ],
    ids=["date", "time", "boolean", "string", "empty-string"],
)
def test_every_kind_of_stored_result_is_a_result(root, result, shown):
    formula = TableCell(formula="of:=[.A2]", **result)
    if shown:
        formula.addElement(P(text=shown))
    sheet = [row(text("a"), text("b")), row(number(1), formula), row(text("n"))]
    # The third row puts a NULL in the sheet, so the scan runs.
    source = save(root / "kinds.ods", {"Calc": sheet})

    assert formula_warnings(source) == []


def test_a_formula_in_the_header_row_is_not_a_data_cell(root):
    sheet = [row(text("a"), uncomputed("of:=1+1")), row(number(1)), row(number(2))]
    source = save(root / "header.ods", {"Calc": sheet})

    assert formula_warnings(source) == []


# ---------------------------------------------------------------------------
# Positions: repeated rows and columns are expanded, as pandas expands them
# ---------------------------------------------------------------------------


def test_a_repeated_row_counts_once_per_row(root):
    data = row(number(1), uncomputed("of:=[.A2]*2"), numberrowsrepeated=4)
    source = save(root / "rows.ods", {"Calc": [row(text("a"), text("b")), data]})

    (warning,) = formula_warnings(source)

    assert "4 cells" in warning and "spreadsheet column B" in warning


def test_a_repeated_row_that_starts_at_the_header_counts_only_its_data_rows(root):
    # The text after the formula keeps column B in the frame: pandas drops a
    # trailing column that holds nothing at all.
    repeated = row(text("a"), uncomputed("of:=1"), text("c"), numberrowsrepeated=3)
    source = save(root / "top.ods", {"Calc": [repeated, row(number(1))]})

    (warning,) = formula_warnings(source)

    assert "2 cells" in warning


def test_a_repeated_cell_is_every_column_it_covers(root):
    sheet = [
        row(text("a"), text("b"), text("c")),
        row(number(1), uncomputed("of:=1", numbercolumnsrepeated=2)),
    ]
    source = save(root / "columns.ods", {"Calc": sheet})

    (warning,) = formula_warnings(source)

    assert "spreadsheet column B" in warning and "spreadsheet column C" in warning
    assert "2 cells in all" in warning


def test_blank_and_covered_cells_before_a_formula_keep_its_letter(root):
    """An empty run and a merged-over cell are positions, as pandas reads them."""
    sheet = [
        row(text("a"), text("b"), text("c"), text("d"), text("e")),
        row(
            TableCell(numbercolumnsrepeated=2),
            CoveredTableCell(),
            number(1),
            uncomputed("of:=[.D2]"),
        ),
    ]
    source = save(root / "gaps.ods", {"Calc": sheet})

    (warning,) = formula_warnings(source)

    assert "e (spreadsheet column E)" in warning


# ---------------------------------------------------------------------------
# Through the verbs
# ---------------------------------------------------------------------------


def test_attach_carries_the_warning_and_sql_rebuilds_the_column(root):
    """The next step the warning offers is measured to work."""
    server_module._reset()
    try:
        attached = call("attach", database=str(program_written(root / "book.ods")))

        assert attached["ok"] is True
        assert any("never computed" in w for w in attached["warnings"])
        rebuilt = call(
            "query",
            nickname=attached["nickname"],
            sql="SELECT qty * unit_price FROM sales ORDER BY qty",
        )
        assert [r[0] for r in rebuilt["rows"]] == [2.5, 5.0, 7.5]
    finally:
        server_module._reset()
