"""A formula a program wrote has no result in the file, and loads as NULL.

An ``.xlsx`` stores each formula beside the result the spreadsheet application
last computed for it, and pandas reads that stored result (openpyxl with
``data_only=True``). A workbook written by a program — openpyxl, this server's
own writer, most export pipelines — stores the formula and no result, so the
column loads entirely NULL and nothing says why (task 20).

Nothing here evaluates a formula. The cells are found and the attach warns,
naming the sheet, the column and the count. The fixtures are real workbooks:
written by openpyxl, which stores no results, and — for the case that must not
warn — the same workbook with the results written in where Excel writes them,
inside each cell's ``<v>``.
"""

from __future__ import annotations

import asyncio
import json
import re
import zipfile
from pathlib import Path

import openpyxl
import pytest
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.loader import read_file
from localdata_mcp.slots import Registry


@pytest.fixture(autouse=True)
def root(tmp_path):
    shared = tmp_path / "root"
    shared.mkdir()
    config_module.use(Config(roots=(shared,)))
    return shared


def program_written(path: Path, rows: int = 3) -> Path:
    """A two-sheet workbook whose ``Sales`` sheet computes a column by formula."""
    workbook = openpyxl.Workbook()
    sales = workbook.active
    sales.title = "Sales"
    sales.append(["qty", "Unit Price", "Line Total", "note"])
    for row in range(2, rows + 2):
        sales.append([row - 1, 2.5, f"=A{row}*B{row}", "x"])
    plain = workbook.create_sheet("Plain")
    plain.append(["a"])
    plain.append([1])
    workbook.save(path)
    return path


def with_cell_xml(path: Path, rewrite) -> Path:
    """The same workbook with the first sheet's XML passed through ``rewrite``."""
    member = "xl/worksheets/sheet1.xml"
    with zipfile.ZipFile(path) as source:
        parts = {name: source.read(name) for name in source.namelist()}
    parts[member] = rewrite(parts[member].decode("utf-8")).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for name, data in parts.items():
            target.writestr(name, data)
    return path


def cache_results(xml: str, rows=None) -> str:
    """Write each formula's result into its ``<v>``, as a spreadsheet application does.

    ``rows`` limits it to those sheet rows, to build a partly computed sheet.
    """

    def computed(match: re.Match) -> str:
        row = int(match.group(1))
        if rows is not None and row not in rows:
            return match.group(0)
        return match.group(0).replace("<v />", f"<v>{(row - 1) * 2.5}</v>")

    return re.sub(r'<c r="C(\d+)"><f>[^<]*</f><v />', computed, xml)


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
    """The condition under test, measured: the column is all NULL, silently, today."""
    source = program_written(root / "book.xlsx")

    sales = read_file(source).tables[0].frame

    assert sales["Line Total"].isna().all()
    assert sales["qty"].tolist() == [1, 2, 3]


def test_the_cached_fixture_really_holds_the_results(root):
    source = with_cell_xml(program_written(root / "book.xlsx"), cache_results)

    sales = read_file(source).tables[0].frame

    assert sales["Line Total"].tolist() == [2.5, 5.0, 7.5]


# ---------------------------------------------------------------------------
# The warning
# ---------------------------------------------------------------------------


def test_uncomputed_formulas_are_named_by_sheet_column_and_count(root):
    source = program_written(root / "book.xlsx")

    (warning,) = formula_warnings(source)

    assert "Sales" in warning
    # The column under the name it landed as, and where it sits in the sheet.
    assert "line_total" in warning and "spreadsheet column C" in warning
    assert "3 cells" in warning
    assert "never computed" in warning and "NULL" in warning
    # The first formula is quoted, so the column can be rebuilt in SQL.
    assert "=A2*B2" in warning
    assert "Plain" not in warning


def test_a_workbook_saved_with_its_results_says_nothing(root):
    source = with_cell_xml(program_written(root / "book.xlsx"), cache_results)

    assert formula_warnings(source) == []


def test_only_the_cells_without_a_result_are_counted(root):
    source = with_cell_xml(
        program_written(root / "book.xlsx", rows=4),
        lambda xml: cache_results(xml, rows={2, 3}),
    )

    (warning,) = formula_warnings(source)

    assert "2 cells" in warning


def test_an_empty_text_result_is_a_result(root):
    """Excel stores a formula answering "" as ``t="str"`` with an empty ``<v>``."""

    def empty_text(xml: str) -> str:
        return re.sub(
            r'<c r="(C\d+)"><f>([^<]*)</f><v />',
            r'<c r="\1" t="str"><f>\2</f><v></v>',
            xml,
        )

    source = with_cell_xml(program_written(root / "book.xlsx"), empty_text)

    assert formula_warnings(source) == []


def test_a_shared_formula_without_a_result_is_counted(root):
    """Excel writes a filled-down formula once and refers to it from the other cells."""

    def shared(xml: str) -> str:
        xml = xml.replace(
            '<c r="C2"><f>A2*B2</f><v />',
            '<c r="C2"><f t="shared" ref="C2:C4" si="0">A2*B2</f><v />',
        )
        return re.sub(
            r'<c r="(C[34])"><f>[^<]*</f><v />',
            r'<c r="\1"><f t="shared" si="0"/>',
            xml,
        )

    source = with_cell_xml(program_written(root / "book.xlsx"), shared)

    (warning,) = formula_warnings(source)

    assert "3 cells" in warning


def test_two_formula_columns_are_both_named(root):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Calc"
    sheet.append(["a", "double", "triple"])
    for row in range(2, 4):
        sheet.append([row, f"=A{row}*2", f"=A{row}*3"])
    source = root / "calc.xlsx"
    workbook.save(source)

    (warning,) = formula_warnings(source)

    assert "double" in warning and "triple" in warning
    assert "4 cells" in warning


def test_a_formula_in_the_header_row_is_not_a_data_cell(root):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["a", "=1+1"])
    sheet.append([1, 2])
    source = root / "header.xlsx"
    workbook.save(source)

    assert formula_warnings(source) == []


def test_an_xlsm_is_read_the_same_way(root):
    source = program_written(root / "book.xlsx")
    macro = root / "book.xlsm"
    macro.write_bytes(source.read_bytes())

    (warning,) = formula_warnings(macro)

    assert "3 cells" in warning


# ---------------------------------------------------------------------------
# Through the verbs
# ---------------------------------------------------------------------------


def test_attach_carries_the_warning_and_sql_rebuilds_the_column(root):
    """The next step the warning offers is measured to work."""
    server_module._reset()
    try:
        source = program_written(root / "book.xlsx")

        attached = call("attach", database=str(source))

        assert attached["ok"] is True
        assert any("never computed" in w for w in attached["warnings"])
        rebuilt = call(
            "query",
            nickname=attached["nickname"],
            sql="SELECT qty * unit_price FROM sales ORDER BY qty",
        )
        assert [row[0] for row in rebuilt["rows"]] == [2.5, 5.0, 7.5]
    finally:
        server_module._reset()


def test_the_registry_lands_the_null_column_and_the_warning_together(root):
    config_module.use(Config(roots=(root,)))
    registry = Registry()
    try:
        source = program_written(root / "book.xlsx")
        registry.attach(str(source), "book")

        info = registry.describe("book", "sales")

        assert any("never computed" in note for note in info.notes)
        _, rows = registry.query("book", "SELECT count(line_total) FROM sales")
        assert rows[0][0] == 0
    finally:
        registry.close()
