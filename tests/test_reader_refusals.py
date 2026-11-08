"""The readers' refusals and error guards that no corpus file reaches (#158).

Each of these is a branch of a reader that the rest of the suite never ran: a
document with nothing tabular in it, a workbook that will not open, a Numbers
table that is only empty grid. They pin the exact words and which exception
wins, so that moving the code cannot change either unnoticed.

A Numbers file is stood in for by a fake ``numbers_parser``, because the cases
here are a failure inside the library and a table that holds nothing, neither
of which a real file can be made to produce on demand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from localdata_mcp import readers as readers_module
from localdata_mcp.errors import LoadError
from localdata_mcp.readers import workbooks as workbooks_module


def _written(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_xml_root_with_no_children_is_refused(tmp_path):
    path = _written(tmp_path / "x.xml", "<root/>")
    with pytest.raises(LoadError) as caught:
        readers_module.read_xml(path)
    assert str(caught.value) == "Could not read x.xml: <root> holds no elements."


def test_xml_where_nothing_repeats_is_refused_naming_each_tag(tmp_path):
    path = _written(tmp_path / "x.xml", "<root><a>1</a><b>2</b></root>")
    with pytest.raises(LoadError) as caught:
        readers_module.read_xml(path)
    assert str(caught.value) == (
        "Could not read x.xml: <root> holds one each of <a>, <b>, "
        "so nothing in it repeats as rows do."
    )


def test_json_scalar_is_refused_as_holding_no_table(tmp_path):
    path = _written(tmp_path / "x.json", "42")
    with pytest.raises(LoadError) as caught:
        readers_module.read_json(path)
    assert str(caught.value) == (
        "Could not read x.json: it holds no table. This reader takes an array "
        "of objects, or an object with arrays of objects in it — at any depth, "
        "each becoming a table of its own."
    )


def _read_xlsx(path: Path):
    return readers_module.workbook("openpyxl", "excel")(path)


def test_workbook_that_will_not_open_is_refused_with_its_cause(tmp_path):
    path = tmp_path / "x.xlsx"
    path.write_bytes(b"this is not a workbook")
    with pytest.raises(LoadError) as caught:
        _read_xlsx(path)
    assert str(caught.value).startswith("Could not read x.xlsx: ")
    assert caught.value.__cause__ is not None


def test_workbook_load_error_from_the_read_is_not_rewrapped(tmp_path, monkeypatch):
    original = LoadError("refused further down")

    def refuse(*_args, **_kwargs):
        raise original

    monkeypatch.setattr(workbooks_module.pd, "read_excel", refuse)
    with pytest.raises(LoadError) as caught:
        _read_xlsx(tmp_path / "x.xlsx")
    assert caught.value is original


def test_workbook_with_every_sheet_empty_is_refused(tmp_path):
    path = tmp_path / "x.xlsx"
    pd.DataFrame().to_excel(path, index=False)
    with pytest.raises(LoadError) as caught:
        _read_xlsx(path)
    assert str(caught.value) == "Could not read x.xlsx: every sheet in it is empty."


class _FakeTable:
    """A Numbers table: a name, and rows that may fail when they are asked for."""

    def __init__(self, name, rows):
        self.name = name
        self._rows = rows

    def rows(self, values_only=False):
        assert values_only is True
        if isinstance(self._rows, Exception):
            raise self._rows
        return self._rows


class _FakeSheet:
    def __init__(self, name, tables):
        self.name = name
        self.tables = tables


class _FakeParser:
    """Stands in for the ``numbers_parser`` module: only ``Document`` is used."""

    def __init__(self, outcome):
        self._outcome = outcome

    def Document(self, _path):  # noqa: N802 - the library's own spelling
        if isinstance(self._outcome, Exception):
            raise self._outcome

        class _Document:
            sheets = self._outcome

        return _Document


def _read_numbers(tmp_path, monkeypatch, outcome):
    parser = _FakeParser(outcome)
    monkeypatch.setattr(workbooks_module, "require_library", lambda *_args: parser)
    return readers_module.read_numbers(tmp_path / "x.numbers")


def test_numbers_failure_opening_the_document_is_refused(tmp_path, monkeypatch):
    with pytest.raises(LoadError) as caught:
        _read_numbers(tmp_path, monkeypatch, ValueError("damaged package"))
    assert str(caught.value) == "Could not read x.numbers: damaged package"
    assert isinstance(caught.value.__cause__, ValueError)


def test_numbers_failure_reading_rows_is_refused(tmp_path, monkeypatch):
    """The rows are read inside the guard, not after it."""
    sheets = [_FakeSheet("S", [_FakeTable("T", ValueError("bad cell"))])]
    with pytest.raises(LoadError) as caught:
        _read_numbers(tmp_path, monkeypatch, sheets)
    assert str(caught.value) == "Could not read x.numbers: bad cell"
    assert isinstance(caught.value.__cause__, ValueError)


def test_numbers_load_error_is_not_rewrapped(tmp_path, monkeypatch):
    original = LoadError("refused further down")
    with pytest.raises(LoadError) as caught:
        _read_numbers(tmp_path, monkeypatch, original)
    assert caught.value is original


def test_numbers_table_of_empty_grid_is_skipped(tmp_path, monkeypatch):
    sheets = [
        _FakeSheet(
            "S",
            [
                _FakeTable("Blank", [[None, None], [None, None]]),
                _FakeTable("Data", [["a", None], [1, None]]),
            ],
        )
    ]
    result = _read_numbers(tmp_path, monkeypatch, sheets)
    assert [table.name for table in result.tables] == ["Data"]
    assert list(result.tables[0].frame.columns) == ["a"]


def test_numbers_with_no_table_holding_rows_is_refused(tmp_path, monkeypatch):
    sheets = [_FakeSheet("S", [_FakeTable("Blank", [[None], [None]])])]
    with pytest.raises(LoadError) as caught:
        _read_numbers(tmp_path, monkeypatch, sheets)
    assert str(caught.value) == "Could not read x.numbers: it holds no table with rows."


def _array_refusal(name: str, kind: str) -> str:
    return (
        f"Could not read {name}: it is an array of {kind}, and a table needs an "
        "array of objects — each one a row, its keys the columns."
    )


@pytest.mark.parametrize("text", ['[{"a": 1}, null]', '[null, {"a": 1}]', "[null]"])
def test_json_null_in_a_top_level_array_is_refused_like_a_number(tmp_path, text):
    """A null is a non-object, whatever position it holds (#159)."""
    path = _written(tmp_path / "x.json", text)
    with pytest.raises(LoadError) as caught:
        readers_module.read_json(path)
    assert str(caught.value) == _array_refusal("x.json", "null")


@pytest.mark.parametrize("text", ["- a: 1\n- null\n", "- ~\n- a: 1\n", "-\n"])
def test_yaml_null_in_a_top_level_list_is_refused_like_a_number(tmp_path, text):
    """YAML has three spellings of null; all parse to the same refused value."""
    pytest.importorskip("yaml")
    path = _written(tmp_path / "x.yaml", text)
    with pytest.raises(LoadError) as caught:
        readers_module.read_yaml(path)
    assert str(caught.value) == _array_refusal("x.yaml", "null")


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ('[1, {"a": 1}]', "a number"),
        ("[null, 1]", "null"),
        ("[1, null]", "a number"),
        ('[{"a": 1}, "s", null]', "a string"),
    ],
)
def test_json_array_refusal_names_the_first_non_object(tmp_path, text, kind):
    path = _written(tmp_path / "x.json", text)
    with pytest.raises(LoadError) as caught:
        readers_module.read_json(path)
    assert str(caught.value) == _array_refusal("x.json", kind)


def test_json_empty_top_level_array_is_one_empty_table(tmp_path):
    result = readers_module.read_json(_written(tmp_path / "x.json", "[]"))
    assert [table.name for table in result.tables] == [None]
    assert result.tables[0].frame.shape == (0, 0)
    assert result.notes == ()


def test_json_null_in_a_nested_array_is_still_no_table(tmp_path):
    path = _written(tmp_path / "x.json", '{"t": [{"a": 1}, null]}')
    with pytest.raises(LoadError) as caught:
        readers_module.read_json(path)
    assert str(caught.value).startswith("Could not read x.json: it holds no table.")


def test_json_array_of_objects_still_loads(tmp_path):
    path = _written(tmp_path / "x.json", '[{"a": 1}, {"a": null}]')
    result = readers_module.read_json(path)
    frame = result.tables[0].frame
    assert frame.shape == (2, 1)
    assert list(frame.columns) == ["a"]
    assert frame["a"].isna().tolist() == [False, True]
