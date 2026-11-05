"""The standard separator is the default, and a declared one always wins (#153).

Project first principle 7, in Chris's words (2026-11-05): *"We need to comply
with the standard, tsv uses tabs, csv uses comma. If we have to guess that's
what we go for. If we are told, this is a different story."*

So ``.csv`` reads and writes at a comma (RFC 4180) and ``.tsv`` at a tab (the
IANA ``text/tab-separated-values`` registration) when nothing is declared. A
``delimiter`` the caller passes wins on every suffix. ``.txt`` has no published
standard, so nothing is assumed for it and the call still asks for one.

The rule is stated once, on the format table, and every path that reads or
writes delimited text resolves through ``formats.resolve_separator``: ``attach``,
``create(source=…)`` read whole or in chunks, and ``query(path=…)``. These tests
drive each path, at the loader and over the wire, and pin the round trip: a file
this server writes by default is one it reads back by default.

A ``;`` file named ``.csv`` read at the default comma is still caught by the
single-column warning, which names the ``;`` it found in the header. That
warning is the net for the default, so it is pinned here too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_server import call

from localdata_mcp import config as config_module
from localdata_mcp import export as export_module
from localdata_mcp import formats as formats_module
from localdata_mcp import loader as loader_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.errors import ExportError, LoadError
from localdata_mcp.formats import DELIMITED, FORMATS, Format


@pytest.fixture()
def root(tmp_path):
    """An allowed root and a fresh registry for every test."""
    allowed = (tmp_path / "root").resolve()
    allowed.mkdir()
    config_module.use(Config(roots=(allowed,)))
    server_module._reset()
    yield allowed
    server_module._reset()


def _write(path: Path, separator: str) -> Path:
    path.write_text(f"name{separator}salary\nAda{separator}120000\nBo{separator}90\n")
    return path


# ---------------------------------------------------------------------------
# The table and the resolver
# ---------------------------------------------------------------------------


def test_each_delimited_suffix_carries_its_published_standard():
    """Comma for .csv (RFC 4180), tab for .tsv (IANA), nothing for .txt."""
    assert FORMATS[".csv"].standard_delimiter == ","
    assert FORMATS[".tsv"].standard_delimiter == "\t"
    assert FORMATS[".txt"].standard_delimiter is None


def test_no_format_without_a_separator_claims_a_standard_one():
    """Derived from the table, so a format added later is covered too."""
    for suffix, fmt in FORMATS.items():
        if suffix not in DELIMITED:
            assert fmt.standard_delimiter is None, suffix


def test_a_standard_separator_needs_a_delimited_format():
    """A standard separator on a format with no separator is a contradiction."""
    with pytest.raises(ValueError, match="delimited"):
        Format(
            suffix=".odd",
            reader=lambda path: None,
            writer=lambda columns, rows, path: 0,
            standard_delimiter=",",
        )


def test_a_standard_separator_is_one_character():
    with pytest.raises(ValueError, match="one character"):
        Format(
            suffix=".odd",
            reader=lambda path: None,
            writer=lambda columns, rows, path: 0,
            delimited=True,
            standard_delimiter="||",
        )


@pytest.mark.parametrize("reading", [True, False])
def test_the_resolver_takes_the_standard_when_nothing_is_declared(reading):
    resolve = formats_module.resolve_separator
    assert resolve("sales.csv", None, reading=reading) == ","
    assert resolve("sales.tsv", None, reading=reading) == "\t"
    assert resolve("SALES.CSV", None, reading=reading) == ","


@pytest.mark.parametrize("reading", [True, False])
def test_the_resolver_lets_a_declared_separator_win_on_every_suffix(reading):
    resolve = formats_module.resolve_separator
    for suffix in sorted(DELIMITED):
        assert resolve(f"data{suffix}", ";", reading=reading) == ";", suffix


def test_the_resolver_refuses_txt_on_the_read_side_as_a_load_error():
    with pytest.raises(LoadError, match="delimiter"):
        formats_module.resolve_separator("notes.txt", None, reading=True)


def test_the_resolver_refuses_txt_on_the_write_side_as_an_export_error():
    with pytest.raises(ExportError, match="delimiter"):
        formats_module.resolve_separator("notes.txt", None, reading=False)


def test_the_refusal_says_why_txt_differs_and_what_to_do():
    """The refusal is read by an agent: it names the gap and the next step."""
    with pytest.raises(LoadError) as read:
        formats_module.resolve_separator("notes.txt", None, reading=True)
    message = str(read.value)
    assert "notes.txt" in message
    assert "no published standard" in message
    assert "first line" in message
    assert "warnings" in message
    # Nothing in it may claim that .csv or .tsv lack a default any more.
    assert "as likely" not in message
    assert "does not decide" not in message

    with pytest.raises(ExportError) as written:
        formats_module.resolve_separator("notes.txt", None, reading=False)
    message = str(written.value)
    assert "no published standard" in message
    assert ".csv" in message and ".tsv" in message


# ---------------------------------------------------------------------------
# The loader: read whole and read in chunks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".tsv", "\t")])
def test_both_read_paths_use_the_standard_when_nothing_is_declared(
    tmp_path, suffix, separator
):
    path = _write(tmp_path / f"data{suffix}", separator)

    whole = loader_module.read_file(path)
    assert list(whole.tables[0].frame.columns) == ["name", "salary"]
    assert whole.notes == ()

    chunked = loader_module.read_source(path)
    assert [c.name for c in chunked.tables[0].columns] == ["name", "salary"]
    assert chunked.notes == ()


@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_both_read_paths_let_a_declared_separator_win(tmp_path, suffix):
    path = _write(tmp_path / f"data{suffix}", ";")

    whole = loader_module.read_file(path, delimiter=";")
    assert list(whole.tables[0].frame.columns) == ["name", "salary"]

    chunked = loader_module.read_source(path, delimiter=";")
    assert [c.name for c in chunked.tables[0].columns] == ["name", "salary"]


def test_both_read_paths_still_refuse_txt_without_a_delimiter(tmp_path):
    path = _write(tmp_path / "data.txt", ",")

    with pytest.raises(LoadError, match="no published standard"):
        loader_module.read_file(path)
    with pytest.raises(LoadError, match="no published standard"):
        loader_module.read_source(path)


def test_the_table_readers_and_writers_use_the_standard_too(tmp_path):
    """`READERS` and `WRITERS` are views of the same table, so they agree."""
    path = _write(tmp_path / "data.tsv", "\t")
    read = formats_module.READERS[".tsv"](path)
    assert list(read.tables[0].frame.columns) == ["name", "salary"]

    target = tmp_path / "out.tsv"
    target.touch()
    formats_module.WRITERS[".tsv"](["a", "b"], [(1, 2)], target)
    assert target.read_text().splitlines() == ["a\tb", "1\t2"]

    with pytest.raises(LoadError, match="no published standard"):
        formats_module.READERS[".txt"](_write(tmp_path / "data.txt", ","))


# ---------------------------------------------------------------------------
# Over the wire: attach, create and query
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".tsv", "\t")])
def test_attach_reads_at_the_standard_when_nothing_is_declared(root, suffix, separator):
    attached = call("attach", database=str(_write(root / f"staff{suffix}", separator)))

    assert attached["ok"] is True, attached
    assert [c["name"] for c in attached["loaded"][0]["columns"]] == [
        "name",
        "salary",
    ]
    assert "warnings" not in attached


@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_attach_uses_a_declared_separator_over_the_standard(root, suffix):
    attached = call(
        "attach", database=str(_write(root / f"staff{suffix}", ";")), delimiter=";"
    )

    assert attached["ok"] is True, attached
    assert [c["name"] for c in attached["loaded"][0]["columns"]] == [
        "name",
        "salary",
    ]


def test_attach_still_refuses_txt_without_a_delimiter(root):
    refused = call("attach", database=str(_write(root / "staff.txt", ",")))

    assert refused["ok"] is False
    assert "no published standard" in refused["error"]
    assert "delimiter" in refused["error"]


def test_a_semicolon_csv_read_at_the_default_warns_and_names_the_semicolon(root):
    """The single-column warning is the net under the default."""
    attached = call("attach", database=str(_write(root / "euro.csv", ";")))

    assert attached["ok"] is True
    assert len(attached["loaded"][0]["columns"]) == 1
    (warning,) = attached["warnings"]
    assert "';'" in warning
    assert "delimiter" in warning


def test_the_warning_s_next_step_works(root):
    """Detach and attach again at the separator the warning named: real columns."""
    path = _write(root / "euro.csv", ";")
    first = call("attach", database=str(path), nickname="euro")
    assert len(first["loaded"][0]["columns"]) == 1
    call("detach", nickname="euro")

    again = call("attach", database=str(path), nickname="euro", delimiter=";")
    assert [c["name"] for c in again["loaded"][0]["columns"]] == ["name", "salary"]
    assert "warnings" not in again


@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".tsv", "\t")])
def test_create_from_a_source_reads_at_the_standard(root, suffix, separator):
    call("attach", database=str(_write(root / "base.csv", ",")), nickname="base")

    added = call(
        "create",
        nickname="base",
        type="table",
        source=str(_write(root / f"extra{suffix}", separator)),
        table="extra",
    )

    assert added["ok"] is True, added
    assert [c["name"] for c in added["columns"]] == ["name", "salary"]


def test_create_from_a_source_lets_a_declared_separator_win(root):
    call("attach", database=str(_write(root / "base.csv", ",")), nickname="base")

    added = call(
        "create",
        nickname="base",
        type="table",
        source=str(_write(root / "extra.tsv", ";")),
        table="extra",
        delimiter=";",
    )

    assert added["ok"] is True, added
    assert [c["name"] for c in added["columns"]] == ["name", "salary"]


def test_create_from_a_txt_source_is_still_refused_without_a_delimiter(root):
    call("attach", database=str(_write(root / "base.csv", ",")), nickname="base")

    refused = call(
        "create",
        nickname="base",
        type="table",
        source=str(_write(root / "extra.txt", ",")),
        table="extra",
    )

    assert refused["ok"] is False
    assert "no published standard" in refused["error"]


@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".tsv", "\t")])
def test_query_writes_at_the_standard_and_it_reads_back_by_default(
    root, suffix, separator
):
    """The round trip, with nothing declared in either direction."""
    call("attach", database=str(_write(root / "staff.csv", ",")), nickname="staff")
    target = root / f"out{suffix}"

    written = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM staff ORDER BY name",
        path=str(target),
    )

    assert written["ok"] is True, written
    assert target.read_text().splitlines() == [
        f"name{separator}salary",
        f"Ada{separator}120000",
        f"Bo{separator}90",
    ]

    back = call("attach", database=str(target), nickname="back")
    assert back["ok"] is True, back
    assert [c["name"] for c in back["loaded"][0]["columns"]] == ["name", "salary"]
    assert "warnings" not in back
    rows = call("query", nickname="back", sql="SELECT * FROM out ORDER BY name")
    assert rows["rows"] == [["Ada", 120000], ["Bo", 90]]


def test_query_uses_a_declared_separator_over_the_standard(root):
    call("attach", database=str(_write(root / "staff.csv", ",")), nickname="staff")
    target = root / "out.tsv"

    written = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM staff ORDER BY name",
        path=str(target),
        delimiter="|",
    )

    assert written["ok"] is True, written
    assert target.read_text().splitlines()[0] == "name|salary"


def test_query_to_txt_is_still_refused_without_a_delimiter(root):
    call("attach", database=str(_write(root / "staff.csv", ",")), nickname="staff")
    target = root / "out.txt"

    refused = call(
        "query", nickname="staff", sql="SELECT * FROM staff", path=str(target)
    )

    assert refused["ok"] is False
    assert "no published standard" in refused["error"]
    assert not target.exists()


def test_export_rows_writes_csv_at_a_comma_by_default(root):
    target = root / "plain.csv"
    export_module.export_rows(["a", "b"], [(1, 2)], str(target))
    assert target.read_text().splitlines() == ["a,b", "1,2"]


# ---------------------------------------------------------------------------
# What the agent is told
# ---------------------------------------------------------------------------


def test_the_server_instructions_state_the_standard_defaults():
    instructions = server_module.mcp.instructions
    assert "required on every call" not in instructions
    assert "as likely to be separated" not in instructions
    assert "RFC 4180" in instructions
    assert ".txt" in instructions
