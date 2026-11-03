"""A refused PRAGMA says how to ask for the same read as a query (#137, DuckDB 2.0).

The DuckDB reader runs a statement only if DuckDB's own parser calls it a query
(``tests/test_duckdb_reader_gate.py``). Measured with ``extract_statements``:
on DuckDB 1.5.5 the parser rewrote a reading ``PRAGMA`` — ``show_tables``,
``table_info('x')``, ``version``, ``database_size`` — into a ``SELECT``; on
2.0.0 every one of them parses as kind ``PRAGMA``, so the gate refuses it. The
gate stays fail-closed and keeps asking the parser rather than a list of PRAGMA
names, so the refusal has to carry the way out: the query spellings of the same
reads. Every spelling the refusal names is run here through ``query``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import duckdb
import foreign
import pytest
from fastmcp import Client
from sqlalchemy import Integer, Text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module

#: Whether the running DuckDB parses a reading PRAGMA as kind ``PRAGMA``.
#: Measured: 1.5.5 rewrote ``PRAGMA show_tables`` to a ``SELECT``, 2.0.0 does
#: not. Read from the parser itself, not from a version number, so the tests
#: below assert what is true of the DuckDB they run against.
_PARSER_KEEPS_PRAGMA = [
    statement.type.name
    for statement in duckdb.connect().extract_statements("PRAGMA show_tables")
] == ["PRAGMA"]


@pytest.fixture(autouse=True)
def session(tmp_path):
    """A fresh registry and a fresh allowed root for every test."""
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(config_module.Config(roots=(root,)))
    server_module._reset()
    yield root
    server_module._reset()


def call(tool: str, **arguments):
    """Call a tool over the protocol and return its decoded payload."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


def attach(root: Path, *, writable: bool) -> Path:
    """``warehouse.db`` with one table ``sales`` of two rows, attached as ``wh``."""
    target = foreign.build_database(
        root / "warehouse.db",
        "sales",
        [("region", Text), ("amount", Integer)],
        [("north", 100), ("south", 250)],
        dialect="duckdb",
    )
    grant = {"writable": True} if writable else {}
    attached = call("attach", database=str(target), nickname="wh", **grant)
    assert attached["ok"] is True, attached
    return target


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under the allowed root and its bytes, to show nothing moved."""
    return {path.name: path.read_bytes() for path in sorted(root.iterdir())}


# ---------------------------------------------------------------------------
# The query spellings the refusal names, each run through ``query``
# ---------------------------------------------------------------------------

#: Each query spelling of a reading PRAGMA that the refusal names, with the
#: column that identifies its answer and the value expected in it. All parse
#: as a ``SELECT`` on both 1.5.5 and 2.0.0 (measured with
#: ``extract_statements``).
_REPHRASES = [
    ("SHOW TABLES", "name", "sales"),
    ("SHOW ALL TABLES", "name", "sales"),
    ("DESCRIBE sales", "column_name", "region"),
    ("SELECT * FROM pragma_table_info('sales')", "name", "region"),
    ("SELECT * FROM pragma_version()", "library_version", f"v{duckdb.__version__}"),
    ("SELECT * FROM pragma_database_size()", "database_name", "warehouse"),
]


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize(("sql", "column", "value"), _REPHRASES)
def test_each_rephrase_the_refusal_names_answers_and_changes_nothing(
    session, writable, sql, column, value
):
    attach(session, writable=writable)
    before = snapshot(session)

    answer = call("query", nickname="wh", sql=sql)

    assert answer["ok"] is True, answer
    position = answer["columns"].index(column)
    assert answer["rows"][0][position] == value, answer
    assert snapshot(session) == before


def test_the_pragma_refusal_names_every_rephrase(session):
    """``PRAGMA enable_profiling`` parses as a PRAGMA on 1.5.5 and 2.0.0 alike."""
    attach(session, writable=False)

    answer = call("query", nickname="wh", sql="PRAGMA enable_profiling")

    assert answer["ok"] is False, answer
    error = answer["error"]
    assert "This statement asks to run a PRAGMA that is not a query" in error, error
    for spelling in (
        "SHOW TABLES",
        "SHOW ALL TABLES",
        "DESCRIBE <table>",
        "SELECT * FROM pragma_table_info('<table>')",
        "pragma_version()",
        "pragma_database_size()",
    ):
        assert spelling in error, error


#: The reading PRAGMAs whose parse changed between 1.5.5 and 2.0.0, with the
#: first value each answered on 1.5.5.
_READING_PRAGMAS = [
    ("PRAGMA show_tables", "sales"),
    ("PRAGMA table_info('sales')", 0),
]


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize(("sql", "first"), _READING_PRAGMAS)
def test_a_reading_pragma_answers_exactly_where_duckdb_parses_it_as_a_query(
    session, writable, sql, first
):
    """Refused, with the way out, where the parser keeps it a PRAGMA (2.0.0)."""
    attach(session, writable=writable)

    answer = call("query", nickname="wh", sql=sql)

    if _PARSER_KEEPS_PRAGMA:
        assert answer["ok"] is False, answer
        assert answer["error"].startswith("query reads; it does not write."), answer
        assert "pragma_table_info('<table>')" in answer["error"], answer
    else:
        assert answer["ok"] is True, answer
        assert answer["rows"][0][0] == first, answer
