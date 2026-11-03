"""A refused PIVOT without an ``IN`` list says how to name its values (#137).

DuckDB's ``PIVOT t ON k USING sum(v)`` does not know its output columns until
it has read the distinct values of ``k``, so the parser expands it. Measured
with ``extract_statements``: on 1.5.5 and 1.5.6 it is two statements,
``CREATE`` and ``SELECT``, the ``CREATE`` with no source text of its own; on
2.0.0 it is one statement of kind ``MULTI``. Either way it is not one query,
and the reader gate refuses it (fail-closed). On 2.0.0 the refusal used to name
it ``MULTI``, which tells an agent nothing. It now says what was asked and how
to write it as one query: name the values with ``IN (…)``, which parses as a
``SELECT`` on both versions, and list them with ``SELECT DISTINCT``.

The same expansion happens when the PIVOT sits inside a subquery or after a
``WITH`` (measured on both versions), so it is recognised from the parse, not
from the first word of the text.
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
from localdata_mcp import dialects
from localdata_mcp import server as server_module

#: The words every PIVOT refusal carries: what was asked, the one-query form,
#: and how to find the values to put in it.
_PIVOT_WORDS = (
    "PIVOT without naming its values in an IN list",
    "PIVOT t ON k IN ('a', 'b') USING sum(v)",
    "SELECT DISTINCT k FROM t",
)


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


def refusal(sql: str) -> str | None:
    """What the gate says of ``sql`` on a bare connection."""
    connection = duckdb.connect()
    try:
        return dialects._duckdb_read_refusal(connection, sql)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "sql",
    [
        "PIVOT t ON k USING sum(v)",
        "pivot t on k using sum(v)",
        "/* c */ PIVOT t ON k USING sum(v)",
        "PIVOT_WIDER t ON k USING sum(v)",
        "SELECT * FROM (PIVOT t ON k USING sum(v))",
        "WITH x AS (SELECT * FROM t) PIVOT x ON k USING sum(v)",
    ],
)
def test_a_pivot_without_an_in_list_is_refused_with_how_to_name_its_values(sql):
    found = refusal(sql)

    assert found is not None, sql
    assert found.startswith(_PIVOT_WORDS[0]), found
    for words in _PIVOT_WORDS[1:]:
        assert words in found, found


def test_a_pivot_with_an_in_list_is_one_query():
    assert refusal("PIVOT t ON k IN ('a', 'b') USING sum(v)") is None


def test_a_column_called_pivot_does_not_make_a_create_a_pivot():
    """The word alone decides nothing: a write next to it is named as itself."""
    found = refusal("SELECT 1 AS pivot; CREATE TABLE x (a INTEGER)")

    assert found is not None and found.startswith("CREATE"), found


@pytest.mark.parametrize("writable", [False, True])
def test_the_pivot_refusal_reaches_the_client_and_changes_nothing(session, writable):
    target = attach(session, writable=writable)
    before = {path.name: path.read_bytes() for path in sorted(session.iterdir())}

    answer = call("query", nickname="wh", sql="PIVOT sales ON region USING sum(amount)")

    assert answer["ok"] is False, answer
    error = answer["error"]
    assert error.startswith("query reads; it does not write."), error
    assert f"This statement asks to {_PIVOT_WORDS[0]}" in error, error
    for words in _PIVOT_WORDS[1:]:
        assert words in error, error
    call("detach", nickname="wh")
    server_module._reset()
    after = {path.name: path.read_bytes() for path in sorted(session.iterdir())}
    assert after == before
    assert target.exists()


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize(
    ("sql", "rows"),
    [
        ("SELECT DISTINCT region FROM sales ORDER BY region", [["north"], ["south"]]),
        (
            "PIVOT sales ON region IN ('north', 'south') USING sum(amount)",
            [[100, 250]],
        ),
    ],
)
def test_the_rephrase_the_refusal_names_answers(session, writable, sql, rows):
    attach(session, writable=writable)

    answer = call("query", nickname="wh", sql=sql)

    assert answer["ok"] is True, answer
    assert answer["rows"] == rows, answer
