"""DuckDB's ``force_checkpoint()`` is refused in this server's words (#138).

The DuckDB reader runs one ``CALL`` kind of statement, so that ``CHECKPOINT``
keeps working (#137). ``force_checkpoint()`` is the one function measured that
DuckDB itself refuses inside the reader's ``BEGIN TRANSACTION READ ONLY``, and
it does so in its own words: "TransactionContext Error: Cannot FORCE
CHECKPOINT: the current transaction has been started for this database",
which talks about a transaction to an agent that never opened one. Nothing is
written either way, so this is about the words only.

The gate now finds the call with DuckDB's own tokenizer, an identifier
``force_checkpoint`` followed by ``(``, so a literal or a column holding the
word is still a read, and refuses it before DuckDB is asked, naming what runs
instead. Measured on DuckDB 2.0.0: ``CHECKPOINT`` and ``FORCE CHECKPOINT``
run in that transaction; every spelling of the call below fails there.
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
from localdata_mcp.dialects import _duckdb_read_refusal


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


def attach(root: Path, *, writable: bool) -> None:
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


_CALLS = [
    "CALL force_checkpoint()",
    "call FORCE_CHECKPOINT ( )",
    'CALL "force_checkpoint"()',
    "CALL main.force_checkpoint()",
    "CALL force_checkpoint/**/()",
    "SELECT * FROM force_checkpoint()",
]


@pytest.mark.parametrize("sql", _CALLS)
def test_duckdb_itself_refuses_each_spelling_in_a_read_only_transaction(tmp_path, sql):
    """What the gate stands in front of: DuckDB's own refusal, in its words."""
    connection = duckdb.connect(str(tmp_path / "x.duckdb"))
    connection.execute("BEGIN TRANSACTION READ ONLY")
    with pytest.raises(duckdb.TransactionException, match="Cannot FORCE CHECKPOINT"):
        connection.execute(sql)
    connection.close()


@pytest.mark.parametrize("sql", _CALLS)
def test_the_gate_names_force_checkpoint_before_duckdb_is_asked(tmp_path, sql):
    connection = duckdb.connect(str(tmp_path / "x.duckdb"))
    refused = _duckdb_read_refusal(connection, sql)
    connection.close()

    assert refused is not None, sql
    assert refused.startswith("call force_checkpoint()"), refused
    assert "CHECKPOINT" in refused and "FORCE CHECKPOINT" in refused, refused


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 'force_checkpoint()'",
        "SELECT 1 AS force_checkpoint",
        "CHECKPOINT",
        "FORCE CHECKPOINT",
        "CALL checkpoint()",
        "CALL pragma_version()",
    ],
)
def test_a_read_or_a_checkpoint_is_not_refused_for_the_word(tmp_path, sql):
    connection = duckdb.connect(str(tmp_path / "x.duckdb"))
    refused = _duckdb_read_refusal(connection, sql)
    connection.close()

    assert refused is None, (sql, refused)


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize("sql", _CALLS)
def test_query_answers_force_checkpoint_in_this_servers_words(session, writable, sql):
    attach(session, writable=writable)

    answer = call("query", nickname="wh", sql=sql)

    assert answer["ok"] is False, answer
    error = answer["error"]
    assert "force_checkpoint()" in error, error
    assert "TransactionContext" not in error, error
    assert "current transaction has been started" not in error, error
    assert "CHECKPOINT" in error, error


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize("sql", ["CHECKPOINT", "FORCE CHECKPOINT"])
def test_the_checkpoints_the_refusal_offers_run_through_query(session, writable, sql):
    """Measured, as the refusal offers them: both answer, with no rows."""
    attach(session, writable=writable)

    answer = call("query", nickname="wh", sql=sql)

    assert answer["ok"] is True, answer
    assert call("query", nickname="wh", sql="SELECT count(*) FROM sales")["rows"] == [
        [2]
    ]
