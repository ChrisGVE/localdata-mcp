"""How ``query`` words a write it refuses on a writable DuckDB datasource (#116).

On a DuckDB slot attached ``writable=true`` the reader is held to reading by a
``READ ONLY`` transaction (#79), and DuckDB refuses the write itself. Its
words for that are "TransactionContext Error: Cannot write to database … -
transaction is launched in read-only mode", which speaks of a transaction to
an agent that opened none and names no verb to use instead. The statement
that ends the transaction early is refused by this server, in its own words;
these hold the write DuckDB refuses to the same wording.

Driven over the protocol, as ``test_server.py`` drives the rest of the
surface, because the wording is what reaches the client.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import foreign
import pytest
from fastmcp import Client
from sqlalchemy import Integer, Text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config


@pytest.fixture(autouse=True)
def session(tmp_path):
    """A fresh registry and a fresh allowed root for every test."""
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,)))
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


def attach_writable_duckdb(root: Path, reach: str) -> None:
    target = root / "warehouse.db"
    foreign.build_database(
        target,
        "sales",
        [("region", Text), ("amount", Integer)],
        [("north", 100), ("south", 250)],
        dialect="duckdb",
    )
    database = str(target) if reach == "file" else f"duckdb:///{target}"
    attached = call("attach", database=database, nickname="wh", writable=True)
    assert attached["ok"] is True, attached


@pytest.mark.parametrize("reach", ["file", "url"])
@pytest.mark.parametrize(
    ("sql", "kind"),
    [
        ("INSERT INTO sales VALUES ('east', 1) RETURNING region", "INSERT"),
        ("INSERT INTO sales VALUES ('east', 1)", "INSERT"),
        ("UPDATE sales SET amount = 0", "UPDATE"),
        ("DELETE FROM sales", "DELETE"),
        ("CREATE TABLE extra (a INTEGER)", "CREATE"),
        ("DROP TABLE sales", "DROP"),
        ("SELECT 1; INSERT INTO sales VALUES ('east', 1)", "INSERT"),
    ],
)
def test_a_write_duckdb_refuses_is_refused_in_the_servers_words(
    session, reach, sql, kind
):
    attach_writable_duckdb(session, reach)

    refused = call("query", nickname="wh", sql=sql)

    assert refused["ok"] is False, refused
    error = refused["error"]
    assert error.startswith("query reads; it does not write."), error
    assert f"asks to {kind}" in error, error
    for verb in ("create", "update", "drop"):
        assert verb in error, error
    assert "TransactionContext" not in error, error
    assert "read-only mode" not in error, error
    assert call("query", nickname="wh", sql="SELECT count(*) FROM sales")["rows"] == [
        [2]
    ]


def test_the_transaction_control_refusal_keeps_its_wording(session):
    """The refusal #79 already worded, unchanged by the translation beside it."""
    attach_writable_duckdb(session, "file")

    refused = call(
        "query",
        nickname="wh",
        sql="COMMIT; INSERT INTO sales VALUES ('east', 1) RETURNING region",
    )

    assert refused["ok"] is False
    assert refused["error"].startswith(
        "query reads; it does not write. This statement asks to begin, commit "
        "or roll back a transaction of its own"
    )


def test_an_ordinary_error_on_a_writable_duckdb_reader_is_not_called_a_write(
    session,
):
    """Only DuckDB's read-only refusal is translated; a typo still says typo."""
    attach_writable_duckdb(session, "file")

    missing = call("query", nickname="wh", sql="SELECT * FROM nowhere")

    assert missing["ok"] is False
    assert "does not write" not in missing["error"], missing
    assert "nowhere" in missing["error"], missing
