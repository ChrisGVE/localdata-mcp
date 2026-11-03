"""A comment inside an ``EXPLAIN`` prefix is not read as the explained statement (#137).

The DuckDB reader answers an ``EXPLAIN`` only over a query, because ``EXPLAIN
ANALYZE`` runs what it explains. It finds where the explained statement begins
with DuckDB's own tokenizer (``dialects._duckdb_explained``). Measured with
``duckdb.tokenize('EXPLAIN /* b */ ANALYZE SELECT 1')``: on 1.5.5 the comment
produced no token; on 2.0.0 it is a token of its own, ``token_type.comment``, at
offset 8. Read as a word, that comment was ``/``, so ``ANALYZE`` was taken for
the first word of the explained statement, the text did not parse as a query,
and a read was refused. The contract is unchanged: a comment anywhere in the
prefix is skipped, and an ``EXPLAIN`` of anything but a query is still refused.
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


@pytest.mark.parametrize(
    ("sql", "refused"),
    [
        ("EXPLAIN /* b */ ANALYZE SELECT 1", None),
        ("EXPLAIN -- why\nANALYZE SELECT 1", None),
        ("EXPLAIN /* x */ (FORMAT json) SELECT 1", None),
        ("EXPLAIN (FORMAT json /* x */, ANALYZE) SELECT 1", None),
        ("EXPLAIN /* ( */ ANALYZE SELECT 1", None),
        (
            "EXPLAIN /* b */ ANALYZE COPY (SELECT 1) TO 'x.wal'",
            "EXPLAIN a COPY statement",
        ),
        (
            "EXPLAIN /* ) */ (ANALYZE) COPY (SELECT 1) TO 'x.wal'",
            "EXPLAIN a COPY statement",
        ),
    ],
)
def test_a_comment_in_the_explain_prefix_is_not_read_as_the_statement(sql, refused):
    connection = duckdb.connect()
    try:
        found = dialects._duckdb_read_refusal(connection, sql)
    finally:
        connection.close()

    if refused is None:
        assert found is None
    else:
        assert found is not None and found.startswith(refused), found


@pytest.mark.parametrize("writable", [False, True])
def test_a_commented_explain_analyze_answers_through_query(session, writable):
    attach(session, writable=writable)

    answer = call(
        "query",
        nickname="wh",
        sql="/* a */ EXPLAIN /* b */ ANALYZE SELECT * FROM sales",
    )

    assert answer["ok"] is True, answer
    assert answer["rows"][0][0] == "analyzed_plan", answer
