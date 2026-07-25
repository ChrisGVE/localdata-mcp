"""End-to-end tests over the MCP protocol itself.

These drive the server the way a client does — listing tools, calling them by
name with JSON arguments, reading structured results back — rather than calling
the Python functions underneath. That distinction matters: a tool can be
perfectly correct as a function and still be unusable over the wire because its
signature does not serialise, its docstring never reaches the client, or its
return value is not JSON.

Everything runs in-process against the same server object ``main()`` serves, so
no subprocess, no registration and no port are involved.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(autouse=True)
def session(monkeypatch, tmp_path):
    """A fresh registry and a fresh allowed root for every test."""
    root = tmp_path / "root"
    root.mkdir()
    for asset in ASSETS.iterdir():
        (root / asset.name).write_bytes(asset.read_bytes())
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


def _build_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE departments (department TEXT, floor INTEGER)")
    connection.executemany(
        "INSERT INTO departments VALUES (?, ?)",
        [("Engineering", 3), ("Sales", 1), ("Marketing", 2)],
    )
    connection.commit()
    connection.close()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_every_tool_is_advertised_with_a_description():
    async def _run():
        async with Client(server_module.mcp) as client:
            return await client.list_tools()

    tools = asyncio.run(_run())
    names = {tool.name for tool in tools}
    assert names == {
        "attach_datasource",
        "list_tables",
        "describe_table",
        "query",
        "export_query",
    }
    # A tool with no description is invisible to an agent choosing between them.
    for tool in tools:
        assert tool.description and len(tool.description) > 20, tool.name

    by_name = {tool.name: tool for tool in tools}
    for name in names:
        assert by_name[name].inputSchema["properties"], name


def test_the_nickname_is_required_everywhere_it_routes():
    """It names the engine to execute against, so it cannot be optional."""

    async def _run():
        async with Client(server_module.mcp) as client:
            return await client.list_tools()

    by_name = {tool.name: tool for tool in asyncio.run(_run())}
    for name in ("attach_datasource", "describe_table", "query", "export_query"):
        assert "nickname" in by_name[name].inputSchema["required"], name
    # Listing everything is the one case where it is genuinely optional.
    assert "nickname" not in by_name["list_tables"].inputSchema.get("required", [])


# ---------------------------------------------------------------------------
# A datasource is a database named by its nickname
# ---------------------------------------------------------------------------


def test_a_file_attaches_as_a_database_holding_one_table(session):
    attached = call(
        "attach_datasource", database=str(session / "simple.csv"), nickname="staff"
    )
    assert attached["ok"] is True
    assert attached["kind"] == "file"
    assert attached["tables"] == ["staff.simple"]
    assert attached["evicted"] is None


def test_attach_then_query_over_the_protocol(session):
    call("attach_datasource", database=str(session / "simple.csv"), nickname="staff")

    answer = call(
        "query", nickname="staff", sql="SELECT sum(salary) AS t FROM staff.simple"
    )
    assert answer["ok"] is True
    assert answer["columns"] == ["t"]
    assert answer["rows"][0][0] == 345000


def test_list_tables_reports_the_datasources_and_the_posture(session):
    call("attach_datasource", database=str(session / "simple.csv"), nickname="staff")
    listing = call("list_tables")

    assert listing["ok"] is True
    assert listing["datasources"] == [
        {
            "nickname": "staff",
            "kind": "file",
            "source": str(session / "simple.csv"),
            "tables": ["staff.simple"],
        }
    ]
    assert listing["slots_used"] == 1
    assert listing["slots_available"] == 10
    # The posture is reported so the LLM can see what it may reach.
    assert str(session) in listing["roots"]
    assert listing["path_limited"] is True


def test_list_tables_can_be_narrowed_to_one_datasource(session):
    call("attach_datasource", database=str(session / "simple.csv"), nickname="staff")
    call(
        "attach_datasource", database=str(session / "mixed_tabs.tsv"), nickname="tabbed"
    )

    listing = call("list_tables", nickname="staff")
    assert [entry["nickname"] for entry in listing["datasources"]] == ["staff"]


def test_describe_table_names_columns_and_types(session):
    call("attach_datasource", database=str(session / "simple.csv"), nickname="staff")
    described = call("describe_table", nickname="staff", table="simple")

    assert described["ok"] is True
    assert described["table"] == "staff.simple"
    types = {column["name"]: column["type"] for column in described["columns"]}
    assert types["salary"] == "INTEGER"
    assert types["department"] == "TEXT"


def test_query_limit_is_reported_as_truncation(session):
    call("attach_datasource", database=str(session / "simple.csv"), nickname="staff")

    answer = call("query", nickname="staff", sql="SELECT * FROM staff.simple", limit=2)
    assert answer["row_count"] == 2
    assert answer["truncated"] is True

    full = call("query", nickname="staff", sql="SELECT * FROM staff.simple", limit=0)
    assert full["row_count"] == 5
    assert full["truncated"] is False


# ---------------------------------------------------------------------------
# Joining across datasources — the capability the tool exists for
# ---------------------------------------------------------------------------


def test_a_file_joins_a_database_in_one_statement(session):
    _build_database(session / "hr.db")

    attached = call("attach_datasource", database=str(session / "hr.db"), nickname="hr")
    assert attached["ok"] is True
    assert attached["kind"] == "database"
    assert attached["tables"] == ["hr.departments"]

    call("attach_datasource", database=str(session / "simple.csv"), nickname="staff")
    answer = call(
        "query",
        nickname="staff",
        sql=(
            "SELECT s.name, d.floor FROM staff.simple s "
            "JOIN hr.departments d ON s.department = d.department "
            "ORDER BY s.name"
        ),
    )
    assert answer["ok"] is True
    assert answer["rows"][0] == ["Alice Johnson", 3]


def test_reattaching_a_nickname_replaces_it_rather_than_refusing(session):
    """The nickname is a handle; pointing it somewhere else is a normal act."""
    call("attach_datasource", database=str(session / "simple.csv"), nickname="slot")
    second = call(
        "attach_datasource", database=str(session / "mixed_tabs.tsv"), nickname="slot"
    )
    assert second["ok"] is True
    assert second["tables"] == ["slot.mixed_tabs"]
    assert call("list_tables")["slots_used"] == 1


# ---------------------------------------------------------------------------
# The slot limit, and what an eviction has to say
# ---------------------------------------------------------------------------


def test_the_eviction_is_reported_in_the_attachment_that_caused_it(session):
    config_module.use(Config(roots=(session,), slots=2))
    call("attach_datasource", database=str(session / "simple.csv"), nickname="a")
    call("attach_datasource", database=str(session / "mixed_tabs.tsv"), nickname="b")

    third = call(
        "attach_datasource", database=str(session / "no_header.csv"), nickname="c"
    )

    assert third["ok"] is True
    assert third["evicted"]["nickname"] == "a"
    assert third["evicted"]["tables"] == ["simple"]
    assert third["evicted"]["source"] == str(session / "simple.csv")


def test_querying_an_evicted_datasource_says_it_was_evicted(session):
    config_module.use(Config(roots=(session,), slots=1))
    call("attach_datasource", database=str(session / "simple.csv"), nickname="a")
    call("attach_datasource", database=str(session / "mixed_tabs.tsv"), nickname="b")

    answer = call("query", nickname="a", sql="SELECT * FROM a.simple")
    assert answer["ok"] is False
    assert "evicted" in answer["error"]
    # Enough to rebuild it without guessing.
    assert str(session / "simple.csv") in answer["error"]


# ---------------------------------------------------------------------------
# Errors reach the caller as answers, not as exceptions
# ---------------------------------------------------------------------------


def test_a_bad_path_is_answered_not_raised(session, tmp_path):
    answer = call(
        "attach_datasource", database=str(tmp_path / "elsewhere.csv"), nickname="x"
    )
    assert answer["ok"] is False
    assert "outside the allowed paths" in answer["error"]


def test_a_missing_file_is_answered(session):
    answer = call(
        "attach_datasource", database=str(session / "absent.csv"), nickname="x"
    )
    assert answer["ok"] is False
    assert "No such file" in answer["error"]


def test_an_unusable_nickname_is_answered(session):
    answer = call(
        "attach_datasource", database=str(session / "simple.csv"), nickname="my-data"
    )
    assert answer["ok"] is False
    assert "my-data" in answer["error"]


def test_an_unknown_nickname_is_answered(session):
    answer = call("query", nickname="nothing", sql="SELECT 1")
    assert answer["ok"] is False
    assert "nothing" in answer["error"]


def test_a_network_url_is_answered_while_the_network_is_closed(session):
    answer = call(
        "attach_datasource",
        database="postgresql://user:hunter2@db.example.com/sales",
        nickname="pg",
    )
    assert answer["ok"] is False
    assert "network" in answer["error"]
    assert "hunter2" not in answer["error"]


def test_invalid_sql_returns_the_engine_message(session):
    call("attach_datasource", database=str(session / "simple.csv"), nickname="staff")
    answer = call("query", nickname="staff", sql="SELECT * FROM nonexistent")
    assert answer["ok"] is False
    assert "no such table" in answer["error"].lower()


def test_mixed_columns_are_flagged_on_attach(session):
    """The signal that keeps an agent from trusting a wrong average."""
    attached = call(
        "attach_datasource",
        database=str(session / "messy_mixed_types.csv"),
        nickname="messy",
    )
    assert attached["ok"] is True
    assert any("coerce text to 0" in warning for warning in attached["warnings"])
    assert any("messy.messy_mixed_types" in w for w in attached["warnings"])


def test_the_mixed_column_detail_is_available_on_describe(session):
    call(
        "attach_datasource",
        database=str(session / "messy_mixed_types.csv"),
        nickname="messy",
    )
    described = call("describe_table", nickname="messy", table="messy_mixed_types")
    assert "id" in described["mixed_columns"]


# ---------------------------------------------------------------------------
# Export, including the overwrite boundary
# ---------------------------------------------------------------------------


def test_export_writes_and_reports(session):
    call("attach_datasource", database=str(session / "simple.csv"), nickname="staff")
    target = session / "out.csv"

    result = call(
        "export_query",
        nickname="staff",
        sql="SELECT name, salary FROM staff.simple ORDER BY name",
        path=str(target),
    )
    assert result["ok"] is True
    assert result["rows_written"] == 5
    assert result["replaced_existing"] is False
    assert target.read_text().splitlines()[0] == "name,salary"


def test_export_refuses_to_clobber_then_accepts_the_flag(session):
    call("attach_datasource", database=str(session / "simple.csv"), nickname="staff")
    target = session / "out.csv"
    target.write_text("existing content\n")

    refused = call(
        "export_query",
        nickname="staff",
        sql="SELECT name FROM staff.simple",
        path=str(target),
    )
    assert refused["ok"] is False
    assert "overwrite=true" in refused["error"]
    assert target.read_text() == "existing content\n"

    allowed = call(
        "export_query",
        nickname="staff",
        sql="SELECT name FROM staff.simple",
        path=str(target),
        overwrite=True,
    )
    assert allowed["ok"] is True
    assert allowed["replaced_existing"] is True
    assert target.read_text().splitlines()[0] == "name"


def test_export_ignores_the_display_limit(session):
    """A query capped at 2 rows for display must still export all 5."""
    call("attach_datasource", database=str(session / "simple.csv"), nickname="staff")
    target = session / "full.csv"
    result = call(
        "export_query",
        nickname="staff",
        sql="SELECT * FROM staff.simple",
        path=str(target),
    )
    assert result["rows_written"] == 5
