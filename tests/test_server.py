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

from localdata_mcp import paths as paths_module
from localdata_mcp import server as server_module

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(autouse=True)
def session(monkeypatch, tmp_path):
    """A fresh workspace and a fresh allowed root for every test."""
    root = tmp_path / "root"
    root.mkdir()
    for asset in ASSETS.iterdir():
        (root / asset.name).write_bytes(asset.read_bytes())
    monkeypatch.setenv(paths_module.ROOT_ENV_VAR, str(root))
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
        "load_file",
        "attach_database",
        "list_tables",
        "describe_table",
        "query",
        "export_query",
    }
    # A tool with no description is invisible to an agent choosing between them.
    for tool in tools:
        assert tool.description and len(tool.description) > 20, tool.name

    # Every tool that takes arguments must document them; list_tables takes none.
    by_name = {tool.name: tool for tool in tools}
    assert by_name["list_tables"].inputSchema["properties"] == {}
    for name in (
        "load_file",
        "attach_database",
        "describe_table",
        "query",
        "export_query",
    ):
        assert by_name[name].inputSchema["properties"], name


# ---------------------------------------------------------------------------
# The core flow: load, inspect, query, export
# ---------------------------------------------------------------------------


def test_load_then_query_over_the_protocol(session):
    loaded = call("load_file", path=str(session / "simple.csv"))
    assert loaded["ok"] is True
    assert loaded["rows"] == 5
    assert loaded["table"] == "simple"

    answer = call("query", sql="SELECT sum(salary) AS total FROM simple")
    assert answer["ok"] is True
    assert answer["columns"] == ["total"]
    assert answer["rows"][0][0] == 345000


def test_list_tables_reports_what_is_loaded(session):
    call("load_file", path=str(session / "simple.csv"))
    listing = call("list_tables")
    assert listing["ok"] is True
    assert [entry["table"] for entry in listing["loaded"]] == ["simple"]
    assert listing["root"] == str(session)


def test_describe_table_names_columns_and_types(session):
    call("load_file", path=str(session / "simple.csv"))
    described = call("describe_table", table="simple")
    assert described["ok"] is True
    types = {column["name"]: column["type"] for column in described["columns"]}
    assert types["salary"] == "INTEGER"
    assert types["department"] == "TEXT"


def test_query_limit_is_reported_as_truncation(session):
    call("load_file", path=str(session / "simple.csv"))
    answer = call("query", sql="SELECT * FROM simple", limit=2)
    assert answer["row_count"] == 2
    assert answer["truncated"] is True

    full = call("query", sql="SELECT * FROM simple", limit=0)
    assert full["row_count"] == 5
    assert full["truncated"] is False


# ---------------------------------------------------------------------------
# Joining across sources — the capability the tool exists for
# ---------------------------------------------------------------------------


def test_join_a_csv_against_an_attached_database(session):
    _build_database(session / "hr.db")

    attached = call("attach_database", path=str(session / "hr.db"), alias="hr")
    assert attached["ok"] is True
    assert attached["tables"] == ["hr.departments"]

    call("load_file", path=str(session / "simple.csv"))
    answer = call(
        "query",
        sql=(
            "SELECT s.name, d.floor FROM simple s "
            "JOIN hr.departments d ON s.department = d.department "
            "ORDER BY s.name"
        ),
    )
    assert answer["ok"] is True
    assert answer["rows"][0] == ["Alice Johnson", 3]


def test_attaching_the_same_alias_twice_is_refused(session):
    _build_database(session / "hr.db")
    assert call("attach_database", path=str(session / "hr.db"), alias="hr")["ok"]
    second = call("attach_database", path=str(session / "hr.db"), alias="hr")
    assert second["ok"] is False
    assert "already attached" in second["error"]


# ---------------------------------------------------------------------------
# Errors reach the caller as answers, not as exceptions
# ---------------------------------------------------------------------------


def test_a_bad_path_is_answered_not_raised(session, tmp_path):
    answer = call("load_file", path=str(tmp_path / "elsewhere.csv"))
    assert answer["ok"] is False
    assert "outside the allowed root" in answer["error"]


def test_a_missing_file_is_answered(session):
    answer = call("load_file", path=str(session / "absent.csv"))
    assert answer["ok"] is False
    assert "No such file" in answer["error"]


def test_invalid_sql_returns_the_engine_message(session):
    call("load_file", path=str(session / "simple.csv"))
    answer = call("query", sql="SELECT * FROM nonexistent")
    assert answer["ok"] is False
    assert "no such table" in answer["error"].lower()


def test_mixed_columns_are_flagged_on_load(session):
    """The signal that keeps an agent from trusting a wrong average."""
    loaded = call("load_file", path=str(session / "messy_mixed_types.csv"))
    assert loaded["ok"] is True
    # `id` holds 1, 2, '3a', … — genuinely more than one storage class.
    assert "id" in loaded["mixed_columns"]
    assert any("coerce text to 0" in w for w in loaded["warnings"])


# ---------------------------------------------------------------------------
# Export, including the overwrite boundary
# ---------------------------------------------------------------------------


def test_export_writes_and_reports(session):
    call("load_file", path=str(session / "simple.csv"))
    target = session / "out.csv"

    result = call(
        "export_query",
        sql="SELECT name, salary FROM simple ORDER BY name",
        path=str(target),
    )
    assert result["ok"] is True
    assert result["rows_written"] == 5
    assert result["replaced_existing"] is False
    assert target.read_text().splitlines()[0] == "name,salary"


def test_export_refuses_to_clobber_then_accepts_the_flag(session):
    call("load_file", path=str(session / "simple.csv"))
    target = session / "out.csv"
    target.write_text("existing content\n")

    refused = call("export_query", sql="SELECT name FROM simple", path=str(target))
    assert refused["ok"] is False
    assert "overwrite=true" in refused["error"]
    assert target.read_text() == "existing content\n"

    allowed = call(
        "export_query",
        sql="SELECT name FROM simple",
        path=str(target),
        overwrite=True,
    )
    assert allowed["ok"] is True
    assert allowed["replaced_existing"] is True
    assert target.read_text().splitlines()[0] == "name"


def test_export_ignores_the_display_limit(session):
    """A query capped at 2 rows for display must still export all 5."""
    call("load_file", path=str(session / "simple.csv"))
    target = session / "full.csv"
    result = call("export_query", sql="SELECT * FROM simple", path=str(target))
    assert result["rows_written"] == 5
