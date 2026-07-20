"""Tests for regex, schema export, and audit tool registration."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from localdata_mcp import DatabaseManager


@pytest.fixture
def manager():
    """Create a DatabaseManager instance."""
    return DatabaseManager()


class TestToolRegistration:
    """Verify that new tools are registered on the MCP server."""

    def test_search_data_tool_exists(self, manager):
        assert hasattr(manager, "search_data")
        assert callable(manager.search_data)

    def test_transform_data_tool_exists(self, manager):
        assert hasattr(manager, "transform_data")
        assert callable(manager.transform_data)

    def test_export_schema_tool_exists(self, manager):
        assert hasattr(manager, "export_schema")
        assert callable(manager.export_schema)

    def test_get_query_log_tool_exists(self, manager):
        assert hasattr(manager, "get_query_log")
        assert callable(manager.get_query_log)

    def test_get_error_log_tool_exists(self, manager):
        assert hasattr(manager, "get_error_log")
        assert callable(manager.get_error_log)

    def test_tools_registered_on_mcp_server(self, manager):
        """Verify add_tool is called for each new tool during registration."""
        mock_server = MagicMock()
        manager._register_tools(mock_server)
        registered = [
            call.args[0].__name__ for call in mock_server.add_tool.call_args_list
        ]
        assert "search_data" in registered
        assert "transform_data" in registered
        assert "export_schema" in registered
        assert "get_query_log" in registered
        assert "get_error_log" in registered


class TestGetQueryLog:
    """Tests for get_query_log tool."""

    def test_returns_empty_entries(self, manager):
        from localdata_mcp.query_audit import get_query_audit_buffer

        get_query_audit_buffer().clear()
        result = json.loads(manager.get_query_log())
        assert "entries" in result
        assert "total_entries" in result
        assert result["total_entries"] == 0

    def test_filters_by_database(self, manager):
        from localdata_mcp.query_audit import get_query_audit_buffer

        buf = get_query_audit_buffer()
        buf.clear()
        buf.record_query("db1", "SELECT 1", "success", 10.0, rows_returned=1)
        buf.record_query("db2", "SELECT 2", "success", 20.0, rows_returned=1)

        result = json.loads(manager.get_query_log(database="db1"))
        assert result["total_entries"] == 1
        assert result["entries"][0]["database"] == "db1"
        buf.clear()


class TestGetErrorLog:
    """Tests for get_error_log tool."""

    def test_returns_only_errors_and_timeouts(self, manager):
        from localdata_mcp.query_audit import get_query_audit_buffer

        buf = get_query_audit_buffer()
        buf.clear()
        buf.record_query("db1", "SELECT 1", "success", 10.0)
        buf.record_query("db1", "BAD QUERY", "error", 5.0, error_type="SyntaxError")
        buf.record_query("db1", "SLOW Q", "timeout", 30000.0, error_type="Timeout")

        result = json.loads(manager.get_error_log())
        assert result["total_entries"] == 2
        statuses = {e["status"] for e in result["entries"]}
        assert statuses <= {"error", "timeout"}
        buf.clear()


class TestRecordAudit:
    """Tests for audit recording in execute_query."""

    def test_record_audit_helper(self, manager):
        """Test the _record_audit helper records entries."""
        import time as time_mod

        from localdata_mcp.query_audit import get_query_audit_buffer

        buf = get_query_audit_buffer()
        buf.clear()
        start = time_mod.time()
        manager._record_audit("testdb", "SELECT 1", "success", start, rows_returned=5)
        entries = buf.get_entries(database="testdb")
        assert len(entries) == 1
        assert entries[0].rows_returned == 5
        buf.clear()


class TestExportSchema:
    """Tests for export_schema tool."""

    def test_unknown_format_returns_error(self, manager):
        """Unknown format should return an error dict."""
        manager.connections["test"] = MagicMock()
        with patch(
            "localdata_mcp.localdata_mcp.DatabaseManager._get_connection",
            return_value=MagicMock(),
        ):
            with patch("localdata_mcp.schema_export.SchemaIntrospector"):
                result = json.loads(manager.export_schema("test", format="invalid_fmt"))
                assert "error" in result
                assert "invalid_fmt" in result["error"]


class TestToolCountClaims:
    """Keep the shipped tool surface and every claim about it in agreement.

    The tool count appeared in six places — the package metadata, the plugin
    manifest, the README, the docs index and the tools reference — and drifted
    from reality as domains were added. Worse, the reference documented a
    `get_metrics` tool that was never registered at all, so a reader could not
    tell a real tool from a fictional one. These tests make the documentation
    answerable to the registry rather than to whoever edited it last.
    """

    @staticmethod
    def _registered_tool_names(manager: DatabaseManager) -> set:
        names = set()

        class _Recorder:
            def add_tool(self, func):
                names.add(func.__name__)

        manager._register_tools(_Recorder())
        return names

    @staticmethod
    def _documented_tool_names() -> set:
        import re

        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "docs", "tools-reference.md"
        )
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        # Headings after this point name workflows, not tools.
        body = text[: text.index("## Composition Patterns")]
        return set(re.findall(r"^### (\w+)$", body, re.MULTILINE))

    def test_every_registered_tool_is_documented(self, manager):
        registered = self._registered_tool_names(manager)
        documented = self._documented_tool_names()

        assert (
            registered - documented == set()
        ), "tools are registered but missing from docs/tools-reference.md"

    def test_no_documented_tool_is_fictional(self, manager):
        registered = self._registered_tool_names(manager)
        documented = self._documented_tool_names()

        assert (
            documented - registered == set()
        ), "docs/tools-reference.md documents tools the server does not register"

    def test_advertised_count_matches_the_registry(self, manager):
        """Every file that states a tool count must state the true one."""
        count = len(self._registered_tool_names(manager))
        root = os.path.dirname(os.path.dirname(__file__))

        for relative in (
            "pyproject.toml",
            "server.json",
            ".claude-plugin/plugin.json",
            "README.md",
            "docs/index.md",
            "docs/tools-reference.md",
        ):
            with open(os.path.join(root, relative), encoding="utf-8") as handle:
                text = handle.read()
            assert (
                f"{count} tools" in text or f"{count} MCP tools" in text
            ), f"{relative} does not state the true tool count of {count}"
