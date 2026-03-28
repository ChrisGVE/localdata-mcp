"""Smoke test to verify MCP test client works."""

import pytest

from .mcp_test_client import call_tool


@pytest.mark.integration
class TestSmoke:
    def test_list_databases_returns_json(self):
        result = call_tool("list_databases", {})
        assert isinstance(result, (dict, str))

    def test_connect_sqlite_and_query(self, small_sqlite_db):
        # Connect using the actual parameter names from connect_database()
        call_tool(
            "connect_database",
            {
                "name": "smoke_test",
                "db_type": "sqlite",
                "conn_string": small_sqlite_db,
            },
        )
        try:
            # Query
            result = call_tool(
                "execute_query",
                {
                    "name": "smoke_test",
                    "query": "SELECT COUNT(*) as cnt FROM test_data",
                },
            )
            assert (
                "1000" in str(result) or result.get("data", [{}])[0].get("cnt") == 1000
            )
        finally:
            # Disconnect
            call_tool("disconnect_database", {"name": "smoke_test"})
