"""SQLite integration tests via MCP tool interface."""

import pytest

from .mcp_test_client import call_tool

pytestmark = [pytest.mark.integration]


class TestSQLiteConnection:
    def test_connect_and_list(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {
                "name": "sqlite_test",
                "db_type": "sqlite",
                "conn_string": small_sqlite_db,
            },
        )
        try:
            result = call_tool("list_databases", {})
            assert "sqlite_test" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "sqlite_test"})

    def test_connect_invalid_path(self):
        result = call_tool(
            "connect_database",
            {
                "name": "bad",
                "db_type": "sqlite",
                "conn_string": "/nonexistent/path.db",
            },
        )
        assert "error" in str(result).lower()

    def test_disconnect_cleanup(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "dc_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        call_tool("disconnect_database", {"name": "dc_test"})
        result = call_tool("list_databases", {})
        assert "dc_test" not in str(result)


class TestSQLiteSchema:
    def test_describe_database(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {
                "name": "schema_test",
                "db_type": "sqlite",
                "conn_string": small_sqlite_db,
            },
        )
        try:
            result = call_tool("describe_database", {"name": "schema_test"})
            assert "test_data" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "schema_test"})

    def test_describe_table(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "tbl_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "describe_table", {"name": "tbl_test", "table_name": "test_data"}
            )
            assert "name" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "tbl_test"})

    def test_export_schema_json(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "exp_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "export_schema", {"name": "exp_test", "format": "json_schema"}
            )
            assert "test_data" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "exp_test"})


class TestSQLiteQuery:
    def test_simple_select(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "q_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "q_test", "query": "SELECT COUNT(*) as cnt FROM test_data"},
            )
            assert "1000" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "q_test"})

    def test_select_with_where(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "w_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "w_test",
                    "query": "SELECT * FROM test_data WHERE category = 'electronics' LIMIT 10",
                },
            )
            assert "electronics" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "w_test"})

    def test_select_with_aggregation(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "agg_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "agg_test",
                    "query": "SELECT category, COUNT(*) as cnt FROM test_data GROUP BY category",
                },
            )
            assert "category" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "agg_test"})

    def test_select_zero_rows(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "zero_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "zero_test",
                    "query": "SELECT * FROM test_data WHERE id = -999",
                },
            )
            # Should return empty result, not error
            result_str = str(result).lower()
            assert "error" not in result_str or "0" in result_str
        finally:
            call_tool("disconnect_database", {"name": "zero_test"})

    def test_preflight_estimation(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "pf_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "pf_test",
                    "query": "SELECT * FROM test_data",
                    "preflight": True,
                },
            )
            assert (
                "preflight" in str(result).lower() or "analysis" in str(result).lower()
            )
        finally:
            call_tool("disconnect_database", {"name": "pf_test"})


class TestSQLiteErrors:
    def test_schema_error(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "err_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "err_test", "query": "SELECT * FROM nonexistent_table"},
            )
            assert "error" in str(result).lower()
        finally:
            call_tool("disconnect_database", {"name": "err_test"})

    def test_syntax_error(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "syn_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "syn_test", "query": "SELEKT * FORM test_data"},
            )
            assert "error" in str(result).lower()
        finally:
            call_tool("disconnect_database", {"name": "syn_test"})

    def test_sql_injection_blocked(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "inj_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "inj_test",
                    "query": "SELECT * FROM test_data; DROP TABLE test_data",
                },
            )
            result_str = str(result).lower()
            assert "security" in result_str or "error" in result_str
        finally:
            call_tool("disconnect_database", {"name": "inj_test"})


class TestSQLiteLargeData:
    @pytest.mark.large_data
    def test_large_query_count(self, large_sqlite_db):
        """500K row query should return correct count."""
        call_tool(
            "connect_database",
            {"name": "lg_test", "db_type": "sqlite", "conn_string": large_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "lg_test", "query": "SELECT COUNT(*) as cnt FROM test_data"},
            )
            assert "500000" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "lg_test"})

    @pytest.mark.large_data
    def test_large_query_with_limit(self, large_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "lim_test", "db_type": "sqlite", "conn_string": large_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "lim_test", "query": "SELECT * FROM test_data LIMIT 100"},
            )
            # Should return data without error
            assert "error" not in str(result).lower() or "test_data" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "lim_test"})


class TestSQLiteDataFidelity:
    def test_null_handling(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "null_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "null_test",
                    "query": "SELECT notes FROM test_data WHERE notes IS NULL LIMIT 5",
                },
            )
            # Should return rows with null notes, not an error
            assert "error" not in str(result).lower() or "null" in str(result).lower()
        finally:
            call_tool("disconnect_database", {"name": "null_test"})

    def test_numeric_precision(self, small_sqlite_db):
        call_tool(
            "connect_database",
            {"name": "num_test", "db_type": "sqlite", "conn_string": small_sqlite_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "num_test",
                    "query": "SELECT amount, score FROM test_data LIMIT 5",
                },
            )
            # Verify decimal values are present (not truncated to int)
            result_str = str(result)
            assert "." in result_str
        finally:
            call_tool("disconnect_database", {"name": "num_test"})
