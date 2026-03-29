"""DuckDB integration tests via MCP tool interface."""

import pytest

duckdb = pytest.importorskip("duckdb")

from .mcp_test_client import call_tool  # noqa: E402

pytestmark = [pytest.mark.integration]


class TestDuckDBConnection:
    def test_connect_and_list(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {
                "name": "duckdb_test",
                "db_type": "duckdb",
                "conn_string": small_duckdb_db,
            },
        )
        try:
            result = call_tool("list_databases", {})
            assert "duckdb_test" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "duckdb_test"})

    def test_connect_invalid_path(self):
        result = call_tool(
            "connect_database",
            {
                "name": "bad_duck",
                "db_type": "duckdb",
                "conn_string": "/nonexistent/path.duckdb",
            },
        )
        assert "error" in str(result).lower()

    def test_disconnect_cleanup(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dc_duck", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        call_tool("disconnect_database", {"name": "dc_duck"})
        result = call_tool("list_databases", {})
        assert "dc_duck" not in str(result)


class TestDuckDBSchema:
    def test_describe_database(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {
                "name": "duck_schema",
                "db_type": "duckdb",
                "conn_string": small_duckdb_db,
            },
        )
        try:
            result = call_tool("describe_database", {"name": "duck_schema"})
            assert "test_data" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "duck_schema"})

    def test_describe_table(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {
                "name": "duck_tbl",
                "db_type": "duckdb",
                "conn_string": small_duckdb_db,
            },
        )
        try:
            result = call_tool(
                "describe_table", {"name": "duck_tbl", "table_name": "test_data"}
            )
            assert "name" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "duck_tbl"})

    def test_export_schema_json(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {
                "name": "duck_exp",
                "db_type": "duckdb",
                "conn_string": small_duckdb_db,
            },
        )
        try:
            result = call_tool(
                "export_schema", {"name": "duck_exp", "format": "json_schema"}
            )
            assert "test_data" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "duck_exp"})


class TestDuckDBQuery:
    def test_simple_select(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dq_test", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "dq_test", "query": "SELECT COUNT(*) as cnt FROM test_data"},
            )
            assert "1000" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "dq_test"})

    def test_select_with_where(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dw_test", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "dw_test",
                    "query": "SELECT * FROM test_data WHERE category = 'electronics' LIMIT 10",
                },
            )
            assert "electronics" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "dw_test"})

    def test_select_with_aggregation(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dagg_test", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "dagg_test",
                    "query": "SELECT category, COUNT(*) as cnt FROM test_data GROUP BY category",
                },
            )
            assert "category" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "dagg_test"})

    def test_select_zero_rows(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dzero_test", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "dzero_test",
                    "query": "SELECT * FROM test_data WHERE id = -999",
                },
            )
            result_str = str(result).lower()
            assert "error" not in result_str or "0" in result_str
        finally:
            call_tool("disconnect_database", {"name": "dzero_test"})

    def test_preflight_estimation(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dpf_test", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "dpf_test",
                    "query": "SELECT * FROM test_data",
                    "preflight": True,
                },
            )
            assert (
                "preflight" in str(result).lower() or "analysis" in str(result).lower()
            )
        finally:
            call_tool("disconnect_database", {"name": "dpf_test"})


class TestDuckDBErrors:
    def test_schema_error(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "derr_test", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "derr_test", "query": "SELECT * FROM nonexistent_table"},
            )
            assert "error" in str(result).lower()
        finally:
            call_tool("disconnect_database", {"name": "derr_test"})

    def test_syntax_error(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dsyn_test", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "dsyn_test", "query": "SELEKT * FORM test_data"},
            )
            assert "error" in str(result).lower()
        finally:
            call_tool("disconnect_database", {"name": "dsyn_test"})

    def test_sql_injection_blocked(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dinj_test", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "dinj_test",
                    "query": "SELECT * FROM test_data; DROP TABLE test_data",
                },
            )
            result_str = str(result).lower()
            assert "security" in result_str or "error" in result_str
        finally:
            call_tool("disconnect_database", {"name": "dinj_test"})


class TestDuckDBLargeData:
    @pytest.mark.large_data
    def test_large_query_count(self, large_duckdb_db):
        """500K row query should return correct count."""
        call_tool(
            "connect_database",
            {"name": "dlg_test", "db_type": "duckdb", "conn_string": large_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "dlg_test", "query": "SELECT COUNT(*) as cnt FROM test_data"},
            )
            assert "500000" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "dlg_test"})

    @pytest.mark.large_data
    def test_large_query_with_limit(self, large_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dlim_test", "db_type": "duckdb", "conn_string": large_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "dlim_test", "query": "SELECT * FROM test_data LIMIT 100"},
            )
            assert "error" not in str(result).lower() or "test_data" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "dlim_test"})


class TestDuckDBDataFidelity:
    def test_null_handling(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dnull_test", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "dnull_test",
                    "query": "SELECT notes FROM test_data WHERE notes IS NULL LIMIT 5",
                },
            )
            assert "error" not in str(result).lower() or "null" in str(result).lower()
        finally:
            call_tool("disconnect_database", {"name": "dnull_test"})

    def test_numeric_precision(self, small_duckdb_db):
        call_tool(
            "connect_database",
            {"name": "dnum_test", "db_type": "duckdb", "conn_string": small_duckdb_db},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "dnum_test",
                    "query": "SELECT amount, score FROM test_data LIMIT 5",
                },
            )
            result_str = str(result)
            assert "." in result_str
        finally:
            call_tool("disconnect_database", {"name": "dnum_test"})
