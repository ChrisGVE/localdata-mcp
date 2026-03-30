"""MS SQL Server integration tests via MCP tool interface.

Requires a running SQL Server instance. Start via Docker:

    docker run -e ACCEPT_EULA=Y -e 'SA_PASSWORD=TestPass123!' \
        -p 11433:1433 -d mcr.microsoft.com/mssql/server:2022-latest
"""

import os

import pytest
from sqlalchemy import create_engine, text

from .mcp_test_client import call_tool

MSSQL_URL = os.environ.get(
    "TEST_MSSQL_URL", "mssql+pymssql://sa:TestPass123!@localhost:11433/master"
)


def _mssql_available():
    try:
        engine = create_engine(MSSQL_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.mssql,
    pytest.mark.skipif(not _mssql_available(), reason="MSSQL not available"),
]


@pytest.fixture(scope="module", autouse=True)
def setup_mssql_data():
    """Create a test_data table with 1000 rows in the MSSQL instance."""
    engine = create_engine(MSSQL_URL)
    with engine.connect() as conn:
        conn.execute(
            text(
                "IF OBJECT_ID('dbo.test_data', 'U') IS NOT NULL "
                "DROP TABLE dbo.test_data"
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE dbo.test_data (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    name NVARCHAR(100),
                    email NVARCHAR(200),
                    amount DECIMAL(10,2),
                    category NVARCHAR(50),
                    score FLOAT,
                    is_active BIT,
                    created_at DATETIME2 DEFAULT GETDATE()
                )
                """
            )
        )
        for i in range(1000):
            conn.execute(
                text(
                    "INSERT INTO dbo.test_data "
                    "(name, email, amount, category, score, is_active) "
                    "VALUES (:n, :e, :a, :c, :s, :active)"
                ),
                {
                    "n": f"user_{i}",
                    "e": f"user_{i}@test.com",
                    "a": float(i * 1.5),
                    "c": ["P", "Q", "R"][i % 3],
                    "s": float(i % 100) / 100.0,
                    "active": i % 2,
                },
            )
        conn.commit()
    engine.dispose()
    yield
    engine = create_engine(MSSQL_URL)
    with engine.connect() as conn:
        conn.execute(
            text(
                "IF OBJECT_ID('dbo.test_data', 'U') IS NOT NULL "
                "DROP TABLE dbo.test_data"
            )
        )
        conn.commit()
    engine.dispose()


def _connect(name):
    """Connect to MSSQL via MCP and assert success."""
    result = call_tool(
        "connect_database",
        {"name": name, "db_type": "mssql", "conn_string": MSSQL_URL},
    )
    # MSSQL system tables (spt_monitor) have columns like 'total_errors' so we
    # cannot do a naive "error" substring check.  Instead verify the result
    # dict reports success.
    if isinstance(result, dict):
        assert result.get("success") is True, f"Connection failed: {result}"
    else:
        assert "error" not in str(result).lower(), f"Connection failed: {result}"


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------


class TestMSSQLConnection:
    def test_connect_and_list(self):
        _connect("mssql_conn")
        try:
            result = call_tool("list_databases", {})
            assert "mssql_conn" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "mssql_conn"})

    def test_describe_database(self):
        _connect("mssql_desc")
        try:
            result = call_tool("describe_database", {"name": "mssql_desc"})
            assert "test_data" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "mssql_desc"})

    def test_disconnect_cleanup(self):
        _connect("mssql_dc")
        call_tool("disconnect_database", {"name": "mssql_dc"})
        result = call_tool("list_databases", {})
        assert "mssql_dc" not in str(result)


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------


class TestMSSQLQueries:
    def test_count(self):
        _connect("mssql_cnt")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mssql_cnt",
                    "query": "SELECT COUNT(*) AS cnt FROM test_data",
                },
            )
            assert "1000" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "mssql_cnt"})

    def test_where_filter(self):
        _connect("mssql_whr")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mssql_whr",
                    "query": (
                        "SELECT id, name, email, amount, category, score "
                        "FROM test_data WHERE category = 'P'"
                    ),
                },
            )
            result_str = str(result)
            assert "P" in result_str
            assert "error" not in result_str.lower()
        finally:
            call_tool("disconnect_database", {"name": "mssql_whr"})

    def test_group_by_aggregation(self):
        _connect("mssql_grp")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mssql_grp",
                    "query": (
                        "SELECT category, COUNT(*) AS cnt, "
                        "SUM(amount) AS total "
                        "FROM test_data GROUP BY category"
                    ),
                },
            )
            result_str = str(result)
            assert "category" in result_str.lower() or "P" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mssql_grp"})

    def test_top_limit(self):
        _connect("mssql_top")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mssql_top",
                    "query": (
                        "SELECT TOP 5 id, name, email, "
                        "amount, category, score "
                        "FROM test_data ORDER BY id"
                    ),
                },
            )
            result_str = str(result)
            assert "user_0" in result_str
            assert "error" not in result_str.lower()
        finally:
            call_tool("disconnect_database", {"name": "mssql_top"})


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestMSSQLSchema:
    def test_describe_table(self):
        _connect("mssql_dtbl")
        try:
            result = call_tool(
                "describe_table",
                {"name": "mssql_dtbl", "table_name": "test_data"},
            )
            result_str = str(result).lower()
            assert "name" in result_str
            assert "amount" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mssql_dtbl"})

    def test_export_schema_json(self):
        _connect("mssql_ejs")
        try:
            result = call_tool(
                "export_schema",
                {"name": "mssql_ejs", "format": "json_schema"},
            )
            assert "test_data" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "mssql_ejs"})

    def test_export_schema_typescript(self):
        _connect("mssql_ets")
        try:
            result = call_tool(
                "export_schema",
                {"name": "mssql_ets", "format": "typescript"},
            )
            result_str = str(result)
            assert "test_data" in result_str.lower() or "TestData" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mssql_ets"})


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestMSSQLErrors:
    def test_nonexistent_table(self):
        _connect("mssql_enx")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mssql_enx",
                    "query": "SELECT * FROM nonexistent_table_xyz",
                },
            )
            assert "error" in str(result).lower()
        finally:
            call_tool("disconnect_database", {"name": "mssql_enx"})

    def test_malformed_sql(self):
        _connect("mssql_eml")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mssql_eml",
                    "query": "SELEKT * FORM test_data",
                },
            )
            assert "error" in str(result).lower()
        finally:
            call_tool("disconnect_database", {"name": "mssql_eml"})


# ---------------------------------------------------------------------------
# Data fidelity tests
# ---------------------------------------------------------------------------


class TestMSSQLDataFidelity:
    def test_decimal_precision(self):
        _connect("mssql_dec")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mssql_dec",
                    "query": "SELECT amount FROM test_data WHERE id = 2",
                },
            )
            # user_1 has amount = 1 * 1.5 = 1.50
            result_str = str(result)
            assert "1.5" in result_str or "1.50" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mssql_dec"})

    def test_bit_boolean_handling(self):
        _connect("mssql_bit")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mssql_bit",
                    "query": (
                        "SELECT is_active, COUNT(*) AS cnt "
                        "FROM test_data GROUP BY is_active"
                    ),
                },
            )
            result_str = str(result)
            # Should have two groups (0 and 1) with 500 each
            assert "500" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mssql_bit"})

    def test_nvarchar_unicode(self):
        """Verify NVARCHAR data round-trips correctly."""
        _connect("mssql_uni")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mssql_uni",
                    "query": (
                        "SELECT name, email FROM test_data WHERE name = 'user_42'"
                    ),
                },
            )
            result_str = str(result)
            assert "user_42" in result_str
            assert "user_42@test.com" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mssql_uni"})

    def test_select_star(self):
        """SELECT * with DECIMAL and BIT columns no longer crashes."""
        _connect("mssql_star")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mssql_star",
                    "query": "SELECT TOP 5 * FROM test_data",
                },
            )
            result_str = str(result)
            assert "error" not in result_str.lower(), f"SELECT * crashed: {result_str}"
            assert "user_" in result_str, (
                f"Expected user data in SELECT * results: {result_str}"
            )
        finally:
            call_tool("disconnect_database", {"name": "mssql_star"})
