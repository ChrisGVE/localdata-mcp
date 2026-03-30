"""MySQL integration tests via MCP tool interface.

Requires a running MySQL instance. Default:
    docker run -e MYSQL_ROOT_PASSWORD=rootpass -e MYSQL_DATABASE=testdb \
        -e MYSQL_USER=testuser -e MYSQL_PASSWORD=testpass \
        -p 13306:3306 -d mysql:8
"""

import os

import pytest
from sqlalchemy import create_engine, text

from .mcp_test_client import call_tool

MYSQL_URL = os.environ.get(
    "TEST_MYSQL_URL",
    "mysql+mysqlconnector://testuser:testpass@localhost:13306/testdb",
)


def _mysql_available():
    """Check whether the MySQL test instance is reachable."""
    try:
        engine = create_engine(MYSQL_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.mysql,
    pytest.mark.skipif(not _mysql_available(), reason="MySQL not available"),
]


@pytest.fixture(scope="module", autouse=True)
def setup_mysql_data():
    """Create a test_data table with 1000 rows for the test module."""
    engine = create_engine(MYSQL_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS test_data"))
        conn.execute(
            text(
                """
            CREATE TABLE test_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(200),
                amount DECIMAL(10,2),
                category VARCHAR(50),
                score FLOAT,
                is_active TINYINT(1),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            )
        )
        for i in range(1000):
            conn.execute(
                text(
                    "INSERT INTO test_data (name, email, amount, category, score, is_active) "
                    "VALUES (:n, :e, :a, :c, :s, :active)"
                ),
                {
                    "n": f"user_{i}",
                    "e": f"user_{i}@test.com",
                    "a": float(i * 1.5),
                    "c": ["X", "Y", "Z"][i % 3],
                    "s": float(i % 100) / 100.0,
                    "active": i % 2,
                },
            )
        conn.commit()
    engine.dispose()
    yield
    engine = create_engine(MYSQL_URL)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS test_data"))
        conn.commit()
    engine.dispose()


def _connect(name):
    """Connect to MySQL via MCP and assert success."""
    result = call_tool(
        "connect_database",
        {"name": name, "db_type": "mysql", "conn_string": MYSQL_URL},
    )
    assert "error" not in str(result).lower(), f"Connection failed: {result}"


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------


class TestMySQLConnection:
    def test_connect_and_describe(self):
        """Connect to MySQL, describe the database, then disconnect."""
        _connect("mysql_conn")
        try:
            result = call_tool("describe_database", {"name": "mysql_conn"})
            assert "test_data" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "mysql_conn"})

    def test_connect_and_list(self):
        """Connected database appears in list_databases output."""
        _connect("mysql_list")
        try:
            result = call_tool("list_databases", {})
            assert "mysql_list" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "mysql_list"})

    def test_disconnect_removes_from_list(self):
        """After disconnect the database no longer appears."""
        _connect("mysql_disc")
        call_tool("disconnect_database", {"name": "mysql_disc"})
        result = call_tool("list_databases", {})
        assert "mysql_disc" not in str(result)


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------


class TestMySQLQuery:
    def test_count(self):
        """SELECT COUNT(*) returns 1000."""
        _connect("mysql_cnt")
        try:
            result = call_tool(
                "execute_query",
                {"name": "mysql_cnt", "query": "SELECT COUNT(*) AS cnt FROM test_data"},
            )
            assert "1000" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "mysql_cnt"})

    def test_where_filter(self):
        """WHERE filter returns only matching rows."""
        _connect("mysql_whr")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mysql_whr",
                    "query": (
                        "SELECT id, name, category FROM test_data "
                        "WHERE category = 'X' LIMIT 5"
                    ),
                },
            )
            result_str = str(result)
            assert "X" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mysql_whr"})

    def test_group_by_aggregation(self):
        """GROUP BY returns one row per category."""
        _connect("mysql_grp")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mysql_grp",
                    "query": (
                        "SELECT category, COUNT(*) AS cnt "
                        "FROM test_data GROUP BY category"
                    ),
                },
            )
            result_str = str(result)
            for cat in ("X", "Y", "Z"):
                assert cat in result_str
        finally:
            call_tool("disconnect_database", {"name": "mysql_grp"})

    def test_limit(self):
        """LIMIT constrains the number of returned rows."""
        _connect("mysql_lim")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mysql_lim",
                    "query": "SELECT id, name, category FROM test_data LIMIT 10",
                },
            )
            result_str = str(result)
            assert "user_" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mysql_lim"})


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestMySQLSchema:
    def test_describe_table(self):
        """describe_table returns column metadata."""
        _connect("mysql_dtbl")
        try:
            result = call_tool(
                "describe_table", {"name": "mysql_dtbl", "table_name": "test_data"}
            )
            result_str = str(result)
            assert "name" in result_str
            assert "amount" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mysql_dtbl"})

    def test_export_schema_json_schema(self):
        """export_schema with json_schema format contains table info."""
        _connect("mysql_ejs")
        try:
            result = call_tool(
                "export_schema", {"name": "mysql_ejs", "format": "json_schema"}
            )
            assert "test_data" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "mysql_ejs"})

    def test_export_schema_typescript(self):
        """export_schema with typescript format produces output."""
        _connect("mysql_ets")
        try:
            result = call_tool(
                "export_schema", {"name": "mysql_ets", "format": "typescript"}
            )
            result_str = str(result).lower()
            # TypeScript export uses PascalCase (TestData) for interface name
            assert "testdata" in result_str or "test_data" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mysql_ets"})

    def test_export_schema_python(self):
        """export_schema with python format produces output."""
        _connect("mysql_epy")
        try:
            result = call_tool(
                "export_schema", {"name": "mysql_epy", "format": "python"}
            )
            result_str = str(result).lower()
            # Python export uses PascalCase class name (TestData)
            assert "testdata" in result_str or "test_data" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mysql_epy"})


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------


class TestMySQLErrors:
    def test_nonexistent_table(self):
        """Query against a missing table returns an error dict."""
        _connect("mysql_enx")
        try:
            result = call_tool(
                "execute_query",
                {"name": "mysql_enx", "query": "SELECT * FROM no_such_table"},
            )
            assert "error" in str(result).lower()
        finally:
            call_tool("disconnect_database", {"name": "mysql_enx"})

    def test_malformed_sql(self):
        """Malformed SQL returns an error dict."""
        _connect("mysql_eml")
        try:
            result = call_tool(
                "execute_query",
                {"name": "mysql_eml", "query": "SELEKT * FORM test_data"},
            )
            assert "error" in str(result).lower()
        finally:
            call_tool("disconnect_database", {"name": "mysql_eml"})


# ---------------------------------------------------------------------------
# Data fidelity tests
# ---------------------------------------------------------------------------


class TestMySQLDataFidelity:
    def test_decimal_precision(self):
        """DECIMAL(10,2) values preserve their precision."""
        _connect("mysql_dec")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mysql_dec",
                    "query": "SELECT amount FROM test_data WHERE id = 5",
                },
            )
            # id=5 → i=4 (0-based AUTO_INCREMENT starts at 1), amount = 4 * 1.5 = 6.00
            result_str = str(result)
            assert "6.0" in result_str
        finally:
            call_tool("disconnect_database", {"name": "mysql_dec"})

    def test_tinyint_boolean(self):
        """TINYINT(1) boolean values are returned correctly."""
        _connect("mysql_bool")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mysql_bool",
                    "query": (
                        "SELECT is_active, COUNT(*) AS cnt "
                        "FROM test_data GROUP BY is_active"
                    ),
                },
            )
            result_str = str(result)
            # Should contain both boolean states (0/1 or false/true)
            assert "0" in result_str or "false" in result_str.lower()
            assert "1" in result_str or "true" in result_str.lower()
        finally:
            call_tool("disconnect_database", {"name": "mysql_bool"})

    def test_select_star(self):
        """SELECT * with DECIMAL and TINYINT columns no longer crashes."""
        _connect("mysql_star")
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "mysql_star",
                    "query": "SELECT * FROM test_data LIMIT 5",
                },
            )
            result_str = str(result)
            assert "error" not in result_str.lower(), f"SELECT * crashed: {result_str}"
            assert "user_" in result_str, (
                f"Expected user data in SELECT * results: {result_str}"
            )
        finally:
            call_tool("disconnect_database", {"name": "mysql_star"})
