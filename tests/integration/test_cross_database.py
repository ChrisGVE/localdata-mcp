"""Cross-database consistency tests.

Verify that common operations behave consistently across all available
database types: error handling, schema export, describe_database, and
basic queries.
"""

import csv
import json
import os
import sqlite3

import pytest
import yaml

from .mcp_test_client import call_tool

pytestmark = [pytest.mark.integration]

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

ROWS = [
    {"id": 1, "name": "Alice", "value": 100},
    {"id": 2, "name": "Bob", "value": 200},
    {"id": 3, "name": "Charlie", "value": 300},
]


# ---------------------------------------------------------------------------
# File-creation helpers
# ---------------------------------------------------------------------------


def _create_sqlite(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE data_table (id INTEGER, name TEXT, value INTEGER)")
    conn.executemany("INSERT INTO data_table VALUES (:id, :name, :value)", ROWS)
    conn.commit()
    conn.close()


def _create_csv(path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "value"])
        w.writeheader()
        w.writerows(ROWS)


def _create_json(path):
    with open(path, "w") as f:
        json.dump({"key": "value", "count": 42}, f)


def _create_yaml(path):
    with open(path, "w") as f:
        yaml.dump({"key": "value", "count": 42}, f)


def _create_xml(path):
    lines = ['<?xml version="1.0"?>', "<data>"]
    for row in ROWS:
        lines.append("  <row>")
        for k, v in row.items():
            lines.append(f"    <{k}>{v}</{k}>")
        lines.append("  </row>")
    lines.append("</data>")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Database types that expose a SQL interface (use execute_query, export_schema)
SQL_DB_TYPES = ["sqlite", "csv", "xml"]

# Database types that expose a tree interface (use get_children, list_keys)
TREE_DB_TYPES = ["json", "yaml"]

ALL_DB_TYPES = SQL_DB_TYPES + TREE_DB_TYPES

# Map db_type -> (file extension, creator function)
_DB_CREATORS = {
    "sqlite": (".db", _create_sqlite),
    "csv": (".csv", _create_csv),
    "json": (".json", _create_json),
    "yaml": (".yaml", _create_yaml),
    "xml": (".xml", _create_xml),
}


@pytest.fixture(params=ALL_DB_TYPES)
def any_db(request, tmp_path):
    """Yield (db_type, connection_name, file_path) for every database type."""
    db_type = request.param
    ext, creator = _DB_CREATORS[db_type]
    path = str(tmp_path / f"cross_test{ext}")
    creator(path)
    conn_name = f"cross_{db_type}"
    yield db_type, conn_name, path


@pytest.fixture(params=SQL_DB_TYPES)
def sql_db(request, tmp_path):
    """Yield (db_type, connection_name, file_path) for SQL-capable databases."""
    db_type = request.param
    ext, creator = _DB_CREATORS[db_type]
    path = str(tmp_path / f"cross_sql{ext}")
    creator(path)
    conn_name = f"cross_sql_{db_type}"
    yield db_type, conn_name, path


# ---------------------------------------------------------------------------
# Skip markers for Docker-only databases
# ---------------------------------------------------------------------------

skip_no_postgres = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL"),
    reason="PostgreSQL not available (set TEST_POSTGRES_URL)",
)
skip_no_mysql = pytest.mark.skipif(
    not os.environ.get("TEST_MYSQL_URL"),
    reason="MySQL not available (set TEST_MYSQL_URL)",
)
skip_no_mssql = pytest.mark.skipif(
    not os.environ.get("TEST_MSSQL_URL"),
    reason="MSSQL not available (set TEST_MSSQL_URL)",
)


# ===================================================================
# 1. Connection lifecycle consistency (all databases)
# ===================================================================


class TestConnectionLifecycle:
    """Connect -> describe -> disconnect should work cleanly for every type."""

    def test_connect_describe_disconnect(self, any_db):
        db_type, conn_name, path = any_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            result = call_tool("describe_database", {"name": conn_name})
            result_str = str(result)
            # Should return something meaningful, not an error
            assert (
                "error" not in result_str.lower() or "error_log" in result_str.lower()
            ), f"describe_database failed for {db_type}: {result_str[:200]}"
        finally:
            call_tool("disconnect_database", {"name": conn_name})


# ===================================================================
# 2. Describe database consistency (all databases)
# ===================================================================


class TestDescribeDatabase:
    """describe_database should return a non-error result for all types."""

    def test_describe_returns_content(self, any_db):
        db_type, conn_name, path = any_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            result = call_tool("describe_database", {"name": conn_name})
            # Must return something non-empty
            assert result is not None
            result_str = str(result)
            assert len(result_str) > 2, (
                f"describe_database returned empty for {db_type}"
            )
        finally:
            call_tool("disconnect_database", {"name": conn_name})


# ===================================================================
# 3. Error classification consistency (SQL databases)
# ===================================================================


class TestErrorClassification:
    """Query errors should return dicts with 'error' key, not crash."""

    def test_query_nonexistent_table(self, sql_db):
        db_type, conn_name, path = sql_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": conn_name, "query": "SELECT * FROM nonexistent_table_xyz"},
            )
            result_str = str(result).lower()
            assert "error" in result_str, (
                f"Expected error for nonexistent table on {db_type}, got: {result_str[:200]}"
            )
        finally:
            call_tool("disconnect_database", {"name": conn_name})

    def test_malformed_sql(self, sql_db):
        db_type, conn_name, path = sql_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": conn_name, "query": "SELECTTTT BADSQL FROM"},
            )
            result_str = str(result).lower()
            assert "error" in result_str, (
                f"Expected error for malformed SQL on {db_type}, got: {result_str[:200]}"
            )
        finally:
            call_tool("disconnect_database", {"name": conn_name})


# ===================================================================
# 4. Schema export consistency (SQL databases)
# ===================================================================


class TestSchemaExport:
    """export_schema should return valid output in each format."""

    def test_export_json_schema(self, sql_db):
        db_type, conn_name, path = sql_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            result = call_tool(
                "export_schema",
                {"name": conn_name, "tables": "data_table", "format": "json_schema"},
            )
            result_str = str(result)
            # Should contain the table name and some schema structure
            assert "data_table" in result_str, (
                f"json_schema missing table name for {db_type}: {result_str[:200]}"
            )
        finally:
            call_tool("disconnect_database", {"name": conn_name})

    def test_export_typescript(self, sql_db):
        db_type, conn_name, path = sql_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            result = call_tool(
                "export_schema",
                {"name": conn_name, "tables": "data_table", "format": "typescript"},
            )
            result_str = str(result)
            # TypeScript export should contain 'interface' keyword
            assert (
                "interface" in result_str.lower() or "data_table" in result_str.lower()
            ), f"typescript export unexpected for {db_type}: {result_str[:200]}"
        finally:
            call_tool("disconnect_database", {"name": conn_name})

    def test_export_python_dataclass(self, sql_db):
        db_type, conn_name, path = sql_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            result = call_tool(
                "export_schema",
                {
                    "name": conn_name,
                    "tables": "data_table",
                    "format": "python_dataclass",
                },
            )
            result_str = str(result)
            # Python dataclass export should contain 'class' or 'dataclass'
            assert (
                "class" in result_str.lower() or "data_table" in result_str.lower()
            ), f"python_dataclass export unexpected for {db_type}: {result_str[:200]}"
        finally:
            call_tool("disconnect_database", {"name": conn_name})


# ===================================================================
# 5. Basic query consistency (SQL databases)
# ===================================================================


class TestBasicQuery:
    """SQL databases should return consistent results for simple queries."""

    def test_select_all(self, sql_db):
        db_type, conn_name, path = sql_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": conn_name, "query": "SELECT * FROM data_table"},
            )
            result_str = str(result)
            assert "Alice" in result_str, (
                f"Missing expected data for {db_type}: {result_str[:200]}"
            )
            assert "Bob" in result_str
            assert "Charlie" in result_str
        finally:
            call_tool("disconnect_database", {"name": conn_name})

    def test_count(self, sql_db):
        db_type, conn_name, path = sql_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": conn_name, "query": "SELECT COUNT(*) as cnt FROM data_table"},
            )
            assert "3" in str(result), (
                f"Expected count 3 for {db_type}, got: {str(result)[:200]}"
            )
        finally:
            call_tool("disconnect_database", {"name": conn_name})


# ===================================================================
# 6. Tree operation consistency (JSON, YAML)
# ===================================================================


class TestTreeOperations:
    """Tree databases should support get_children and list_keys."""

    @pytest.fixture(params=TREE_DB_TYPES)
    def tree_db(self, request, tmp_path):
        db_type = request.param
        ext, creator = _DB_CREATORS[db_type]
        path = str(tmp_path / f"cross_tree{ext}")
        creator(path)
        conn_name = f"cross_tree_{db_type}"
        yield db_type, conn_name, path

    def test_get_children_at_root(self, tree_db):
        db_type, conn_name, path = tree_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            # Root level contains a single 'root' node
            result = call_tool("get_children", {"name": conn_name})
            result_str = str(result)
            assert "root" in result_str, (
                f"get_children missing 'root' for {db_type}: {result_str[:200]}"
            )
        finally:
            call_tool("disconnect_database", {"name": conn_name})

    def test_list_keys_at_root_node(self, tree_db):
        db_type, conn_name, path = tree_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            # list_keys requires a path; use 'root' to see top-level keys
            result = call_tool("list_keys", {"name": conn_name, "path": "root"})
            result_str = str(result)
            assert "key" in result_str, (
                f"list_keys missing 'key' for {db_type}: {result_str[:200]}"
            )
        finally:
            call_tool("disconnect_database", {"name": conn_name})

    def test_get_value(self, tree_db):
        db_type, conn_name, path = tree_db
        call_tool(
            "connect_database",
            {"name": conn_name, "db_type": db_type, "conn_string": path},
        )
        try:
            result = call_tool(
                "get_value",
                {"name": conn_name, "path": "root", "key": "key"},
            )
            assert "value" in str(result), (
                f"get_value unexpected for {db_type}: {str(result)[:200]}"
            )
        finally:
            call_tool("disconnect_database", {"name": conn_name})
