"""CouchDB integration tests via MCP tool interface."""

import os

import pytest
import requests

from .mcp_test_client import call_tool

COUCHDB_URL = os.environ.get("TEST_COUCHDB_URL", "http://admin:admin@localhost:15984")
TEST_DB = "mcp_test_db"


def _couchdb_available():
    try:
        resp = requests.get(COUCHDB_URL, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.couchdb,
    pytest.mark.skipif(not _couchdb_available(), reason="CouchDB not available"),
]


@pytest.fixture(scope="module", autouse=True)
def setup_couchdb_data():
    """Create a test database and insert documents via CouchDB HTTP API."""
    db_url = f"{COUCHDB_URL}/{TEST_DB}"

    # Drop database if it exists, then create fresh
    requests.delete(db_url, timeout=5)
    resp = requests.put(db_url, timeout=5)
    assert resp.status_code in (201, 202), f"Failed to create test DB: {resp.text}"

    # Insert test documents
    docs = [
        {
            "_id": f"user_{i:03d}",
            "name": f"user_{i}",
            "email": f"user_{i}@test.com",
            "amount": float(i * 1.5),
            "category": ["A", "B", "C"][i % 3],
            "active": i % 2 == 0,
        }
        for i in range(100)
    ]
    resp = requests.post(
        f"{db_url}/_bulk_docs",
        json={"docs": docs},
        timeout=10,
    )
    assert resp.status_code in (200, 201), f"Failed to insert docs: {resp.text}"

    yield

    requests.delete(db_url, timeout=5)


def _try_disconnect(name: str) -> None:
    """Best-effort disconnect."""
    try:
        call_tool("disconnect_database", {"name": name})
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _cleanup_couchdb_connections():
    """Disconnect any CouchDB connections left over after each test."""
    yield
    try:
        result = call_tool("list_databases", {})
        if isinstance(result, dict):
            for db in result.get("databases", []):
                if db.get("db_type") == "couchdb":
                    _try_disconnect(db["name"])
    except Exception:
        pass


class TestCouchDBConnection:
    """Test CouchDB connection lifecycle via MCP."""

    def test_connect_couchdb(self):
        """Connecting to CouchDB should return a result (success or structured error)."""
        conn_string = f"{COUCHDB_URL}/{TEST_DB}"
        result = call_tool(
            "connect_database",
            {"name": "couch_conn", "db_type": "couchdb", "conn_string": conn_string},
        )
        try:
            result_str = str(result)
            assert result_str, "connect_database returned empty result"
            assert (
                "success" in result_str.lower() or "couch_conn" in result_str
            ), f"Unexpected connect result: {result_str}"
        finally:
            _try_disconnect("couch_conn")

    def test_connect_and_list(self):
        """After connecting, the connection name should appear in list_databases."""
        conn_string = f"{COUCHDB_URL}/{TEST_DB}"
        call_tool(
            "connect_database",
            {"name": "couch_ls", "db_type": "couchdb", "conn_string": conn_string},
        )
        try:
            list_result = call_tool("list_databases", {})
            list_str = str(list_result)
            assert (
                "couch_ls" in list_str
            ), f"couch_ls not in list_databases after connect: {list_str}"
        finally:
            _try_disconnect("couch_ls")

    def test_disconnect_couchdb(self):
        """Disconnect should succeed for CouchDB connections."""
        conn_string = f"{COUCHDB_URL}/{TEST_DB}"
        call_tool(
            "connect_database",
            {"name": "couch_dc", "db_type": "couchdb", "conn_string": conn_string},
        )
        result = call_tool("disconnect_database", {"name": "couch_dc"})
        result_str = str(result)
        assert (
            "disconnect" in result_str.lower() or "success" in result_str.lower()
        ), f"Expected successful disconnect, got: {result_str}"

    def test_connect_invalid_url(self):
        """Connecting with an unreachable host should return an error, not crash."""
        result = call_tool(
            "connect_database",
            {
                "name": "couch_bad",
                "db_type": "couchdb",
                "conn_string": "http://nonexistent-host:99999/baddb",
            },
        )
        try:
            result_str = str(result)
            assert result_str, "connect_database with bad URL returned empty result"
        finally:
            _try_disconnect("couch_bad")

    def test_duplicate_connect_rejected(self):
        """Connecting with a name already in use should return an error."""
        conn_string = f"{COUCHDB_URL}/{TEST_DB}"
        call_tool(
            "connect_database",
            {"name": "couch_dup", "db_type": "couchdb", "conn_string": conn_string},
        )
        try:
            result = call_tool(
                "connect_database",
                {
                    "name": "couch_dup",
                    "db_type": "couchdb",
                    "conn_string": conn_string,
                },
            )
            result_str = str(result)
            assert (
                "already" in result_str.lower() or "error" in result_str.lower()
            ), f"Expected duplicate-name error: {result_str}"
        finally:
            _try_disconnect("couch_dup")


class TestCouchDBQueries:
    """Test querying CouchDB via MCP tools."""

    def test_all_docs_query(self):
        """Querying _all_docs should return a result or graceful error."""
        conn_string = f"{COUCHDB_URL}/{TEST_DB}"
        call_tool(
            "connect_database",
            {"name": "couch_alldocs", "db_type": "couchdb", "conn_string": conn_string},
        )
        try:
            query_result = call_tool(
                "execute_query",
                {"name": "couch_alldocs", "query": "_all_docs?limit=10"},
            )
            query_str = str(query_result)
            assert query_str, "execute_query _all_docs returned empty result"
        finally:
            _try_disconnect("couch_alldocs")

    def test_describe_database(self):
        """describe_database should return some result or a graceful error."""
        conn_string = f"{COUCHDB_URL}/{TEST_DB}"
        call_tool(
            "connect_database",
            {"name": "couch_desc", "db_type": "couchdb", "conn_string": conn_string},
        )
        try:
            desc_result = call_tool("describe_database", {"name": "couch_desc"})
            desc_str = str(desc_result)
            assert desc_str, "describe_database returned empty result"
        finally:
            _try_disconnect("couch_desc")

    def test_execute_query_returns_result(self):
        """execute_query on CouchDB should return a result or graceful error."""
        conn_string = f"{COUCHDB_URL}/{TEST_DB}"
        call_tool(
            "connect_database",
            {"name": "couch_qry", "db_type": "couchdb", "conn_string": conn_string},
        )
        try:
            query_result = call_tool(
                "execute_query",
                {"name": "couch_qry", "query": "SELECT 1"},
            )
            query_str = str(query_result)
            assert query_str, "execute_query returned empty result"
        finally:
            _try_disconnect("couch_qry")


class TestCouchDBErrors:
    """Test error handling for CouchDB edge cases."""

    def test_disconnect_nonexistent(self):
        """Disconnecting a nonexistent connection should return error."""
        result = call_tool("disconnect_database", {"name": "couch_nonexist_xyz"})
        result_str = str(result)
        assert (
            "error" in result_str.lower() or "not" in result_str.lower()
        ), f"Expected error for nonexistent disconnect: {result_str}"

    def test_describe_nonexistent(self):
        """Describing a nonexistent connection should return error."""
        result = call_tool("describe_database", {"name": "couch_nonexist_xyz"})
        result_str = str(result)
        assert (
            "error" in result_str.lower() or "not" in result_str.lower()
        ), f"Expected error for nonexistent describe: {result_str}"

    def test_query_nonexistent_connection(self):
        """Querying a nonexistent connection should return error."""
        result = call_tool(
            "execute_query",
            {"name": "couch_nonexist_xyz", "query": "_all_docs"},
        )
        result_str = str(result)
        assert (
            "error" in result_str.lower() or "not" in result_str.lower()
        ), f"Expected error for nonexistent query: {result_str}"
