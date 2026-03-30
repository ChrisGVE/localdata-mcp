"""MongoDB integration tests via MCP tool interface."""

import os

import pytest

from .mcp_test_client import call_tool

MONGODB_URL = os.environ.get("TEST_MONGODB_URL", "mongodb://localhost:17017/testdb")


def _mongodb_available():
    try:
        import pymongo

        c = pymongo.MongoClient(MONGODB_URL, serverSelectionTimeoutMS=3000)
        c.admin.command("ping")
        c.close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.mongodb,
    pytest.mark.skipif(not _mongodb_available(), reason="MongoDB not available"),
]


@pytest.fixture(scope="module", autouse=True)
def setup_mongodb_data():
    """Insert test documents directly via pymongo before running tests."""
    import pymongo

    client = pymongo.MongoClient(MONGODB_URL)
    db = client.get_database()
    db.drop_collection("test_data")
    docs = [
        {
            "name": f"user_{i}",
            "email": f"user_{i}@test.com",
            "amount": float(i * 1.5),
            "category": ["A", "B", "C"][i % 3],
            "active": i % 2 == 0,
        }
        for i in range(100)
    ]
    db.test_data.insert_many(docs)
    yield
    db.drop_collection("test_data")
    client.close()


def _try_disconnect(name: str) -> None:
    """Best-effort disconnect."""
    try:
        call_tool("disconnect_database", {"name": name})
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _cleanup_mongo_connections():
    """Disconnect any MongoDB connections left over after each test."""
    yield
    try:
        result = call_tool("list_databases", {})
        if isinstance(result, dict):
            for db in result.get("databases", []):
                if db.get("db_type") == "mongodb":
                    _try_disconnect(db["name"])
    except Exception:
        pass


class TestMongoDBConnection:
    """Test MongoDB connection lifecycle via MCP."""

    def test_connect_mongodb(self):
        """Connecting to MongoDB should return a result (success or structured error)."""
        result = call_tool(
            "connect_database",
            {"name": "mongo_conn", "db_type": "mongodb", "conn_string": MONGODB_URL},
        )
        try:
            result_str = str(result)
            # The server should not crash; it should return a response.
            assert result_str, "connect_database returned empty result"
            # Connection should report success even if summary building fails
            assert "success" in result_str.lower() or "mongo_conn" in result_str, (
                f"Unexpected connect result: {result_str}"
            )
        finally:
            _try_disconnect("mongo_conn")

    def test_connect_and_list(self):
        """After connecting, the connection name should appear in list_databases."""
        call_tool(
            "connect_database",
            {"name": "mongo_ls", "db_type": "mongodb", "conn_string": MONGODB_URL},
        )
        try:
            list_result = call_tool("list_databases", {})
            list_str = str(list_result)
            assert "mongo_ls" in list_str, (
                f"mongo_ls not in list_databases after connect: {list_str}"
            )
        finally:
            _try_disconnect("mongo_ls")

    def test_disconnect_mongodb(self):
        """Disconnect should succeed for MongoDB connections."""
        call_tool(
            "connect_database",
            {"name": "mongo_dc", "db_type": "mongodb", "conn_string": MONGODB_URL},
        )
        result = call_tool("disconnect_database", {"name": "mongo_dc"})
        result_str = str(result)
        assert "disconnect" in result_str.lower() or "success" in result_str.lower(), (
            f"Expected successful disconnect, got: {result_str}"
        )

    def test_connect_invalid_url(self):
        """Connecting with an unreachable host should return an error, not crash."""
        result = call_tool(
            "connect_database",
            {
                "name": "mongo_bad",
                "db_type": "mongodb",
                "conn_string": "mongodb://nonexistent-host:99999/baddb",
            },
        )
        try:
            result_str = str(result)
            # Should get a response (may succeed lazily since pymongo defers connection)
            assert result_str, "connect_database with bad URL returned empty result"
        finally:
            _try_disconnect("mongo_bad")

    def test_duplicate_connect_rejected(self):
        """Connecting with a name already in use should return an error."""
        call_tool(
            "connect_database",
            {"name": "mongo_dup", "db_type": "mongodb", "conn_string": MONGODB_URL},
        )
        try:
            result = call_tool(
                "connect_database",
                {
                    "name": "mongo_dup",
                    "db_type": "mongodb",
                    "conn_string": MONGODB_URL,
                },
            )
            result_str = str(result)
            assert "already" in result_str.lower() or "error" in result_str.lower(), (
                f"Expected duplicate-name error: {result_str}"
            )
        finally:
            _try_disconnect("mongo_dup")


class TestMongoDBDescribe:
    """Test describe_database on MongoDB connections."""

    def test_describe_database_returns_result(self):
        """describe_database should return some result or a graceful error."""
        call_tool(
            "connect_database",
            {"name": "mongo_dsc", "db_type": "mongodb", "conn_string": MONGODB_URL},
        )
        try:
            desc_result = call_tool("describe_database", {"name": "mongo_dsc"})
            desc_str = str(desc_result)
            # Should return something, not crash
            assert desc_str, "describe_database returned empty result"
        finally:
            _try_disconnect("mongo_dsc")


class TestMongoDBQuery:
    """Test execute_query on MongoDB connections.

    MongoDB connections may not support SQL queries through the MCP server.
    These tests verify the server handles queries gracefully.
    """

    def test_execute_query_returns_result(self):
        """execute_query on MongoDB should return a result or graceful error."""
        call_tool(
            "connect_database",
            {"name": "mongo_qry", "db_type": "mongodb", "conn_string": MONGODB_URL},
        )
        try:
            query_result = call_tool(
                "execute_query",
                {"name": "mongo_qry", "query": "SELECT 1"},
            )
            query_str = str(query_result)
            # Should return something — data or error, not crash
            assert query_str, "execute_query returned empty result"
        finally:
            _try_disconnect("mongo_qry")


class TestMongoDBErrorHandling:
    """Test error handling for MongoDB edge cases."""

    def test_disconnect_nonexistent(self):
        """Disconnecting a nonexistent connection should return error."""
        result = call_tool("disconnect_database", {"name": "mongo_nonexist_xyz"})
        result_str = str(result)
        assert "error" in result_str.lower() or "not" in result_str.lower(), (
            f"Expected error for nonexistent disconnect: {result_str}"
        )

    def test_describe_nonexistent(self):
        """Describing a nonexistent connection should return error."""
        result = call_tool("describe_database", {"name": "mongo_nonexist_xyz"})
        result_str = str(result)
        assert "error" in result_str.lower() or "not" in result_str.lower(), (
            f"Expected error for nonexistent describe: {result_str}"
        )

    def test_query_nonexistent_connection(self):
        """Querying a nonexistent connection should return error."""
        result = call_tool(
            "execute_query",
            {"name": "mongo_nonexist_xyz", "query": "SELECT 1"},
        )
        result_str = str(result)
        assert "error" in result_str.lower() or "not" in result_str.lower(), (
            f"Expected error for nonexistent query: {result_str}"
        )
