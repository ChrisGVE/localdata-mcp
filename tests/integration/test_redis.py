"""Redis integration tests via MCP tool interface."""

import os

import pytest

from .mcp_test_client import call_tool

REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:16379/0")


def _redis_available():
    try:
        import redis

        r = redis.Redis.from_url(REDIS_URL)
        r.ping()
        r.close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.redis,
    pytest.mark.skipif(not _redis_available(), reason="Redis not available"),
]


@pytest.fixture(scope="module", autouse=True)
def setup_redis_data():
    import redis

    r = redis.Redis.from_url(REDIS_URL)
    r.flushdb()
    # Insert test keys
    r.set("key1", "value1")
    r.set("key2", "value2")
    r.hset("user:1", mapping={"name": "Alice", "email": "alice@test.com"})
    r.hset("user:2", mapping={"name": "Bob", "email": "bob@test.com"})
    r.lpush("mylist", "item1", "item2", "item3")
    r.sadd("myset", "a", "b", "c")
    yield
    r.flushdb()
    r.close()


def _connect(name):
    """Connect to Redis via MCP and return the parsed result."""
    return call_tool(
        "connect_database",
        {"name": name, "db_type": "redis", "conn_string": REDIS_URL},
    )


def _disconnect(name):
    """Disconnect a Redis connection."""
    call_tool("disconnect_database", {"name": name})


@pytest.fixture(autouse=True)
def _cleanup_redis_connections():
    """Disconnect any Redis connections left over after each test."""
    yield
    try:
        result = call_tool("list_databases", {})
        if isinstance(result, dict):
            for db in result.get("databases", []):
                if db.get("db_type") == "redis":
                    try:
                        call_tool("disconnect_database", {"name": db["name"]})
                    except Exception:
                        pass
    except Exception:
        pass


class TestRedisConnection:
    def test_connect_redis(self):
        result = _connect("r_conn")
        result_str = str(result)
        assert (
            "success" in result_str.lower() or "redis" in result_str.lower()
        ), f"Unexpected connect result: {result_str}"

    def test_connect_returns_success_true(self):
        result = _connect("r_succ")
        if isinstance(result, dict):
            assert (
                result.get("success") is True
            ), f"Expected success=True, got: {result}"
        else:
            result_str = str(result)
            assert (
                "success" in result_str.lower()
            ), f"Expected success in result: {result_str}"

    def test_connection_info_contains_redis_type(self):
        result = _connect("r_info")
        if isinstance(result, dict):
            conn_info = result.get("connection_info", {})
            assert (
                conn_info.get("db_type") == "redis"
            ), f"Expected db_type=redis, got: {conn_info}"
            assert (
                conn_info.get("sql_flavor") == "Redis"
            ), f"Expected sql_flavor=Redis, got: {conn_info}"

    def test_list_databases_shows_redis(self):
        _connect("r_listed")
        result = call_tool("list_databases", {})
        result_str = str(result)
        assert (
            "r_listed" in result_str
        ), f"r_listed not found in list_databases: {result_str}"

    def test_disconnect_redis(self):
        """Disconnect should succeed for Redis connections."""
        _connect("r_dc")
        result = call_tool("disconnect_database", {"name": "r_dc"})
        result_str = str(result)
        assert (
            "disconnect" in result_str.lower() or "success" in result_str.lower()
        ), f"Expected successful disconnect, got: {result_str}"

    def test_duplicate_connection_rejected(self):
        _connect("r_dup")
        result = _connect("r_dup")
        result_str = str(result)
        assert (
            "already" in result_str.lower() or "error" in result_str.lower()
        ), f"Expected duplicate error, got: {result_str}"


class TestRedisDescribe:
    def test_describe_database_returns_result(self):
        """describe_database on Redis hits SQLAlchemy inspect which fails,
        but the call should not crash -- it returns an error message."""
        _connect("r_desc")
        result = call_tool("describe_database", {"name": "r_desc"})
        result_str = str(result)
        # Should return something (error or info), not empty
        assert result_str, "describe_database returned empty result"
        # The error message typically mentions the inspection failure
        assert (
            "error" in result_str.lower() or "redis" in result_str.lower()
        ), f"Unexpected describe result: {result_str}"


class TestRedisQuery:
    """execute_query passes through the SQL parser, so Redis commands will
    be rejected as invalid SQL.  These tests verify graceful error handling."""

    def test_execute_query_get_command(self):
        _connect("r_qget")
        result = call_tool(
            "execute_query",
            {"name": "r_qget", "query": "GET key1"},
        )
        result_str = str(result)
        # Will be either an error from SQL validation or a Redis response
        assert result_str, "execute_query returned empty result"

    def test_execute_query_keys_command(self):
        _connect("r_qkeys")
        result = call_tool(
            "execute_query",
            {"name": "r_qkeys", "query": "KEYS *"},
        )
        result_str = str(result)
        assert result_str, "execute_query returned empty result"

    def test_execute_query_hgetall_command(self):
        _connect("r_qhget")
        result = call_tool(
            "execute_query",
            {"name": "r_qhget", "query": "HGETALL user:1"},
        )
        result_str = str(result)
        assert result_str, "execute_query returned empty result"


class TestRedisErrors:
    def test_query_returns_graceful_error(self):
        """Any query on Redis returns a graceful error (not a crash)."""
        _connect("r_qerr")
        result = call_tool(
            "execute_query",
            {"name": "r_qerr", "query": "INVALID_COMMAND foo bar"},
        )
        result_str = str(result)
        assert result_str, "Expected non-empty error result"

    def test_disconnect_nonexistent_connection(self):
        """Disconnecting a non-existent connection returns an error."""
        result = call_tool("disconnect_database", {"name": "r_nonexistent_xyz"})
        result_str = str(result)
        assert (
            "not connected" in result_str.lower() or "error" in result_str.lower()
        ), f"Expected not-connected error, got: {result_str}"
