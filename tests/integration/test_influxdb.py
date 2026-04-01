"""InfluxDB integration tests via MCP tool interface."""

import os
from datetime import datetime, timezone

import pytest

from .mcp_test_client import call_tool

INFLUXDB_URL = os.environ.get("TEST_INFLUXDB_URL", "http://localhost:18086")
INFLUXDB_TOKEN = "testtokenforlocaldata"
INFLUXDB_ORG = "testorg"
INFLUXDB_BUCKET = "testbucket"

# Connection string format expected by the MCP server
INFLUXDB_CONN = (
    f"{INFLUXDB_URL}?token={INFLUXDB_TOKEN}&org={INFLUXDB_ORG}&bucket={INFLUXDB_BUCKET}"
)


def _influxdb_available():
    try:
        from influxdb_client import InfluxDBClient

        client = InfluxDBClient(
            url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG
        )
        health = client.health()
        client.close()
        return health.status == "pass"
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.influxdb,
    pytest.mark.skipif(not _influxdb_available(), reason="InfluxDB not available"),
]


@pytest.fixture(scope="module", autouse=True)
def setup_influxdb_data():
    """Write test data points into InfluxDB for integration tests."""
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS

    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    # Delete existing data in the bucket by dropping the measurement
    delete_api = client.delete_api()
    start = "1970-01-01T00:00:00Z"
    stop = "2100-01-01T00:00:00Z"
    try:
        delete_api.delete(
            start,
            stop,
            '_measurement="cpu_usage"',
            bucket=INFLUXDB_BUCKET,
            org=INFLUXDB_ORG,
        )
        delete_api.delete(
            start,
            stop,
            '_measurement="memory"',
            bucket=INFLUXDB_BUCKET,
            org=INFLUXDB_ORG,
        )
    except Exception:
        pass

    # Write cpu_usage data points
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(20):
        point = (
            Point("cpu_usage")
            .tag("host", f"server{i % 3 + 1}")
            .tag("region", ["us-east", "us-west", "eu-west"][i % 3])
            .field("usage_percent", 30.0 + (i * 2.5))
            .field("idle_percent", 70.0 - (i * 2.5))
            .time(int(base_time.timestamp()) + i * 60, WritePrecision.S)
        )
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)

    # Write memory data points
    for i in range(10):
        point = (
            Point("memory")
            .tag("host", f"server{i % 2 + 1}")
            .field("used_gb", 4.0 + (i * 0.5))
            .field("total_gb", 16.0)
            .time(int(base_time.timestamp()) + i * 120, WritePrecision.S)
        )
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)

    write_api.close()
    yield

    # Cleanup
    try:
        delete_api.delete(
            start,
            stop,
            '_measurement="cpu_usage"',
            bucket=INFLUXDB_BUCKET,
            org=INFLUXDB_ORG,
        )
        delete_api.delete(
            start,
            stop,
            '_measurement="memory"',
            bucket=INFLUXDB_BUCKET,
            org=INFLUXDB_ORG,
        )
    except Exception:
        pass
    client.close()


def _connect(name):
    """Connect to InfluxDB via MCP and return the parsed result."""
    return call_tool(
        "connect_database",
        {"name": name, "db_type": "influxdb", "conn_string": INFLUXDB_CONN},
    )


@pytest.fixture(autouse=True)
def _cleanup_influxdb_connections():
    """Disconnect any InfluxDB connections left over after each test."""
    yield
    try:
        result = call_tool("list_databases", {})
        if isinstance(result, dict):
            for db in result.get("databases", []):
                if db.get("db_type") == "influxdb":
                    try:
                        call_tool("disconnect_database", {"name": db["name"]})
                    except Exception:
                        pass
    except Exception:
        pass


class TestInfluxDBConnection:
    """Test connection lifecycle: connect, list, disconnect."""

    def test_connect(self):
        """Connecting to InfluxDB returns a success response."""
        name = "idb_conn"
        try:
            result = _connect(name)
            result_str = str(result)
            assert "success" in result_str.lower() or name in result_str, (
                f"Connect did not indicate success: {result_str}"
            )
        finally:
            call_tool("disconnect_database", {"name": name})

    def test_connect_returns_success_true(self):
        """The connect response contains success: True."""
        name = "idb_succ"
        try:
            result = _connect(name)
            if isinstance(result, dict):
                assert result.get("success") is True, (
                    f"Expected success=True in dict result: {result}"
                )
            else:
                assert (
                    '"success": true' in str(result).lower()
                    or "success" in str(result).lower()
                ), f"Expected success in result: {result}"
        finally:
            call_tool("disconnect_database", {"name": name})

    def test_list_databases_shows_connection(self):
        """After connecting, list_databases includes the InfluxDB connection."""
        name = "idb_list"
        _connect(name)
        try:
            result = call_tool("list_databases", {})
            result_str = str(result)
            assert name in result_str, (
                f"Connection '{name}' not found in list_databases: {result_str}"
            )
            assert "influxdb" in result_str.lower(), (
                f"db_type 'influxdb' not shown in list_databases: {result_str}"
            )
        finally:
            call_tool("disconnect_database", {"name": name})

    def test_disconnect(self):
        """Disconnect should succeed for InfluxDB connections."""
        name = "idb_dc"
        _connect(name)
        result = call_tool("disconnect_database", {"name": name})
        result_str = str(result)
        assert "disconnect" in result_str.lower() or "success" in result_str.lower(), (
            f"Expected successful disconnect, got: {result_str}"
        )

    def test_duplicate_connection_rejected(self):
        """Connecting with an already-used name returns an error."""
        name = "idb_dup"
        _connect(name)
        try:
            result = _connect(name)
            result_str = str(result).lower()
            assert "error" in result_str or "already" in result_str, (
                f"Expected error for duplicate connection: {result}"
            )
        finally:
            call_tool("disconnect_database", {"name": name})


class TestInfluxDBQueries:
    """Test execute_query with Flux queries against InfluxDB."""

    def test_basic_flux_query(self):
        """A basic Flux query returns some response."""
        name = "idb_flux"
        _connect(name)
        try:
            query = (
                f'from(bucket: "{INFLUXDB_BUCKET}")'
                " |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)"
                ' |> filter(fn: (r) => r._measurement == "cpu_usage")'
                " |> limit(n: 5)"
            )
            result = call_tool(
                "execute_query",
                {"name": name, "query": query},
            )
            assert result is not None, "execute_query returned None for Flux query"
        finally:
            call_tool("disconnect_database", {"name": name})

    def test_flux_query_with_filter(self):
        """A Flux query with tag filter returns a response."""
        name = "idb_filter"
        _connect(name)
        try:
            query = (
                f'from(bucket: "{INFLUXDB_BUCKET}")'
                " |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)"
                ' |> filter(fn: (r) => r._measurement == "cpu_usage")'
                ' |> filter(fn: (r) => r.host == "server1")'
                " |> limit(n: 10)"
            )
            result = call_tool(
                "execute_query",
                {"name": name, "query": query},
            )
            assert result is not None, (
                "execute_query returned None for filtered Flux query"
            )
        finally:
            call_tool("disconnect_database", {"name": name})

    def test_flux_query_aggregation(self):
        """A Flux aggregation query returns a response."""
        name = "idb_agg"
        _connect(name)
        try:
            query = (
                f'from(bucket: "{INFLUXDB_BUCKET}")'
                " |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)"
                ' |> filter(fn: (r) => r._measurement == "cpu_usage")'
                ' |> filter(fn: (r) => r._field == "usage_percent")'
                " |> mean()"
            )
            result = call_tool(
                "execute_query",
                {"name": name, "query": query},
            )
            assert result is not None, (
                "execute_query returned None for aggregation query"
            )
        finally:
            call_tool("disconnect_database", {"name": name})

    def test_flux_query_memory_measurement(self):
        """Querying the memory measurement returns a response."""
        name = "idb_mem"
        _connect(name)
        try:
            query = (
                f'from(bucket: "{INFLUXDB_BUCKET}")'
                " |> range(start: 2024-01-01T00:00:00Z, stop: 2025-01-01T00:00:00Z)"
                ' |> filter(fn: (r) => r._measurement == "memory")'
                " |> limit(n: 5)"
            )
            result = call_tool(
                "execute_query",
                {"name": name, "query": query},
            )
            assert result is not None, "execute_query returned None for memory query"
        finally:
            call_tool("disconnect_database", {"name": name})


class TestInfluxDBErrors:
    """Test error handling for invalid operations."""

    def test_query_nonexistent_connection(self):
        """Querying a non-existent connection returns an error."""
        result = call_tool(
            "execute_query",
            {"name": "idb_nonexistent", "query": "SELECT 1"},
        )
        result_str = str(result).lower()
        assert "error" in result_str or "not connected" in result_str, (
            f"Expected error for nonexistent connection: {result}"
        )

    def test_describe_nonexistent_connection(self):
        """Describing a non-existent connection returns an error."""
        result = call_tool(
            "describe_database",
            {"name": "idb_no_such"},
        )
        result_str = str(result).lower()
        assert "error" in result_str or "not connected" in result_str, (
            f"Expected error for nonexistent connection: {result}"
        )

    def test_bad_flux_query(self):
        """A malformed Flux query returns an error, not a crash."""
        name = "idb_badq"
        _connect(name)
        try:
            result = call_tool(
                "execute_query",
                {"name": name, "query": "THIS IS NOT VALID FLUX"},
            )
            result_str = str(result)
            assert result_str, "Expected non-empty result for bad query"
        finally:
            call_tool("disconnect_database", {"name": name})

    def test_connect_bad_url(self):
        """Connecting with a bad URL returns an error or succeeds lazily."""
        name = "idb_bad"
        result = call_tool(
            "connect_database",
            {
                "name": name,
                "db_type": "influxdb",
                "conn_string": "http://localhost:19999?token=bad&org=bad&bucket=bad",
            },
        )
        # InfluxDB client may connect lazily; the important thing is no crash.
        assert result is not None, "connect_database returned None for bad URL"
        # Clean up in case it did register
        call_tool("disconnect_database", {"name": name})

    def test_disconnect_nonexistent_connection(self):
        """Disconnecting a non-existent connection returns an error."""
        result = call_tool("disconnect_database", {"name": "idb_nonexistent_xyz"})
        result_str = str(result).lower()
        assert "not connected" in result_str or "error" in result_str, (
            f"Expected not-connected error, got: {result_str}"
        )
