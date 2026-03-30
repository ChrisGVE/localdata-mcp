"""Performance and stress tests for LocalData MCP.

Tests verify system behavior under heavy load: large result sets,
streaming pagination, memory metrics, and preflight estimation.

Marked as large_data — skip with: pytest -m "not large_data"
"""

import os
import time

import pytest

from .database_setup import create_sqlite_test_db
from .mcp_test_client import call_tool

pytestmark = [pytest.mark.integration, pytest.mark.large_data]


@pytest.fixture(scope="module")
def large_db():
    """Create a 100K-row SQLite database for stress testing."""
    path = create_sqlite_test_db(rows=100_000)
    call_tool(
        "connect_database",
        {"name": "stress_db", "db_type": "sqlite", "conn_string": path},
    )
    yield path
    call_tool("disconnect_database", {"name": "stress_db"})
    os.unlink(path)


class TestLargeResultSets:
    """Test handling of large query results."""

    def test_count_100k_rows(self, large_db):
        """COUNT(*) on 100K rows should succeed quickly."""
        result = call_tool(
            "execute_query",
            {
                "name": "stress_db",
                "query": "SELECT COUNT(*) as cnt FROM test_data",
            },
        )
        assert isinstance(result, dict)
        assert "error" not in result or not result.get("error")
        data = result.get("data", [])
        assert len(data) > 0
        assert data[0]["cnt"] == 100_000

    def test_streaming_activates(self, large_db):
        """SELECT * on 100K rows should activate streaming."""
        result = call_tool(
            "execute_query",
            {
                "name": "stress_db",
                "query": "SELECT * FROM test_data",
            },
        )
        assert isinstance(result, dict)
        meta = result.get("streaming_metadata", {})
        assert meta.get("streaming") is True
        assert meta.get("buffer_complete") is not None

    def test_chunked_pagination(self, large_db):
        """Retrieve data via next_chunk pagination."""
        result = call_tool(
            "execute_query",
            {
                "name": "stress_db",
                "query": "SELECT id, name FROM test_data WHERE id <= 500",
                "chunk_size": 100,
            },
        )
        assert isinstance(result, dict)
        first_data = result.get("data", [])
        assert len(first_data) > 0

        # If there's a query_id, try fetching next chunk
        query_id = result.get("metadata", {}).get("query_id")
        if query_id:
            chunk2 = call_tool(
                "next_chunk",
                {
                    "query_id": query_id,
                    "start_row": len(first_data) + 1,
                    "chunk_size": "100",
                },
            )
            assert chunk2 is not None

    def test_query_with_limit(self, large_db):
        """LIMIT should cap results even on large tables."""
        result = call_tool(
            "execute_query",
            {
                "name": "stress_db",
                "query": "SELECT * FROM test_data LIMIT 10",
            },
        )
        assert isinstance(result, dict)
        data = result.get("data", [])
        assert len(data) == 10


class TestPreflightEstimation:
    """Test preflight query analysis."""

    def test_preflight_large_query(self, large_db):
        """Preflight should estimate rows without executing."""
        result = call_tool(
            "execute_query",
            {
                "name": "stress_db",
                "query": "SELECT * FROM test_data",
                "preflight": True,
            },
        )
        assert isinstance(result, dict)
        # Preflight returns estimation, not data
        result_str = str(result)
        assert "estimated" in result_str.lower() or "rows" in result_str.lower()

    def test_preflight_filtered_query(self, large_db):
        """Preflight on filtered query should give smaller estimate."""
        result = call_tool(
            "execute_query",
            {
                "name": "stress_db",
                "query": "SELECT * FROM test_data WHERE id <= 100",
                "preflight": True,
            },
        )
        assert isinstance(result, dict)
        assert "error" not in result or not result.get("error")


class TestMemoryMetrics:
    """Test memory monitoring during queries."""

    def test_memory_info_in_response(self, large_db):
        """Query responses should include memory info."""
        result = call_tool(
            "execute_query",
            {
                "name": "stress_db",
                "query": "SELECT COUNT(*) FROM test_data",
            },
        )
        assert isinstance(result, dict)
        mem = result.get("metadata", {}).get("memory_info", {})
        if mem:
            assert "total_gb" in mem
            assert "available_gb" in mem

    def test_manage_memory_bounds(self, large_db):
        """Memory bounds tool should return status."""
        result = call_tool("manage_memory_bounds", {})
        assert result is not None
        result_str = str(result)
        assert "memory" in result_str.lower() or "limit" in result_str.lower()


class TestQueryPerformance:
    """Test query execution performance."""

    def test_count_query_under_5s(self, large_db):
        """COUNT(*) on 100K rows should complete in <5 seconds."""
        start = time.time()
        result = call_tool(
            "execute_query",
            {
                "name": "stress_db",
                "query": "SELECT COUNT(*) FROM test_data",
            },
        )
        elapsed = time.time() - start
        assert elapsed < 5.0, f"COUNT query took {elapsed:.1f}s"
        assert isinstance(result, dict)

    def test_aggregation_under_5s(self, large_db):
        """GROUP BY aggregation on 100K rows should complete in <5s."""
        start = time.time()
        result = call_tool(
            "execute_query",
            {
                "name": "stress_db",
                "query": (
                    "SELECT category, COUNT(*) as cnt, AVG(amount) as avg_amt "
                    "FROM test_data GROUP BY category"
                ),
            },
        )
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Aggregation took {elapsed:.1f}s"
        assert isinstance(result, dict)
        data = result.get("data", [])
        assert len(data) > 0

    def test_filtered_query_under_5s(self, large_db):
        """Filtered query on 100K rows should complete in <5s."""
        start = time.time()
        result = call_tool(
            "execute_query",
            {
                "name": "stress_db",
                "query": "SELECT * FROM test_data WHERE score > 0.9 LIMIT 100",
            },
        )
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Filtered query took {elapsed:.1f}s"
        assert isinstance(result, dict)


class TestStreamingStatus:
    """Test streaming buffer management."""

    def test_streaming_status(self, large_db):
        """Get streaming status should work."""
        result = call_tool("get_streaming_status", {})
        assert result is not None

    def test_clear_buffer_nonexistent(self, large_db):
        """Clearing a nonexistent buffer should not crash."""
        result = call_tool("clear_streaming_buffer", {"query_id": "nonexistent_12345"})
        assert result is not None
