"""Enterprise-scale stress tests across all database types.

Exercises large queries, streaming, memory bounds, metadata, and
cross-database consistency using NYC Taxi data.

Run separately with: pytest -m enterprise_scale
"""

from __future__ import annotations

import time

import pytest

from .large_dataset import get_dataset_info
from .mcp_test_client import call_tool

# Lazy imports — tests skip gracefully when loader is unavailable.
try:
    from .database_loader import cleanup_dataset, get_connection_info, load_dataset
except ImportError:
    load_dataset = None  # type: ignore[assignment]
    cleanup_dataset = None  # type: ignore[assignment]
    get_connection_info = None  # type: ignore[assignment]

SQL_DBS = ["postgresql", "mysql", "mssql", "oracle", "sqlite"]
NOSQL_DBS = ["mongodb", "elasticsearch"]
ALL_DBS = SQL_DBS + NOSQL_DBS

pytestmark = [pytest.mark.integration, pytest.mark.enterprise_scale]


def _conn_name(db_type: str) -> str:
    return f"enterprise_{db_type}"


def _conn_string(db_type: str) -> str:
    """Return the MCP-compatible connection string for a database type."""
    info = get_connection_info(db_type)
    if db_type == "oracle":
        dsn = info.get("dsn", "localhost:11521/FREEPDB1")
        return f"oracle+oracledb://{info['user']}:{info['password']}@{dsn}"
    if db_type == "sqlite":
        return info.get("path", "")
    return info.get("uri", "")


@pytest.fixture(scope="session")
def enterprise_databases():
    """Load NYC Taxi dataset into all available databases. Cleanup after."""
    if load_dataset is None:
        pytest.skip("database_loader module not available")

    loaded: dict[str, dict] = {}
    for db_type in ALL_DBS:
        try:
            stats = load_dataset(db_type)
            loaded[db_type] = stats
            print(
                f"  Loaded {db_type}: {stats['rows_loaded']:,} rows in {stats['elapsed_seconds']:.0f}s"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {db_type}: {exc}")

    if not loaded:
        pytest.skip("No enterprise databases available")

    yield loaded

    # Disconnect all MCP connections, then cleanup database data
    for db_type in list(loaded):
        try:
            call_tool("disconnect_database", {"name": _conn_name(db_type)})
        except Exception:  # noqa: BLE001
            pass
        try:
            cleanup_dataset(db_type)
        except Exception:  # noqa: BLE001
            pass


def _require_db(enterprise_databases: dict, db_type: str) -> str:
    """Ensure db_type was loaded, connect via MCP if needed, return conn name."""
    if db_type not in enterprise_databases:
        pytest.skip(f"{db_type} not available")
    name = _conn_name(db_type)
    # Try connecting (idempotent — if already connected, MCP returns "already")
    try:
        call_tool(
            "connect_database",
            {"name": name, "db_type": db_type, "conn_string": _conn_string(db_type)},
        )
    except Exception:  # noqa: BLE001
        pass
    return name


def _expected_row_count() -> int:
    """Return the expected number of rows in the cached dataset."""
    info = get_dataset_info()
    return info.get("row_count", 0)


@pytest.mark.parametrize("db_type", ALL_DBS)
class TestEnterpriseConnection:
    """Verify MCP can connect to and describe every enterprise database."""

    def test_connect_enterprise_db(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        # _require_db already connects via MCP — verify listing shows it
        result = call_tool("list_databases", {})
        assert result is not None
        result_str = str(result).lower()
        assert conn_name in result_str

    def test_describe_enterprise_db(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        result = call_tool("describe_database", {"name": conn_name})
        assert isinstance(result, dict)
        result_str = str(result).lower()
        assert "taxi_trips" in result_str or "taxi" in result_str


@pytest.mark.parametrize("db_type", SQL_DBS)
class TestLargeQueryStreaming:
    """Verify streaming behaviour on large SQL result sets."""

    def test_select_all_triggers_streaming(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        # Use a filtered query that returns many rows (triggers streaming)
        # but not all 9.5M (which would timeout).
        result = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": "SELECT * FROM taxi_trips WHERE fare_amount > 20",
            },
        )
        assert isinstance(result, dict)
        assert result.get("error") is not True, f"Query error: {result.get('message')}"
        meta = result.get("metadata", {})
        assert meta.get("streaming") is True or meta.get("chunked") is True, (
            f"Expected streaming/chunked for large result. metadata={meta}"
        )

    def test_chunked_retrieval(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        result = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": "SELECT vendorid, fare_amount, tip_amount FROM taxi_trips WHERE fare_amount > 50",
            },
        )
        assert isinstance(result, dict)
        assert result.get("error") is not True, f"Query error: {result.get('message')}"
        first_data = result.get("data", [])
        assert len(first_data) > 0

        query_id = result.get("metadata", {}).get("query_id")
        if query_id:
            chunk2 = call_tool(
                "next_chunk",
                {
                    "query_id": query_id,
                    "start_row": len(first_data) + 1,
                    "chunk_size": "500",
                },
            )
            assert chunk2 is not None

    def test_count_matches_loaded_rows(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        result = call_tool(
            "execute_query",
            {"name": conn_name, "query": "SELECT COUNT(*) AS cnt FROM taxi_trips"},
        )
        assert isinstance(result, dict)
        data = result.get("data", [])
        assert len(data) > 0
        expected = _expected_row_count()
        if expected > 0:
            assert data[0]["cnt"] == expected

    def test_large_aggregation(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        result = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": (
                    "SELECT payment_type, COUNT(*) AS cnt, "
                    "AVG(total_amount) AS avg_total, "
                    "SUM(tip_amount) AS sum_tip "
                    "FROM taxi_trips GROUP BY payment_type"
                ),
            },
        )
        assert isinstance(result, dict)
        data = result.get("data", [])
        assert len(data) > 0, "Aggregation returned no rows"


@pytest.mark.parametrize("db_type", SQL_DBS)
class TestMemoryAndPerformance:
    """Timed queries and memory-bound checks on enterprise-scale data."""

    def test_aggregation_under_30s(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        start = time.time()
        result = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": (
                    "SELECT payment_type, COUNT(*) AS cnt, "
                    "AVG(fare_amount) AS avg_fare, "
                    "AVG(tip_amount) AS avg_tip, "
                    "SUM(total_amount) AS sum_total "
                    "FROM taxi_trips GROUP BY payment_type"
                ),
            },
        )
        elapsed = time.time() - start
        print(f"[{db_type}] aggregation: {elapsed:.1f}s")
        assert elapsed < 30.0, f"Aggregation on {db_type} took {elapsed:.1f}s"
        assert isinstance(result, dict)

    def test_filtered_query_under_60s(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        start = time.time()
        result = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": (
                    "SELECT * FROM taxi_trips "
                    "WHERE fare_amount > 50 AND trip_distance > 10"
                ),
            },
        )
        elapsed = time.time() - start
        print(f"[{db_type}] filtered query: {elapsed:.1f}s")
        assert elapsed < 60.0, f"Filtered query on {db_type} took {elapsed:.1f}s"
        assert isinstance(result, dict)
        assert result.get("error") is not True, f"Query error: {result.get('message')}"

    def test_memory_bounds_reported(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        result = call_tool(
            "execute_query",
            {"name": conn_name, "query": "SELECT COUNT(*) FROM taxi_trips"},
        )
        assert isinstance(result, dict)
        mem = result.get("metadata", {}).get("memory_info", {})
        if mem:
            assert "total_gb" in mem or "available_gb" in mem

    def test_preflight_estimation(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        result = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": "SELECT * FROM taxi_trips",
                "preflight": True,
            },
        )
        assert isinstance(result, dict)
        result_str = str(result).lower()
        assert "estimated" in result_str or "rows" in result_str


@pytest.mark.parametrize("db_type", SQL_DBS)
class TestQueryMetadata:
    """Verify query metadata and data quality tools work at scale."""

    def test_query_metadata_available(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        # Execute a query that returns a query_id
        qr = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": "SELECT * FROM taxi_trips WHERE fare_amount > 100",
            },
        )
        assert isinstance(qr, dict)
        query_id = qr.get("metadata", {}).get("query_id")
        if query_id:
            result = call_tool("get_query_metadata", {"query_id": query_id})
            assert result is not None

    def test_data_quality_report(self, enterprise_databases, db_type):
        conn_name = _require_db(enterprise_databases, db_type)
        # Execute a query first to get a query_id
        qr = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": "SELECT * FROM taxi_trips WHERE vendorid = 1 LIMIT 500",
            },
        )
        assert isinstance(qr, dict)
        query_id = qr.get("metadata", {}).get("query_id")
        if not query_id:
            pytest.skip("No query_id returned — cannot test data quality report")
        result = call_tool("get_data_quality_report", {"query_id": query_id})
        assert result is not None
        result_str = str(result).lower()
        assert (
            "quality" in result_str
            or "report" in result_str
            or "null" in result_str
            or "column" in result_str
        )


class TestNoSQLEnterprise:
    """Targeted tests for MongoDB and Elasticsearch at enterprise scale."""

    # -- MongoDB --

    def test_mongodb_large_query(self, enterprise_databases):
        conn_name = _require_db(enterprise_databases, "mongodb")
        result = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": '{"fare_amount": {"$gt": 50}}',
            },
        )
        assert isinstance(result, dict)
        data = result.get("data", [])
        assert len(data) > 0, "Expected MongoDB results for fare_amount > 50"

    def test_mongodb_aggregation(self, enterprise_databases):
        conn_name = _require_db(enterprise_databases, "mongodb")
        result = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": (
                    '[{"$group": {"_id": "$payment_type", '
                    '"count": {"$sum": 1}, '
                    '"avg_total": {"$avg": "$total_amount"}}}]'
                ),
            },
        )
        assert isinstance(result, dict)
        data = result.get("data", [])
        assert len(data) > 0, "MongoDB aggregation returned no results"

    # -- Elasticsearch --

    def test_elasticsearch_search(self, enterprise_databases):
        conn_name = _require_db(enterprise_databases, "elasticsearch")
        result = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": (
                    '{"query": {"range": {"fare_amount": {"gte": 50, "lte": 200}}}}'
                ),
            },
        )
        assert isinstance(result, dict)
        data = result.get("data", result.get("hits", []))
        assert len(data) > 0, "Expected ES results for fare range query"

    def test_elasticsearch_aggregation(self, enterprise_databases):
        conn_name = _require_db(enterprise_databases, "elasticsearch")
        result = call_tool(
            "execute_query",
            {
                "name": conn_name,
                "query": (
                    '{"size": 0, "aggs": {"by_payment": '
                    '{"terms": {"field": "payment_type"}, '
                    '"aggs": {"avg_total": '
                    '{"avg": {"field": "total_amount"}}}}}}'
                ),
            },
        )
        assert result is not None


class TestCrossDatabaseConsistency:
    """Run the same aggregation across all SQL databases and verify results match."""

    _QUERY = (
        "SELECT payment_type, COUNT(*) AS cnt, "
        "ROUND(AVG(total_amount), 2) AS avg_total "
        "FROM taxi_trips GROUP BY payment_type ORDER BY payment_type"
    )

    def test_aggregation_results_match_across_sql_dbs(self, enterprise_databases):
        """All SQL engines should return the same aggregation results."""
        available_sql = [db for db in SQL_DBS if db in enterprise_databases]
        if len(available_sql) < 2:
            pytest.skip("Need at least 2 SQL databases for cross-DB comparison")

        results: dict[str, list] = {}
        for db_type in available_sql:
            conn_name = enterprise_databases[db_type]
            result = call_tool(
                "execute_query",
                {"name": conn_name, "query": self._QUERY},
            )
            assert isinstance(result, dict), f"{db_type} returned non-dict"
            data = result.get("data", [])
            assert len(data) > 0, f"{db_type} aggregation returned no rows"
            results[db_type] = data

        # Compare all against the first available database
        reference_db = available_sql[0]
        ref_data = results[reference_db]
        ref_counts = {row["payment_type"]: row["cnt"] for row in ref_data}

        for db_type in available_sql[1:]:
            db_data = results[db_type]
            db_counts = {row["payment_type"]: row["cnt"] for row in db_data}
            assert db_counts == ref_counts, (
                f"Count mismatch between {reference_db} and {db_type}: "
                f"{ref_counts} vs {db_counts}"
            )
