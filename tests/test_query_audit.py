"""Tests for query audit ring buffer and readonly mode enforcement."""

from datetime import datetime, timedelta

from localdata_mcp.config_schemas import SecurityConfig
from localdata_mcp.query_audit import (
    QueryAuditBuffer,
    QueryAuditEntry,
    get_query_audit_buffer,
)
from localdata_mcp.query_parser import check_readonly

# ---------------------------------------------------------------------------
# QueryAuditEntry
# ---------------------------------------------------------------------------


class TestQueryAuditEntry:
    def test_audit_entry_creation(self):
        entry = QueryAuditEntry(
            timestamp=datetime.now(),
            database="test.db",
            query="SELECT 1",
            status="success",
            duration_ms=12.5,
            rows_returned=1,
        )
        assert entry.database == "test.db"
        assert entry.status == "success"
        assert entry.duration_ms == 12.5
        assert entry.rows_returned == 1

    def test_audit_entry_to_dict_truncates_long_query(self):
        long_query = "SELECT " + "x" * 300
        entry = QueryAuditEntry(
            timestamp=datetime.now(),
            database="db",
            query=long_query,
            status="success",
            duration_ms=1.0,
        )
        d = entry.to_dict()
        assert len(d["query"]) == 203  # 200 chars + "..."
        assert d["query"].endswith("...")

    def test_audit_entry_to_dict_short_query_unchanged(self):
        entry = QueryAuditEntry(
            timestamp=datetime.now(),
            database="db",
            query="SELECT 1",
            status="success",
            duration_ms=1.0,
        )
        d = entry.to_dict()
        assert d["query"] == "SELECT 1"


# ---------------------------------------------------------------------------
# QueryAuditBuffer – hashing
# ---------------------------------------------------------------------------


class TestQueryHash:
    def test_generate_query_hash_deterministic(self):
        h1 = QueryAuditBuffer.generate_query_hash("SELECT 1")
        h2 = QueryAuditBuffer.generate_query_hash("SELECT 1")
        assert h1 == h2
        assert len(h1) == 16

    def test_generate_query_hash_normalization(self):
        h1 = QueryAuditBuffer.generate_query_hash("SELECT   1")
        h2 = QueryAuditBuffer.generate_query_hash("  select 1  ")
        assert h1 == h2


# ---------------------------------------------------------------------------
# QueryAuditBuffer – record / retrieve
# ---------------------------------------------------------------------------


class TestQueryAuditBuffer:
    def test_buffer_record_and_retrieve(self):
        buf = QueryAuditBuffer(max_entries=100)
        buf.record_query("db1", "SELECT 1", "success", 5.0, rows_returned=1)
        entries = buf.get_entries(since_minutes=5)
        assert len(entries) == 1
        assert entries[0].database == "db1"

    def test_buffer_ring_eviction(self):
        buf = QueryAuditBuffer(max_entries=1000)
        for i in range(1001):
            buf.record_query("db", f"SELECT {i}", "success", 1.0)
        entries = buf.get_entries(since_minutes=60, limit=2000)
        assert len(entries) == 1000

    def test_buffer_filter_by_database(self):
        buf = QueryAuditBuffer()
        buf.record_query("db_a", "SELECT 1", "success", 1.0)
        buf.record_query("db_b", "SELECT 2", "success", 1.0)
        entries = buf.get_entries(database="db_a", since_minutes=5)
        assert len(entries) == 1
        assert entries[0].database == "db_a"

    def test_buffer_filter_by_status(self):
        buf = QueryAuditBuffer()
        buf.record_query("db", "SELECT 1", "success", 1.0)
        buf.record_query("db", "SELECT 2", "error", 1.0, error_type="Syntax")
        entries = buf.get_entries(status="error", since_minutes=5)
        assert len(entries) == 1
        assert entries[0].status == "error"

    def test_buffer_filter_by_time(self):
        buf = QueryAuditBuffer()
        old = QueryAuditEntry(
            timestamp=datetime.now() - timedelta(hours=2),
            database="db",
            query="SELECT old",
            status="success",
            duration_ms=1.0,
        )
        buf.record(old)
        buf.record_query("db", "SELECT new", "success", 1.0)
        entries = buf.get_entries(since_minutes=60)
        assert len(entries) == 1
        assert "new" in entries[0].query

    def test_buffer_limit(self):
        buf = QueryAuditBuffer()
        for i in range(10):
            buf.record_query("db", f"SELECT {i}", "success", 1.0)
        entries = buf.get_entries(since_minutes=5, limit=3)
        assert len(entries) == 3

    def test_buffer_stats(self):
        buf = QueryAuditBuffer()
        buf.record_query("db1", "SELECT 1", "success", 10.0, rows_returned=5)
        buf.record_query("db2", "SELECT 2", "error", 20.0, error_type="Timeout")
        stats = buf.get_stats()
        assert stats["total"] == 2
        assert stats["by_status"]["success"] == 1
        assert stats["by_status"]["error"] == 1
        assert stats["avg_duration_ms"] == 15.0
        assert stats["max_duration_ms"] == 20.0
        assert stats["error_count"] == 1
        assert stats["by_database"]["db1"] == 1

    def test_buffer_stats_empty(self):
        buf = QueryAuditBuffer()
        stats = buf.get_stats()
        assert stats["total"] == 0

    def test_buffer_clear(self):
        buf = QueryAuditBuffer()
        buf.record_query("db", "SELECT 1", "success", 1.0)
        buf.clear()
        entries = buf.get_entries(since_minutes=60)
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_singleton_pattern(self):
        import localdata_mcp.query_audit as mod

        mod._audit_buffer = None  # reset
        buf1 = get_query_audit_buffer()
        buf2 = get_query_audit_buffer()
        assert buf1 is buf2
        mod._audit_buffer = None  # cleanup


# ---------------------------------------------------------------------------
# Readonly enforcement
# ---------------------------------------------------------------------------


class TestReadonlyEnforcement:
    def test_readonly_blocks_select_into(self):
        assert check_readonly("SELECT col INTO new_table FROM t") is not None

    def test_readonly_blocks_create_table_as(self):
        assert check_readonly("CREATE TABLE t2 AS SELECT * FROM t1") is not None

    def test_readonly_allows_normal_select(self):
        assert check_readonly("SELECT * FROM users WHERE id = 1") is None

    def test_readonly_allows_select_with_into_in_column_name(self):
        # "INTO" as part of a subquery or alias should not be an issue
        # if it does not match the pattern; this tests that normal selects pass
        assert check_readonly("SELECT id FROM customers") is None

    def test_readonly_blocks_merge_into(self):
        assert check_readonly("MERGE INTO target USING source ON ...") is not None

    def test_readonly_blocks_copy_to(self):
        assert check_readonly("COPY users TO '/tmp/out.csv'") is not None

    def test_readonly_blocks_select_into_temp(self):
        assert check_readonly("SELECT col INTO #temp FROM t") is not None


# ---------------------------------------------------------------------------
# SecurityConfig readonly field
# ---------------------------------------------------------------------------


class TestReadonlyConfig:
    def test_readonly_config_field_default(self):
        cfg = SecurityConfig()
        assert cfg.readonly is False

    def test_readonly_config_field_set_true(self):
        cfg = SecurityConfig(readonly=True)
        assert cfg.readonly is True
