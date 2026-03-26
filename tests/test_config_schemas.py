"""Tests for v1.6.0 configuration schemas."""

import pytest

from localdata_mcp.config_schemas import (
    BlobHandling,
    ConnectionsConfig,
    EvictionPolicy,
    MemoryConfig,
    QueryConfig,
    SecurityConfig,
    StagingConfig,
)


class TestStagingConfig:
    def test_defaults(self):
        c = StagingConfig()
        assert c.max_concurrent == 10
        assert c.max_size_mb == 2048
        assert c.max_total_mb == 10240
        assert c.timeout_minutes == 30
        assert c.eviction_policy == EvictionPolicy.LRU

    def test_string_eviction_policy(self):
        c = StagingConfig(eviction_policy="oldest")
        assert c.eviction_policy == EvictionPolicy.OLDEST

    def test_invalid_max_concurrent(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            StagingConfig(max_concurrent=0)

    def test_max_total_less_than_max_size(self):
        with pytest.raises(ValueError, match="max_total_mb"):
            StagingConfig(max_size_mb=5000, max_total_mb=1000)

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout_minutes"):
            StagingConfig(timeout_minutes=-1)


class TestMemoryConfig:
    def test_defaults(self):
        c = MemoryConfig()
        assert c.max_budget_mb == 512
        assert c.budget_percent == 10
        assert c.low_memory_threshold_gb == 1.0

    def test_invalid_budget_percent(self):
        with pytest.raises(ValueError, match="budget_percent"):
            MemoryConfig(budget_percent=0)

    def test_invalid_budget_percent_over_100(self):
        with pytest.raises(ValueError, match="budget_percent"):
            MemoryConfig(budget_percent=101)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="low_memory_threshold_gb"):
            MemoryConfig(low_memory_threshold_gb=0)


class TestQueryConfig:
    def test_defaults(self):
        c = QueryConfig()
        assert c.default_chunk_size == 100
        assert c.buffer_timeout_seconds == 600
        assert c.blob_handling == BlobHandling.EXCLUDE
        assert c.blob_max_size_mb == 5
        assert c.preflight_default is False

    def test_string_blob_handling(self):
        c = QueryConfig(blob_handling="include")
        assert c.blob_handling == BlobHandling.INCLUDE

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="default_chunk_size"):
            QueryConfig(default_chunk_size=0)

    def test_invalid_blob_max_size(self):
        with pytest.raises(ValueError, match="blob_max_size_mb"):
            QueryConfig(blob_max_size_mb=-1)


class TestConnectionsConfig:
    def test_defaults(self):
        c = ConnectionsConfig()
        assert c.max_concurrent == 10
        assert c.timeout_seconds == 30

    def test_invalid_concurrent(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            ConnectionsConfig(max_concurrent=0)

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout_seconds"):
            ConnectionsConfig(timeout_seconds=-5)


class TestSecurityConfig:
    def test_defaults(self):
        c = SecurityConfig()
        assert c.allowed_paths == ["."]
        assert c.blocked_keywords == []
        assert c.max_query_length == 10000

    def test_custom_values(self):
        c = SecurityConfig(
            allowed_paths=[".", "/data"],
            max_query_length=5000,
            blocked_keywords=["DROP"],
        )
        assert len(c.allowed_paths) == 2
        assert c.blocked_keywords == ["DROP"]

    def test_invalid_query_length(self):
        with pytest.raises(ValueError, match="max_query_length"):
            SecurityConfig(max_query_length=0)
