"""Tests for staging_manager module."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from localdata_mcp.staging_manager import StagingDatabase, StagingManager


class TestStagingDatabase:
    """Tests for StagingDatabase dataclass."""

    def test_staging_database_creation(self, tmp_path):
        """Test basic dataclass instantiation."""
        db_path = tmp_path / "test.sqlite"
        db_path.touch()
        now = datetime.now()

        staging = StagingDatabase(
            name="test_db",
            parent_connection="main",
            source_query="SELECT 1",
            created_at=now,
            file_path=db_path,
        )

        assert staging.name == "test_db"
        assert staging.parent_connection == "main"
        assert staging.source_query == "SELECT 1"
        assert staging.created_at == now
        assert staging.file_path == db_path
        assert staging.size_bytes == 0
        assert staging.row_count == 0
        assert staging.expires_at is None

    def test_staging_database_to_dict(self, tmp_path):
        """Test to_dict returns all fields and truncates long queries."""
        db_path = tmp_path / "test.sqlite"
        db_path.touch()
        now = datetime(2025, 1, 1, 12, 0, 0)
        expires = datetime(2025, 1, 1, 12, 30, 0)

        staging = StagingDatabase(
            name="test_db",
            parent_connection="main",
            source_query="SELECT 1",
            created_at=now,
            file_path=db_path,
            size_bytes=1048576,
            row_count=100,
            expires_at=expires,
        )

        d = staging.to_dict()
        assert d["name"] == "test_db"
        assert d["type"] == "staging"
        assert d["parent_connection"] == "main"
        assert d["source_query"] == "SELECT 1"
        assert d["created_at"] == now.isoformat()
        assert d["size_mb"] == 1.0
        assert d["row_count"] == 100
        assert d["expires_at"] == expires.isoformat()

    def test_staging_database_to_dict_truncates_query(self, tmp_path):
        """Test that to_dict truncates queries longer than 200 chars."""
        db_path = tmp_path / "test.sqlite"
        db_path.touch()
        long_query = "SELECT " + "x" * 300

        staging = StagingDatabase(
            name="test_db",
            parent_connection="main",
            source_query=long_query,
            created_at=datetime.now(),
            file_path=db_path,
        )

        d = staging.to_dict()
        assert len(d["source_query"]) == 203  # 200 + "..."
        assert d["source_query"].endswith("...")

    def test_staging_database_to_dict_no_expiry(self, tmp_path):
        """Test to_dict when expires_at is None."""
        db_path = tmp_path / "test.sqlite"
        db_path.touch()

        staging = StagingDatabase(
            name="test_db",
            parent_connection="main",
            source_query="SELECT 1",
            created_at=datetime.now(),
            file_path=db_path,
        )

        d = staging.to_dict()
        assert d["expires_at"] is None


class TestStagingManagerConfig:
    """Tests for StagingManager configuration."""

    def test_staging_manager_default_config(self):
        """Test that defaults are applied when config_manager unavailable."""
        manager = StagingManager(config={})

        # With empty config, properties return hardcoded defaults
        assert manager.max_concurrent == 10
        assert manager.max_size_mb == 2048
        assert manager.max_total_mb == 10240
        assert manager.timeout_minutes == 30

    def test_staging_manager_custom_config(self):
        """Test that custom config values are respected."""
        config = {
            "max_concurrent": 5,
            "max_size_mb": 512,
            "max_total_mb": 4096,
            "timeout_minutes": 15,
        }
        manager = StagingManager(config=config)

        assert manager.max_concurrent == 5
        assert manager.max_size_mb == 512
        assert manager.max_total_mb == 4096
        assert manager.timeout_minutes == 15


class TestStagingManagerOperations:
    """Tests for StagingManager create/remove/evict operations."""

    def test_create_staging(self):
        """Test that create_staging creates a file and tracks the entry."""
        manager = StagingManager(config={"timeout_minutes": 10})
        staging = manager.create_staging("mydb", "SELECT * FROM t")

        try:
            assert staging.name.startswith("staging_mydb_")
            assert staging.parent_connection == "mydb"
            assert staging.source_query == "SELECT * FROM t"
            assert staging.file_path.exists()
            assert staging.expires_at is not None
            assert staging.name in manager._staging_dbs
        finally:
            manager.remove_staging(staging.name)

    def test_create_staging_unique_names(self):
        """Test that two rapid calls produce different names."""
        manager = StagingManager(config={"max_concurrent": 10})
        s1 = manager.create_staging("db", "SELECT 1")
        s2 = manager.create_staging("db", "SELECT 2")

        try:
            assert s1.name != s2.name
            assert len(manager._staging_dbs) == 2
        finally:
            manager.remove_staging(s1.name)
            manager.remove_staging(s2.name)

    def test_create_staging_evicts_on_limit(self):
        """Test that exceeding max_concurrent triggers LRU eviction."""
        manager = StagingManager(config={"max_concurrent": 1, "timeout_minutes": 30})
        s1 = manager.create_staging("db", "SELECT 1")
        first_path = s1.file_path

        s2 = manager.create_staging("db", "SELECT 2")

        try:
            # Only one entry should remain after eviction
            assert len(manager._staging_dbs) == 1
            # First file should have been cleaned up
            assert not first_path.exists()
            # Second should exist
            assert s2.name in manager._staging_dbs
            assert s2.file_path.exists()
        finally:
            manager.remove_staging(s2.name)

    def test_remove_staging(self):
        """Test that remove_staging deletes the file and entry."""
        manager = StagingManager(config={})
        staging = manager.create_staging("db", "SELECT 1")
        path = staging.file_path
        name = staging.name

        assert path.exists()
        result = manager.remove_staging(name)

        assert result is True
        assert not path.exists()
        assert name not in manager._staging_dbs

    def test_remove_staging_nonexistent(self):
        """Test that removing a nonexistent staging returns False."""
        manager = StagingManager(config={})
        result = manager.remove_staging("does_not_exist")
        assert result is False

    def test_evict_lru_selects_oldest(self):
        """Test that _evict_lru removes the entry with oldest last_accessed."""
        manager = StagingManager(config={"max_concurrent": 10, "timeout_minutes": 30})
        s1 = manager.create_staging("db", "SELECT 1")
        s2 = manager.create_staging("db", "SELECT 2")
        s3 = manager.create_staging("db", "SELECT 3")

        # Manually set last_accessed so s2 is the oldest
        s1.last_accessed = datetime.now()
        s2.last_accessed = datetime.now() - timedelta(hours=2)
        s3.last_accessed = datetime.now() - timedelta(hours=1)

        try:
            manager._evict_lru()

            assert s2.name not in manager._staging_dbs
            assert s1.name in manager._staging_dbs
            assert s3.name in manager._staging_dbs
        finally:
            for name in list(manager._staging_dbs):
                manager.remove_staging(name)
