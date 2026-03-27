"""Tests for staging_manager new methods: list, update, parent-lookup, status, singleton."""

import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from localdata_mcp.staging_manager import (
    StagingDatabase,
    StagingManager,
    get_staging_manager,
    initialize_staging_manager,
)
import localdata_mcp.staging_manager as staging_module


@pytest.fixture()
def manager():
    """Create a StagingManager with generous limits and stop it after the test."""
    mgr = StagingManager(config={"max_concurrent": 20, "timeout_minutes": 30})
    yield mgr
    mgr.stop()


@pytest.fixture()
def populated_manager(manager):
    """Manager with three staging databases across two parents."""
    s1 = manager.create_staging("parent_a", "SELECT 1")
    s2 = manager.create_staging("parent_b", "SELECT 2")
    s3 = manager.create_staging("parent_a", "SELECT 3")
    return manager, [s1, s2, s3]


class TestListStaging:
    """Tests for list_staging method."""

    def test_list_staging_all(self, populated_manager):
        mgr, stagings = populated_manager
        result = mgr.list_staging()
        assert len(result) == 3
        # Should be dicts with expected keys
        for entry in result:
            assert "name" in entry
            assert "type" in entry
            assert entry["type"] == "staging"

    def test_list_staging_filtered(self, populated_manager):
        mgr, _ = populated_manager
        result = mgr.list_staging(parent_connection="parent_a")
        assert len(result) == 2
        for entry in result:
            assert entry["parent_connection"] == "parent_a"

    def test_list_staging_empty(self):
        mgr = StagingManager(config={})
        try:
            result = mgr.list_staging()
            assert result == []
        finally:
            mgr.stop()

    def test_list_staging_sorted_by_created_at(self, manager):
        """Entries should be sorted newest-first."""
        s1 = manager.create_staging("db", "SELECT 1")
        s1.created_at = datetime(2025, 1, 1)
        s2 = manager.create_staging("db", "SELECT 2")
        s2.created_at = datetime(2025, 6, 1)
        result = manager.list_staging()
        assert result[0]["created_at"] > result[1]["created_at"]

    def test_list_staging_filter_no_match(self, populated_manager):
        mgr, _ = populated_manager
        result = mgr.list_staging(parent_connection="nonexistent")
        assert result == []


class TestUpdateStagingSize:
    """Tests for update_staging_size method."""

    def test_update_staging_size(self, manager, tmp_path):
        staging = manager.create_staging("db", "SELECT 1")
        # Write some data so file has a real size
        staging.file_path.write_bytes(b"x" * 1024)
        old_accessed = staging.last_accessed

        time.sleep(0.01)
        result = manager.update_staging_size(staging.name, row_count=42)

        assert result is True
        assert staging.row_count == 42
        assert staging.size_bytes == 1024
        assert staging.last_accessed > old_accessed

    def test_update_nonexistent(self, manager):
        result = manager.update_staging_size("does_not_exist", row_count=10)
        assert result is False

    def test_update_staging_size_missing_file(self, manager):
        """When the file is missing, size_bytes stays unchanged but row_count updates."""
        staging = manager.create_staging("db", "SELECT 1")
        staging.file_path.unlink(missing_ok=True)
        staging.size_bytes = 999

        result = manager.update_staging_size(staging.name, row_count=5)
        assert result is True
        assert staging.row_count == 5
        # size_bytes unchanged because stat() failed
        assert staging.size_bytes == 999


class TestGetStagingByParent:
    """Tests for get_staging_by_parent method."""

    def test_get_staging_by_parent(self, populated_manager):
        mgr, stagings = populated_manager
        result = mgr.get_staging_by_parent("parent_a")
        assert len(result) == 2
        for s in result:
            assert isinstance(s, StagingDatabase)
            assert s.parent_connection == "parent_a"

    def test_get_staging_by_parent_empty(self, manager):
        result = manager.get_staging_by_parent("nonexistent")
        assert result == []


class TestCleanupByParent:
    """Tests for cleanup_by_parent method."""

    def test_cleanup_by_parent(self, populated_manager):
        mgr, stagings = populated_manager
        count = mgr.cleanup_by_parent("parent_a")
        assert count == 2
        # Only parent_b entry should remain
        remaining = mgr.list_staging()
        assert len(remaining) == 1
        assert remaining[0]["parent_connection"] == "parent_b"

    def test_cleanup_by_parent_returns_count(self, populated_manager):
        mgr, _ = populated_manager
        count = mgr.cleanup_by_parent("parent_b")
        assert count == 1

    def test_cleanup_by_parent_no_match(self, manager):
        count = manager.cleanup_by_parent("nonexistent")
        assert count == 0


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status_structure(self, manager):
        status = manager.get_status()
        assert "active_count" in status
        assert "total_size_mb" in status
        assert "disk_space" in status
        assert "config" in status
        assert "oldest_staging" in status
        assert "cleanup_thread_alive" in status
        assert isinstance(status["config"], dict)
        assert "max_concurrent" in status["config"]

    def test_get_status_empty(self, manager):
        status = manager.get_status()
        assert status["active_count"] == 0
        assert status["total_size_mb"] == 0.0
        assert status["oldest_staging"] is None

    def test_get_status_with_entries(self, populated_manager):
        mgr, stagings = populated_manager
        stagings[0].size_bytes = 1024 * 1024  # 1 MB
        stagings[1].size_bytes = 2 * 1024 * 1024  # 2 MB
        status = mgr.get_status()
        assert status["active_count"] == 3
        assert status["total_size_mb"] == 3.0
        assert status["oldest_staging"] is not None
        assert status["cleanup_thread_alive"] is True

    def test_get_status_config_values(self, manager):
        status = manager.get_status()
        assert status["config"]["max_concurrent"] == 20
        assert status["config"]["timeout_minutes"] == 30


class TestSingleton:
    """Tests for get_staging_manager and initialize_staging_manager."""

    def test_get_staging_manager_singleton(self):
        """get_staging_manager returns the same instance on repeated calls."""
        # Reset module-level singleton
        staging_module._staging_manager = None
        try:
            m1 = get_staging_manager()
            m2 = get_staging_manager()
            assert m1 is m2
        finally:
            m1.stop()
            staging_module._staging_manager = None

    def test_initialize_staging_manager(self):
        """initialize_staging_manager replaces the existing singleton."""
        staging_module._staging_manager = None
        try:
            m1 = initialize_staging_manager(config={"timeout_minutes": 5})
            assert m1.timeout_minutes == 5

            m2 = initialize_staging_manager(config={"timeout_minutes": 15})
            assert m2.timeout_minutes == 15
            assert m2 is not m1
            # Module singleton should be updated
            assert get_staging_manager() is m2
        finally:
            m2.stop()
            staging_module._staging_manager = None

    def test_initialize_stops_previous(self):
        """initialize_staging_manager stops the previous instance."""
        staging_module._staging_manager = None
        try:
            m1 = initialize_staging_manager()
            assert m1._running is True

            m2 = initialize_staging_manager()
            assert m1._running is False
            assert m2._running is True
        finally:
            m2.stop()
            staging_module._staging_manager = None
