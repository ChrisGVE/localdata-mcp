"""Tests for staging_manager access, disk space, and cleanup methods."""

import time
from datetime import datetime

import pytest

from localdata_mcp.staging_manager import StagingManager


class TestStagingManagerAccess:
    """Tests for get_staging and touch_staging access methods."""

    def test_get_staging_updates_last_accessed(self):
        """Test that get_staging updates the last_accessed timestamp."""
        manager = StagingManager(config={"timeout_minutes": 30})
        staging = manager.create_staging("db", "SELECT 1")
        old_accessed = staging.last_accessed

        time.sleep(0.01)
        result = manager.get_staging(staging.name)

        try:
            assert result is not None
            assert result.name == staging.name
            assert result.last_accessed > old_accessed
        finally:
            manager.stop()

    def test_get_staging_nonexistent_returns_none(self):
        """Test that get_staging returns None for unknown name."""
        manager = StagingManager(config={})
        result = manager.get_staging("nonexistent")

        try:
            assert result is None
        finally:
            manager.stop()

    def test_touch_staging_returns_true(self):
        """Test that touch_staging returns True for existing entry."""
        manager = StagingManager(config={"timeout_minutes": 30})
        staging = manager.create_staging("db", "SELECT 1")
        old_accessed = staging.last_accessed

        time.sleep(0.01)
        result = manager.touch_staging(staging.name)

        try:
            assert result is True
            assert staging.last_accessed > old_accessed
        finally:
            manager.stop()

    def test_touch_staging_nonexistent_returns_false(self):
        """Test that touch_staging returns False for unknown name."""
        manager = StagingManager(config={})
        result = manager.touch_staging("nonexistent")

        try:
            assert result is False
        finally:
            manager.stop()


class TestStagingManagerDiskSpace:
    """Tests for check_disk_space."""

    def test_check_disk_space_structure(self):
        """Test that check_disk_space returns all expected keys."""
        manager = StagingManager(config={})
        result = manager.check_disk_space()

        try:
            expected_keys = {
                "total_gb",
                "available_gb",
                "used_percent",
                "used_by_staging_mb",
                "can_create_staging",
            }
            assert set(result.keys()) == expected_keys
            assert isinstance(result["total_gb"], float)
            assert isinstance(result["available_gb"], float)
            assert isinstance(result["used_percent"], float)
            assert isinstance(result["used_by_staging_mb"], float)
            assert isinstance(result["can_create_staging"], bool)
        finally:
            manager.stop()

    def test_check_disk_space_staging_bytes(self):
        """Test that used_by_staging_mb reflects tracked staging sizes."""
        manager = StagingManager(config={"timeout_minutes": 30})
        s1 = manager.create_staging("db", "SELECT 1")
        s1.size_bytes = 2 * 1024 * 1024  # 2 MB
        s2 = manager.create_staging("db", "SELECT 2")
        s2.size_bytes = 3 * 1024 * 1024  # 3 MB

        result = manager.check_disk_space()

        try:
            assert result["used_by_staging_mb"] == 5.0
        finally:
            manager.stop()


class TestStagingManagerCleanup:
    """Tests for cleanup thread and stop/cleanup_all."""

    def test_cleanup_thread_starts(self):
        """Test that cleanup thread is daemon and alive after init."""
        manager = StagingManager(config={})

        try:
            assert manager._cleanup_thread is not None
            assert manager._cleanup_thread.daemon is True
            assert manager._cleanup_thread.is_alive()
        finally:
            manager.stop()

    def test_stop_sets_running_false(self):
        """Test that stop() sets _running to False."""
        manager = StagingManager(config={})
        assert manager._running is True

        manager.stop()

        assert manager._running is False

    def test_cleanup_all_removes_everything(self):
        """Test that _cleanup_all removes all staging databases."""
        manager = StagingManager(config={"timeout_minutes": 30})
        s1 = manager.create_staging("db", "SELECT 1")
        s2 = manager.create_staging("db", "SELECT 2")
        s3 = manager.create_staging("db", "SELECT 3")
        paths = [s1.file_path, s2.file_path, s3.file_path]

        assert len(manager._staging_dbs) == 3

        manager._cleanup_all()

        assert len(manager._staging_dbs) == 0
        for p in paths:
            assert not p.exists()

        manager.stop()
