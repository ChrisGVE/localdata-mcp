"""Tests for DiskBudgetConfig and DiskMonitor."""

import os
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import pytest

from localdata_mcp.config_schemas import DiskBudgetConfig
from localdata_mcp.disk_monitor import DiskMonitor

DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])


# --- DiskBudgetConfig tests ---


class TestDiskBudgetConfigDefaults:
    def test_defaults(self):
        cfg = DiskBudgetConfig()
        assert cfg.max_staging_size_mb == 2048
        assert cfg.max_total_staging_mb == 10240
        assert cfg.disk_warning_threshold == 0.90
        assert cfg.headroom_mb == 500
        assert cfg.check_interval_rows == 1000


class TestDiskBudgetConfigValidation:
    def test_negative_max_staging_size(self):
        with pytest.raises(ValueError, match="max_staging_size_mb must be positive"):
            DiskBudgetConfig(max_staging_size_mb=-1)

    def test_zero_max_staging_size(self):
        with pytest.raises(ValueError, match="max_staging_size_mb must be positive"):
            DiskBudgetConfig(max_staging_size_mb=0)

    def test_total_less_than_per_db(self):
        with pytest.raises(
            ValueError, match="max_total_staging_mb must be >= max_staging_size_mb"
        ):
            DiskBudgetConfig(max_staging_size_mb=500, max_total_staging_mb=100)

    def test_threshold_zero(self):
        with pytest.raises(ValueError, match="disk_warning_threshold must be 0-1"):
            DiskBudgetConfig(disk_warning_threshold=0.0)

    def test_threshold_above_one(self):
        with pytest.raises(ValueError, match="disk_warning_threshold must be 0-1"):
            DiskBudgetConfig(disk_warning_threshold=1.5)

    def test_negative_headroom(self):
        with pytest.raises(ValueError, match="headroom_mb must be positive"):
            DiskBudgetConfig(headroom_mb=-10)

    def test_zero_check_interval(self):
        with pytest.raises(ValueError, match="check_interval_rows must be positive"):
            DiskBudgetConfig(check_interval_rows=0)

    def test_threshold_exactly_one_is_valid(self):
        cfg = DiskBudgetConfig(disk_warning_threshold=1.0)
        assert cfg.disk_warning_threshold == 1.0


# --- DiskMonitor tests ---


class TestDiskMonitorSkipNonInterval:
    def test_skips_non_interval_rows(self, tmp_path):
        db_file = tmp_path / "staging.db"
        db_file.write_bytes(b"\x00" * 100)
        monitor = DiskMonitor(db_file, DiskBudgetConfig(check_interval_rows=100))

        # Rows 1-99 should all skip the check
        for row in range(1, 100):
            can_continue, reason = monitor.check_can_continue(row)
            assert can_continue is True
            assert reason is None

    def test_checks_at_interval(self, tmp_path):
        db_file = tmp_path / "staging.db"
        db_file.write_bytes(b"\x00" * 100)
        config = DiskBudgetConfig(check_interval_rows=50)
        monitor = DiskMonitor(db_file, config)

        # Row 50 is an interval check; file is tiny so should pass
        with patch(
            "shutil.disk_usage",
            return_value=DiskUsage(
                total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3
            ),
        ):
            can_continue, reason = monitor.check_can_continue(50)
            assert can_continue is True
            assert reason is None


class TestDiskMonitorStagingLimit:
    def test_aborts_when_staging_too_large(self, tmp_path):
        db_file = tmp_path / "staging.db"
        # Write >90% of 10MB limit = >9MB
        db_file.write_bytes(b"\x00" * (10 * 1024 * 1024))
        config = DiskBudgetConfig(max_staging_size_mb=10, check_interval_rows=1)
        monitor = DiskMonitor(db_file, config)

        can_continue, reason = monitor.check_can_continue(1)
        assert can_continue is False
        assert "Staging limit reached" in reason


class TestDiskMonitorHeadroom:
    def test_aborts_when_free_space_low(self, tmp_path):
        db_file = tmp_path / "staging.db"
        db_file.write_bytes(b"\x00" * 100)
        config = DiskBudgetConfig(headroom_mb=1000, check_interval_rows=1)
        monitor = DiskMonitor(db_file, config)

        # Simulate only 100MB free
        with patch(
            "shutil.disk_usage",
            return_value=DiskUsage(
                total=500 * 1024**3, used=499 * 1024**3, free=100 * 1024 * 1024
            ),
        ):
            can_continue, reason = monitor.check_can_continue(1)
            assert can_continue is False
            assert "System disk critical" in reason


class TestDiskMonitorThreshold:
    def test_aborts_when_usage_percent_high(self, tmp_path):
        db_file = tmp_path / "staging.db"
        db_file.write_bytes(b"\x00" * 100)
        config = DiskBudgetConfig(
            disk_warning_threshold=0.80,
            headroom_mb=1,
            check_interval_rows=1,
        )
        monitor = DiskMonitor(db_file, config)

        # 95% used
        total = 100 * 1024**3
        used = int(total * 0.95)
        free = total - used
        with patch(
            "shutil.disk_usage",
            return_value=DiskUsage(total=total, used=used, free=free),
        ):
            can_continue, reason = monitor.check_can_continue(1)
            assert can_continue is False
            assert "Disk usage above 80%" in reason


class TestDiskMonitorAllOk:
    def test_continues_when_within_limits(self, tmp_path):
        db_file = tmp_path / "staging.db"
        db_file.write_bytes(b"\x00" * 100)
        config = DiskBudgetConfig(
            max_staging_size_mb=2048,
            headroom_mb=500,
            disk_warning_threshold=0.90,
            check_interval_rows=1,
        )
        monitor = DiskMonitor(db_file, config)

        total = 500 * 1024**3
        used = int(total * 0.50)
        free = total - used
        with patch(
            "shutil.disk_usage",
            return_value=DiskUsage(total=total, used=used, free=free),
        ):
            can_continue, reason = monitor.check_can_continue(1)
            assert can_continue is True
            assert reason is None


class TestDiskMonitorMissingFile:
    def test_graceful_when_file_missing(self, tmp_path):
        db_file = tmp_path / "nonexistent.db"
        config = DiskBudgetConfig(check_interval_rows=1)
        monitor = DiskMonitor(db_file, config)

        can_continue, reason = monitor.check_can_continue(1)
        assert can_continue is True
        assert reason is None


class TestGetCurrentSizeMb:
    def test_returns_zero_initially(self, tmp_path):
        db_file = tmp_path / "staging.db"
        monitor = DiskMonitor(db_file)
        assert monitor.get_current_size_mb() == 0.0

    def test_returns_size_after_check(self, tmp_path):
        db_file = tmp_path / "staging.db"
        db_file.write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2MB
        config = DiskBudgetConfig(check_interval_rows=1)
        monitor = DiskMonitor(db_file, config)

        with patch(
            "shutil.disk_usage",
            return_value=DiskUsage(
                total=500 * 1024**3, used=100 * 1024**3, free=400 * 1024**3
            ),
        ):
            monitor.check_can_continue(1)

        assert abs(monitor.get_current_size_mb() - 2.0) < 0.01


class TestGetSystemDiskStatus:
    def test_returns_disk_info(self, tmp_path):
        db_file = tmp_path / "staging.db"
        db_file.write_bytes(b"\x00" * 100)
        config = DiskBudgetConfig(headroom_mb=500)
        monitor = DiskMonitor(db_file, config)

        total = 500 * 1024**3
        used = 250 * 1024**3
        free = 250 * 1024**3
        with patch(
            "shutil.disk_usage",
            return_value=DiskUsage(total=total, used=used, free=free),
        ):
            status = monitor.get_system_disk_status()

        assert status["total_gb"] == 500.0
        assert status["available_gb"] == 250.0
        assert status["used_percent"] == 50.0
        assert status["meets_headroom"] is True

    def test_returns_error_on_os_error(self, tmp_path):
        db_file = tmp_path / "staging.db"
        db_file.write_bytes(b"\x00" * 100)
        monitor = DiskMonitor(db_file)

        with patch("shutil.disk_usage", side_effect=OSError("fail")):
            status = monitor.get_system_disk_status()

        assert "error" in status
