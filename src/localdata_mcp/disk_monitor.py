"""Disk space monitoring during query streaming."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config_schemas import DiskBudgetConfig

logger = logging.getLogger(__name__)


class DiskMonitor:
    """Monitors disk usage during streaming to staging databases."""

    def __init__(self, staging_path: Path, config: Optional[DiskBudgetConfig] = None):
        self._path = staging_path
        self._config = config or DiskBudgetConfig()
        self._current_size = 0

    def check_can_continue(self, rows_written: int) -> Tuple[bool, Optional[str]]:
        """Check if streaming can continue.

        Returns (can_continue, abort_reason).
        Only checks every check_interval_rows rows.
        """
        if rows_written % self._config.check_interval_rows != 0:
            return True, None

        # Check staging file size
        try:
            self._current_size = self._path.stat().st_size
        except OSError:
            return True, None

        staging_mb = self._current_size / (1024 * 1024)
        limit_mb = self._config.max_staging_size_mb
        if staging_mb > limit_mb * 0.90:
            reason = f"Staging limit reached ({staging_mb:.1f}MB of {limit_mb}MB)"
            logger.warning("Disk abort: %s", reason)
            return False, reason

        # Check system disk
        try:
            usage = shutil.disk_usage(self._path.parent)
        except OSError:
            return True, None

        free_mb = usage.free / (1024 * 1024)
        if free_mb < self._config.headroom_mb:
            reason = (
                f"System disk critical ({free_mb:.0f}MB free, "
                f"need {self._config.headroom_mb}MB)"
            )
            logger.error("Disk abort: %s", reason)
            return False, reason

        used_pct = usage.used / usage.total
        if used_pct > self._config.disk_warning_threshold:
            reason = (
                f"Disk usage above "
                f"{self._config.disk_warning_threshold * 100:.0f}% "
                f"({used_pct * 100:.1f}%)"
            )
            logger.warning("Disk abort: %s", reason)
            return False, reason

        return True, None

    def get_current_size_mb(self) -> float:
        """Return current staging file size in megabytes."""
        return self._current_size / (1024 * 1024)

    def get_system_disk_status(self) -> Dict[str, Any]:
        """Return current system disk usage information."""
        try:
            usage = shutil.disk_usage(self._path.parent)
            return {
                "total_gb": round(usage.total / (1024**3), 2),
                "available_gb": round(usage.free / (1024**3), 2),
                "used_percent": round((usage.used / usage.total) * 100, 1),
                "meets_headroom": (usage.free > self._config.headroom_mb * 1024 * 1024),
            }
        except OSError:
            return {"error": "Unable to read disk usage"}
