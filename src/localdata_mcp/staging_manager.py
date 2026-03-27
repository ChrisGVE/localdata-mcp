"""Staging database manager with LRU eviction for LocalData MCP."""

import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StagingDatabase:
    """Metadata for a staging SQLite database."""

    name: str
    parent_connection: str
    source_query: str
    created_at: datetime
    file_path: Path
    size_bytes: int = 0
    row_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of this staging database."""
        sq = self.source_query
        if len(sq) > 200:
            sq = sq[:200] + "..."
        return {
            "name": self.name,
            "type": "staging",
            "parent_connection": self.parent_connection,
            "source_query": sq,
            "created_at": self.created_at.isoformat(),
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
            "row_count": self.row_count,
            "expires_at": (self.expires_at.isoformat() if self.expires_at else None),
        }


class StagingManager:
    """Manages temporary staging SQLite databases with LRU eviction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._staging_dbs: Dict[str, StagingDatabase] = {}
        self._lock = threading.RLock()
        self._config = config or self._load_config()
        self._running = True
        self._cleanup_thread: Optional[threading.Thread] = None
        self._start_cleanup_thread()

    _DEFAULTS = {
        "max_concurrent": 10,
        "max_size_mb": 2048,
        "max_total_mb": 10240,
        "timeout_minutes": 30,
    }

    def _load_config(self) -> Dict[str, Any]:
        """Load staging configuration from config manager."""
        try:
            from .config_manager import get_config_manager

            cfg = get_config_manager().get_staging_config()
            return {k: getattr(cfg, k) for k in self._DEFAULTS}
        except Exception:
            return dict(self._DEFAULTS)

    @property
    def max_concurrent(self) -> int:
        return self._config.get("max_concurrent", 10)

    @property
    def max_size_mb(self) -> int:
        return self._config.get("max_size_mb", 2048)

    @property
    def max_total_mb(self) -> int:
        return self._config.get("max_total_mb", 10240)

    @property
    def timeout_minutes(self) -> int:
        return self._config.get("timeout_minutes", 30)

    def create_staging(
        self, parent_connection: str, source_query: str
    ) -> StagingDatabase:
        """Create a new staging database, evicting LRU if at capacity."""
        with self._lock:
            if len(self._staging_dbs) >= self.max_concurrent:
                self._evict_lru()

            total_bytes = sum(s.size_bytes for s in self._staging_dbs.values())
            if total_bytes > self.max_total_mb * 1024 * 1024:
                self._evict_lru()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"staging_{parent_connection}_{ts}"

            # Ensure unique name if one already exists with the same timestamp
            counter = 1
            base_name = name
            while name in self._staging_dbs:
                name = f"{base_name}_{counter}"
                counter += 1

            fd, path = tempfile.mkstemp(suffix=".sqlite", prefix="localdata_staging_")
            os.close(fd)

            staging = StagingDatabase(
                name=name,
                parent_connection=parent_connection,
                source_query=source_query,
                created_at=datetime.now(),
                file_path=Path(path),
                expires_at=datetime.now() + timedelta(minutes=self.timeout_minutes),
            )
            self._staging_dbs[name] = staging
            logger.info("Created staging database: %s", name)
            return staging

    def _evict_lru(self) -> None:
        """Evict least recently used staging database."""
        if not self._staging_dbs:
            return
        oldest = min(self._staging_dbs.values(), key=lambda s: s.last_accessed)
        self.remove_staging(oldest.name)

    def remove_staging(self, name: str) -> bool:
        """Remove a staging database and delete its file."""
        with self._lock:
            staging = self._staging_dbs.pop(name, None)
            if staging is None:
                return False
            try:
                staging.file_path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(
                    "Failed to delete staging file %s: %s",
                    staging.file_path,
                    e,
                )
            logger.info("Removed staging database: %s", name)
            return True

    def get_staging(self, name: str) -> Optional[StagingDatabase]:
        """Retrieve staging database and update last_accessed."""
        with self._lock:
            staging = self._staging_dbs.get(name)
            if staging:
                staging.last_accessed = datetime.now()
            return staging

    def touch_staging(self, name: str) -> bool:
        """Update last_accessed without returning full object."""
        with self._lock:
            staging = self._staging_dbs.get(name)
            if staging:
                staging.last_accessed = datetime.now()
                return True
            return False

    def list_staging(
        self, parent_connection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List active staging databases, optionally filtered by parent."""
        with self._lock:
            entries = list(self._staging_dbs.values())
        if parent_connection:
            entries = [s for s in entries if s.parent_connection == parent_connection]
        entries.sort(key=lambda s: s.created_at, reverse=True)
        return [s.to_dict() for s in entries]

    def update_staging_size(self, name: str, row_count: int) -> bool:
        """Update size and row count for a staging database."""
        with self._lock:
            staging = self._staging_dbs.get(name)
            if not staging:
                return False
            try:
                staging.size_bytes = staging.file_path.stat().st_size
            except OSError:
                pass
            staging.row_count = row_count
            staging.last_accessed = datetime.now()
            return True

    def get_staging_by_parent(self, parent_connection: str) -> List[StagingDatabase]:
        """Get all staging databases for a parent connection."""
        with self._lock:
            return [
                s
                for s in self._staging_dbs.values()
                if s.parent_connection == parent_connection
            ]

    def cleanup_by_parent(self, parent_connection: str) -> int:
        """Remove all staging databases for a parent connection."""
        targets = self.get_staging_by_parent(parent_connection)
        return sum(1 for s in targets if self.remove_staging(s.name))

    def get_status(self) -> Dict[str, Any]:
        """Get staging system health status."""
        with self._lock:
            dbs = list(self._staging_dbs.values())
        total_bytes = sum(s.size_bytes for s in dbs)
        oldest = min(dbs, key=lambda s: s.created_at) if dbs else None
        return {
            "active_count": len(dbs),
            "total_size_mb": round(total_bytes / (1024**2), 2),
            "disk_space": self.check_disk_space(),
            "config": {
                "max_concurrent": self.max_concurrent,
                "max_size_mb": self.max_size_mb,
                "max_total_mb": self.max_total_mb,
                "timeout_minutes": self.timeout_minutes,
            },
            "oldest_staging": oldest.name if oldest else None,
            "cleanup_thread_alive": bool(
                self._cleanup_thread and self._cleanup_thread.is_alive()
            ),
        }

    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space in temp directory."""
        usage = shutil.disk_usage(tempfile.gettempdir())
        staging_bytes = sum(s.size_bytes for s in self._staging_dbs.values())
        return {
            "total_gb": round(usage.total / (1024**3), 2),
            "available_gb": round(usage.free / (1024**3), 2),
            "used_percent": round((usage.used / usage.total) * 100, 1),
            "used_by_staging_mb": round(staging_bytes / (1024**2), 2),
            "can_create_staging": usage.free > 500 * 1024 * 1024,
        }

    def _start_cleanup_thread(self) -> None:
        """Start background thread for expired staging cleanup."""
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        """Periodically remove expired staging databases."""
        while self._running:
            time.sleep(60)
            try:
                with self._lock:
                    now = datetime.now()
                    expired = [
                        name
                        for name, s in self._staging_dbs.items()
                        if s.expires_at and now > s.expires_at
                    ]
                for name in expired:
                    self.remove_staging(name)
                    logger.info("Auto-cleaned expired staging: %s", name)
            except Exception as e:
                logger.debug("Cleanup loop error: %s", e)

    def stop(self) -> None:
        """Stop the staging manager and clean up all databases."""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        self._cleanup_all()

    def _cleanup_all(self) -> None:
        """Remove all staging databases."""
        with self._lock:
            names = list(self._staging_dbs.keys())
        for name in names:
            self.remove_staging(name)
        logger.info("Cleaned up %d staging databases", len(names))


_staging_manager: Optional[StagingManager] = None


def get_staging_manager() -> StagingManager:
    """Return the singleton StagingManager, creating one if needed."""
    global _staging_manager
    if _staging_manager is None:
        _staging_manager = StagingManager()
    return _staging_manager


def initialize_staging_manager(
    config: Optional[Dict[str, Any]] = None,
) -> StagingManager:
    """Create (or recreate) the singleton StagingManager with given config."""
    global _staging_manager
    if _staging_manager is not None:
        _staging_manager.stop()
    _staging_manager = StagingManager(config=config)
    return _staging_manager
