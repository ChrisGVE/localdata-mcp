"""Staging database manager with LRU eviction for LocalData MCP."""

import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

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

    def _load_config(self) -> Dict[str, Any]:
        """Load staging configuration from config manager."""
        try:
            from .config_manager import get_config_manager

            cfg = get_config_manager().get_staging_config()
            return {
                "max_concurrent": cfg.max_concurrent,
                "max_size_mb": cfg.max_size_mb,
                "max_total_mb": cfg.max_total_mb,
                "timeout_minutes": cfg.timeout_minutes,
            }
        except Exception:
            return {
                "max_concurrent": 10,
                "max_size_mb": 2048,
                "max_total_mb": 10240,
                "timeout_minutes": 30,
            }

    @property
    def max_concurrent(self) -> int:
        """Maximum number of concurrent staging databases."""
        return self._config.get("max_concurrent", 10)

    @property
    def max_size_mb(self) -> int:
        """Maximum size in MB for a single staging database."""
        return self._config.get("max_size_mb", 2048)

    @property
    def max_total_mb(self) -> int:
        """Maximum total size in MB for all staging databases."""
        return self._config.get("max_total_mb", 10240)

    @property
    def timeout_minutes(self) -> int:
        """Timeout in minutes before a staging database expires."""
        return self._config.get("timeout_minutes", 30)

    def create_staging(
        self, parent_connection: str, source_query: str
    ) -> StagingDatabase:
        """Create a new staging database.

        If the maximum number of concurrent staging databases is reached,
        the least recently used one is evicted first. Similarly, if total
        size exceeds the configured limit, eviction is triggered.

        Args:
            parent_connection: Name of the parent database connection.
            source_query: The SQL query that populates this staging db.

        Returns:
            The newly created StagingDatabase instance.
        """
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
        """Remove a staging database and clean up its file.

        Args:
            name: The name of the staging database to remove.

        Returns:
            True if the staging database was found and removed, False otherwise.
        """
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
