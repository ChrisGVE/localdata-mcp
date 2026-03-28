"""Query execution audit log with ring buffer for LocalData MCP."""

import hashlib
import logging
import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

QueryStatus = Literal["success", "error", "timeout"]


@dataclass
class QueryAuditEntry:
    """Single audit log entry for a query execution."""

    timestamp: datetime
    database: str
    query: str
    status: QueryStatus
    duration_ms: float
    rows_returned: Optional[int] = None
    memory_used_mb: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    query_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry to dictionary, truncating long queries."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "database": self.database,
            "query": (
                self.query[:200] + "..." if len(self.query) > 200 else self.query
            ),
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "rows_returned": self.rows_returned,
            "memory_used_mb": round(self.memory_used_mb, 2),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class QueryAuditBuffer:
    """Thread-safe ring buffer for query audit entries."""

    def __init__(self, max_entries: int = 1000):
        self._buffer: Deque[QueryAuditEntry] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    @staticmethod
    def generate_query_hash(query: str) -> str:
        """Generate a short hash for a query, normalizing whitespace."""
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def record(self, entry: QueryAuditEntry) -> None:
        """Append an entry to the ring buffer."""
        with self._lock:
            self._buffer.append(entry)

    def record_query(
        self,
        database: str,
        query: str,
        status: QueryStatus,
        duration_ms: float,
        rows_returned: Optional[int] = None,
        memory_used_mb: float = 0.0,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """Record a query execution and return its hash."""
        qhash = self.generate_query_hash(query)
        entry = QueryAuditEntry(
            timestamp=datetime.now(),
            database=database,
            query=query,
            status=status,
            duration_ms=duration_ms,
            rows_returned=rows_returned,
            memory_used_mb=memory_used_mb,
            error_type=error_type,
            error_message=error_message,
            query_hash=qhash,
        )
        self.record(entry)
        return qhash

    def get_entries(
        self,
        database: Optional[str] = None,
        status: Optional[str] = None,
        since_minutes: int = 60,
        limit: int = 50,
    ) -> List[QueryAuditEntry]:
        """Retrieve filtered entries from the buffer."""
        cutoff = datetime.now() - timedelta(minutes=since_minutes)
        with self._lock:
            entries = list(self._buffer)
        filtered = [
            e
            for e in entries
            if e.timestamp >= cutoff
            and (database is None or e.database == database)
            and (status is None or e.status == status)
        ]
        filtered.sort(key=lambda e: e.timestamp, reverse=True)
        return filtered[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics over the buffer."""
        with self._lock:
            entries = list(self._buffer)
        if not entries:
            return {
                "total": 0,
                "by_status": {},
                "by_database": {},
                "avg_duration_ms": 0,
            }
        by_status: Dict[str, int] = {}
        by_db: Dict[str, int] = {}
        durations: List[float] = []
        for e in entries:
            by_status[e.status] = by_status.get(e.status, 0) + 1
            by_db[e.database] = by_db.get(e.database, 0) + 1
            durations.append(e.duration_ms)
        return {
            "total": len(entries),
            "by_status": by_status,
            "by_database": by_db,
            "avg_duration_ms": round(sum(durations) / len(durations), 2),
            "max_duration_ms": round(max(durations), 2),
            "error_count": by_status.get("error", 0) + by_status.get("timeout", 0),
        }

    def clear(self) -> None:
        """Remove all entries from the buffer."""
        with self._lock:
            self._buffer.clear()


_audit_buffer: Optional[QueryAuditBuffer] = None


def get_query_audit_buffer() -> QueryAuditBuffer:
    """Return the global singleton audit buffer, creating it if needed."""
    global _audit_buffer
    if _audit_buffer is None:
        _audit_buffer = QueryAuditBuffer()
    return _audit_buffer
