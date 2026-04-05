"""Security data models for LocalData MCP.

Enums, dataclasses, and configuration types used by the security subsystem.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional


class SecurityThreatLevel(Enum):
    """Security threat severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""

    QUERY_BLOCKED = "query_blocked"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    PARAMETER_VIOLATION = "parameter_violation"
    COMPLEXITY_VIOLATION = "complexity_violation"
    AUDIT_LOG = "audit_log"


class AttackPattern(Enum):
    """Known SQL injection attack patterns."""

    UNION_INJECTION = "union_injection"
    TIME_BASED_BLIND = "time_based_blind"
    BOOLEAN_BLIND = "boolean_blind"
    ERROR_BASED = "error_based"
    INFORMATION_EXTRACTION = "information_extraction"
    COMMENT_INJECTION = "comment_injection"
    STACKED_QUERIES = "stacked_queries"
    FUNCTION_ABUSE = "function_abuse"


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    # Rate limiting
    queries_per_minute: int = 60
    queries_per_hour: int = 1000
    burst_limit: int = 10

    # Query complexity limits
    max_query_length: int = 10000
    max_joins: int = 10
    max_subqueries: int = 5
    max_functions: int = 20

    # Resource protection
    memory_threshold_mb: float = 500.0
    cpu_threshold_percent: float = 80.0
    max_concurrent_queries: int = 10

    # Audit settings
    audit_enabled: bool = True
    audit_all_queries: bool = True
    audit_failed_only: bool = False
    retain_audit_days: int = 30

    # Threat detection
    enable_pattern_detection: bool = True
    enable_anomaly_detection: bool = True
    block_suspicious_patterns: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.queries_per_minute <= 0:
            raise ValueError("queries_per_minute must be positive")
        if self.burst_limit <= 0:
            raise ValueError("burst_limit must be positive")
        if self.max_query_length <= 0:
            raise ValueError("max_query_length must be positive")


@dataclass
class SecurityEvent:
    """Security event record."""

    timestamp: float
    event_type: SecurityEventType
    threat_level: SecurityThreatLevel
    database_name: str
    connection_id: Optional[str]
    query_fingerprint: str
    query_text: Optional[str]
    attack_pattern: Optional[AttackPattern]
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "database_name": self.database_name,
            "connection_id": self.connection_id,
            "query_fingerprint": self.query_fingerprint,
            "query_text": self.query_text,
            "attack_pattern": self.attack_pattern.value
            if self.attack_pattern
            else None,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass
class RateLimitState:
    """Rate limiting state for a connection."""

    queries_this_minute: Deque[float] = field(default_factory=deque)
    queries_this_hour: Deque[float] = field(default_factory=deque)
    burst_count: int = 0
    last_query_time: float = 0.0
    violations: int = 0
    blocked_until: float = 0.0

    def cleanup_old_queries(self, current_time: float):
        """Remove queries outside the time windows."""
        # Clean minute window
        while (
            self.queries_this_minute and current_time - self.queries_this_minute[0] > 60
        ):
            self.queries_this_minute.popleft()

        # Clean hour window
        while (
            self.queries_this_hour and current_time - self.queries_this_hour[0] > 3600
        ):
            self.queries_this_hour.popleft()


@dataclass
class QueryComplexity:
    """Analysis of query complexity."""

    length: int = 0
    joins: int = 0
    subqueries: int = 0
    functions: int = 0
    tables: int = 0
    conditions: int = 0
    unions: int = 0
    complexity_score: float = 0.0

    def calculate_score(self):
        """Calculate overall complexity score."""
        self.complexity_score = (
            self.length * 0.001
            + self.joins * 2.0
            + self.subqueries * 3.0
            + self.functions * 1.0
            + self.unions * 5.0
            + self.conditions * 0.5
        )
