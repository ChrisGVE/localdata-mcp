"""Security manager for LocalData MCP.

Enterprise-grade security management system providing rate limiting,
resource monitoring, query validation, and audit logging.

Heavy logic is delegated to sibling modules (rate_limiter, resources,
events, validation) so that each file stays within the 300-line limit.
"""

import logging
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..query_parser import QueryParser, SQLSecurityError, get_query_parser
from ..timeout_manager import get_timeout_manager, TimeoutReason
from ..connection_manager import get_enhanced_connection_manager
from ..logging_manager import get_logging_manager, get_logger

from .models import (
    AttackPattern,
    QueryComplexity,
    RateLimitState,
    SecurityConfig,
    SecurityEvent,
    SecurityEventType,
    SecurityThreatLevel,
)
from .detection import (
    compile_attack_patterns,
    detect_attack_patterns,
    create_query_fingerprint,
    analyze_query_complexity,
)
from .rate_limiter import check_rate_limits as _check_rate_limits
from .resources import (
    check_resource_limits as _check_resource_limits,
    start_resource_monitoring,
)
from .events import (
    log_security_event as _log_security_event,
    get_security_events as _get_security_events,
    get_security_statistics as _get_security_statistics,
)
from .validation import (
    validate_query_security as _validate_query_security,
    secure_query_execution as _secure_query_execution,
)

# Get structured logger
logger = get_logger(__name__)


class SecurityManager:
    """Enterprise-grade security management system."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize the security manager.

        Args:
            config: Security configuration. Uses defaults if None.
        """
        self.config = config or SecurityConfig()
        self.query_parser = get_query_parser()
        self.timeout_manager = get_timeout_manager()
        self.connection_manager = get_enhanced_connection_manager()

        # Rate limiting state
        self._rate_limits: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._rate_limit_lock = threading.RLock()

        # Security event storage
        self._security_events: Deque[SecurityEvent] = deque(maxlen=10000)
        self._events_lock = threading.Lock()

        # Attack pattern detection
        self._attack_patterns = compile_attack_patterns()

        # Resource monitoring
        self._resource_monitor_active = True
        start_resource_monitoring(
            self.config,
            self._rate_limits,
            self._rate_limit_lock,
            self._security_events,
            self._events_lock,
            lambda: self._resource_monitor_active,
        )

        logging_manager = get_logging_manager()
        with logging_manager.context(
            operation="security_manager_init", component="security_manager"
        ):
            logger.info(
                "SecurityManager initialized",
                threat_detection_enabled=True,
                rate_limiting_enabled=True,
                resource_monitoring_enabled=True,
                attack_patterns_loaded=len(self._attack_patterns),
            )

    def create_query_fingerprint(self, query: str) -> str:
        """Create a cryptographic fingerprint of the query.

        Args:
            query: SQL query string

        Returns:
            SHA-256 hexdigest of the normalized query.
        """
        return create_query_fingerprint(query)

    def analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze query complexity for resource limit enforcement.

        Args:
            query: SQL query string

        Returns:
            Detailed complexity analysis.
        """
        return analyze_query_complexity(query)

    def detect_attack_patterns(self, query: str) -> List[AttackPattern]:
        """Detect known attack patterns in query.

        Args:
            query: SQL query string

        Returns:
            List of detected attack patterns.
        """
        return detect_attack_patterns(query, self._attack_patterns)

    def check_rate_limits(
        self, connection_id: str, database_name: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if connection exceeds rate limits.

        Args:
            connection_id: Connection identifier.
            database_name: Database name.

        Returns:
            Tuple of (is_allowed, error_message).
        """
        return _check_rate_limits(
            connection_id,
            database_name,
            self.config,
            self._rate_limits,
            self._rate_limit_lock,
            self._log_security_event,
        )

    def check_resource_limits(self, database_name: str) -> Tuple[bool, Optional[str]]:
        """Check system resource limits before query execution.

        Args:
            database_name: Database name.

        Returns:
            Tuple of (is_allowed, error_message).
        """
        return _check_resource_limits(
            database_name,
            self.config,
            self.timeout_manager,
            self._log_security_event,
        )

    def validate_query_security(
        self, query: str, database_name: str, connection_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Comprehensive security validation for SQL queries.

        Args:
            query: SQL query string.
            database_name: Database name.
            connection_id: Optional connection identifier.

        Returns:
            Tuple of (is_valid, error_message, security_metadata).
        """
        return _validate_query_security(
            query,
            database_name,
            connection_id,
            self.config,
            self.query_parser,
            self.create_query_fingerprint,
            self.analyze_query_complexity,
            self.detect_attack_patterns,
            self.check_rate_limits,
            self.check_resource_limits,
            self._log_security_event,
        )

    @contextmanager
    def secure_query_execution(
        self, query: str, database_name: str, connection_id: Optional[str] = None
    ):
        """Context manager for secure query execution.

        Args:
            query: SQL query string.
            database_name: Database name.
            connection_id: Optional connection identifier.

        Yields:
            Execution context dict with security metadata.

        Raises:
            SQLSecurityError: If security validation fails.
        """
        with _secure_query_execution(
            query,
            database_name,
            connection_id,
            self.validate_query_security,
            self.create_query_fingerprint,
            self.config,
            self._log_security_event,
        ) as ctx:
            yield ctx

    def _log_security_event(self, event: SecurityEvent) -> None:
        """Log a security event using structured logging.

        Args:
            event: SecurityEvent to log.
        """
        _log_security_event(event, self._security_events, self._events_lock)

    def get_security_events(
        self,
        limit: Optional[int] = None,
        event_types: Optional[List[SecurityEventType]] = None,
        threat_levels: Optional[List[SecurityThreatLevel]] = None,
    ) -> List[SecurityEvent]:
        """Get security events with optional filtering.

        Args:
            limit: Maximum number of events to return.
            event_types: Filter by event types.
            threat_levels: Filter by threat levels.

        Returns:
            Filtered security events.
        """
        return _get_security_events(
            self._security_events,
            self._events_lock,
            limit=limit,
            event_types=event_types,
            threat_levels=threat_levels,
        )

    def get_security_statistics(self) -> Dict[str, Any]:
        """Get comprehensive security statistics.

        Returns:
            Security statistics and metrics.
        """
        return _get_security_statistics(
            self._security_events,
            self._events_lock,
            self._rate_limits,
            self._rate_limit_lock,
            self.config,
        )

    def close(self) -> None:
        """Close the security manager and clean up resources."""
        self._resource_monitor_active = False
        logging_manager = get_logging_manager()
        with logging_manager.context(
            operation="security_manager_close", component="security_manager"
        ):
            logger.info(
                "SecurityManager shutdown complete",
                events_processed=len(self._security_events),
                connections_monitored=len(self._rate_limits),
            )


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get or create global security manager instance.

    Returns:
        Global security manager.
    """
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def initialize_security_manager(
    config: Optional[SecurityConfig] = None,
) -> SecurityManager:
    """Initialize a new global security manager instance.

    Args:
        config: Optional security configuration.

    Returns:
        New security manager.
    """
    global _security_manager
    if _security_manager is not None:
        _security_manager.close()
    _security_manager = SecurityManager(config)
    return _security_manager


# Convenience functions for common operations


def validate_query_security(
    query: str, database_name: str, connection_id: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Validate query security using global security manager.

    Args:
        query: SQL query string.
        database_name: Database name.
        connection_id: Optional connection identifier.

    Returns:
        Tuple of (is_valid, error_message).
    """
    security_manager = get_security_manager()
    is_valid, error_msg, _ = security_manager.validate_query_security(
        query, database_name, connection_id
    )
    return is_valid, error_msg


def secure_query_execution(
    query: str, database_name: str, connection_id: Optional[str] = None
):
    """Context manager for secure query execution.

    Args:
        query: SQL query string.
        database_name: Database name.
        connection_id: Optional connection identifier.

    Returns:
        Context manager for secure execution.
    """
    security_manager = get_security_manager()
    return security_manager.secure_query_execution(query, database_name, connection_id)
