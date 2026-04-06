"""Resource monitoring for the security system.

Standalone functions extracted from SecurityManager to keep module sizes
within the 300-line project limit.
"""

import logging
import threading
import time
from typing import Callable, Deque, Dict, Optional, Tuple

import psutil

from ..logging_manager import get_logger, get_logging_manager
from .models import (
    RateLimitState,
    SecurityConfig,
    SecurityEvent,
    SecurityEventType,
    SecurityThreatLevel,
)

logger = get_logger(__name__)


def check_resource_limits(
    database_name: str,
    config: SecurityConfig,
    timeout_manager,
    log_event_fn: Callable[[SecurityEvent], None],
) -> Tuple[bool, Optional[str]]:
    """Check system resource limits before query execution.

    Args:
        database_name: Database name.
        config: Security configuration.
        timeout_manager: Timeout manager instance for active-op queries.
        log_event_fn: Callback to log a SecurityEvent.

    Returns:
        Tuple of (is_allowed, error_message).
    """
    try:
        # Check memory usage
        memory_mb = psutil.virtual_memory().used / 1024 / 1024
        if memory_mb > config.memory_threshold_mb:
            log_event_fn(
                SecurityEvent(
                    timestamp=time.time(),
                    event_type=SecurityEventType.RESOURCE_EXHAUSTION,
                    threat_level=SecurityThreatLevel.HIGH,
                    database_name=database_name,
                    connection_id=None,
                    query_fingerprint="resource_limit",
                    query_text=None,
                    attack_pattern=None,
                    message=(
                        f"Memory threshold exceeded: "
                        f"{memory_mb:.1f}MB > {config.memory_threshold_mb}MB"
                    ),
                    metadata={
                        "memory_mb": memory_mb,
                        "threshold": config.memory_threshold_mb,
                    },
                )
            )
            return False, f"System memory threshold exceeded: {memory_mb:.1f}MB"

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > config.cpu_threshold_percent:
            log_event_fn(
                SecurityEvent(
                    timestamp=time.time(),
                    event_type=SecurityEventType.RESOURCE_EXHAUSTION,
                    threat_level=SecurityThreatLevel.MEDIUM,
                    database_name=database_name,
                    connection_id=None,
                    query_fingerprint="resource_limit",
                    query_text=None,
                    attack_pattern=None,
                    message=(
                        f"CPU threshold exceeded: "
                        f"{cpu_percent:.1f}% > {config.cpu_threshold_percent}%"
                    ),
                    metadata={
                        "cpu_percent": cpu_percent,
                        "threshold": config.cpu_threshold_percent,
                    },
                )
            )
            return False, f"System CPU threshold exceeded: {cpu_percent:.1f}%"

        # Check active query count using timeout manager
        active_operations = timeout_manager.get_active_operations()
        db_operations = [
            op
            for op in active_operations.values()
            if op["database_name"] == database_name
        ]

        if len(db_operations) >= config.max_concurrent_queries:
            return (
                False,
                (
                    f"Maximum concurrent queries exceeded: "
                    f"{len(db_operations)}/{config.max_concurrent_queries}"
                ),
            )

        return True, None

    except Exception as e:
        logging_manager = get_logging_manager()
        with logging_manager.context(
            operation="resource_limit_check_error",
            component="security_manager",
            database_name=database_name,
        ):
            logger.warning(
                "Resource limit check failed, allowing query",
                error_type=type(e).__name__,
                error_message=str(e),
            )
        return True, None  # Allow query if we can't check resources


def start_resource_monitoring(
    config: SecurityConfig,
    rate_limits: Dict[str, RateLimitState],
    rate_limit_lock: threading.RLock,
    security_events: Deque[SecurityEvent],
    events_lock: threading.Lock,
    active_flag_getter: Callable[[], bool],
) -> threading.Thread:
    """Start background resource monitoring thread.

    Args:
        config: Security configuration.
        rate_limits: Shared rate-limit state dictionary.
        rate_limit_lock: Lock protecting *rate_limits*.
        security_events: Shared deque of security events.
        events_lock: Lock protecting *security_events*.
        active_flag_getter: Callable returning whether monitoring is active.

    Returns:
        The started daemon thread.
    """

    def monitor():
        while active_flag_getter():
            try:
                # Clean up old rate limit data
                current_time = time.time()
                with rate_limit_lock:
                    for rate_state in rate_limits.values():
                        rate_state.cleanup_old_queries(current_time)

                # Clean up old security events
                cutoff_time = current_time - (config.retain_audit_days * 86400)
                with events_lock:
                    while (
                        security_events and security_events[0].timestamp < cutoff_time
                    ):
                        security_events.popleft()

                time.sleep(60)  # Run cleanup every minute

            except Exception as e:
                logging_manager = get_logging_manager()
                logging_manager.log_error(
                    e, "security_manager", operation="resource_monitoring"
                )
                time.sleep(10)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    logging_manager = get_logging_manager()
    with logging_manager.context(
        operation="resource_monitoring_start", component="security_manager"
    ):
        logger.info(
            "Security resource monitoring started",
            cleanup_interval=60,
            audit_retention_days=config.retain_audit_days,
        )
    return thread
