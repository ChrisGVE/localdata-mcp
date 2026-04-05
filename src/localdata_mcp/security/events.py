"""Security event logging and statistics.

Standalone functions extracted from SecurityManager to keep module sizes
within the 300-line project limit.
"""

import threading
import time
from collections import defaultdict
from typing import Any, Deque, Dict, List, Optional

import psutil

from ..logging_manager import get_logging_manager, get_logger
from .models import (
    RateLimitState,
    SecurityConfig,
    SecurityEvent,
    SecurityEventType,
    SecurityThreatLevel,
)

logger = get_logger(__name__)


def log_security_event(
    event: SecurityEvent,
    security_events: Deque[SecurityEvent],
    events_lock: threading.Lock,
) -> None:
    """Log a security event using structured logging.

    Args:
        event: SecurityEvent to log.
        security_events: Shared deque of security events.
        events_lock: Lock protecting *security_events*.
    """
    with events_lock:
        security_events.append(event)

    # Log using structured logging system
    logging_manager = get_logging_manager()

    # Map threat levels to severity for structured logging
    severity_mapping = {
        SecurityThreatLevel.LOW: "low",
        SecurityThreatLevel.MEDIUM: "medium",
        SecurityThreatLevel.HIGH: "high",
        SecurityThreatLevel.CRITICAL: "critical",
    }

    # Use the structured logging security event method
    logging_manager.log_security_event(
        event.event_type.value,
        severity_mapping[event.threat_level],
        event.message,
        query_fingerprint=event.query_fingerprint,
        connection_id=event.connection_id,
        database_name=event.database_name,
        client_ip=event.metadata.get("client_ip"),
        timestamp=event.timestamp,
        metadata=event.metadata,
    )


def get_security_events(
    security_events: Deque[SecurityEvent],
    events_lock: threading.Lock,
    limit: Optional[int] = None,
    event_types: Optional[List[SecurityEventType]] = None,
    threat_levels: Optional[List[SecurityThreatLevel]] = None,
) -> List[SecurityEvent]:
    """Get security events with optional filtering.

    Args:
        security_events: Shared deque of security events.
        events_lock: Lock protecting *security_events*.
        limit: Maximum number of events to return.
        event_types: Filter by event types.
        threat_levels: Filter by threat levels.

    Returns:
        Filtered security events.
    """
    with events_lock:
        events = list(security_events)

    # Apply filters
    if event_types:
        events = [e for e in events if e.event_type in event_types]

    if threat_levels:
        events = [e for e in events if e.threat_level in threat_levels]

    # Sort by timestamp (newest first)
    events.sort(key=lambda e: e.timestamp, reverse=True)

    if limit:
        events = events[:limit]

    return events


def get_security_statistics(
    security_events: Deque[SecurityEvent],
    events_lock: threading.Lock,
    rate_limits: Dict[str, RateLimitState],
    rate_limit_lock: threading.RLock,
    config: SecurityConfig,
) -> Dict[str, Any]:
    """Get comprehensive security statistics.

    Args:
        security_events: Shared deque of security events.
        events_lock: Lock protecting *security_events*.
        rate_limits: Shared rate-limit state dictionary.
        rate_limit_lock: Lock protecting *rate_limits*.
        config: Security configuration.

    Returns:
        Security statistics and metrics.
    """
    with events_lock:
        events = list(security_events)

    current_time = time.time()
    hour_ago = current_time - 3600
    day_ago = current_time - 86400

    # Basic statistics
    total_events = len(events)
    events_last_hour = len([e for e in events if e.timestamp >= hour_ago])
    events_last_day = len([e for e in events if e.timestamp >= day_ago])

    # Event type breakdown
    event_type_counts: Dict[str, int] = defaultdict(int)
    for event in events:
        event_type_counts[event.event_type.value] += 1

    # Threat level breakdown
    threat_level_counts: Dict[str, int] = defaultdict(int)
    for event in events:
        threat_level_counts[event.threat_level.value] += 1

    # Attack pattern breakdown
    attack_pattern_counts: Dict[str, int] = defaultdict(int)
    for event in events:
        if event.attack_pattern:
            attack_pattern_counts[event.attack_pattern.value] += 1

    # Rate limiting statistics
    with rate_limit_lock:
        rate_limit_stats = {}
        for connection_id, rate_state in rate_limits.items():
            rate_limit_stats[connection_id] = {
                "queries_this_minute": len(rate_state.queries_this_minute),
                "queries_this_hour": len(rate_state.queries_this_hour),
                "violations": rate_state.violations,
                "blocked_until": rate_state.blocked_until,
                "currently_blocked": current_time < rate_state.blocked_until,
            }

    # Resource statistics
    try:
        memory_mb = psutil.virtual_memory().used / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
    except Exception:
        memory_mb = 0
        cpu_percent = 0

    return {
        "event_statistics": {
            "total_events": total_events,
            "events_last_hour": events_last_hour,
            "events_last_day": events_last_day,
            "event_types": dict(event_type_counts),
            "threat_levels": dict(threat_level_counts),
            "attack_patterns": dict(attack_pattern_counts),
        },
        "rate_limiting": {
            "active_connections": len(rate_limits),
            "connections": rate_limit_stats,
            "config": {
                "queries_per_minute": config.queries_per_minute,
                "queries_per_hour": config.queries_per_hour,
                "burst_limit": config.burst_limit,
            },
        },
        "resource_monitoring": {
            "current_memory_mb": memory_mb,
            "memory_threshold_mb": config.memory_threshold_mb,
            "current_cpu_percent": cpu_percent,
            "cpu_threshold_percent": config.cpu_threshold_percent,
            "memory_warning": memory_mb > config.memory_threshold_mb,
            "cpu_warning": cpu_percent > config.cpu_threshold_percent,
        },
        "configuration": {
            "audit_enabled": config.audit_enabled,
            "pattern_detection_enabled": config.enable_pattern_detection,
            "block_suspicious_patterns": config.block_suspicious_patterns,
            "max_query_length": config.max_query_length,
            "max_joins": config.max_joins,
            "max_subqueries": config.max_subqueries,
        },
    }
