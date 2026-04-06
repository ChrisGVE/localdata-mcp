"""Rate limiting logic for the security system.

Standalone functions extracted from SecurityManager to keep module sizes
within the 300-line project limit.
"""

import threading
import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

from ..logging_manager import get_logger
from .models import (
    RateLimitState,
    SecurityConfig,
    SecurityEvent,
    SecurityEventType,
    SecurityThreatLevel,
)

logger = get_logger(__name__)


def check_rate_limits(
    connection_id: str,
    database_name: str,
    config: SecurityConfig,
    rate_limits: Dict[str, RateLimitState],
    rate_limit_lock: threading.RLock,
    log_event_fn: Callable[[SecurityEvent], None],
) -> Tuple[bool, Optional[str]]:
    """Check if connection exceeds rate limits.

    Args:
        connection_id: Connection identifier.
        database_name: Database name.
        config: Security configuration.
        rate_limits: Shared rate-limit state dictionary.
        rate_limit_lock: Lock protecting *rate_limits*.
        log_event_fn: Callback to log a SecurityEvent.

    Returns:
        Tuple of (is_allowed, error_message).
    """
    current_time = time.time()

    with rate_limit_lock:
        rate_state = rate_limits[connection_id]
        rate_state.cleanup_old_queries(current_time)

        # Check if currently blocked
        if current_time < rate_state.blocked_until:
            remaining = int(rate_state.blocked_until - current_time)
            return (
                False,
                f"Connection blocked for {remaining} seconds due to rate limit violations",
            )

        # Check minute limit
        if len(rate_state.queries_this_minute) >= config.queries_per_minute:
            rate_state.violations += 1
            rate_state.blocked_until = current_time + min(
                60, rate_state.violations * 10
            )

            log_event_fn(
                SecurityEvent(
                    timestamp=current_time,
                    event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                    threat_level=SecurityThreatLevel.MEDIUM,
                    database_name=database_name,
                    connection_id=connection_id,
                    query_fingerprint="rate_limit",
                    query_text=None,
                    attack_pattern=None,
                    message=(
                        f"Minute rate limit exceeded: "
                        f"{len(rate_state.queries_this_minute)}/{config.queries_per_minute}"
                    ),
                    metadata={"violations": rate_state.violations},
                )
            )

            return (
                False,
                (
                    f"Rate limit exceeded: "
                    f"{len(rate_state.queries_this_minute)}/{config.queries_per_minute} "
                    f"queries per minute"
                ),
            )

        # Check hour limit
        if len(rate_state.queries_this_hour) >= config.queries_per_hour:
            rate_state.violations += 1
            rate_state.blocked_until = current_time + min(
                300, rate_state.violations * 60
            )

            log_event_fn(
                SecurityEvent(
                    timestamp=current_time,
                    event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                    threat_level=SecurityThreatLevel.HIGH,
                    database_name=database_name,
                    connection_id=connection_id,
                    query_fingerprint="rate_limit",
                    query_text=None,
                    attack_pattern=None,
                    message=(
                        f"Hour rate limit exceeded: "
                        f"{len(rate_state.queries_this_hour)}/{config.queries_per_hour}"
                    ),
                    metadata={"violations": rate_state.violations},
                )
            )

            return (
                False,
                (
                    f"Rate limit exceeded: "
                    f"{len(rate_state.queries_this_hour)}/{config.queries_per_hour} "
                    f"queries per hour"
                ),
            )

        # Check burst limit
        if current_time - rate_state.last_query_time < 1.0:
            rate_state.burst_count += 1
        else:
            rate_state.burst_count = 0

        if rate_state.burst_count > config.burst_limit:
            rate_state.violations += 1
            rate_state.blocked_until = current_time + 5  # Short block for burst

            log_event_fn(
                SecurityEvent(
                    timestamp=current_time,
                    event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                    threat_level=SecurityThreatLevel.MEDIUM,
                    database_name=database_name,
                    connection_id=connection_id,
                    query_fingerprint="burst_limit",
                    query_text=None,
                    attack_pattern=None,
                    message=(
                        f"Burst limit exceeded: "
                        f"{rate_state.burst_count}/{config.burst_limit} "
                        f"queries per second"
                    ),
                    metadata={"violations": rate_state.violations},
                )
            )

            return False, "Burst limit exceeded: too many queries in short time"

        # Record query
        rate_state.queries_this_minute.append(current_time)
        rate_state.queries_this_hour.append(current_time)
        rate_state.last_query_time = current_time

        return True, None
