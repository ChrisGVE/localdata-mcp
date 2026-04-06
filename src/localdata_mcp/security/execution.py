"""Secure query execution context manager.

Extracted from SecurityManager to keep module sizes within the 300-line limit.
"""

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

from ..query_parser import SQLSecurityError
from .models import (
    SecurityConfig,
    SecurityEvent,
    SecurityEventType,
    SecurityThreatLevel,
)


@contextmanager
def secure_query_execution(
    query: str,
    database_name: str,
    connection_id: Optional[str],
    validate_fn: Callable[
        [str, str, Optional[str]],
        Tuple[bool, Optional[str], Dict[str, Any]],
    ],
    fingerprint_fn: Callable[[str], str],
    config: SecurityConfig,
    log_event_fn: Callable[[SecurityEvent], None],
):
    """Context manager for secure query execution with comprehensive monitoring.

    Args:
        query: SQL query string.
        database_name: Database name.
        connection_id: Optional connection identifier.
        validate_fn: Callable to validate query security.
        fingerprint_fn: Callable to create query fingerprint.
        config: Security configuration.
        log_event_fn: Callable to log security events.

    Yields:
        Execution context dict with security metadata.

    Raises:
        SQLSecurityError: If security validation fails.
    """
    start_time = time.time()
    query_fingerprint = fingerprint_fn(query)

    is_valid, error_msg, security_metadata = validate_fn(
        query, database_name, connection_id
    )

    if not is_valid:
        raise SQLSecurityError(f"Security validation failed: {error_msg}")

    execution_context = {
        "query_fingerprint": query_fingerprint,
        "database_name": database_name,
        "connection_id": connection_id,
        "start_time": start_time,
        "security_metadata": security_metadata,
        "execution_metadata": {},
    }

    try:
        yield execution_context

        execution_time = time.time() - start_time
        execution_context["execution_metadata"]["execution_time"] = execution_time
        execution_context["execution_metadata"]["success"] = True

        if config.audit_enabled:
            log_event_fn(
                SecurityEvent(
                    timestamp=start_time,
                    event_type=SecurityEventType.AUDIT_LOG,
                    threat_level=SecurityThreatLevel.LOW,
                    database_name=database_name,
                    connection_id=connection_id,
                    query_fingerprint=query_fingerprint,
                    query_text=query[:200],
                    attack_pattern=None,
                    message=f"Query executed successfully (time: {execution_time:.3f}s)",
                    metadata={
                        "security_metadata": security_metadata,
                        "execution_time": execution_time,
                    },
                )
            )

    except Exception as e:
        execution_time = time.time() - start_time
        execution_context["execution_metadata"]["execution_time"] = execution_time
        execution_context["execution_metadata"]["success"] = False
        execution_context["execution_metadata"]["error"] = str(e)

        if config.audit_enabled:
            log_event_fn(
                SecurityEvent(
                    timestamp=start_time,
                    event_type=SecurityEventType.AUDIT_LOG,
                    threat_level=SecurityThreatLevel.MEDIUM,
                    database_name=database_name,
                    connection_id=connection_id,
                    query_fingerprint=query_fingerprint,
                    query_text=query[:200],
                    attack_pattern=None,
                    message=f"Query execution failed: {str(e)[:200]}",
                    metadata={
                        "security_metadata": security_metadata,
                        "execution_time": execution_time,
                        "error": str(e),
                    },
                )
            )

        raise
