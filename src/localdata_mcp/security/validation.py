"""Query security validation and secure execution.

Standalone functions extracted from SecurityManager to keep module sizes
within the 300-line project limit.
"""

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..logging_manager import get_logging_manager, get_logger
from ..query_parser import SQLSecurityError
from .models import (
    AttackPattern,
    QueryComplexity,
    SecurityConfig,
    SecurityEvent,
    SecurityEventType,
    SecurityThreatLevel,
)

logger = get_logger(__name__)


def validate_query_security(
    query: str,
    database_name: str,
    connection_id: Optional[str],
    config: SecurityConfig,
    query_parser,
    fingerprint_fn: Callable[[str], str],
    complexity_fn: Callable[[str], QueryComplexity],
    detect_fn: Callable[[str], List[AttackPattern]],
    rate_limit_fn: Callable[[str, str], Tuple[bool, Optional[str]]],
    resource_fn: Callable[[str], Tuple[bool, Optional[str]]],
    log_event_fn: Callable[[SecurityEvent], None],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Comprehensive security validation for SQL queries.

    Args:
        query: SQL query string.
        database_name: Database name.
        connection_id: Optional connection identifier.
        config: Security configuration.
        query_parser: QueryParser instance for basic validation.
        fingerprint_fn: Callable to create query fingerprint.
        complexity_fn: Callable to analyze query complexity.
        detect_fn: Callable to detect attack patterns.
        rate_limit_fn: Callable to check rate limits.
        resource_fn: Callable to check resource limits.
        log_event_fn: Callable to log security events.

    Returns:
        Tuple of (is_valid, error_message, security_metadata).
    """
    start_time = time.time()
    query_fingerprint = fingerprint_fn(query)

    security_metadata: Dict[str, Any] = {
        "fingerprint": query_fingerprint,
        "validation_time": 0.0,
        "threat_level": SecurityThreatLevel.LOW,
        "checks_performed": [],
    }

    try:
        # 1. Basic SQL validation from Task 1
        security_metadata["checks_performed"].append("basic_sql_validation")
        is_valid, error_msg = query_parser.validate_query(query)
        if not is_valid:
            log_event_fn(
                SecurityEvent(
                    timestamp=start_time,
                    event_type=SecurityEventType.QUERY_BLOCKED,
                    threat_level=SecurityThreatLevel.HIGH,
                    database_name=database_name,
                    connection_id=connection_id,
                    query_fingerprint=query_fingerprint,
                    query_text=query[:200],
                    attack_pattern=None,
                    message=f"Basic SQL validation failed: {error_msg}",
                    metadata={"validation_error": error_msg},
                )
            )
            security_metadata["threat_level"] = SecurityThreatLevel.HIGH
            return False, error_msg, security_metadata

        # 2. Rate limiting check
        if connection_id:
            security_metadata["checks_performed"].append("rate_limiting")
            rate_allowed, rate_error = rate_limit_fn(connection_id, database_name)
            if not rate_allowed:
                security_metadata["threat_level"] = SecurityThreatLevel.MEDIUM
                return False, rate_error, security_metadata

        # 3. Resource limit check
        security_metadata["checks_performed"].append("resource_limits")
        resource_allowed, resource_error = resource_fn(database_name)
        if not resource_allowed:
            security_metadata["threat_level"] = SecurityThreatLevel.HIGH
            return False, resource_error, security_metadata

        # 4. Query complexity analysis
        security_metadata["checks_performed"].append("complexity_analysis")
        complexity = complexity_fn(query)
        security_metadata["complexity"] = {
            "score": complexity.complexity_score,
            "length": complexity.length,
            "joins": complexity.joins,
            "subqueries": complexity.subqueries,
        }

        # Check complexity limits
        result = _check_complexity_limits(
            query,
            database_name,
            connection_id,
            config,
            complexity,
            query_fingerprint,
            start_time,
            security_metadata,
            log_event_fn,
        )
        if result is not None:
            return result

        # 5. Attack pattern detection
        result = _check_attack_patterns(
            query,
            database_name,
            connection_id,
            config,
            query_fingerprint,
            start_time,
            security_metadata,
            detect_fn,
            log_event_fn,
        )
        if result is not None:
            return result

        # All checks passed
        validation_time = time.time() - start_time
        security_metadata["validation_time"] = validation_time

        # Log successful validation if audit is enabled
        if config.audit_enabled and config.audit_all_queries:
            log_event_fn(
                SecurityEvent(
                    timestamp=start_time,
                    event_type=SecurityEventType.AUDIT_LOG,
                    threat_level=SecurityThreatLevel.LOW,
                    database_name=database_name,
                    connection_id=connection_id,
                    query_fingerprint=query_fingerprint,
                    query_text=query[:500],
                    attack_pattern=None,
                    message=f"Query validation passed (validation time: {validation_time:.3f}s)",
                    metadata=security_metadata.copy(),
                )
            )

        return True, None, security_metadata

    except Exception as e:
        logging_manager = get_logging_manager()
        logging_manager.log_error(
            e,
            "security_manager",
            database_name=database_name,
            connection_id=connection_id,
            validation_time=time.time() - start_time,
        )
        security_metadata["validation_time"] = time.time() - start_time
        security_metadata["error"] = str(e)
        return False, f"Security validation error: {e}", security_metadata


def _check_complexity_limits(
    query: str,
    database_name: str,
    connection_id: Optional[str],
    config: SecurityConfig,
    complexity: QueryComplexity,
    query_fingerprint: str,
    start_time: float,
    security_metadata: Dict[str, Any],
    log_event_fn: Callable[[SecurityEvent], None],
) -> Optional[Tuple[bool, Optional[str], Dict[str, Any]]]:
    """Check query complexity limits, returning a result tuple if violated."""
    if complexity.length > config.max_query_length:
        log_event_fn(
            SecurityEvent(
                timestamp=start_time,
                event_type=SecurityEventType.COMPLEXITY_VIOLATION,
                threat_level=SecurityThreatLevel.MEDIUM,
                database_name=database_name,
                connection_id=connection_id,
                query_fingerprint=query_fingerprint,
                query_text=query[:200],
                attack_pattern=None,
                message=f"Query length exceeds limit: {complexity.length}/{config.max_query_length}",
                metadata={"complexity": complexity.__dict__},
            )
        )
        security_metadata["threat_level"] = SecurityThreatLevel.MEDIUM
        return (
            False,
            f"Query too long: {complexity.length}/{config.max_query_length} characters",
            security_metadata,
        )

    if complexity.joins > config.max_joins:
        security_metadata["threat_level"] = SecurityThreatLevel.MEDIUM
        return (
            False,
            f"Too many joins: {complexity.joins}/{config.max_joins}",
            security_metadata,
        )

    if complexity.subqueries > config.max_subqueries:
        security_metadata["threat_level"] = SecurityThreatLevel.MEDIUM
        return (
            False,
            f"Too many subqueries: {complexity.subqueries}/{config.max_subqueries}",
            security_metadata,
        )

    return None


def _check_attack_patterns(
    query: str,
    database_name: str,
    connection_id: Optional[str],
    config: SecurityConfig,
    query_fingerprint: str,
    start_time: float,
    security_metadata: Dict[str, Any],
    detect_fn: Callable[[str], List[AttackPattern]],
    log_event_fn: Callable[[SecurityEvent], None],
) -> Optional[Tuple[bool, Optional[str], Dict[str, Any]]]:
    """Check for attack patterns, returning a result tuple if blocked."""
    if not config.enable_pattern_detection:
        return None

    security_metadata["checks_performed"].append("attack_pattern_detection")
    detected_patterns = detect_fn(query)
    security_metadata["attack_patterns"] = [p.value for p in detected_patterns]

    if detected_patterns and config.block_suspicious_patterns:
        for pattern in detected_patterns:
            log_event_fn(
                SecurityEvent(
                    timestamp=start_time,
                    event_type=SecurityEventType.INJECTION_ATTEMPT,
                    threat_level=SecurityThreatLevel.CRITICAL,
                    database_name=database_name,
                    connection_id=connection_id,
                    query_fingerprint=query_fingerprint,
                    query_text=query[:200],
                    attack_pattern=pattern,
                    message=f"Detected attack pattern: {pattern.value}",
                    metadata={"all_patterns": [p.value for p in detected_patterns]},
                )
            )

        security_metadata["threat_level"] = SecurityThreatLevel.CRITICAL
        return (
            False,
            f"Suspicious patterns detected: {', '.join(p.value for p in detected_patterns)}",
            security_metadata,
        )

    elif detected_patterns:
        # Log but don't block
        log_event_fn(
            SecurityEvent(
                timestamp=start_time,
                event_type=SecurityEventType.SUSPICIOUS_PATTERN,
                threat_level=SecurityThreatLevel.MEDIUM,
                database_name=database_name,
                connection_id=connection_id,
                query_fingerprint=query_fingerprint,
                query_text=query[:200],
                attack_pattern=detected_patterns[0],
                message=(
                    f"Suspicious patterns detected but allowed: "
                    f"{', '.join(p.value for p in detected_patterns)}"
                ),
                metadata={"all_patterns": [p.value for p in detected_patterns]},
            )
        )
        security_metadata["threat_level"] = SecurityThreatLevel.MEDIUM

    return None


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

    # Perform security validation
    is_valid, error_msg, security_metadata = validate_fn(
        query, database_name, connection_id
    )

    if not is_valid:
        raise SQLSecurityError(f"Security validation failed: {error_msg}")

    # Create execution context
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

        # Query executed successfully
        execution_time = time.time() - start_time
        execution_context["execution_metadata"]["execution_time"] = execution_time
        execution_context["execution_metadata"]["success"] = True

        # Log successful execution
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
        # Query execution failed
        execution_time = time.time() - start_time
        execution_context["execution_metadata"]["execution_time"] = execution_time
        execution_context["execution_metadata"]["success"] = False
        execution_context["execution_metadata"]["error"] = str(e)

        # Log failed execution
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
