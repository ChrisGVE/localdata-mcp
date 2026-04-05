"""Backward-compatibility shim — real implementation in security/ sub-package."""

from .security import (  # noqa: F401
    AttackPattern,
    QueryComplexity,
    RateLimitState,
    SecurityConfig,
    SecurityEvent,
    SecurityEventType,
    SecurityManager,
    SecurityThreatLevel,
    get_security_manager,
    initialize_security_manager,
    secure_query_execution,
    validate_query_security,
)

__all__ = [
    "AttackPattern",
    "QueryComplexity",
    "RateLimitState",
    "SecurityConfig",
    "SecurityEvent",
    "SecurityEventType",
    "SecurityManager",
    "SecurityThreatLevel",
    "get_security_manager",
    "initialize_security_manager",
    "secure_query_execution",
    "validate_query_security",
]
