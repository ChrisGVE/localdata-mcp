"""Security sub-package for LocalData MCP.

Re-exports all public symbols from the original security_manager module
to maintain backward compatibility.
"""

from .detection import (
    analyze_query_complexity,
    compile_attack_patterns,
    create_query_fingerprint,
    detect_attack_patterns,
)
from .manager import (
    SecurityManager,
    get_security_manager,
    initialize_security_manager,
    secure_query_execution,
    validate_query_security,
)
from .models import (
    AttackPattern,
    QueryComplexity,
    RateLimitState,
    SecurityConfig,
    SecurityEvent,
    SecurityEventType,
    SecurityThreatLevel,
)

__all__ = [
    # Models
    "SecurityThreatLevel",
    "SecurityEventType",
    "AttackPattern",
    "SecurityConfig",
    "SecurityEvent",
    "RateLimitState",
    "QueryComplexity",
    # Detection functions
    "compile_attack_patterns",
    "detect_attack_patterns",
    "create_query_fingerprint",
    "analyze_query_complexity",
    # Manager
    "SecurityManager",
    "get_security_manager",
    "initialize_security_manager",
    "validate_query_security",
    "secure_query_execution",
]
