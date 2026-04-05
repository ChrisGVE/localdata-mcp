"""Security sub-package for LocalData MCP.

Re-exports all public symbols from the original security_manager module
to maintain backward compatibility.
"""

from .models import (
    SecurityThreatLevel,
    SecurityEventType,
    AttackPattern,
    SecurityConfig,
    SecurityEvent,
    RateLimitState,
    QueryComplexity,
)
from .detection import (
    compile_attack_patterns,
    detect_attack_patterns,
    create_query_fingerprint,
    analyze_query_complexity,
)
from .manager import (
    SecurityManager,
    get_security_manager,
    initialize_security_manager,
    validate_query_security,
    secure_query_execution,
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
