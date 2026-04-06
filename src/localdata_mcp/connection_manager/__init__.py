"""Enhanced Database Connection Management for LocalData MCP.

Provides advanced connection management with per-database configuration,
connection pooling, health monitoring, and resource management. Integrates
with the configuration system and timeout management.
"""

from typing import Optional

from .manager import EnhancedConnectionManager
from .models import (
    ConnectionMetrics,
    ConnectionState,
    HealthCheckResult,
    ResourceLimit,
    ResourceType,
)

# Global enhanced connection manager instance
_enhanced_connection_manager: Optional[EnhancedConnectionManager] = None


def get_enhanced_connection_manager() -> EnhancedConnectionManager:
    """Get or create global enhanced connection manager instance.

    Returns:
        Global connection manager.
    """
    global _enhanced_connection_manager
    if _enhanced_connection_manager is None:
        _enhanced_connection_manager = EnhancedConnectionManager()
    return _enhanced_connection_manager


def initialize_enhanced_connection_manager() -> EnhancedConnectionManager:
    """Initialize a new global enhanced connection manager instance.

    Returns:
        New connection manager.
    """
    global _enhanced_connection_manager
    if _enhanced_connection_manager is not None:
        _enhanced_connection_manager.close_all()
    _enhanced_connection_manager = EnhancedConnectionManager()
    return _enhanced_connection_manager


__all__ = [
    "ConnectionMetrics",
    "ConnectionState",
    "EnhancedConnectionManager",
    "HealthCheckResult",
    "ResourceLimit",
    "ResourceType",
    "get_enhanced_connection_manager",
    "initialize_enhanced_connection_manager",
]
