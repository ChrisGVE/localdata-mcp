"""
ShimRegistry and Enhanced ShimAdapter Base Classes for LocalData MCP v2.0 Integration Framework.

This package provides the foundational shim adapter framework for pipeline component integration
with lifecycle management, dynamic discovery, and performance monitoring.

Key Features:
- Enhanced ShimAdapter with lifecycle hooks (initialize, activate, deactivate, cleanup)
- ShimRegistry for centralized adapter management and discovery
- Automatic adapter discovery and validation
- Performance monitoring and health checks
- Dependency injection and configuration support
- Plugin-style architecture for extensibility

Design Principles:
- Intention-Driven Interface: Simple adapter registration and discovery by analytical goals
- Context-Aware Composition: Adapters aware of pipeline context and dependencies
- Progressive Disclosure: Basic registration with advanced configuration options
- Streaming-First: Memory-efficient adapter operations and discovery
- Modular Domain Integration: Easy integration with existing domains and new additions
"""

from ._types import (
    AdapterLifecycleState,
    AdapterConfig,
    AdapterMetrics,
    HealthCheckResult,
)
from ._enhanced_adapter import EnhancedShimAdapter
from ._registry import ShimRegistry
from ._utils import (
    create_shim_registry,
    create_adapter_config,
    validate_adapter_dependencies,
    monitor_adapter_performance,
)

__all__ = [
    # Types and configuration
    "AdapterLifecycleState",
    "AdapterConfig",
    "AdapterMetrics",
    "HealthCheckResult",
    # Enhanced adapter
    "EnhancedShimAdapter",
    # Registry
    "ShimRegistry",
    # Factory functions
    "create_shim_registry",
    "create_adapter_config",
    # Utility functions
    "validate_adapter_dependencies",
    "monitor_adapter_performance",
]
