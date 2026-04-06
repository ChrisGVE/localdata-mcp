"""
ShimRegistry for centralized adapter management with dynamic discovery and lifecycle management.

Provides adapter registration, lifecycle management, automatic discovery,
health monitoring, conversion path discovery, and statistics.
"""

import importlib
import inspect
import logging
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from ....logging_manager import get_logger
from ..compatibility_matrix import PipelineCompatibilityMatrix
from ..interfaces import (
    ConversionPath,
    ConversionRegistry,
    ConversionRequest,
    ConversionResult,
    DataFormat,
)
from ._enhanced_adapter import EnhancedShimAdapter
from ._types import AdapterConfig, AdapterLifecycleState, HealthCheckResult

logger = get_logger(__name__)


class ShimRegistry(ConversionRegistry):
    """
    Central registry for managing shim adapters with dynamic discovery and lifecycle management.

    Features:
    - Dynamic adapter registration and discovery
    - Adapter lifecycle management (load, unload, reload)
    - Dependency resolution between adapters
    - Performance monitoring and health checks
    - Plugin-style architecture for extensibility
    - Integration with PipelineCompatibilityMatrix
    """

    def __init__(
        self,
        compatibility_matrix: Optional[PipelineCompatibilityMatrix] = None,
        enable_auto_discovery: bool = True,
        max_concurrent_health_checks: int = 5,
        health_check_interval_seconds: int = 300,
    ):
        """
        Initialize ShimRegistry.

        Args:
            compatibility_matrix: Optional compatibility matrix for pathway discovery
            enable_auto_discovery: Enable automatic adapter discovery
            max_concurrent_health_checks: Maximum concurrent health check operations
            health_check_interval_seconds: Interval between automatic health checks
        """
        self.compatibility_matrix = compatibility_matrix
        self.enable_auto_discovery = enable_auto_discovery
        self.max_concurrent_health_checks = max_concurrent_health_checks
        self.health_check_interval_seconds = health_check_interval_seconds

        # Adapter storage
        self._adapters: Dict[str, EnhancedShimAdapter] = {}
        self._adapter_configs: Dict[str, AdapterConfig] = {}
        self._adapter_lock = threading.RLock()

        # Discovery and loading
        self._discovery_paths: List[str] = []
        self._adapter_classes: Dict[str, Type[EnhancedShimAdapter]] = {}

        # Health monitoring
        self._health_check_executor = ThreadPoolExecutor(
            max_workers=max_concurrent_health_checks,
            thread_name_prefix="registry_health",
        )
        self._last_health_check = 0.0

        # Statistics
        self._stats = {
            "registrations": 0,
            "discoveries": 0,
            "health_checks": 0,
            "adapter_errors": 0,
        }

        logger.info(
            "ShimRegistry initialized",
            auto_discovery=enable_auto_discovery,
            health_check_interval=health_check_interval_seconds,
        )

    def register_adapter(
        self, adapter: EnhancedShimAdapter, config: Optional[AdapterConfig] = None
    ) -> None:
        """
        Register a shim adapter with the registry.

        Args:
            adapter: Adapter instance to register
            config: Optional configuration for the adapter
        """
        with self._adapter_lock:
            if adapter.adapter_id in self._adapters:
                logger.warning(
                    f"Adapter '{adapter.adapter_id}' already registered, replacing"
                )
                # Cleanup existing adapter
                existing = self._adapters[adapter.adapter_id]
                existing.cleanup()

            self._adapters[adapter.adapter_id] = adapter

            if config:
                self._adapter_configs[adapter.adapter_id] = config
                adapter.config = config

            # Set registry reference
            adapter._registry_ref = weakref.ref(self)

            self._stats["registrations"] += 1

            logger.info(f"Registered adapter '{adapter.adapter_id}'")

    def unregister_adapter(self, adapter_id: str) -> bool:
        """
        Unregister and cleanup an adapter.

        Args:
            adapter_id: ID of adapter to unregister

        Returns:
            True if unregistered successfully, False otherwise
        """
        with self._adapter_lock:
            if adapter_id not in self._adapters:
                logger.warning(f"Adapter '{adapter_id}' not found for unregistration")
                return False

            adapter = self._adapters[adapter_id]

            # Check for dependents
            dependents = adapter.get_dependents()
            if dependents:
                dependent_ids = [d.adapter_id for d in dependents]
                logger.error(
                    f"Cannot unregister adapter '{adapter_id}' - has dependents: {dependent_ids}"
                )
                return False

            # Cleanup adapter
            try:
                adapter.cleanup()
            except Exception as e:
                logger.error(f"Error during adapter cleanup: {e}")

            # Remove from registry
            del self._adapters[adapter_id]
            self._adapter_configs.pop(adapter_id, None)

            logger.info(f"Unregistered adapter '{adapter_id}'")
            return True

    def get_adapter(self, adapter_id: str) -> Optional[EnhancedShimAdapter]:
        """Get adapter by ID."""
        with self._adapter_lock:
            return self._adapters.get(adapter_id)

    def list_adapters(self) -> List[str]:
        """Get list of registered adapter IDs."""
        with self._adapter_lock:
            return list(self._adapters.keys())

    def get_active_adapters(self) -> List[EnhancedShimAdapter]:
        """Get list of active adapters."""
        with self._adapter_lock:
            return [
                adapter
                for adapter in self._adapters.values()
                if adapter.state == AdapterLifecycleState.ACTIVE
            ]

    # Lifecycle Management

    def initialize_all_adapters(self) -> Dict[str, bool]:
        """
        Initialize all registered adapters in dependency order.

        Returns:
            Dictionary mapping adapter IDs to initialization success
        """
        results = {}

        with self._adapter_lock:
            # Sort adapters by initialization order
            sorted_adapters = sorted(
                self._adapters.values(), key=lambda a: a.config.initialization_order
            )

            for adapter in sorted_adapters:
                try:
                    results[adapter.adapter_id] = adapter.initialize()
                except Exception as e:
                    logger.error(
                        f"Failed to initialize adapter '{adapter.adapter_id}': {e}"
                    )
                    results[adapter.adapter_id] = False

        successful = sum(1 for success in results.values() if success)
        logger.info(f"Initialized {successful}/{len(results)} adapters")

        return results

    def activate_all_adapters(self) -> Dict[str, bool]:
        """
        Activate all initialized adapters.

        Returns:
            Dictionary mapping adapter IDs to activation success
        """
        results = {}

        with self._adapter_lock:
            for adapter in self._adapters.values():
                if adapter.state == AdapterLifecycleState.INITIALIZED:
                    try:
                        results[adapter.adapter_id] = adapter.activate()
                    except Exception as e:
                        logger.error(
                            f"Failed to activate adapter '{adapter.adapter_id}': {e}"
                        )
                        results[adapter.adapter_id] = False
                else:
                    results[adapter.adapter_id] = False

        successful = sum(1 for success in results.values() if success)
        logger.info(f"Activated {successful}/{len(results)} adapters")

        return results

    def shutdown_all_adapters(self) -> Dict[str, bool]:
        """
        Shutdown all adapters gracefully.

        Returns:
            Dictionary mapping adapter IDs to shutdown success
        """
        results = {}

        with self._adapter_lock:
            # Shutdown in reverse dependency order
            adapters_to_shutdown = list(self._adapters.values())
            adapters_to_shutdown.reverse()

            for adapter in adapters_to_shutdown:
                try:
                    results[adapter.adapter_id] = adapter.cleanup()
                except Exception as e:
                    logger.error(
                        f"Failed to shutdown adapter '{adapter.adapter_id}': {e}"
                    )
                    results[adapter.adapter_id] = False

        successful = sum(1 for success in results.values() if success)
        logger.info(f"Shutdown {successful}/{len(results)} adapters")

        return results

    # Automatic Discovery

    def add_discovery_path(self, module_path: str) -> None:
        """
        Add a module path for automatic adapter discovery.

        Args:
            module_path: Python module path to scan for adapters
        """
        if module_path not in self._discovery_paths:
            self._discovery_paths.append(module_path)
            logger.info(f"Added discovery path: {module_path}")

    def discover_adapters(self) -> Dict[str, Type[EnhancedShimAdapter]]:
        """
        Discover adapter classes from configured paths.

        Returns:
            Dictionary mapping adapter class names to types
        """
        discovered = {}

        for module_path in self._discovery_paths:
            try:
                module = importlib.import_module(module_path)

                # Find adapter classes in module
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, EnhancedShimAdapter)
                        and obj is not EnhancedShimAdapter
                    ):
                        discovered[name] = obj
                        self._adapter_classes[name] = obj

                        logger.debug(f"Discovered adapter class: {name}")

            except Exception as e:
                logger.error(f"Failed to discover adapters in {module_path}: {e}")

        self._stats["discoveries"] += 1
        logger.info(f"Discovered {len(discovered)} adapter classes")

        return discovered

    def create_adapter_from_class(
        self,
        class_name: str,
        adapter_id: str,
        config: Optional[AdapterConfig] = None,
        **kwargs,
    ) -> Optional[EnhancedShimAdapter]:
        """
        Create adapter instance from discovered class.

        Args:
            class_name: Name of discovered adapter class
            adapter_id: Unique ID for adapter instance
            config: Optional configuration
            **kwargs: Additional arguments for adapter constructor

        Returns:
            Created adapter instance or None if failed
        """
        if class_name not in self._adapter_classes:
            logger.error(f"Adapter class '{class_name}' not found")
            return None

        try:
            adapter_class = self._adapter_classes[class_name]
            adapter = adapter_class(adapter_id=adapter_id, config=config, **kwargs)

            logger.info(f"Created adapter '{adapter_id}' from class '{class_name}'")
            return adapter

        except Exception as e:
            logger.error(
                f"Failed to create adapter '{adapter_id}' from class '{class_name}': {e}"
            )
            return None

    # Health Monitoring

    def perform_health_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Perform health checks on all registered adapters.

        Returns:
            Dictionary mapping adapter IDs to health check results
        """
        results = {}

        with self._adapter_lock:
            adapters = list(self._adapters.values())

        # Perform health checks concurrently
        future_to_adapter = {}

        for adapter in adapters:
            if adapter.state not in [
                AdapterLifecycleState.DISPOSED,
                AdapterLifecycleState.UNINITIALIZED,
            ]:
                future = self._health_check_executor.submit(
                    adapter.perform_health_check
                )
                future_to_adapter[future] = adapter

        for future in as_completed(future_to_adapter):
            adapter = future_to_adapter[future]
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results[adapter.adapter_id] = result
            except Exception as e:
                logger.error(
                    f"Health check failed for adapter '{adapter.adapter_id}': {e}"
                )
                results[adapter.adapter_id] = HealthCheckResult(
                    adapter_id=adapter.adapter_id,
                    is_healthy=False,
                    status="health_check_error",
                    issues=[f"Health check error: {e}"],
                )

        self._stats["health_checks"] += 1
        self._last_health_check = time.time()

        # Log summary
        healthy_count = sum(1 for r in results.values() if r.is_healthy)
        logger.info(
            f"Health check completed: {healthy_count}/{len(results)} adapters healthy"
        )

        return results

    # Conversion Path Discovery

    def find_conversion_path(
        self, source_format: DataFormat, target_format: DataFormat
    ) -> Optional[ConversionPath]:
        """Find optimal conversion path between formats."""
        if self.compatibility_matrix:
            return self.compatibility_matrix.find_conversion_path(
                source_format, target_format
            )
        return None

    def get_compatible_adapters(
        self, request: ConversionRequest
    ) -> List[Tuple[EnhancedShimAdapter, float]]:
        """
        Get adapters that can handle the conversion request with confidence scores.

        Args:
            request: Conversion request to evaluate

        Returns:
            List of (adapter, confidence) tuples sorted by confidence
        """
        compatible = []

        with self._adapter_lock:
            for adapter in self._adapters.values():
                if adapter.state == AdapterLifecycleState.ACTIVE:
                    try:
                        confidence = adapter.can_convert(request)
                        if confidence > 0:
                            compatible.append((adapter, confidence))
                    except Exception as e:
                        logger.error(
                            f"Error checking compatibility for adapter '{adapter.adapter_id}': {e}"
                        )

        # Sort by confidence descending
        compatible.sort(key=lambda x: x[1], reverse=True)

        return compatible

    # Statistics and Monitoring

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics and health information."""
        with self._adapter_lock:
            adapter_states = {}
            for state in AdapterLifecycleState:
                count = sum(1 for a in self._adapters.values() if a.state == state)
                adapter_states[state.value] = count

            total_conversions = sum(
                a.metrics.total_conversions for a in self._adapters.values()
            )
            total_errors = sum(
                a.metrics.failed_conversions for a in self._adapters.values()
            )

            return {
                "total_adapters": len(self._adapters),
                "adapter_states": adapter_states,
                "discovery_paths": len(self._discovery_paths),
                "discovered_classes": len(self._adapter_classes),
                "total_conversions": total_conversions,
                "total_errors": total_errors,
                "error_rate": total_errors / max(total_conversions, 1),
                "registry_stats": self._stats.copy(),
                "last_health_check": self._last_health_check,
            }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown_all_adapters()
        self._health_check_executor.shutdown(wait=True)
