"""
EnhancedShimAdapter with lifecycle management and performance monitoring.

Extends the basic ShimAdapter interface with lifecycle hooks,
performance monitoring, health checks, and dependency management.
"""

import time
import logging
import threading
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor
import weakref

from ..interfaces import (
    ShimAdapter,
    ConversionRequest,
    ConversionResult,
    DataFormat,
    ValidationResult,
)
from ..base_adapters import BaseShimAdapter
from ....logging_manager import get_logger
from ._types import (
    AdapterLifecycleState,
    AdapterConfig,
    AdapterMetrics,
    HealthCheckResult,
)

logger = get_logger(__name__)


class EnhancedShimAdapter(ShimAdapter):
    """
    Enhanced ShimAdapter with lifecycle management and performance monitoring.

    Extends the basic ShimAdapter interface with:
    - Lifecycle hooks (initialize, activate, deactivate, cleanup)
    - Performance monitoring and metrics collection
    - Health check support
    - Configuration and dependency management
    - Context awareness for pipeline integration
    """

    def __init__(self, adapter_id: str, config: Optional[AdapterConfig] = None):
        """
        Initialize Enhanced ShimAdapter.

        Args:
            adapter_id: Unique identifier for this adapter
            config: Optional configuration for adapter operation
        """
        super().__init__(adapter_id)
        self.config = config or AdapterConfig(adapter_id=adapter_id)

        # Lifecycle management
        self._state = AdapterLifecycleState.UNINITIALIZED
        self._state_lock = threading.RLock()

        # Performance monitoring
        self.metrics = AdapterMetrics(adapter_id=adapter_id)
        self._metrics_lock = threading.Lock()

        # Dependencies and context
        self._dependencies: Dict[str, "EnhancedShimAdapter"] = {}
        self._dependents: Set["EnhancedShimAdapter"] = set()
        self._registry_ref: Optional[weakref.ReferenceType] = None

        # Health monitoring
        self._last_health_check = 0.0
        self._health_check_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{adapter_id}_health"
        )

        logger.debug(f"EnhancedShimAdapter '{adapter_id}' created")

    @property
    def state(self) -> AdapterLifecycleState:
        """Get current adapter lifecycle state."""
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: AdapterLifecycleState) -> None:
        """Set adapter lifecycle state with logging."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state

            if old_state != new_state:
                logger.info(
                    f"Adapter '{self.adapter_id}' state changed: {old_state.value} -> {new_state.value}"
                )

                # Update metrics
                if new_state == AdapterLifecycleState.ERROR:
                    with self._metrics_lock:
                        self.metrics.error_count += 1

    # Lifecycle Management Methods

    def initialize(self) -> bool:
        """
        Initialize the adapter with configuration and dependencies.

        Returns:
            True if initialization successful, False otherwise
        """
        with self._state_lock:
            if self._state != AdapterLifecycleState.UNINITIALIZED:
                logger.warning(
                    f"Adapter '{self.adapter_id}' already initialized (state: {self._state.value})"
                )
                return self._state != AdapterLifecycleState.ERROR

            try:
                logger.info(f"Initializing adapter '{self.adapter_id}'")
                start_time = time.time()

                # Call implementation-specific initialization
                if self._initialize_impl():
                    self._set_state(AdapterLifecycleState.INITIALIZED)

                    # Update metrics
                    with self._metrics_lock:
                        self.metrics.uptime_seconds = (
                            time.time() - self.metrics.creation_timestamp
                        )

                    initialization_time = time.time() - start_time
                    logger.info(
                        f"Adapter '{self.adapter_id}' initialized successfully in {initialization_time:.2f}s"
                    )
                    return True
                else:
                    self._set_state(AdapterLifecycleState.ERROR)
                    logger.error(f"Adapter '{self.adapter_id}' initialization failed")
                    return False

            except Exception as e:
                self._set_state(AdapterLifecycleState.ERROR)
                logger.error(f"Adapter '{self.adapter_id}' initialization error: {e}")
                return False

    def activate(self) -> bool:
        """
        Activate the adapter for processing requests.

        Returns:
            True if activation successful, False otherwise
        """
        with self._state_lock:
            if self._state not in [
                AdapterLifecycleState.INITIALIZED,
                AdapterLifecycleState.INACTIVE,
            ]:
                logger.warning(
                    f"Cannot activate adapter '{self.adapter_id}' in state: {self._state.value}"
                )
                return False

            try:
                logger.info(f"Activating adapter '{self.adapter_id}'")

                if self._activate_impl():
                    self._set_state(AdapterLifecycleState.ACTIVE)
                    logger.info(f"Adapter '{self.adapter_id}' activated successfully")
                    return True
                else:
                    self._set_state(AdapterLifecycleState.ERROR)
                    logger.error(f"Adapter '{self.adapter_id}' activation failed")
                    return False

            except Exception as e:
                self._set_state(AdapterLifecycleState.ERROR)
                logger.error(f"Adapter '{self.adapter_id}' activation error: {e}")
                return False

    def deactivate(self) -> bool:
        """
        Deactivate the adapter (stop processing but keep resources).

        Returns:
            True if deactivation successful, False otherwise
        """
        with self._state_lock:
            if self._state != AdapterLifecycleState.ACTIVE:
                logger.warning(
                    f"Adapter '{self.adapter_id}' not active (state: {self._state.value})"
                )
                return True

            try:
                logger.info(f"Deactivating adapter '{self.adapter_id}'")

                if self._deactivate_impl():
                    self._set_state(AdapterLifecycleState.INACTIVE)
                    logger.info(f"Adapter '{self.adapter_id}' deactivated successfully")
                    return True
                else:
                    logger.error(f"Adapter '{self.adapter_id}' deactivation failed")
                    return False

            except Exception as e:
                logger.error(f"Adapter '{self.adapter_id}' deactivation error: {e}")
                return False

    def cleanup(self) -> bool:
        """
        Cleanup adapter resources and dispose.

        Returns:
            True if cleanup successful, False otherwise
        """
        with self._state_lock:
            if self._state == AdapterLifecycleState.DISPOSED:
                return True

            try:
                logger.info(f"Cleaning up adapter '{self.adapter_id}'")

                # First deactivate if active
                if self._state == AdapterLifecycleState.ACTIVE:
                    self.deactivate()

                # Call implementation-specific cleanup
                if self._cleanup_impl():
                    self._set_state(AdapterLifecycleState.DISPOSED)

                    # Shutdown health check executor
                    self._health_check_executor.shutdown(wait=False)

                    logger.info(f"Adapter '{self.adapter_id}' cleaned up successfully")
                    return True
                else:
                    logger.error(f"Adapter '{self.adapter_id}' cleanup failed")
                    return False

            except Exception as e:
                logger.error(f"Adapter '{self.adapter_id}' cleanup error: {e}")
                return False

    # Performance Monitoring

    def _update_metrics(
        self, execution_time: float, success: bool, data_size_mb: float = 0.0
    ):
        """Update adapter performance metrics."""
        with self._metrics_lock:
            self.metrics.total_conversions += 1

            if success:
                self.metrics.successful_conversions += 1
            else:
                self.metrics.failed_conversions += 1

            self.metrics.last_execution_time = execution_time
            self.metrics.total_data_processed_mb += data_size_mb

            # Update average execution time (exponential moving average)
            alpha = 0.1  # Smoothing factor
            if self.metrics.average_execution_time == 0:
                self.metrics.average_execution_time = execution_time
            else:
                self.metrics.average_execution_time = (
                    alpha * execution_time
                    + (1 - alpha) * self.metrics.average_execution_time
                )

    def get_metrics(self) -> AdapterMetrics:
        """Get current adapter metrics."""
        with self._metrics_lock:
            return AdapterMetrics(**self.metrics.__dict__)

    # Health Checking

    def perform_health_check(self) -> HealthCheckResult:
        """
        Perform comprehensive health check on the adapter.

        Returns:
            Health check result with status and recommendations
        """
        check_time = time.time()

        with self._metrics_lock:
            self.metrics.health_check_count += 1
            self.metrics.last_health_check = check_time

        try:
            issues = []
            warnings = []

            # Check adapter state
            if self._state == AdapterLifecycleState.ERROR:
                issues.append("Adapter is in error state")
            elif self._state == AdapterLifecycleState.DISPOSED:
                issues.append("Adapter has been disposed")
            elif self._state == AdapterLifecycleState.UNINITIALIZED:
                issues.append("Adapter is not initialized")

            # Check error rate
            with self._metrics_lock:
                if self.metrics.total_conversions > 0:
                    error_rate = (
                        self.metrics.failed_conversions / self.metrics.total_conversions
                    )
                    if error_rate > 0.1:  # 10% error rate threshold
                        issues.append(f"High error rate: {error_rate:.2%}")
                    elif error_rate > 0.05:  # 5% warning threshold
                        warnings.append(f"Elevated error rate: {error_rate:.2%}")

                # Check recent errors
                if (
                    self.metrics.last_error_timestamp
                    and check_time - self.metrics.last_error_timestamp < 300
                ):  # 5 minutes
                    warnings.append("Recent error occurred")

                # Check performance
                if self.metrics.average_execution_time > 10.0:  # 10 seconds threshold
                    warnings.append(
                        f"Slow average execution time: {self.metrics.average_execution_time:.2f}s"
                    )

                metrics_snapshot = {
                    "total_conversions": self.metrics.total_conversions,
                    "error_rate": self.metrics.failed_conversions
                    / max(self.metrics.total_conversions, 1),
                    "average_execution_time": self.metrics.average_execution_time,
                    "uptime_seconds": check_time - self.metrics.creation_timestamp,
                }

            # Call implementation-specific health check
            impl_result = self._health_check_impl()
            if impl_result:
                issues.extend(impl_result.get("issues", []))
                warnings.extend(impl_result.get("warnings", []))
                metrics_snapshot.update(impl_result.get("metrics", {}))

            is_healthy = len(issues) == 0
            status = "healthy" if is_healthy else "unhealthy"

            return HealthCheckResult(
                adapter_id=self.adapter_id,
                is_healthy=is_healthy,
                status=status,
                issues=issues,
                warnings=warnings,
                metrics_snapshot=metrics_snapshot,
                check_timestamp=check_time,
            )

        except Exception as e:
            logger.error(f"Health check failed for adapter '{self.adapter_id}': {e}")
            return HealthCheckResult(
                adapter_id=self.adapter_id,
                is_healthy=False,
                status="health_check_failed",
                issues=[f"Health check error: {e}"],
                check_timestamp=check_time,
            )

    # Dependency Management

    def add_dependency(self, adapter: "EnhancedShimAdapter") -> None:
        """Add a dependency on another adapter."""
        if adapter.adapter_id in self._dependencies:
            return

        self._dependencies[adapter.adapter_id] = adapter
        adapter._dependents.add(self)
        logger.debug(f"Added dependency: '{self.adapter_id}' -> '{adapter.adapter_id}'")

    def remove_dependency(self, adapter: "EnhancedShimAdapter") -> None:
        """Remove a dependency on another adapter."""
        if adapter.adapter_id in self._dependencies:
            del self._dependencies[adapter.adapter_id]
            adapter._dependents.discard(self)
            logger.debug(
                f"Removed dependency: '{self.adapter_id}' -> '{adapter.adapter_id}'"
            )

    def get_dependencies(self) -> List["EnhancedShimAdapter"]:
        """Get list of adapter dependencies."""
        return list(self._dependencies.values())

    def get_dependents(self) -> List["EnhancedShimAdapter"]:
        """Get list of adapters that depend on this one."""
        return list(self._dependents)

    # Enhanced Conversion with Monitoring

    def convert(self, request: ConversionRequest) -> ConversionResult:
        """
        Perform conversion with enhanced monitoring and error handling.

        Args:
            request: Conversion request

        Returns:
            Conversion result with enhanced metadata
        """
        if self._state != AdapterLifecycleState.ACTIVE:
            logger.error(
                f"Adapter '{self.adapter_id}' not active for conversion (state: {self._state.value})"
            )
            return ConversionResult(
                converted_data=request.source_data,
                success=False,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.source_format,
                errors=[f"Adapter not active (state: {self._state.value})"],
                request_id=request.request_id,
            )

        start_time = time.time()
        data_size_mb = 0.0

        try:
            # Estimate data size for metrics
            if hasattr(request.source_data, "memory_usage"):
                data_size_mb = request.source_data.memory_usage(deep=True).sum() / (
                    1024 * 1024
                )
            elif hasattr(request.source_data, "nbytes"):
                data_size_mb = request.source_data.nbytes / (1024 * 1024)

            # Call the actual conversion implementation
            result = self._convert_impl(request)

            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, result.success, data_size_mb)

            # Enhance result metadata
            if not hasattr(result, "performance_metrics"):
                result.performance_metrics = {}

            result.performance_metrics.update(
                {
                    "adapter_id": self.adapter_id,
                    "adapter_state": self._state.value,
                    "execution_time": execution_time,
                    "data_size_mb": data_size_mb,
                }
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, False, data_size_mb)

            # Record error
            with self._metrics_lock:
                self.metrics.last_error = str(e)
                self.metrics.last_error_timestamp = time.time()

            logger.error(f"Conversion failed in adapter '{self.adapter_id}': {e}")

            return ConversionResult(
                converted_data=request.source_data,
                success=False,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.source_format,
                errors=[f"Conversion error: {e}"],
                performance_metrics={
                    "adapter_id": self.adapter_id,
                    "execution_time": execution_time,
                    "error": str(e),
                },
                request_id=request.request_id,
                execution_time=execution_time,
            )

    # Abstract methods for implementation

    def _initialize_impl(self) -> bool:
        """Implementation-specific initialization logic."""
        return True

    def _activate_impl(self) -> bool:
        """Implementation-specific activation logic."""
        return True

    def _deactivate_impl(self) -> bool:
        """Implementation-specific deactivation logic."""
        return True

    def _cleanup_impl(self) -> bool:
        """Implementation-specific cleanup logic."""
        return True

    def _health_check_impl(self) -> Optional[Dict[str, Any]]:
        """Implementation-specific health check logic."""
        return None

    def _convert_impl(self, request: ConversionRequest) -> ConversionResult:
        """Implementation-specific conversion logic."""
        # Default to base adapter behavior
        if hasattr(self, "can_convert") and hasattr(self, "estimate_cost"):
            # This adapter extends from existing base classes
            return super().convert(request)
        else:
            # Basic pass-through implementation
            return ConversionResult(
                converted_data=request.source_data,
                success=True,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.source_format,
                request_id=request.request_id,
            )
