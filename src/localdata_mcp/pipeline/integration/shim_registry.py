"""
ShimRegistry and Enhanced ShimAdapter Base Classes for LocalData MCP v2.0 Integration Framework.

This module provides the foundational shim adapter framework for pipeline component integration
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

import time
import logging
import importlib
import inspect
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

from .interfaces import (
    ShimAdapter, ConversionRequest, ConversionResult, ConversionCost,
    DataFormat, ValidationResult, ConversionRegistry
)
from .base_adapters import BaseShimAdapter
from .compatibility_matrix import PipelineCompatibilityMatrix
from ...logging_manager import get_logger

logger = get_logger(__name__)


class AdapterLifecycleState(Enum):
    """Adapter lifecycle states for managing adapter operations."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISPOSED = "disposed"


@dataclass
class AdapterConfig:
    """Configuration for adapter initialization and operation."""
    adapter_id: str
    config_params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    initialization_order: int = 100  # Lower values initialize first
    enable_metrics: bool = True
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 60
    max_initialization_time_seconds: int = 30
    auto_restart_on_error: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterMetrics:
    """Performance and operational metrics for adapters."""
    adapter_id: str
    total_conversions: int = 0
    successful_conversions: int = 0
    failed_conversions: int = 0
    average_execution_time: float = 0.0
    last_execution_time: Optional[float] = None
    total_data_processed_mb: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_timestamp: Optional[float] = None
    health_check_count: int = 0
    last_health_check: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    uptime_seconds: float = 0.0
    creation_timestamp: float = field(default_factory=time.time)


@dataclass  
class HealthCheckResult:
    """Result of adapter health check."""
    adapter_id: str
    is_healthy: bool
    status: str
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    check_timestamp: float = field(default_factory=time.time)


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
        self._dependencies: Dict[str, 'EnhancedShimAdapter'] = {}
        self._dependents: Set['EnhancedShimAdapter'] = set()
        self._registry_ref: Optional[weakref.ReferenceType] = None
        
        # Health monitoring
        self._last_health_check = 0.0
        self._health_check_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{adapter_id}_health")
        
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
                logger.info(f"Adapter '{self.adapter_id}' state changed: {old_state.value} -> {new_state.value}")
                
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
                logger.warning(f"Adapter '{self.adapter_id}' already initialized (state: {self._state.value})")
                return self._state != AdapterLifecycleState.ERROR
            
            try:
                logger.info(f"Initializing adapter '{self.adapter_id}'")
                start_time = time.time()
                
                # Call implementation-specific initialization
                if self._initialize_impl():
                    self._set_state(AdapterLifecycleState.INITIALIZED)
                    
                    # Update metrics
                    with self._metrics_lock:
                        self.metrics.uptime_seconds = time.time() - self.metrics.creation_timestamp
                    
                    initialization_time = time.time() - start_time
                    logger.info(f"Adapter '{self.adapter_id}' initialized successfully in {initialization_time:.2f}s")
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
            if self._state not in [AdapterLifecycleState.INITIALIZED, AdapterLifecycleState.INACTIVE]:
                logger.warning(f"Cannot activate adapter '{self.adapter_id}' in state: {self._state.value}")
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
                logger.warning(f"Adapter '{self.adapter_id}' not active (state: {self._state.value})")
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
    
    def _update_metrics(self, execution_time: float, success: bool, data_size_mb: float = 0.0):
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
                    alpha * execution_time + 
                    (1 - alpha) * self.metrics.average_execution_time
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
                    error_rate = self.metrics.failed_conversions / self.metrics.total_conversions
                    if error_rate > 0.1:  # 10% error rate threshold
                        issues.append(f"High error rate: {error_rate:.2%}")
                    elif error_rate > 0.05:  # 5% warning threshold
                        warnings.append(f"Elevated error rate: {error_rate:.2%}")
                
                # Check recent errors
                if (self.metrics.last_error_timestamp and
                    check_time - self.metrics.last_error_timestamp < 300):  # 5 minutes
                    warnings.append("Recent error occurred")
                
                # Check performance
                if self.metrics.average_execution_time > 10.0:  # 10 seconds threshold
                    warnings.append(f"Slow average execution time: {self.metrics.average_execution_time:.2f}s")
                
                metrics_snapshot = {
                    'total_conversions': self.metrics.total_conversions,
                    'error_rate': self.metrics.failed_conversions / max(self.metrics.total_conversions, 1),
                    'average_execution_time': self.metrics.average_execution_time,
                    'uptime_seconds': check_time - self.metrics.creation_timestamp
                }
            
            # Call implementation-specific health check
            impl_result = self._health_check_impl()
            if impl_result:
                issues.extend(impl_result.get('issues', []))
                warnings.extend(impl_result.get('warnings', []))
                metrics_snapshot.update(impl_result.get('metrics', {}))
            
            is_healthy = len(issues) == 0
            status = "healthy" if is_healthy else "unhealthy"
            
            return HealthCheckResult(
                adapter_id=self.adapter_id,
                is_healthy=is_healthy,
                status=status,
                issues=issues,
                warnings=warnings,
                metrics_snapshot=metrics_snapshot,
                check_timestamp=check_time
            )
            
        except Exception as e:
            logger.error(f"Health check failed for adapter '{self.adapter_id}': {e}")
            return HealthCheckResult(
                adapter_id=self.adapter_id,
                is_healthy=False,
                status="health_check_failed",
                issues=[f"Health check error: {e}"],
                check_timestamp=check_time
            )
    
    # Dependency Management
    
    def add_dependency(self, adapter: 'EnhancedShimAdapter') -> None:
        """Add a dependency on another adapter."""
        if adapter.adapter_id in self._dependencies:
            return
        
        self._dependencies[adapter.adapter_id] = adapter
        adapter._dependents.add(self)
        logger.debug(f"Added dependency: '{self.adapter_id}' -> '{adapter.adapter_id}'")
    
    def remove_dependency(self, adapter: 'EnhancedShimAdapter') -> None:
        """Remove a dependency on another adapter."""
        if adapter.adapter_id in self._dependencies:
            del self._dependencies[adapter.adapter_id]
            adapter._dependents.discard(self)
            logger.debug(f"Removed dependency: '{self.adapter_id}' -> '{adapter.adapter_id}'")
    
    def get_dependencies(self) -> List['EnhancedShimAdapter']:
        """Get list of adapter dependencies."""
        return list(self._dependencies.values())
    
    def get_dependents(self) -> List['EnhancedShimAdapter']:
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
            logger.error(f"Adapter '{self.adapter_id}' not active for conversion (state: {self._state.value})")
            return ConversionResult(
                converted_data=request.source_data,
                success=False,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.source_format,
                errors=[f"Adapter not active (state: {self._state.value})"],
                request_id=request.request_id
            )
        
        start_time = time.time()
        data_size_mb = 0.0
        
        try:
            # Estimate data size for metrics
            if hasattr(request.source_data, 'memory_usage'):
                data_size_mb = request.source_data.memory_usage(deep=True).sum() / (1024 * 1024)
            elif hasattr(request.source_data, 'nbytes'):
                data_size_mb = request.source_data.nbytes / (1024 * 1024)
            
            # Call the actual conversion implementation
            result = self._convert_impl(request)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, result.success, data_size_mb)
            
            # Enhance result metadata
            if not hasattr(result, 'performance_metrics'):
                result.performance_metrics = {}
            
            result.performance_metrics.update({
                'adapter_id': self.adapter_id,
                'adapter_state': self._state.value,
                'execution_time': execution_time,
                'data_size_mb': data_size_mb
            })
            
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
                    'adapter_id': self.adapter_id,
                    'execution_time': execution_time,
                    'error': str(e)
                },
                request_id=request.request_id,
                execution_time=execution_time
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
        if hasattr(self, 'can_convert') and hasattr(self, 'estimate_cost'):
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
                request_id=request.request_id
            )


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
    
    def __init__(self, 
                 compatibility_matrix: Optional[PipelineCompatibilityMatrix] = None,
                 enable_auto_discovery: bool = True,
                 max_concurrent_health_checks: int = 5,
                 health_check_interval_seconds: int = 300):
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
            thread_name_prefix="registry_health"
        )
        self._last_health_check = 0.0
        
        # Statistics
        self._stats = {
            'registrations': 0,
            'discoveries': 0,
            'health_checks': 0,
            'adapter_errors': 0
        }
        
        logger.info("ShimRegistry initialized",
                   auto_discovery=enable_auto_discovery,
                   health_check_interval=health_check_interval_seconds)
    
    def register_adapter(self, adapter: EnhancedShimAdapter, config: Optional[AdapterConfig] = None) -> None:
        """
        Register a shim adapter with the registry.
        
        Args:
            adapter: Adapter instance to register
            config: Optional configuration for the adapter
        """
        with self._adapter_lock:
            if adapter.adapter_id in self._adapters:
                logger.warning(f"Adapter '{adapter.adapter_id}' already registered, replacing")
                # Cleanup existing adapter
                existing = self._adapters[adapter.adapter_id]
                existing.cleanup()
            
            self._adapters[adapter.adapter_id] = adapter
            
            if config:
                self._adapter_configs[adapter.adapter_id] = config
                adapter.config = config
            
            # Set registry reference
            adapter._registry_ref = weakref.ref(self)
            
            self._stats['registrations'] += 1
            
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
                logger.error(f"Cannot unregister adapter '{adapter_id}' - has dependents: {dependent_ids}")
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
                adapter for adapter in self._adapters.values()
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
                self._adapters.values(),
                key=lambda a: a.config.initialization_order
            )
            
            for adapter in sorted_adapters:
                try:
                    results[adapter.adapter_id] = adapter.initialize()
                except Exception as e:
                    logger.error(f"Failed to initialize adapter '{adapter.adapter_id}': {e}")
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
                        logger.error(f"Failed to activate adapter '{adapter.adapter_id}': {e}")
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
                    logger.error(f"Failed to shutdown adapter '{adapter.adapter_id}': {e}")
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
                    if (inspect.isclass(obj) and
                        issubclass(obj, EnhancedShimAdapter) and
                        obj is not EnhancedShimAdapter):
                        
                        discovered[name] = obj
                        self._adapter_classes[name] = obj
                        
                        logger.debug(f"Discovered adapter class: {name}")
                
            except Exception as e:
                logger.error(f"Failed to discover adapters in {module_path}: {e}")
        
        self._stats['discoveries'] += 1
        logger.info(f"Discovered {len(discovered)} adapter classes")
        
        return discovered
    
    def create_adapter_from_class(self, 
                                 class_name: str, 
                                 adapter_id: str,
                                 config: Optional[AdapterConfig] = None,
                                 **kwargs) -> Optional[EnhancedShimAdapter]:
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
            logger.error(f"Failed to create adapter '{adapter_id}' from class '{class_name}': {e}")
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
            if adapter.state not in [AdapterLifecycleState.DISPOSED, AdapterLifecycleState.UNINITIALIZED]:
                future = self._health_check_executor.submit(adapter.perform_health_check)
                future_to_adapter[future] = adapter
        
        for future in as_completed(future_to_adapter):
            adapter = future_to_adapter[future]
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results[adapter.adapter_id] = result
            except Exception as e:
                logger.error(f"Health check failed for adapter '{adapter.adapter_id}': {e}")
                results[adapter.adapter_id] = HealthCheckResult(
                    adapter_id=adapter.adapter_id,
                    is_healthy=False,
                    status="health_check_error",
                    issues=[f"Health check error: {e}"]
                )
        
        self._stats['health_checks'] += 1
        self._last_health_check = time.time()
        
        # Log summary
        healthy_count = sum(1 for r in results.values() if r.is_healthy)
        logger.info(f"Health check completed: {healthy_count}/{len(results)} adapters healthy")
        
        return results
    
    # Conversion Path Discovery
    
    def find_conversion_path(self, 
                           source_format: DataFormat, 
                           target_format: DataFormat) -> Optional[ConversionPath]:
        """Find optimal conversion path between formats."""
        if self.compatibility_matrix:
            return self.compatibility_matrix.find_conversion_path(source_format, target_format)
        return None
    
    def get_compatible_adapters(self, 
                              request: ConversionRequest) -> List[Tuple[EnhancedShimAdapter, float]]:
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
                        logger.error(f"Error checking compatibility for adapter '{adapter.adapter_id}': {e}")
        
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
                'total_adapters': len(self._adapters),
                'adapter_states': adapter_states,
                'discovery_paths': len(self._discovery_paths),
                'discovered_classes': len(self._adapter_classes),
                'total_conversions': total_conversions,
                'total_errors': total_errors,
                'error_rate': total_errors / max(total_conversions, 1),
                'registry_stats': self._stats.copy(),
                'last_health_check': self._last_health_check
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown_all_adapters()
        self._health_check_executor.shutdown(wait=True)


# Factory Functions

def create_shim_registry(compatibility_matrix: Optional[PipelineCompatibilityMatrix] = None,
                        **kwargs) -> ShimRegistry:
    """Create a ShimRegistry with standard configuration."""
    return ShimRegistry(compatibility_matrix=compatibility_matrix, **kwargs)


def create_adapter_config(adapter_id: str, **kwargs) -> AdapterConfig:
    """Create an AdapterConfig with sensible defaults."""
    return AdapterConfig(adapter_id=adapter_id, **kwargs)


# Utility Functions

def validate_adapter_dependencies(adapters: List[EnhancedShimAdapter]) -> ValidationResult:
    """
    Validate adapter dependency graph for cycles and missing dependencies.
    
    Args:
        adapters: List of adapters to validate
        
    Returns:
        Validation result with dependency issues
    """
    errors = []
    warnings = []
    adapter_ids = {a.adapter_id for a in adapters}
    
    # Check for missing dependencies
    for adapter in adapters:
        for dep_id in adapter.config.dependencies:
            if dep_id not in adapter_ids:
                errors.append(f"Adapter '{adapter.adapter_id}' has missing dependency: '{dep_id}'")
    
    # Check for circular dependencies using DFS
    def has_cycle(adapter_id: str, visited: Set[str], rec_stack: Set[str]) -> bool:
        visited.add(adapter_id)
        rec_stack.add(adapter_id)
        
        adapter = next((a for a in adapters if a.adapter_id == adapter_id), None)
        if not adapter:
            return False
        
        for dep_id in adapter.config.dependencies:
            if dep_id not in visited:
                if has_cycle(dep_id, visited, rec_stack):
                    return True
            elif dep_id in rec_stack:
                return True
        
        rec_stack.remove(adapter_id)
        return False
    
    visited = set()
    for adapter in adapters:
        if adapter.adapter_id not in visited:
            if has_cycle(adapter.adapter_id, visited, set()):
                errors.append(f"Circular dependency detected involving adapter '{adapter.adapter_id}'")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        score=1.0 if len(errors) == 0 else 0.0,
        errors=errors,
        warnings=warnings,
        details={'total_adapters_checked': len(adapters)}
    )


def monitor_adapter_performance(adapter: EnhancedShimAdapter, 
                               duration_seconds: int = 60) -> Dict[str, Any]:
    """
    Monitor adapter performance over a specified duration.
    
    Args:
        adapter: Adapter to monitor
        duration_seconds: Monitoring duration in seconds
        
    Returns:
        Performance monitoring report
    """
    start_time = time.time()
    start_metrics = adapter.get_metrics()
    
    # Wait for monitoring period
    time.sleep(duration_seconds)
    
    end_time = time.time()
    end_metrics = adapter.get_metrics()
    
    # Calculate deltas
    conversion_delta = end_metrics.total_conversions - start_metrics.total_conversions
    error_delta = end_metrics.failed_conversions - start_metrics.failed_conversions
    data_delta = end_metrics.total_data_processed_mb - start_metrics.total_data_processed_mb
    
    actual_duration = end_time - start_time
    
    return {
        'monitoring_duration': actual_duration,
        'conversions_per_second': conversion_delta / actual_duration,
        'errors_per_second': error_delta / actual_duration,
        'data_throughput_mb_per_second': data_delta / actual_duration,
        'current_error_rate': error_delta / max(conversion_delta, 1),
        'current_avg_execution_time': end_metrics.average_execution_time,
        'adapter_state': adapter.state.value,
        'snapshot': {
            'start_metrics': start_metrics,
            'end_metrics': end_metrics
        }
    }