"""
Unit tests for ShimRegistry and EnhancedShimAdapter implementation.

Tests cover:
- Enhanced ShimAdapter lifecycle management
- ShimRegistry adapter management and discovery
- Health monitoring and performance metrics
- Dependency resolution and validation
- Error handling and edge cases
"""

import time
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future
import pytest

from src.localdata_mcp.pipeline.integration.shim_registry import (
    EnhancedShimAdapter, ShimRegistry, AdapterConfig, AdapterMetrics,
    HealthCheckResult, AdapterLifecycleState, create_shim_registry,
    create_adapter_config, validate_adapter_dependencies, monitor_adapter_performance
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    ConversionRequest, ConversionResult, DataFormat, ConversionCost,
    ValidationResult
)
from src.localdata_mcp.pipeline.integration.compatibility_matrix import PipelineCompatibilityMatrix


class TestAdapterConfig:
    """Test AdapterConfig functionality."""
    
    def test_adapter_config_creation(self):
        """Test creating AdapterConfig with defaults."""
        config = AdapterConfig(adapter_id="test_adapter")
        
        assert config.adapter_id == "test_adapter"
        assert config.config_params == {}
        assert config.dependencies == []
        assert config.initialization_order == 100
        assert config.enable_metrics is True
        assert config.enable_health_checks is True
        assert config.health_check_interval_seconds == 60
        assert config.max_initialization_time_seconds == 30
        assert config.auto_restart_on_error is True
        assert config.metadata == {}
    
    def test_adapter_config_with_custom_values(self):
        """Test creating AdapterConfig with custom values."""
        config = AdapterConfig(
            adapter_id="custom_adapter",
            config_params={"param1": "value1"},
            dependencies=["dep1", "dep2"],
            initialization_order=50,
            enable_metrics=False,
            metadata={"version": "1.0"}
        )
        
        assert config.adapter_id == "custom_adapter"
        assert config.config_params == {"param1": "value1"}
        assert config.dependencies == ["dep1", "dep2"]
        assert config.initialization_order == 50
        assert config.enable_metrics is False
        assert config.metadata == {"version": "1.0"}


class TestAdapterMetrics:
    """Test AdapterMetrics functionality."""
    
    def test_adapter_metrics_creation(self):
        """Test creating AdapterMetrics with defaults."""
        metrics = AdapterMetrics(adapter_id="test_adapter")
        
        assert metrics.adapter_id == "test_adapter"
        assert metrics.total_conversions == 0
        assert metrics.successful_conversions == 0
        assert metrics.failed_conversions == 0
        assert metrics.average_execution_time == 0.0
        assert metrics.last_execution_time is None
        assert metrics.total_data_processed_mb == 0.0
        assert metrics.error_count == 0
        assert metrics.last_error is None
        assert metrics.last_error_timestamp is None
        assert metrics.health_check_count == 0
        assert metrics.last_health_check is None
        assert metrics.memory_usage_mb == 0.0
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.uptime_seconds == 0.0
        assert isinstance(metrics.creation_timestamp, float)


class MockAdapter(EnhancedShimAdapter):
    """Mock adapter for testing."""
    
    def __init__(self, adapter_id: str, config: AdapterConfig = None):
        super().__init__(adapter_id, config)
        self.init_called = False
        self.activate_called = False
        self.deactivate_called = False
        self.cleanup_called = False
        self.conversion_count = 0
        self.should_fail_init = False
        self.should_fail_activate = False
        self.should_fail_conversion = False
    
    def can_convert(self, request: ConversionRequest) -> float:
        """Return fixed confidence for testing."""
        if request.source_format == DataFormat.PANDAS_DATAFRAME:
            return 0.9
        return 0.0
    
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """Return fixed cost estimate for testing."""
        return ConversionCost(
            computational_cost=0.1,
            memory_cost_mb=10.0,
            time_estimate_seconds=1.0
        )
    
    def get_supported_conversions(self):
        """Return mock supported conversions."""
        return [(DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)]
    
    def _initialize_impl(self) -> bool:
        """Mock initialization implementation."""
        self.init_called = True
        if self.should_fail_init:
            raise RuntimeError("Mock initialization failure")
        return True
    
    def _activate_impl(self) -> bool:
        """Mock activation implementation."""
        self.activate_called = True
        if self.should_fail_activate:
            raise RuntimeError("Mock activation failure")
        return True
    
    def _deactivate_impl(self) -> bool:
        """Mock deactivation implementation."""
        self.deactivate_called = True
        return True
    
    def _cleanup_impl(self) -> bool:
        """Mock cleanup implementation."""
        self.cleanup_called = True
        return True
    
    def _health_check_impl(self):
        """Mock health check implementation."""
        return {
            'issues': [],
            'warnings': [],
            'metrics': {'custom_metric': 42}
        }
    
    def _convert_impl(self, request: ConversionRequest) -> ConversionResult:
        """Mock conversion implementation."""
        self.conversion_count += 1
        
        if self.should_fail_conversion:
            raise RuntimeError("Mock conversion failure")
        
        return ConversionResult(
            converted_data="converted_data",
            success=True,
            original_format=request.source_format,
            target_format=request.target_format,
            actual_format=request.target_format,
            request_id=request.request_id
        )


class TestEnhancedShimAdapter:
    """Test EnhancedShimAdapter functionality."""
    
    def test_adapter_creation(self):
        """Test creating an enhanced adapter."""
        config = AdapterConfig(adapter_id="test_adapter")
        adapter = MockAdapter("test_adapter", config)
        
        assert adapter.adapter_id == "test_adapter"
        assert adapter.config == config
        assert adapter.state == AdapterLifecycleState.UNINITIALIZED
        assert isinstance(adapter.metrics, AdapterMetrics)
        assert adapter.metrics.adapter_id == "test_adapter"
    
    def test_adapter_lifecycle_success(self):
        """Test successful adapter lifecycle."""
        adapter = MockAdapter("test_adapter")
        
        # Test initialization
        assert adapter.initialize() is True
        assert adapter.state == AdapterLifecycleState.INITIALIZED
        assert adapter.init_called is True
        
        # Test activation
        assert adapter.activate() is True
        assert adapter.state == AdapterLifecycleState.ACTIVE
        assert adapter.activate_called is True
        
        # Test deactivation
        assert adapter.deactivate() is True
        assert adapter.state == AdapterLifecycleState.INACTIVE
        assert adapter.deactivate_called is True
        
        # Test cleanup
        assert adapter.cleanup() is True
        assert adapter.state == AdapterLifecycleState.DISPOSED
        assert adapter.cleanup_called is True
    
    def test_adapter_lifecycle_failures(self):
        """Test adapter lifecycle with failures."""
        adapter = MockAdapter("test_adapter")
        
        # Test initialization failure
        adapter.should_fail_init = True
        assert adapter.initialize() is False
        assert adapter.state == AdapterLifecycleState.ERROR
        
        # Reset for activation test
        adapter = MockAdapter("test_adapter")
        adapter.initialize()
        
        # Test activation failure
        adapter.should_fail_activate = True
        assert adapter.activate() is False
        assert adapter.state == AdapterLifecycleState.ERROR
    
    def test_adapter_state_transitions(self):
        """Test invalid state transitions."""
        adapter = MockAdapter("test_adapter")
        
        # Cannot activate uninitialized adapter
        assert adapter.activate() is False
        
        # Cannot deactivate inactive adapter
        assert adapter.deactivate() is True  # Should succeed (no-op)
        
        # Initialize and activate
        adapter.initialize()
        adapter.activate()
        
        # Cannot initialize active adapter
        assert adapter.initialize() is True  # Should succeed (already initialized)
        assert adapter.state == AdapterLifecycleState.ACTIVE
    
    def test_adapter_conversion_success(self):
        """Test successful conversion with metrics."""
        adapter = MockAdapter("test_adapter")
        adapter.initialize()
        adapter.activate()
        
        request = ConversionRequest(
            source_data="test_data",
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            request_id="test_request"
        )
        
        result = adapter.convert(request)
        
        assert result.success is True
        assert result.converted_data == "converted_data"
        assert result.request_id == "test_request"
        assert adapter.conversion_count == 1
        
        # Check metrics were updated
        metrics = adapter.get_metrics()
        assert metrics.total_conversions == 1
        assert metrics.successful_conversions == 1
        assert metrics.failed_conversions == 0
        assert metrics.average_execution_time > 0
    
    def test_adapter_conversion_failure(self):
        """Test conversion failure with error tracking."""
        adapter = MockAdapter("test_adapter")
        adapter.initialize()
        adapter.activate()
        adapter.should_fail_conversion = True
        
        request = ConversionRequest(
            source_data="test_data",
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            request_id="test_request"
        )
        
        result = adapter.convert(request)
        
        assert result.success is False
        assert "Mock conversion failure" in str(result.errors)
        assert adapter.conversion_count == 1
        
        # Check metrics were updated
        metrics = adapter.get_metrics()
        assert metrics.total_conversions == 1
        assert metrics.successful_conversions == 0
        assert metrics.failed_conversions == 1
        assert metrics.last_error is not None
        assert metrics.last_error_timestamp is not None
    
    def test_adapter_conversion_inactive(self):
        """Test conversion when adapter is not active."""
        adapter = MockAdapter("test_adapter")
        adapter.initialize()
        # Don't activate
        
        request = ConversionRequest(
            source_data="test_data",
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            request_id="test_request"
        )
        
        result = adapter.convert(request)
        
        assert result.success is False
        assert "not active" in str(result.errors[0])
    
    def test_adapter_health_check(self):
        """Test adapter health check functionality."""
        adapter = MockAdapter("test_adapter")
        adapter.initialize()
        adapter.activate()
        
        health_result = adapter.perform_health_check()
        
        assert isinstance(health_result, HealthCheckResult)
        assert health_result.adapter_id == "test_adapter"
        assert health_result.is_healthy is True
        assert health_result.status == "healthy"
        assert health_result.issues == []
        assert 'custom_metric' in health_result.metrics_snapshot
        assert health_result.metrics_snapshot['custom_metric'] == 42
        
        # Check metrics were updated
        metrics = adapter.get_metrics()
        assert metrics.health_check_count == 1
        assert metrics.last_health_check is not None
    
    def test_adapter_health_check_error_state(self):
        """Test health check when adapter is in error state."""
        adapter = MockAdapter("test_adapter")
        adapter._set_state(AdapterLifecycleState.ERROR)
        
        health_result = adapter.perform_health_check()
        
        assert health_result.is_healthy is False
        assert health_result.status == "unhealthy"
        assert any("error state" in issue.lower() for issue in health_result.issues)
    
    def test_adapter_dependency_management(self):
        """Test adapter dependency management."""
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        
        # Add dependency
        adapter2.add_dependency(adapter1)
        
        assert len(adapter2.get_dependencies()) == 1
        assert adapter2.get_dependencies()[0] == adapter1
        assert len(adapter1.get_dependents()) == 1
        assert list(adapter1.get_dependents())[0] == adapter2
        
        # Remove dependency
        adapter2.remove_dependency(adapter1)
        
        assert len(adapter2.get_dependencies()) == 0
        assert len(adapter1.get_dependents()) == 0
    
    def test_adapter_metrics_update(self):
        """Test adapter metrics updating."""
        adapter = MockAdapter("test_adapter")
        
        # Simulate successful conversion
        adapter._update_metrics(1.5, True, 10.0)
        
        metrics = adapter.get_metrics()
        assert metrics.total_conversions == 1
        assert metrics.successful_conversions == 1
        assert metrics.failed_conversions == 0
        assert metrics.last_execution_time == 1.5
        assert metrics.average_execution_time == 1.5
        assert metrics.total_data_processed_mb == 10.0
        
        # Simulate failed conversion
        adapter._update_metrics(2.0, False, 5.0)
        
        metrics = adapter.get_metrics()
        assert metrics.total_conversions == 2
        assert metrics.successful_conversions == 1
        assert metrics.failed_conversions == 1
        assert metrics.last_execution_time == 2.0
        assert metrics.total_data_processed_mb == 15.0
        # Average should be updated (exponential moving average)
        assert 1.5 < metrics.average_execution_time < 2.0


class TestShimRegistry:
    """Test ShimRegistry functionality."""
    
    def test_registry_creation(self):
        """Test creating a ShimRegistry."""
        registry = ShimRegistry()
        
        assert registry.enable_auto_discovery is True
        assert registry.max_concurrent_health_checks == 5
        assert registry.health_check_interval_seconds == 300
        assert len(registry.list_adapters()) == 0
    
    def test_registry_with_compatibility_matrix(self):
        """Test creating registry with compatibility matrix."""
        matrix = PipelineCompatibilityMatrix()
        registry = ShimRegistry(compatibility_matrix=matrix)
        
        assert registry.compatibility_matrix == matrix
    
    def test_adapter_registration(self):
        """Test registering adapters with the registry."""
        registry = ShimRegistry()
        adapter = MockAdapter("test_adapter")
        config = AdapterConfig(adapter_id="test_adapter")
        
        registry.register_adapter(adapter, config)
        
        assert "test_adapter" in registry.list_adapters()
        assert registry.get_adapter("test_adapter") == adapter
        assert adapter.config == config
        assert adapter._registry_ref is not None
    
    def test_adapter_unregistration(self):
        """Test unregistering adapters."""
        registry = ShimRegistry()
        adapter = MockAdapter("test_adapter")
        
        registry.register_adapter(adapter)
        assert "test_adapter" in registry.list_adapters()
        
        result = registry.unregister_adapter("test_adapter")
        
        assert result is True
        assert "test_adapter" not in registry.list_adapters()
        assert adapter.cleanup_called is True
        assert adapter.state == AdapterLifecycleState.DISPOSED
    
    def test_adapter_unregistration_with_dependents(self):
        """Test unregistering adapter with dependents fails."""
        registry = ShimRegistry()
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        
        # Set up dependency
        adapter2.add_dependency(adapter1)
        
        registry.register_adapter(adapter1)
        registry.register_adapter(adapter2)
        
        # Should fail to unregister adapter with dependents
        result = registry.unregister_adapter("adapter1")
        assert result is False
        assert "adapter1" in registry.list_adapters()
    
    def test_adapter_unregistration_nonexistent(self):
        """Test unregistering non-existent adapter."""
        registry = ShimRegistry()
        
        result = registry.unregister_adapter("nonexistent")
        assert result is False
    
    def test_adapter_replacement(self):
        """Test replacing an existing adapter."""
        registry = ShimRegistry()
        adapter1 = MockAdapter("test_adapter")
        adapter2 = MockAdapter("test_adapter")  # Same ID
        
        registry.register_adapter(adapter1)
        registry.register_adapter(adapter2)  # Should replace adapter1
        
        assert registry.get_adapter("test_adapter") == adapter2
        assert adapter1.cleanup_called is True
    
    def test_initialize_all_adapters(self):
        """Test initializing all registered adapters."""
        registry = ShimRegistry()
        
        adapter1 = MockAdapter("adapter1", AdapterConfig("adapter1", initialization_order=1))
        adapter2 = MockAdapter("adapter2", AdapterConfig("adapter2", initialization_order=2))
        
        registry.register_adapter(adapter1)
        registry.register_adapter(adapter2)
        
        results = registry.initialize_all_adapters()
        
        assert results["adapter1"] is True
        assert results["adapter2"] is True
        assert adapter1.init_called is True
        assert adapter2.init_called is True
        assert adapter1.state == AdapterLifecycleState.INITIALIZED
        assert adapter2.state == AdapterLifecycleState.INITIALIZED
    
    def test_initialize_all_adapters_with_failure(self):
        """Test initializing adapters with one failure."""
        registry = ShimRegistry()
        
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        adapter2.should_fail_init = True
        
        registry.register_adapter(adapter1)
        registry.register_adapter(adapter2)
        
        results = registry.initialize_all_adapters()
        
        assert results["adapter1"] is True
        assert results["adapter2"] is False
        assert adapter1.state == AdapterLifecycleState.INITIALIZED
        assert adapter2.state == AdapterLifecycleState.ERROR
    
    def test_activate_all_adapters(self):
        """Test activating all initialized adapters."""
        registry = ShimRegistry()
        
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        
        registry.register_adapter(adapter1)
        registry.register_adapter(adapter2)
        
        # Initialize first
        registry.initialize_all_adapters()
        
        results = registry.activate_all_adapters()
        
        assert results["adapter1"] is True
        assert results["adapter2"] is True
        assert adapter1.state == AdapterLifecycleState.ACTIVE
        assert adapter2.state == AdapterLifecycleState.ACTIVE
    
    def test_shutdown_all_adapters(self):
        """Test shutting down all adapters."""
        registry = ShimRegistry()
        
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        
        registry.register_adapter(adapter1)
        registry.register_adapter(adapter2)
        
        registry.initialize_all_adapters()
        registry.activate_all_adapters()
        
        results = registry.shutdown_all_adapters()
        
        assert results["adapter1"] is True
        assert results["adapter2"] is True
        assert adapter1.state == AdapterLifecycleState.DISPOSED
        assert adapter2.state == AdapterLifecycleState.DISPOSED
    
    def test_get_active_adapters(self):
        """Test getting active adapters."""
        registry = ShimRegistry()
        
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        
        registry.register_adapter(adapter1)
        registry.register_adapter(adapter2)
        
        # Initially no active adapters
        assert len(registry.get_active_adapters()) == 0
        
        # Initialize and activate adapter1 only
        adapter1.initialize()
        adapter1.activate()
        
        active = registry.get_active_adapters()
        assert len(active) == 1
        assert active[0] == adapter1
    
    def test_discovery_path_management(self):
        """Test managing discovery paths."""
        registry = ShimRegistry()
        
        registry.add_discovery_path("test.module1")
        registry.add_discovery_path("test.module2")
        
        assert len(registry._discovery_paths) == 2
        assert "test.module1" in registry._discovery_paths
        assert "test.module2" in registry._discovery_paths
        
        # Adding duplicate path should not create duplicate
        registry.add_discovery_path("test.module1")
        assert len(registry._discovery_paths) == 2
    
    @patch('importlib.import_module')
    def test_discover_adapters(self, mock_import):
        """Test discovering adapter classes."""
        registry = ShimRegistry()
        
        # Mock module with adapter class
        mock_module = MagicMock()
        mock_module.__dict__ = {'TestAdapter': MockAdapter}
        mock_import.return_value = mock_module
        
        # Mock inspect.getmembers
        with patch('inspect.getmembers') as mock_getmembers:
            mock_getmembers.return_value = [
                ('TestAdapter', MockAdapter),
                ('NotAnAdapter', str),  # Should be ignored
                ('EnhancedShimAdapter', EnhancedShimAdapter)  # Should be ignored (base class)
            ]
            
            registry.add_discovery_path("test.module")
            discovered = registry.discover_adapters()
            
            assert 'TestAdapter' in discovered
            assert discovered['TestAdapter'] == MockAdapter
            assert len(discovered) == 1  # Only MockAdapter should be discovered
    
    def test_create_adapter_from_class(self):
        """Test creating adapter from discovered class."""
        registry = ShimRegistry()
        registry._adapter_classes['MockAdapter'] = MockAdapter
        
        adapter = registry.create_adapter_from_class(
            'MockAdapter', 
            'test_instance',
            config=AdapterConfig('test_instance')
        )
        
        assert adapter is not None
        assert isinstance(adapter, MockAdapter)
        assert adapter.adapter_id == 'test_instance'
    
    def test_create_adapter_from_nonexistent_class(self):
        """Test creating adapter from non-existent class."""
        registry = ShimRegistry()
        
        adapter = registry.create_adapter_from_class('NonExistent', 'test_instance')
        
        assert adapter is None
    
    def test_perform_health_checks(self):
        """Test performing health checks on all adapters."""
        registry = ShimRegistry()
        
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        
        registry.register_adapter(adapter1)
        registry.register_adapter(adapter2)
        
        # Initialize and activate adapters
        adapter1.initialize()
        adapter1.activate()
        adapter2.initialize()
        adapter2.activate()
        
        results = registry.perform_health_checks()
        
        assert len(results) == 2
        assert "adapter1" in results
        assert "adapter2" in results
        assert results["adapter1"].is_healthy is True
        assert results["adapter2"].is_healthy is True
    
    def test_get_compatible_adapters(self):
        """Test getting compatible adapters for conversion request."""
        registry = ShimRegistry()
        
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        
        registry.register_adapter(adapter1)
        registry.register_adapter(adapter2)
        
        # Activate adapters
        adapter1.initialize()
        adapter1.activate()
        adapter2.initialize()
        adapter2.activate()
        
        request = ConversionRequest(
            source_data="test_data",
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            request_id="test_request"
        )
        
        compatible = registry.get_compatible_adapters(request)
        
        # Both adapters should be compatible (confidence 0.9)
        assert len(compatible) == 2
        assert compatible[0][1] == 0.9  # Should be sorted by confidence
        assert compatible[1][1] == 0.9
    
    def test_get_registry_stats(self):
        """Test getting registry statistics."""
        registry = ShimRegistry()
        
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        
        registry.register_adapter(adapter1)
        registry.register_adapter(adapter2)
        
        # Initialize one adapter
        adapter1.initialize()
        
        stats = registry.get_registry_stats()
        
        assert stats['total_adapters'] == 2
        assert stats['adapter_states']['uninitialized'] == 1
        assert stats['adapter_states']['initialized'] == 1
        assert 'total_conversions' in stats
        assert 'error_rate' in stats
        assert 'registry_stats' in stats
    
    def test_context_manager(self):
        """Test registry as context manager."""
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        
        with ShimRegistry() as registry:
            registry.register_adapter(adapter1)
            registry.register_adapter(adapter2)
            
            registry.initialize_all_adapters()
            registry.activate_all_adapters()
            
            assert adapter1.state == AdapterLifecycleState.ACTIVE
            assert adapter2.state == AdapterLifecycleState.ACTIVE
        
        # After context exit, adapters should be cleaned up
        assert adapter1.state == AdapterLifecycleState.DISPOSED
        assert adapter2.state == AdapterLifecycleState.DISPOSED


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_shim_registry(self):
        """Test creating registry with factory function."""
        matrix = PipelineCompatibilityMatrix()
        registry = create_shim_registry(compatibility_matrix=matrix)
        
        assert isinstance(registry, ShimRegistry)
        assert registry.compatibility_matrix == matrix
    
    def test_create_adapter_config(self):
        """Test creating config with factory function."""
        config = create_adapter_config(
            "test_adapter",
            dependencies=["dep1"],
            enable_metrics=False
        )
        
        assert isinstance(config, AdapterConfig)
        assert config.adapter_id == "test_adapter"
        assert config.dependencies == ["dep1"]
        assert config.enable_metrics is False


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_validate_adapter_dependencies_success(self):
        """Test successful dependency validation."""
        adapter1 = MockAdapter("adapter1", AdapterConfig("adapter1"))
        adapter2 = MockAdapter("adapter2", AdapterConfig("adapter2", dependencies=["adapter1"]))
        
        result = validate_adapter_dependencies([adapter1, adapter2])
        
        assert result.is_valid is True
        assert result.score == 1.0
        assert len(result.errors) == 0
    
    def test_validate_adapter_dependencies_missing(self):
        """Test dependency validation with missing dependency."""
        adapter1 = MockAdapter("adapter1", AdapterConfig("adapter1", dependencies=["missing"]))
        
        result = validate_adapter_dependencies([adapter1])
        
        assert result.is_valid is False
        assert result.score == 0.0
        assert len(result.errors) == 1
        assert "missing dependency" in result.errors[0]
    
    def test_validate_adapter_dependencies_circular(self):
        """Test dependency validation with circular dependency."""
        adapter1 = MockAdapter("adapter1", AdapterConfig("adapter1", dependencies=["adapter2"]))
        adapter2 = MockAdapter("adapter2", AdapterConfig("adapter2", dependencies=["adapter1"]))
        
        result = validate_adapter_dependencies([adapter1, adapter2])
        
        assert result.is_valid is False
        assert result.score == 0.0
        assert len(result.errors) >= 1
        assert any("circular dependency" in error.lower() for error in result.errors)
    
    @patch('time.sleep')
    def test_monitor_adapter_performance(self, mock_sleep):
        """Test monitoring adapter performance."""
        adapter = MockAdapter("test_adapter")
        
        # Simulate some activity
        adapter._update_metrics(1.0, True, 10.0)
        initial_metrics = adapter.get_metrics()
        
        # Simulate more activity during monitoring
        mock_sleep.side_effect = lambda _: adapter._update_metrics(2.0, True, 15.0)
        
        report = monitor_adapter_performance(adapter, duration_seconds=60)
        
        assert 'monitoring_duration' in report
        assert 'conversions_per_second' in report
        assert 'errors_per_second' in report
        assert 'data_throughput_mb_per_second' in report
        assert 'current_error_rate' in report
        assert 'adapter_state' in report
        assert 'snapshot' in report
        assert 'start_metrics' in report['snapshot']
        assert 'end_metrics' in report['snapshot']


# Integration Tests

class TestShimRegistryIntegration:
    """Integration tests for ShimRegistry with real scenarios."""
    
    def test_full_lifecycle_integration(self):
        """Test complete adapter lifecycle in registry."""
        with ShimRegistry() as registry:
            # Create and register adapters
            adapter1 = MockAdapter("adapter1", AdapterConfig("adapter1", initialization_order=1))
            adapter2 = MockAdapter("adapter2", AdapterConfig("adapter2", initialization_order=2))
            
            registry.register_adapter(adapter1)
            registry.register_adapter(adapter2)
            
            # Initialize all
            init_results = registry.initialize_all_adapters()
            assert all(init_results.values())
            
            # Activate all
            activate_results = registry.activate_all_adapters()
            assert all(activate_results.values())
            
            # Perform conversions
            request = ConversionRequest(
                source_data="test_data",
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                request_id="test_request_1"
            )
            
            result1 = adapter1.convert(request)
            result2 = adapter2.convert(request)
            
            assert result1.success is True
            assert result2.success is True
            
            # Check health
            health_results = registry.perform_health_checks()
            assert all(r.is_healthy for r in health_results.values())
            
            # Get statistics
            stats = registry.get_registry_stats()
            assert stats['total_adapters'] == 2
            assert stats['total_conversions'] == 2
            assert stats['error_rate'] == 0.0
        
        # Verify cleanup occurred
        assert adapter1.state == AdapterLifecycleState.DISPOSED
        assert adapter2.state == AdapterLifecycleState.DISPOSED
    
    def test_error_handling_integration(self):
        """Test error handling in integrated scenarios."""
        registry = ShimRegistry()
        
        # Create adapter that fails initialization
        bad_adapter = MockAdapter("bad_adapter")
        bad_adapter.should_fail_init = True
        
        good_adapter = MockAdapter("good_adapter")
        
        registry.register_adapter(bad_adapter)
        registry.register_adapter(good_adapter)
        
        # Initialize - one should fail
        init_results = registry.initialize_all_adapters()
        assert init_results["bad_adapter"] is False
        assert init_results["good_adapter"] is True
        
        # Only good adapter should be active
        activate_results = registry.activate_all_adapters()
        assert activate_results["bad_adapter"] is False
        assert activate_results["good_adapter"] is True
        
        # Health checks should show the issue
        health_results = registry.perform_health_checks()
        assert health_results["bad_adapter"].is_healthy is False
        assert health_results["good_adapter"].is_healthy is True
        
        # Statistics should reflect the problems
        stats = registry.get_registry_stats()
        assert stats['adapter_states']['error'] == 1
        assert stats['adapter_states']['active'] == 1
        
        registry.shutdown_all_adapters()
    
    def test_concurrent_operations(self):
        """Test concurrent operations on registry."""
        registry = ShimRegistry()
        adapters = [MockAdapter(f"adapter_{i}") for i in range(10)]
        
        # Register all adapters
        for adapter in adapters:
            registry.register_adapter(adapter)
        
        # Initialize and activate
        registry.initialize_all_adapters()
        registry.activate_all_adapters()
        
        # Perform concurrent conversions
        request = ConversionRequest(
            source_data="test_data",
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            request_id="concurrent_test"
        )
        
        def convert_with_adapter(adapter):
            return adapter.convert(request)
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(convert_with_adapter, adapter) for adapter in adapters]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All conversions should succeed
        assert len(results) == 10
        assert all(result.success for result in results)
        
        # Health check should still work
        health_results = registry.perform_health_checks()
        assert len(health_results) == 10
        assert all(r.is_healthy for r in health_results.values())
        
        registry.shutdown_all_adapters()


if __name__ == "__main__":
    pytest.main([__file__])