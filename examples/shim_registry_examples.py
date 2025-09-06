"""
Integration examples demonstrating ShimRegistry and EnhancedShimAdapter usage.

These examples show how to:
1. Create and register custom adapters
2. Manage adapter lifecycle
3. Perform automatic discovery
4. Monitor performance and health
5. Handle dependencies
6. Integration with PipelineCompatibilityMatrix
"""

import time
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

from src.localdata_mcp.pipeline.integration.shim_registry import (
    EnhancedShimAdapter, ShimRegistry, AdapterConfig,
    create_shim_registry, create_adapter_config
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    ConversionRequest, ConversionResult, ConversionCost, DataFormat
)
from src.localdata_mcp.pipeline.integration.compatibility_matrix import (
    PipelineCompatibilityMatrix
)


# Example 1: Creating Custom Adapters

class PandasToNumpyAdapter(EnhancedShimAdapter):
    """Example adapter converting Pandas DataFrame to NumPy array."""
    
    def __init__(self, adapter_id: str = "pandas_to_numpy", config: Optional[AdapterConfig] = None):
        super().__init__(adapter_id, config)
        self.conversion_stats = {"numeric_conversions": 0, "data_loss_warnings": 0}
    
    def can_convert(self, request: ConversionRequest) -> float:
        """Return confidence for Pandas to NumPy conversion."""
        if (request.source_format == DataFormat.PANDAS_DATAFRAME and
            request.target_format == DataFormat.NUMPY_ARRAY):
            return 0.95
        return 0.0
    
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """Estimate conversion cost."""
        if isinstance(request.source_data, pd.DataFrame):
            rows, cols = request.source_data.shape
            data_size = rows * cols * 8 / (1024 * 1024)  # Rough MB estimate
            
            return ConversionCost(
                computational_cost=0.1 + (data_size / 1000),
                memory_cost_mb=data_size * 2,  # Input + output
                time_estimate_seconds=data_size / 100,  # 100MB/sec processing
                quality_impact=-0.1 if request.source_data.isnull().sum().sum() > 0 else 0.0
            )
        
        return ConversionCost(
            computational_cost=0.5,
            memory_cost_mb=10.0,
            time_estimate_seconds=1.0
        )
    
    def get_supported_conversions(self):
        """Return supported conversion paths."""
        return [(DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)]
    
    def _initialize_impl(self) -> bool:
        """Initialize the adapter."""
        print(f"[{self.adapter_id}] Initializing Pandas to NumPy adapter")
        # Perform any setup needed
        return True
    
    def _activate_impl(self) -> bool:
        """Activate the adapter."""
        print(f"[{self.adapter_id}] Activating adapter")
        return True
    
    def _deactivate_impl(self) -> bool:
        """Deactivate the adapter."""
        print(f"[{self.adapter_id}] Deactivating adapter")
        return True
    
    def _cleanup_impl(self) -> bool:
        """Cleanup adapter resources."""
        print(f"[{self.adapter_id}] Cleaning up adapter")
        print(f"  Final stats: {self.conversion_stats}")
        return True
    
    def _health_check_impl(self) -> Dict[str, Any]:
        """Perform adapter-specific health checks."""
        issues = []
        warnings = []
        
        if self.conversion_stats["data_loss_warnings"] > 10:
            warnings.append("High number of data loss warnings")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'metrics': {
                'numeric_conversions': self.conversion_stats["numeric_conversions"],
                'data_loss_warnings': self.conversion_stats["data_loss_warnings"]
            }
        }
    
    def _convert_impl(self, request: ConversionRequest) -> ConversionResult:
        """Convert Pandas DataFrame to NumPy array."""
        if not isinstance(request.source_data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(request.source_data)}")
        
        df = request.source_data
        warnings = []
        
        # Check for non-numeric data
        numeric_only = df.select_dtypes(include=[np.number])
        if len(numeric_only.columns) < len(df.columns):
            non_numeric_cols = set(df.columns) - set(numeric_only.columns)
            warnings.append(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
            df = numeric_only
            self.conversion_stats["data_loss_warnings"] += 1
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            warnings.append("DataFrame contains missing values, filling with 0")
            df = df.fillna(0)
            self.conversion_stats["data_loss_warnings"] += 1
        
        # Convert to NumPy
        numpy_array = df.values
        self.conversion_stats["numeric_conversions"] += 1
        
        return ConversionResult(
            converted_data=numpy_array,
            success=True,
            original_format=request.source_format,
            target_format=request.target_format,
            actual_format=DataFormat.NUMPY_ARRAY,
            metadata={
                'original_columns': list(request.source_data.columns),
                'original_shape': request.source_data.shape,
                'converted_shape': numpy_array.shape,
                'numeric_columns_only': len(numeric_only.columns) < len(request.source_data.columns)
            },
            warnings=warnings,
            request_id=request.request_id
        )


class NumpyToPandasAdapter(EnhancedShimAdapter):
    """Example adapter converting NumPy array to Pandas DataFrame."""
    
    def __init__(self, adapter_id: str = "numpy_to_pandas", config: Optional[AdapterConfig] = None):
        super().__init__(adapter_id, config)
    
    def can_convert(self, request: ConversionRequest) -> float:
        """Return confidence for NumPy to Pandas conversion."""
        if (request.source_format == DataFormat.NUMPY_ARRAY and
            request.target_format == DataFormat.PANDAS_DATAFRAME):
            return 0.9
        return 0.0
    
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """Estimate conversion cost."""
        if isinstance(request.source_data, np.ndarray):
            data_size = request.source_data.nbytes / (1024 * 1024)
            
            return ConversionCost(
                computational_cost=0.05 + (data_size / 2000),
                memory_cost_mb=data_size * 1.5,  # Pandas has overhead
                time_estimate_seconds=data_size / 200  # Faster than reverse conversion
            )
        
        return ConversionCost(
            computational_cost=0.3,
            memory_cost_mb=5.0,
            time_estimate_seconds=0.5
        )
    
    def get_supported_conversions(self):
        """Return supported conversion paths."""
        return [(DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME)]
    
    def _convert_impl(self, request: ConversionRequest) -> ConversionResult:
        """Convert NumPy array to Pandas DataFrame."""
        if not isinstance(request.source_data, np.ndarray):
            raise ValueError(f"Expected ndarray, got {type(request.source_data)}")
        
        array = request.source_data
        warnings = []
        
        # Handle different array shapes
        if array.ndim == 1:
            # 1D array to single column DataFrame
            df = pd.DataFrame({'column_0': array})
            warnings.append("1D array converted to single-column DataFrame")
        elif array.ndim == 2:
            # 2D array to DataFrame with generated column names
            columns = [f'column_{i}' for i in range(array.shape[1])]
            df = pd.DataFrame(array, columns=columns)
        else:
            # Higher dimensional arrays - flatten to 2D
            reshaped = array.reshape(array.shape[0], -1)
            columns = [f'column_{i}' for i in range(reshaped.shape[1])]
            df = pd.DataFrame(reshaped, columns=columns)
            warnings.append(f"Reshaped {array.ndim}D array to 2D: {array.shape} -> {reshaped.shape}")
        
        return ConversionResult(
            converted_data=df,
            success=True,
            original_format=request.source_format,
            target_format=request.target_format,
            actual_format=DataFormat.PANDAS_DATAFRAME,
            metadata={
                'original_shape': array.shape,
                'original_dtype': str(array.dtype),
                'converted_columns': list(df.columns),
                'converted_shape': df.shape
            },
            warnings=warnings,
            request_id=request.request_id
        )


# Example 2: Basic Registry Usage

def example_basic_registry_usage():
    """Demonstrate basic registry operations."""
    print("=== Example 2: Basic Registry Usage ===")
    
    # Create registry
    registry = create_shim_registry()
    
    # Create and configure adapters
    pandas_config = create_adapter_config(
        "pandas_to_numpy",
        initialization_order=1,
        enable_metrics=True,
        metadata={"version": "1.0", "author": "example"}
    )
    
    numpy_config = create_adapter_config(
        "numpy_to_pandas", 
        initialization_order=2,
        enable_metrics=True
    )
    
    pandas_adapter = PandasToNumpyAdapter(config=pandas_config)
    numpy_adapter = NumpyToPandasAdapter(config=numpy_config)
    
    # Register adapters
    registry.register_adapter(pandas_adapter, pandas_config)
    registry.register_adapter(numpy_adapter, numpy_config)
    
    print(f"Registered adapters: {registry.list_adapters()}")
    
    # Initialize and activate
    print("\nInitializing adapters...")
    init_results = registry.initialize_all_adapters()
    for adapter_id, success in init_results.items():
        print(f"  {adapter_id}: {'✓' if success else '✗'}")
    
    print("\nActivating adapters...")
    activate_results = registry.activate_all_adapters()
    for adapter_id, success in activate_results.items():
        print(f"  {adapter_id}: {'✓' if success else '✗'}")
    
    # Show active adapters
    active_adapters = registry.get_active_adapters()
    print(f"\nActive adapters: {[a.adapter_id for a in active_adapters]}")
    
    # Perform a conversion
    print("\nPerforming conversion...")
    test_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['x', 'y', 'z', 'w', 'v']  # Non-numeric column
    })
    
    request = ConversionRequest(
        source_data=test_data,
        source_format=DataFormat.PANDAS_DATAFRAME,
        target_format=DataFormat.NUMPY_ARRAY,
        request_id="example_conversion"
    )
    
    # Find compatible adapters
    compatible = registry.get_compatible_adapters(request)
    print(f"Compatible adapters: {[(a.adapter_id, conf) for a, conf in compatible]}")
    
    if compatible:
        best_adapter, confidence = compatible[0]
        result = best_adapter.convert(request)
        
        print(f"Conversion result:")
        print(f"  Success: {result.success}")
        print(f"  Shape: {result.converted_data.shape}")
        print(f"  Warnings: {result.warnings}")
        print(f"  Metadata: {result.metadata}")
    
    # Check health
    print("\nPerforming health checks...")
    health_results = registry.perform_health_checks()
    for adapter_id, health in health_results.items():
        print(f"  {adapter_id}: {'✓ Healthy' if health.is_healthy else '✗ Unhealthy'}")
        if health.warnings:
            print(f"    Warnings: {health.warnings}")
    
    # Get statistics
    print("\nRegistry statistics:")
    stats = registry.get_registry_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    print("\nShutting down...")
    shutdown_results = registry.shutdown_all_adapters()
    for adapter_id, success in shutdown_results.items():
        print(f"  {adapter_id}: {'✓' if success else '✗'}")


# Example 3: Adapter Dependencies

class DataValidationAdapter(EnhancedShimAdapter):
    """Example adapter that validates data before conversion."""
    
    def __init__(self, adapter_id: str = "data_validator", config: Optional[AdapterConfig] = None):
        super().__init__(adapter_id, config)
        self.validation_count = 0
    
    def can_convert(self, request: ConversionRequest) -> float:
        """This adapter can validate any data format."""
        return 0.5  # Medium confidence for all formats
    
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """Validation is fast and low-cost."""
        return ConversionCost(
            computational_cost=0.01,
            memory_cost_mb=1.0,
            time_estimate_seconds=0.1
        )
    
    def get_supported_conversions(self):
        """Validation is format-agnostic."""
        return [(fmt, fmt) for fmt in DataFormat]  # Pass-through for all formats
    
    def _convert_impl(self, request: ConversionRequest) -> ConversionResult:
        """Validate data and pass through unchanged."""
        self.validation_count += 1
        warnings = []
        
        # Perform validation checks
        if request.source_data is None:
            raise ValueError("Source data cannot be None")
        
        if isinstance(request.source_data, pd.DataFrame):
            if request.source_data.empty:
                warnings.append("DataFrame is empty")
            if request.source_data.isnull().sum().sum() > len(request.source_data) * 0.5:
                warnings.append("More than 50% of data is missing")
        
        return ConversionResult(
            converted_data=request.source_data,  # Pass through
            success=True,
            original_format=request.source_format,
            target_format=request.target_format,
            actual_format=request.source_format,
            metadata={'validated': True, 'validation_id': self.validation_count},
            warnings=warnings,
            request_id=request.request_id
        )


def example_adapter_dependencies():
    """Demonstrate adapter dependencies and ordering."""
    print("\n=== Example 3: Adapter Dependencies ===")
    
    registry = create_shim_registry()
    
    # Create validator adapter (should initialize first)
    validator_config = create_adapter_config(
        "data_validator",
        initialization_order=1,
        metadata={"purpose": "validation"}
    )
    validator = DataValidationAdapter(config=validator_config)
    
    # Create converter adapter that depends on validator
    converter_config = create_adapter_config(
        "pandas_to_numpy",
        initialization_order=2,
        dependencies=["data_validator"],
        metadata={"purpose": "conversion"}
    )
    converter = PandasToNumpyAdapter(config=converter_config)
    
    # Set up dependency relationship
    converter.add_dependency(validator)
    
    registry.register_adapter(validator, validator_config)
    registry.register_adapter(converter, converter_config)
    
    print(f"Adapter dependencies:")
    print(f"  {converter.adapter_id} depends on: {[d.adapter_id for d in converter.get_dependencies()]}")
    print(f"  {validator.adapter_id} dependents: {[d.adapter_id for d in validator.get_dependents()]}")
    
    # Initialize in dependency order
    registry.initialize_all_adapters()
    registry.activate_all_adapters()
    
    # Demonstrate that dependencies affect shutdown order
    print("\nShutdown order (reverse dependency):")
    shutdown_results = registry.shutdown_all_adapters()
    for adapter_id in shutdown_results:
        print(f"  Shutting down {adapter_id}")


# Example 4: Integration with Compatibility Matrix

def example_compatibility_matrix_integration():
    """Demonstrate integration with PipelineCompatibilityMatrix."""
    print("\n=== Example 4: Compatibility Matrix Integration ===")
    
    # Create compatibility matrix
    matrix = PipelineCompatibilityMatrix()
    
    # Create registry with matrix
    registry = create_shim_registry(compatibility_matrix=matrix)
    
    # Register adapters
    pandas_adapter = PandasToNumpyAdapter()
    numpy_adapter = NumpyToPandasAdapter()
    
    registry.register_adapter(pandas_adapter)
    registry.register_adapter(numpy_adapter)
    registry.initialize_all_adapters()
    registry.activate_all_adapters()
    
    # Test compatibility checking
    compatibility = matrix.get_compatibility(
        DataFormat.PANDAS_DATAFRAME,
        DataFormat.NUMPY_ARRAY
    )
    
    print(f"Compatibility score: {compatibility.score}")
    print(f"Direct compatible: {compatibility.direct_compatible}")
    print(f"Conversion required: {compatibility.conversion_required}")
    print(f"Recommendations: {compatibility.recommendations}")
    
    # Find conversion path
    path = registry.find_conversion_path(
        DataFormat.PANDAS_DATAFRAME,
        DataFormat.NUMPY_ARRAY
    )
    
    if path:
        print(f"\nConversion path found:")
        print(f"  Steps: {len(path.steps)}")
        print(f"  Total cost: {path.total_cost.computational_cost}")
        print(f"  Success probability: {path.success_probability}")
    
    registry.shutdown_all_adapters()


# Example 5: Performance Monitoring

def example_performance_monitoring():
    """Demonstrate performance monitoring and metrics collection."""
    print("\n=== Example 5: Performance Monitoring ===")
    
    registry = create_shim_registry()
    adapter = PandasToNumpyAdapter()
    
    registry.register_adapter(adapter)
    registry.initialize_all_adapters()
    registry.activate_all_adapters()
    
    # Perform multiple conversions to generate metrics
    print("Performing conversions to generate metrics...")
    for i in range(10):
        test_data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randint(0, 10, 100)
        })
        
        request = ConversionRequest(
            source_data=test_data,
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            request_id=f"perf_test_{i}"
        )
        
        result = adapter.convert(request)
        if not result.success:
            print(f"  Conversion {i} failed: {result.errors}")
        
        # Simulate some processing delay
        time.sleep(0.01)
    
    # Get adapter metrics
    metrics = adapter.get_metrics()
    print(f"\nAdapter Metrics:")
    print(f"  Total conversions: {metrics.total_conversions}")
    print(f"  Successful conversions: {metrics.successful_conversions}")
    print(f"  Failed conversions: {metrics.failed_conversions}")
    print(f"  Average execution time: {metrics.average_execution_time:.4f}s")
    print(f"  Total data processed: {metrics.total_data_processed_mb:.2f}MB")
    print(f"  Uptime: {metrics.uptime_seconds:.2f}s")
    
    # Perform health check
    health = adapter.perform_health_check()
    print(f"\nHealth Check:")
    print(f"  Status: {health.status}")
    print(f"  Healthy: {health.is_healthy}")
    print(f"  Custom metrics: {health.metrics_snapshot.get('numeric_conversions', 'N/A')}")
    
    registry.shutdown_all_adapters()


# Example 6: Error Handling and Recovery

class UnreliableAdapter(EnhancedShimAdapter):
    """Example adapter that occasionally fails for testing error handling."""
    
    def __init__(self, adapter_id: str = "unreliable_adapter", config: Optional[AdapterConfig] = None):
        super().__init__(adapter_id, config)
        self.conversion_count = 0
        self.failure_rate = 0.3  # 30% failure rate
    
    def can_convert(self, request: ConversionRequest) -> float:
        return 0.7
    
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        return ConversionCost(
            computational_cost=0.2,
            memory_cost_mb=5.0,
            time_estimate_seconds=0.5
        )
    
    def get_supported_conversions(self):
        return [(DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)]
    
    def _convert_impl(self, request: ConversionRequest) -> ConversionResult:
        self.conversion_count += 1
        
        # Simulate random failures
        import random
        if random.random() < self.failure_rate:
            raise RuntimeError(f"Simulated failure on conversion {self.conversion_count}")
        
        return ConversionResult(
            converted_data="mock_conversion",
            success=True,
            original_format=request.source_format,
            target_format=request.target_format,
            actual_format=request.target_format,
            request_id=request.request_id
        )


def example_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n=== Example 6: Error Handling and Recovery ===")
    
    registry = create_shim_registry()
    
    # Create unreliable adapter
    unreliable = UnreliableAdapter()
    registry.register_adapter(unreliable)
    registry.initialize_all_adapters()
    registry.activate_all_adapters()
    
    # Perform conversions and track errors
    successes = 0
    failures = 0
    
    print("Performing conversions with unreliable adapter...")
    for i in range(20):
        request = ConversionRequest(
            source_data=pd.DataFrame({'A': [1, 2, 3]}),
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            request_id=f"test_{i}"
        )
        
        result = unreliable.convert(request)
        if result.success:
            successes += 1
        else:
            failures += 1
            print(f"  Conversion {i} failed: {result.errors[0] if result.errors else 'Unknown error'}")
    
    print(f"\nResults: {successes} successes, {failures} failures")
    
    # Check adapter health after errors
    health = unreliable.perform_health_check()
    print(f"\nHealth Status: {health.status}")
    print(f"Issues: {health.issues}")
    print(f"Warnings: {health.warnings}")
    
    # Check error metrics
    metrics = unreliable.get_metrics()
    error_rate = metrics.failed_conversions / metrics.total_conversions
    print(f"Error rate: {error_rate:.1%}")
    print(f"Last error: {metrics.last_error}")
    
    registry.shutdown_all_adapters()


# Example 7: Context Manager Usage

def example_context_manager():
    """Demonstrate using registry as context manager."""
    print("\n=== Example 7: Context Manager Usage ===")
    
    print("Using registry as context manager...")
    
    with create_shim_registry() as registry:
        # Set up adapters
        adapter = PandasToNumpyAdapter()
        registry.register_adapter(adapter)
        registry.initialize_all_adapters()
        registry.activate_all_adapters()
        
        print(f"Adapter state inside context: {adapter.state.value}")
        
        # Perform operations
        test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        request = ConversionRequest(
            source_data=test_data,
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            request_id="context_test"
        )
        
        result = adapter.convert(request)
        print(f"Conversion success: {result.success}")
    
    # After context exit
    print(f"Adapter state after context exit: {adapter.state.value}")


def main():
    """Run all examples."""
    print("ShimRegistry Integration Examples")
    print("=" * 50)
    
    try:
        example_basic_registry_usage()
        example_adapter_dependencies()
        example_compatibility_matrix_integration()
        example_performance_monitoring()
        example_error_handling()
        example_context_manager()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()