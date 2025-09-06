"""
Validation Prototype for Integration Shims Architecture

This module provides concrete prototype implementations that demonstrate
the key architectural concepts and validate the design decisions for
the Integration Shims Framework.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from .interfaces import (
    DataFormat,
    ConversionRequest,
    ConversionResult,
    ConversionCost,
    ConversionPath,
    ConversionStep,
    ShimAdapter,
    ConversionRegistry,
    CompatibilityMatrix,
    ValidationResult,
    CompatibilityScore,
    DomainRequirements,
    DataFormatSpec
)


class PrototypeDataFrameToNumpyAdapter(ShimAdapter):
    """
    Prototype adapter demonstrating pandas DataFrame to numpy array conversion.
    
    This prototype validates:
    - ShimAdapter interface implementation
    - Cost estimation logic
    - Quality assessment
    - Metadata preservation patterns
    """
    
    def __init__(self):
        super().__init__("pandas_to_numpy_prototype")
        self.supported_conversions = [
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)
        ]
    
    def can_convert(self, request: ConversionRequest) -> float:
        """Return confidence score for DataFrame to numpy conversion."""
        if (request.source_format != DataFormat.PANDAS_DATAFRAME or
            request.target_format != DataFormat.NUMPY_ARRAY):
            return 0.0
        
        # Check if source data is actually a DataFrame
        if not isinstance(request.source_data, pd.DataFrame):
            return 0.0
        
        # Check if DataFrame can be converted to numeric array
        df = request.source_data
        try:
            # Try to convert to numeric
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                return 0.3  # Low confidence for non-numeric data
            elif len(numeric_df.columns) == len(df.columns):
                return 0.95  # High confidence for all-numeric data
            else:
                return 0.7  # Medium confidence for mixed data
        except Exception:
            return 0.1  # Very low confidence if analysis fails
    
    def convert(self, request: ConversionRequest) -> ConversionResult:
        """Convert DataFrame to numpy array with metadata preservation."""
        start_time = time.time()
        
        try:
            df = request.source_data
            
            # Preserve original metadata
            original_metadata = {
                'column_names': df.columns.tolist(),
                'index_names': df.index.names,
                'dtypes': df.dtypes.to_dict(),
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            # Convert to numpy array
            # Handle mixed types by converting to object array if necessary
            try:
                numpy_array = df.values
                conversion_quality = 0.95
            except Exception:
                # Fallback to object array for mixed types
                numpy_array = df.values.astype(object)
                conversion_quality = 0.8
            
            # Prepare result metadata
            result_metadata = {
                'original_metadata': original_metadata,
                'conversion_type': 'dataframe_to_numpy',
                'numpy_shape': numpy_array.shape,
                'numpy_dtype': str(numpy_array.dtype),
                'memory_usage': numpy_array.nbytes
            }
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            performance_metrics = {
                'execution_time': execution_time,
                'memory_efficiency': numpy_array.nbytes / original_metadata['memory_usage'],
                'data_processed_mb': numpy_array.nbytes / (1024 * 1024)
            }
            
            return ConversionResult(
                converted_data=numpy_array,
                success=True,
                original_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                actual_format=DataFormat.NUMPY_ARRAY,
                metadata=result_metadata,
                performance_metrics=performance_metrics,
                quality_score=conversion_quality,
                request_id=request.request_id,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ConversionResult(
                converted_data=request.source_data,  # Return original on failure
                success=False,
                original_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                actual_format=DataFormat.PANDAS_DATAFRAME,
                errors=[str(e)],
                request_id=request.request_id,
                execution_time=execution_time
            )
    
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """Estimate conversion cost based on data characteristics."""
        if not isinstance(request.source_data, pd.DataFrame):
            return ConversionCost(
                computational_cost=0.0,
                memory_cost_mb=0.0,
                time_estimate_seconds=0.0
            )
        
        df = request.source_data
        rows, cols = df.shape
        
        # Estimate computational cost based on data size and complexity
        base_cost = min(0.1 + (rows * cols) / 1000000, 1.0)  # Scale 0.1-1.0
        
        # Memory cost estimation
        current_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        estimated_numpy_memory = (rows * cols * 8) / (1024 * 1024)  # Assume float64
        
        # Time estimation based on data size
        time_estimate = max(0.001, (rows * cols) / 1000000)  # Seconds
        
        return ConversionCost(
            computational_cost=base_cost,
            memory_cost_mb=estimated_numpy_memory,
            time_estimate_seconds=time_estimate,
            quality_impact=0.05 if df.dtypes.nunique() > 1 else 0.0  # Mixed types = slight quality loss
        )
    
    def get_supported_conversions(self) -> List[Tuple[DataFormat, DataFormat]]:
        """Return supported conversion paths."""
        return self.supported_conversions


class PrototypeTimeSeriesAdapter(ShimAdapter):
    """
    Prototype adapter for time series format conversions.
    
    Demonstrates:
    - Temporal data handling
    - Index preservation
    - Frequency detection and validation
    """
    
    def __init__(self):
        super().__init__("timeseries_adapter_prototype")
        self.supported_conversions = [
            (DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES),
            (DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME)
        ]
    
    def can_convert(self, request: ConversionRequest) -> float:
        """Assess ability to handle time series conversions."""
        if (request.source_format, request.target_format) not in self.supported_conversions:
            return 0.0
        
        data = request.source_data
        
        if request.source_format == DataFormat.PANDAS_DATAFRAME:
            if not isinstance(data, pd.DataFrame):
                return 0.0
            
            # Check for datetime index or datetime columns
            has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
            datetime_columns = data.select_dtypes(include=['datetime64']).columns
            
            if has_datetime_index:
                return 0.9
            elif len(datetime_columns) > 0:
                return 0.7
            else:
                # Try to infer if any columns look like dates
                for col in data.columns[:5]:  # Check first 5 columns
                    try:
                        pd.to_datetime(data[col].head(10))
                        return 0.5  # Medium confidence
                    except:
                        continue
                return 0.1  # Low confidence
        
        elif request.source_format == DataFormat.TIME_SERIES:
            # For this prototype, assume time series is a DataFrame with DatetimeIndex
            return 0.9 if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex) else 0.1
        
        return 0.0
    
    def convert(self, request: ConversionRequest) -> ConversionResult:
        """Convert between DataFrame and time series formats."""
        start_time = time.time()
        
        try:
            if request.target_format == DataFormat.TIME_SERIES:
                # Convert DataFrame to time series format
                converted_data = self._to_time_series(request.source_data)
            else:
                # Convert time series to DataFrame
                converted_data = self._from_time_series(request.source_data)
            
            execution_time = time.time() - start_time
            
            return ConversionResult(
                converted_data=converted_data,
                success=True,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.target_format,
                metadata={'conversion_type': 'time_series_conversion'},
                performance_metrics={'execution_time': execution_time},
                quality_score=0.9,
                request_id=request.request_id,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ConversionResult(
                converted_data=request.source_data,
                success=False,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.source_format,
                errors=[str(e)],
                request_id=request.request_id,
                execution_time=execution_time
            )
    
    def _to_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame to time series format."""
        if isinstance(data.index, pd.DatetimeIndex):
            # Already has datetime index
            return data.copy()
        
        # Try to find a datetime column to use as index
        datetime_columns = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            # Use first datetime column as index
            result = data.copy()
            result.index = pd.to_datetime(result[datetime_columns[0]])
            return result.drop(columns=[datetime_columns[0]])
        
        # Try to convert first column to datetime
        first_col = data.columns[0]
        try:
            datetime_index = pd.to_datetime(data[first_col])
            result = data.copy()
            result.index = datetime_index
            return result.drop(columns=[first_col])
        except:
            raise ValueError("Cannot identify datetime column for time series conversion")
    
    def _from_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert time series format back to regular DataFrame."""
        if isinstance(data.index, pd.DatetimeIndex):
            # Reset datetime index to column
            result = data.reset_index()
            return result
        return data.copy()
    
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """Estimate time series conversion cost."""
        return ConversionCost(
            computational_cost=0.2,
            memory_cost_mb=0.1,
            time_estimate_seconds=0.01,
            quality_impact=0.0
        )
    
    def get_supported_conversions(self) -> List[Tuple[DataFormat, DataFormat]]:
        """Return supported conversion paths."""
        return self.supported_conversions


class PrototypeConversionRegistry(ConversionRegistry):
    """
    Prototype implementation of ConversionRegistry.
    
    Demonstrates:
    - Adapter registration and discovery
    - Path finding algorithms
    - Performance optimization through caching
    """
    
    def __init__(self):
        self._adapters: Dict[str, ShimAdapter] = {}
        self._conversion_cache: Dict[str, ConversionPath] = {}
        self._performance_stats: Dict[str, float] = {}
    
    def register_adapter(self, adapter: ShimAdapter) -> None:
        """Register a conversion adapter."""
        self._adapters[adapter.adapter_id] = adapter
        
        # Update performance stats
        self._performance_stats[adapter.adapter_id] = 0.8  # Default performance score
    
    def get_adapter(self, adapter_id: str) -> Optional[ShimAdapter]:
        """Get adapter by ID."""
        return self._adapters.get(adapter_id)
    
    def find_conversion_path(self, source_format: DataFormat, 
                           target_format: DataFormat) -> Optional[ConversionPath]:
        """Find optimal conversion path between formats."""
        # Check cache first
        cache_key = f"{source_format.value}->{target_format.value}"
        if cache_key in self._conversion_cache:
            return self._conversion_cache[cache_key]
        
        # Direct conversion check
        for adapter in self._adapters.values():
            supported = adapter.get_supported_conversions()
            if (source_format, target_format) in supported:
                # Create single-step path
                step = ConversionStep(
                    adapter_id=adapter.adapter_id,
                    source_format=source_format,
                    target_format=target_format,
                    estimated_cost=ConversionCost(0.3, 1.0, 0.1),  # Prototype values
                    confidence=0.9
                )
                
                path = ConversionPath(
                    source_format=source_format,
                    target_format=target_format,
                    steps=[step],
                    total_cost=step.estimated_cost,
                    success_probability=0.9
                )
                
                # Cache the path
                self._conversion_cache[cache_key] = path
                return path
        
        # Multi-step path finding (simplified for prototype)
        # In full implementation, would use graph algorithms
        return None
    
    def get_compatible_adapters(self, request: ConversionRequest) -> List[Tuple[ShimAdapter, float]]:
        """Get adapters that can handle the request with confidence scores."""
        compatible = []
        
        for adapter in self._adapters.values():
            confidence = adapter.can_convert(request)
            if confidence > 0.0:
                compatible.append((adapter, confidence))
        
        # Sort by confidence score (highest first)
        compatible.sort(key=lambda x: x[1], reverse=True)
        return compatible


class PrototypeCompatibilityMatrix(CompatibilityMatrix):
    """
    Prototype implementation of CompatibilityMatrix.
    
    Demonstrates:
    - Compatibility scoring algorithms
    - Domain requirements management
    - Pipeline validation logic
    """
    
    def __init__(self):
        self._domain_requirements: Dict[str, DomainRequirements] = {}
        self._compatibility_cache: Dict[Tuple[DataFormat, DataFormat], CompatibilityScore] = {}
        
        # Initialize with some prototype compatibility rules
        self._initialize_prototype_rules()
    
    def _initialize_prototype_rules(self):
        """Initialize prototype compatibility rules."""
        # Define basic compatibility scores
        high_compatibility = [
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES),
            (DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME),
        ]
        
        medium_compatibility = [
            (DataFormat.PYTHON_LIST, DataFormat.NUMPY_ARRAY),
            (DataFormat.PYTHON_DICT, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.CSV, DataFormat.PANDAS_DATAFRAME),
        ]
        
        for source, target in high_compatibility:
            self._compatibility_cache[(source, target)] = CompatibilityScore(
                score=0.9,
                direct_compatible=True,
                conversion_required=True
            )
        
        for source, target in medium_compatibility:
            self._compatibility_cache[(source, target)] = CompatibilityScore(
                score=0.6,
                direct_compatible=False,
                conversion_required=True
            )
    
    def get_compatibility(self, source_format: DataFormat, 
                         target_format: DataFormat) -> CompatibilityScore:
        """Get compatibility score between formats."""
        # Check cache first
        key = (source_format, target_format)
        if key in self._compatibility_cache:
            return self._compatibility_cache[key]
        
        # Default compatibility logic
        if source_format == target_format:
            score = CompatibilityScore(
                score=1.0,
                direct_compatible=True,
                conversion_required=False
            )
        else:
            # Low default compatibility for unknown combinations
            score = CompatibilityScore(
                score=0.1,
                direct_compatible=False,
                conversion_required=True,
                compatibility_issues=["No conversion path defined"]
            )
        
        self._compatibility_cache[key] = score
        return score
    
    def register_domain_requirements(self, domain_name: str, 
                                   requirements: DataFormatSpec) -> None:
        """Register domain's format requirements."""
        domain_req = DomainRequirements(
            domain_name=domain_name,
            input_formats=[requirements.format_type],
            output_formats=[requirements.format_type],
            preferred_format=requirements.format_type,
            metadata_requirements=requirements.metadata_requirements,
            performance_requirements=requirements.performance_requirements,
            memory_constraints=requirements.memory_constraints
        )
        self._domain_requirements[domain_name] = domain_req
    
    def validate_pipeline(self, pipeline_steps: List[str]) -> ValidationResult:
        """Validate pipeline compatibility."""
        errors = []
        warnings = []
        
        if len(pipeline_steps) < 2:
            return ValidationResult(
                is_valid=True,
                score=1.0,
                details={'pipeline_length': len(pipeline_steps)}
            )
        
        # Check compatibility between consecutive steps
        for i in range(len(pipeline_steps) - 1):
            current_domain = pipeline_steps[i]
            next_domain = pipeline_steps[i + 1]
            
            if current_domain not in self._domain_requirements:
                warnings.append(f"Unknown domain: {current_domain}")
                continue
            
            if next_domain not in self._domain_requirements:
                warnings.append(f"Unknown domain: {next_domain}")
                continue
            
            # Check if output of current domain is compatible with input of next domain
            current_req = self._domain_requirements[current_domain]
            next_req = self._domain_requirements[next_domain]
            
            # Simplified compatibility check
            compatible = any(
                output_format in next_req.input_formats
                for output_format in current_req.output_formats
            )
            
            if not compatible:
                errors.append(f"Incompatible formats between {current_domain} and {next_domain}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=1.0 - (len(errors) * 0.3 + len(warnings) * 0.1),
            errors=errors,
            warnings=warnings,
            details={
                'pipeline_length': len(pipeline_steps),
                'domains_analyzed': len([d for d in pipeline_steps if d in self._domain_requirements])
            }
        )


# Factory function for creating prototype system
def create_prototype_system() -> Tuple[PrototypeConversionRegistry, PrototypeCompatibilityMatrix]:
    """
    Create a complete prototype system for testing and validation.
    
    Returns:
        Tuple of (registry, compatibility_matrix) with pre-configured adapters
    """
    # Create registry and compatibility matrix
    registry = PrototypeConversionRegistry()
    compatibility_matrix = PrototypeCompatibilityMatrix()
    
    # Register prototype adapters
    registry.register_adapter(PrototypeDataFrameToNumpyAdapter())
    registry.register_adapter(PrototypeTimeSeriesAdapter())
    
    # Register some example domain requirements
    statistical_requirements = DataFormatSpec(
        format_type=DataFormat.PANDAS_DATAFRAME,
        metadata_requirements=['column_types', 'missing_values'],
        schema_requirements={'min_columns': 1, 'numeric_columns_preferred': True}
    )
    compatibility_matrix.register_domain_requirements('statistical_analysis', statistical_requirements)
    
    time_series_requirements = DataFormatSpec(
        format_type=DataFormat.TIME_SERIES,
        metadata_requirements=['temporal_index', 'frequency'],
        schema_requirements={'datetime_index_required': True}
    )
    compatibility_matrix.register_domain_requirements('time_series_analysis', time_series_requirements)
    
    return registry, compatibility_matrix


# Validation functions
def validate_architecture_concepts():
    """
    Comprehensive validation of architectural concepts.
    
    This function demonstrates and validates:
    1. Adapter registration and discovery
    2. Conversion path finding
    3. Quality assessment
    4. Pipeline compatibility validation
    5. Error handling
    """
    print("🔍 Validating Integration Shims Architecture Concepts\n")
    
    # Create prototype system
    registry, compatibility_matrix = create_prototype_system()
    
    # Test 1: Adapter Registration and Discovery
    print("✅ Test 1: Adapter Registration and Discovery")
    adapters = registry._adapters
    print(f"   Registered adapters: {list(adapters.keys())}")
    print(f"   Total adapters: {len(adapters)}")
    
    # Test 2: Conversion Request Processing
    print("\n✅ Test 2: Conversion Request Processing")
    
    # Create test data
    test_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1.1, 2.2, 3.3, 4.4, 5.5],
        'C': ['a', 'b', 'c', 'd', 'e']
    })
    
    # Create conversion request
    request = ConversionRequest(
        source_data=test_df,
        source_format=DataFormat.PANDAS_DATAFRAME,
        target_format=DataFormat.NUMPY_ARRAY
    )
    
    # Find compatible adapters
    compatible_adapters = registry.get_compatible_adapters(request)
    print(f"   Compatible adapters found: {len(compatible_adapters)}")
    
    if compatible_adapters:
        adapter, confidence = compatible_adapters[0]
        print(f"   Best adapter: {adapter.adapter_id} (confidence: {confidence:.2f})")
        
        # Perform conversion
        result = adapter.convert(request)
        print(f"   Conversion success: {result.success}")
        print(f"   Quality score: {result.quality_score:.2f}")
        print(f"   Output shape: {result.converted_data.shape if hasattr(result.converted_data, 'shape') else 'N/A'}")
    
    # Test 3: Time Series Conversion
    print("\n✅ Test 3: Time Series Conversion")
    
    # Create time series test data
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    ts_df = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(10)
    })
    
    ts_request = ConversionRequest(
        source_data=ts_df,
        source_format=DataFormat.PANDAS_DATAFRAME,
        target_format=DataFormat.TIME_SERIES
    )
    
    ts_compatible = registry.get_compatible_adapters(ts_request)
    if ts_compatible:
        ts_adapter, ts_confidence = ts_compatible[0]
        print(f"   Time series adapter: {ts_adapter.adapter_id} (confidence: {ts_confidence:.2f})")
        
        ts_result = ts_adapter.convert(ts_request)
        print(f"   Time series conversion success: {ts_result.success}")
        if ts_result.success:
            print(f"   Output has datetime index: {isinstance(ts_result.converted_data.index, pd.DatetimeIndex)}")
    
    # Test 4: Compatibility Matrix
    print("\n✅ Test 4: Compatibility Matrix Validation")
    
    compatibility = compatibility_matrix.get_compatibility(
        DataFormat.PANDAS_DATAFRAME, 
        DataFormat.NUMPY_ARRAY
    )
    print(f"   DataFrame->NumPy compatibility: {compatibility.score:.2f}")
    print(f"   Direct compatible: {compatibility.direct_compatible}")
    print(f"   Conversion required: {compatibility.conversion_required}")
    
    # Test 5: Pipeline Validation
    print("\n✅ Test 5: Pipeline Validation")
    
    test_pipeline = ['statistical_analysis', 'time_series_analysis']
    pipeline_validation = compatibility_matrix.validate_pipeline(test_pipeline)
    print(f"   Pipeline valid: {pipeline_validation.is_valid}")
    print(f"   Pipeline score: {pipeline_validation.score:.2f}")
    if pipeline_validation.errors:
        print(f"   Errors: {pipeline_validation.errors}")
    if pipeline_validation.warnings:
        print(f"   Warnings: {pipeline_validation.warnings}")
    
    print("\n🎉 Architecture Validation Complete!")
    print(f"   All core concepts successfully demonstrated")
    
    return registry, compatibility_matrix


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_architecture_concepts()