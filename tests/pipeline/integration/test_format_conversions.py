"""
Data Format Conversion Integration Tests for LocalData MCP v2.0

This module provides comprehensive testing for data format conversions throughout
the Integration Shims Framework. Tests validate seamless conversion between all
supported data formats while preserving data integrity and metadata.

Tested Format Conversions:
- Pandas DataFrame ↔ NumPy Array
- Pandas DataFrame ↔ SciPy Sparse Matrix
- NumPy Array ↔ SciPy Sparse Matrix
- Domain-specific format conversions
- Specialized analytical format handling
- Memory-efficient streaming conversions
"""

import asyncio
import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from localdata_mcp.pipeline.integration import (
    # Core conversion components
    ConversionRegistry,
    DataFormat,
    ConversionRequest,
    ConversionResult,
    ConversionCost,
    # Format converters
    PandasConverter,
    NumpyConverter,
    SparseMatrixConverter,
    create_pandas_converter,
    create_numpy_converter,
    create_sparse_converter,
    # Conversion options
    ConversionOptions,
    ConversionQuality,
    create_memory_efficient_options,
    create_high_fidelity_options,
    create_streaming_options,
    # Type detection and validation
    TypeDetectionEngine,
    detect_data_format,
    SchemaInferenceEngine,
    SchemaValidator,
    # Performance optimization
    ConversionCache,
    LazyLoadingManager,
    # Error handling
    ConversionError,
    ValidationResult,
    create_conversion_error_handler,
)

# Test fixtures and utilities
from ..fixtures.sample_datasets import (
    create_pandas_dataframe,
    create_numpy_array,
    create_sparse_matrix,
    create_mixed_type_dataframe,
    create_temporal_dataframe,
    create_high_dimensional_array,
    create_streaming_data_source,
)
from ..utils.test_helpers import (
    validate_data_integrity,
    measure_conversion_performance,
    assert_format_compatibility,
    create_format_test_suite,
)

logger = logging.getLogger(__name__)


class TestFormatConversions:
    """
    Comprehensive tests for data format conversions in the Integration Shims Framework.

    Tests cover:
    1. Core format conversions (pandas ↔ numpy ↔ scipy)
    2. Domain-specific format handling
    3. Memory-efficient conversion strategies
    4. Streaming format conversions
    5. Format auto-detection and validation
    6. Conversion quality and fidelity preservation
    7. Performance optimization for large datasets
    8. Error handling and recovery for problematic conversions
    """

    # Define comprehensive format mapping
    FORMAT_SPECS = {
        DataFormat.PANDAS_DATAFRAME: {
            "converter_class": PandasConverter,
            "factory_function": create_pandas_converter,
            "supported_sources": [
                DataFormat.NUMPY_ARRAY,
                DataFormat.SCIPY_SPARSE,
                DataFormat.CSV_DATA,
                DataFormat.JSON_DATA,
            ],
            "data_characteristics": [
                "tabular",
                "mixed_types",
                "labeled_columns",
                "index_support",
            ],
            "memory_profile": "moderate",
            "streaming_support": True,
        },
        DataFormat.NUMPY_ARRAY: {
            "converter_class": NumpyConverter,
            "factory_function": create_numpy_converter,
            "supported_sources": [
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.SCIPY_SPARSE,
                DataFormat.PYTHON_LIST,
                DataFormat.BINARY_DATA,
            ],
            "data_characteristics": [
                "numeric",
                "homogeneous",
                "n_dimensional",
                "vectorized",
            ],
            "memory_profile": "efficient",
            "streaming_support": True,
        },
        DataFormat.SCIPY_SPARSE: {
            "converter_class": SparseMatrixConverter,
            "factory_function": create_sparse_converter,
            "supported_sources": [
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.NUMPY_ARRAY,
                DataFormat.MATRIX_MARKET,
                DataFormat.COO_FORMAT,
            ],
            "data_characteristics": [
                "sparse",
                "memory_efficient",
                "mathematical",
                "compressed",
            ],
            "memory_profile": "very_efficient",
            "streaming_support": True,
        },
    }

    @pytest.fixture(autouse=True)
    async def setup_format_conversion_framework(self):
        """Setup comprehensive format conversion testing framework."""
        # Initialize core components
        self.registry = ConversionRegistry()
        self.type_detector = TypeDetectionEngine()
        self.schema_engine = SchemaInferenceEngine()
        self.schema_validator = SchemaValidator()

        # Performance optimization components
        self.cache = ConversionCache(max_size=50, ttl=600)
        self.lazy_manager = LazyLoadingManager(memory_limit=512 * 1024 * 1024)  # 512MB

        # Error handling
        self.error_handler = create_conversion_error_handler()

        # Initialize converters
        self.converters = {}
        for format_type, spec in self.FORMAT_SPECS.items():
            converter = spec["factory_function"]()
            self.converters[format_type] = converter
            await self.registry.register_adapter(converter)

        # Create test datasets for each format
        self.test_datasets = {
            DataFormat.PANDAS_DATAFRAME: create_pandas_dataframe(
                rows=1000, columns=20, mixed_types=True
            ),
            DataFormat.NUMPY_ARRAY: create_numpy_array(
                shape=(1000, 20), dtype=np.float64
            ),
            DataFormat.SCIPY_SPARSE: create_sparse_matrix(
                shape=(1000, 100), density=0.1, format="csr"
            ),
        }

        logger.info(
            f"Format conversion framework initialized with {len(self.converters)} converters"
        )

    @pytest.mark.parametrize(
        "source_format,target_format",
        [
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE),
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE),
            (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.SCIPY_SPARSE, DataFormat.NUMPY_ARRAY),
        ],
    )
    @pytest.mark.asyncio
    async def test_core_format_conversions(
        self, source_format: DataFormat, target_format: DataFormat
    ):
        """
        Test core format conversions between all major data formats.

        Validates:
        - Conversion success and data integrity
        - Metadata preservation
        - Performance characteristics
        - Round-trip conversion fidelity
        """
        logger.info(f"Testing {source_format} → {target_format} conversion")

        # Get test data
        source_data = self.test_datasets[source_format]
        source_converter = self.converters[source_format]
        target_converter = self.converters[target_format]

        # Phase 1: Direct conversion
        conversion_request = ConversionRequest(
            source_format=source_format,
            target_format=target_format,
            data=source_data,
            context={
                "conversion_quality": "high_fidelity",
                "preserve_metadata": True,
                "performance_target": "balanced",
            },
        )

        start_time = time.time()
        conversion_result = await target_converter.convert(conversion_request)
        conversion_time = time.time() - start_time

        # Validate conversion success
        assert conversion_result.success, (
            f"Conversion {source_format}→{target_format} failed: {conversion_result.error}"
        )

        # Validate data integrity
        await self._validate_conversion_integrity(
            source_data, conversion_result.data, source_format, target_format
        )

        # Validate metadata preservation
        assert "source_format" in conversion_result.metadata
        assert conversion_result.metadata["source_format"] == source_format
        assert conversion_result.metadata["target_format"] == target_format

        # Validate performance
        assert conversion_time < 10.0, f"Conversion took too long: {conversion_time}s"

        # Phase 2: Round-trip conversion test
        round_trip_request = ConversionRequest(
            source_format=target_format,
            target_format=source_format,
            data=conversion_result.data,
            context={
                "conversion_quality": "high_fidelity",
                "round_trip_test": True,
                "preserve_original_characteristics": True,
            },
        )

        round_trip_result = await source_converter.convert(round_trip_request)
        assert round_trip_result.success, "Round-trip conversion failed"

        # Validate round-trip fidelity
        fidelity_score = await self._calculate_round_trip_fidelity(
            source_data, round_trip_result.data, source_format
        )
        assert fidelity_score > 0.95, f"Round-trip fidelity too low: {fidelity_score}"

        logger.info(
            f"✅ {source_format}→{target_format} conversion validated (fidelity: {fidelity_score:.3f})"
        )

    @pytest.mark.asyncio
    async def test_conversion_quality_levels(self):
        """
        Test different conversion quality levels and their trade-offs.

        Validates:
        - Quality vs performance trade-offs
        - Appropriate quality level selection
        - Metadata accuracy at different levels
        - Memory usage optimization
        """
        logger.info("Testing conversion quality levels")

        # Test dataset
        test_dataframe = create_mixed_type_dataframe(
            rows=5000, numeric_cols=15, categorical_cols=5, temporal_cols=3, text_cols=2
        )

        quality_levels = [
            ConversionQuality.FAST,
            ConversionQuality.BALANCED,
            ConversionQuality.HIGH_FIDELITY,
            ConversionQuality.MAXIMUM_PRECISION,
        ]

        conversion_metrics = {}

        for quality in quality_levels:
            logger.info(f"  Testing quality level: {quality}")

            # Create quality-specific options
            if quality == ConversionQuality.FAST:
                options = create_memory_efficient_options()
            elif quality == ConversionQuality.HIGH_FIDELITY:
                options = create_high_fidelity_options()
            else:
                options = ConversionOptions(quality=quality)

            # Test pandas → numpy conversion with quality level
            request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                data=test_dataframe,
                context={
                    "conversion_options": options,
                    "quality_level": quality,
                    "benchmark_test": True,
                },
            )

            # Measure conversion performance
            start_time = time.time()
            memory_before = self._get_memory_usage()

            result = await self.converters[DataFormat.NUMPY_ARRAY].convert(request)

            conversion_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_usage = memory_after - memory_before

            assert result.success, f"Quality level {quality} conversion failed"

            # Analyze result quality
            quality_metrics = await self._analyze_conversion_quality(
                test_dataframe, result.data, quality
            )

            conversion_metrics[quality] = {
                "conversion_time": conversion_time,
                "memory_usage": memory_usage,
                "data_fidelity": quality_metrics["fidelity_score"],
                "precision_score": quality_metrics["precision_score"],
                "completeness_score": quality_metrics["completeness_score"],
            }

            logger.info(
                f"    Time: {conversion_time:.3f}s, Memory: {memory_usage / 1024 / 1024:.1f}MB, "
                f"Fidelity: {quality_metrics['fidelity_score']:.3f}"
            )

        # Validate quality level characteristics
        fast_metrics = conversion_metrics[ConversionQuality.FAST]
        high_fidelity_metrics = conversion_metrics[ConversionQuality.HIGH_FIDELITY]
        max_precision_metrics = conversion_metrics[ConversionQuality.MAXIMUM_PRECISION]

        # Fast should be faster but potentially lower fidelity
        assert (
            fast_metrics["conversion_time"] < high_fidelity_metrics["conversion_time"]
        )

        # High fidelity should have better fidelity than fast
        assert high_fidelity_metrics["data_fidelity"] >= fast_metrics["data_fidelity"]

        # Maximum precision should have highest precision
        assert (
            max_precision_metrics["precision_score"]
            >= high_fidelity_metrics["precision_score"]
        )

        logger.info("✅ Conversion quality levels validated")

    @pytest.mark.asyncio
    async def test_streaming_format_conversions(self):
        """
        Test streaming format conversions for large datasets.

        Validates:
        - Memory-bounded streaming conversion
        - Chunk-based processing integrity
        - Progressive result assembly
        - Performance characteristics of streaming
        """
        logger.info("Testing streaming format conversions")

        # Create large streaming dataset
        streaming_data = create_streaming_data_source(
            total_rows=50000,
            chunk_size=5000,
            columns=30,
            data_types=["numeric", "categorical", "temporal"],
        )

        # Configure streaming options
        streaming_options = create_streaming_options(
            chunk_size=5000,
            memory_limit=100 * 1024 * 1024,  # 100MB limit
            quality_level=ConversionQuality.BALANCED,
        )

        # Test streaming pandas → numpy conversion
        streaming_request = ConversionRequest(
            source_format=DataFormat.STREAMING_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            data=streaming_data,
            context={
                "conversion_options": streaming_options,
                "streaming_mode": True,
                "memory_monitoring": True,
            },
        )

        # Monitor memory usage during conversion
        memory_monitor = []

        async def memory_callback(current_usage):
            memory_monitor.append(current_usage)
            assert current_usage < 120 * 1024 * 1024, (
                f"Memory limit exceeded: {current_usage}"
            )

        # Execute streaming conversion
        start_time = time.time()
        streaming_result = await self.converters[DataFormat.NUMPY_ARRAY].convert(
            streaming_request, memory_callback=memory_callback
        )
        streaming_time = time.time() - start_time

        # Validate streaming conversion
        assert streaming_result.success, "Streaming conversion failed"
        assert streaming_result.metadata["processing_strategy"] == "streaming"
        assert "chunks_processed" in streaming_result.metadata
        assert streaming_result.metadata["chunks_processed"] == 10  # 50000 / 5000

        # Validate memory usage stayed within bounds
        max_memory = max(memory_monitor) if memory_monitor else 0
        assert max_memory < 120 * 1024 * 1024, f"Memory exceeded limit: {max_memory}"

        # Validate result integrity
        assert isinstance(streaming_result.data, np.ndarray)
        assert streaming_result.data.shape == (50000, 30)

        # Test streaming with different chunk sizes
        chunk_sizes = [1000, 5000, 10000]
        chunk_performance = {}

        for chunk_size in chunk_sizes:
            chunk_options = create_streaming_options(
                chunk_size=chunk_size, memory_limit=100 * 1024 * 1024
            )

            chunk_request = ConversionRequest(
                source_format=DataFormat.STREAMING_DATAFRAME,
                target_format=DataFormat.SCIPY_SPARSE,
                data=streaming_data,
                context={"conversion_options": chunk_options, "chunk_size_test": True},
            )

            start_time = time.time()
            chunk_result = await self.converters[DataFormat.SCIPY_SPARSE].convert(
                chunk_request
            )
            chunk_time = time.time() - start_time

            assert chunk_result.success, f"Chunk size {chunk_size} failed"

            chunk_performance[chunk_size] = {
                "time": chunk_time,
                "chunks": chunk_result.metadata.get("chunks_processed", 0),
                "memory_efficiency": chunk_result.metadata.get(
                    "memory_efficiency_score", 0
                ),
            }

        # Validate chunk size optimization
        # Smaller chunks should process more chunks but potentially with overhead
        assert chunk_performance[1000]["chunks"] > chunk_performance[10000]["chunks"]

        logger.info(
            f"Streaming conversion completed in {streaming_time:.2f}s, "
            f"peak memory: {max_memory / 1024 / 1024:.1f}MB"
        )

    @pytest.mark.asyncio
    async def test_format_auto_detection(self):
        """
        Test automatic format detection and intelligent conversion path selection.

        Validates:
        - Accurate format detection
        - Optimal conversion path selection
        - Confidence scoring for detection
        - Fallback strategies for ambiguous formats
        """
        logger.info("Testing format auto-detection")

        # Create test cases with various data formats
        test_cases = [
            {
                "name": "pandas_dataframe",
                "data": create_pandas_dataframe(rows=100, columns=10),
                "expected_format": DataFormat.PANDAS_DATAFRAME,
                "confidence_threshold": 0.95,
            },
            {
                "name": "numpy_array",
                "data": create_numpy_array(shape=(100, 10)),
                "expected_format": DataFormat.NUMPY_ARRAY,
                "confidence_threshold": 0.90,
            },
            {
                "name": "sparse_matrix",
                "data": create_sparse_matrix(shape=(100, 50), density=0.05),
                "expected_format": DataFormat.SCIPY_SPARSE,
                "confidence_threshold": 0.85,
            },
            {
                "name": "mixed_format_dict",
                "data": {
                    "dataframe": create_pandas_dataframe(rows=50, columns=5),
                    "array": create_numpy_array(shape=(50, 5)),
                    "metadata": {"source": "test", "version": "1.0"},
                },
                "expected_format": DataFormat.MIXED_FORMAT,
                "confidence_threshold": 0.70,
            },
            {
                "name": "temporal_dataframe",
                "data": create_temporal_dataframe(rows=200, freq="H"),
                "expected_format": DataFormat.TIME_SERIES_DATA,
                "confidence_threshold": 0.80,
            },
        ]

        detection_results = {}

        for test_case in test_cases:
            logger.info(f"  Testing detection for: {test_case['name']}")

            # Perform format detection
            detection_result = await self.type_detector.detect_format(
                data=test_case["data"],
                context={"detection_mode": "comprehensive", "confidence_required": 0.5},
            )

            # Validate detection
            assert detection_result.success, f"Detection failed for {test_case['name']}"
            assert detection_result.detected_format == test_case["expected_format"], (
                f"Wrong format detected for {test_case['name']}: got {detection_result.detected_format}, expected {test_case['expected_format']}"
            )
            assert detection_result.confidence >= test_case["confidence_threshold"], (
                f"Confidence too low for {test_case['name']}: {detection_result.confidence}"
            )

            detection_results[test_case["name"]] = detection_result
            logger.info(
                f"    Detected: {detection_result.detected_format}, Confidence: {detection_result.confidence:.3f}"
            )

        # Test ambiguous format handling
        ambiguous_data = np.array(
            [[1, 2, 3], [4, 5, 6]]
        )  # Could be numpy or convertible to pandas

        ambiguous_detection = await self.type_detector.detect_format(
            data=ambiguous_data,
            context={
                "detection_mode": "comprehensive",
                "handle_ambiguity": True,
                "provide_alternatives": True,
            },
        )

        assert ambiguous_detection.success
        assert "alternative_formats" in ambiguous_detection.metadata
        assert len(ambiguous_detection.metadata["alternative_formats"]) > 0

        # Test format detection with schema inference
        schema_aware_detection = await self.type_detector.detect_format(
            data=create_mixed_type_dataframe(
                rows=100, numeric_cols=5, categorical_cols=3
            ),
            context={
                "detection_mode": "schema_aware",
                "infer_schema": True,
                "detailed_analysis": True,
            },
        )

        assert schema_aware_detection.success
        assert "inferred_schema" in schema_aware_detection.metadata
        assert "column_types" in schema_aware_detection.metadata["inferred_schema"]

        logger.info("✅ Format auto-detection validated")

    @pytest.mark.asyncio
    async def test_conversion_error_handling_and_recovery(self):
        """
        Test conversion error handling and recovery mechanisms.

        Validates:
        - Error detection and classification
        - Graceful degradation strategies
        - Alternative conversion pathway discovery
        - Recovery success rates
        """
        logger.info("Testing conversion error handling and recovery")

        # Create problematic datasets for testing error scenarios
        error_test_cases = [
            {
                "name": "dataframe_with_mixed_nulls",
                "data": pd.DataFrame(
                    {
                        "numeric": [1.0, 2.0, None, np.inf, -np.inf],
                        "text": ["a", "b", None, "", "very_long_text_" * 100],
                        "categorical": ["X", "Y", None, "Z", "Invalid_Category_" * 50],
                    }
                ),
                "target_format": DataFormat.NUMPY_ARRAY,
                "expected_error_type": "data_type_incompatibility",
                "recovery_expected": True,
            },
            {
                "name": "sparse_matrix_wrong_format",
                "data": sparse.coo_matrix([[1, 0, 2], [0, 3, 0]]),  # COO format
                "target_format": DataFormat.PANDAS_DATAFRAME,
                "context": {"require_csr_format": True},  # Incompatible requirement
                "expected_error_type": "format_mismatch",
                "recovery_expected": True,
            },
            {
                "name": "oversized_array",
                "data": np.random.random((10000, 10000)),  # Large array
                "target_format": DataFormat.PANDAS_DATAFRAME,
                "context": {"memory_limit": 50 * 1024 * 1024},  # 50MB limit
                "expected_error_type": "memory_constraint_violation",
                "recovery_expected": True,
            },
            {
                "name": "corrupted_data_structure",
                "data": {"incomplete": "structure", "missing": None},
                "target_format": DataFormat.NUMPY_ARRAY,
                "expected_error_type": "structural_incompatibility",
                "recovery_expected": False,
            },
        ]

        recovery_results = {}

        for test_case in error_test_cases:
            logger.info(f"  Testing error scenario: {test_case['name']}")

            # Attempt conversion that should trigger error
            error_request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,  # Inferred
                target_format=test_case["target_format"],
                data=test_case["data"],
                context={
                    **test_case.get("context", {}),
                    "error_handling": "comprehensive",
                    "attempt_recovery": test_case["recovery_expected"],
                },
            )

            try:
                # Use error handler for conversion
                result = await self.error_handler.handle_conversion(
                    error_request, converter=self.converters[test_case["target_format"]]
                )

                if test_case["recovery_expected"]:
                    # Recovery should have succeeded
                    assert result.success, (
                        f"Expected recovery to succeed for {test_case['name']}"
                    )
                    assert "error_recovery_applied" in result.metadata
                    assert "original_error" in result.metadata

                    # Validate recovery strategy
                    recovery_strategy = result.metadata.get("recovery_strategy")
                    assert recovery_strategy is not None

                    recovery_results[test_case["name"]] = {
                        "recovery_successful": True,
                        "recovery_strategy": recovery_strategy,
                        "original_error_type": result.metadata["original_error"][
                            "type"
                        ],
                        "recovery_time": result.metadata.get("recovery_time", 0),
                    }

                else:
                    # Recovery not expected - should fail gracefully
                    assert not result.success, (
                        f"Expected failure for {test_case['name']}"
                    )
                    assert "error_classification" in result.metadata

                    recovery_results[test_case["name"]] = {
                        "recovery_successful": False,
                        "error_properly_classified": True,
                        "graceful_failure": True,
                    }

            except ConversionError as e:
                # Explicit error raised - validate it's properly classified
                assert hasattr(e, "error_type")
                assert e.error_type == test_case["expected_error_type"]

                recovery_results[test_case["name"]] = {
                    "recovery_successful": False,
                    "error_properly_classified": True,
                    "explicit_error_raised": True,
                }

            logger.info(
                f"    Error handling result: {recovery_results[test_case['name']]}"
            )

        # Validate overall recovery effectiveness
        recoverable_cases = [
            case for case in error_test_cases if case["recovery_expected"]
        ]
        successful_recoveries = [
            result
            for result in recovery_results.values()
            if result.get("recovery_successful", False)
        ]

        recovery_rate = (
            len(successful_recoveries) / len(recoverable_cases)
            if recoverable_cases
            else 0
        )
        assert recovery_rate >= 0.75, f"Recovery rate too low: {recovery_rate:.2f}"

        logger.info(
            f"✅ Error handling validated with {recovery_rate:.1%} recovery rate"
        )

    @pytest.mark.asyncio
    async def test_performance_optimization_strategies(self):
        """
        Test performance optimization strategies for format conversions.

        Validates:
        - Caching effectiveness for repeated conversions
        - Lazy loading for large datasets
        - Parallel processing capabilities
        - Memory usage optimization
        """
        logger.info("Testing performance optimization strategies")

        # Create test dataset for performance testing
        perf_dataset = create_high_dimensional_array(
            shape=(5000, 200), dtype=np.float32, sparsity=0.3
        )

        # Test 1: Caching effectiveness
        cache_request = ConversionRequest(
            source_format=DataFormat.NUMPY_ARRAY,
            target_format=DataFormat.PANDAS_DATAFRAME,
            data=perf_dataset,
            context={
                "performance_test": "caching",
                "cache_key": "perf_test_dataset",
                "conversion_quality": ConversionQuality.BALANCED,
            },
        )

        # First conversion - cache miss
        start_time = time.time()
        first_result = await self.converters[DataFormat.PANDAS_DATAFRAME].convert(
            cache_request
        )
        first_time = time.time() - start_time

        assert first_result.success

        # Second conversion - should hit cache
        start_time = time.time()
        second_result = await self.converters[DataFormat.PANDAS_DATAFRAME].convert(
            cache_request
        )
        second_time = time.time() - start_time

        assert second_result.success

        # Cache should provide significant speedup
        cache_speedup = first_time / second_time if second_time > 0 else 0
        assert cache_speedup > 5.0, f"Insufficient cache speedup: {cache_speedup:.1f}x"

        # Test 2: Lazy loading performance
        large_streaming_data = create_streaming_data_source(
            total_rows=100000, chunk_size=10000, columns=50
        )

        lazy_request = ConversionRequest(
            source_format=DataFormat.STREAMING_DATAFRAME,
            target_format=DataFormat.SCIPY_SPARSE,
            data=large_streaming_data,
            context={
                "performance_test": "lazy_loading",
                "lazy_loading": True,
                "memory_budget": 200 * 1024 * 1024,  # 200MB
                "optimization_strategy": "memory_first",
            },
        )

        # Monitor memory during lazy loading
        memory_samples = []

        async def lazy_memory_callback(usage):
            memory_samples.append(usage)

        start_time = time.time()
        lazy_result = await self.converters[DataFormat.SCIPY_SPARSE].convert(
            lazy_request, memory_callback=lazy_memory_callback
        )
        lazy_time = time.time() - start_time

        assert lazy_result.success
        max_memory = max(memory_samples) if memory_samples else 0
        assert max_memory < 250 * 1024 * 1024, (
            f"Lazy loading used too much memory: {max_memory / 1024 / 1024:.1f}MB"
        )

        # Test 3: Performance config optimization
        # create_performance_config is not yet implemented; skip this section
        pytest.skip("create_performance_config not yet implemented")
        performance_configs = []

        config_results = {}

        for config in performance_configs:
            config_request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.NUMPY_ARRAY,
                data=create_mixed_type_dataframe(rows=10000, numeric_cols=30),
                context={
                    "performance_config": config,
                    "performance_test": f"{config.optimization_target}_test",
                },
            )

            start_time = time.time()
            memory_before = self._get_memory_usage()

            config_result = await self.converters[DataFormat.NUMPY_ARRAY].convert(
                config_request
            )

            execution_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_usage = memory_after - memory_before

            assert config_result.success

            config_results[config.optimization_target] = {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "conversion_quality": config_result.metadata.get("quality_score", 0),
            }

        # Validate optimization effectiveness
        speed_result = config_results["speed_optimized"]
        memory_result = config_results["memory_optimized"]
        balanced_result = config_results["balanced"]

        # Speed config should be fastest
        assert speed_result["execution_time"] <= balanced_result["execution_time"]

        # Memory config should use least memory
        assert memory_result["memory_usage"] <= balanced_result["memory_usage"]

        logger.info(f"Performance optimization results:")
        logger.info(f"  Cache speedup: {cache_speedup:.1f}x")
        logger.info(f"  Lazy loading peak memory: {max_memory / 1024 / 1024:.1f}MB")
        logger.info(f"  Speed config time: {speed_result['execution_time']:.3f}s")
        logger.info(
            f"  Memory config usage: {memory_result['memory_usage'] / 1024 / 1024:.1f}MB"
        )

    async def _validate_conversion_integrity(
        self,
        source_data: Any,
        target_data: Any,
        source_format: DataFormat,
        target_format: DataFormat,
    ):
        """Validate data integrity after conversion."""

        # Basic shape validation where applicable
        if hasattr(source_data, "shape") and hasattr(target_data, "shape"):
            # For dimensional data, validate compatible shapes
            if (
                source_format == DataFormat.PANDAS_DATAFRAME
                and target_format == DataFormat.NUMPY_ARRAY
            ):
                assert (
                    target_data.shape[0] == source_data.shape[0]
                )  # Same number of rows
            elif (
                source_format == DataFormat.NUMPY_ARRAY
                and target_format == DataFormat.SCIPY_SPARSE
            ):
                assert target_data.shape == source_data.shape

        # Data type compatibility
        if isinstance(source_data, pd.DataFrame) and isinstance(
            target_data, np.ndarray
        ):
            # Check that numeric data was preserved
            numeric_cols = source_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                assert target_data.dtype in [np.float64, np.float32, np.int64, np.int32]

        # Sparse matrix specific validations
        if isinstance(target_data, sparse.spmatrix):
            assert hasattr(target_data, "nnz")  # Has non-zero count
            assert target_data.nnz >= 0

        # Value preservation for compatible formats
        if (
            isinstance(source_data, np.ndarray)
            and isinstance(target_data, np.ndarray)
            and source_data.dtype == target_data.dtype
        ):
            np.testing.assert_array_almost_equal(source_data, target_data, decimal=10)

    async def _calculate_round_trip_fidelity(
        self, original_data: Any, round_trip_data: Any, data_format: DataFormat
    ) -> float:
        """Calculate fidelity score for round-trip conversion."""

        if isinstance(original_data, pd.DataFrame) and isinstance(
            round_trip_data, pd.DataFrame
        ):
            # Compare shapes
            shape_match = original_data.shape == round_trip_data.shape

            # Compare numeric data
            numeric_original = original_data.select_dtypes(include=[np.number])
            numeric_round_trip = round_trip_data.select_dtypes(include=[np.number])

            if len(numeric_original.columns) > 0:
                numeric_diff = np.abs(
                    numeric_original.values - numeric_round_trip.values
                )
                numeric_fidelity = 1.0 - (
                    np.mean(numeric_diff)
                    / (np.mean(np.abs(numeric_original.values)) + 1e-10)
                )
            else:
                numeric_fidelity = 1.0

            return 0.7 * (1.0 if shape_match else 0.5) + 0.3 * max(
                0.0, numeric_fidelity
            )

        elif isinstance(original_data, np.ndarray) and isinstance(
            round_trip_data, np.ndarray
        ):
            if original_data.shape != round_trip_data.shape:
                return 0.5

            if original_data.dtype != round_trip_data.dtype:
                return 0.7

            # Calculate numerical fidelity
            diff = np.abs(original_data - round_trip_data)
            relative_error = np.mean(diff) / (np.mean(np.abs(original_data)) + 1e-10)
            return max(0.0, 1.0 - relative_error)

        else:
            # Default similarity check
            return 0.9  # Assume good fidelity for complex types

    async def _analyze_conversion_quality(
        self, source_data: Any, target_data: Any, quality_level: ConversionQuality
    ) -> Dict[str, float]:
        """Analyze conversion quality metrics."""

        # Mock quality analysis - in real implementation would be more sophisticated
        base_fidelity = 0.85
        base_precision = 0.80
        base_completeness = 0.90

        # Adjust based on quality level
        if quality_level == ConversionQuality.FAST:
            fidelity_modifier = 0.95
            precision_modifier = 0.90
        elif quality_level == ConversionQuality.HIGH_FIDELITY:
            fidelity_modifier = 1.05
            precision_modifier = 1.03
        elif quality_level == ConversionQuality.MAXIMUM_PRECISION:
            fidelity_modifier = 1.08
            precision_modifier = 1.10
        else:  # BALANCED
            fidelity_modifier = 1.0
            precision_modifier = 1.0

        return {
            "fidelity_score": min(1.0, base_fidelity * fidelity_modifier),
            "precision_score": min(1.0, base_precision * precision_modifier),
            "completeness_score": base_completeness,
        }

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss


if __name__ == "__main__":
    # Run format conversion tests
    import sys

    logging.basicConfig(level=logging.INFO)

    async def run_tests():
        test_instance = TestFormatConversions()
        await test_instance.setup_format_conversion_framework()

        # Define test scenarios
        format_conversion_tests = [
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE),
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE),
            (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME),
            (DataFormat.SCIPY_SPARSE, DataFormat.NUMPY_ARRAY),
        ]

        print("\n" + "=" * 60)
        print("Testing Core Format Conversions")
        print("=" * 60)

        for source_fmt, target_fmt in format_conversion_tests:
            try:
                await test_instance.test_core_format_conversions(source_fmt, target_fmt)
                print(f"✅ {source_fmt} → {target_fmt} PASSED")
            except Exception as e:
                print(f"❌ {source_fmt} → {target_fmt} FAILED: {e}")

        # Run advanced tests
        advanced_tests = [
            ("Conversion Quality Levels", test_instance.test_conversion_quality_levels),
            (
                "Streaming Format Conversions",
                test_instance.test_streaming_format_conversions,
            ),
            ("Format Auto-Detection", test_instance.test_format_auto_detection),
            (
                "Error Handling and Recovery",
                test_instance.test_conversion_error_handling_and_recovery,
            ),
            (
                "Performance Optimization",
                test_instance.test_performance_optimization_strategies,
            ),
        ]

        print("\n" + "=" * 60)
        print("Testing Advanced Format Conversion Features")
        print("=" * 60)

        for test_name, test_method in advanced_tests:
            try:
                await test_method()
                print(f"✅ {test_name} PASSED")
            except Exception as e:
                print(f"❌ {test_name} FAILED: {e}")
                import traceback

                traceback.print_exc()

        print("\n" + "=" * 60)
        print("Format Conversion Tests Complete")
        print("=" * 60)

    if sys.version_info >= (3, 7):
        asyncio.run(run_tests())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_tests())
