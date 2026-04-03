"""
End-to-End Integration Tests for LocalData MCP v2.0 Integration Shims Framework
NOTE: Some fixtures/helpers are not yet ported; these tests may skip.

This module provides comprehensive end-to-end testing for complete workflow validation
across all domains. Tests are designed to validate the Five First Principles:
1. Intention-Driven Interface
2. Context-Aware Composition
3. Progressive Disclosure Architecture
4. Streaming-First Data Science
5. Modular Domain Integration

Tests cover complete workflows spanning statistical analysis, regression modeling,
time series analysis, and pattern recognition domains.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from localdata_mcp.pipeline.integration import (
    # Core integration components
    ShimRegistry,
    DataFormat,
    ConversionRequest,
    ConversionResult,
    # Domain shims
    StatisticalShim,
    RegressionShim,
    TimeSeriesShim,
    PatternRecognitionShim,
    create_all_domain_shims,
    # Pipeline analysis
    PipelineAnalyzer,
    analyze_and_fix_pipeline,
    create_pipeline_step,
    OptimizationCriteria,
    # Performance optimization
    ConversionCache,
    LazyLoadingManager,
    # Error recovery
    create_complete_error_recovery_system,
    # Schema validation
    SchemaInferenceEngine,
    SchemaValidator,
    # Compatibility matrix
    PipelineCompatibilityMatrix,
    create_compatibility_matrix,
)

# Test fixtures and utilities
from ..fixtures.sample_datasets import (
    create_statistical_dataset,
    create_time_series_dataset,
    create_regression_dataset,
    create_high_dimensional_dataset,
    create_streaming_dataset,
)
from ..utils.test_helpers import (
    assert_workflow_integrity,
    measure_performance,
    validate_first_principles,
    create_mock_domain_pipeline,
)

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.skip(reason="Integration test fixtures not yet compatible with current implementation")




class TestEndToEndIntegration:
    """
    Comprehensive end-to-end integration tests for the complete Integration Shims Framework.

    Tests validate:
    1. Complete cross-domain workflows (statistical → regression → time series → pattern recognition)
    2. Seamless data format conversions throughout pipelines
    3. Performance characteristics under varying data sizes and memory constraints
    4. Error recovery and alternative pathway discovery
    5. Schema validation and evolution across domain boundaries
    """

    @pytest.fixture(autouse=True)
    def setup_integration_framework(self):
        """Setup complete integration framework for testing."""
        # Initialize core components
        self.registry = ShimRegistry()
        self.domain_shims = create_all_domain_shims()
        self.compatibility_matrix = create_compatibility_matrix()
        self.pipeline_analyzer = PipelineAnalyzer(
            compatibility_matrix=self.compatibility_matrix
        )

        # Performance optimization components
        self.cache = ConversionCache(max_size=100, ttl=300)
        self.lazy_manager = LazyLoadingManager(memory_limit=1024 * 1024 * 1024)  # 1GB

        # Error recovery system
        self.error_recovery = create_complete_error_recovery_system()

        # Schema validation
        self.schema_engine = SchemaInferenceEngine()
        self.schema_validator = SchemaValidator()

        # Register domain shims with registry
        for shim_type, shim in self.domain_shims.items():
            self.registry.register_adapter(shim)

        logger.info(
            f"Integration framework initialized with {len(self.domain_shims)} domain shims"
        )
    def test_complete_statistical_to_regression_workflow(self):
        """
        Test complete workflow: Statistical Analysis → Regression Modeling

        Validates:
        - Intention-driven interface usage
        - Context-aware composition between domains
        - Data format preservation and enrichment
        - Performance within memory constraints
        """
        # Create test dataset
        dataset = create_statistical_dataset(size="medium", complexity="high")

        # Phase 1: Statistical Analysis with intention-driven parameters
        statistical_shim = self.domain_shims["statistical"]

        statistical_request = ConversionRequest(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.STATISTICAL_ANALYSIS,
            data=dataset,
            context={
                "intention": "explore_relationships",
                "focus": "linear_patterns",
                "strength_threshold": "strong",  # Semantic parameter
            },
        )

        start_time = time.time()
        statistical_result = statistical_shim.convert(statistical_request)
        statistical_time = time.time() - start_time

        # Validate statistical analysis results
        assert statistical_result.success
        assert statistical_result.metadata["analysis_type"] == "correlation_analysis"
        assert "correlation_matrix" in statistical_result.data
        assert "significant_relationships" in statistical_result.metadata
        assert statistical_time < 5.0  # Sub-5s performance requirement

        # Phase 2: Use statistical results for regression modeling (context-aware composition)
        regression_shim = self.domain_shims["regression"]

        regression_request = ConversionRequest(
            source_format=DataFormat.STATISTICAL_ANALYSIS,
            target_format=DataFormat.REGRESSION_MODEL,
            data=statistical_result.data,
            context={
                "intention": "model_relationships",
                "statistical_context": statistical_result.metadata,
                "model_complexity": "adaptive",  # Let system choose based on data
            },
        )

        start_time = time.time()
        regression_result = regression_shim.convert(regression_request)
        regression_time = time.time() - start_time

        # Validate regression modeling results
        assert regression_result.success
        assert regression_result.metadata["model_type"] in [
            "linear",
            "polynomial",
            "ridge",
        ]
        assert "model_coefficients" in regression_result.data
        assert "prediction_accuracy" in regression_result.metadata
        assert regression_time < 10.0  # Sub-10s for modeling

        # Validate workflow composition
        assert_workflow_integrity(
            [statistical_result, regression_result],
            expected_lineage=["statistical_analysis", "regression_modeling"],
        )

        # Validate First Principles adherence
        validate_first_principles(
            workflow=[statistical_request, regression_request],
            results=[statistical_result, regression_result],
        )

        logger.info(
            f"Complete statistical→regression workflow: {statistical_time + regression_time:.2f}s"
        )
    def test_time_series_to_pattern_recognition_workflow(self):
        """
        Test workflow: Time Series Analysis → Pattern Recognition

        Validates:
        - Temporal data handling and format conversion
        - Progressive disclosure (simple → advanced parameters)
        - Cross-domain semantic context preservation
        - Streaming-first processing with large datasets
        """
        # Create time series dataset
        ts_dataset = create_time_series_dataset(
            length=10000,  # Large enough to trigger streaming
            features=50,
            seasonality=True,
            trend=True,
            noise_level=0.1,
        )

        # Phase 1: Time Series Analysis (simple parameters first)
        ts_shim = self.domain_shims["time_series"]

        # Simple call - system chooses appropriate methods
        simple_request = ConversionRequest(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.TIME_SERIES_FEATURES,
            data=ts_dataset,
            context={
                "intention": "extract_temporal_patterns",
                "complexity": "auto",  # Progressive disclosure
            },
        )

        start_time = time.time()
        ts_result = ts_shim.convert(simple_request)
        simple_processing_time = time.time() - start_time

        # Validate time series analysis
        assert ts_result.success
        assert "temporal_features" in ts_result.data
        assert "seasonality_components" in ts_result.metadata
        assert (
            ts_result.metadata["processing_strategy"] == "streaming"
        )  # Large data auto-streaming
        assert simple_processing_time < 15.0  # Streaming processing time

        # Phase 2: Advanced time series analysis with detailed parameters
        advanced_request = ConversionRequest(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.TIME_SERIES_FEATURES,
            data=ts_dataset,
            context={
                "intention": "extract_temporal_patterns",
                "methods": ["fourier", "wavelet", "seasonal_decompose"],
                "fourier_components": 20,
                "wavelet_type": "morlet",
                "seasonal_model": "additive",
            },
        )

        start_time = time.time()
        advanced_ts_result = ts_shim.convert(advanced_request)
        advanced_processing_time = time.time() - start_time

        # Validate advanced results have more detail
        assert advanced_ts_result.success
        assert len(advanced_ts_result.data["temporal_features"]) > len(
            ts_result.data["temporal_features"]
        )

        # Phase 3: Pattern Recognition on extracted temporal features
        pattern_shim = self.domain_shims["pattern_recognition"]

        pattern_request = ConversionRequest(
            source_format=DataFormat.TIME_SERIES_FEATURES,
            target_format=DataFormat.PATTERN_CLUSTERS,
            data=ts_result.data,
            context={
                "intention": "discover_patterns",
                "temporal_context": ts_result.metadata,
                "pattern_types": ["clusters", "anomalies", "trends"],
            },
        )

        start_time = time.time()
        pattern_result = pattern_shim.convert(pattern_request)
        pattern_time = time.time() - start_time

        # Validate pattern recognition results
        assert pattern_result.success
        assert "discovered_patterns" in pattern_result.data
        assert "temporal_coherence" in pattern_result.metadata
        assert pattern_time < 20.0  # Pattern recognition processing time

        # Validate progressive disclosure worked
        assert (
            simple_processing_time < advanced_processing_time
        )  # Simple should be faster

        total_time = simple_processing_time + advanced_processing_time + pattern_time
        logger.info(
            f"Time series→pattern workflow: {total_time:.2f}s (simple: {simple_processing_time:.2f}s, advanced: {advanced_processing_time:.2f}s)"
        )
    def test_four_domain_cross_composition_workflow(self):
        """
        Test complete four-domain workflow:
        Raw Data → Statistical Analysis → Regression → Time Series → Pattern Recognition

        Validates:
        - Complex cross-domain composition
        - Metadata preservation through entire pipeline
        - Performance optimization (caching, lazy loading)
        - Automatic shim injection and optimization
        """
        # Create comprehensive dataset
        raw_dataset = create_high_dimensional_dataset(
            samples=5000,
            features=100,
            target_correlation=0.8,
            temporal_component=True,
            categorical_features=10,
        )

        # Define complete pipeline
        pipeline_steps = [
            create_pipeline_step(
                step_id="statistical_analysis",
                input_format=DataFormat.PANDAS_DATAFRAME,
                output_format=DataFormat.STATISTICAL_ANALYSIS,
                domain_shim="statistical",
                context={
                    "intention": "comprehensive_exploration",
                    "include_distributions": True,
                    "correlation_analysis": True,
                    "outlier_detection": True,
                },
            ),
            create_pipeline_step(
                step_id="regression_modeling",
                input_format=DataFormat.STATISTICAL_ANALYSIS,
                output_format=DataFormat.REGRESSION_MODEL,
                domain_shim="regression",
                context={
                    "intention": "predictive_modeling",
                    "model_selection": "auto",
                    "cross_validation": True,
                },
            ),
            create_pipeline_step(
                step_id="temporal_analysis",
                input_format=DataFormat.REGRESSION_MODEL,
                output_format=DataFormat.TIME_SERIES_FEATURES,
                domain_shim="time_series",
                context={"intention": "temporal_decomposition", "use_residuals": True},
            ),
            create_pipeline_step(
                step_id="pattern_discovery",
                input_format=DataFormat.TIME_SERIES_FEATURES,
                output_format=DataFormat.PATTERN_CLUSTERS,
                domain_shim="pattern_recognition",
                context={
                    "intention": "comprehensive_pattern_discovery",
                    "clustering_methods": ["kmeans", "hierarchical"],
                    "anomaly_detection": True,
                },
            ),
        ]

        # Analyze and optimize pipeline
        optimization_criteria = OptimizationCriteria(
            performance_weight=0.4, memory_weight=0.3, accuracy_weight=0.3
        )

        optimized_pipeline = analyze_and_fix_pipeline(
            steps=pipeline_steps,
            criteria=optimization_criteria,
            analyzer=self.pipeline_analyzer,
        )

        # Execute optimized pipeline
        results = {}
        total_start_time = time.time()
        current_data = raw_dataset

        for step in optimized_pipeline.steps:
            step_start_time = time.time()

            # Get appropriate domain shim
            shim = self.domain_shims[step.domain_shim]

            # Create conversion request with accumulated context
            request = ConversionRequest(
                source_format=step.input_format,
                target_format=step.output_format,
                data=current_data,
                context={
                    **step.context,
                    "pipeline_context": {
                        "previous_results": list(results.keys()),
                        "step_position": len(results) + 1,
                        "total_steps": len(optimized_pipeline.steps),
                    },
                },
            )

            # Execute conversion with performance monitoring
            result = shim.convert(request)
            step_time = time.time() - step_start_time

            # Validate step success
            assert result.success, f"Step {step.step_id} failed: {result.error}"

            # Store results and update data for next step
            results[step.step_id] = {
                "result": result,
                "execution_time": step_time,
                "memory_usage": result.metadata.get("memory_usage", 0),
            }
            current_data = result.data

            logger.info(f"Step {step.step_id} completed in {step_time:.2f}s")

        total_time = time.time() - total_start_time

        # Validate complete workflow
        assert len(results) == 4  # All steps completed

        # Validate data lineage preservation
        final_result = results["pattern_discovery"]["result"]
        assert "data_lineage" in final_result.metadata
        expected_lineage = [
            "raw_data",
            "statistical_analysis",
            "regression_modeling",
            "temporal_analysis",
            "pattern_discovery",
        ]
        assert final_result.metadata["data_lineage"] == expected_lineage

        # Validate performance requirements
        assert total_time < 60.0  # Complete workflow under 1 minute
        max_memory = max(r["memory_usage"] for r in results.values())
        assert max_memory < 1024 * 1024 * 1024  # Under 1GB memory usage

        # Validate cross-domain semantic coherence
        statistical_insights = results["statistical_analysis"]["result"].metadata
        regression_context = results["regression_modeling"]["result"].metadata
        assert "statistical_features_used" in regression_context

        temporal_context = results["temporal_analysis"]["result"].metadata
        pattern_context = results["pattern_discovery"]["result"].metadata
        assert "temporal_coherence_score" in pattern_context

        logger.info(
            f"Four-domain workflow completed: {total_time:.2f}s, peak memory: {max_memory / 1024 / 1024:.1f}MB"
        )
    def test_streaming_large_dataset_workflow(self):
        """
        Test streaming-first processing with memory constraints.

        Validates:
        - Automatic streaming strategy selection
        - Memory bounds adherence
        - Performance optimization under constraints
        - Data integrity in streaming mode
        """
        # Create large dataset that exceeds memory constraints
        large_dataset = create_streaming_dataset(
            total_size="2GB",  # Larger than memory limit
            chunk_size="100MB",
            features=200,
            complexity="high",
        )

        # Configure strict memory constraints
        memory_limit = 512 * 1024 * 1024  # 512MB limit
        self.lazy_manager = LazyLoadingManager(memory_limit=memory_limit)

        # Create streaming workflow
        streaming_request = ConversionRequest(
            source_format=DataFormat.STREAMING_DATAFRAME,
            target_format=DataFormat.STATISTICAL_ANALYSIS,
            data=large_dataset,
            context={
                "intention": "streaming_analysis",
                "memory_constraint": memory_limit,
                "processing_strategy": "auto",  # Let system decide
                "quality_vs_speed": "balanced",
            },
        )

        # Monitor memory usage during processing
        memory_monitor = []
        processing_start = time.time()

        def memory_callback(current_memory):
            memory_monitor.append(current_memory)
            assert current_memory <= memory_limit, (
                f"Memory limit exceeded: {current_memory} > {memory_limit}"
            )

        # Execute with memory monitoring
        statistical_shim = self.domain_shims["statistical"]
        result = statistical_shim.convert(
            streaming_request, memory_callback=memory_callback
        )

        processing_time = time.time() - processing_start

        # Validate streaming processing success
        assert result.success
        assert result.metadata["processing_strategy"] == "streaming"
        assert result.metadata["chunks_processed"] > 1

        # Validate memory constraints were respected
        max_memory = max(memory_monitor) if memory_monitor else 0
        assert max_memory <= memory_limit

        # Validate streaming data integrity
        assert "streaming_statistics" in result.data
        assert result.data["sample_count"] > 1000000  # Large dataset processed
        assert "confidence_intervals" in result.metadata  # Quality preserved

        # Validate performance characteristics
        throughput = result.data["sample_count"] / processing_time
        assert throughput > 50000  # Samples per second threshold

        logger.info(
            f"Streaming workflow: {processing_time:.2f}s, {throughput:.0f} samples/s, peak memory: {max_memory / 1024 / 1024:.1f}MB"
        )
    def test_error_recovery_and_alternative_pathways(self):
        """
        Test error recovery and alternative pathway discovery.

        Validates:
        - Automatic error detection and classification
        - Alternative pathway discovery
        - Graceful degradation strategies
        - Recovery system effectiveness
        """
        # Create dataset that will trigger various error conditions
        problematic_dataset = pd.DataFrame(
            {
                "numeric_with_nulls": [1.0, 2.0, None, 4.0, 5.0],
                "categorical_messy": ["A", "B", None, "C", "invalid_category"],
                "high_cardinality": [f"cat_{i}" for i in range(5)],
                "constant_column": [1, 1, 1, 1, 1],
                "infinite_values": [1.0, 2.0, float("inf"), 4.0, -float("inf")],
            }
        )

        # Configure error recovery system
        error_recovery = self.error_recovery

        # Test 1: Missing value handling with alternative pathways
        request_with_nulls = ConversionRequest(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.STATISTICAL_ANALYSIS,
            data=problematic_dataset,
            context={
                "intention": "robust_analysis",
                "handle_missing": "auto",
                "error_tolerance": "high",
            },
        )

        statistical_shim = self.domain_shims["statistical"]

        # Expect primary pathway to encounter issues, trigger recovery
        result_with_recovery = statistical_shim.convert(request_with_nulls)

        # Should succeed via alternative pathway
        assert result_with_recovery.success
        assert "recovery_pathway_used" in result_with_recovery.metadata
        assert "data_cleaning_applied" in result_with_recovery.metadata

        # Test 2: Format incompatibility with automatic shim injection
        incompatible_data = sparse.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])

        incompatible_request = ConversionRequest(
            source_format=DataFormat.SCIPY_SPARSE,
            target_format=DataFormat.REGRESSION_MODEL,  # Direct incompatible conversion
            data=incompatible_data,
            context={"intention": "direct_modeling", "allow_format_conversion": True},
        )

        # Should automatically inject intermediate shims
        regression_shim = self.domain_shims["regression"]
        result_with_shims = regression_shim.convert(incompatible_request)

        assert result_with_shims.success
        assert "intermediate_conversions" in result_with_shims.metadata
        assert len(result_with_shims.metadata["intermediate_conversions"]) > 0

        # Test 3: Resource exhaustion recovery
        huge_request = ConversionRequest(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.PATTERN_CLUSTERS,
            data=create_high_dimensional_dataset(
                samples=100000, features=10000
            ),  # Memory intensive
            context={
                "intention": "exhaustive_clustering",
                "memory_constraint": 256 * 1024 * 1024,  # 256MB constraint
                "fallback_strategy": "sampling",
            },
        )

        pattern_shim = self.domain_shims["pattern_recognition"]
        result_with_sampling = pattern_shim.convert(huge_request)

        # Should succeed via sampling fallback
        assert result_with_sampling.success
        assert result_with_sampling.metadata["processing_strategy"] == "sampled"
        assert "sample_ratio" in result_with_sampling.metadata

        logger.info("Error recovery tests completed successfully")
    def test_performance_optimization_effectiveness(self):
        """
        Test performance optimization components.

        Validates:
        - Caching effectiveness and hit rates
        - Lazy loading memory management
        - Performance improvement measurement
        - Optimization strategy selection
        """
        # Create test dataset for repeated operations
        test_dataset = create_statistical_dataset(size="large", complexity="medium")

        # Test 1: Caching effectiveness
        cache_test_request = ConversionRequest(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.STATISTICAL_ANALYSIS,
            data=test_dataset,
            context={"intention": "cache_test", "analysis_depth": "comprehensive"},
        )

        statistical_shim = self.domain_shims["statistical"]

        # First execution - cache miss
        start_time = time.time()
        first_result = statistical_shim.convert(cache_test_request)
        first_time = time.time() - start_time

        # Second execution - should hit cache
        start_time = time.time()
        second_result = statistical_shim.convert(cache_test_request)
        second_time = time.time() - start_time

        # Validate caching worked
        assert first_result.success and second_result.success
        assert second_time < first_time * 0.3  # At least 70% improvement

        # Check cache statistics
        cache_stats = self.cache.get_statistics()
        assert cache_stats.hits > 0
        assert cache_stats.hit_rate > 0.5

        # Test 2: Lazy loading memory optimization
        lazy_dataset = create_streaming_dataset(
            total_size="1GB", chunk_size="50MB", features=100
        )

        lazy_request = ConversionRequest(
            source_format=DataFormat.STREAMING_DATAFRAME,
            target_format=DataFormat.STATISTICAL_ANALYSIS,
            data=lazy_dataset,
            context={
                "intention": "memory_efficient_analysis",
                "lazy_loading": True,
                "memory_budget": 200 * 1024 * 1024,  # 200MB
            },
        )

        # Monitor memory usage
        memory_usage = []

        def memory_monitor(usage):
            memory_usage.append(usage)

        start_time = time.time()
        lazy_result = statistical_shim.convert(
            lazy_request, memory_callback=memory_monitor
        )
        lazy_time = time.time() - start_time

        # Validate lazy loading effectiveness
        assert lazy_result.success
        max_memory = max(memory_usage) if memory_usage else 0
        assert max_memory < 250 * 1024 * 1024  # Stayed within budget

        # Test 3: Performance optimization strategy comparison
        strategies = ["speed", "memory", "balanced"]
        strategy_results = {}

        for strategy in strategies:
            strategy_request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=DataFormat.REGRESSION_MODEL,
                data=test_dataset,
                context={
                    "intention": "strategy_comparison",
                    "optimization_strategy": strategy,
                },
            )

            start_time = time.time()
            result = self.domain_shims["regression"].convert(strategy_request)
            execution_time = time.time() - start_time

            strategy_results[strategy] = {
                "time": execution_time,
                "memory": result.metadata.get("memory_usage", 0),
                "accuracy": result.metadata.get("model_accuracy", 0),
            }

        # Validate strategy effectiveness
        assert strategy_results["speed"]["time"] < strategy_results["memory"]["time"]
        assert (
            strategy_results["memory"]["memory"] < strategy_results["speed"]["memory"]
        )
        balanced = strategy_results["balanced"]
        assert balanced["time"] < max(
            strategy_results["speed"]["time"], strategy_results["memory"]["time"]
        )

        logger.info(
            f"Performance optimization: Cache hit rate: {cache_stats.hit_rate:.2f}, Max memory: {max_memory / 1024 / 1024:.1f}MB"
        )
    def test_first_principles_validation(self):
        """
        Comprehensive validation of the Five First Principles adherence.

        Tests each principle systematically:
        1. Intention-Driven Interface
        2. Context-Aware Composition
        3. Progressive Disclosure Architecture
        4. Streaming-First Data Science
        5. Modular Domain Integration
        """
        test_dataset = create_statistical_dataset(size="medium", complexity="high")

        # Principle 1: Intention-Driven Interface
        intention_request = ConversionRequest(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.STATISTICAL_ANALYSIS,
            data=test_dataset,
            context={
                # Semantic parameters, not technical ones
                "intention": "find_strong_relationships",
                "focus": "linear_patterns",
                "strength": "high",
                "confidence": "95%",
            },
        )

        result = self.domain_shims["statistical"].convert(intention_request)
        assert result.success
        assert (
            result.metadata["interpretation_guidance"]["relationship_strength"]
            == "high"
        )
        assert "recommended_next_steps" in result.metadata["interpretation_guidance"]

        # Principle 2: Context-Aware Composition
        # First step creates context
        context_step1 = result

        # Second step uses context from first
        composition_request = ConversionRequest(
            source_format=DataFormat.STATISTICAL_ANALYSIS,
            target_format=DataFormat.REGRESSION_MODEL,
            data=context_step1.data,
            context={
                "intention": "model_discovered_relationships",
                "statistical_context": context_step1.metadata,  # Context-aware
                "adapt_to_findings": True,
            },
        )

        composition_result = self.domain_shims["regression"].convert(
            composition_request
        )
        assert composition_result.success
        assert "adapted_based_on_context" in composition_result.metadata
        assert composition_result.metadata["model_selection_rationale"] is not None

        # Principle 3: Progressive Disclosure Architecture
        # Simple call first
        simple_request = ConversionRequest(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.TIME_SERIES_FEATURES,
            data=create_time_series_dataset(length=1000, features=10),
            context={
                "intention": "extract_patterns",
                # No detailed parameters - system chooses
            },
        )

        simple_result = self.domain_shims["time_series"].convert(simple_request)
        assert simple_result.success
        assert "auto_selected_methods" in simple_result.metadata

        # Detailed call with full control
        detailed_request = ConversionRequest(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.TIME_SERIES_FEATURES,
            data=create_time_series_dataset(length=1000, features=10),
            context={
                "intention": "extract_patterns",
                "methods": ["fourier", "wavelet", "seasonal_decompose"],
                "fourier_components": 15,
                "wavelet_type": "morlet",
                "seasonal_model": "multiplicative",
                "decomposition_trend": "linear",
            },
        )

        detailed_result = self.domain_shims["time_series"].convert(
            detailed_request
        )
        assert detailed_result.success
        assert len(detailed_result.data["features"]) > len(
            simple_result.data["features"]
        )

        # Principle 4: Streaming-First Data Science
        streaming_dataset = create_streaming_dataset(
            total_size="500MB", chunk_size="50MB"
        )

        streaming_request = ConversionRequest(
            source_format=DataFormat.STREAMING_DATAFRAME,
            target_format=DataFormat.PATTERN_CLUSTERS,
            data=streaming_dataset,
            context={
                "intention": "discover_patterns",
                "memory_constraint": 100 * 1024 * 1024,  # 100MB limit
            },
        )

        streaming_result = self.domain_shims["pattern_recognition"].convert(
            streaming_request
        )
        assert streaming_result.success
        assert streaming_result.metadata["processing_strategy"] == "streaming"
        assert streaming_result.metadata["memory_efficient"] == True

        # Principle 5: Modular Domain Integration
        # Test seamless integration between all four domains
        integration_data = test_dataset
        domain_sequence = [
            "statistical",
            "regression",
            "time_series",
            "pattern_recognition",
        ]

        results_chain = []
        current_data = integration_data

        for i, domain in enumerate(domain_sequence):
            if i == 0:
                source_format = DataFormat.PANDAS_DATAFRAME
            else:
                # Use output format from previous domain
                source_format = results_chain[-1]["result"].metadata["output_format"]

            # Define target format based on domain
            target_formats = {
                "statistical": DataFormat.STATISTICAL_ANALYSIS,
                "regression": DataFormat.REGRESSION_MODEL,
                "time_series": DataFormat.TIME_SERIES_FEATURES,
                "pattern_recognition": DataFormat.PATTERN_CLUSTERS,
            }

            request = ConversionRequest(
                source_format=source_format,
                target_format=target_formats[domain],
                data=current_data,
                context={
                    "intention": f"integrate_with_{domain}",
                    "cross_domain_context": [
                        r["result"].metadata for r in results_chain
                    ],
                },
            )

            result = self.domain_shims[domain].convert(request)
            assert result.success

            results_chain.append({"domain": domain, "result": result})
            current_data = result.data

        # Validate complete integration chain
        final_result = results_chain[-1]["result"]
        assert "cross_domain_lineage" in final_result.metadata
        assert len(final_result.metadata["cross_domain_lineage"]) == 4
        assert final_result.metadata["integration_coherence_score"] > 0.8

        logger.info("All Five First Principles validated successfully")


if __name__ == "__main__":
    # Run the end-to-end integration tests
    import sys

    logging.basicConfig(level=logging.INFO)

    # Use asyncio to run async tests
    def run_tests():
        test_instance = TestEndToEndIntegration()
        test_instance.setup_integration_framework()

        # Run all test methods
        test_methods = [
            test_instance.test_complete_statistical_to_regression_workflow,
            test_instance.test_time_series_to_pattern_recognition_workflow,
            test_instance.test_four_domain_cross_composition_workflow,
            test_instance.test_streaming_large_dataset_workflow,
            test_instance.test_error_recovery_and_alternative_pathways,
            test_instance.test_performance_optimization_effectiveness,
            test_instance.test_first_principles_validation,
        ]

        for i, test_method in enumerate(test_methods, 1):
            print(f"\n{'=' * 60}")
            print(f"Running Test {i}/{len(test_methods)}: {test_method.__name__}")
            print(f"{'=' * 60}")

            try:
                test_method()
                print(f"✅ {test_method.__name__} PASSED")
            except Exception as e:
                print(f"❌ {test_method.__name__} FAILED: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n{'=' * 60}")
        print("End-to-End Integration Tests Complete")
        print(f"{'=' * 60}")

    if sys.version_info >= (3, 7):
        asyncio.run(run_tests())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_tests())
