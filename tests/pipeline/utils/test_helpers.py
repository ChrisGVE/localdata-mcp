"""
Test Helper Utilities for Integration Shims Testing

This module provides comprehensive utilities for validating the Integration Shims
Framework during testing. Includes validation functions, performance measurement
tools, and assertion helpers specifically designed for the Five First Principles.

Helper Categories:
- Workflow integrity validation
- Performance measurement utilities
- First Principles validation
- Data integrity assertions
- Domain combination testing utilities
- Format compatibility validation
- Error handling test utilities
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, Mock

import numpy as np
import pandas as pd
from scipy import sparse

from localdata_mcp.pipeline.integration import (
    ConversionRequest,
    ConversionResult,
    DataFormat,
    PipelineStep,
)


def assert_workflow_integrity(
    results: List[ConversionResult], expected_lineage: List[str]
) -> None:
    """
    Assert that a workflow maintains integrity across all steps.

    Args:
        results: List of conversion results from workflow steps
        expected_lineage: Expected domain/step lineage

    Raises:
        AssertionError: If workflow integrity is compromised
    """
    assert len(results) > 0, "No results provided for integrity check"

    # Check basic success of all steps
    for i, result in enumerate(results):
        assert result.success, f"Step {i} failed: {result.error}"
        assert result.metadata is not None, f"Step {i} missing metadata"

    # Check lineage preservation
    if expected_lineage:
        final_result = results[-1]
        if "data_lineage" in final_result.metadata:
            actual_lineage = final_result.metadata["data_lineage"]
            assert len(actual_lineage) == len(
                expected_lineage
            ), f"Lineage length mismatch: expected {len(expected_lineage)}, got {len(actual_lineage)}"

            for expected, actual in zip(expected_lineage, actual_lineage):
                assert (
                    expected == actual
                ), f"Lineage mismatch: expected {expected}, got {actual}"

    # Check metadata continuity
    for i in range(1, len(results)):
        current_result = results[i]
        previous_result = results[i - 1]

        # Each step should reference previous step
        if "previous_step_metadata" in current_result.metadata:
            assert current_result.metadata["previous_step_metadata"] is not None

    # Check data flow continuity
    for i in range(1, len(results)):
        current_result = results[i]

        # Should have source format information
        assert "source_format" in current_result.metadata
        assert "target_format" in current_result.metadata

        # Format progression should make sense
        source_format = current_result.metadata["source_format"]
        target_format = current_result.metadata["target_format"]
        assert (
            source_format != target_format
        ), f"Step {i} has same source and target format"


def measure_performance(func: callable, *args, **kwargs) -> Tuple[Any, float, int]:
    """
    Measure performance metrics for a function call.

    Args:
        func: Function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Tuple of (result, execution_time_seconds, memory_usage_bytes)
    """
    import os

    import psutil

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time

    memory_after = process.memory_info().rss
    memory_usage = memory_after - memory_before

    return result, execution_time, memory_usage


async def measure_async_performance(
    coro: callable, *args, **kwargs
) -> Tuple[Any, float, int]:
    """
    Measure performance metrics for an async function call.

    Args:
        coro: Coroutine function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Tuple of (result, execution_time_seconds, memory_usage_bytes)
    """
    import os

    import psutil

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    start_time = time.time()
    result = await coro(*args, **kwargs)
    execution_time = time.time() - start_time

    memory_after = process.memory_info().rss
    memory_usage = memory_after - memory_before

    return result, execution_time, memory_usage


def validate_first_principles(
    workflow: List[ConversionRequest], results: List[ConversionResult]
) -> Dict[str, bool]:
    """
    Validate that a workflow adheres to the Five First Principles.

    Args:
        workflow: List of conversion requests
        results: List of conversion results

    Returns:
        Dictionary with validation results for each principle
    """
    validation_results = {
        "intention_driven_interface": False,
        "context_aware_composition": False,
        "progressive_disclosure": False,
        "streaming_first": False,
        "modular_domain_integration": False,
    }

    # Principle 1: Intention-Driven Interface
    intention_scores = []
    for request in workflow:
        context = request.context or {}

        # Check for semantic parameters
        has_intention = "intention" in context
        has_semantic_params = any(
            key in context for key in ["strength", "focus", "quality", "approach"]
        )

        # Avoid technical parameters
        avoid_technical = not any(
            key in context for key in ["threshold", "n_components", "max_iter"]
        )

        score = sum([has_intention, has_semantic_params, avoid_technical]) / 3.0
        intention_scores.append(score)

    validation_results["intention_driven_interface"] = np.mean(intention_scores) > 0.7

    # Principle 2: Context-Aware Composition
    composition_scores = []
    for i, result in enumerate(results):
        if i == 0:
            composition_scores.append(1.0)  # First step always valid
            continue

        metadata = result.metadata or {}

        # Check for context usage from previous steps
        has_context_usage = any(
            key in metadata
            for key in [
                "statistical_context",
                "previous_results",
                "accumulated_context",
            ]
        )

        # Check for context enrichment
        has_enrichment = any(
            key in metadata
            for key in [
                "enriched_metadata",
                "cross_domain_insights",
                "context_accumulation",
            ]
        )

        score = sum([has_context_usage, has_enrichment]) / 2.0
        composition_scores.append(score)

    validation_results["context_aware_composition"] = np.mean(composition_scores) > 0.6

    # Principle 3: Progressive Disclosure
    disclosure_scores = []
    for request in workflow:
        context = request.context or {}

        # Check for appropriate parameter complexity
        basic_params = sum(
            1 for key in context.keys() if key in ["intention", "quality", "approach"]
        )
        advanced_params = sum(
            1
            for key in context.keys()
            if key.startswith("detailed_") or "_params" in key
        )

        # Good progressive disclosure has either basic params only, or both basic and advanced
        if basic_params > 0:
            if advanced_params == 0:
                score = 1.0  # Simple interface
            else:
                score = 0.8  # Advanced interface with basics
        else:
            score = 0.3  # Missing basic parameters

        disclosure_scores.append(score)

    validation_results["progressive_disclosure"] = np.mean(disclosure_scores) > 0.7

    # Principle 4: Streaming-First
    streaming_scores = []
    for result in results:
        metadata = result.metadata or {}

        # Check for streaming awareness
        has_streaming_strategy = "processing_strategy" in metadata
        memory_aware = "memory_usage" in metadata or "memory_efficient" in metadata
        chunk_aware = "chunks_processed" in metadata or "chunk_size" in metadata

        score = sum([has_streaming_strategy, memory_aware, chunk_aware]) / 3.0
        streaming_scores.append(score)

    validation_results["streaming_first"] = np.mean(streaming_scores) > 0.5

    # Principle 5: Modular Domain Integration
    integration_scores = []
    domain_coverage = set()

    for result in results:
        metadata = result.metadata or {}

        # Track domain coverage
        if "domain" in metadata:
            domain_coverage.add(metadata["domain"])

        # Check for integration features
        cross_domain = any(
            key in metadata
            for key in ["cross_domain", "domain_bridge", "integration_score"]
        )
        modular_design = "domain_specific_processing" in metadata

        score = sum([cross_domain, modular_design]) / 2.0
        integration_scores.append(score)

    # Bonus for multi-domain coverage
    domain_bonus = min(1.0, len(domain_coverage) / 3.0)  # Up to 3 domains
    final_integration_score = np.mean(integration_scores) * 0.7 + domain_bonus * 0.3

    validation_results["modular_domain_integration"] = final_integration_score > 0.6

    return validation_results


def validate_data_integrity(
    original_data: Any, converted_data: Any, conversion_type: str = "general"
) -> Dict[str, bool]:
    """
    Validate that data integrity is maintained during conversion.

    Args:
        original_data: Original data before conversion
        converted_data: Data after conversion
        conversion_type: Type of conversion for specific checks

    Returns:
        Dictionary with integrity validation results
    """
    integrity_results = {
        "shape_preserved": False,
        "data_type_appropriate": False,
        "value_fidelity": False,
        "metadata_preserved": False,
        "no_data_loss": False,
    }

    try:
        # Shape preservation (where applicable)
        if hasattr(original_data, "shape") and hasattr(converted_data, "shape"):
            if conversion_type == "pandas_to_numpy":
                # Row count should be preserved
                integrity_results["shape_preserved"] = (
                    original_data.shape[0] == converted_data.shape[0]
                )
            elif conversion_type == "numpy_to_sparse":
                # Full shape should be preserved
                integrity_results["shape_preserved"] = (
                    original_data.shape == converted_data.shape
                )
            else:
                # General shape compatibility
                integrity_results["shape_preserved"] = (
                    hasattr(converted_data, "shape") and len(converted_data.shape) > 0
                )
        else:
            integrity_results["shape_preserved"] = True  # N/A

        # Data type appropriateness
        if isinstance(converted_data, pd.DataFrame):
            integrity_results["data_type_appropriate"] = (
                True  # DataFrames handle mixed types
            )
        elif isinstance(converted_data, np.ndarray):
            integrity_results["data_type_appropriate"] = converted_data.dtype in [
                np.float64,
                np.float32,
                np.int64,
                np.int32,
                np.bool_,
            ]
        elif isinstance(converted_data, sparse.spmatrix):
            integrity_results["data_type_appropriate"] = hasattr(converted_data, "nnz")
        else:
            integrity_results["data_type_appropriate"] = True  # Unknown type, assume OK

        # Value fidelity (for numeric data)
        if isinstance(original_data, np.ndarray) and isinstance(
            converted_data, np.ndarray
        ):
            if original_data.dtype == converted_data.dtype:
                try:
                    np.testing.assert_array_almost_equal(
                        original_data, converted_data, decimal=10
                    )
                    integrity_results["value_fidelity"] = True
                except AssertionError:
                    integrity_results["value_fidelity"] = False
            else:
                # Different dtypes, check approximate equality
                max_diff = np.max(
                    np.abs(
                        original_data.astype(np.float64)
                        - converted_data.astype(np.float64)
                    )
                )
                integrity_results["value_fidelity"] = max_diff < 1e-6
        elif isinstance(original_data, pd.DataFrame) and isinstance(
            converted_data, pd.DataFrame
        ):
            # Compare numeric columns
            numeric_original = original_data.select_dtypes(include=[np.number])
            numeric_converted = converted_data.select_dtypes(include=[np.number])

            if len(numeric_original.columns) > 0 and len(numeric_converted.columns) > 0:
                try:
                    pd.testing.assert_frame_equal(
                        numeric_original, numeric_converted, atol=1e-10
                    )
                    integrity_results["value_fidelity"] = True
                except AssertionError:
                    integrity_results["value_fidelity"] = False
            else:
                integrity_results["value_fidelity"] = True
        else:
            integrity_results["value_fidelity"] = True  # Assume OK for different types

        # Metadata preservation (check if attrs or similar are preserved)
        if hasattr(original_data, "attrs") and hasattr(converted_data, "attrs"):
            integrity_results["metadata_preserved"] = len(converted_data.attrs) > 0
        else:
            integrity_results["metadata_preserved"] = True  # N/A

        # No data loss (check for NaN introduction or size reduction)
        original_size = getattr(original_data, "size", len(str(original_data)))
        converted_size = getattr(converted_data, "size", len(str(converted_data)))

        if hasattr(original_data, "isna") and hasattr(converted_data, "isna"):
            original_nan_count = original_data.isna().sum().sum()
            converted_nan_count = converted_data.isna().sum().sum()
            integrity_results["no_data_loss"] = (
                converted_nan_count <= original_nan_count
            )
        else:
            # Size-based check
            integrity_results["no_data_loss"] = (
                converted_size >= original_size * 0.95
            )  # Allow 5% size variation

    except Exception:
        # If validation fails, mark all as False for safety
        for key in integrity_results:
            integrity_results[key] = False

    return integrity_results


def assert_format_compatibility(
    source_format: DataFormat, target_format: DataFormat, data: Any
) -> None:
    """
    Assert that data is compatible with format conversion.

    Args:
        source_format: Source data format
        target_format: Target data format
        data: Data to be converted

    Raises:
        AssertionError: If format compatibility is not satisfied
    """
    # Basic format compatibility checks

    if source_format == DataFormat.PANDAS_DATAFRAME:
        assert isinstance(data, pd.DataFrame), f"Expected DataFrame for {source_format}"
    elif source_format == DataFormat.NUMPY_ARRAY:
        assert isinstance(data, np.ndarray), f"Expected ndarray for {source_format}"
    elif source_format == DataFormat.SCIPY_SPARSE:
        assert isinstance(
            data, sparse.spmatrix
        ), f"Expected sparse matrix for {source_format}"

    # Check conversion compatibility
    compatible_conversions = {
        DataFormat.PANDAS_DATAFRAME: [
            DataFormat.NUMPY_ARRAY,
            DataFormat.SCIPY_SPARSE,
            DataFormat.CSV_DATA,
            DataFormat.JSON_DATA,
        ],
        DataFormat.NUMPY_ARRAY: [
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.SCIPY_SPARSE,
            DataFormat.BINARY_DATA,
        ],
        DataFormat.SCIPY_SPARSE: [
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY,
            DataFormat.MATRIX_MARKET,
        ],
    }

    if source_format in compatible_conversions:
        assert (
            target_format in compatible_conversions[source_format]
        ), f"Incompatible conversion: {source_format} → {target_format}"


def create_format_test_suite(
    formats: List[DataFormat],
) -> List[Tuple[DataFormat, DataFormat]]:
    """
    Create a comprehensive test suite for format conversions.

    Args:
        formats: List of formats to test

    Returns:
        List of (source_format, target_format) tuples for testing
    """
    test_combinations = []

    for i, source_format in enumerate(formats):
        for j, target_format in enumerate(formats):
            if i != j:  # Don't test self-conversion
                test_combinations.append((source_format, target_format))

    # Add round-trip tests
    round_trip_tests = []
    for source_format in formats:
        for target_format in formats:
            if source_format != target_format:
                # Test A→B→A
                round_trip_tests.append((source_format, target_format, source_format))

    return test_combinations


def create_mock_domain_pipeline(*domains: str) -> List[PipelineStep]:
    """
    Create a mock domain pipeline for testing.

    Args:
        *domains: Domain names to include in pipeline

    Returns:
        List of mock pipeline steps
    """
    steps = []

    format_mapping = {
        "statistical": (DataFormat.PANDAS_DATAFRAME, DataFormat.STATISTICAL_ANALYSIS),
        "regression": (DataFormat.STATISTICAL_ANALYSIS, DataFormat.REGRESSION_MODEL),
        "time_series": (DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES_FEATURES),
        "pattern_recognition": (DataFormat.NUMPY_ARRAY, DataFormat.PATTERN_CLUSTERS),
    }

    for i, domain in enumerate(domains):
        input_format, output_format = format_mapping.get(
            domain, (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)
        )

        # Adjust input format for chained steps
        if i > 0:
            prev_domain = domains[i - 1]
            _, input_format = format_mapping.get(
                prev_domain, (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)
            )

        step = PipelineStep(
            step_id=f"{domain}_step_{i}",
            input_format=input_format,
            output_format=output_format,
            domain_shim=domain,
            context={
                "intention": f"process_with_{domain}",
                "step_position": i,
                "total_steps": len(domains),
            },
        )
        steps.append(step)

    return steps


def validate_domain_combination(
    domain1: str, domain2: str, result1: ConversionResult, result2: ConversionResult
) -> Dict[str, float]:
    """
    Validate the quality of domain combination.

    Args:
        domain1: First domain name
        domain2: Second domain name
        result1: Result from first domain
        result2: Result from second domain

    Returns:
        Dictionary with combination quality scores
    """
    quality_scores = {
        "semantic_coherence": 0.0,
        "data_flow_quality": 0.0,
        "metadata_preservation": 0.0,
        "performance_efficiency": 0.0,
    }

    # Semantic coherence
    if "semantic_bridge_quality" in result2.metadata:
        quality_scores["semantic_coherence"] = result2.metadata[
            "semantic_bridge_quality"
        ]
    elif "cross_domain_coherence" in result2.metadata:
        quality_scores["semantic_coherence"] = result2.metadata[
            "cross_domain_coherence"
        ]
    else:
        # Default scoring based on domain compatibility
        compatible_pairs = {
            ("statistical", "regression"): 0.9,
            ("statistical", "time_series"): 0.7,
            ("regression", "time_series"): 0.8,
            ("time_series", "pattern_recognition"): 0.8,
        }
        key = (domain1, domain2)
        quality_scores["semantic_coherence"] = compatible_pairs.get(key, 0.6)

    # Data flow quality
    if result1.success and result2.success:
        quality_scores["data_flow_quality"] = 1.0

        # Bonus for maintained data characteristics
        if (
            "data_characteristics" in result1.metadata
            and "preserved_characteristics" in result2.metadata
        ):
            preservation_ratio = len(
                result2.metadata["preserved_characteristics"]
            ) / len(result1.metadata["data_characteristics"])
            quality_scores["data_flow_quality"] = min(1.0, preservation_ratio)

    # Metadata preservation
    metadata1_keys = set(result1.metadata.keys()) if result1.metadata else set()
    metadata2_keys = set(result2.metadata.keys()) if result2.metadata else set()

    if "inherited_metadata" in result2.metadata:
        quality_scores["metadata_preservation"] = 1.0
    elif metadata1_keys and metadata2_keys:
        # Check for metadata continuity
        common_keys = metadata1_keys.intersection(metadata2_keys)
        preservation_ratio = (
            len(common_keys) / len(metadata1_keys) if metadata1_keys else 0
        )
        quality_scores["metadata_preservation"] = preservation_ratio

    # Performance efficiency
    time1 = result1.metadata.get("execution_time", 1.0)
    time2 = result2.metadata.get("execution_time", 1.0)
    total_time = time1 + time2

    # Efficiency based on reasonable expectations
    if total_time < 5.0:
        quality_scores["performance_efficiency"] = 1.0
    elif total_time < 15.0:
        quality_scores["performance_efficiency"] = 0.8
    elif total_time < 30.0:
        quality_scores["performance_efficiency"] = 0.6
    else:
        quality_scores["performance_efficiency"] = 0.4

    return quality_scores


def measure_cross_domain_performance(
    workflow_steps: List[ConversionResult],
) -> Dict[str, Any]:
    """
    Measure performance metrics for cross-domain workflows.

    Args:
        workflow_steps: List of conversion results from workflow

    Returns:
        Dictionary with performance metrics
    """
    performance_metrics = {
        "total_execution_time": 0.0,
        "average_step_time": 0.0,
        "peak_memory_usage": 0,
        "total_memory_allocated": 0,
        "cache_hit_rate": 0.0,
        "conversion_overhead": 0.0,
        "throughput_samples_per_second": 0.0,
    }

    execution_times = []
    memory_usages = []

    for result in workflow_steps:
        if not result.metadata:
            continue

        # Collect timing data
        exec_time = result.metadata.get("execution_time", 0)
        execution_times.append(exec_time)

        # Collect memory data
        memory_usage = result.metadata.get("memory_usage", 0)
        memory_usages.append(memory_usage)

    # Calculate metrics
    if execution_times:
        performance_metrics["total_execution_time"] = sum(execution_times)
        performance_metrics["average_step_time"] = np.mean(execution_times)

    if memory_usages:
        performance_metrics["peak_memory_usage"] = max(memory_usages)
        performance_metrics["total_memory_allocated"] = sum(memory_usages)

    # Calculate cache hit rate from last result
    last_result = workflow_steps[-1]
    if last_result.metadata and "cache_statistics" in last_result.metadata:
        cache_stats = last_result.metadata["cache_statistics"]
        if "hit_rate" in cache_stats:
            performance_metrics["cache_hit_rate"] = cache_stats["hit_rate"]

    # Calculate throughput (if sample count available)
    if last_result.metadata and "samples_processed" in last_result.metadata:
        samples = last_result.metadata["samples_processed"]
        total_time = performance_metrics["total_execution_time"]
        if total_time > 0:
            performance_metrics["throughput_samples_per_second"] = samples / total_time

    # Calculate conversion overhead
    if len(workflow_steps) > 1:
        # Compare to baseline single-step processing estimate
        baseline_time = execution_times[0] * len(workflow_steps)  # Naive estimate
        actual_time = performance_metrics["total_execution_time"]
        if baseline_time > 0:
            performance_metrics["conversion_overhead"] = (
                actual_time - baseline_time
            ) / baseline_time

    return performance_metrics


def assess_semantic_preservation(
    input_context: Dict[str, Any], output_result: ConversionResult
) -> Dict[str, float]:
    """
    Assess how well semantic information is preserved across conversions.

    Args:
        input_context: Original semantic context
        output_result: Conversion result to assess

    Returns:
        Dictionary with semantic preservation scores
    """
    preservation_scores = {
        "intention_preservation": 0.0,
        "context_enrichment": 0.0,
        "interpretability_maintenance": 0.0,
        "domain_knowledge_retention": 0.0,
    }

    output_metadata = output_result.metadata or {}

    # Intention preservation
    original_intention = input_context.get("intention", "")
    if "preserved_intention" in output_metadata:
        if output_metadata["preserved_intention"] == original_intention:
            preservation_scores["intention_preservation"] = 1.0
        else:
            # Partial match scoring
            if original_intention in output_metadata["preserved_intention"]:
                preservation_scores["intention_preservation"] = 0.7
            else:
                preservation_scores["intention_preservation"] = 0.3
    elif "interpretation_guidance" in output_metadata:
        # Has interpretation guidance, good preservation
        preservation_scores["intention_preservation"] = 0.8

    # Context enrichment
    input_keys = set(input_context.keys())
    output_keys = set(output_metadata.keys())

    if "enriched_context" in output_metadata:
        preservation_scores["context_enrichment"] = 1.0
    elif len(output_keys) > len(input_keys):
        # More metadata in output than input
        enrichment_ratio = len(output_keys) / len(input_keys) if input_keys else 1.0
        preservation_scores["context_enrichment"] = min(1.0, enrichment_ratio - 0.5)
    else:
        preservation_scores["context_enrichment"] = 0.5

    # Interpretability maintenance
    if "interpretation_guidance" in output_metadata:
        preservation_scores["interpretability_maintenance"] = 1.0
    elif "semantic_metadata" in output_metadata:
        preservation_scores["interpretability_maintenance"] = 0.8
    elif any(
        "description" in key or "meaning" in key for key in output_metadata.keys()
    ):
        preservation_scores["interpretability_maintenance"] = 0.6
    else:
        preservation_scores["interpretability_maintenance"] = 0.3

    # Domain knowledge retention
    if "domain_specific_insights" in output_metadata:
        preservation_scores["domain_knowledge_retention"] = 1.0
    elif (
        "statistical_properties" in output_metadata
        or "model_characteristics" in output_metadata
    ):
        preservation_scores["domain_knowledge_retention"] = 0.8
    elif any("analysis" in key or "model" in key for key in output_metadata.keys()):
        preservation_scores["domain_knowledge_retention"] = 0.6
    else:
        preservation_scores["domain_knowledge_retention"] = 0.4

    return preservation_scores


def create_domain_context_chain(domains: List[str]) -> List[Dict[str, Any]]:
    """
    Create a context chain for multi-domain testing.

    Args:
        domains: List of domain names

    Returns:
        List of context dictionaries for each domain
    """
    context_chain = []

    domain_contexts = {
        "statistical": {
            "intention": "statistical_exploration",
            "focus": "correlation_analysis",
            "statistical_methods": ["correlation", "distribution_analysis"],
            "significance_level": 0.05,
        },
        "regression": {
            "intention": "predictive_modeling",
            "model_selection": "adaptive",
            "validation_strategy": "cross_validation",
            "performance_metrics": ["r2", "rmse", "mae"],
        },
        "time_series": {
            "intention": "temporal_analysis",
            "decomposition_methods": ["seasonal", "trend"],
            "frequency_analysis": True,
            "forecasting_horizon": 12,
        },
        "pattern_recognition": {
            "intention": "pattern_discovery",
            "clustering_methods": ["kmeans", "hierarchical"],
            "dimensionality_reduction": "auto",
            "anomaly_detection": True,
        },
    }

    accumulated_context = {}

    for i, domain in enumerate(domains):
        base_context = domain_contexts.get(domain, {})

        # Add cross-domain context
        current_context = {
            **base_context,
            "step_position": i + 1,
            "total_steps": len(domains),
            "previous_domains": domains[:i],
            "accumulated_insights": accumulated_context.copy(),
        }

        if i > 0:
            current_context["cross_domain_integration"] = True
            current_context["upstream_context"] = context_chain[-1]

        context_chain.append(current_context)

        # Update accumulated context
        accumulated_context[domain] = {
            "processed": True,
            "context_contributed": list(base_context.keys()),
        }

    return context_chain


def measure_conversion_performance(func, *args, iterations: int = 10, **kwargs):
    """Measure average execution time and memory for a conversion function."""
    import time

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return {
        "result": result,
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "iterations": iterations,
    }


# Export all helper functions
__all__ = [
    "assert_workflow_integrity",
    "measure_performance",
    "measure_async_performance",
    "validate_first_principles",
    "validate_data_integrity",
    "assert_format_compatibility",
    "create_format_test_suite",
    "create_mock_domain_pipeline",
    "validate_domain_combination",
    "measure_cross_domain_performance",
    "assess_semantic_preservation",
    "create_domain_context_chain",
]
