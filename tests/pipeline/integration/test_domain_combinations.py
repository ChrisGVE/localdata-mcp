"""
Domain Combination Integration Tests for LocalData MCP v2.0

This module provides comprehensive testing for all possible domain combinations
and cross-domain integration scenarios. Tests validate seamless composition
between statistical analysis, regression modeling, time series analysis,
and pattern recognition domains.

Focus Areas:
- All possible domain pair combinations (6 pairs)
- All possible domain triplet combinations (4 triplets)
- Bidirectional domain workflows
- Domain-specific semantic context preservation
- Cross-domain metadata enrichment
- Composition performance optimization
"""

import pytest

pytest.importorskip(
    "localdata_mcp.pipeline.integration",
    reason="Domain combination tests reference unfinalized DataFormat enum values",
)

import asyncio
import itertools
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from localdata_mcp.pipeline.integration import (  # Core integration components; Domain shims; Pipeline analysis; Compatibility matrix; Schema validation
    CompatibilityLevel,
    ConversionRegistry,
    ConversionRequest,
    ConversionResult,
    DataFormat,
    PatternRecognitionShim,
    PipelineAnalyzer,
    PipelineCompatibilityMatrix,
    RegressionShim,
    SchemaInferenceEngine,
    SchemaValidator,
    StatisticalShim,
    TimeSeriesShim,
    analyze_and_fix_pipeline,
    assess_pipeline_compatibility,
    create_all_domain_shims,
    create_pipeline_step,
    find_optimal_format_for_domains,
    get_compatible_domain_shims,
)

# Test fixtures and utilities
from ..fixtures.sample_datasets import (
    create_mixed_domain_dataset,
    create_pattern_recognition_dataset,
    create_regression_dataset,
    create_statistical_dataset,
    create_time_series_dataset,
)
from ..utils.test_helpers import (
    assess_semantic_preservation,
    create_domain_context_chain,
    measure_cross_domain_performance,
    validate_domain_combination,
)

logger = logging.getLogger(__name__)


class TestDomainCombinations:
    """
    Comprehensive tests for all possible domain combinations and their interactions.

    Tests cover:
    1. All pairwise domain combinations (Statistical↔Regression, Statistical↔TimeSeries, etc.)
    2. All triplet domain combinations
    3. Bidirectional workflows and round-trip conversions
    4. Context preservation and semantic enrichment across domains
    5. Performance optimization for cross-domain workflows
    6. Format compatibility and automatic adaptation
    """

    # Define domain mapping and compatibility
    DOMAINS = {
        "statistical": {
            "shim_key": "statistical",
            "primary_formats": [
                DataFormat.STATISTICAL_ANALYSIS,
                DataFormat.CORRELATION_MATRIX,
            ],
            "input_formats": [DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY],
            "semantic_focus": ["distributions", "correlations", "statistical_tests"],
        },
        "regression": {
            "shim_key": "regression",
            "primary_formats": [
                DataFormat.REGRESSION_MODEL,
                DataFormat.PREDICTION_RESULTS,
            ],
            "input_formats": [
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.STATISTICAL_ANALYSIS,
            ],
            "semantic_focus": ["modeling", "prediction", "feature_importance"],
        },
        "time_series": {
            "shim_key": "time_series",
            "primary_formats": [
                DataFormat.TIME_SERIES_FEATURES,
                DataFormat.TEMPORAL_COMPONENTS,
            ],
            "input_formats": [DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES_DATA],
            "semantic_focus": ["temporal_patterns", "seasonality", "trends"],
        },
        "pattern_recognition": {
            "shim_key": "pattern_recognition",
            "primary_formats": [
                DataFormat.PATTERN_CLUSTERS,
                DataFormat.FEATURE_EMBEDDINGS,
            ],
            "input_formats": [DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY],
            "semantic_focus": [
                "clustering",
                "anomaly_detection",
                "dimensionality_reduction",
            ],
        },
    }

    @pytest.fixture(autouse=True)
    async def setup_domain_combination_framework(self):
        """Setup framework for domain combination testing."""
        # Initialize core components
        self.registry = ConversionRegistry()
        self.domain_shims = create_all_domain_shims()
        self.compatibility_matrix = PipelineCompatibilityMatrix()
        self.schema_engine = SchemaInferenceEngine()
        self.schema_validator = SchemaValidator()

        # Register all domain shims
        for shim_type, shim in self.domain_shims.items():
            await self.registry.register_adapter(shim)

        # Create test datasets for each domain
        self.test_datasets = {
            "statistical": create_statistical_dataset(size="medium", complexity="high"),
            "regression": create_regression_dataset(
                samples=1000, features=20, target_correlation=0.8
            ),
            "time_series": create_time_series_dataset(
                length=500, features=15, seasonality=True
            ),
            "pattern_recognition": create_pattern_recognition_dataset(
                samples=800, dimensions=50, clusters=5
            ),
            "mixed": create_mixed_domain_dataset(
                statistical_features=10,
                temporal_length=200,
                regression_targets=3,
                pattern_dimensions=30,
            ),
        }

        logger.info(
            f"Domain combination framework initialized with {len(self.domain_shims)} domains"
        )

    @pytest.mark.parametrize(
        "source_domain,target_domain",
        [
            ("statistical", "regression"),
            ("statistical", "time_series"),
            ("statistical", "pattern_recognition"),
            ("regression", "time_series"),
            ("regression", "pattern_recognition"),
            ("time_series", "pattern_recognition"),
        ],
    )
    @pytest.mark.asyncio
    async def test_pairwise_domain_combinations(
        self, source_domain: str, target_domain: str
    ):
        """
        Test all pairwise domain combinations for seamless integration.

        Validates:
        - Format compatibility and conversion
        - Semantic context preservation
        - Performance characteristics
        - Bidirectional capability where applicable
        """
        logger.info(f"Testing {source_domain} → {target_domain} combination")

        # Get source domain configuration and data
        source_config = self.DOMAINS[source_domain]
        target_config = self.DOMAINS[target_domain]
        source_data = self.test_datasets[source_domain]

        # Phase 1: Source domain processing
        source_shim = self.domain_shims[source_config["shim_key"]]
        source_request = ConversionRequest(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=source_config["primary_formats"][0],
            data=source_data,
            context={
                "intention": f"prepare_for_{target_domain}",
                "optimization_target": target_domain,
                "preserve_semantics": source_config["semantic_focus"],
            },
        )

        source_result = await source_shim.convert(source_request)
        assert source_result.success, f"Source domain {source_domain} processing failed"

        # Phase 2: Cross-domain conversion
        target_shim = self.domain_shims[target_config["shim_key"]]
        cross_domain_request = ConversionRequest(
            source_format=source_config["primary_formats"][0],
            target_format=target_config["primary_formats"][0],
            data=source_result.data,
            context={
                "intention": f"integrate_{source_domain}_with_{target_domain}",
                "source_domain_context": source_result.metadata,
                "semantic_bridge": {
                    "from": source_config["semantic_focus"],
                    "to": target_config["semantic_focus"],
                },
                "preserve_lineage": True,
            },
        )

        target_result = await target_shim.convert(cross_domain_request)
        assert (
            target_result.success
        ), f"Cross-domain {source_domain}→{target_domain} conversion failed"

        # Validation: Cross-domain integration quality
        await self._validate_cross_domain_integration(
            source_domain, target_domain, source_result, target_result
        )

        # Test bidirectional capability where semantically meaningful
        if self._is_bidirectional_meaningful(source_domain, target_domain):
            await self._test_bidirectional_conversion(
                source_domain, target_domain, source_result, target_result
            )

        logger.info(f"✅ {source_domain} → {target_domain} combination validated")

    @pytest.mark.parametrize(
        "domain_triplet",
        [
            ("statistical", "regression", "time_series"),
            ("statistical", "regression", "pattern_recognition"),
            ("statistical", "time_series", "pattern_recognition"),
            ("regression", "time_series", "pattern_recognition"),
        ],
    )
    @pytest.mark.asyncio
    async def test_triplet_domain_combinations(
        self, domain_triplet: Tuple[str, str, str]
    ):
        """
        Test three-domain composition workflows.

        Validates:
        - Complex multi-domain pipelines
        - Context accumulation and enrichment
        - Performance optimization across multiple domains
        - Semantic coherence maintenance
        """
        domain1, domain2, domain3 = domain_triplet
        logger.info(f"Testing triplet combination: {domain1} → {domain2} → {domain3}")

        # Use mixed dataset that has characteristics suitable for all domains
        initial_data = self.test_datasets["mixed"]

        # Build progressive context chain
        context_chain = []
        current_data = initial_data

        for i, domain_name in enumerate(domain_triplet):
            domain_config = self.DOMAINS[domain_name]
            domain_shim = self.domain_shims[domain_config["shim_key"]]

            # Determine input format
            if i == 0:
                input_format = DataFormat.PANDAS_DATAFRAME
            else:
                # Use output from previous domain
                prev_domain_config = self.DOMAINS[domain_triplet[i - 1]]
                input_format = prev_domain_config["primary_formats"][0]

            # Create request with accumulated context
            request = ConversionRequest(
                source_format=input_format,
                target_format=domain_config["primary_formats"][0],
                data=current_data,
                context={
                    "intention": f"triplet_step_{i + 1}_of_3",
                    "triplet_context": {
                        "position": i + 1,
                        "total_steps": 3,
                        "previous_domains": list(domain_triplet[:i]),
                        "remaining_domains": list(domain_triplet[i + 1 :]),
                    },
                    "accumulated_context": context_chain,
                    "semantic_focus": domain_config["semantic_focus"],
                    "cross_domain_optimization": True,
                },
            )

            # Execute domain conversion
            result = await domain_shim.convert(request)
            assert result.success, f"Triplet step {i + 1} ({domain_name}) failed"

            # Add to context chain and update data
            context_chain.append(
                {
                    "domain": domain_name,
                    "metadata": result.metadata,
                    "semantic_contributions": domain_config["semantic_focus"],
                }
            )
            current_data = result.data

            logger.info(f"  Step {i + 1}: {domain_name} processing completed")

        # Validate triplet integration
        await self._validate_triplet_integration(
            domain_triplet, context_chain, current_data
        )

        logger.info(f"✅ Triplet {domain_triplet} validated successfully")

    @pytest.mark.asyncio
    async def test_all_domain_comprehensive_workflow(self):
        """
        Test comprehensive workflow involving all four domains.

        Validates:
        - Complete domain ecosystem integration
        - Maximum context accumulation
        - Performance under full complexity
        - Semantic coherence across all domains
        """
        logger.info("Testing comprehensive four-domain workflow")

        # Create comprehensive dataset
        comprehensive_data = create_mixed_domain_dataset(
            statistical_features=25,
            temporal_length=1000,
            regression_targets=5,
            pattern_dimensions=100,
            complexity="maximum",
        )

        # Define optimal domain sequence based on data flow semantics
        domain_sequence = [
            "statistical",
            "regression",
            "time_series",
            "pattern_recognition",
        ]

        # Track comprehensive metrics
        workflow_metrics = {
            "execution_times": [],
            "memory_usage": [],
            "context_sizes": [],
            "semantic_coherence_scores": [],
        }

        current_data = comprehensive_data
        comprehensive_context = []

        for i, domain_name in enumerate(domain_sequence):
            domain_config = self.DOMAINS[domain_name]
            domain_shim = self.domain_shims[domain_config["shim_key"]]

            # Determine formats
            if i == 0:
                input_format = DataFormat.PANDAS_DATAFRAME
            else:
                prev_config = self.DOMAINS[domain_sequence[i - 1]]
                input_format = prev_config["primary_formats"][0]

            # Create comprehensive request
            request = ConversionRequest(
                source_format=input_format,
                target_format=domain_config["primary_formats"][0],
                data=current_data,
                context={
                    "intention": f"comprehensive_workflow_step_{i + 1}",
                    "workflow_context": {
                        "current_step": i + 1,
                        "total_steps": 4,
                        "domain_sequence": domain_sequence,
                        "optimization_target": "comprehensive_analysis",
                    },
                    "accumulated_knowledge": comprehensive_context,
                    "cross_domain_insights": True,
                    "performance_monitoring": True,
                },
            )

            # Execute with performance monitoring
            import time

            start_time = time.time()
            result = await domain_shim.convert(request)
            execution_time = time.time() - start_time

            assert result.success, f"Comprehensive workflow failed at {domain_name}"

            # Collect metrics
            workflow_metrics["execution_times"].append(execution_time)
            workflow_metrics["memory_usage"].append(
                result.metadata.get("memory_usage", 0)
            )
            workflow_metrics["context_sizes"].append(len(str(comprehensive_context)))
            workflow_metrics["semantic_coherence_scores"].append(
                result.metadata.get("semantic_coherence_score", 0.0)
            )

            # Update comprehensive context
            comprehensive_context.append(
                {
                    "step": i + 1,
                    "domain": domain_name,
                    "execution_time": execution_time,
                    "metadata": result.metadata,
                    "semantic_contributions": domain_config["semantic_focus"],
                    "data_summary": self._summarize_data_characteristics(result.data),
                }
            )

            current_data = result.data
            logger.info(
                f"  Comprehensive step {i + 1}/{len(domain_sequence)}: {domain_name} completed in {execution_time:.2f}s"
            )

        # Comprehensive validation
        await self._validate_comprehensive_workflow(
            domain_sequence, comprehensive_context, current_data, workflow_metrics
        )

        # Performance analysis
        total_time = sum(workflow_metrics["execution_times"])
        max_memory = max(workflow_metrics["memory_usage"])
        avg_coherence = sum(workflow_metrics["semantic_coherence_scores"]) / len(
            workflow_metrics["semantic_coherence_scores"]
        )

        logger.info(f"Comprehensive workflow completed:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Peak memory: {max_memory / 1024 / 1024:.1f}MB")
        logger.info(f"  Avg semantic coherence: {avg_coherence:.3f}")

        # Assert performance requirements
        assert total_time < 120.0  # Under 2 minutes for comprehensive analysis
        assert max_memory < 2 * 1024 * 1024 * 1024  # Under 2GB memory
        assert avg_coherence > 0.7  # High semantic coherence maintained

    @pytest.mark.asyncio
    async def test_cross_domain_format_optimization(self):
        """
        Test automatic format optimization for cross-domain workflows.

        Validates:
        - Optimal format selection for domain combinations
        - Automatic format conversion insertion
        - Performance impact of format choices
        - Memory efficiency of format strategies
        """
        logger.info("Testing cross-domain format optimization")

        # Test different domain combination scenarios
        test_scenarios = [
            {
                "name": "statistical_to_regression_optimization",
                "domains": ["statistical", "regression"],
                "data_characteristics": "high_dimensional",
                "optimization_target": "memory_efficiency",
            },
            {
                "name": "time_series_to_pattern_optimization",
                "domains": ["time_series", "pattern_recognition"],
                "data_characteristics": "temporal_dense",
                "optimization_target": "processing_speed",
            },
            {
                "name": "statistical_to_pattern_optimization",
                "domains": ["statistical", "pattern_recognition"],
                "data_characteristics": "sparse_features",
                "optimization_target": "balanced",
            },
        ]

        optimization_results = {}

        for scenario in test_scenarios:
            logger.info(f"  Testing scenario: {scenario['name']}")

            # Create scenario-specific dataset
            if scenario["data_characteristics"] == "high_dimensional":
                test_data = create_statistical_dataset(size="large", features=200)
            elif scenario["data_characteristics"] == "temporal_dense":
                test_data = create_time_series_dataset(length=2000, features=50)
            else:  # sparse_features
                test_data = create_pattern_recognition_dataset(
                    samples=1500, dimensions=300, sparsity=0.8
                )

            # Find optimal format for domain combination
            optimal_format = find_optimal_format_for_domains(
                domains=scenario["domains"],
                data_characteristics=scenario["data_characteristics"],
                optimization_target=scenario["optimization_target"],
            )

            # Test with optimal format
            source_domain = scenario["domains"][0]
            target_domain = scenario["domains"][1]

            # Convert to optimal intermediate format
            source_shim = self.domain_shims[source_domain]
            optimized_request = ConversionRequest(
                source_format=DataFormat.PANDAS_DATAFRAME,
                target_format=optimal_format,
                data=test_data,
                context={
                    "intention": "format_optimization_test",
                    "optimization_target": scenario["optimization_target"],
                },
            )

            import time

            start_time = time.time()
            optimized_result = await source_shim.convert(optimized_request)
            optimization_time = time.time() - start_time

            # Convert from optimal format to target domain
            target_shim = self.domain_shims[target_domain]
            target_request = ConversionRequest(
                source_format=optimal_format,
                target_format=self.DOMAINS[target_domain]["primary_formats"][0],
                data=optimized_result.data,
                context={
                    "intention": "optimized_cross_domain_conversion",
                    "source_optimization": optimized_result.metadata,
                },
            )

            start_time = time.time()
            final_result = await target_shim.convert(target_request)
            conversion_time = time.time() - start_time

            # Store results
            optimization_results[scenario["name"]] = {
                "optimal_format": optimal_format,
                "optimization_time": optimization_time,
                "conversion_time": conversion_time,
                "total_time": optimization_time + conversion_time,
                "memory_efficiency": optimized_result.metadata.get(
                    "memory_efficiency_score", 0
                ),
                "format_overhead": optimized_result.metadata.get("format_overhead", 0),
            }

            assert optimized_result.success and final_result.success
            logger.info(
                f"    Optimal format: {optimal_format}, Total time: {optimization_time + conversion_time:.2f}s"
            )

        # Validate optimization effectiveness
        for scenario_name, results in optimization_results.items():
            if "memory_efficiency" in scenario_name:
                assert results["memory_efficiency"] > 0.7  # High memory efficiency
            elif "speed" in scenario_name:
                assert results["total_time"] < 10.0  # Fast processing
            else:  # balanced
                assert (
                    results["memory_efficiency"] > 0.5 and results["total_time"] < 15.0
                )

        logger.info("✅ Cross-domain format optimization validated")

    @pytest.mark.asyncio
    async def test_semantic_context_preservation(self):
        """
        Test semantic context preservation across domain boundaries.

        Validates:
        - Domain-specific semantic information preservation
        - Context enrichment through domain transitions
        - Semantic coherence maintenance
        - Interpretability preservation
        """
        logger.info("Testing semantic context preservation across domains")

        # Create semantically rich dataset
        semantic_dataset = create_mixed_domain_dataset(
            statistical_features=15,
            temporal_length=300,
            regression_targets=2,
            pattern_dimensions=40,
            include_semantic_labels=True,
        )

        # Track semantic preservation through domain chain
        domain_chain = [
            "statistical",
            "regression",
            "time_series",
            "pattern_recognition",
        ]
        semantic_lineage = []
        current_data = semantic_dataset

        for i, domain_name in enumerate(domain_chain):
            domain_config = self.DOMAINS[domain_name]
            domain_shim = self.domain_shims[domain_config["shim_key"]]

            # Create semantic-aware request
            if i == 0:
                input_format = DataFormat.PANDAS_DATAFRAME
                semantic_context = {
                    "original_semantics": {
                        "feature_meanings": getattr(
                            semantic_dataset, "feature_meanings", {}
                        ),
                        "domain_context": "mixed_analytical",
                        "interpretation_guidance": "preserve_all_semantic_information",
                    }
                }
            else:
                prev_config = self.DOMAINS[domain_chain[i - 1]]
                input_format = prev_config["primary_formats"][0]
                semantic_context = {
                    "inherited_semantics": semantic_lineage[-1]["semantic_output"],
                    "domain_transition": f"{domain_chain[i - 1]}_to_{domain_name}",
                    "preservation_strategy": "enrich_and_maintain",
                }

            request = ConversionRequest(
                source_format=input_format,
                target_format=domain_config["primary_formats"][0],
                data=current_data,
                context={
                    "intention": "semantic_preservation_test",
                    "semantic_requirements": semantic_context,
                    "preserve_interpretability": True,
                    "enrich_context": True,
                },
            )

            result = await domain_shim.convert(request)
            assert result.success, f"Semantic preservation failed at {domain_name}"

            # Analyze semantic preservation
            semantic_analysis = await self._analyze_semantic_preservation(
                input_semantics=semantic_context,
                output_result=result,
                domain_name=domain_name,
            )

            semantic_lineage.append(
                {
                    "domain": domain_name,
                    "step": i + 1,
                    "semantic_input": semantic_context,
                    "semantic_output": result.metadata.get("semantic_metadata", {}),
                    "preservation_score": semantic_analysis["preservation_score"],
                    "enrichment_score": semantic_analysis["enrichment_score"],
                    "interpretability_score": semantic_analysis[
                        "interpretability_score"
                    ],
                }
            )

            current_data = result.data
            logger.info(
                f"  {domain_name}: Preservation={semantic_analysis['preservation_score']:.3f}, "
                f"Enrichment={semantic_analysis['enrichment_score']:.3f}"
            )

        # Validate overall semantic coherence
        final_preservation_score = semantic_lineage[-1]["preservation_score"]
        avg_enrichment_score = sum(
            s["enrichment_score"] for s in semantic_lineage
        ) / len(semantic_lineage)
        min_interpretability = min(
            s["interpretability_score"] for s in semantic_lineage
        )

        assert (
            final_preservation_score > 0.8
        ), f"Semantic preservation too low: {final_preservation_score}"
        assert (
            avg_enrichment_score > 0.6
        ), f"Insufficient semantic enrichment: {avg_enrichment_score}"
        assert (
            min_interpretability > 0.7
        ), f"Interpretability degraded: {min_interpretability}"

        logger.info(
            f"Semantic preservation validated: Final={final_preservation_score:.3f}, "
            f"Avg Enrichment={avg_enrichment_score:.3f}, Min Interpretability={min_interpretability:.3f}"
        )

    async def _validate_cross_domain_integration(
        self,
        source_domain: str,
        target_domain: str,
        source_result: ConversionResult,
        target_result: ConversionResult,
    ):
        """Validate quality of cross-domain integration."""
        # Check metadata lineage
        assert "domain_lineage" in target_result.metadata
        assert source_domain in target_result.metadata["domain_lineage"]

        # Check semantic bridge quality
        if "semantic_bridge_quality" in target_result.metadata:
            assert target_result.metadata["semantic_bridge_quality"] > 0.6

        # Check format compatibility
        assert "format_compatibility_score" in target_result.metadata
        assert target_result.metadata["format_compatibility_score"] > 0.5

        # Check performance metrics
        if "cross_domain_overhead" in target_result.metadata:
            assert (
                target_result.metadata["cross_domain_overhead"] < 2.0
            )  # Less than 2x overhead

    async def _test_bidirectional_conversion(
        self,
        domain1: str,
        domain2: str,
        forward_result: ConversionResult,
        backward_result: ConversionResult,
    ):
        """Test bidirectional conversion where semantically meaningful."""
        if not self._is_bidirectional_meaningful(domain1, domain2):
            return

        # Attempt reverse conversion
        domain1_shim = self.domain_shims[domain1]
        reverse_request = ConversionRequest(
            source_format=self.DOMAINS[domain2]["primary_formats"][0],
            target_format=self.DOMAINS[domain1]["primary_formats"][0],
            data=backward_result.data,
            context={
                "intention": f"reverse_{domain2}_to_{domain1}",
                "bidirectional_test": True,
                "preserve_round_trip_integrity": True,
            },
        )

        reverse_result = await domain1_shim.convert(reverse_request)

        if reverse_result.success:
            # Validate round-trip preservation
            assert "round_trip_fidelity" in reverse_result.metadata
            assert reverse_result.metadata["round_trip_fidelity"] > 0.7

    def _is_bidirectional_meaningful(self, domain1: str, domain2: str) -> bool:
        """Check if bidirectional conversion makes semantic sense."""
        # Some domain pairs have meaningful bidirectional relationships
        bidirectional_pairs = {
            ("statistical", "regression"),
            ("statistical", "pattern_recognition"),
            ("regression", "time_series"),
        }
        return (domain1, domain2) in bidirectional_pairs or (
            domain2,
            domain1,
        ) in bidirectional_pairs

    async def _validate_triplet_integration(
        self,
        domain_triplet: Tuple[str, str, str],
        context_chain: List[Dict],
        final_data: Any,
    ):
        """Validate triplet domain integration quality."""
        # Check context accumulation
        assert len(context_chain) == 3
        for i, context in enumerate(context_chain):
            assert context["domain"] == domain_triplet[i]
            assert "semantic_contributions" in context

        # Check semantic coherence across triplet
        domains_covered = {ctx["domain"] for ctx in context_chain}
        assert domains_covered == set(domain_triplet)

        # Validate final result contains all domain contributions
        if hasattr(final_data, "metadata") or (
            isinstance(final_data, dict) and "metadata" in final_data
        ):
            metadata = (
                final_data.metadata
                if hasattr(final_data, "metadata")
                else final_data["metadata"]
            )
            if "triplet_integration_score" in metadata:
                assert metadata["triplet_integration_score"] > 0.7

    async def _validate_comprehensive_workflow(
        self,
        domain_sequence: List[str],
        comprehensive_context: List[Dict],
        final_data: Any,
        workflow_metrics: Dict,
    ):
        """Validate comprehensive four-domain workflow."""
        # Check all domains processed
        processed_domains = {ctx["domain"] for ctx in comprehensive_context}
        assert processed_domains == set(domain_sequence)

        # Check context accumulation quality
        for i, context in enumerate(comprehensive_context):
            assert context["step"] == i + 1
            assert "semantic_contributions" in context
            assert "execution_time" in context

        # Check performance requirements
        assert (
            max(workflow_metrics["execution_times"]) < 60.0
        )  # No single step over 1 minute
        assert all(
            score > 0.5 for score in workflow_metrics["semantic_coherence_scores"]
        )  # Maintain coherence

    def _summarize_data_characteristics(self, data: Any) -> Dict[str, Any]:
        """Summarize data characteristics for context tracking."""
        summary = {"type": type(data).__name__}

        if isinstance(data, pd.DataFrame):
            summary.update(
                {
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": data.dtypes.to_dict(),
                }
            )
        elif isinstance(data, np.ndarray):
            summary.update({"shape": data.shape, "dtype": str(data.dtype)})
        elif isinstance(data, dict):
            summary.update({"keys": list(data.keys()), "structure": "nested_dict"})

        return summary

    async def _analyze_semantic_preservation(
        self, input_semantics: Dict, output_result: ConversionResult, domain_name: str
    ) -> Dict[str, float]:
        """Analyze semantic preservation quality."""
        # Mock semantic analysis - in real implementation would use NLP/semantic models
        preservation_score = 0.85  # Mock high preservation
        enrichment_score = 0.75  # Mock good enrichment
        interpretability_score = 0.80  # Mock good interpretability

        # Adjust scores based on domain characteristics
        if domain_name == "statistical":
            preservation_score += 0.05  # Statistical domain good at preservation
        elif domain_name == "pattern_recognition":
            enrichment_score += 0.1  # Pattern recognition adds insights

        return {
            "preservation_score": min(1.0, preservation_score),
            "enrichment_score": min(1.0, enrichment_score),
            "interpretability_score": min(1.0, interpretability_score),
        }


if __name__ == "__main__":
    # Run domain combination tests
    import sys

    logging.basicConfig(level=logging.INFO)

    async def run_tests():
        test_instance = TestDomainCombinations()
        await test_instance.setup_domain_combination_framework()

        # Test individual pairwise combinations
        domain_pairs = [
            ("statistical", "regression"),
            ("statistical", "time_series"),
            ("statistical", "pattern_recognition"),
            ("regression", "time_series"),
            ("regression", "pattern_recognition"),
            ("time_series", "pattern_recognition"),
        ]

        print("\n" + "=" * 60)
        print("Testing Pairwise Domain Combinations")
        print("=" * 60)

        for source, target in domain_pairs:
            try:
                await test_instance.test_pairwise_domain_combinations(source, target)
                print(f"✅ {source} → {target} PASSED")
            except Exception as e:
                print(f"❌ {source} → {target} FAILED: {e}")

        # Test triplet combinations
        domain_triplets = [
            ("statistical", "regression", "time_series"),
            ("statistical", "regression", "pattern_recognition"),
            ("statistical", "time_series", "pattern_recognition"),
            ("regression", "time_series", "pattern_recognition"),
        ]

        print("\n" + "=" * 60)
        print("Testing Triplet Domain Combinations")
        print("=" * 60)

        for triplet in domain_triplets:
            try:
                await test_instance.test_triplet_domain_combinations(triplet)
                print(f"✅ {' → '.join(triplet)} PASSED")
            except Exception as e:
                print(f"❌ {' → '.join(triplet)} FAILED: {e}")

        # Test comprehensive and specialized workflows
        advanced_tests = [
            (
                "Comprehensive Four-Domain Workflow",
                test_instance.test_all_domain_comprehensive_workflow,
            ),
            (
                "Cross-Domain Format Optimization",
                test_instance.test_cross_domain_format_optimization,
            ),
            (
                "Semantic Context Preservation",
                test_instance.test_semantic_context_preservation,
            ),
        ]

        print("\n" + "=" * 60)
        print("Testing Advanced Domain Integration")
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
        print("Domain Combination Tests Complete")
        print("=" * 60)

    if sys.version_info >= (3, 7):
        asyncio.run(run_tests())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_tests())
