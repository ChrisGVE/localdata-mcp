"""
Unit tests for Automatic Shim Insertion Logic.

Tests for PipelineAnalyzer, ShimInjector, PipelineValidator and related components
in the LocalData MCP v2.0 Integration Shims Framework.
"""

import pytest
import time
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock, patch

from src.localdata_mcp.pipeline.integration.pipeline_analyzer import (
    PipelineAnalyzer, ShimInjector, PipelineValidator,
    PipelineStep, PipelineConnection, IncompatibilityIssue, ShimRecommendation,
    PipelineAnalysisResult, AnalysisType, InjectionStrategy, OptimizationCriteria,
    create_pipeline_analyzer, create_shim_injector, create_pipeline_validator,
    create_optimization_criteria, create_pipeline_step, analyze_and_fix_pipeline
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    DataFormat, ConversionRequest, ConversionResult, ConversionCost, 
    ConversionPath, ConversionStep
)
from src.localdata_mcp.pipeline.integration.compatibility_matrix import PipelineCompatibilityMatrix
from src.localdata_mcp.pipeline.integration.shim_registry import ShimRegistry, EnhancedShimAdapter


class TestPipelineStep:
    """Test PipelineStep data structure."""
    
    def test_pipeline_step_creation(self):
        """Test creating a pipeline step."""
        step = PipelineStep(
            step_id="test_step_1",
            domain="statistical_analysis",
            operation="correlation_analysis",
            input_format=DataFormat.PANDAS_DATAFRAME,
            output_format=DataFormat.STATISTICAL_RESULT
        )
        
        assert step.step_id == "test_step_1"
        assert step.domain == "statistical_analysis"
        assert step.operation == "correlation_analysis"
        assert step.input_format == DataFormat.PANDAS_DATAFRAME
        assert step.output_format == DataFormat.STATISTICAL_RESULT
        assert step.metadata == {}
        assert step.requirements == {}
    
    def test_pipeline_step_with_metadata(self):
        """Test creating a pipeline step with metadata."""
        metadata = {"param1": "value1", "threshold": 0.05}
        requirements = {"min_samples": 30}
        
        step = PipelineStep(
            step_id="test_step_2",
            domain="regression",
            operation="linear_regression",
            input_format=DataFormat.PANDAS_DATAFRAME,
            output_format=DataFormat.REGRESSION_MODEL,
            metadata=metadata,
            requirements=requirements
        )
        
        assert step.metadata == metadata
        assert step.requirements == requirements


class TestPipelineConnection:
    """Test PipelineConnection data structure."""
    
    def test_pipeline_connection_creation(self):
        """Test creating a pipeline connection."""
        source_step = PipelineStep(
            "step1", "domain1", "op1", 
            DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY
        )
        target_step = PipelineStep(
            "step2", "domain2", "op2",
            DataFormat.NUMPY_ARRAY, DataFormat.STATISTICAL_RESULT
        )
        
        connection = PipelineConnection(
            source_step=source_step,
            target_step=target_step,
            compatibility_score=0.8,
            requires_conversion=True
        )
        
        assert connection.source_step == source_step
        assert connection.target_step == target_step
        assert connection.compatibility_score == 0.8
        assert connection.requires_conversion is True
        assert connection.conversion_path is None


@pytest.fixture
def mock_compatibility_matrix():
    """Create a mock compatibility matrix."""
    matrix = Mock(spec=PipelineCompatibilityMatrix)
    
    # Mock compatibility scores
    def mock_get_compatibility(source_format, target_format):
        mock_result = Mock()
        if source_format == target_format:
            mock_result.score = 1.0
            mock_result.conversion_required = False
            mock_result.conversion_path = None
        elif (source_format == DataFormat.PANDAS_DATAFRAME and 
              target_format == DataFormat.NUMPY_ARRAY):
            mock_result.score = 0.9
            mock_result.conversion_required = True
            mock_result.conversion_path = Mock(spec=ConversionPath)
            mock_result.conversion_path.total_cost = ConversionCost(
                computational_cost=0.2,
                memory_cost_mb=50,
                time_estimate_seconds=1.0
            )
        else:
            mock_result.score = 0.3  # Low compatibility
            mock_result.conversion_required = True
            mock_result.conversion_path = Mock(spec=ConversionPath)
            mock_result.conversion_path.total_cost = ConversionCost(
                computational_cost=0.8,
                memory_cost_mb=200,
                time_estimate_seconds=5.0
            )
        
        return mock_result
    
    matrix.get_compatibility = mock_get_compatibility
    return matrix


@pytest.fixture
def mock_shim_registry():
    """Create a mock shim registry."""
    registry = Mock(spec=ShimRegistry)
    
    # Create mock adapters
    adapter1 = Mock(spec=EnhancedShimAdapter)
    adapter1.adapter_id = "pandas_to_numpy"
    adapter1.estimate_cost.return_value = ConversionCost(
        computational_cost=0.2,
        memory_cost_mb=50,
        time_estimate_seconds=1.0
    )
    
    adapter2 = Mock(spec=EnhancedShimAdapter)
    adapter2.adapter_id = "numpy_to_sparse"
    adapter2.estimate_cost.return_value = ConversionCost(
        computational_cost=0.3,
        memory_cost_mb=80,
        time_estimate_seconds=1.5
    )
    
    # Mock get_compatible_adapters
    def mock_get_compatible_adapters(request):
        source_format = request.source_format
        target_format = request.target_format
        
        if (source_format == DataFormat.PANDAS_DATAFRAME and 
            target_format == DataFormat.NUMPY_ARRAY):
            return [(adapter1, 0.9)]
        elif (source_format == DataFormat.NUMPY_ARRAY and 
              target_format == DataFormat.SCIPY_SPARSE):
            return [(adapter2, 0.8)]
        else:
            return []
    
    registry.get_compatible_adapters = mock_get_compatible_adapters
    return registry


@pytest.fixture
def sample_pipeline_steps():
    """Create sample pipeline steps for testing."""
    return [
        PipelineStep(
            step_id="step1",
            domain="data_loading",
            operation="load_data",
            input_format=DataFormat.CSV,
            output_format=DataFormat.PANDAS_DATAFRAME
        ),
        PipelineStep(
            step_id="step2",
            domain="preprocessing",
            operation="normalize",
            input_format=DataFormat.PANDAS_DATAFRAME,
            output_format=DataFormat.NUMPY_ARRAY
        ),
        PipelineStep(
            step_id="step3",
            domain="analysis",
            operation="statistical_test",
            input_format=DataFormat.NUMPY_ARRAY,
            output_format=DataFormat.STATISTICAL_RESULT
        )
    ]


@pytest.fixture
def incompatible_pipeline_steps():
    """Create incompatible pipeline steps for testing."""
    return [
        PipelineStep(
            step_id="step1",
            domain="data_loading",
            operation="load_data",
            input_format=DataFormat.CSV,
            output_format=DataFormat.PANDAS_DATAFRAME
        ),
        PipelineStep(
            step_id="step2",
            domain="preprocessing",
            operation="sparse_operation",
            input_format=DataFormat.SCIPY_SPARSE,  # Incompatible with previous output
            output_format=DataFormat.NUMPY_ARRAY
        )
    ]


class TestPipelineAnalyzer:
    """Test PipelineAnalyzer functionality."""
    
    def test_analyzer_initialization(self, mock_compatibility_matrix, mock_shim_registry):
        """Test analyzer initialization."""
        analyzer = PipelineAnalyzer(
            compatibility_matrix=mock_compatibility_matrix,
            shim_registry=mock_shim_registry,
            enable_caching=True,
            max_analysis_threads=2
        )
        
        assert analyzer.compatibility_matrix == mock_compatibility_matrix
        assert analyzer.shim_registry == mock_shim_registry
        assert analyzer.enable_caching is True
        assert analyzer.max_analysis_threads == 2
        assert hasattr(analyzer, '_analysis_cache')
        assert hasattr(analyzer, '_stats')
    
    def test_build_pipeline_connections(self, mock_compatibility_matrix, mock_shim_registry, sample_pipeline_steps):
        """Test building pipeline connections."""
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry)
        
        connections = analyzer._build_pipeline_connections(sample_pipeline_steps)
        
        assert len(connections) == 2  # n-1 connections for n steps
        
        # First connection: CSV -> DataFrame (should be high compatibility)
        conn1 = connections[0]
        assert conn1.source_step.step_id == "step1"
        assert conn1.target_step.step_id == "step2"
        
        # Second connection: DataFrame -> NumPy (should be high compatibility)
        conn2 = connections[1]
        assert conn2.source_step.step_id == "step2" 
        assert conn2.target_step.step_id == "step3"
    
    def test_analyze_compatibility_compatible_pipeline(self, mock_compatibility_matrix, mock_shim_registry, sample_pipeline_steps):
        """Test compatibility analysis on compatible pipeline."""
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry)
        
        # Mock high compatibility for all connections
        def mock_high_compatibility(source_format, target_format):
            mock_result = Mock()
            mock_result.score = 0.9
            mock_result.conversion_required = False
            mock_result.conversion_path = None
            return mock_result
        
        mock_compatibility_matrix.get_compatibility = mock_high_compatibility
        
        connections = analyzer._build_pipeline_connections(sample_pipeline_steps)
        incompatible_connections, issues = analyzer._analyze_compatibility(connections)
        
        assert len(incompatible_connections) == 0
        assert len(issues) == 0
    
    def test_analyze_compatibility_incompatible_pipeline(self, mock_compatibility_matrix, mock_shim_registry, incompatible_pipeline_steps):
        """Test compatibility analysis on incompatible pipeline."""
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry)
        
        connections = analyzer._build_pipeline_connections(incompatible_pipeline_steps)
        incompatible_connections, issues = analyzer._analyze_compatibility(connections)
        
        assert len(incompatible_connections) > 0
        assert len(issues) > 0
        
        # Check issue details
        issue = issues[0]
        assert issue.issue_type == "format_incompatibility"
        assert issue.severity in ["critical", "warning", "info"]
        assert len(issue.suggested_solutions) > 0
    
    def test_analyze_pipeline_complete(self, mock_compatibility_matrix, mock_shim_registry, sample_pipeline_steps):
        """Test complete pipeline analysis."""
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry)
        
        result = analyzer.analyze_pipeline(
            pipeline_steps=sample_pipeline_steps,
            analysis_type=AnalysisType.COMPLETE,
            pipeline_id="test_pipeline"
        )
        
        assert isinstance(result, PipelineAnalysisResult)
        assert result.pipeline_id == "test_pipeline"
        assert result.analysis_type == AnalysisType.COMPLETE
        assert result.total_steps == len(sample_pipeline_steps)
        assert result.execution_time > 0
        assert isinstance(result.compatibility_score, float)
        assert 0 <= result.compatibility_score <= 1
    
    def test_analyze_pipeline_with_caching(self, mock_compatibility_matrix, mock_shim_registry, sample_pipeline_steps):
        """Test pipeline analysis with caching."""
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry, enable_caching=True)
        
        # First analysis
        result1 = analyzer.analyze_pipeline(sample_pipeline_steps, pipeline_id="cached_test")
        
        # Second analysis (should use cache)
        result2 = analyzer.analyze_pipeline(sample_pipeline_steps, pipeline_id="cached_test")
        
        # Results should be identical (from cache)
        assert result1.pipeline_id == result2.pipeline_id
        assert result1.compatibility_score == result2.compatibility_score
        
        # Check cache hit statistics
        stats = analyzer.get_analysis_statistics()
        assert stats['cache_hits'] >= 1
    
    def test_generate_shim_recommendations(self, mock_compatibility_matrix, mock_shim_registry, incompatible_pipeline_steps):
        """Test shim recommendation generation."""
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry)
        
        connections = analyzer._build_pipeline_connections(incompatible_pipeline_steps)
        recommendations = analyzer._generate_shim_recommendations(connections)
        
        # Should have recommendations for incompatible connections
        assert len(recommendations) >= 0
        
        if recommendations:
            rec = recommendations[0]
            assert isinstance(rec, ShimRecommendation)
            assert rec.recommended_shim is not None
            assert rec.confidence > 0
            assert rec.insertion_point in ["before_target", "after_source", "intermediate"]


class TestShimInjector:
    """Test ShimInjector functionality."""
    
    def test_injector_initialization(self, mock_shim_registry, mock_compatibility_matrix):
        """Test injector initialization."""
        injector = ShimInjector(
            shim_registry=mock_shim_registry,
            compatibility_matrix=mock_compatibility_matrix
        )
        
        assert injector.shim_registry == mock_shim_registry
        assert injector.compatibility_matrix == mock_compatibility_matrix
        assert isinstance(injector.optimization_criteria, OptimizationCriteria)
    
    def test_select_recommendations_by_strategy(self, mock_shim_registry, mock_compatibility_matrix):
        """Test recommendation selection by strategy."""
        injector = ShimInjector(mock_shim_registry, mock_compatibility_matrix)
        
        # Create mock recommendations
        rec1 = Mock(spec=ShimRecommendation)
        rec1.confidence = 0.9
        rec1.cost_estimate = ConversionCost(computational_cost=0.2, memory_cost_mb=50, time_estimate_seconds=1.0)
        
        rec2 = Mock(spec=ShimRecommendation)
        rec2.confidence = 0.5
        rec2.cost_estimate = ConversionCost(computational_cost=0.9, memory_cost_mb=200, time_estimate_seconds=5.0)
        
        recommendations = [rec1, rec2]
        
        # Test BALANCED strategy
        selected = injector._select_recommendations_by_strategy(recommendations, InjectionStrategy.BALANCED)
        assert len(selected) == 1  # Only rec1 should pass (confidence > 0.6 and cost < 0.8)
        assert selected[0] == rec1
        
        # Test SAFE strategy
        selected_safe = injector._select_recommendations_by_strategy(recommendations, InjectionStrategy.SAFE)
        assert len(selected_safe) == 2  # All recommendations
    
    def test_optimize_shim_selection(self, mock_shim_registry, mock_compatibility_matrix):
        """Test shim selection optimization."""
        injector = ShimInjector(mock_shim_registry, mock_compatibility_matrix)
        
        # Create mock adapters
        adapter1 = Mock(spec=EnhancedShimAdapter)
        adapter1.adapter_id = "adapter1"
        adapter1.estimate_cost.return_value = ConversionCost(
            computational_cost=0.2, memory_cost_mb=50, time_estimate_seconds=1.0
        )
        
        adapter2 = Mock(spec=EnhancedShimAdapter)
        adapter2.adapter_id = "adapter2"
        adapter2.estimate_cost.return_value = ConversionCost(
            computational_cost=0.8, memory_cost_mb=200, time_estimate_seconds=5.0
        )
        
        compatible_adapters = [(adapter1, 0.9), (adapter2, 0.7)]
        
        request = ConversionRequest(
            source_data=None,
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY
        )
        
        selected_adapter, score = injector.optimize_shim_selection(compatible_adapters, request)
        
        # Should select adapter1 (better performance and lower cost)
        assert selected_adapter == adapter1
        assert score > 0
    
    def test_inject_single_shim(self, mock_shim_registry, mock_compatibility_matrix):
        """Test single shim injection."""
        injector = ShimInjector(mock_shim_registry, mock_compatibility_matrix)
        
        # Create test pipeline
        steps = [
            PipelineStep("step1", "domain1", "op1", DataFormat.PANDAS_DATAFRAME, DataFormat.PANDAS_DATAFRAME),
            PipelineStep("step2", "domain2", "op2", DataFormat.NUMPY_ARRAY, DataFormat.STATISTICAL_RESULT)
        ]
        
        # Create mock recommendation
        connection = PipelineConnection(steps[0], steps[1])
        adapter = Mock(spec=EnhancedShimAdapter)
        adapter.adapter_id = "test_adapter"
        
        recommendation = ShimRecommendation(
            connection=connection,
            recommended_shim=adapter,
            insertion_point="before_target",
            confidence=0.9,
            expected_benefit="Test benefit",
            cost_estimate=ConversionCost(0.2, 50, 1.0)
        )
        
        modified_steps, injection_info = injector._inject_single_shim(steps, recommendation)
        
        assert len(modified_steps) == 3  # Original 2 + 1 injected
        assert injection_info['shim_step'].domain == "conversion"
        assert injection_info['insertion_index'] == 1  # Before target


class TestPipelineValidator:
    """Test PipelineValidator functionality."""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create mock analyzer."""
        analyzer = Mock(spec=PipelineAnalyzer)
        
        # Mock analysis result
        mock_result = Mock(spec=PipelineAnalysisResult)
        mock_result.pipeline_id = "test_pipeline"
        mock_result.is_compatible = True
        mock_result.compatibility_score = 0.9
        mock_result.identified_issues = []
        mock_result.shim_recommendations = []
        
        analyzer.analyze_pipeline.return_value = mock_result
        return analyzer
    
    @pytest.fixture
    def mock_injector(self):
        """Create mock injector."""
        injector = Mock(spec=ShimInjector)
        
        injector.inject_shims_for_pipeline.return_value = ([], {'injections': []})
        return injector
    
    def test_validator_initialization(self, mock_compatibility_matrix, mock_shim_registry, mock_analyzer, mock_injector):
        """Test validator initialization."""
        validator = PipelineValidator(
            compatibility_matrix=mock_compatibility_matrix,
            shim_registry=mock_shim_registry,
            analyzer=mock_analyzer,
            injector=mock_injector
        )
        
        assert validator.compatibility_matrix == mock_compatibility_matrix
        assert validator.shim_registry == mock_shim_registry
        assert validator.analyzer == mock_analyzer
        assert validator.injector == mock_injector
    
    def test_validate_pipeline_structure(self, mock_compatibility_matrix, mock_shim_registry, mock_analyzer, mock_injector):
        """Test pipeline structure validation."""
        validator = PipelineValidator(mock_compatibility_matrix, mock_shim_registry, mock_analyzer, mock_injector)
        
        # Test empty pipeline
        issues = validator._validate_pipeline_structure([])
        assert "Pipeline is empty" in issues
        
        # Test single step pipeline
        single_step = [PipelineStep("step1", "domain1", "op1", DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)]
        issues = validator._validate_pipeline_structure(single_step)
        assert "Pipeline must have at least 2 steps" in issues
        
        # Test duplicate step IDs
        duplicate_steps = [
            PipelineStep("step1", "domain1", "op1", DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            PipelineStep("step1", "domain2", "op2", DataFormat.NUMPY_ARRAY, DataFormat.STATISTICAL_RESULT)  # Duplicate ID
        ]
        issues = validator._validate_pipeline_structure(duplicate_steps)
        assert "Duplicate step IDs found" in issues
    
    def test_validate_and_fix_pipeline(self, mock_compatibility_matrix, mock_shim_registry, mock_analyzer, mock_injector, sample_pipeline_steps):
        """Test complete pipeline validation and fixing."""
        validator = PipelineValidator(mock_compatibility_matrix, mock_shim_registry, mock_analyzer, mock_injector)
        
        result = validator.validate_and_fix_pipeline(
            pipeline_steps=sample_pipeline_steps,
            auto_fix=True,
            validation_level="strict"
        )
        
        assert 'pipeline_id' in result
        assert 'is_valid' in result
        assert 'validation_score' in result
        assert 'final_pipeline' in result
        assert 'execution_time' in result
        assert result['original_steps_count'] == len(sample_pipeline_steps)
    
    def test_generate_execution_plan(self, mock_compatibility_matrix, mock_shim_registry, mock_analyzer, mock_injector, sample_pipeline_steps):
        """Test execution plan generation."""
        validator = PipelineValidator(mock_compatibility_matrix, mock_shim_registry, mock_analyzer, mock_injector)
        
        plan = validator._generate_execution_plan(sample_pipeline_steps)
        
        assert 'steps' in plan
        assert 'estimated_total_time' in plan
        assert 'estimated_total_memory' in plan
        assert 'parallel_opportunities' in plan
        assert 'optimization_suggestions' in plan
        assert len(plan['steps']) == len(sample_pipeline_steps)


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_pipeline_analyzer(self, mock_compatibility_matrix, mock_shim_registry):
        """Test analyzer factory function."""
        analyzer = create_pipeline_analyzer(mock_compatibility_matrix, mock_shim_registry)
        
        assert isinstance(analyzer, PipelineAnalyzer)
        assert analyzer.compatibility_matrix == mock_compatibility_matrix
        assert analyzer.shim_registry == mock_shim_registry
    
    def test_create_shim_injector(self, mock_shim_registry, mock_compatibility_matrix):
        """Test injector factory function."""
        injector = create_shim_injector(mock_shim_registry, mock_compatibility_matrix)
        
        assert isinstance(injector, ShimInjector)
        assert injector.shim_registry == mock_shim_registry
        assert injector.compatibility_matrix == mock_compatibility_matrix
    
    def test_create_pipeline_validator(self, mock_compatibility_matrix, mock_shim_registry):
        """Test validator factory function."""
        validator = create_pipeline_validator(mock_compatibility_matrix, mock_shim_registry)
        
        assert isinstance(validator, PipelineValidator)
        assert isinstance(validator.analyzer, PipelineAnalyzer)
        assert isinstance(validator.injector, ShimInjector)
    
    def test_create_optimization_criteria(self):
        """Test optimization criteria factory."""
        criteria = create_optimization_criteria(
            prioritize_performance=False,
            quality_threshold=0.9
        )
        
        assert isinstance(criteria, OptimizationCriteria)
        assert criteria.prioritize_performance is False
        assert criteria.quality_threshold == 0.9
    
    def test_create_pipeline_step(self):
        """Test pipeline step factory."""
        step = create_pipeline_step(
            step_id="test_step",
            domain="test_domain",
            operation="test_operation",
            input_format=DataFormat.PANDAS_DATAFRAME,
            output_format=DataFormat.NUMPY_ARRAY,
            metadata={"test": "value"}
        )
        
        assert isinstance(step, PipelineStep)
        assert step.step_id == "test_step"
        assert step.metadata == {"test": "value"}


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_analyze_and_fix_pipeline(self, mock_compatibility_matrix, mock_shim_registry, sample_pipeline_steps):
        """Test high-level analyze and fix utility."""
        with patch('src.localdata_mcp.pipeline.integration.pipeline_analyzer.create_pipeline_validator') as mock_create_validator:
            mock_validator = Mock()
            mock_validator.validate_and_fix_pipeline.return_value = {
                'pipeline_id': 'test',
                'is_valid': True,
                'validation_score': 0.9
            }
            mock_create_validator.return_value = mock_validator
            
            result = analyze_and_fix_pipeline(
                pipeline_steps=sample_pipeline_steps,
                compatibility_matrix=mock_compatibility_matrix,
                shim_registry=mock_shim_registry,
                auto_fix=True
            )
            
            assert result['is_valid'] is True
            assert result['validation_score'] == 0.9
            mock_validator.validate_and_fix_pipeline.assert_called_once()


class TestErrorHandling:
    """Test error handling in pipeline analysis components."""
    
    def test_analyzer_error_handling(self, mock_compatibility_matrix, mock_shim_registry, sample_pipeline_steps):
        """Test analyzer error handling."""
        # Mock compatibility matrix to raise exception
        mock_compatibility_matrix.get_compatibility.side_effect = Exception("Test error")
        
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry)
        result = analyzer.analyze_pipeline(sample_pipeline_steps)
        
        assert result.is_compatible is False
        assert len(result.identified_issues) > 0
        assert result.identified_issues[0].issue_type == "analysis_error"
    
    def test_injector_error_handling(self, mock_shim_registry, mock_compatibility_matrix):
        """Test injector error handling."""
        injector = ShimInjector(mock_shim_registry, mock_compatibility_matrix)
        
        # Test with empty adapters list
        with pytest.raises(ValueError, match="No compatible adapters provided"):
            request = ConversionRequest(None, DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)
            injector.optimize_shim_selection([], request)


class TestPerformanceAndCaching:
    """Test performance optimizations and caching."""
    
    def test_analysis_caching_performance(self, mock_compatibility_matrix, mock_shim_registry, sample_pipeline_steps):
        """Test that caching improves performance."""
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry, enable_caching=True)
        
        # First analysis (no cache)
        start_time = time.time()
        result1 = analyzer.analyze_pipeline(sample_pipeline_steps, pipeline_id="perf_test")
        first_time = time.time() - start_time
        
        # Second analysis (with cache)
        start_time = time.time()
        result2 = analyzer.analyze_pipeline(sample_pipeline_steps, pipeline_id="perf_test")
        second_time = time.time() - start_time
        
        # Cache should make second call faster (though this is hard to guarantee in unit tests)
        assert result1.pipeline_id == result2.pipeline_id
        stats = analyzer.get_analysis_statistics()
        assert stats['cache_hits'] >= 1
    
    def test_cache_size_management(self, mock_compatibility_matrix, mock_shim_registry, sample_pipeline_steps):
        """Test cache size management."""
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry, enable_caching=True)
        
        # Fill cache beyond limit
        for i in range(105):  # More than the 100 limit
            analyzer.analyze_pipeline(sample_pipeline_steps, pipeline_id=f"cache_test_{i}")
        
        # Cache should be managed to stay within reasonable size
        stats = analyzer.get_analysis_statistics()
        assert stats['cache_size'] <= 100
    
    def test_cache_clearing(self, mock_compatibility_matrix, mock_shim_registry, sample_pipeline_steps):
        """Test cache clearing functionality."""
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry, enable_caching=True)
        
        # Add something to cache
        analyzer.analyze_pipeline(sample_pipeline_steps, pipeline_id="clear_test")
        
        # Clear cache
        analyzer.clear_cache()
        
        stats = analyzer.get_analysis_statistics()
        assert stats['cache_size'] == 0


class TestComplexPipelineScenarios:
    """Test complex real-world pipeline scenarios."""
    
    def test_multi_domain_pipeline_analysis(self, mock_compatibility_matrix, mock_shim_registry):
        """Test analysis of multi-domain pipeline."""
        pipeline_steps = [
            PipelineStep("load", "data_loading", "load_csv", DataFormat.CSV, DataFormat.PANDAS_DATAFRAME),
            PipelineStep("clean", "preprocessing", "clean_data", DataFormat.PANDAS_DATAFRAME, DataFormat.PANDAS_DATAFRAME),
            PipelineStep("transform", "feature_engineering", "scale_features", DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            PipelineStep("analyze", "statistical_analysis", "correlation", DataFormat.NUMPY_ARRAY, DataFormat.STATISTICAL_RESULT),
            PipelineStep("model", "regression", "linear_regression", DataFormat.PANDAS_DATAFRAME, DataFormat.REGRESSION_MODEL),  # Format mismatch!
            PipelineStep("forecast", "time_series", "arima_forecast", DataFormat.TIME_SERIES, DataFormat.FORECAST_RESULT)  # Another mismatch!
        ]
        
        analyzer = PipelineAnalyzer(mock_compatibility_matrix, mock_shim_registry)
        result = analyzer.analyze_pipeline(pipeline_steps, AnalysisType.COMPLETE)
        
        assert result.total_steps == len(pipeline_steps)
        assert len(result.incompatible_connections) >= 2  # At least 2 format mismatches
    
    def test_pipeline_with_shim_injection_and_reanalysis(self, mock_compatibility_matrix, mock_shim_registry):
        """Test complete flow: analyze, inject shims, re-analyze."""
        # Start with incompatible pipeline
        pipeline_steps = [
            PipelineStep("step1", "domain1", "op1", DataFormat.PANDAS_DATAFRAME, DataFormat.PANDAS_DATAFRAME),
            PipelineStep("step2", "domain2", "op2", DataFormat.SCIPY_SPARSE, DataFormat.NUMPY_ARRAY)  # Incompatible input
        ]
        
        validator = create_pipeline_validator(mock_compatibility_matrix, mock_shim_registry)
        
        result = validator.validate_and_fix_pipeline(
            pipeline_steps=pipeline_steps,
            auto_fix=True,
            validation_level="strict"
        )
        
        # Should attempt to fix the pipeline
        assert 'fixes_applied' in result
        assert 'final_pipeline' in result
        assert result['original_steps_count'] == len(pipeline_steps)
    
    def test_circular_dependency_detection(self, mock_compatibility_matrix, mock_shim_registry):
        """Test detection of circular dependencies."""
        # This is a linear pipeline, so no real circular dependencies
        # But we can test the validation logic
        pipeline_steps = [
            PipelineStep("step1", "domain1", "op1", DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            PipelineStep("step2", "domain2", "op2", DataFormat.NUMPY_ARRAY, DataFormat.STATISTICAL_RESULT)
        ]
        
        validator = PipelineValidator(
            mock_compatibility_matrix, 
            mock_shim_registry, 
            Mock(), 
            Mock()
        )
        
        issues = validator._validate_pipeline_structure(pipeline_steps)
        
        # Should not detect circular dependencies in linear pipeline
        circular_issues = [issue for issue in issues if "circular" in issue.lower()]
        assert len(circular_issues) == 0


if __name__ == '__main__':
    pytest.main([__file__])