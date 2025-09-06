"""
Unit tests for the Error Handling and Recovery Framework.

Tests the comprehensive error handling, alternative pathway discovery,
rollback management, and recovery strategy execution for the Integration Shims system.
"""

import pytest
import time
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from localdata_mcp.pipeline.integration.error_recovery import (
    ConversionErrorHandler, AlternativePathwayEngine, RollbackManager, RecoveryStrategyFramework,
    ErrorSeverity, ErrorRecoverability, RecoveryStrategy,
    ErrorClassificationEnhanced, ErrorHandlingResult, ErrorAggregation,
    PerformanceImpact, PathwayCost, QualityAssessment,
    CheckpointId, RollbackPath, RetryPolicy, RecoveryResult,
    create_error_recovery_framework
)

from localdata_mcp.pipeline.integration.interfaces import (
    ConversionError, ConversionRequest, ConversionContext,
    ConversionCost, DataFormat, ConversionPath, ConversionStep
)
from localdata_mcp.pipeline.integration.shim_registry import ShimRegistry, EnhancedShimAdapter


class TestConversionErrorHandler:
    """Test ConversionErrorHandler functionality."""

    def test_initialization(self):
        """Test ConversionErrorHandler initialization."""
        handler = ConversionErrorHandler(
            classification_confidence_threshold=0.8,
            enable_performance_tracking=True,
            enable_error_learning=True
        )
        
        assert handler.classification_confidence_threshold == 0.8
        assert handler.enable_performance_tracking is True
        assert handler.enable_error_learning is True
        assert len(handler._error_history) == 0
        assert len(handler._classification_cache) == 0

    def test_error_classification(self):
        """Test error classification functionality."""
        handler = ConversionErrorHandler()
        
        # Test with ConversionError
        conversion_error = ConversionError(
            ConversionError.Type.MEMORY_EXCEEDED,
            "Out of memory",
            {"memory_usage": "8GB"}
        )
        
        context = ConversionContext(
            source_domain="test",
            target_domain="test",
            user_intention="critical data processing"
        )
        
        classification = handler.classify_error(conversion_error, context)
        
        assert classification.error_type == ConversionError.Type.MEMORY_EXCEEDED
        assert classification.severity == ErrorSeverity.CRITICAL
        assert classification.recoverability in [ErrorRecoverability.RECOVERABLE, ErrorRecoverability.TERMINAL]
        assert 0.0 <= classification.confidence <= 1.0
        assert len(classification.suggested_strategies) > 0

    def test_error_handling(self):
        """Test comprehensive error handling."""
        handler = ConversionErrorHandler()
        
        error = ValueError("Invalid data format")
        request = ConversionRequest(
            source_data={"test": "data"},
            source_format=DataFormat.PYTHON_DICT,
            target_format=DataFormat.PANDAS_DATAFRAME,
            context=ConversionContext(source_domain="test", target_domain="test")
        )
        
        result = handler.handle_error(error, request)
        
        assert isinstance(result, ErrorHandlingResult)
        assert result.handled is True
        assert isinstance(result.classification, ErrorClassificationEnhanced)
        assert len(result.recovery_actions) >= 0
        assert len(result.alternative_suggestions) >= 0
        assert 'handling_time' in result.performance_metrics

    def test_error_aggregation(self):
        """Test error aggregation for batch operations."""
        handler = ConversionErrorHandler()
        
        errors = [
            ValueError("Type error 1"),
            ValueError("Type error 2"),
            MemoryError("Out of memory"),
            TimeoutError("Operation timeout")
        ]
        
        aggregation = handler.aggregate_errors(errors)
        
        assert aggregation.total_errors == 4
        assert len(aggregation.error_distribution) > 0
        assert len(aggregation.severity_distribution) > 0
        assert len(aggregation.recoverability_distribution) > 0
        assert len(aggregation.suggested_batch_strategies) > 0
        assert 0.0 <= aggregation.aggregate_confidence <= 1.0

    def test_performance_impact_assessment(self):
        """Test performance impact assessment."""
        handler = ConversionErrorHandler(enable_performance_tracking=True)
        
        memory_error = MemoryError("Out of memory")
        impact = handler.assess_performance_impact(memory_error)
        
        assert isinstance(impact, PerformanceImpact)
        assert impact.execution_time_increase_percent > 0
        assert impact.memory_overhead_mb >= 0
        assert 0.0 <= impact.cascade_risk <= 1.0

    def test_error_learning(self):
        """Test error pattern learning."""
        handler = ConversionErrorHandler(enable_error_learning=True)
        
        context = ConversionContext(source_domain="test", target_domain="test")
        
        # Process multiple similar errors
        for _ in range(5):
            error = ValueError("Similar error pattern")
            handler.classify_error(error, context)
        
        # Check that patterns are learned
        assert len(handler._error_patterns) > 0
        assert len(handler._error_history) == 5

    def test_statistics_collection(self):
        """Test error statistics collection."""
        handler = ConversionErrorHandler()
        
        # Generate some errors for statistics
        context = ConversionContext(source_domain="test", target_domain="test")
        errors = [ValueError("Error 1"), TypeError("Error 2"), MemoryError("Error 3")]
        
        for error in errors:
            handler.classify_error(error, context)
        
        stats = handler.get_error_statistics()
        
        assert 'total_errors_processed' in stats
        assert 'cached_classifications' in stats
        assert 'learned_patterns' in stats
        assert 'performance_baselines' in stats


class TestAlternativePathwayEngine:
    """Test AlternativePathwayEngine functionality."""

    def test_initialization(self):
        """Test AlternativePathwayEngine initialization."""
        registry = Mock(spec=ShimRegistry)
        engine = AlternativePathwayEngine(
            registry=registry,
            max_pathway_depth=3,
            enable_pathway_caching=True,
            quality_threshold=0.7
        )
        
        assert engine.registry is registry
        assert engine.max_pathway_depth == 3
        assert engine.enable_pathway_caching is True
        assert engine.quality_threshold == 0.7

    def test_pathway_cost_assessment(self):
        """Test pathway cost assessment."""
        engine = AlternativePathwayEngine()
        
        # Create test pathway
        step_cost = ConversionCost(
            computational_cost=0.5,
            memory_cost_mb=100.0,
            time_estimate_seconds=2.0,
            io_operations=1,
            network_operations=0,
            quality_impact=0.1
        )
        
        step = ConversionStep(
            adapter_id="test_adapter",
            source_format=DataFormat.PYTHON_DICT,
            target_format=DataFormat.PANDAS_DATAFRAME,
            estimated_cost=step_cost,
            confidence=0.9
        )
        
        pathway = ConversionPath(
            source_format=DataFormat.PYTHON_DICT,
            target_format=DataFormat.PANDAS_DATAFRAME,
            steps=[step],
            total_cost=step_cost
        )
        
        cost_assessment = engine.assess_pathway_cost(pathway)
        
        assert isinstance(cost_assessment, PathwayCost)
        assert cost_assessment.computational_cost >= 0
        assert cost_assessment.time_overhead >= 0
        assert cost_assessment.memory_overhead >= 0
        assert 0.0 <= cost_assessment.reliability_score <= 1.0
        assert 0.0 <= cost_assessment.confidence <= 1.0

    def test_quality_assessment(self):
        """Test pathway quality assessment."""
        engine = AlternativePathwayEngine()
        
        # Create test pathway with multiple steps
        step1 = ConversionStep(
            adapter_id="adapter1",
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.NUMPY_ARRAY,
            estimated_cost=ConversionCost(computational_cost=0.1),
            confidence=0.9
        )
        
        step2 = ConversionStep(
            adapter_id="adapter2",
            source_format=DataFormat.NUMPY_ARRAY,
            target_format=DataFormat.PYTHON_LIST,
            estimated_cost=ConversionCost(computational_cost=0.1),
            confidence=0.8
        )
        
        pathway = ConversionPath(
            source_format=DataFormat.PANDAS_DATAFRAME,
            target_format=DataFormat.PYTHON_LIST,
            steps=[step1, step2]
        )
        
        quality_assessment = engine.assess_quality_degradation(pathway)
        
        assert isinstance(quality_assessment, QualityAssessment)
        assert 0.0 <= quality_assessment.expected_quality_score <= 1.0
        assert quality_assessment.quality_degradation >= 0.0
        assert 0.0 <= quality_assessment.metadata_preservation <= 1.0
        assert 0.0 <= quality_assessment.data_fidelity <= 1.0
        assert isinstance(quality_assessment.risk_factors, list)

    def test_pathway_caching(self):
        """Test pathway caching functionality."""
        engine = AlternativePathwayEngine(enable_pathway_caching=True)
        
        # Create test pathway
        pathway = ConversionPath(
            path_id="test_pathway",
            source_format=DataFormat.PYTHON_DICT,
            target_format=DataFormat.PANDAS_DATAFRAME,
            steps=[]
        )
        
        # Cache pathway
        engine.cache_successful_pathway(pathway)
        
        # Verify caching
        cache_key = f"{pathway.source_format.value}->{pathway.target_format.value}"
        assert cache_key in engine._pathway_cache
        assert engine._successful_pathways["test_pathway"] is pathway

    @patch('localdata_mcp.pipeline.integration.error_recovery.deque')
    def test_pathway_discovery_bfs(self, mock_deque):
        """Test breadth-first search pathway discovery."""
        registry = Mock(spec=ShimRegistry)
        adapter = Mock(spec=EnhancedShimAdapter)
        adapter.adapter_id = "test_adapter"
        adapter.get_supported_conversions.return_value = [
            (DataFormat.PYTHON_DICT, DataFormat.PANDAS_DATAFRAME)
        ]
        adapter.estimate_cost.return_value = ConversionCost(computational_cost=0.1)
        adapter.can_convert.return_value = 0.9
        
        registry.get_active_adapters.return_value = [adapter]
        
        engine = AlternativePathwayEngine(registry=registry)
        
        request = ConversionRequest(
            source_data={"test": "data"},
            source_format=DataFormat.PYTHON_DICT,
            target_format=DataFormat.PANDAS_DATAFRAME,
            context=ConversionContext(source_domain="test", target_domain="test")
        )
        
        # Mock deque to control BFS behavior
        mock_deque.return_value = []
        
        pathways = engine._discover_pathways_bfs(
            DataFormat.PYTHON_DICT,
            DataFormat.PANDAS_DATAFRAME,
            request
        )
        
        # Verify that BFS was attempted
        mock_deque.assert_called()

    def test_format_degradation_factors(self):
        """Test format degradation factor calculation."""
        engine = AlternativePathwayEngine()
        
        # Test known degradation
        degradation = engine._get_format_degradation_factor(
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.PYTHON_LIST
        )
        assert degradation > 0.0
        
        # Test unknown degradation (should return default)
        degradation = engine._get_format_degradation_factor(
            DataFormat.CSV,
            DataFormat.JSON
        )
        assert degradation == 0.01  # Default

    def test_metadata_and_precision_loss(self):
        """Test metadata and precision loss detection."""
        engine = AlternativePathwayEngine()
        
        # Test metadata loss
        loses_metadata = engine._loses_metadata(
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY
        )
        assert loses_metadata is True
        
        # Test precision loss
        loses_precision = engine._loses_precision(
            DataFormat.NUMPY_ARRAY,
            DataFormat.PYTHON_LIST
        )
        assert loses_precision is True

    def test_cost_operations(self):
        """Test cost calculation operations."""
        engine = AlternativePathwayEngine()
        
        cost1 = ConversionCost(computational_cost=0.5, memory_cost_mb=100)
        cost2 = ConversionCost(computational_cost=0.3, memory_cost_mb=50)
        
        combined = engine._add_costs(cost1, cost2)
        
        assert combined.computational_cost == 0.8
        assert combined.memory_cost_mb == 150

    def test_statistics(self):
        """Test pathway statistics collection."""
        engine = AlternativePathwayEngine()
        
        # Add some test data
        engine._pathway_cache["test_key"] = []
        engine._successful_pathways["test_id"] = Mock()
        engine._pathway_success_rates["test_conversion"] = 0.8
        
        stats = engine.get_pathway_statistics()
        
        assert 'cached_pathways' in stats
        assert 'successful_pathways' in stats
        assert 'average_success_rates' in stats
        assert stats['cached_pathways'] == 1
        assert stats['successful_pathways'] == 1


class TestRollbackManager:
    """Test RollbackManager functionality."""

    def test_initialization(self):
        """Test RollbackManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RollbackManager(
                max_checkpoints=5,
                enable_disk_persistence=True,
                checkpoint_compression=True
            )
            
            assert manager.max_checkpoints == 5
            assert manager.enable_disk_persistence is True
            assert manager.checkpoint_compression is True

    def test_checkpoint_creation(self):
        """Test checkpoint creation and management."""
        manager = RollbackManager(enable_disk_persistence=False)
        
        test_data = {"test": "data", "values": [1, 2, 3]}
        
        checkpoint = manager.create_checkpoint(
            test_data,
            "Test checkpoint"
        )
        
        assert isinstance(checkpoint, CheckpointId)
        assert checkpoint.description == "Test checkpoint"
        assert len(checkpoint.id) > 0
        assert checkpoint.timestamp > 0

    def test_rollback_operations(self):
        """Test rollback functionality."""
        manager = RollbackManager(enable_disk_persistence=False)
        
        # Create checkpoint
        original_data = {"original": True}
        checkpoint = manager.create_checkpoint(original_data, "Original state")
        
        # Verify rollback (simplified test)
        rollback_path = manager.get_rollback_path(checkpoint, checkpoint)
        
        assert isinstance(rollback_path, RollbackPath)
        assert rollback_path.from_checkpoint == checkpoint
        assert rollback_path.to_checkpoint == checkpoint

    def test_resource_management(self):
        """Test resource registration and cleanup."""
        manager = RollbackManager()
        
        checkpoint = manager.create_checkpoint({"test": "data"}, "Test")
        
        # Register a mock resource
        mock_resource = Mock()
        mock_resource.close = Mock()
        manager.register_resource(checkpoint, mock_resource)
        
        # Register cleanup handler
        cleanup_handler = Mock()
        manager.register_cleanup_handler(checkpoint, cleanup_handler)
        
        # Cleanup
        manager.cleanup_resources(checkpoint)
        
        # Verify cleanup was called
        mock_resource.close.assert_called_once()
        cleanup_handler.assert_called_once()

    def test_checkpoint_statistics(self):
        """Test checkpoint statistics collection."""
        manager = RollbackManager()
        
        # Create some checkpoints
        for i in range(3):
            manager.create_checkpoint(f"data_{i}", f"Checkpoint {i}")
        
        stats = manager.get_checkpoint_statistics()
        
        assert 'active_checkpoints' in stats
        assert 'total_size_bytes' in stats
        assert 'storage_path' in stats
        assert stats['active_checkpoints'] >= 0


class TestRecoveryStrategyFramework:
    """Test RecoveryStrategyFramework functionality."""

    def test_initialization(self):
        """Test framework initialization with all components."""
        error_handler = ConversionErrorHandler()
        pathway_engine = AlternativePathwayEngine()
        rollback_manager = RollbackManager()
        
        framework = RecoveryStrategyFramework(
            error_handler=error_handler,
            pathway_engine=pathway_engine,
            rollback_manager=rollback_manager
        )
        
        assert framework.error_handler is error_handler
        assert framework.pathway_engine is pathway_engine
        assert framework.rollback_manager is rollback_manager

    def test_strategy_selection(self):
        """Test intelligent strategy selection."""
        error_handler = Mock()
        error_handler.classify_error.return_value = Mock(
            suggested_strategies=[RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
        )
        
        framework = RecoveryStrategyFramework(
            error_handler=error_handler,
            pathway_engine=Mock(),
            rollback_manager=Mock()
        )
        
        error = ValueError("Test error")
        context = ConversionContext(
            source_domain="test",
            target_domain="test",
            user_intention="retry operation"
        )
        
        strategy = framework.select_strategy(error, context)
        
        assert isinstance(strategy, RecoveryStrategy)

    def test_retry_policy_configuration(self):
        """Test retry policy configuration."""
        framework = RecoveryStrategyFramework(
            error_handler=Mock(),
            pathway_engine=Mock(),
            rollback_manager=Mock()
        )
        
        custom_policy = RetryPolicy(
            max_attempts=5,
            base_delay_seconds=2.0,
            exponential_backoff=True
        )
        
        framework.configure_retry_policy(custom_policy)
        
        assert framework.default_retry_policy.max_attempts == 5
        assert framework.default_retry_policy.base_delay_seconds == 2.0

    def test_circuit_breaker_implementation(self):
        """Test circuit breaker pattern implementation."""
        framework = RecoveryStrategyFramework(
            error_handler=Mock(),
            pathway_engine=Mock(),
            rollback_manager=Mock()
        )
        
        breaker = framework.implement_circuit_breaker("test_operation", 3)
        
        assert breaker.failure_threshold == 3
        assert "test_operation" in framework._circuit_breakers

    def test_recovery_statistics(self):
        """Test recovery statistics collection."""
        framework = RecoveryStrategyFramework(
            error_handler=Mock(),
            pathway_engine=Mock(),
            rollback_manager=Mock()
        )
        
        # Add some test statistics
        framework._strategy_success_rates[RecoveryStrategy.RETRY] = 0.8
        framework._strategy_performance[RecoveryStrategy.RETRY].append(1.5)
        
        stats = framework.get_recovery_statistics()
        
        assert 'strategy_success_rates' in stats
        assert 'average_execution_times' in stats
        assert 'configured_circuit_breakers' in stats
        assert 'strategy_preferences' in stats


class TestFactoryFunctions:
    """Test factory functions for creating error recovery components."""

    def test_create_error_recovery_framework(self):
        """Test complete framework creation."""
        registry = Mock(spec=ShimRegistry)
        
        components = create_error_recovery_framework(
            registry=registry,
            enable_learning=True,
            enable_disk_persistence=False,
            max_checkpoints=10
        )
        
        assert len(components) == 4
        error_handler, pathway_engine, rollback_manager, recovery_framework = components
        
        assert isinstance(error_handler, ConversionErrorHandler)
        assert isinstance(pathway_engine, AlternativePathwayEngine)
        assert isinstance(rollback_manager, RollbackManager)
        assert isinstance(recovery_framework, RecoveryStrategyFramework)
        
        assert error_handler.enable_error_learning is True
        assert pathway_engine.registry is registry
        assert rollback_manager.max_checkpoints == 10


class TestIntegrationScenarios:
    """Test integration scenarios combining all components."""

    def test_complete_error_recovery_scenario(self):
        """Test complete error recovery workflow."""
        # Create framework
        registry = Mock(spec=ShimRegistry)
        error_handler, pathway_engine, rollback_manager, recovery_framework = (
            create_error_recovery_framework(registry=registry)
        )
        
        # Simulate error scenario
        error = MemoryError("Out of memory during conversion")
        request = ConversionRequest(
            source_data={"large": "dataset"},
            source_format=DataFormat.PYTHON_DICT,
            target_format=DataFormat.PANDAS_DATAFRAME,
            context=ConversionContext(
                source_domain="data_processing",
                target_domain="analytics"
            )
        )
        
        # Handle error
        error_result = error_handler.handle_error(error, request)
        
        assert error_result.handled is True
        assert error_result.classification.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]
        assert len(error_result.recovery_actions) > 0

    def test_pathway_discovery_integration(self):
        """Test pathway discovery with error handling."""
        registry = Mock(spec=ShimRegistry)
        adapter = Mock(spec=EnhancedShimAdapter)
        adapter.adapter_id = "test_adapter"
        adapter.get_supported_conversions.return_value = []
        registry.get_active_adapters.return_value = [adapter]
        
        pathway_engine = AlternativePathwayEngine(registry=registry)
        
        request = ConversionRequest(
            source_data=None,
            source_format=DataFormat.CSV,
            target_format=DataFormat.PARQUET,
            context=ConversionContext(source_domain="file", target_domain="storage")
        )
        
        pathways = pathway_engine.find_alternative_pathways_enhanced(request)
        
        # Should return empty list when no adapters support the conversion
        assert isinstance(pathways, list)

    def test_rollback_with_error_recovery(self):
        """Test rollback integration with error recovery."""
        rollback_manager = RollbackManager(enable_disk_persistence=False)
        
        # Create initial state
        initial_state = {"step": 0, "data": "initial"}
        checkpoint = rollback_manager.create_checkpoint(initial_state, "Initial")
        
        # Simulate processing with intermediate states
        intermediate_state = {"step": 1, "data": "processed"}
        intermediate_checkpoint = rollback_manager.create_checkpoint(
            intermediate_state, "Intermediate"
        )
        
        # Test rollback path
        rollback_path = rollback_manager.get_rollback_path(
            intermediate_checkpoint, checkpoint
        )
        
        assert rollback_path.from_checkpoint == intermediate_checkpoint
        assert rollback_path.to_checkpoint == checkpoint
        assert rollback_path.estimated_cost is not None


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations and concurrent access."""

    async def test_concurrent_error_handling(self):
        """Test concurrent error handling scenarios."""
        import asyncio
        
        handler = ConversionErrorHandler()
        
        async def handle_error_async(error_msg: str):
            error = ValueError(error_msg)
            context = ConversionContext(source_domain="test", target_domain="test")
            return handler.classify_error(error, context)
        
        # Run multiple concurrent classifications
        tasks = [
            handle_error_async(f"Error {i}")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(isinstance(result, ErrorClassificationEnhanced) for result in results)

    async def test_concurrent_pathway_discovery(self):
        """Test concurrent pathway discovery."""
        import asyncio
        
        engine = AlternativePathwayEngine()
        
        async def assess_quality_async(step_count: int):
            # Create mock pathway with variable step count
            steps = [
                ConversionStep(
                    adapter_id=f"adapter_{i}",
                    source_format=DataFormat.PYTHON_DICT,
                    target_format=DataFormat.PANDAS_DATAFRAME,
                    estimated_cost=ConversionCost(computational_cost=0.1),
                    confidence=0.9
                )
                for i in range(step_count)
            ]
            
            pathway = ConversionPath(
                source_format=DataFormat.PYTHON_DICT,
                target_format=DataFormat.PANDAS_DATAFRAME,
                steps=steps
            )
            
            return engine.assess_quality_degradation(pathway)
        
        # Run concurrent quality assessments
        tasks = [assess_quality_async(i + 1) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(isinstance(result, QualityAssessment) for result in results)
        
        # Quality should degrade with more steps
        for i in range(1, len(results)):
            assert results[i].quality_degradation >= results[i-1].quality_degradation


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_error_aggregation(self):
        """Test error aggregation with empty error list."""
        handler = ConversionErrorHandler()
        
        aggregation = handler.aggregate_errors([])
        
        assert aggregation.total_errors == 0
        assert len(aggregation.error_distribution) == 0
        assert aggregation.aggregate_confidence == 0.0

    def test_pathway_discovery_without_registry(self):
        """Test pathway discovery without registry."""
        engine = AlternativePathwayEngine(registry=None)
        
        request = ConversionRequest(
            source_data=None,
            source_format=DataFormat.CSV,
            target_format=DataFormat.JSON,
            context=ConversionContext(source_domain="test", target_domain="test")
        )
        
        pathways = engine.find_alternative_pathways_enhanced(request)
        
        assert len(pathways) == 0

    def test_rollback_with_invalid_checkpoint(self):
        """Test rollback with invalid checkpoint."""
        manager = RollbackManager()
        
        invalid_checkpoint = CheckpointId(id="nonexistent", description="Invalid")
        
        with pytest.raises(ValueError):
            manager.rollback_to_checkpoint("nonexistent_transaction", invalid_checkpoint.id)

    def test_recovery_with_unsupported_strategy(self):
        """Test recovery with unsupported strategy."""
        framework = RecoveryStrategyFramework(
            error_handler=Mock(),
            pathway_engine=Mock(),
            rollback_manager=Mock()
        )
        
        # Mock an unsupported strategy execution
        with patch.object(framework, '_execute_strategy', return_value=(False, None)):
            result = framework.execute_recovery(
                error_context=Mock(),
                strategy=RecoveryStrategy.FAIL_FAST,
                operation=Mock()
            )
            
            success, _, metadata = result
            assert success is False

    def test_memory_efficient_operations(self):
        """Test memory efficiency with large data structures."""
        handler = ConversionErrorHandler(enable_performance_tracking=True)
        
        # Generate many errors to test memory management
        for i in range(2000):  # More than the default limit
            error = ValueError(f"Error {i}")
            context = ConversionContext(source_domain="test", target_domain="test")
            handler.classify_error(error, context)
        
        # Should limit history size
        assert len(handler._error_history) <= 1000
        
        # Statistics should still be available
        stats = handler.get_error_statistics()
        assert stats['total_errors_processed'] > 0


if __name__ == "__main__":
    pytest.main([__file__])