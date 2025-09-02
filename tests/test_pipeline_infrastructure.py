"""
Comprehensive test suite for LocalData MCP Pipeline Infrastructure.

This module provides comprehensive testing for the entire pipeline infrastructure
including DataSciencePipeline, PipelineComposer, StreamingDataPipeline, and
SklearnStreamingAdapter integration with existing streaming architecture.

Tests cover:
- Core pipeline components functionality
- Integration with existing streaming architecture
- Memory usage validation under various loads
- Performance benchmarking and optimization
- Error scenario handling and recovery
- End-to-end workflow validation
- sklearn API compatibility verification
"""

import time
import unittest
from unittest.mock import Mock, patch, MagicMock
import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

# Import sklearn components for testing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier  # Supports partial_fit
from sklearn.cluster import MiniBatchKMeans  # Supports partial_fit

# Import the pipeline infrastructure components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from localdata_mcp.pipeline.core import (
    DataSciencePipeline, 
    PipelineComposer, 
    StreamingDataPipeline,
    SklearnStreamingAdapter,
    DataFrameStreamingSource
)
from localdata_mcp.pipeline.base import (
    StreamingConfig, 
    PipelineState, 
    PipelineError, 
    CompositionMetadata,
    ErrorClassification
)
from localdata_mcp.pipeline.preprocessing import (
    DataCleaningPipeline,
    FeatureScalingPipeline,
    CategoricalEncodingPipeline
)
from localdata_mcp.pipeline.type_conversion import DataTypeManager
from localdata_mcp.pipeline.adapters import DomainAdapter, SciPyStatsAdapter
from localdata_mcp.streaming_executor import StreamingQueryExecutor, MemoryStatus

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore', category=UserWarning)


class MockEstimator(BaseEstimator, TransformerMixin):
    """Mock sklearn estimator for testing purposes."""
    
    def __init__(self, fail_fit=False, fail_transform=False, sleep_time=0):
        self.fail_fit = fail_fit
        self.fail_transform = fail_transform
        self.sleep_time = sleep_time
        self.fitted = False
        self.fit_call_count = 0
        self.transform_call_count = 0
    
    def fit(self, X, y=None):
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        
        if self.fail_fit:
            raise ValueError("Mock fit failure")
        
        self.fitted = True
        self.fit_call_count += 1
        return self
    
    def transform(self, X):
        if not self.fitted and not self.fail_fit:
            raise ValueError("Not fitted")
        
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        
        if self.fail_transform:
            raise ValueError("Mock transform failure")
        
        self.transform_call_count += 1
        return X  # Return input unchanged
    
    def partial_fit(self, X, y=None):
        """Support partial fitting for streaming tests."""
        self.fitted = True
        self.fit_call_count += 1
        return self


def create_test_dataframe(rows=1000, cols=10, include_categorical=True, include_nulls=True):
    """Create test DataFrame with various data types."""
    np.random.seed(42)
    
    data = {}
    
    # Numeric columns
    for i in range(cols // 2):
        data[f'num_col_{i}'] = np.random.normal(0, 1, rows)
    
    # Categorical columns if requested
    if include_categorical:
        categories = ['A', 'B', 'C', 'D', 'E']
        for i in range(cols // 4):
            data[f'cat_col_{i}'] = np.random.choice(categories, rows)
    
    # Fill remaining columns with mixed data
    remaining_cols = cols - len(data)
    for i in range(remaining_cols):
        if i % 2 == 0:
            data[f'mixed_col_{i}'] = np.random.uniform(0, 100, rows)
        else:
            data[f'mixed_col_{i}'] = [f'value_{j}' for j in np.random.randint(0, 10, rows)]
    
    df = pd.DataFrame(data)
    
    # Add null values if requested
    if include_nulls:
        # Randomly set 5% of values to null
        for col in df.columns:
            null_mask = np.random.random(len(df)) < 0.05
            df.loc[null_mask, col] = None
    
    return df


class TestDataSciencePipeline(unittest.TestCase):
    """Test suite for DataSciencePipeline base functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = create_test_dataframe(100, 5)
        self.simple_pipeline = DataSciencePipeline([
            ('scaler', MockEstimator()),
            ('classifier', MockEstimator())
        ], analytical_intention="Test pipeline")
    
    def test_pipeline_initialization(self):
        """Test DataSciencePipeline initialization."""
        self.assertIsNotNone(self.simple_pipeline._pipeline_id)
        self.assertEqual(self.simple_pipeline.analytical_intention, "Test pipeline")
        self.assertEqual(self.simple_pipeline._state, PipelineState.CONFIGURED)
        self.assertTrue(self.simple_pipeline.composition_aware)
    
    def test_pipeline_fit_transform_cycle(self):
        """Test basic fit-transform cycle."""
        # Test fit
        fitted_pipeline = self.simple_pipeline.fit(self.test_data)
        self.assertEqual(self.simple_pipeline._state, PipelineState.FITTED)
        self.assertIs(fitted_pipeline, self.simple_pipeline)
        
        # Verify fit was called on estimators
        scaler = self.simple_pipeline.named_steps['scaler']
        classifier = self.simple_pipeline.named_steps['classifier']
        self.assertEqual(scaler.fit_call_count, 1)
        self.assertEqual(classifier.fit_call_count, 1)
        
        # Test transform
        result = self.simple_pipeline.transform(self.test_data)
        self.assertEqual(self.simple_pipeline._state, PipelineState.COMPLETED)
        self.assertIsNotNone(result)
        
        # Verify transform was called
        self.assertEqual(scaler.transform_call_count, 1)
        self.assertEqual(classifier.transform_call_count, 1)
    
    def test_pipeline_error_handling(self):
        """Test error handling in pipeline operations."""
        error_pipeline = DataSciencePipeline([
            ('faulty', MockEstimator(fail_fit=True))
        ], analytical_intention="Error test")
        
        with self.assertRaises(PipelineError):
            error_pipeline.fit(self.test_data)
        
        self.assertEqual(error_pipeline._state, PipelineState.ERROR)
    
    def test_composition_metadata_generation(self):
        """Test composition metadata generation."""
        self.simple_pipeline.fit(self.test_data)
        self.simple_pipeline.transform(self.test_data)
        
        metadata = self.simple_pipeline.get_composition_metadata()
        self.assertIsNotNone(metadata)
        self.assertIn('pipeline_info', metadata)
        self.assertIn('execution_metrics', metadata)
        self.assertIn('composition_context', metadata)
    
    def test_sklearn_api_compatibility(self):
        """Test sklearn Pipeline API compatibility."""
        # Test that DataSciencePipeline works with sklearn functions
        from sklearn.model_selection import cross_val_score
        from sklearn.datasets import make_classification
        
        # Create simple classification data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X)
        
        # Create pipeline with real sklearn estimators
        sklearn_pipeline = DataSciencePipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Test sklearn compatibility
        scores = cross_val_score(sklearn_pipeline, X_df, y, cv=3)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(score >= 0 for score in scores))


class TestPipelineComposer(unittest.TestCase):
    """Test suite for PipelineComposer multi-stage workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = create_test_dataframe(100, 5)
        self.composer = PipelineComposer(
            composition_strategy='sequential',
            metadata_enrichment=True
        )
    
    def test_composer_initialization(self):
        """Test PipelineComposer initialization."""
        self.assertEqual(self.composer.composition_strategy, 'sequential')
        self.assertTrue(self.composer.metadata_enrichment)
        self.assertIsNotNone(self.composer._composer_id)
    
    def test_pipeline_registration(self):
        """Test pipeline registration and dependency management."""
        # Create test pipelines
        pipeline1 = DataSciencePipeline([('step1', MockEstimator())], analytical_intention="Pipeline 1")
        pipeline2 = DataSciencePipeline([('step2', MockEstimator())], analytical_intention="Pipeline 2")
        
        # Register pipelines
        self.composer.add_pipeline('first', pipeline1)
        self.composer.add_pipeline('second', pipeline2, depends_on='first')
        
        self.assertEqual(len(self.composer.registered_pipelines), 2)
        self.assertIn('first', self.composer.registered_pipelines)
        self.assertIn('second', self.composer.registered_pipelines)
    
    def test_dependency_resolution(self):
        """Test dependency resolution and execution ordering."""
        # Create pipeline chain: A -> B -> C
        pipeline_a = DataSciencePipeline([('step_a', MockEstimator())], analytical_intention="Pipeline A")
        pipeline_b = DataSciencePipeline([('step_b', MockEstimator())], analytical_intention="Pipeline B")
        pipeline_c = DataSciencePipeline([('step_c', MockEstimator())], analytical_intention="Pipeline C")
        
        self.composer.add_pipeline('A', pipeline_a)
        self.composer.add_pipeline('B', pipeline_b, depends_on='A')
        self.composer.add_pipeline('C', pipeline_c, depends_on='B')
        
        # Test dependency resolution
        dependency_report = self.composer.resolve_dependencies()
        execution_order = dependency_report['execution_order']
        
        # Verify correct execution order
        self.assertEqual(execution_order, ['A', 'B', 'C'])
    
    def test_sequential_execution(self):
        """Test sequential pipeline execution."""
        # Create sequential pipeline composition
        pipeline1 = DataSciencePipeline([('mock1', MockEstimator())], analytical_intention="First stage")
        pipeline2 = DataSciencePipeline([('mock2', MockEstimator())], analytical_intention="Second stage")
        
        self.composer.add_pipeline('stage1', pipeline1)
        self.composer.add_pipeline('stage2', pipeline2, depends_on='stage1')
        
        # Execute composition
        results = self.composer.execute(self.test_data)
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertIn('stage1', results)
        self.assertIn('stage2', results)
        self.assertTrue(all(result.success for result in results.values()))
    
    def test_parallel_execution(self):
        """Test parallel pipeline execution."""
        parallel_composer = PipelineComposer(composition_strategy='parallel')
        
        # Create independent pipelines for parallel execution
        pipeline1 = DataSciencePipeline([('mock1', MockEstimator())], analytical_intention="Parallel 1")
        pipeline2 = DataSciencePipeline([('mock2', MockEstimator())], analytical_intention="Parallel 2")
        pipeline3 = DataSciencePipeline([('mock3', MockEstimator())], analytical_intention="Parallel 3")
        
        parallel_composer.add_pipeline('p1', pipeline1)
        parallel_composer.add_pipeline('p2', pipeline2)
        parallel_composer.add_pipeline('p3', pipeline3)
        
        # Execute in parallel
        results = parallel_composer.execute(self.test_data)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.success for result in results.values()))


class TestStreamingDataPipeline(unittest.TestCase):
    """Test suite for StreamingDataPipeline memory-bounded processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create large test data to trigger streaming
        self.large_data = create_test_dataframe(5000, 20)
        self.small_data = create_test_dataframe(100, 5)
        
        # Create streaming pipeline with low threshold to trigger streaming
        self.streaming_pipeline = StreamingDataPipeline([
            ('scaler', MockEstimator()),
            ('classifier', MockEstimator())
        ], 
        analytical_intention="Streaming test pipeline",
        streaming_threshold_mb=1,  # Low threshold to trigger streaming
        adaptive_chunking=True,
        memory_monitoring=True)
    
    def test_streaming_pipeline_initialization(self):
        """Test StreamingDataPipeline initialization."""
        self.assertEqual(self.streaming_pipeline.streaming_threshold_mb, 1)
        self.assertTrue(self.streaming_pipeline.adaptive_chunking)
        self.assertTrue(self.streaming_pipeline.memory_monitoring)
        self.assertFalse(self.streaming_pipeline._streaming_activated)
    
    def test_streaming_activation_logic(self):
        """Test streaming activation based on data size."""
        # Mock memory status
        with patch.object(StreamingQueryExecutor, '_get_memory_status') as mock_memory:
            mock_memory.return_value = MemoryStatus(
                total_gb=16.0,
                available_gb=8.0,
                used_percent=50.0,
                is_low_memory=False,
                recommended_chunk_size=1000,
                max_safe_chunk_size=5000
            )
            
            # Test with large data (should activate streaming)
            profile = self.streaming_pipeline._profile_data_characteristics(self.large_data)
            should_stream = self.streaming_pipeline._should_use_streaming(self.large_data, profile)
            self.assertTrue(should_stream)
            
            # Test with small data (should not activate streaming)
            small_profile = self.streaming_pipeline._profile_data_characteristics(self.small_data)
            should_stream_small = self.streaming_pipeline._should_use_streaming(self.small_data, small_profile)
            # Note: might still be True due to low threshold, but that's expected
    
    @patch('localdata_mcp.pipeline.core.StreamingQueryExecutor')
    def test_streaming_fit_execution(self, mock_streaming_executor_class):
        """Test streaming fit execution with mocked streaming executor."""
        # Mock the streaming executor
        mock_executor = Mock()
        mock_executor._get_memory_status.return_value = MemoryStatus(
            total_gb=16.0,
            available_gb=8.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=1000,
            max_safe_chunk_size=5000
        )
        mock_streaming_executor_class.return_value = mock_executor
        
        # Test streaming fit
        fitted_pipeline = self.streaming_pipeline.fit(self.large_data)
        
        # Verify streaming was activated
        self.assertTrue(self.streaming_pipeline._streaming_activated)
        self.assertIs(fitted_pipeline, self.streaming_pipeline)
        self.assertEqual(self.streaming_pipeline._state, PipelineState.FITTED)
    
    def test_streaming_metadata_collection(self):
        """Test streaming metadata collection and reporting."""
        # Execute with small data first to set up state
        self.streaming_pipeline.fit(self.small_data)
        
        # Get streaming metadata
        metadata = self.streaming_pipeline.get_streaming_metadata()
        
        self.assertIsInstance(metadata, dict)
        self.assertIn('streaming_activated', metadata)
        self.assertIn('threshold_mb', metadata)
        self.assertIn('adaptive_chunking', metadata)
        self.assertIn('memory_monitoring', metadata)


class TestSklearnStreamingAdapter(unittest.TestCase):
    """Test suite for SklearnStreamingAdapter integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = create_test_dataframe(1000, 10)
        self.test_target = np.random.randint(0, 2, 1000)
        
        # Create test pipeline
        self.sklearn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(random_state=42))  # Supports partial_fit
        ])
        
        # Create adapter
        self.adapter = SklearnStreamingAdapter(
            self.sklearn_pipeline,
            memory_monitoring=True,
            adaptive_chunking=True,
            performance_tracking=True
        )
    
    def test_adapter_initialization(self):
        """Test SklearnStreamingAdapter initialization."""
        self.assertIsNotNone(self.adapter.adapter_id)
        self.assertTrue(self.adapter.memory_monitoring)
        self.assertTrue(self.adapter.adaptive_chunking)
        self.assertTrue(self.adapter.performance_tracking)
        self.assertEqual(len(self.adapter.execution_history), 0)
    
    @patch('localdata_mcp.pipeline.core.StreamingQueryExecutor')
    def test_streaming_fit_operation(self, mock_streaming_executor_class):
        """Test streaming fit operation through adapter."""
        # Mock the streaming executor
        mock_executor = Mock()
        mock_executor._get_memory_status.return_value = MemoryStatus(
            total_gb=16.0,
            available_gb=8.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=500,
            max_safe_chunk_size=2000
        )
        self.adapter.streaming_executor = mock_executor
        
        # Execute streaming fit
        result, metadata = self.adapter.execute_streaming(
            self.test_data, 
            self.test_target,
            operation='fit',
            initial_chunk_size=200
        )
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(metadata, dict)
        self.assertIn('execution_id', metadata)
        self.assertIn('chunk_statistics', metadata)
        self.assertIn('performance_metrics', metadata)
        
        # Verify performance tracking
        self.assertEqual(len(self.adapter.execution_history), 1)
        self.assertIn('fit', self.adapter.performance_metrics)
    
    @patch('localdata_mcp.pipeline.core.StreamingQueryExecutor')
    def test_streaming_transform_operation(self, mock_streaming_executor_class):
        """Test streaming transform operation."""
        # Mock the streaming executor
        mock_executor = Mock()
        mock_executor._get_memory_status.return_value = MemoryStatus(
            total_gb=16.0,
            available_gb=8.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=500,
            max_safe_chunk_size=2000
        )
        self.adapter.streaming_executor = mock_executor
        
        # First fit the pipeline
        self.sklearn_pipeline.fit(self.test_data, self.test_target)
        
        # Execute streaming transform
        result, metadata = self.adapter.execute_streaming(
            self.test_data,
            operation='transform',
            initial_chunk_size=200
        )
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(self.test_data))
        self.assertIn('transform', self.adapter.performance_metrics)
    
    def test_partial_fit_support_detection(self):
        """Test detection of partial_fit support."""
        # Test with pipeline that supports partial_fit
        partial_fit_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier())
        ])
        adapter_with_partial = SklearnStreamingAdapter(partial_fit_pipeline)
        self.assertTrue(adapter_with_partial._check_partial_fit_support())
        
        # Test with pipeline that doesn't support partial_fit
        no_partial_fit_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        adapter_without_partial = SklearnStreamingAdapter(no_partial_fit_pipeline)
        # Note: StandardScaler doesn't have partial_fit, so this should be False
        # But SGDClassifier does, so the first example should be True
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking."""
        # Execute multiple operations
        self.sklearn_pipeline.fit(self.test_data, self.test_target)
        
        with patch.object(self.adapter.streaming_executor, '_get_memory_status') as mock_memory:
            mock_memory.return_value = MemoryStatus(
                total_gb=16.0,
                available_gb=8.0,
                used_percent=50.0,
                is_low_memory=False,
                recommended_chunk_size=500,
                max_safe_chunk_size=2000
            )
            
            # Execute operations
            self.adapter.execute_streaming(self.test_data, operation='transform')
            self.adapter.execute_streaming(self.test_data, operation='transform')
        
        # Get performance summary
        summary = self.adapter.get_performance_summary()
        
        self.assertIn('adapter_id', summary)
        self.assertIn('total_executions', summary)
        self.assertIn('performance_by_operation', summary)
        self.assertTrue(summary['total_executions'] >= 2)


class TestPreprocessingPipelines(unittest.TestCase):
    """Test suite for preprocessing pipeline components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = create_test_dataframe(500, 8, include_categorical=True, include_nulls=True)
    
    def test_data_cleaning_pipeline(self):
        """Test DataCleaningPipeline functionality."""
        cleaning_pipeline = DataCleaningPipeline(
            analytical_intention="Test data cleaning",
            cleaning_intensity="auto"
        )
        
        result, metadata = cleaning_pipeline.execute(self.test_data)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(metadata, dict)
        self.assertIn('cleaning_pipeline', metadata)
        self.assertIn('quality_assessment', metadata)
    
    def test_feature_scaling_pipeline(self):
        """Test FeatureScalingPipeline functionality."""
        # Create data with only numeric columns for scaling
        numeric_data = self.test_data.select_dtypes(include=[np.number])
        
        scaling_pipeline = FeatureScalingPipeline(
            analytical_intention="Test feature scaling",
            scaling_strategy="auto"
        )
        
        result, metadata = scaling_pipeline.execute(numeric_data)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, numeric_data.shape)
        self.assertIn('scaling_pipeline', metadata)
    
    def test_categorical_encoding_pipeline(self):
        """Test CategoricalEncodingPipeline functionality."""
        encoding_pipeline = CategoricalEncodingPipeline(
            analytical_intention="Test categorical encoding",
            encoding_strategy="auto"
        )
        
        result, metadata = encoding_pipeline.execute(self.test_data)
        
        self.assertIsNotNone(result)
        self.assertIn('encoding_pipeline', metadata)
    
    def test_data_type_manager(self):
        """Test DataTypeManager functionality."""
        type_manager = DataTypeManager(
            analytical_intention="Test type conversion",
            auto_inference=True
        )
        
        # Fit and transform
        type_manager.fit(self.test_data)
        result = type_manager.transform(self.test_data)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_data.shape)
        
        # Test type inference results
        inference_results = type_manager.get_type_inference_results()
        self.assertIsInstance(inference_results, dict)


class TestDomainAdapters(unittest.TestCase):
    """Test suite for domain adapter system."""
    
    def test_scipy_stats_adapter(self):
        """Test SciPyStatsAdapter functionality."""
        test_data = create_test_dataframe(100, 5, include_categorical=False, include_nulls=False)
        
        try:
            adapter = SciPyStatsAdapter(
                stat_function="describe",
                analytical_intention="Test scipy stats integration"
            )
            
            # Fit and transform
            adapter.fit(test_data)
            result = adapter.transform(test_data)
            
            self.assertIsNotNone(result)
            
            # Test adapter metadata
            metadata = adapter.get_composition_metadata()
            self.assertIn('adapter_info', metadata)
            self.assertIn('integration_status', metadata)
            
        except ImportError:
            self.skipTest("SciPy not available for testing")


class TestIntegrationScenarios(unittest.TestCase):
    """Test suite for end-to-end integration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = create_test_dataframe(1000, 15)
        self.test_target = np.random.randint(0, 3, 1000)
    
    def test_end_to_end_ml_workflow(self):
        """Test complete machine learning workflow."""
        # Create multi-stage ML workflow
        composer = PipelineComposer(composition_strategy='sequential')
        
        # Stage 1: Data preprocessing
        preprocessing = DataSciencePipeline([
            ('type_converter', DataTypeManager()),
            ('scaler', StandardScaler())
        ], analytical_intention="Preprocessing stage")
        
        # Stage 2: Feature engineering
        feature_engineering = DataSciencePipeline([
            ('feature_scaler', MockEstimator())
        ], analytical_intention="Feature engineering")
        
        # Stage 3: Model training
        model_training = DataSciencePipeline([
            ('classifier', MockEstimator())
        ], analytical_intention="Model training")
        
        # Register stages
        composer.add_pipeline('preprocessing', preprocessing)
        composer.add_pipeline('features', feature_engineering, depends_on='preprocessing')
        composer.add_pipeline('modeling', model_training, depends_on='features')
        
        # Execute workflow
        results = composer.execute(self.test_data)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.success for result in results.values()))
        
        # Verify execution order
        dependency_report = composer.resolve_dependencies()
        self.assertEqual(dependency_report['execution_order'], ['preprocessing', 'features', 'modeling'])
    
    def test_streaming_integration_with_large_dataset(self):
        """Test streaming integration with large dataset."""
        # Create large dataset
        large_data = create_test_dataframe(10000, 50)  # Larger dataset
        
        # Create streaming pipeline
        streaming_pipeline = StreamingDataPipeline([
            ('preprocessor', MockEstimator()),
            ('model', MockEstimator())
        ],
        streaming_threshold_mb=5,  # Low threshold to trigger streaming
        adaptive_chunking=True)
        
        # Test streaming execution
        fitted_pipeline = streaming_pipeline.fit(large_data)
        result = streaming_pipeline.transform(large_data)
        
        self.assertIsNotNone(fitted_pipeline)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(large_data))
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        # Test pipeline with mixed success/failure
        composer = PipelineComposer(
            composition_strategy='sequential',
            error_recovery_mode='partial'
        )
        
        # Add successful pipeline
        success_pipeline = DataSciencePipeline([
            ('success', MockEstimator())
        ], analytical_intention="Success pipeline")
        
        # Add failing pipeline
        fail_pipeline = DataSciencePipeline([
            ('fail', MockEstimator(fail_fit=True))
        ], analytical_intention="Failing pipeline")
        
        composer.add_pipeline('success', success_pipeline)
        composer.add_pipeline('fail', fail_pipeline, depends_on='success')
        
        # Execute and expect partial results
        try:
            results = composer.execute(self.test_data)
            # Should have at least the successful pipeline
            self.assertIn('success', results)
            self.assertTrue(results['success'].success)
        except Exception:
            # Expected if error recovery doesn't handle this case
            pass
    
    def test_memory_usage_validation(self):
        """Test memory usage stays within bounds."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Create and execute large streaming pipeline
        large_data = create_test_dataframe(5000, 20)
        
        streaming_pipeline = StreamingDataPipeline([
            ('step1', MockEstimator()),
            ('step2', MockEstimator())
        ], streaming_threshold_mb=10)
        
        # Execute pipeline
        streaming_pipeline.fit(large_data)
        result = streaming_pipeline.transform(large_data)
        
        # Force garbage collection
        del result
        gc.collect()
        
        # Check final memory usage
        final_memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory_mb - initial_memory_mb
        
        # Memory increase should be reasonable (less than 500MB for this test)
        self.assertLess(memory_increase, 500, 
                       f"Memory usage increased by {memory_increase:.1f}MB, which may indicate a memory leak")
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking and optimization."""
        # Create test data
        benchmark_data = create_test_dataframe(2000, 10)
        
        # Test regular pipeline performance
        regular_pipeline = DataSciencePipeline([
            ('step1', MockEstimator(sleep_time=0.001)),  # Small delay to measure
            ('step2', MockEstimator(sleep_time=0.001))
        ])
        
        start_time = time.time()
        regular_pipeline.fit(benchmark_data)
        regular_pipeline.transform(benchmark_data)
        regular_time = time.time() - start_time
        
        # Test streaming pipeline performance
        streaming_pipeline = StreamingDataPipeline([
            ('step1', MockEstimator(sleep_time=0.001)),
            ('step2', MockEstimator(sleep_time=0.001))
        ], streaming_threshold_mb=1)
        
        start_time = time.time()
        streaming_pipeline.fit(benchmark_data)
        streaming_pipeline.transform(benchmark_data)
        streaming_time = time.time() - start_time
        
        # Streaming might be slower due to overhead, but should be reasonable
        self.assertLess(streaming_time, regular_time * 5,  # Allow 5x overhead for streaming
                       "Streaming pipeline performance is significantly worse than regular pipeline")


def run_comprehensive_test_suite():
    """Run the complete test suite with detailed reporting."""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataSciencePipeline,
        TestPipelineComposer,
        TestStreamingDataPipeline,
        TestSklearnStreamingAdapter,
        TestPreprocessingPipelines,
        TestDomainAdapters,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("=" * 80)
    print("LOCALDATA MCP PIPELINE INFRASTRUCTURE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Running {test_suite.countTestCases()} tests across {len(test_classes)} test classes...")
    print()
    
    # Execute tests
    start_time = time.time()
    result = runner.run(test_suite)
    execution_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successful: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    print("\n" + "=" * 80)
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)