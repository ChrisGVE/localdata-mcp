"""
Unit tests for StreamingDataPipeline class.

This module provides comprehensive testing for the StreamingDataPipeline class,
validating streaming capabilities, memory management, and sklearn compatibility.

Test Categories:
- Streaming activation and threshold detection
- Memory-bounded chunk processing
- Adaptive chunk sizing
- Integration with StreamingQueryExecutor
- sklearn Pipeline API compatibility under streaming
- Error handling and recovery during streaming
- Performance validation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Any, Dict

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDClassifier, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

from src.localdata_mcp.pipeline.core import StreamingDataPipeline, DataFrameStreamingSource
from src.localdata_mcp.pipeline.base import (
    PipelineState, 
    PipelineError, 
    CompositionMetadata,
    StreamingConfig,
    ErrorClassification
)
from src.localdata_mcp.streaming_executor import MemoryStatus, StreamingQueryExecutor


class TestDataFrameStreamingSource:
    """Test DataFrameStreamingSource functionality."""
    
    def test_dataframe_streaming_source_initialization(self):
        """Test DataFrameStreamingSource initialization."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10]})
        source = DataFrameStreamingSource(df)
        
        assert source.dataframe.equals(df)
        assert source.estimate_total_rows() == 5
        assert source.estimate_memory_per_row() > 0
    
    def test_chunk_iterator(self):
        """Test chunk iterator functionality."""
        df = pd.DataFrame({'x': range(10), 'y': range(10, 20)})
        source = DataFrameStreamingSource(df)
        
        chunks = list(source.get_chunk_iterator(chunk_size=3))
        
        assert len(chunks) == 4  # 10 rows / 3 = 3 full chunks + 1 partial
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3
        assert len(chunks[3]) == 1  # Partial chunk
        
        # Verify data integrity
        reconstructed = pd.concat(chunks, ignore_index=True)
        pd.testing.assert_frame_equal(reconstructed, df)
    
    def test_memory_estimation(self):
        """Test memory per row estimation."""
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.random(1000),
            'str_col': ['test_string'] * 1000
        })
        source = DataFrameStreamingSource(df)
        
        memory_per_row = source.estimate_memory_per_row()
        assert memory_per_row > 0
        assert isinstance(memory_per_row, float)
        
        # Should be reasonable estimate (not too small or huge)
        assert 100 < memory_per_row < 10000  # Between 100 bytes and 10KB per row


class TestStreamingDataPipelineInitialization:
    """Test StreamingDataPipeline initialization and configuration."""
    
    def test_basic_initialization(self):
        """Test basic StreamingDataPipeline initialization."""
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        assert pipeline.state == PipelineState.INITIALIZED
        assert len(pipeline.steps) == 2
        assert pipeline.streaming_threshold_mb == 1024  # Default threshold
        assert pipeline.adaptive_chunking is True
        assert pipeline.memory_monitoring is True
        assert pipeline._streaming_activated is False
        assert isinstance(pipeline.streaming_config, StreamingConfig)
    
    def test_custom_streaming_configuration(self):
        """Test initialization with custom streaming parameters."""
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ], 
        streaming_threshold_mb=512,
        adaptive_chunking=False,
        memory_monitoring=False)
        
        assert pipeline.streaming_threshold_mb == 512
        assert pipeline.adaptive_chunking is False
        assert pipeline.memory_monitoring is False
        assert pipeline.streaming_config.threshold_mb == 512
        assert pipeline.streaming_config.memory_limit_mb == 1024  # 2x threshold
    
    def test_explicit_streaming_config(self):
        """Test initialization with explicit StreamingConfig."""
        streaming_config = StreamingConfig(
            enabled=True,
            threshold_mb=256,
            chunk_size_adaptive=True,
            initial_chunk_size=500,
            memory_limit_mb=2048
        )
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(random_state=42))
        ], streaming_config=streaming_config)
        
        assert pipeline.streaming_config.enabled is True
        assert pipeline.streaming_config.threshold_mb == 256
        assert pipeline.streaming_config.initial_chunk_size == 500
        assert pipeline.streaming_config.memory_limit_mb == 2048


class TestStreamingActivationLogic:
    """Test streaming activation and threshold detection."""
    
    def test_streaming_not_activated_small_dataset(self):
        """Test that streaming is not activated for small datasets."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=5, random_state=42))
        ], streaming_threshold_mb=1000)  # High threshold
        
        with patch('src.localdata_mcp.pipeline.core.get_logging_manager') as mock_logging:
            mock_logging_instance = Mock()
            mock_logging_instance.log_query_start.return_value = "test_request_id"
            mock_logging_instance.context.return_value.__enter__ = Mock()
            mock_logging_instance.context.return_value.__exit__ = Mock()
            mock_logging.return_value = mock_logging_instance
            
            pipeline.fit(X_df, y)
        
        assert pipeline._streaming_activated is False
        assert pipeline._streaming_executor is None
    
    def test_streaming_activated_by_size_threshold(self):
        """Test streaming activation based on data size threshold."""
        # Create dataset that exceeds threshold
        X = pd.DataFrame(np.random.random((500, 20)))  # Larger dataset
        y = np.random.randint(0, 2, 500)
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(random_state=42))  # Use SGD for partial_fit support
        ], streaming_threshold_mb=0.01)  # Very low threshold to trigger streaming
        
        with patch('src.localdata_mcp.pipeline.core.get_logging_manager') as mock_logging:
            mock_logging_instance = Mock()
            mock_logging_instance.log_query_start.return_value = "test_request_id"
            mock_logging_instance.context.return_value.__enter__ = Mock()
            mock_logging_instance.context.return_value.__exit__ = Mock()
            mock_logging.return_value = mock_logging_instance
            
            pipeline.fit(X, y)
        
        assert pipeline._streaming_activated is True
        assert pipeline._streaming_executor is not None
    
    def test_streaming_activated_by_row_count(self):
        """Test streaming activation based on row count threshold."""
        # Create dataset with many rows but small memory footprint
        X = pd.DataFrame({'feature': range(150000)})  # >100K rows
        y = np.random.randint(0, 2, 150000)
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(random_state=42))
        ], streaming_threshold_mb=1000)  # High memory threshold
        
        with patch('src.localdata_mcp.pipeline.core.get_logging_manager') as mock_logging:
            mock_logging_instance = Mock()
            mock_logging_instance.log_query_start.return_value = "test_request_id" 
            mock_logging_instance.context.return_value.__enter__ = Mock()
            mock_logging_instance.context.return_value.__exit__ = Mock()
            mock_logging.return_value = mock_logging_instance
            
            pipeline.fit(X, y)
        
        assert pipeline._streaming_activated is True
    
    def test_streaming_activated_by_memory_pressure(self):
        """Test streaming activation due to memory pressure."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(random_state=42))
        ], streaming_threshold_mb=1000, memory_monitoring=True)
        
        # Mock high memory usage
        mock_memory_status = MemoryStatus(
            total_gb=8.0,
            available_gb=0.5,  # Low available memory
            used_percent=95.0,  # High usage
            is_low_memory=True,
            recommended_chunk_size=100,
            max_safe_chunk_size=200
        )
        
        with patch('src.localdata_mcp.pipeline.core.get_logging_manager') as mock_logging:
            mock_logging_instance = Mock()
            mock_logging_instance.log_query_start.return_value = "test_request_id"
            mock_logging_instance.context.return_value.__enter__ = Mock()
            mock_logging_instance.context.return_value.__exit__ = Mock()
            mock_logging.return_value = mock_logging_instance
            
            # Mock StreamingQueryExecutor to return low memory status
            with patch('src.localdata_mcp.pipeline.core.StreamingQueryExecutor') as mock_executor_class:
                mock_executor = Mock()
                mock_executor._get_memory_status.return_value = mock_memory_status
                mock_executor_class.return_value = mock_executor
                
                pipeline.fit(X_df, y)
        
        assert pipeline._streaming_activated is True


class TestStreamingPipelineExecution:
    """Test streaming pipeline execution and chunk processing."""
    
    @patch('src.localdata_mcp.pipeline.core.get_logging_manager')
    def test_streaming_fit_execution(self, mock_get_logging):
        """Test streaming fit with chunk processing."""
        mock_logging = Mock()
        mock_logging.log_query_start.return_value = "test_request_123"
        mock_logging.context.return_value.__enter__ = Mock()
        mock_logging.context.return_value.__exit__ = Mock()
        mock_get_logging.return_value = mock_logging
        
        # Create dataset that will trigger streaming
        X = pd.DataFrame(np.random.random((1000, 10)))
        y = np.random.randint(0, 2, 1000)
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(random_state=42))  # Supports partial_fit
        ], streaming_threshold_mb=0.01)  # Force streaming
        
        # Mock memory status
        mock_memory_status = MemoryStatus(
            total_gb=8.0,
            available_gb=4.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=250,
            max_safe_chunk_size=500
        )
        
        with patch('src.localdata_mcp.pipeline.core.StreamingQueryExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor._get_memory_status.return_value = mock_memory_status
            mock_executor_class.return_value = mock_executor
            
            result = pipeline.fit(X, y)
        
        assert result is pipeline
        assert pipeline.state == PipelineState.FITTED
        assert pipeline._streaming_activated is True
        assert len(pipeline._chunk_fit_results) > 0
    
    @patch('src.localdata_mcp.pipeline.core.get_logging_manager')
    def test_streaming_transform_execution(self, mock_get_logging):
        """Test streaming transform with chunk processing."""
        mock_logging = Mock()
        mock_logging.log_query_start.return_value = "test_request_123"
        mock_logging.context.return_value.__enter__ = Mock()
        mock_logging.context.return_value.__exit__ = Mock()
        mock_get_logging.return_value = mock_logging
        
        # Create dataset and fit first
        X_train, y_train = make_classification(n_samples=500, n_features=8, random_state=42)
        X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(8)])
        
        X_test = pd.DataFrame(np.random.random((800, 8)), columns=[f'feature_{i}' for i in range(8)])
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(random_state=42))
        ], streaming_threshold_mb=0.01)  # Force streaming
        
        mock_memory_status = MemoryStatus(
            total_gb=8.0,
            available_gb=4.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=200,
            max_safe_chunk_size=400
        )
        
        with patch('src.localdata_mcp.pipeline.core.StreamingQueryExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor._get_memory_status.return_value = mock_memory_status
            mock_executor_class.return_value = mock_executor
            
            # Fit first
            pipeline.fit(X_train_df, y_train)
            
            # Then transform
            result = pipeline.transform(X_test)
        
        assert len(result) == len(X_test)
        assert len(pipeline._chunk_transform_results) > 0
        assert pipeline.state == PipelineState.COMPLETED
    
    def test_chunk_aggregation_dataframe(self):
        """Test chunk result aggregation for DataFrames."""
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler())
        ])
        
        # Mock chunk results
        chunk1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        chunk2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        chunk3 = pd.DataFrame({'a': [9, 10], 'b': [11, 12]})
        
        pipeline._chunk_transform_results = [chunk1, chunk2, chunk3]
        
        result = pipeline._aggregate_chunk_transform_results()
        
        expected = pd.DataFrame({'a': [1, 2, 5, 6, 9, 10], 'b': [3, 4, 7, 8, 11, 12]})
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
    
    def test_chunk_aggregation_numpy_array(self):
        """Test chunk result aggregation for numpy arrays."""
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler())
        ])
        
        # Mock chunk results as numpy arrays
        chunk1 = np.array([[1, 2], [3, 4]])
        chunk2 = np.array([[5, 6], [7, 8]])
        chunk3 = np.array([[9, 10]])
        
        pipeline._chunk_transform_results = [chunk1, chunk2, chunk3]
        
        result = pipeline._aggregate_chunk_transform_results()
        
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        np.testing.assert_array_equal(result, expected)


class TestAdaptiveChunkSizing:
    """Test adaptive chunk sizing functionality."""
    
    def test_adaptive_chunk_size_calculation(self):
        """Test adaptive chunk size calculation."""
        df = pd.DataFrame(np.random.random((1000, 20)))
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler())
        ], adaptive_chunking=True)
        
        # Mock memory status
        memory_status = MemoryStatus(
            total_gb=8.0,
            available_gb=4.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=300,
            max_safe_chunk_size=600
        )
        
        chunk_size = pipeline._calculate_adaptive_chunk_size(df, memory_status)
        
        assert isinstance(chunk_size, int)
        assert 100 <= chunk_size <= 10000  # Within reasonable bounds
    
    def test_adaptive_chunk_size_low_memory(self):
        """Test adaptive chunk size under memory pressure."""
        df = pd.DataFrame(np.random.random((1000, 10)))
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler())
        ], adaptive_chunking=True)
        
        # Mock low memory status
        low_memory_status = MemoryStatus(
            total_gb=8.0,
            available_gb=0.5,
            used_percent=95.0,
            is_low_memory=True,
            recommended_chunk_size=50,
            max_safe_chunk_size=100
        )
        
        chunk_size = pipeline._calculate_adaptive_chunk_size(df, low_memory_status)
        
        assert chunk_size == 50  # Should use recommended size for low memory
    
    def test_dynamic_chunk_size_adjustment(self):
        """Test dynamic chunk size adjustment during processing."""
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler())
        ], adaptive_chunking=True)
        
        memory_status = MemoryStatus(
            total_gb=8.0,
            available_gb=4.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=500,
            max_safe_chunk_size=1000
        )
        
        # Test reducing chunk size for slow processing
        new_size = pipeline._adaptive_chunk_sizing(
            current_chunk_size=1000,
            processing_time=6.0,  # Slow processing
            memory_status=memory_status
        )
        assert new_size < 1000  # Should reduce
        
        # Test increasing chunk size for fast processing
        new_size = pipeline._adaptive_chunk_sizing(
            current_chunk_size=500,
            processing_time=0.5,  # Fast processing
            memory_status=memory_status
        )
        assert new_size > 500  # Should increase
        
        # Test memory pressure override
        low_memory_status = MemoryStatus(
            total_gb=8.0,
            available_gb=0.3,
            used_percent=96.0,
            is_low_memory=True,
            recommended_chunk_size=100,
            max_safe_chunk_size=200
        )
        
        new_size = pipeline._adaptive_chunk_sizing(
            current_chunk_size=1000,
            processing_time=0.5,  # Fast processing
            memory_status=low_memory_status
        )
        assert new_size <= 100  # Should respect memory constraints


class TestStreamingErrorHandling:
    """Test error handling during streaming operations."""
    
    @patch('src.localdata_mcp.pipeline.core.get_logging_manager')
    def test_streaming_fit_error_handling(self, mock_get_logging):
        """Test error handling during streaming fit."""
        mock_logging = Mock()
        mock_logging.log_query_start.return_value = "test_request_123"
        mock_logging.context.return_value.__enter__ = Mock()
        mock_logging.context.return_value.__exit__ = Mock()
        mock_get_logging.return_value = mock_logging
        
        X = pd.DataFrame(np.random.random((500, 5)))
        y = np.random.randint(0, 2, 500)
        
        # Create pipeline with failing estimator
        mock_classifier = Mock()
        mock_classifier.fit.side_effect = ValueError("Streaming fit failed")
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', mock_classifier)
        ], streaming_threshold_mb=0.01)  # Force streaming
        
        mock_memory_status = MemoryStatus(
            total_gb=8.0,
            available_gb=4.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=200,
            max_safe_chunk_size=400
        )
        
        with patch('src.localdata_mcp.pipeline.core.StreamingQueryExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor._get_memory_status.return_value = mock_memory_status
            mock_executor_class.return_value = mock_executor
            
            with pytest.raises(PipelineError) as excinfo:
                pipeline.fit(X, y)
            
            # Verify error details
            error = excinfo.value
            assert "StreamingDataPipeline fit failed" in str(error)
            assert error.pipeline_stage == "streaming_fit"
            assert len(error.recovery_suggestions) > 0
            assert pipeline.state == PipelineState.ERROR
    
    @patch('src.localdata_mcp.pipeline.core.get_logging_manager')
    def test_streaming_transform_error_handling(self, mock_get_logging):
        """Test error handling during streaming transform."""
        mock_logging = Mock()
        mock_logging.log_query_start.return_value = "test_request_123"
        mock_logging.context.return_value.__enter__ = Mock()
        mock_logging.context.return_value.__exit__ = Mock()
        mock_get_logging.return_value = mock_logging
        
        X_train, y_train = make_classification(n_samples=200, n_features=5, random_state=42)
        X_test = pd.DataFrame(np.random.random((300, 5)))
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(random_state=42))
        ], streaming_threshold_mb=0.01)  # Force streaming
        
        mock_memory_status = MemoryStatus(
            total_gb=8.0,
            available_gb=4.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=100,
            max_safe_chunk_size=200
        )
        
        with patch('src.localdata_mcp.pipeline.core.StreamingQueryExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor._get_memory_status.return_value = mock_memory_status
            mock_executor_class.return_value = mock_executor
            
            # Fit successfully
            pipeline.fit(X_train, y_train)
            
            # Mock transform failure
            with patch.object(pipeline, '_transform_chunk', side_effect=RuntimeError("Transform chunk failed")):
                with pytest.raises(PipelineError) as excinfo:
                    pipeline.transform(X_test)
                
                error = excinfo.value
                assert "StreamingDataPipeline transform failed" in str(error)
                assert error.pipeline_stage == "streaming_transform"


class TestStreamingMetadata:
    """Test streaming metadata and monitoring functionality."""
    
    def test_streaming_metadata_generation(self):
        """Test streaming metadata generation."""
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler())
        ], streaming_threshold_mb=100, adaptive_chunking=True, memory_monitoring=True)
        
        # Mock some streaming execution state
        pipeline._streaming_activated = True
        pipeline._chunk_fit_results = [{'chunk': 1}, {'chunk': 2}]
        pipeline._chunk_transform_results = [pd.DataFrame({'a': [1, 2]}), pd.DataFrame({'a': [3, 4]})]
        pipeline._memory_snapshots = [
            MemoryStatus(8.0, 4.0, 50.0, False, 500, 1000),
            MemoryStatus(8.0, 3.5, 55.0, False, 450, 900)
        ]
        pipeline._adaptive_chunk_size = 750
        
        metadata = pipeline.get_streaming_metadata()
        
        assert metadata['streaming_activated'] is True
        assert metadata['threshold_mb'] == 100
        assert metadata['adaptive_chunking'] is True
        assert metadata['memory_monitoring'] is True
        assert metadata['chunk_fit_results'] == 2
        assert metadata['chunk_transform_results'] == 2
        assert metadata['memory_snapshots'] == 2
        assert metadata['final_chunk_size'] == 750
        assert 'memory_efficiency' in metadata
        assert metadata['memory_efficiency']['initial_memory_gb'] == 4.0
        assert metadata['memory_efficiency']['final_memory_gb'] == 3.5
    
    def test_streaming_cache_clearing(self):
        """Test streaming cache clearing functionality."""
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler())
        ])
        
        # Populate streaming caches
        pipeline._chunk_fit_results = [{'chunk': 1}, {'chunk': 2}]
        pipeline._chunk_transform_results = [pd.DataFrame({'a': [1]})]
        pipeline._memory_snapshots = [MemoryStatus(8.0, 4.0, 50.0, False, 500, 1000)]
        
        # Mock streaming executor with buffers
        mock_executor = Mock()
        mock_executor._result_buffers = {'query1': Mock(), 'query2': Mock()}
        pipeline._streaming_executor = mock_executor
        
        pipeline.clear_streaming_cache()
        
        assert len(pipeline._chunk_fit_results) == 0
        assert len(pipeline._chunk_transform_results) == 0
        assert len(pipeline._memory_snapshots) == 0
        
        # Verify executor buffers were cleared
        mock_executor.clear_buffer.assert_called()


class TestStreamingPipelineIntegration:
    """Test integration with existing sklearn workflows and LocalData MCP systems."""
    
    @patch('src.localdata_mcp.pipeline.core.get_logging_manager')
    def test_sklearn_api_compatibility_with_streaming(self, mock_get_logging):
        """Test that streaming pipeline maintains sklearn API compatibility."""
        mock_logging = Mock()
        mock_logging.log_query_start.return_value = "test_request_123"
        mock_logging.context.return_value.__enter__ = Mock()
        mock_logging.context.return_value.__exit__ = Mock()
        mock_get_logging.return_value = mock_logging
        
        X, y = make_classification(n_samples=300, n_features=6, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(6)])
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(random_state=42))
        ], streaming_threshold_mb=0.01)  # Force streaming
        
        mock_memory_status = MemoryStatus(
            total_gb=8.0,
            available_gb=4.0,
            used_percent=50.0,
            is_low_memory=False,
            recommended_chunk_size=150,
            max_safe_chunk_size=300
        )
        
        with patch('src.localdata_mcp.pipeline.core.StreamingQueryExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor._get_memory_status.return_value = mock_memory_status
            mock_executor_class.return_value = mock_executor
            
            # Test sklearn methods work with streaming
            pipeline.fit(X_df, y)
            predictions = pipeline.predict(X_df)
            transformed = pipeline.transform(X_df)
            
            assert len(predictions) == len(X_df)
            assert len(transformed) == len(X_df)
            
            # Test sklearn properties are accessible
            assert hasattr(pipeline, 'named_steps')
            assert 'scaler' in pipeline.named_steps
            assert 'classifier' in pipeline.named_steps
    
    def test_composition_metadata_with_streaming(self):
        """Test composition metadata generation with streaming active."""
        X, y = make_regression(n_samples=100, n_features=4, random_state=42)
        X_df = pd.DataFrame(X, columns=['price', 'size', 'location', 'age'])
        
        pipeline = StreamingDataPipeline(
            steps=[('scaler', StandardScaler()), ('regressor', SGDRegressor(random_state=42))],
            analytical_intention="Predict housing prices with streaming processing",
            composition_aware=True,
            streaming_threshold_mb=0.01  # Force streaming
        )
        
        with patch('src.localdata_mcp.pipeline.core.get_logging_manager') as mock_logging:
            mock_logging_instance = Mock()
            mock_logging_instance.log_query_start.return_value = "test_request_id"
            mock_logging_instance.context.return_value.__enter__ = Mock()
            mock_logging_instance.context.return_value.__exit__ = Mock()
            mock_logging.return_value = mock_logging_instance
            
            mock_memory_status = MemoryStatus(8.0, 4.0, 50.0, False, 200, 400)
            
            with patch('src.localdata_mcp.pipeline.core.StreamingQueryExecutor') as mock_executor_class:
                mock_executor = Mock()
                mock_executor._get_memory_status.return_value = mock_memory_status
                mock_executor_class.return_value = mock_executor
                
                pipeline.fit(X_df, y)
                pipeline.transform(X_df)
        
        metadata = pipeline.composition_metadata
        assert metadata is not None
        assert metadata.domain == "regression"
        assert metadata.transformation_summary["streaming_enabled"] is True
        
        # Verify streaming metadata is included
        streaming_metadata = pipeline.get_streaming_metadata()
        assert streaming_metadata['streaming_activated'] is True


class TestStreamingPerformanceValidation:
    """Test streaming pipeline performance characteristics."""
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large dataset processing."""
        # Note: This would be a performance test with actual large data in production
        # Here we simulate the scenario with mocks
        
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(random_state=42))
        ], streaming_threshold_mb=100)
        
        # Simulate processing large dataset
        mock_memory_snapshots = [
            MemoryStatus(8.0, 4.0, 50.0, False, 500, 1000),  # Initial
            MemoryStatus(8.0, 3.8, 52.5, False, 480, 960),   # Processing
            MemoryStatus(8.0, 3.9, 51.2, False, 490, 980)    # Final
        ]
        
        pipeline._memory_snapshots = mock_memory_snapshots
        
        metadata = pipeline.get_streaming_metadata()
        memory_efficiency = metadata['memory_efficiency']
        
        # Should maintain reasonable memory usage
        assert memory_efficiency['peak_memory_usage_percent'] < 60  # Less than 60% peak
        assert memory_efficiency['final_memory_gb'] >= 3.5  # Still have available memory
    
    def test_chunk_processing_performance(self):
        """Test chunk processing performance metrics."""
        pipeline = StreamingDataPipeline([
            ('scaler', StandardScaler())
        ])
        
        # Mock successful chunk fit results with timing
        pipeline._chunk_fit_results = [
            {'chunk_number': 1, 'processing_time': 0.1, 'samples_processed': 100, 'success': True},
            {'chunk_number': 2, 'processing_time': 0.12, 'samples_processed': 100, 'success': True},
            {'chunk_number': 3, 'processing_time': 0.08, 'samples_processed': 80, 'success': True}
        ]
        
        pipeline._aggregate_chunk_fit_results()
        
        aggregation_metadata = pipeline._execution_context['fit_aggregation']
        
        assert aggregation_metadata['successful_chunks'] == 3
        assert aggregation_metadata['failed_chunks'] == 0
        assert aggregation_metadata['total_samples'] == 280
        assert aggregation_metadata['total_processing_time'] == 0.3
        assert aggregation_metadata['average_chunk_time'] == 0.1


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])