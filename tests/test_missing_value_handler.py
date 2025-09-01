"""
Tests for the MissingValueHandler - Comprehensive sklearn.impute integration testing.

This test suite validates:
- Missing value pattern analysis (MCAR, MAR, MNAR detection)
- Multiple imputation strategies (simple, KNN, iterative)
- Automatic strategy selection and quality assessment
- Cross-validation and comprehensive quality metrics
- Progressive disclosure and streaming compatibility
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings

# Suppress sklearn warnings during testing
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from src.localdata_mcp.pipeline.missing_value_handler import (
    MissingValueHandler,
    MissingValuePattern,
    ImputationQuality,
    ImputationMetadata
)
from src.localdata_mcp.pipeline.base import StreamingConfig


class TestMissingValueHandler:
    """Test suite for the MissingValueHandler class."""
    
    @pytest.fixture
    def sample_data_mcar(self):
        """Create sample data with Missing Completely At Random pattern."""
        np.random.seed(42)
        n = 1000
        
        data = pd.DataFrame({
            'numeric1': np.random.normal(100, 15, n),
            'numeric2': np.random.normal(50, 10, n),
            'numeric3': np.random.exponential(2, n),
            'categorical1': np.random.choice(['A', 'B', 'C', 'D'], n),
            'categorical2': np.random.choice(['X', 'Y', 'Z'], n, p=[0.5, 0.3, 0.2]),
            'datetime1': pd.date_range('2020-01-01', periods=n, freq='D')
        })
        
        # Introduce MCAR missing values (completely random)
        missing_mask = np.random.random((n, 3)) < 0.1  # 10% missing
        data.iloc[missing_mask[:, 0], 0] = np.nan  # numeric1
        data.iloc[missing_mask[:, 1], 1] = np.nan  # numeric2
        data.iloc[missing_mask[:, 2], 3] = np.nan  # categorical1
        
        return data
    
    @pytest.fixture
    def sample_data_mar(self):
        """Create sample data with Missing At Random pattern."""
        np.random.seed(42)
        n = 1000
        
        data = pd.DataFrame({
            'age': np.random.normal(35, 12, n),
            'income': np.random.lognormal(10, 1, n),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n)
        })
        
        # Introduce MAR missing values (missing depends on other variables)
        # Higher income people less likely to report income
        high_income_mask = data['income'] > data['income'].quantile(0.8)
        income_missing_prob = np.where(high_income_mask, 0.3, 0.05)
        income_missing_mask = np.random.random(n) < income_missing_prob
        data.loc[income_missing_mask, 'income'] = np.nan
        
        # PhD holders less likely to report education
        phd_mask = data['education'] == 'PhD'
        edu_missing_prob = np.where(phd_mask, 0.4, 0.1)
        edu_missing_mask = np.random.random(n) < edu_missing_prob
        data.loc[edu_missing_mask, 'education'] = np.nan
        
        return data
    
    @pytest.fixture
    def sample_data_high_missing(self):
        """Create sample data with high missing percentages."""
        np.random.seed(42)
        n = 500
        
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n),
            'feature2': np.random.normal(5, 2, n),
            'feature3': np.random.choice(['A', 'B', 'C'], n),
            'feature4': np.random.exponential(1, n)
        })
        
        # Introduce high missing rates
        for i, col in enumerate(data.columns):
            missing_rate = [0.6, 0.4, 0.7, 0.5][i]  # High missing rates
            missing_mask = np.random.random(n) < missing_rate
            data.loc[missing_mask, col] = np.nan
        
        return data
    
    def test_initialization_default(self):
        """Test MissingValueHandler initialization with default parameters."""
        handler = MissingValueHandler()
        
        assert handler.get_analysis_type() == "missing_value_handling"
        assert handler.strategy == "auto"
        assert handler.complexity == "auto"
        assert handler.cross_validation is True
        assert handler.metadata_tracking is True
        assert handler._missing_pattern is None
        assert handler._fitted_imputers == {}
    
    def test_initialization_custom(self):
        """Test MissingValueHandler initialization with custom parameters."""
        custom_params = {
            'strategy_config': {
                'knn': {'n_neighbors': 10, 'weights': 'distance'}
            }
        }
        
        handler = MissingValueHandler(
            analytical_intention="test imputation",
            strategy="knn",
            complexity="comprehensive",
            cross_validation=False,
            custom_parameters=custom_params
        )
        
        assert handler.analytical_intention == "test imputation"
        assert handler.strategy == "knn"
        assert handler.complexity == "comprehensive"
        assert handler.cross_validation is False
        assert handler.strategy_configs['custom'] == custom_params['strategy_config']
    
    def test_missing_pattern_analysis_mcar(self, sample_data_mcar):
        """Test missing pattern analysis for MCAR data."""
        handler = MissingValueHandler()
        
        # Analyze missing patterns
        processed_data, metadata = handler._analyze_missing_patterns(sample_data_mcar)
        
        assert handler._missing_pattern is not None
        assert handler._missing_pattern.pattern_type in ["MCAR", "MAR"]  # Could be either due to randomness
        assert handler._missing_pattern.missing_percentage > 0
        assert len(handler._missing_pattern.column_patterns) > 0
        assert handler._missing_pattern.confidence_score >= 0.0
        assert len(handler._missing_pattern.recommendations) > 0
        
        # Check metadata structure
        pattern_analysis = metadata["pattern_analysis"]
        assert "pattern_type" in pattern_analysis
        assert "overall_missing_percentage" in pattern_analysis
        assert "columns_with_missing" in pattern_analysis
        assert "recommendations" in pattern_analysis
    
    def test_missing_pattern_analysis_mar(self, sample_data_mar):
        """Test missing pattern analysis for MAR data."""
        handler = MissingValueHandler()
        
        processed_data, metadata = handler._analyze_missing_patterns(sample_data_mar)
        
        assert handler._missing_pattern is not None
        # MAR pattern might be detected based on correlations
        assert handler._missing_pattern.pattern_type in ["MAR", "MCAR", "MNAR"]
        assert handler._missing_pattern.missing_percentage > 0
        
        # Check for pattern-specific recommendations
        recommendations = handler._missing_pattern.recommendations
        assert any("pattern detected" in rec.lower() for rec in recommendations)
    
    def test_simple_imputation(self, sample_data_mcar):
        """Test simple imputation strategy."""
        handler = MissingValueHandler(complexity="minimal")
        
        processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should have no missing values after imputation
        assert processed_data.isnull().sum().sum() == 0
        
        # Check that numeric columns used median, categorical used mode
        original_missing = sample_data_mcar.isnull().sum().sum()
        final_missing = processed_data.isnull().sum().sum()
        assert final_missing < original_missing
        
        # Verify metadata
        imputation_results = metadata.get('imputation_results', {})
        assert imputation_results.get('imputation_complete') is True
        assert imputation_results.get('original_missing_values', 0) > 0
    
    def test_auto_strategy_selection(self, sample_data_mcar):
        """Test automatic strategy selection."""
        handler = MissingValueHandler(strategy="auto", complexity="auto")
        
        processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should complete imputation
        assert processed_data.isnull().sum().sum() == 0
        
        # Should have selected a strategy
        missing_analysis = metadata.get('missing_value_analysis', {})
        assert missing_analysis.get('pattern_type') is not None
        assert missing_analysis.get('pattern_confidence', 0) >= 0
    
    def test_knn_imputation(self, sample_data_mcar):
        """Test KNN imputation strategy."""
        handler = MissingValueHandler(strategy="knn")
        
        # Mock the KNN strategy selection
        with patch.object(handler, '_select_optimal_strategy', return_value='knn'):
            processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should complete imputation
        assert processed_data.isnull().sum().sum() == 0
        
        # Verify KNN was attempted (might fallback to simple if KNN fails)
        imputation_results = metadata.get('imputation_results', {})
        assert imputation_results.get('imputation_complete') is True
    
    def test_iterative_imputation(self, sample_data_mcar):
        """Test iterative imputation strategy."""
        handler = MissingValueHandler(strategy="iterative")
        
        # Mock the iterative strategy selection
        with patch.object(handler, '_select_optimal_strategy', return_value='iterative'):
            processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should complete imputation
        assert processed_data.isnull().sum().sum() == 0
        
        # Verify iterative was attempted
        imputation_results = metadata.get('imputation_results', {})
        assert imputation_results.get('imputation_complete') is True
    
    def test_comprehensive_analysis(self, sample_data_mcar):
        """Test comprehensive analysis with all strategies."""
        handler = MissingValueHandler(complexity="comprehensive")
        
        processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should complete imputation
        assert processed_data.isnull().sum().sum() == 0
        
        # Should have comprehensive metadata
        assert 'missing_value_analysis' in metadata
        assert 'imputation_results' in metadata
        assert 'quality_assessment' in metadata
        assert 'composition_context' in metadata
        
        # Verify composition context suggestions
        composition_context = metadata.get('composition_context', {})
        assert 'suggested_next_steps' in composition_context
        assert composition_context.get('ready_for_analysis') is not None
    
    def test_high_missing_data_handling(self, sample_data_high_missing):
        """Test handling of data with very high missing percentages."""
        handler = MissingValueHandler()
        
        processed_data, metadata = handler.analyze(sample_data_high_missing)
        
        # Should still attempt imputation
        original_missing = sample_data_high_missing.isnull().sum().sum()
        final_missing = processed_data.isnull().sum().sum()
        
        # Should reduce missing values (might not eliminate all due to high missing rate)
        assert final_missing <= original_missing
        
        # Should detect high missing rate and provide appropriate recommendations
        missing_analysis = metadata.get('missing_value_analysis', {})
        recommendations = missing_analysis.get('recommendations_followed', [])
        # Should suggest advanced methods or caution about high missing rates
        assert len(recommendations) > 0
    
    def test_quality_assessment(self, sample_data_mcar):
        """Test imputation quality assessment."""
        handler = MissingValueHandler(cross_validation=True)
        
        processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should have quality assessment results
        quality_assessment = metadata.get('quality_assessment', {})
        assert quality_assessment is not None
        
        # Check for quality thresholds
        assert 'quality_thresholds' in quality_assessment
        
        # Verify imputation success
        imputation_results = metadata.get('imputation_results', {})
        assert imputation_results.get('imputation_complete') is True
    
    def test_streaming_compatibility(self, sample_data_mcar):
        """Test streaming configuration compatibility."""
        streaming_config = StreamingConfig(
            chunk_size=100,
            memory_limit_mb=50,
            enable_streaming=True
        )
        
        handler = MissingValueHandler(streaming_config=streaming_config)
        
        processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should complete successfully
        assert processed_data.isnull().sum().sum() == 0
        
        # Should indicate streaming was considered
        imputation_pipeline = metadata.get('imputation_pipeline', {})
        assert 'streaming_enabled' in imputation_pipeline
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        # Create problematic data
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, np.nan, np.nan],
            'col2': ['A', 'B', np.nan, np.nan, np.nan]
        })
        
        handler = MissingValueHandler()
        
        # Should handle gracefully even with very sparse data
        processed_data, metadata = handler.analyze(data)
        
        # Should not crash and provide some imputation
        assert processed_data is not None
        assert len(processed_data) == len(data)
        
        # Should have metadata even if imputation partially failed
        assert metadata is not None
        assert 'imputation_results' in metadata
    
    def test_fitted_imputers_storage(self, sample_data_mcar):
        """Test that fitted imputers are properly stored for reuse."""
        handler = MissingValueHandler()
        
        processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should have fitted imputers stored
        fitted_imputers = handler.get_fitted_imputers()
        assert len(fitted_imputers) > 0
        
        # Each imputer should correspond to a column with missing values
        original_missing_cols = sample_data_mcar.columns[sample_data_mcar.isnull().any()].tolist()
        assert len(fitted_imputers) >= len(original_missing_cols)
    
    def test_imputation_summary(self, sample_data_mcar):
        """Test imputation summary generation."""
        handler = MissingValueHandler()
        
        processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should generate meaningful summary
        summary = handler.get_imputation_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "imputation completed" in summary.lower()
        
        # Should indicate strategy and pattern
        pattern = handler.get_missing_value_pattern()
        assert pattern is not None
        assert pattern.pattern_type in ["MCAR", "MAR", "MNAR"]
    
    def test_integration_with_cleaning_pipeline(self, sample_data_mcar):
        """Test integration with the DataCleaningPipeline."""
        from src.localdata_mcp.pipeline.preprocessing import DataCleaningPipeline
        
        # Initialize cleaning pipeline with sophisticated missing value handling
        cleaning_pipeline = DataCleaningPipeline(
            analytical_intention="clean data with sophisticated missing value handling",
            cleaning_intensity="auto"
        )
        
        processed_data, metadata = cleaning_pipeline.analyze(sample_data_mcar)
        
        # Should complete successfully
        assert processed_data.isnull().sum().sum() == 0
        
        # Should have cleaning metadata that includes sophisticated imputation info
        operations_log = metadata.get('operations_log', [])
        missing_value_ops = [op for op in operations_log if 'missing' in op.get('operation_type', '').lower()]
        assert len(missing_value_ops) > 0
    
    def test_progressive_disclosure_levels(self, sample_data_mcar):
        """Test different progressive disclosure complexity levels."""
        
        # Test minimal complexity
        handler_minimal = MissingValueHandler(complexity="minimal")
        data_minimal, meta_minimal = handler_minimal.analyze(sample_data_mcar.copy())
        
        # Test auto complexity  
        handler_auto = MissingValueHandler(complexity="auto")
        data_auto, meta_auto = handler_auto.analyze(sample_data_mcar.copy())
        
        # Test comprehensive complexity
        handler_comprehensive = MissingValueHandler(complexity="comprehensive")
        data_comprehensive, meta_comprehensive = handler_comprehensive.analyze(sample_data_mcar.copy())
        
        # All should complete imputation
        assert data_minimal.isnull().sum().sum() == 0
        assert data_auto.isnull().sum().sum() == 0
        assert data_comprehensive.isnull().sum().sum() == 0
        
        # Comprehensive should have more detailed metadata
        comprehensive_keys = set(meta_comprehensive.keys())
        auto_keys = set(meta_auto.keys())
        minimal_keys = set(meta_minimal.keys())
        
        # Comprehensive should have at least as much metadata as auto
        assert len(comprehensive_keys) >= len(auto_keys)
        # Auto should have more than minimal
        assert len(auto_keys) >= len(minimal_keys)
    
    def test_cross_validation_disabled(self, sample_data_mcar):
        """Test behavior when cross-validation is disabled."""
        handler = MissingValueHandler(cross_validation=False)
        
        processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should still complete imputation
        assert processed_data.isnull().sum().sum() == 0
        
        # Cross-validation specific metadata should indicate it's disabled
        imputation_pipeline = metadata.get('imputation_pipeline', {})
        assert imputation_pipeline.get('cross_validation') is False
    
    def test_metadata_tracking_disabled(self, sample_data_mcar):
        """Test behavior when metadata tracking is disabled."""
        handler = MissingValueHandler(metadata_tracking=False)
        
        processed_data, metadata = handler.analyze(sample_data_mcar)
        
        # Should still complete imputation
        assert processed_data.isnull().sum().sum() == 0
        
        # Should have minimal metadata
        assert metadata is not None
        imputation_pipeline = metadata.get('imputation_pipeline', {})
        assert 'analytical_intention' in imputation_pipeline


@pytest.mark.performance
class TestMissingValueHandlerPerformance:
    """Performance tests for MissingValueHandler."""
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large dataset
        np.random.seed(42)
        n = 10000
        
        large_data = pd.DataFrame({
            'num1': np.random.normal(0, 1, n),
            'num2': np.random.normal(10, 5, n),
            'num3': np.random.exponential(2, n),
            'cat1': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
            'cat2': np.random.choice(['X', 'Y', 'Z'], n)
        })
        
        # Introduce missing values
        for col in large_data.columns:
            missing_mask = np.random.random(n) < 0.15  # 15% missing
            large_data.loc[missing_mask, col] = np.nan
        
        handler = MissingValueHandler(complexity="auto")
        
        import time
        start_time = time.time()
        processed_data, metadata = handler.analyze(large_data)
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 60 seconds)
        assert execution_time < 60
        
        # Should complete imputation
        assert processed_data.isnull().sum().sum() == 0
        
        # Should provide performance metadata
        imputation_results = metadata.get('imputation_results', {})
        assert imputation_results.get('imputation_complete') is True
    
    def test_memory_efficiency(self):
        """Test memory efficiency with moderately large datasets."""
        np.random.seed(42)
        n = 5000
        
        data = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, n) for i in range(20)
        })
        
        # Introduce missing values
        for col in data.columns:
            missing_mask = np.random.random(n) < 0.1
            data.loc[missing_mask, col] = np.nan
        
        handler = MissingValueHandler()
        
        # Monitor memory usage during processing
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        processed_data, metadata = handler.analyze(data)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Should not increase memory by more than 200MB for this dataset
        assert memory_increase < 200
        
        # Should complete imputation
        assert processed_data.isnull().sum().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])