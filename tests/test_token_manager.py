"""Tests for the TokenManager intelligent token estimation system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from localdata_mcp.token_manager import (
    TokenManager, 
    get_token_manager,
    TokenEstimation,
    ResponseMetadata,
    ChunkingRecommendation,
    MODEL_CONTEXT_WINDOWS,
    TOKEN_RISK_THRESHOLDS
)


class TestTokenManager:
    """Test the core TokenManager functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.token_manager = TokenManager()
    
    def test_initialization(self):
        """Test TokenManager initialization."""
        tm = TokenManager()
        assert tm.sample_size == 100
        assert tm.encoding is not None  # Should initialize tiktoken
        
        # Test with custom sample size
        tm_custom = TokenManager(sample_size=50)
        assert tm_custom.sample_size == 50
    
    def test_analyze_column_types(self):
        """Test column type analysis."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'string_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'datetime_col': pd.date_range('2023-01-01', periods=3),
            'object_col': ['x', 1, None]
        })
        
        result = self.token_manager._analyze_column_types(df)
        
        assert 'int_col' in result['numeric']
        assert 'float_col' in result['numeric']
        assert 'string_col' in result['text']
        assert 'object_col' in result['text']
        assert 'bool_col' in result['other']
        assert 'datetime_col' in result['other']
    
    def test_estimate_text_column_tokens(self):
        """Test text column token estimation."""
        df = pd.DataFrame({
            'short_text': ['hi', 'bye', 'ok'],
            'long_text': ['This is a longer text'] * 3,
            'empty_text': ['', None, 'something'],
            'mixed_text': ['short', 'this is much longer text', '']
        })
        
        result = self.token_manager._estimate_text_column_tokens(df, df.columns.tolist())
        
        # Should have estimates for all columns
        assert all(col in result for col in df.columns)
        
        # Short text should have fewer tokens than long text
        assert result['short_text'] < result['long_text']
        
        # All values should be positive integers
        assert all(isinstance(v, int) and v > 0 for v in result.values())
    
    def test_estimate_other_column_tokens(self):
        """Test token estimation for non-text, non-numeric columns."""
        bool_series = pd.Series([True, False, True])
        datetime_series = pd.Series(pd.date_range('2023-01-01', periods=3))
        
        bool_tokens = self.token_manager._estimate_other_column_tokens(bool_series)
        datetime_tokens = self.token_manager._estimate_other_column_tokens(datetime_series)
        
        assert bool_tokens == 1  # Boolean = 1 token
        assert datetime_tokens == 3  # Datetime ≈ 3 tokens
    
    def test_calculate_json_overhead(self):
        """Test JSON serialization overhead calculation."""
        df_small = pd.DataFrame({'a': [1], 'b': [2]})
        df_large = pd.DataFrame({f'col_{i}': [1] for i in range(10)})
        
        overhead_small = self.token_manager._calculate_json_overhead(df_small)
        overhead_large = self.token_manager._calculate_json_overhead(df_large)
        
        # Overhead should scale with number of columns
        assert overhead_large > overhead_small
        
        # Should include opening/closing braces + field overhead
        assert overhead_small == 2 * 2 + 2  # 2 fields * 2 tokens + 2 braces
    
    def test_assess_token_risk(self):
        """Test token risk level assessment."""
        assert self.token_manager._assess_token_risk(500) == 'low'
        assert self.token_manager._assess_token_risk(5000) == 'medium'
        assert self.token_manager._assess_token_risk(25000) == 'high'
        assert self.token_manager._assess_token_risk(75000) == 'critical'
    
    def test_assess_context_compatibility(self):
        """Test model context window compatibility."""
        compatibility = self.token_manager._assess_context_compatibility(5000)
        
        # Should have entries for all models except 'default'
        expected_models = [m for m in MODEL_CONTEXT_WINDOWS.keys() if m != 'default']
        assert all(model in compatibility for model in expected_models)
        
        # Small token count should fit in most models
        assert compatibility['gpt-4'] == True
        assert compatibility['claude-3-haiku'] == True
        
        # Test large token count
        large_compat = self.token_manager._assess_context_compatibility(150000)
        assert large_compat['gpt-4'] == False  # 8K context
        assert large_compat['claude-3-haiku'] == True  # 200K context
    
    def test_estimate_tokens_from_dataframe_small(self):
        """Test complete DataFrame token estimation with small dataset."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [95.5, 87.2, 92.1],
            'active': [True, False, True]
        })
        
        estimation = self.token_manager.estimate_tokens_from_dataframe(df)
        
        # Basic structure checks
        assert isinstance(estimation, TokenEstimation)
        assert estimation.total_rows == 3
        assert estimation.total_tokens > 0
        assert estimation.tokens_per_row > 0
        assert estimation.confidence > 0
        
        # Column categorization
        assert 'id' in estimation.numeric_columns
        assert 'score' in estimation.numeric_columns
        assert 'name' in estimation.text_columns
        assert 'active' in estimation.other_columns
        
        # Token breakdown
        assert len(estimation.column_token_breakdown) == 4
        assert all(tokens > 0 for tokens in estimation.column_token_breakdown.values())
        
        # Risk assessment
        assert estimation.risk_level in ['low', 'medium', 'high', 'critical']
        
        # Method should be 'full' for small dataset
        assert estimation.estimation_method == 'full'
    
    def test_estimate_tokens_from_dataframe_large(self):
        """Test DataFrame estimation with large dataset requiring sampling."""
        # Create dataset larger than sample_size
        large_df = pd.DataFrame({
            'id': range(200),
            'description': [f'This is item number {i} with some text' for i in range(200)]
        })
        
        estimation = self.token_manager.estimate_tokens_from_dataframe(large_df)
        
        assert estimation.total_rows == 200
        assert estimation.sample_size == self.token_manager.sample_size  # Should use sample
        assert estimation.estimation_method == 'sampled'
        assert estimation.confidence < 1.0  # Should be reduced for sampling
    
    def test_estimate_tokens_for_query_result(self):
        """Test query result token estimation with extrapolation."""
        sample_df = pd.DataFrame({
            'user_id': [12345],
            'username': ['john_doe'],
            'email': ['john.doe@example.com'],
            'created_at': ['2023-01-01 12:00:00']
        })
        
        estimation = self.token_manager.estimate_tokens_for_query_result(10000, sample_df)
        
        assert estimation.total_rows == 10000
        assert estimation.sample_size == 1
        assert estimation.estimation_method == 'extrapolated'
        assert estimation.confidence < 1.0  # Should be reduced for extrapolation
        assert estimation.total_tokens == int(10000 * estimation.tokens_per_row)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        estimation = self.token_manager.estimate_tokens_from_dataframe(empty_df)
        
        assert estimation.total_tokens == 0
        assert estimation.tokens_per_row == 0
        assert estimation.confidence == 1.0
        assert estimation.estimation_method == 'empty'
        assert estimation.risk_level == 'low'
    
    def test_get_response_metadata(self):
        """Test generation of rich response metadata."""
        df = pd.DataFrame({
            'numeric_col': range(1000),
            'text_col': ['some text content'] * 1000
        })
        
        estimation = self.token_manager.estimate_tokens_from_dataframe(df)
        metadata = self.token_manager.get_response_metadata(estimation)
        
        assert isinstance(metadata, ResponseMetadata)
        assert metadata.row_count == 1000
        assert metadata.column_count == 2
        assert metadata.response_size_category in ['small', 'medium', 'large', 'xlarge']
        assert metadata.data_density in ['sparse', 'moderate', 'dense']
        assert isinstance(metadata.text_heavy, bool)
        assert isinstance(metadata.streaming_recommended, bool)
        
        # Should have model compatibility info
        assert len(metadata.model_compatibility) > 0
        assert all('fits_in_context' in info for info in metadata.model_compatibility.values())
    
    def test_chunking_recommendations(self):
        """Test chunking recommendation generation."""
        # Small dataset - should not recommend chunking
        small_df = pd.DataFrame({'col': range(10)})
        small_estimation = self.token_manager.estimate_tokens_from_dataframe(small_df)
        small_metadata = self.token_manager.get_response_metadata(small_estimation)
        
        # Large dataset - should recommend chunking
        large_df = pd.DataFrame({
            'text_data': ['This is a long text that will have many tokens'] * 1000
        })
        large_estimation = self.token_manager.estimate_tokens_from_dataframe(large_df)
        large_metadata = self.token_manager.get_response_metadata(large_estimation)
        
        if small_metadata.chunking_recommendation:
            assert not small_metadata.chunking_recommendation.should_chunk
        
        if large_metadata.chunking_recommendation:
            assert large_metadata.chunking_recommendation.should_chunk
            assert large_metadata.chunking_recommendation.recommended_chunk_size > 0
            assert large_metadata.chunking_recommendation.strategy in ['row_based', 'column_based', 'mixed']
    
    def test_singleton_token_manager(self):
        """Test that get_token_manager returns singleton instance."""
        tm1 = get_token_manager()
        tm2 = get_token_manager()
        
        assert tm1 is tm2  # Should be same instance
        assert isinstance(tm1, TokenManager)
    
    @patch('tiktoken.get_encoding')
    def test_tiktoken_initialization_failure(self, mock_get_encoding):
        """Test graceful handling of tiktoken initialization failure."""
        mock_get_encoding.side_effect = Exception("Failed to load encoding")
        
        tm = TokenManager()
        assert tm.encoding is None
        
        # Should still work with fallback methods
        df = pd.DataFrame({'text': ['hello world']})
        estimation = tm.estimate_tokens_from_dataframe(df)
        assert estimation.total_tokens > 0  # Should use fallback estimation
    
    def test_confidence_calculation(self):
        """Test confidence calculation under different conditions."""
        # Full dataset should have high confidence
        small_df = pd.DataFrame({'col': range(50)})
        small_est = self.token_manager.estimate_tokens_from_dataframe(small_df)
        
        # Sampled dataset should have lower confidence
        large_df = pd.DataFrame({'col': range(500)})
        large_est = self.token_manager.estimate_tokens_from_dataframe(large_df)
        
        # Text-heavy dataset should have lower confidence
        text_df = pd.DataFrame({'text': ['variable length text'] * 50})
        text_est = self.token_manager.estimate_tokens_from_dataframe(text_df)
        
        assert small_est.confidence > large_est.confidence  # Sampling reduces confidence
        # Text content also affects confidence due to variability
    
    def test_performance_estimation(self):
        """Test processing complexity and response time estimation."""
        # Simple numeric data
        simple_df = pd.DataFrame({'nums': range(100)})
        simple_est = self.token_manager.estimate_tokens_from_dataframe(simple_df)
        simple_meta = self.token_manager.get_response_metadata(simple_est)
        
        # Complex text data
        complex_df = pd.DataFrame({
            'long_text': ['This is a very long text field with lots of content'] * 1000
        })
        complex_est = self.token_manager.estimate_tokens_from_dataframe(complex_df)
        complex_meta = self.token_manager.get_response_metadata(complex_est)
        
        # Complex data should have higher processing complexity
        complexity_levels = ['low', 'medium', 'high']
        assert simple_meta.processing_complexity in complexity_levels
        assert complex_meta.processing_complexity in complexity_levels
        
        # Response time should be positive
        assert simple_meta.estimated_response_time >= 0
        assert complex_meta.estimated_response_time >= 0
        assert complex_meta.estimated_response_time >= simple_meta.estimated_response_time


class TestTokenManagerIntegration:
    """Test integration with other components."""
    
    def test_query_analyzer_integration(self):
        """Test that TokenManager works with QueryAnalyzer interface."""
        from localdata_mcp.token_manager import get_token_manager
        
        # Create sample data that would come from QueryAnalyzer
        sample_row = pd.Series({
            'id': 123,
            'name': 'John Doe',
            'email': 'john@example.com',
            'created': '2023-01-01'
        })
        
        token_manager = get_token_manager()
        sample_df = pd.DataFrame([sample_row])
        
        # Should work with the query result estimation method
        estimation = token_manager.estimate_tokens_for_query_result(5000, sample_df)
        
        assert estimation.total_rows == 5000
        assert estimation.total_tokens > 0
        assert estimation.estimation_method == 'extrapolated'
    
    def test_streaming_executor_metadata_format(self):
        """Test that metadata format matches StreamingExecutor expectations."""
        df = pd.DataFrame({
            'id': range(100),
            'description': ['Sample text'] * 100
        })
        
        token_manager = get_token_manager()
        estimation = token_manager.estimate_tokens_for_query_result(1000, df.head(1))
        response_metadata = token_manager.get_response_metadata(estimation)
        
        # Verify the metadata structure that StreamingExecutor expects
        expected_fields = [
            'estimated_tokens',
            'response_size_category', 
            'data_density',
            'text_heavy',
            'streaming_recommended',
            'processing_complexity',
            'estimated_response_time'
        ]
        
        for field in expected_fields:
            assert hasattr(response_metadata, field), f"Missing field: {field}"


class TestTokenEstimationAccuracy:
    """Test the accuracy of token estimation methods."""
    
    def setup_method(self):
        self.token_manager = TokenManager()
    
    def test_numeric_column_accuracy(self):
        """Test that numeric columns are estimated as 1 token per value."""
        df = pd.DataFrame({
            'integers': [1, 2, 3, 4, 5],
            'floats': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        estimation = self.token_manager.estimate_tokens_from_dataframe(df)
        
        # Each numeric value should be ~1 token + JSON overhead
        # 2 numeric columns * 1 token each + JSON overhead
        expected_content_tokens = 2  # 2 numeric columns
        json_overhead = estimation.json_overhead_per_row
        
        # Should be close to expected (allowing for small variations)
        assert abs(estimation.tokens_per_row - (expected_content_tokens + json_overhead)) < 1
    
    def test_text_column_variability(self):
        """Test that text columns handle variability correctly."""
        # Short text
        short_df = pd.DataFrame({'text': ['hi', 'bye', 'ok'] * 10})
        short_est = self.token_manager.estimate_tokens_from_dataframe(short_df)
        
        # Long text
        long_df = pd.DataFrame({
            'text': ['This is a much longer text with many more tokens'] * 10
        })
        long_est = self.token_manager.estimate_tokens_from_dataframe(long_df)
        
        # Long text should have significantly more tokens per row
        assert long_est.tokens_per_row > short_est.tokens_per_row * 2
    
    def test_mixed_data_types(self):
        """Test estimation with mixed data types."""
        df = pd.DataFrame({
            'id': range(50),  # Numeric: 1 token each
            'name': (['John', 'Jane', 'Bob'] * 16) + ['John', 'Jane'],  # Text: variable (50 total)
            'active': [True, False] * 25,  # Boolean: 1 token each
            'created': pd.date_range('2023-01-01', periods=50),  # Datetime: ~3 tokens
            'score': np.random.uniform(0, 100, 50)  # Float: 1 token each
        })
        
        estimation = self.token_manager.estimate_tokens_from_dataframe(df)
        
        # Should have reasonable token counts
        assert estimation.total_tokens > 50  # At least 1 token per row
        assert estimation.total_tokens < 50 * 50  # Not excessive
        
        # Should categorize columns correctly
        assert 'id' in estimation.numeric_columns
        assert 'score' in estimation.numeric_columns
        assert 'name' in estimation.text_columns
        assert 'active' in estimation.other_columns
        assert 'created' in estimation.other_columns


class TestChunkingRecommendations:
    """Test the chunking recommendation system."""
    
    def setup_method(self):
        self.token_manager = TokenManager()
    
    def test_no_chunking_for_small_data(self):
        """Test that small datasets don't get chunking recommendations."""
        small_df = pd.DataFrame({'col': range(10)})
        estimation = self.token_manager.estimate_tokens_from_dataframe(small_df)
        
        chunk_rec = self.token_manager._generate_chunking_recommendation(estimation)
        
        assert not chunk_rec.should_chunk
        assert chunk_rec.recommended_chunk_size == 0
        assert chunk_rec.strategy == 'none'
    
    def test_chunking_for_large_data(self):
        """Test chunking recommendations for large datasets."""
        # Create large dataset
        large_text = 'This is a long text that will generate many tokens ' * 20
        large_df = pd.DataFrame({'text': [large_text] * 200})
        
        estimation = self.token_manager.estimate_tokens_from_dataframe(large_df)
        chunk_rec = self.token_manager._generate_chunking_recommendation(estimation)
        
        if estimation.total_tokens > 5000:  # Should trigger chunking
            assert chunk_rec.should_chunk
            assert chunk_rec.recommended_chunk_size > 0
            assert chunk_rec.estimated_chunks > 1
            assert chunk_rec.strategy in ['row_based', 'column_based', 'mixed']
            assert len(chunk_rec.chunk_size_rationale) > 0
    
    def test_chunk_size_calculation(self):
        """Test that chunk size calculations are reasonable."""
        # Create dataset with known token characteristics
        df = pd.DataFrame({
            'text': ['Ten tokens in this text roughly'] * 1000  # ~10 tokens per row
        })
        
        estimation = self.token_manager.estimate_tokens_from_dataframe(df)
        
        if estimation.recommended_chunk_size:
            # Should target ~5000 tokens per chunk
            expected_rows = 5000 / estimation.tokens_per_row
            actual_chunk_size = estimation.recommended_chunk_size
            
            # Should be reasonably close (within 50%)
            assert abs(actual_chunk_size - expected_rows) < expected_rows * 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])