"""
Unit tests for Pipeline Compatibility Matrix.

Tests the core functionality of domain compatibility assessment, format conversion
pathway discovery, and pipeline validation for the Integration Shims Framework.
"""

import unittest
from unittest.mock import Mock, patch
import pytest

from src.localdata_mcp.pipeline.integration.compatibility_matrix import (
    PipelineCompatibilityMatrix,
    CompatibilityLevel,
    DomainProfile,
    create_compatibility_matrix,
    assess_pipeline_compatibility,
    find_optimal_format_for_domains
)
from src.localdata_mcp.pipeline.integration.interfaces import (
    DataFormat,
    DomainRequirements,
    CompatibilityScore,
    ValidationResult
)


class TestPipelineCompatibilityMatrix(unittest.TestCase):
    """Test cases for PipelineCompatibilityMatrix."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.matrix = create_compatibility_matrix()
    
    def test_initialization(self):
        """Test compatibility matrix initialization."""
        self.assertIsInstance(self.matrix, PipelineCompatibilityMatrix)
        self.assertEqual(len(self.matrix.list_domains()), 4)  # 4 standard domains
        
        domains = self.matrix.list_domains()
        expected_domains = ['statistical_analysis', 'regression_modeling', 'time_series', 'pattern_recognition']
        self.assertCountEqual(domains, expected_domains)
    
    def test_get_compatibility_same_format(self):
        """Test compatibility scoring for identical formats."""
        score = self.matrix.get_compatibility(DataFormat.PANDAS_DATAFRAME, DataFormat.PANDAS_DATAFRAME)
        
        self.assertIsInstance(score, CompatibilityScore)
        self.assertEqual(score.score, 1.0)
        self.assertTrue(score.direct_compatible)
        self.assertFalse(score.conversion_required)
        self.assertIsNone(score.conversion_path)
    
    def test_get_compatibility_different_formats(self):
        """Test compatibility scoring for different formats."""
        score = self.matrix.get_compatibility(DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)
        
        self.assertIsInstance(score, CompatibilityScore)
        self.assertGreater(score.score, 0.8)  # Should be high compatibility
        self.assertFalse(score.direct_compatible)
        self.assertTrue(score.conversion_required)
        self.assertIsNotNone(score.conversion_path)
    
    def test_compatibility_levels(self):
        """Test compatibility level classification."""
        # Perfect compatibility
        level = self.matrix.get_compatibility_level(1.0)
        self.assertEqual(level, CompatibilityLevel.PERFECT)
        
        # High compatibility
        level = self.matrix.get_compatibility_level(0.85)
        self.assertEqual(level, CompatibilityLevel.HIGH)
        
        # Moderate compatibility
        level = self.matrix.get_compatibility_level(0.65)
        self.assertEqual(level, CompatibilityLevel.MODERATE)
        
        # Low compatibility
        level = self.matrix.get_compatibility_level(0.4)
        self.assertEqual(level, CompatibilityLevel.LOW)
        
        # Incompatible
        level = self.matrix.get_compatibility_level(0.1)
        self.assertEqual(level, CompatibilityLevel.INCOMPATIBLE)
    
    def test_domain_profiles(self):
        """Test domain profile retrieval."""
        stats_profile = self.matrix.get_domain_profile('statistical_analysis')
        self.assertIsNotNone(stats_profile)
        self.assertEqual(stats_profile.domain_name, 'statistical_analysis')
        self.assertIn(DataFormat.PANDAS_DATAFRAME, stats_profile.base_requirements.input_formats)
        
        # Test non-existent domain
        missing_profile = self.matrix.get_domain_profile('non_existent_domain')
        self.assertIsNone(missing_profile)
    
    def test_validate_pipeline_single_step(self):
        """Test pipeline validation with single step."""
        result = self.matrix.validate_pipeline(['statistical_analysis'])
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.score, 1.0)
    
    def test_validate_pipeline_compatible_steps(self):
        """Test pipeline validation with compatible steps."""
        # Statistical analysis -> Regression modeling should be compatible
        result = self.matrix.validate_pipeline(['statistical_analysis', 'regression_modeling'])
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertGreater(result.score, 0.5)  # Should have reasonable compatibility
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_pipeline_unknown_domain(self):
        """Test pipeline validation with unknown domain."""
        result = self.matrix.validate_pipeline(['unknown_domain', 'statistical_analysis'])
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        self.assertIn('Unknown domain', result.errors[0])
    
    def test_validate_pipeline_complex(self):
        """Test complex pipeline validation."""
        # Test all 4 domains in sequence
        domains = ['statistical_analysis', 'regression_modeling', 'time_series', 'pattern_recognition']
        result = self.matrix.validate_pipeline(domains)
        
        self.assertIsInstance(result, ValidationResult)
        # Complex pipeline might have some compatibility issues but should not fail completely
        self.assertGreaterEqual(result.score, 0.3)
    
    def test_caching_behavior(self):
        """Test that caching works correctly."""
        # Enable caching and make same request twice
        matrix_with_cache = create_compatibility_matrix(enable_caching=True)
        
        # First request
        score1 = matrix_with_cache.get_compatibility(DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)
        
        # Second request should hit cache
        score2 = matrix_with_cache.get_compatibility(DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)
        
        # Should return same object/values
        self.assertEqual(score1.score, score2.score)
        
        # Check statistics
        stats = matrix_with_cache.get_statistics()
        self.assertEqual(stats['total_assessments'], 2)
        self.assertEqual(stats['cache_hit_rate'], 0.5)  # 1 out of 2 was cache hit
    
    def test_register_custom_domain(self):
        """Test registering a custom domain."""
        custom_requirements = DomainRequirements(
            domain_name="custom_domain",
            input_formats=[DataFormat.NUMPY_ARRAY],
            output_formats=[DataFormat.PYTHON_DICT],
            preferred_format=DataFormat.NUMPY_ARRAY
        )
        
        self.matrix.register_domain_requirements("custom_domain", custom_requirements)
        
        # Verify domain was registered
        self.assertIn("custom_domain", self.matrix.list_domains())
        
        custom_profile = self.matrix.get_domain_profile("custom_domain")
        self.assertIsNotNone(custom_profile)
        self.assertEqual(custom_profile.domain_name, "custom_domain")
    
    def test_conversion_path_creation(self):
        """Test conversion path creation."""
        path = self.matrix.find_conversion_path(DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY)
        
        self.assertIsNotNone(path)
        self.assertEqual(path.source_format, DataFormat.PANDAS_DATAFRAME)
        self.assertEqual(path.target_format, DataFormat.NUMPY_ARRAY)
        self.assertEqual(len(path.steps), 1)  # Direct conversion
        self.assertGreater(path.success_probability, 0.5)
    
    def test_format_family_compatibility(self):
        """Test format family-based compatibility scoring."""
        # Test tabular format family
        score = self.matrix.get_compatibility(DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES)
        self.assertGreater(score.score, 0.8)  # Should be high as both are tabular
        
        # Test array format family
        score = self.matrix.get_compatibility(DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE)
        self.assertGreater(score.score, 0.6)  # Should be moderate to high
        
        # Test cross-family compatibility
        score = self.matrix.get_compatibility(DataFormat.PYTHON_LIST, DataFormat.STATISTICAL_RESULT)
        self.assertLess(score.score, 0.6)  # Should be lower
    
    def test_compatibility_issues_and_recommendations(self):
        """Test that compatibility issues and recommendations are generated."""
        # Test problematic conversion
        score = self.matrix.get_compatibility(DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME)
        
        self.assertGreater(len(score.compatibility_issues + score.recommendations), 0)
        
        # Check for specific warnings about memory usage
        issues_text = ' '.join(score.compatibility_issues)
        self.assertIn('memory', issues_text.lower())


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_assess_pipeline_compatibility(self):
        """Test pipeline compatibility assessment utility."""
        result = assess_pipeline_compatibility(['statistical_analysis', 'regression_modeling'])
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
    
    def test_find_optimal_format_for_domains(self):
        """Test finding optimal format for multiple domains."""
        domains = ['statistical_analysis', 'regression_modeling']
        optimal_format = find_optimal_format_for_domains(domains)
        
        self.assertIsNotNone(optimal_format)
        # Should likely return PANDAS_DATAFRAME as it's versatile
        self.assertEqual(optimal_format, DataFormat.PANDAS_DATAFRAME)
    
    def test_find_optimal_format_empty_list(self):
        """Test finding optimal format with empty domain list."""
        optimal_format = find_optimal_format_for_domains([])
        self.assertIsNone(optimal_format)
    
    def test_find_optimal_format_single_domain(self):
        """Test finding optimal format with single domain."""
        optimal_format = find_optimal_format_for_domains(['time_series'])
        self.assertIsNotNone(optimal_format)
    
    def test_suggest_pipeline_improvements(self):
        """Test pipeline improvement suggestions."""
        from src.localdata_mcp.pipeline.integration.compatibility_matrix import suggest_pipeline_improvements
        
        suggestions = suggest_pipeline_improvements(['statistical_analysis', 'pattern_recognition'])
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Should contain format-related suggestions
        suggestions_text = ' '.join(suggestions)
        self.assertTrue(any(word in suggestions_text.lower() 
                           for word in ['format', 'conversion', 'intermediate']))


class TestCompatibilityMatrix(unittest.TestCase):
    """Integration tests for the compatibility matrix."""
    
    def test_end_to_end_pipeline_assessment(self):
        """Test complete end-to-end pipeline assessment."""
        matrix = create_compatibility_matrix()
        
        # Create a realistic data science pipeline
        pipeline = [
            'statistical_analysis',    # Data exploration
            'pattern_recognition',     # Feature engineering/clustering  
            'regression_modeling',     # Predictive modeling
            'time_series'             # Time series forecasting
        ]
        
        result = matrix.validate_pipeline(pipeline)
        
        # Should complete without errors
        self.assertIsInstance(result, ValidationResult)
        
        # May have warnings but should not be completely invalid
        if not result.is_valid:
            # If invalid, should have meaningful error messages
            self.assertGreater(len(result.errors), 0)
            self.assertTrue(all(isinstance(error, str) and len(error) > 10 
                               for error in result.errors))
        
        # Should have step-by-step details
        self.assertGreater(len(result.details), 0)
        
        # Should provide suggestions if score is low
        if result.score < 0.7:
            self.assertGreater(len(result.suggestions), 0)


if __name__ == '__main__':
    unittest.main()