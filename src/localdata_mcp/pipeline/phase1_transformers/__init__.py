"""
Phase 1 Tool Transformers - sklearn-compatible wrappers for profile_table, detect_data_types, and analyze_distributions.

This package implements sklearn BaseEstimator and TransformerMixin interfaces for the existing Phase 1 tools
to enable seamless pipeline integration while maintaining 100% API compatibility.

Key Features:
- Full sklearn pipeline compatibility
- Preserved streaming capabilities
- Memory-efficient processing
- Comprehensive parameter validation
- Backward compatible interfaces
"""

from .profile_table import ProfileTableTransformer
from .data_type_detector import DataTypeDetectorTransformer
from .distribution_analyzer import DistributionAnalyzerTransformer

__all__ = [
    "ProfileTableTransformer",
    "DataTypeDetectorTransformer",
    "DistributionAnalyzerTransformer",
]
