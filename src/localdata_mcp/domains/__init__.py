"""
Domain-specific analysis modules for the LocalData MCP pipeline.

This package contains specialized analysis domains that integrate with the
core pipeline framework, each implementing specific analytical capabilities
using sklearn-compatible transformers.

Available domains:
- statistical_analysis: Comprehensive statistical analysis tools
"""

from .statistical_analysis import (
    HypothesisTestingTransformer,
    ANOVAAnalysisTransformer,
    NonParametricTestTransformer,
    ExperimentalDesignTransformer,
    run_hypothesis_test,
    perform_anova,
    analyze_experiment_design,
    calculate_effect_sizes,
)

__all__ = [
    "HypothesisTestingTransformer",
    "ANOVAAnalysisTransformer", 
    "NonParametricTestTransformer",
    "ExperimentalDesignTransformer",
    "run_hypothesis_test",
    "perform_anova",
    "analyze_experiment_design",
    "calculate_effect_sizes",
]