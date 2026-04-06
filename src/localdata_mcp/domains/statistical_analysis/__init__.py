"""
Statistical Analysis Domain - Comprehensive statistical analysis capabilities.

This package implements advanced statistical analysis tools including hypothesis testing,
ANOVA, non-parametric tests, and experimental design using scipy.stats and sklearn integration.

Key Features:
- Hypothesis testing (t-tests, chi-square, normality, correlation tests)
- ANOVA analysis (one-way, two-way, post-hoc, effect sizes)
- Non-parametric tests (Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman)
- Experimental design (power analysis, effect sizes, confidence intervals)
- Full sklearn pipeline compatibility
- Streaming-compatible processing
- Comprehensive result formatting
"""

from ._anova import ANOVAAnalysisTransformer

# Base types
from ._base import StatisticalTestResult
from ._experimental import ExperimentalDesignTransformer

# Convenience functions
from ._functions import (
    analyze_experiment_design,
    calculate_effect_sizes,
    perform_anova,
    run_hypothesis_test,
)

# Transformers
from ._hypothesis import HypothesisTestingTransformer
from ._nonparametric import NonParametricTestTransformer

__all__ = [
    "StatisticalTestResult",
    "HypothesisTestingTransformer",
    "ANOVAAnalysisTransformer",
    "NonParametricTestTransformer",
    "ExperimentalDesignTransformer",
    "run_hypothesis_test",
    "perform_anova",
    "analyze_experiment_design",
    "calculate_effect_sizes",
]
