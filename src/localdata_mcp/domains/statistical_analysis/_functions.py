"""
Statistical Analysis Domain - MCP Tool Functions.

Convenience functions that wrap the transformer classes for use as MCP tools.
"""

from typing import Any, Dict, Optional, Union

import pandas as pd

from ._anova import ANOVAAnalysisTransformer
from ._experimental import ExperimentalDesignTransformer
from ._hypothesis import HypothesisTestingTransformer


# MCP Tool Functions
def run_hypothesis_test(
    data: Union[pd.DataFrame, str],
    test_type: str = "auto",
    alpha: float = 0.05,
    alternative: str = "two-sided",
    **kwargs,
) -> Dict[str, Any]:
    """
    Run comprehensive hypothesis testing analysis.

    Args:
        data: DataFrame or path to data file
        test_type: Type of hypothesis test to perform
        alpha: Significance level
        alternative: Alternative hypothesis direction
        **kwargs: Additional parameters for specific tests

    Returns:
        Dictionary containing test results and interpretations
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith(".csv"):
            data = pd.read_csv(data)
        elif data.endswith(".json"):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")

    # Initialize and run hypothesis testing transformer
    transformer = HypothesisTestingTransformer(
        test_type=test_type, alpha=alpha, alternative=alternative, **kwargs
    )

    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)

    return result_df.iloc[0].to_dict()


def perform_anova(
    data: Union[pd.DataFrame, str],
    anova_type: str = "one_way",
    alpha: float = 0.05,
    post_hoc: str = "tukey",
    **kwargs,
) -> Dict[str, Any]:
    """
    Perform comprehensive ANOVA analysis.

    Args:
        data: DataFrame or path to data file
        anova_type: Type of ANOVA analysis
        alpha: Significance level
        post_hoc: Post-hoc test method
        **kwargs: Additional ANOVA parameters

    Returns:
        Dictionary containing ANOVA results and post-hoc tests
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith(".csv"):
            data = pd.read_csv(data)
        elif data.endswith(".json"):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")

    # Initialize and run ANOVA transformer
    transformer = ANOVAAnalysisTransformer(
        anova_type=anova_type, alpha=alpha, post_hoc=post_hoc, **kwargs
    )

    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)

    return result_df.iloc[0].to_dict()


def analyze_experiment_design(
    data: Union[pd.DataFrame, str],
    analysis_type: str = "power_analysis",
    effect_size: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.80,
    **kwargs,
) -> Dict[str, Any]:
    """
    Analyze experimental design parameters and power.

    Args:
        data: DataFrame or path to data file
        analysis_type: Type of experimental design analysis
        effect_size: Expected effect size
        alpha: Type I error rate
        power: Desired statistical power
        **kwargs: Additional experimental design parameters

    Returns:
        Dictionary containing experimental design analysis results
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith(".csv"):
            data = pd.read_csv(data)
        elif data.endswith(".json"):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")

    # Initialize and run experimental design transformer
    transformer = ExperimentalDesignTransformer(
        analysis_type=analysis_type,
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        **kwargs,
    )

    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)

    return result_df.iloc[0].to_dict()


def calculate_effect_sizes(
    data: Union[pd.DataFrame, str], test_type: str = "auto", **kwargs
) -> Dict[str, Any]:
    """
    Calculate comprehensive effect sizes for statistical analyses.

    Args:
        data: DataFrame or path to data file
        test_type: Type of statistical test for effect size calculation
        **kwargs: Additional parameters for effect size calculations

    Returns:
        Dictionary containing calculated effect sizes and interpretations
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith(".csv"):
            data = pd.read_csv(data)
        elif data.endswith(".json"):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")

    # Use experimental design transformer for effect size calculations
    transformer = ExperimentalDesignTransformer(analysis_type="effect_size", **kwargs)

    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)

    return result_df.iloc[0].to_dict()
