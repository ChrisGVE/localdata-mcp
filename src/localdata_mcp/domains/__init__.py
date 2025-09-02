"""
Domain-specific analysis modules for the LocalData MCP pipeline.

This package contains specialized analysis domains that integrate with the
core pipeline framework, each implementing specific analytical capabilities
using sklearn-compatible transformers.

Available domains:
- statistical_analysis: Comprehensive statistical analysis tools
- regression_modeling: Comprehensive regression analysis and modeling capabilities
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

from .regression_modeling import (
    # Core transformers
    LinearRegressionTransformer,
    RegularizedRegressionTransformer,
    LogisticRegressionTransformer,
    PolynomialRegressionTransformer,
    ResidualAnalysisTransformer,
    FeatureSelectionTransformer,
    
    # Pipeline
    RegressionModelingPipeline,
    
    # High-level functions
    fit_regression_model,
    evaluate_model_performance,
    analyze_residuals,
    select_features,
    
    # Result classes
    RegressionModelResult,
    ResidualAnalysisResult
)

__all__ = [
    # Statistical Analysis Domain
    "HypothesisTestingTransformer",
    "ANOVAAnalysisTransformer", 
    "NonParametricTestTransformer",
    "ExperimentalDesignTransformer",
    "run_hypothesis_test",
    "perform_anova",
    "analyze_experiment_design",
    "calculate_effect_sizes",
    
    # Regression Modeling Domain
    "LinearRegressionTransformer",
    "RegularizedRegressionTransformer",
    "LogisticRegressionTransformer",
    "PolynomialRegressionTransformer",
    "ResidualAnalysisTransformer",
    "FeatureSelectionTransformer",
    "RegressionModelingPipeline",
    "fit_regression_model",
    "evaluate_model_performance",
    "analyze_residuals",
    "select_features",
    "RegressionModelResult",
    "ResidualAnalysisResult"
]