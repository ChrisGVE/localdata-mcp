"""
Regression & Modeling Domain - Comprehensive regression analysis and modeling capabilities.

This package implements advanced regression analysis tools including linear models,
generalized linear models, regularization, and comprehensive model evaluation
using scikit-learn's linear models and statistical extensions.

Key Features:
- Linear Regression Suite (OLS, Ridge, Lasso, ElasticNet)
- Generalized Linear Models (Logistic, Poisson, Gamma regression)
- Non-linear Extensions (Polynomial features, splines, kernel methods)
- Model Evaluation Suite (metrics, residual analysis, cross-validation)
- Feature selection and importance analysis
- Full sklearn pipeline compatibility
- Streaming-compatible processing
- Comprehensive result formatting
"""

# Base types
from ._base import RegressionModelResult, ResidualAnalysisResult
from ._feature_selection import FeatureSelectionTransformer

# Convenience functions
from ._functions import (
    analyze_residuals,
    evaluate_model_performance,
    fit_regression_model,
    select_features,
)

# Transformers
from ._linear import LinearRegressionTransformer
from ._logistic import LogisticRegressionTransformer

# Pipeline
from ._pipeline import RegressionModelingPipeline
from ._polynomial import PolynomialRegressionTransformer
from ._regularized import RegularizedRegressionTransformer
from ._residuals import ResidualAnalysisTransformer

__all__ = [
    "RegressionModelResult",
    "ResidualAnalysisResult",
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
]
