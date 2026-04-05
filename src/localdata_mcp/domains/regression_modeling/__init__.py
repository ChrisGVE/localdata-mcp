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

# Transformers
from ._linear import LinearRegressionTransformer
from ._regularized import RegularizedRegressionTransformer
from ._logistic import LogisticRegressionTransformer
from ._polynomial import PolynomialRegressionTransformer
from ._residuals import ResidualAnalysisTransformer
from ._feature_selection import FeatureSelectionTransformer

# Pipeline
from ._pipeline import RegressionModelingPipeline

# Convenience functions
from ._functions import (
    fit_regression_model,
    evaluate_model_performance,
    analyze_residuals,
    select_features,
)

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
