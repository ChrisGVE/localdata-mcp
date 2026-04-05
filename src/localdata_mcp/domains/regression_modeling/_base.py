"""
Regression & Modeling Domain - Base types and result classes.

Contains shared imports, result dataclasses, and common utilities
used across all regression modeling sub-modules.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    max_error,
    mean_squared_log_error,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
    RidgeCV,
    LassoCV,
    ElasticNetCV,
    TweedieRegressor,
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.kernel_ridge import KernelRidge
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, normal_ad
from statsmodels.stats.outliers_influence import OLSInfluence

from ...logging_manager import get_logger
from ...pipeline.base import (
    AnalysisPipelineBase,
    PipelineResult,
    CompositionMetadata,
    StreamingConfig,
    PipelineState,
)

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


@dataclass
class RegressionModelResult:
    """Standardized result structure for regression models."""

    model_type: str
    model_params: Dict[str, Any]
    fitted_model: Any

    # Performance metrics
    r2_score: float
    adjusted_r2: Optional[float] = None
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0

    # Statistical measures
    aic: Optional[float] = None
    bic: Optional[float] = None
    log_likelihood: Optional[float] = None

    # Model diagnostics
    residuals: Optional[np.ndarray] = None
    fitted_values: Optional[np.ndarray] = None
    standardized_residuals: Optional[np.ndarray] = None

    # Feature analysis
    feature_names: List[str] = field(default_factory=list)
    coefficients: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
    selected_features: Optional[List[str]] = None

    # Cross-validation results
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None

    # Additional info
    n_features: int = 0
    n_samples: int = 0
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result_dict = {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "performance_metrics": {
                "r2_score": self.r2_score,
                "mse": self.mse,
                "mae": self.mae,
                "rmse": self.rmse,
            },
            "n_features": self.n_features,
            "n_samples": self.n_samples,
        }

        if self.adjusted_r2 is not None:
            result_dict["performance_metrics"]["adjusted_r2"] = self.adjusted_r2
        if self.aic is not None:
            result_dict["performance_metrics"]["aic"] = self.aic
        if self.bic is not None:
            result_dict["performance_metrics"]["bic"] = self.bic

        if self.cv_scores is not None:
            result_dict["cross_validation"] = {
                "cv_scores": self.cv_scores,
                "cv_mean": self.cv_mean,
                "cv_std": self.cv_std,
            }

        if self.coefficients is not None:
            result_dict["feature_analysis"] = {
                "feature_names": self.feature_names,
                "coefficients": self.coefficients.tolist()
                if hasattr(self.coefficients, "tolist")
                else self.coefficients,
                "selected_features": self.selected_features,
            }

        if self.feature_importance is not None:
            result_dict["feature_analysis"]["feature_importance"] = (
                self.feature_importance.tolist()
                if hasattr(self.feature_importance, "tolist")
                else self.feature_importance
            )

        if self.assumptions_met:
            result_dict["model_diagnostics"] = {"assumptions_met": self.assumptions_met}

        if self.convergence_info:
            result_dict["convergence_info"] = self.convergence_info

        return result_dict


@dataclass
class ResidualAnalysisResult:
    """Result structure for residual analysis."""

    residuals: np.ndarray
    standardized_residuals: np.ndarray
    fitted_values: np.ndarray
    studentized_residuals: Optional[np.ndarray] = None

    # Diagnostic tests
    normality_test: Dict[str, Any] = field(default_factory=dict)
    homoscedasticity_test: Dict[str, Any] = field(default_factory=dict)
    autocorrelation_test: Dict[str, Any] = field(default_factory=dict)

    # Outlier detection
    outliers: Optional[List[int]] = None
    leverage: Optional[np.ndarray] = None
    cooks_distance: Optional[np.ndarray] = None

    # Summary statistics
    residual_stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result_dict = {
            "residual_statistics": self.residual_stats,
            "diagnostic_tests": {
                "normality": self.normality_test,
                "homoscedasticity": self.homoscedasticity_test,
                "autocorrelation": self.autocorrelation_test,
            },
        }

        if self.outliers is not None:
            result_dict["outlier_analysis"] = {
                "outlier_indices": self.outliers,
                "n_outliers": len(self.outliers),
            }

        if self.leverage is not None:
            result_dict["influence_measures"] = {
                "leverage_available": True,
                "cooks_distance_available": self.cooks_distance is not None,
            }

        return result_dict
