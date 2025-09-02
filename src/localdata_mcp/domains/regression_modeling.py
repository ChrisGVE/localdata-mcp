"""
Regression & Modeling Domain - Comprehensive regression analysis and modeling capabilities.

This module implements advanced regression analysis tools including linear models,
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
    r2_score, mean_squared_error, mean_absolute_error, 
    explained_variance_score, max_error, mean_squared_log_error
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
    RidgeCV, LassoCV, ElasticNetCV, TweedieRegressor
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.kernel_ridge import KernelRidge
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, normal_ad
from statsmodels.stats.outliers_influence import OLSInfluence

from ..logging_manager import get_logger
from ..pipeline.base import (
    AnalysisPipelineBase, PipelineResult, CompositionMetadata, 
    StreamingConfig, PipelineState
)

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result_dict = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'performance_metrics': {
                'r2_score': self.r2_score,
                'mse': self.mse,
                'mae': self.mae,
                'rmse': self.rmse
            },
            'n_features': self.n_features,
            'n_samples': self.n_samples
        }
        
        if self.adjusted_r2 is not None:
            result_dict['performance_metrics']['adjusted_r2'] = self.adjusted_r2
        if self.aic is not None:
            result_dict['performance_metrics']['aic'] = self.aic
        if self.bic is not None:
            result_dict['performance_metrics']['bic'] = self.bic
            
        if self.cv_scores is not None:
            result_dict['cross_validation'] = {
                'cv_scores': self.cv_scores,
                'cv_mean': self.cv_mean,
                'cv_std': self.cv_std
            }
            
        if self.coefficients is not None:
            result_dict['feature_analysis'] = {
                'feature_names': self.feature_names,
                'coefficients': self.coefficients.tolist() if hasattr(self.coefficients, 'tolist') else self.coefficients,
                'selected_features': self.selected_features
            }
            
        if self.feature_importance is not None:
            result_dict['feature_analysis']['feature_importance'] = (
                self.feature_importance.tolist() if hasattr(self.feature_importance, 'tolist') else self.feature_importance
            )
            
        if self.assumptions_met:
            result_dict['model_diagnostics'] = {
                'assumptions_met': self.assumptions_met
            }
            
        if self.convergence_info:
            result_dict['convergence_info'] = self.convergence_info
            
        return result_dict


@dataclass
class ResidualAnalysisResult:
    """Result structure for residual analysis."""
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    studentized_residuals: Optional[np.ndarray] = None
    fitted_values: np.ndarray
    
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
            'residual_statistics': self.residual_stats,
            'diagnostic_tests': {
                'normality': self.normality_test,
                'homoscedasticity': self.homoscedasticity_test,
                'autocorrelation': self.autocorrelation_test
            }
        }
        
        if self.outliers is not None:
            result_dict['outlier_analysis'] = {
                'outlier_indices': self.outliers,
                'n_outliers': len(self.outliers)
            }
            
        if self.leverage is not None:
            result_dict['influence_measures'] = {
                'leverage_available': True,
                'cooks_distance_available': self.cooks_distance is not None
            }
            
        return result_dict


class LinearRegressionTransformer(BaseEstimator, TransformerMixin, RegressorMixin):
    """
    sklearn-compatible transformer for linear regression analysis.
    
    Implements ordinary least squares regression with comprehensive
    statistical analysis, diagnostics, and model evaluation.
    
    Parameters:
    -----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model
    normalize : bool, default=False
        Whether to normalize the features before regression
    include_diagnostics : bool, default=True
        Whether to include comprehensive model diagnostics
    alpha : float, default=0.05
        Significance level for statistical tests
    """
    
    def __init__(self, fit_intercept=True, normalize=False, include_diagnostics=True, alpha=0.05):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.include_diagnostics = include_diagnostics
        self.alpha = alpha
        
        # Initialize model
        self.model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
        self.result_ = None
        
    def fit(self, X, y, sample_weight=None, feature_names=None):
        """
        Fit linear regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, shape (n_samples,), optional
            Individual weights for each sample
        feature_names : list of str, optional
            Names of features
            
        Returns:
        --------
        self : LinearRegressionTransformer
            Returns self for method chaining
        """
        start_time = time.time()
        logger.info("Starting linear regression fit")
        
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = feature_names
        
        # Fit the model
        try:
            self.model.fit(X, y, sample_weight=sample_weight)
            
            # Calculate comprehensive metrics
            y_pred = self.model.predict(X)
            residuals = y - y_pred
            
            # Basic metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            # Adjusted R²
            n_samples, n_features = X.shape
            adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            
            # Statistical analysis using statsmodels for comprehensive metrics
            aic, bic, log_likelihood = None, None, None
            assumptions_met = {}
            
            if self.include_diagnostics:
                try:
                    # Add intercept for statsmodels
                    X_sm = sm.add_constant(X) if self.fit_intercept else X
                    sm_model = sm.OLS(y, X_sm).fit()
                    
                    aic = sm_model.aic
                    bic = sm_model.bic
                    log_likelihood = sm_model.llf
                    
                    # Check assumptions
                    assumptions_met = self._check_assumptions(X, y, residuals, sm_model)
                    
                except Exception as e:
                    logger.warning(f"Failed to compute statistical diagnostics: {e}")
            
            # Create result object
            self.result_ = RegressionModelResult(
                model_type="Linear Regression",
                model_params={
                    'fit_intercept': self.fit_intercept,
                    'normalize': self.normalize
                },
                fitted_model=self.model,
                r2_score=r2,
                adjusted_r2=adjusted_r2,
                mse=mse,
                mae=mae,
                rmse=rmse,
                aic=aic,
                bic=bic,
                log_likelihood=log_likelihood,
                residuals=residuals,
                fitted_values=y_pred,
                standardized_residuals=residuals / np.std(residuals),
                feature_names=feature_names,
                coefficients=self.model.coef_,
                n_features=n_features,
                n_samples=n_samples,
                assumptions_met=assumptions_met
            )
            
            # Add intercept to coefficients if fitted
            if self.fit_intercept:
                self.result_.coefficients = np.concatenate([[self.model.intercept_], self.model.coef_])
                self.result_.feature_names = ['intercept'] + feature_names
            
            fit_time = time.time() - start_time
            logger.info(f"Linear regression fit completed in {fit_time:.3f}s - R² = {r2:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting linear regression: {e}")
            raise
            
        return self
    
    def transform(self, X):
        """
        Transform method returns predictions for pipeline compatibility.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted values
        """
        check_is_fitted(self)
        return self.predict(X)
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted values
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        return self.model.predict(X)
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R².
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True values for X
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights
            
        Returns:
        --------
        score : float
            R² of self.predict(X) wrt. y
        """
        return self.model.score(X, y, sample_weight=sample_weight)
    
    def _check_assumptions(self, X, y, residuals, sm_model):
        """Check linear regression assumptions."""
        assumptions = {}
        
        try:
            # 1. Normality of residuals (Anderson-Darling test)
            ad_stat, ad_p = normal_ad(residuals)
            assumptions['normality'] = ad_p > self.alpha
            
            # 2. Homoscedasticity (Breusch-Pagan test)
            try:
                bp_stat, bp_p, _, _ = het_breuschpagan(residuals, sm_model.model.exog)
                assumptions['homoscedasticity'] = bp_p > self.alpha
            except:
                # Fallback to White test if BP fails
                try:
                    w_stat, w_p, _, _ = het_white(residuals, sm_model.model.exog)
                    assumptions['homoscedasticity'] = w_p > self.alpha
                except:
                    assumptions['homoscedasticity'] = None
            
            # 3. Independence (Durbin-Watson test)
            try:
                from statsmodels.stats.diagnostic import durbin_watson
                dw_stat = durbin_watson(residuals)
                # Rule of thumb: values between 1.5-2.5 suggest no autocorrelation
                assumptions['independence'] = 1.5 <= dw_stat <= 2.5
            except:
                assumptions['independence'] = None
                
        except Exception as e:
            logger.warning(f"Failed to check some model assumptions: {e}")
            
        return assumptions
    
    def get_result(self):
        """Get the comprehensive regression result."""
        check_is_fitted(self)
        return self.result_


class RegularizedRegressionTransformer(BaseEstimator, TransformerMixin, RegressorMixin):
    """
    sklearn-compatible transformer for regularized regression analysis.
    
    Supports Ridge, Lasso, and Elastic Net regression with automatic
    hyperparameter tuning via cross-validation and comprehensive evaluation.
    
    Parameters:
    -----------
    method : str, default='ridge'
        Regularization method: 'ridge', 'lasso', 'elastic_net'
    alpha : float or 'auto', default='auto'
        Regularization strength. If 'auto', uses cross-validation
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (only for elastic_net method)
    cv : int, default=5
        Number of cross-validation folds for hyperparameter tuning
    max_iter : int, default=1000
        Maximum iterations for solver convergence
    """
    
    def __init__(self, method='ridge', alpha='auto', l1_ratio=0.5, cv=5, max_iter=1000):
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.max_iter = max_iter
        
        self.model = None
        self.result_ = None
        
    def fit(self, X, y, sample_weight=None, feature_names=None):
        """
        Fit regularized regression model with optional cross-validation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, shape (n_samples,), optional
            Individual weights for each sample
        feature_names : list of str, optional
            Names of features
            
        Returns:
        --------
        self : RegularizedRegressionTransformer
            Returns self for method chaining
        """
        start_time = time.time()
        logger.info(f"Starting {self.method} regression fit")
        
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = feature_names
        
        try:
            # Select model based on method and alpha tuning preference
            if self.alpha == 'auto':
                if self.method == 'ridge':
                    self.model = RidgeCV(cv=self.cv)
                elif self.method == 'lasso':
                    self.model = LassoCV(cv=self.cv, max_iter=self.max_iter)
                elif self.method == 'elastic_net':
                    self.model = ElasticNetCV(cv=self.cv, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
            else:
                if self.method == 'ridge':
                    self.model = Ridge(alpha=self.alpha)
                elif self.method == 'lasso':
                    self.model = Lasso(alpha=self.alpha, max_iter=self.max_iter)
                elif self.method == 'elastic_net':
                    self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
            
            # Fit the model
            self.model.fit(X, y, sample_weight=sample_weight)
            
            # Get optimal alpha if using CV
            optimal_alpha = getattr(self.model, 'alpha_', self.alpha)
            
            # Calculate comprehensive metrics
            y_pred = self.model.predict(X)
            residuals = y - y_pred
            
            # Basic metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            # Adjusted R²
            n_samples, n_features = X.shape
            adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            
            # Cross-validation scores
            cv_scores = None
            cv_mean = None
            cv_std = None
            if hasattr(self.model, 'mse_path_'):
                # For CV models, get cross-validation R² scores
                cv_scores = cross_val_score(self.model, X, y, cv=self.cv, scoring='r2')
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
            
            # Feature selection analysis
            selected_features = None
            feature_importance = None
            if hasattr(self.model, 'coef_'):
                # For Lasso, identify selected features (non-zero coefficients)
                if self.method == 'lasso':
                    selected_mask = np.abs(self.model.coef_) > 1e-10
                    selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
                
                # Feature importance based on absolute coefficient values
                feature_importance = np.abs(self.model.coef_)
            
            # Convergence information
            convergence_info = {}
            if hasattr(self.model, 'n_iter_'):
                convergence_info['n_iterations'] = getattr(self.model, 'n_iter_', None)
                convergence_info['converged'] = getattr(self.model, 'n_iter_', 0) < self.max_iter
            
            if self.alpha == 'auto':
                convergence_info['optimal_alpha'] = optimal_alpha
                if hasattr(self.model, 'alphas_'):
                    convergence_info['alphas_tested'] = len(self.model.alphas_)
            
            # Create result object
            self.result_ = RegressionModelResult(
                model_type=f"{self.method.title()} Regression",
                model_params={
                    'method': self.method,
                    'alpha': optimal_alpha,
                    'l1_ratio': self.l1_ratio if self.method == 'elastic_net' else None,
                    'cv_folds': self.cv if self.alpha == 'auto' else None,
                    'max_iter': self.max_iter
                },
                fitted_model=self.model,
                r2_score=r2,
                adjusted_r2=adjusted_r2,
                mse=mse,
                mae=mae,
                rmse=rmse,
                residuals=residuals,
                fitted_values=y_pred,
                standardized_residuals=residuals / np.std(residuals),
                feature_names=feature_names,
                coefficients=self.model.coef_,
                feature_importance=feature_importance,
                selected_features=selected_features,
                cv_scores=cv_scores,
                cv_mean=cv_mean,
                cv_std=cv_std,
                n_features=n_features,
                n_samples=n_samples,
                convergence_info=convergence_info
            )
            
            fit_time = time.time() - start_time
            logger.info(f"{self.method.title()} regression fit completed in {fit_time:.3f}s - R² = {r2:.4f}, α = {optimal_alpha:.6f}")
            
        except Exception as e:
            logger.error(f"Error fitting {self.method} regression: {e}")
            raise
            
        return self
    
    def transform(self, X):
        """Transform method returns predictions for pipeline compatibility."""
        check_is_fitted(self)
        return self.predict(X)
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        return self.model.predict(X)
    
    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R²."""
        return self.model.score(X, y, sample_weight=sample_weight)
    
    def get_result(self):
        """Get the comprehensive regression result."""
        check_is_fitted(self)
        return self.result_