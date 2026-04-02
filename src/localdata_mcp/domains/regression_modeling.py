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


class LogisticRegressionTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for logistic regression analysis.
    
    Implements logistic regression for binary and multiclass classification
    with comprehensive model evaluation and diagnostics.
    
    Parameters:
    -----------
    penalty : str, default='l2'
        Regularization penalty: 'l1', 'l2', 'elasticnet', 'none'
    C : float, default=1.0
        Inverse of regularization strength
    solver : str, default='liblinear'
        Optimization algorithm
    multi_class : str, default='auto'
        Multi-class strategy: 'ovr', 'multinomial', 'auto'
    max_iter : int, default=1000
        Maximum iterations for solver convergence
    """
    
    def __init__(self, penalty='l2', C=1.0, solver='liblinear', multi_class='auto', max_iter=1000):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.multi_class = multi_class
        self.max_iter = max_iter
        
        # Initialize model
        self.model = LogisticRegression(
            penalty=penalty, C=C, solver=solver, 
            multi_class=multi_class, max_iter=max_iter
        )
        self.result_ = None
        
    def fit(self, X, y, sample_weight=None, feature_names=None):
        """
        Fit logistic regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (class labels)
        sample_weight : array-like, shape (n_samples,), optional
            Individual weights for each sample
        feature_names : list of str, optional
            Names of features
            
        Returns:
        --------
        self : LogisticRegressionTransformer
            Returns self for method chaining
        """
        start_time = time.time()
        logger.info("Starting logistic regression fit")
        
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Store feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = feature_names
        
        try:
            # Fit the model
            self.model.fit(X, y, sample_weight=sample_weight)
            
            # Get predictions and probabilities
            y_pred = self.model.predict(X)
            y_proba = self.model.predict_proba(X)
            
            # Calculate metrics (using accuracy as proxy for R² in classification)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import classification_report, confusion_matrix
            
            accuracy = accuracy_score(y, y_pred)
            
            # Handle multi-class averaging
            avg_method = 'macro' if len(np.unique(y)) > 2 else 'binary'
            precision = precision_score(y, y_pred, average=avg_method, zero_division=0)
            recall = recall_score(y, y_pred, average=avg_method, zero_division=0)
            f1 = f1_score(y, y_pred, average=avg_method, zero_division=0)
            
            # Pseudo R² (McFadden's R²)
            try:
                from sklearn.metrics import log_loss
                y_null = np.full_like(y, stats.mode(y)[0][0])  # Most frequent class
                null_log_loss = log_loss(y, self.model.predict_proba(X)[:, [0] * len(y)])
                model_log_loss = log_loss(y, y_proba)
                mcfadden_r2 = 1 - (model_log_loss / null_log_loss)
            except:
                mcfadden_r2 = None
            
            # Cross-validation scores
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
            
            # Feature importance (absolute coefficient values)
            if hasattr(self.model, 'coef_'):
                if self.model.coef_.ndim > 1:
                    # Multi-class: average across classes
                    feature_importance = np.mean(np.abs(self.model.coef_), axis=0)
                    coefficients = self.model.coef_
                else:
                    feature_importance = np.abs(self.model.coef_[0])
                    coefficients = self.model.coef_[0]
            else:
                feature_importance = None
                coefficients = None
            
            # Convergence information
            convergence_info = {
                'converged': getattr(self.model, 'n_iter_', [0])[0] < self.max_iter,
                'n_iterations': getattr(self.model, 'n_iter_', None)
            }
            
            # Create result object
            self.result_ = RegressionModelResult(
                model_type="Logistic Regression",
                model_params={
                    'penalty': self.penalty,
                    'C': self.C,
                    'solver': self.solver,
                    'multi_class': self.multi_class,
                    'max_iter': self.max_iter
                },
                fitted_model=self.model,
                r2_score=accuracy,  # Using accuracy as classification equivalent
                mse=1 - accuracy,   # Error rate as MSE equivalent
                mae=1 - accuracy,
                rmse=np.sqrt(1 - accuracy),
                feature_names=feature_names,
                coefficients=coefficients,
                feature_importance=feature_importance,
                cv_scores=cv_scores.tolist(),
                cv_mean=np.mean(cv_scores),
                cv_std=np.std(cv_scores),
                n_features=X.shape[1],
                n_samples=X.shape[0],
                convergence_info=convergence_info,
                # Store additional classification metrics
                additional_info={
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mcfadden_r2': mcfadden_r2,
                    'classes': self.model.classes_.tolist(),
                    'n_classes': len(self.model.classes_)
                }
            )
            
            fit_time = time.time() - start_time
            logger.info(f"Logistic regression fit completed in {fit_time:.3f}s - Accuracy = {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting logistic regression: {e}")
            raise
            
        return self
    
    def transform(self, X):
        """Transform method returns predicted probabilities for pipeline compatibility."""
        check_is_fitted(self)
        return self.predict_proba(X)
    
    def predict(self, X):
        """Make class predictions using the fitted model."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities using the fitted model."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        return self.model.predict_proba(X)
    
    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels."""
        return self.model.score(X, y, sample_weight=sample_weight)
    
    def get_result(self):
        """Get the comprehensive regression result."""
        check_is_fitted(self)
        return self.result_


class PolynomialRegressionTransformer(BaseEstimator, TransformerMixin, RegressorMixin):
    """
    sklearn-compatible transformer for polynomial regression analysis.
    
    Combines PolynomialFeatures with linear regression to model non-linear
    relationships with comprehensive evaluation and overfitting detection.
    
    Parameters:
    -----------
    degree : int, default=2
        Maximum degree of polynomial features
    interaction_only : bool, default=False
        If True, only interaction features are produced
    include_bias : bool, default=True
        If True, include a bias column (intercept)
    regularization : str or None, default=None
        Regularization method: 'ridge', 'lasso', 'elastic_net', or None
    alpha : float, default=1.0
        Regularization strength (if regularization is used)
    """
    
    def __init__(self, degree=2, interaction_only=False, include_bias=True, 
                 regularization=None, alpha=1.0):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.regularization = regularization
        self.alpha = alpha
        
        # Initialize components
        self.poly_features = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )
        
        # Initialize regression model based on regularization
        if regularization == 'ridge':
            self.regressor = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.regressor = Lasso(alpha=alpha)
        elif regularization == 'elastic_net':
            self.regressor = ElasticNet(alpha=alpha)
        else:
            self.regressor = LinearRegression()
            
        self.result_ = None
        
    def fit(self, X, y, sample_weight=None, feature_names=None):
        """
        Fit polynomial regression model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, shape (n_samples,), optional
            Individual weights for each sample
        feature_names : list of str, optional
            Names of original features
            
        Returns:
        --------
        self : PolynomialRegressionTransformer
            Returns self for method chaining
        """
        start_time = time.time()
        logger.info(f"Starting polynomial regression fit (degree={self.degree})")
        
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store original feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.original_feature_names_ = feature_names
        
        try:
            # Create polynomial features
            X_poly = self.poly_features.fit_transform(X)
            
            # Get polynomial feature names
            poly_feature_names = self.poly_features.get_feature_names_out(feature_names)
            
            # Fit the regression model
            self.regressor.fit(X_poly, y, sample_weight=sample_weight)
            
            # Make predictions
            y_pred = self.regressor.predict(X_poly)
            residuals = y - y_pred
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            # Adjusted R² (accounting for increased features)
            n_samples, n_features = X_poly.shape
            adjusted_r2 = 1 - (1 - r2) * (X.shape[0] - 1) / (X.shape[0] - n_features - 1)
            
            # Cross-validation to detect overfitting
            cv_scores = cross_val_score(
                self.regressor, X_poly, y, cv=5, scoring='r2'
            )
            
            # Overfitting detection
            training_r2 = r2
            cv_mean = np.mean(cv_scores)
            overfitting_gap = training_r2 - cv_mean
            is_overfitting = overfitting_gap > 0.1  # Threshold for overfitting
            
            # Feature importance (based on absolute coefficient values)
            if hasattr(self.regressor, 'coef_'):
                feature_importance = np.abs(self.regressor.coef_)
                coefficients = self.regressor.coef_
            else:
                feature_importance = None
                coefficients = None
            
            # Model complexity analysis
            complexity_info = {
                'original_features': X.shape[1],
                'polynomial_features': X_poly.shape[1],
                'degree': self.degree,
                'feature_expansion_ratio': X_poly.shape[1] / X.shape[1],
                'overfitting_detected': is_overfitting,
                'overfitting_gap': overfitting_gap,
                'regularization_used': self.regularization is not None
            }
            
            # Create result object
            self.result_ = RegressionModelResult(
                model_type=f"Polynomial Regression (degree={self.degree})",
                model_params={
                    'degree': self.degree,
                    'interaction_only': self.interaction_only,
                    'include_bias': self.include_bias,
                    'regularization': self.regularization,
                    'alpha': self.alpha if self.regularization else None
                },
                fitted_model=self.regressor,
                r2_score=r2,
                adjusted_r2=adjusted_r2,
                mse=mse,
                mae=mae,
                rmse=rmse,
                residuals=residuals,
                fitted_values=y_pred,
                standardized_residuals=residuals / np.std(residuals),
                feature_names=poly_feature_names.tolist(),
                coefficients=coefficients,
                feature_importance=feature_importance,
                cv_scores=cv_scores.tolist(),
                cv_mean=cv_mean,
                cv_std=np.std(cv_scores),
                n_features=n_features,
                n_samples=n_samples,
                convergence_info=complexity_info
            )
            
            fit_time = time.time() - start_time
            logger.info(f"Polynomial regression fit completed in {fit_time:.3f}s - R² = {r2:.4f} (CV = {cv_mean:.4f})")
            
            if is_overfitting:
                logger.warning(f"Potential overfitting detected - gap = {overfitting_gap:.3f}")
                
        except Exception as e:
            logger.error(f"Error fitting polynomial regression: {e}")
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
        X_poly = self.poly_features.transform(X)
        return self.regressor.predict(X_poly)
    
    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R²."""
        X = check_array(X, accept_sparse=False)
        X_poly = self.poly_features.transform(X)
        return self.regressor.score(X_poly, y, sample_weight=sample_weight)
    
    def get_result(self):
        """Get the comprehensive regression result."""
        check_is_fitted(self)
        return self.result_


class ResidualAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for comprehensive residual analysis.
    
    Performs detailed residual diagnostics including normality tests,
    homoscedasticity tests, outlier detection, and influence measures.
    
    Parameters:
    -----------
    alpha : float, default=0.05
        Significance level for statistical tests
    outlier_threshold : float, default=2.5
        Threshold for outlier detection (in standard deviations)
    include_influence : bool, default=True
        Whether to compute influence measures (leverage, Cook's distance)
    """
    
    def __init__(self, alpha=0.05, outlier_threshold=2.5, include_influence=True):
        self.alpha = alpha
        self.outlier_threshold = outlier_threshold
        self.include_influence = include_influence
        self.result_ = None
        
    def fit(self, X, y, residuals=None, fitted_values=None):
        """
        Perform comprehensive residual analysis.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Original input data
        y : array-like, shape (n_samples,)
            Original target values
        residuals : array-like, shape (n_samples,), optional
            Model residuals. If None, computed from y - fitted_values
        fitted_values : array-like, shape (n_samples,), optional
            Model predictions. If None, y is used as fitted values
            
        Returns:
        --------
        self : ResidualAnalysisTransformer
            Returns self for method chaining
        """
        start_time = time.time()
        logger.info("Starting residual analysis")
        
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Handle residuals and fitted values
        if fitted_values is None:
            fitted_values = y  # Fallback if not provided
        if residuals is None:
            residuals = y - fitted_values
        
        try:
            # Standardized residuals
            residual_std = np.std(residuals)
            standardized_residuals = residuals / residual_std if residual_std > 0 else residuals
            
            # Basic residual statistics
            residual_stats = {
                'mean': np.mean(residuals),
                'std': residual_std,
                'min': np.min(residuals),
                'max': np.max(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            }
            
            # 1. Normality tests
            normality_test = {}
            
            # Shapiro-Wilk test (for n < 5000)
            if len(residuals) < 5000:
                sw_stat, sw_p = stats.shapiro(residuals)
                normality_test['shapiro_wilk'] = {
                    'statistic': sw_stat,
                    'p_value': sw_p,
                    'is_normal': sw_p > self.alpha
                }
            
            # Anderson-Darling test
            try:
                ad_stat, ad_critical, ad_p = stats.anderson(residuals, dist='norm')
                normality_test['anderson_darling'] = {
                    'statistic': ad_stat,
                    'critical_values': ad_critical.tolist(),
                    'significance_levels': [15, 10, 5, 2.5, 1],
                    'is_normal': ad_stat < ad_critical[2]  # 5% level
                }
            except:
                pass
            
            # Jarque-Bera test
            try:
                jb_stat, jb_p = stats.jarque_bera(residuals)
                normality_test['jarque_bera'] = {
                    'statistic': jb_stat,
                    'p_value': jb_p,
                    'is_normal': jb_p > self.alpha
                }
            except:
                pass
            
            # 2. Homoscedasticity tests
            homoscedasticity_test = {}
            
            # Breusch-Pagan test (requires statsmodels)
            try:
                X_sm = sm.add_constant(X)
                bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, X_sm)
                homoscedasticity_test['breusch_pagan'] = {
                    'lm_statistic': bp_stat,
                    'p_value': bp_p,
                    'f_statistic': bp_f,
                    'f_p_value': bp_fp,
                    'is_homoscedastic': bp_p > self.alpha
                }
            except Exception as e:
                logger.debug(f"Breusch-Pagan test failed: {e}")
            
            # White test
            try:
                X_sm = sm.add_constant(X)
                w_stat, w_p, w_f, w_fp = het_white(residuals, X_sm)
                homoscedasticity_test['white'] = {
                    'lm_statistic': w_stat,
                    'p_value': w_p,
                    'f_statistic': w_f,
                    'f_p_value': w_fp,
                    'is_homoscedastic': w_p > self.alpha
                }
            except Exception as e:
                logger.debug(f"White test failed: {e}")
            
            # 3. Autocorrelation test (Durbin-Watson)
            autocorrelation_test = {}
            try:
                from statsmodels.stats.diagnostic import durbin_watson
                dw_stat = durbin_watson(residuals)
                autocorrelation_test['durbin_watson'] = {
                    'statistic': dw_stat,
                    'interpretation': self._interpret_durbin_watson(dw_stat),
                    'no_autocorrelation': 1.5 <= dw_stat <= 2.5
                }
            except Exception as e:
                logger.debug(f"Durbin-Watson test failed: {e}")
            
            # 4. Outlier detection
            outliers = []
            outlier_threshold_abs = self.outlier_threshold
            
            # Using standardized residuals
            outlier_mask = np.abs(standardized_residuals) > outlier_threshold_abs
            outliers = np.where(outlier_mask)[0].tolist()
            
            # 5. Influence measures (if requested and data allows)
            leverage = None
            cooks_distance = None
            studentized_residuals = None
            
            if self.include_influence:
                try:
                    X_sm = sm.add_constant(X)
                    ols_model = sm.OLS(y, X_sm).fit()
                    influence = OLSInfluence(ols_model)
                    
                    leverage = influence.hat_matrix_diag
                    cooks_distance = influence.cooks_distance[0]
                    studentized_residuals = influence.resid_studentized_external
                    
                except Exception as e:
                    logger.debug(f"Influence measures computation failed: {e}")
            
            # Create result object
            self.result_ = ResidualAnalysisResult(
                residuals=residuals,
                standardized_residuals=standardized_residuals,
                studentized_residuals=studentized_residuals,
                fitted_values=fitted_values,
                normality_test=normality_test,
                homoscedasticity_test=homoscedasticity_test,
                autocorrelation_test=autocorrelation_test,
                outliers=outliers,
                leverage=leverage,
                cooks_distance=cooks_distance,
                residual_stats=residual_stats
            )
            
            fit_time = time.time() - start_time
            logger.info(f"Residual analysis completed in {fit_time:.3f}s - {len(outliers)} outliers detected")
            
        except Exception as e:
            logger.error(f"Error in residual analysis: {e}")
            raise
            
        return self
    
    def transform(self, X):
        """Transform method returns standardized residuals for pipeline compatibility."""
        check_is_fitted(self)
        return self.result_.standardized_residuals.reshape(-1, 1)
    
    def _interpret_durbin_watson(self, dw_stat):
        """Interpret Durbin-Watson statistic."""
        if dw_stat < 1.5:
            return "Positive autocorrelation likely"
        elif dw_stat > 2.5:
            return "Negative autocorrelation likely"
        else:
            return "No strong evidence of autocorrelation"
    
    def get_result(self):
        """Get the comprehensive residual analysis result."""
        check_is_fitted(self)
        return self.result_


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for automated feature selection.
    
    Supports multiple feature selection methods including univariate selection,
    model-based selection, and recursive feature elimination with cross-validation.
    
    Parameters:
    -----------
    method : str, default='model_based'
        Feature selection method: 'model_based', 'rfe', 'rfecv', 'univariate'
    estimator : estimator object, optional
        Estimator for model-based selection or RFE
    k : int or 'all', default='all'
        Number of features to select (for univariate and RFE)
    cv : int, default=5
        Cross-validation folds for RFECV
    scoring : str, default='r2'
        Scoring method for evaluation
    """
    
    def __init__(self, method='model_based', estimator=None, k='all', cv=5, scoring='r2'):
        self.method = method
        self.estimator = estimator
        self.k = k
        self.cv = cv
        self.scoring = scoring
        
        # Default estimator
        if estimator is None:
            self.estimator = LassoCV(cv=cv)
        
        self.selector_ = None
        self.result_ = None
        
    def fit(self, X, y, feature_names=None):
        """
        Perform feature selection.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        feature_names : list of str, optional
            Names of features
            
        Returns:
        --------
        self : FeatureSelectionTransformer
            Returns self for method chaining
        """
        start_time = time.time()
        logger.info(f"Starting feature selection using {self.method}")
        
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = feature_names
        
        try:
            # Initialize selector based on method
            if self.method == 'model_based':
                self.selector_ = SelectFromModel(self.estimator)
            elif self.method == 'rfe':
                n_features = self.k if self.k != 'all' else X.shape[1] // 2
                self.selector_ = RFE(self.estimator, n_features_to_select=n_features)
            elif self.method == 'rfecv':
                self.selector_ = RFECV(
                    self.estimator, cv=self.cv, scoring=self.scoring
                )
            elif self.method == 'univariate':
                from sklearn.feature_selection import SelectKBest, f_regression
                k_best = self.k if self.k != 'all' else X.shape[1]
                self.selector_ = SelectKBest(score_func=f_regression, k=k_best)
            else:
                raise ValueError(f"Unknown feature selection method: {self.method}")
            
            # Fit the selector
            self.selector_.fit(X, y)
            
            # Get selected features
            selected_mask = self.selector_.get_support()
            selected_indices = np.where(selected_mask)[0]
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Transform data to selected features only
            X_selected = self.selector_.transform(X)
            
            # Calculate feature importance/scores
            feature_scores = None
            feature_importance = None
            
            if hasattr(self.selector_, 'scores_'):
                # Univariate selection
                feature_scores = self.selector_.scores_
                feature_importance = feature_scores[selected_mask]
            elif hasattr(self.selector_, 'estimator_') and hasattr(self.selector_.estimator_, 'coef_'):
                # Model-based selection
                feature_importance = np.abs(self.selector_.estimator_.coef_)
            elif hasattr(self.selector_, 'ranking_'):
                # RFE/RFECV - convert ranking to importance (inverse ranking)
                max_rank = np.max(self.selector_.ranking_)
                feature_importance = (max_rank - self.selector_.ranking_ + 1)[selected_mask]
            
            # Performance comparison
            comparison_results = {}
            if self.method in ['rfe', 'rfecv', 'model_based']:
                try:
                    # Fit simple linear regression on original and selected features
                    lr_original = LinearRegression().fit(X, y)
                    lr_selected = LinearRegression().fit(X_selected, y)
                    
                    r2_original = lr_original.score(X, y)
                    r2_selected = lr_selected.score(X_selected, y)
                    
                    comparison_results = {
                        'original_features': X.shape[1],
                        'selected_features': X_selected.shape[1],
                        'feature_reduction': 1 - (X_selected.shape[1] / X.shape[1]),
                        'r2_original': r2_original,
                        'r2_selected': r2_selected,
                        'r2_change': r2_selected - r2_original,
                        'performance_per_feature': r2_selected / X_selected.shape[1]
                    }
                except:
                    pass
            
            # Store results
            self.result_ = {
                'method': self.method,
                'selected_features': selected_features,
                'selected_indices': selected_indices.tolist(),
                'n_selected': len(selected_features),
                'n_original': len(feature_names),
                'selection_ratio': len(selected_features) / len(feature_names),
                'feature_importance': feature_importance.tolist() if feature_importance is not None else None,
                'feature_scores': feature_scores.tolist() if feature_scores is not None else None,
                'comparison': comparison_results
            }
            
            # Additional info for specific methods
            if self.method == 'rfecv' and hasattr(self.selector_, 'grid_scores_'):
                self.result_['cv_scores'] = self.selector_.grid_scores_.tolist()
                self.result_['optimal_n_features'] = self.selector_.n_features_
            
            fit_time = time.time() - start_time
            logger.info(f"Feature selection completed in {fit_time:.3f}s - {len(selected_features)}/{len(feature_names)} features selected")
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise
            
        return self
    
    def transform(self, X):
        """Transform data using selected features."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        return self.selector_.transform(X)
    
    def get_support(self, indices=False):
        """Get a mask or integer index of the selected features."""
        check_is_fitted(self)
        return self.selector_.get_support(indices=indices)
    
    def get_result(self):
        """Get the comprehensive feature selection result."""
        check_is_fitted(self)
        return self.result_


class RegressionModelingPipeline(AnalysisPipelineBase):
    """
    Complete pipeline for regression analysis and modeling.
    
    Provides high-level interface for fitting, evaluating, and diagnosing
    regression models with comprehensive preprocessing and result formatting.
    
    Parameters:
    -----------
    model_type : str, default='linear'
        Type of regression model: 'linear', 'ridge', 'lasso', 'elastic_net',
        'logistic', 'polynomial'
    preprocessing : str, default='auto'
        Preprocessing level: 'minimal', 'auto', 'comprehensive', 'custom'
    cross_validation : bool, default=True
        Whether to perform cross-validation
    residual_analysis : bool, default=True
        Whether to perform residual diagnostics
    feature_selection : bool, default=False
        Whether to perform automatic feature selection
    """
    
    def __init__(self, model_type='linear', preprocessing='auto', cross_validation=True,
                 residual_analysis=True, feature_selection=False, **kwargs):
        super().__init__()
        self.model_type = model_type
        self.preprocessing_level = preprocessing
        self.cross_validation = cross_validation
        self.residual_analysis = residual_analysis
        self.feature_selection = feature_selection
        self.model_kwargs = kwargs
        
        # Initialize transformers
        self.regressor_ = None
        self.feature_selector_ = None
        self.residual_analyzer_ = None
        
        # Results storage
        self.regression_result_ = None
        self.feature_selection_result_ = None
        self.residual_analysis_result_ = None
        
    def fit(self, X, y, feature_names=None, **fit_params):
        """
        Fit the complete regression modeling pipeline.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        feature_names : list of str, optional
            Names of features
        **fit_params : dict
            Additional parameters passed to model fitting
            
        Returns:
        --------
        self : RegressionModelingPipeline
            Returns self for method chaining
        """
        start_time = time.time()
        logger.info(f"Starting regression modeling pipeline - {self.model_type}")
        
        try:
            # Validate inputs
            X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            # Step 1: Feature selection (if requested)
            X_processed = X.copy()
            if self.feature_selection:
                self.feature_selector_ = FeatureSelectionTransformer(**self.model_kwargs.get('feature_selection_params', {}))
                self.feature_selector_.fit(X_processed, y, feature_names=feature_names)
                X_processed = self.feature_selector_.transform(X_processed)
                
                # Update feature names to selected ones
                selected_features = self.feature_selector_.get_result()['selected_features']
                feature_names = selected_features
                
                self.feature_selection_result_ = self.feature_selector_.get_result()
                logger.info(f"Feature selection completed - {len(selected_features)} features selected")
            
            # Step 2: Fit regression model
            if self.model_type == 'linear':
                self.regressor_ = LinearRegressionTransformer(**self.model_kwargs.get('model_params', {}))
            elif self.model_type in ['ridge', 'lasso', 'elastic_net']:
                model_params = self.model_kwargs.get('model_params', {})
                model_params['method'] = self.model_type
                self.regressor_ = RegularizedRegressionTransformer(**model_params)
            elif self.model_type == 'logistic':
                self.regressor_ = LogisticRegressionTransformer(**self.model_kwargs.get('model_params', {}))
            elif self.model_type == 'polynomial':
                self.regressor_ = PolynomialRegressionTransformer(**self.model_kwargs.get('model_params', {}))
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Fit the model
            self.regressor_.fit(X_processed, y, feature_names=feature_names, **fit_params)
            self.regression_result_ = self.regressor_.get_result()
            
            # Step 3: Residual analysis (if requested and applicable)
            if self.residual_analysis and self.model_type != 'logistic':
                fitted_values = self.regressor_.predict(X_processed)
                residuals = y - fitted_values
                
                self.residual_analyzer_ = ResidualAnalysisTransformer(**self.model_kwargs.get('residual_params', {}))
                self.residual_analyzer_.fit(
                    X_processed, y, 
                    residuals=residuals, 
                    fitted_values=fitted_values
                )
                self.residual_analysis_result_ = self.residual_analyzer_.get_result()
                logger.info("Residual analysis completed")
            
            # Update pipeline state
            self.state = PipelineState.FITTED
            
            fit_time = time.time() - start_time
            logger.info(f"Regression modeling pipeline completed in {fit_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error in regression modeling pipeline: {e}")
            self.state = PipelineState.ERROR
            raise
            
        return self
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        check_is_fitted(self)
        
        # Apply feature selection if used
        X_processed = X.copy()
        if self.feature_selection and self.feature_selector_ is not None:
            X_processed = self.feature_selector_.transform(X_processed)
            
        return self.regressor_.predict(X_processed)
    
    def score(self, X, y):
        """Score the model on test data."""
        check_is_fitted(self)
        
        # Apply feature selection if used
        X_processed = X.copy()
        if self.feature_selection and self.feature_selector_ is not None:
            X_processed = self.feature_selector_.transform(X_processed)
            
        return self.regressor_.score(X_processed, y)
    
    def get_results(self):
        """
        Get comprehensive results from the regression modeling pipeline.
        
        Returns:
        --------
        results : dict
            Dictionary containing all analysis results
        """
        check_is_fitted(self)
        
        results = {
            'model_type': self.model_type,
            'pipeline_config': {
                'preprocessing_level': self.preprocessing_level,
                'cross_validation': self.cross_validation,
                'residual_analysis': self.residual_analysis,
                'feature_selection': self.feature_selection
            }
        }
        
        # Add regression results
        if self.regression_result_:
            results['regression_analysis'] = self.regression_result_.to_dict()
            
        # Add feature selection results
        if self.feature_selection_result_:
            results['feature_selection'] = self.feature_selection_result_
            
        # Add residual analysis results
        if self.residual_analysis_result_:
            results['residual_analysis'] = self.residual_analysis_result_.to_dict()
            
        return results
    
    def get_composition_metadata(self):
        """Get metadata for pipeline composition."""
        return CompositionMetadata(
            domain="regression_modeling",
            analysis_type=self.model_type,
            result_type="model_results",
            compatible_tools=[
                "statistical_analysis",
                "data_profiling", 
                "visualization",
                "model_evaluation"
            ],
            suggested_compositions=[
                {
                    "tool": "statistical_analysis",
                    "purpose": "validate model assumptions",
                    "inputs": ["residuals", "feature_correlations"]
                },
                {
                    "tool": "visualization", 
                    "purpose": "model diagnostics plots",
                    "inputs": ["fitted_vs_residuals", "qq_plot", "feature_importance"]
                }
            ],
            data_artifacts={
                "fitted_model": self.regressor_,
                "residuals": getattr(self.regression_result_, 'residuals', None),
                "feature_importance": getattr(self.regression_result_, 'feature_importance', None),
                "selected_features": self.feature_selection_result_.get('selected_features') if self.feature_selection_result_ else None
            }
        )


# High-level convenience functions
def fit_regression_model(data, target_column, model_type='linear', feature_columns=None,
                        cross_validation=True, residual_analysis=True, **kwargs):
    """
    High-level function to fit a regression model with comprehensive analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    target_column : str
        Name of the target column
    model_type : str, default='linear'
        Type of regression model
    feature_columns : list of str, optional
        List of feature column names. If None, uses all except target
    cross_validation : bool, default=True
        Whether to perform cross-validation
    residual_analysis : bool, default=True
        Whether to perform residual analysis
    **kwargs : dict
        Additional parameters for model configuration
        
    Returns:
    --------
    results : dict
        Comprehensive analysis results
    """
    logger.info(f"Fitting {model_type} regression model")
    
    # Prepare data
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column]
    
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Create and fit pipeline
    pipeline = RegressionModelingPipeline(
        model_type=model_type,
        cross_validation=cross_validation,
        residual_analysis=residual_analysis,
        **kwargs
    )
    
    pipeline.fit(X, y, feature_names=feature_columns)
    
    return pipeline.get_results()


def evaluate_model_performance(model, X_test, y_test, X_train=None, y_train=None):
    """
    Comprehensive model performance evaluation.
    
    Parameters:
    -----------
    model : fitted estimator
        Trained regression model
    X_test : array-like
        Test features
    y_test : array-like
        Test targets
    X_train : array-like, optional
        Training features for comparison
    y_train : array-like, optional
        Training targets for comparison
        
    Returns:
    --------
    evaluation : dict
        Performance metrics and diagnostics
    """
    logger.info("Evaluating model performance")
    
    # Test set predictions
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    test_metrics = {
        'r2_score': r2_score(y_test, y_pred_test),
        'mse': mean_squared_error(y_test, y_pred_test),
        'mae': mean_absolute_error(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'explained_variance': explained_variance_score(y_test, y_pred_test)
    }
    
    evaluation = {
        'test_metrics': test_metrics,
        'test_predictions': y_pred_test.tolist(),
        'test_residuals': (y_test - y_pred_test).tolist()
    }
    
    # Training set comparison if provided
    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train)
        train_metrics = {
            'r2_score': r2_score(y_train, y_pred_train),
            'mse': mean_squared_error(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
        }
        
        evaluation['train_metrics'] = train_metrics
        evaluation['overfitting_check'] = {
            'r2_gap': train_metrics['r2_score'] - test_metrics['r2_score'],
            'mse_ratio': test_metrics['mse'] / train_metrics['mse'],
            'likely_overfitting': train_metrics['r2_score'] - test_metrics['r2_score'] > 0.1
        }
    
    return evaluation


def analyze_residuals(model, X, y, feature_names=None, **kwargs):
    """
    Perform comprehensive residual analysis for a fitted model.
    
    Parameters:
    -----------
    model : fitted estimator
        Trained regression model
    X : array-like
        Feature data
    y : array-like
        Target data
    feature_names : list of str, optional
        Names of features
    **kwargs : dict
        Additional parameters for residual analysis
        
    Returns:
    --------
    analysis : dict
        Residual analysis results
    """
    logger.info("Performing residual analysis")
    
    # Make predictions and calculate residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Perform analysis
    analyzer = ResidualAnalysisTransformer(**kwargs)
    analyzer.fit(X, y, residuals=residuals, fitted_values=y_pred)
    
    return analyzer.get_result().to_dict()


def select_features(X, y, method='model_based', feature_names=None, **kwargs):
    """
    Perform feature selection for regression analysis.
    
    Parameters:
    -----------
    X : array-like
        Feature data
    y : array-like
        Target data
    method : str, default='model_based'
        Feature selection method
    feature_names : list of str, optional
        Names of features
    **kwargs : dict
        Additional parameters for feature selection
        
    Returns:
    --------
    selection_result : dict
        Feature selection results and selected features
    """
    logger.info(f"Performing feature selection using {method}")
    
    # Perform selection
    selector = FeatureSelectionTransformer(method=method, **kwargs)
    selector.fit(X, y, feature_names=feature_names)
    
    result = selector.get_result()
    result['selected_data'] = selector.transform(X)
    
    return result