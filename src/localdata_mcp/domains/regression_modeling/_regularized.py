"""
Regression & Modeling Domain - Regularized Regression Transformer.

Supports Ridge, Lasso, and Elastic Net regression with automatic
hyperparameter tuning via cross-validation and comprehensive evaluation.
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    RidgeCV,
    LassoCV,
    ElasticNetCV,
)

from ._base import RegressionModelResult, logger


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

    def __init__(self, method="ridge", alpha="auto", l1_ratio=0.5, cv=5, max_iter=1000):
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.max_iter = max_iter

        self.model = None
        self.result_ = None

    def _build_model_instance(self):
        """Instantiate the appropriate regularized regression model."""
        if self.alpha == "auto":
            if self.method == "ridge":
                return RidgeCV(cv=self.cv)
            elif self.method == "lasso":
                return LassoCV(cv=self.cv, max_iter=self.max_iter)
            elif self.method == "elastic_net":
                return ElasticNetCV(
                    cv=self.cv, l1_ratio=self.l1_ratio, max_iter=self.max_iter
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")
        else:
            if self.method == "ridge":
                return Ridge(alpha=self.alpha)
            elif self.method == "lasso":
                return Lasso(alpha=self.alpha, max_iter=self.max_iter)
            elif self.method == "elastic_net":
                return ElasticNet(
                    alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")

    def _compute_cv_scores(self, X, y):
        """Compute cross-validation scores if alpha was auto-tuned."""
        if self.alpha != "auto":
            return None, None, None
        cv_scores = cross_val_score(self.model, X, y, cv=self.cv, scoring="r2")
        return cv_scores, np.mean(cv_scores), np.std(cv_scores)

    def _extract_feature_info(self, feature_names):
        """Extract selected features and importance from fitted model."""
        selected_features = None
        feature_importance = None
        if hasattr(self.model, "coef_"):
            if self.method == "lasso":
                selected_mask = np.abs(self.model.coef_) > 1e-10
                selected_features = [
                    feature_names[i] for i, sel in enumerate(selected_mask) if sel
                ]
            feature_importance = np.abs(self.model.coef_)
        return selected_features, feature_importance

    def _build_convergence_info(self, optimal_alpha):
        """Build convergence information dictionary."""
        convergence_info = {}
        n_iter = getattr(self.model, "n_iter_", None)
        if n_iter is not None:
            convergence_info["n_iterations"] = n_iter
            convergence_info["converged"] = n_iter < self.max_iter
        if self.alpha == "auto":
            convergence_info["optimal_alpha"] = optimal_alpha
            if hasattr(self.model, "alphas_"):
                convergence_info["alphas_tested"] = len(self.model.alphas_)
        return convergence_info

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

        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = feature_names

        try:
            self.model = self._build_model_instance()
            self.model.fit(X, y, sample_weight=sample_weight)

            optimal_alpha = getattr(self.model, "alpha_", self.alpha)
            y_pred = self.model.predict(X)
            residuals = y - y_pred

            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mse)
            n_samples, n_features = X.shape
            adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

            cv_scores, cv_mean, cv_std = self._compute_cv_scores(X, y)
            selected_features, feature_importance = self._extract_feature_info(
                feature_names
            )
            convergence_info = self._build_convergence_info(optimal_alpha)

            self.result_ = RegressionModelResult(
                model_type=f"{self.method.title()} Regression",
                model_params={
                    "method": self.method,
                    "alpha": optimal_alpha,
                    "l1_ratio": self.l1_ratio if self.method == "elastic_net" else None,
                    "cv_folds": self.cv if self.alpha == "auto" else None,
                    "max_iter": self.max_iter,
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
                convergence_info=convergence_info,
            )

            fit_time = time.time() - start_time
            logger.info(
                f"{self.method.title()} regression fit completed in {fit_time:.3f}s"
                f" - R² = {r2:.4f}, α = {optimal_alpha:.6f}"
            )

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
