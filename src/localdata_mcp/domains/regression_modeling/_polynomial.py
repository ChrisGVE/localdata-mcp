"""
Regression & Modeling Domain - Polynomial Regression Transformer.

Combines PolynomialFeatures with linear regression to model non-linear
relationships with comprehensive evaluation and overfitting detection.
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._base import RegressionModelResult, logger


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

    def __init__(
        self,
        degree=2,
        interaction_only=False,
        include_bias=True,
        regularization=None,
        alpha=1.0,
    ):
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
        if regularization == "ridge":
            self.regressor = Ridge(alpha=alpha)
        elif regularization == "lasso":
            self.regressor = Lasso(alpha=alpha)
        elif regularization == "elastic_net":
            self.regressor = ElasticNet(alpha=alpha)
        else:
            self.regressor = LinearRegression()

        self.result_ = None

    def _compute_poly_metrics(self, X, X_poly, y, y_pred):
        """Compute regression metrics and overfitting diagnostics for polynomial model."""
        residuals = y - y_pred
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mse)
        n_samples, n_features = X_poly.shape
        adjusted_r2 = 1 - (1 - r2) * (X.shape[0] - 1) / (X.shape[0] - n_features - 1)
        cv_scores = cross_val_score(self.regressor, X_poly, y, cv=5, scoring="r2")
        cv_mean = np.mean(cv_scores)
        overfitting_gap = r2 - cv_mean
        return (
            residuals,
            r2,
            mse,
            mae,
            rmse,
            adjusted_r2,
            cv_scores,
            cv_mean,
            overfitting_gap,
        )

    def _build_complexity_info(self, X, X_poly, overfitting_gap):
        """Build a complexity info dict for the polynomial model."""
        is_overfitting = overfitting_gap > 0.1
        return is_overfitting, {
            "original_features": X.shape[1],
            "polynomial_features": X_poly.shape[1],
            "degree": self.degree,
            "feature_expansion_ratio": X_poly.shape[1] / X.shape[1],
            "overfitting_detected": is_overfitting,
            "overfitting_gap": overfitting_gap,
            "regularization_used": self.regularization is not None,
        }

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

        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.original_feature_names_ = feature_names

        try:
            X_poly = self.poly_features.fit_transform(X)
            poly_feature_names = self.poly_features.get_feature_names_out(feature_names)
            self.regressor.fit(X_poly, y, sample_weight=sample_weight)

            y_pred = self.regressor.predict(X_poly)
            (
                residuals,
                r2,
                mse,
                mae,
                rmse,
                adjusted_r2,
                cv_scores,
                cv_mean,
                overfitting_gap,
            ) = self._compute_poly_metrics(X, X_poly, y, y_pred)
            is_overfitting, complexity_info = self._build_complexity_info(
                X, X_poly, overfitting_gap
            )

            n_samples, n_features = X_poly.shape
            coefficients = getattr(self.regressor, "coef_", None)
            feature_importance = (
                np.abs(coefficients) if coefficients is not None else None
            )

            self.result_ = RegressionModelResult(
                model_type=f"Polynomial Regression (degree={self.degree})",
                model_params={
                    "degree": self.degree,
                    "interaction_only": self.interaction_only,
                    "include_bias": self.include_bias,
                    "regularization": self.regularization,
                    "alpha": self.alpha if self.regularization else None,
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
                convergence_info=complexity_info,
            )

            fit_time = time.time() - start_time
            logger.info(
                f"Polynomial regression fit completed in {fit_time:.3f}s"
                f" - R² = {r2:.4f} (CV = {cv_mean:.4f})"
            )
            if is_overfitting:
                logger.warning(
                    f"Potential overfitting detected - gap = {overfitting_gap:.3f}"
                )

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
