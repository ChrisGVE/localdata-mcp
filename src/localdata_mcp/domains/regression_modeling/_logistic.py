"""
Regression & Modeling Domain - Logistic Regression Transformer.

Implements logistic regression for binary and multiclass classification
with comprehensive model evaluation and diagnostics.
"""

import time

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from ._base import RegressionModelResult, logger


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

    def __init__(
        self, penalty="l2", C=1.0, solver="liblinear", multi_class="auto", max_iter=1000
    ):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.multi_class = multi_class
        self.max_iter = max_iter

        # Initialize model
        self.model = LogisticRegression(
            penalty=penalty, C=C, solver=solver, max_iter=max_iter
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
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
            )
            from sklearn.metrics import classification_report, confusion_matrix

            accuracy = accuracy_score(y, y_pred)

            # Handle multi-class averaging
            avg_method = "macro" if len(np.unique(y)) > 2 else "binary"
            precision = precision_score(y, y_pred, average=avg_method, zero_division=0)
            recall = recall_score(y, y_pred, average=avg_method, zero_division=0)
            f1 = f1_score(y, y_pred, average=avg_method, zero_division=0)

            # Pseudo R² (McFadden's R²)
            try:
                from sklearn.metrics import log_loss

                y_null = np.full_like(y, stats.mode(y)[0][0])  # Most frequent class
                null_log_loss = log_loss(
                    y, self.model.predict_proba(X)[:, [0] * len(y)]
                )
                model_log_loss = log_loss(y, y_proba)
                mcfadden_r2 = 1 - (model_log_loss / null_log_loss)
            except:
                mcfadden_r2 = None

            # Cross-validation scores
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")

            # Feature importance (absolute coefficient values)
            if hasattr(self.model, "coef_"):
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
                "converged": getattr(self.model, "n_iter_", [0])[0] < self.max_iter,
                "n_iterations": getattr(self.model, "n_iter_", None),
            }

            # Create result object
            self.result_ = RegressionModelResult(
                model_type="Logistic Regression",
                model_params={
                    "penalty": self.penalty,
                    "C": self.C,
                    "solver": self.solver,
                    "multi_class": self.multi_class,
                    "max_iter": self.max_iter,
                },
                fitted_model=self.model,
                r2_score=accuracy,  # Using accuracy as classification equivalent
                mse=1 - accuracy,  # Error rate as MSE equivalent
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
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "mcfadden_r2": mcfadden_r2,
                    "classes": self.model.classes_.tolist(),
                    "n_classes": len(self.model.classes_),
                },
            )

            fit_time = time.time() - start_time
            logger.info(
                f"Logistic regression fit completed in {fit_time:.3f}s - Accuracy = {accuracy:.4f}"
            )

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
