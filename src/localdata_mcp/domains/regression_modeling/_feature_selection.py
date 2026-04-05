"""
Regression & Modeling Domain - Feature Selection Transformer.

Supports multiple feature selection methods including univariate selection,
model-based selection, and recursive feature elimination with cross-validation.
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectFromModel, RFE, RFECV

from ._base import logger


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

    def __init__(
        self, method="model_based", estimator=None, k="all", cv=5, scoring="r2"
    ):
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
            if self.method == "model_based":
                self.selector_ = SelectFromModel(self.estimator)
            elif self.method == "rfe":
                n_features = self.k if self.k != "all" else X.shape[1] // 2
                self.selector_ = RFE(self.estimator, n_features_to_select=n_features)
            elif self.method == "rfecv":
                self.selector_ = RFECV(self.estimator, cv=self.cv, scoring=self.scoring)
            elif self.method == "univariate":
                from sklearn.feature_selection import SelectKBest, f_regression

                k_best = self.k if self.k != "all" else X.shape[1]
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

            if hasattr(self.selector_, "scores_"):
                # Univariate selection
                feature_scores = self.selector_.scores_
                feature_importance = feature_scores[selected_mask]
            elif hasattr(self.selector_, "ranking_"):
                # RFE/RFECV - convert ranking to importance (inverse ranking)
                max_rank = np.max(self.selector_.ranking_)
                feature_importance = (max_rank - self.selector_.ranking_ + 1)[
                    selected_mask
                ]
            elif hasattr(self.selector_, "estimator_") and hasattr(
                self.selector_.estimator_, "coef_"
            ):
                # Model-based selection (SelectFromModel)
                # estimator_.coef_ has all features for SelectFromModel
                coef = np.abs(self.selector_.estimator_.coef_)
                if len(coef) == len(selected_mask):
                    feature_importance = coef[selected_mask]
                else:
                    # Estimator was fitted on selected features only
                    feature_importance = coef

            # Performance comparison
            comparison_results = {}
            if self.method in ["rfe", "rfecv", "model_based"]:
                try:
                    # Fit simple linear regression on original and selected features
                    lr_original = LinearRegression().fit(X, y)
                    lr_selected = LinearRegression().fit(X_selected, y)

                    r2_original = lr_original.score(X, y)
                    r2_selected = lr_selected.score(X_selected, y)

                    comparison_results = {
                        "original_features": X.shape[1],
                        "selected_features": X_selected.shape[1],
                        "feature_reduction": 1 - (X_selected.shape[1] / X.shape[1]),
                        "r2_original": r2_original,
                        "r2_selected": r2_selected,
                        "r2_change": r2_selected - r2_original,
                        "performance_per_feature": r2_selected / X_selected.shape[1],
                    }
                except:
                    pass

            # Store results
            self.result_ = {
                "method": self.method,
                "selected_features": selected_features,
                "selected_indices": selected_indices.tolist(),
                "n_selected": len(selected_features),
                "n_original": len(feature_names),
                "selection_ratio": len(selected_features) / len(feature_names),
                "feature_importance": feature_importance.tolist()
                if feature_importance is not None
                else None,
                "feature_scores": feature_scores.tolist()
                if feature_scores is not None
                else None,
                "comparison": comparison_results,
            }

            # Additional info for specific methods
            if self.method == "rfecv" and hasattr(self.selector_, "grid_scores_"):
                self.result_["cv_scores"] = self.selector_.grid_scores_.tolist()
                self.result_["optimal_n_features"] = self.selector_.n_features_

            fit_time = time.time() - start_time
            logger.info(
                f"Feature selection completed in {fit_time:.3f}s - {len(selected_features)}/{len(feature_names)} features selected"
            )

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
