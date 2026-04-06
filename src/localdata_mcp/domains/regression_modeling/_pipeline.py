"""
Regression & Modeling Domain - Regression Modeling Pipeline.

Provides high-level interface for fitting, evaluating, and diagnosing
regression models with comprehensive preprocessing and result formatting.
"""

import time

from sklearn.utils.validation import check_is_fitted, check_X_y

from ._base import AnalysisPipelineBase, CompositionMetadata, PipelineState, logger
from ._feature_selection import FeatureSelectionTransformer
from ._linear import LinearRegressionTransformer
from ._logistic import LogisticRegressionTransformer
from ._polynomial import PolynomialRegressionTransformer
from ._regularized import RegularizedRegressionTransformer
from ._residuals import ResidualAnalysisTransformer


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

    def __init__(
        self,
        model_type="linear",
        preprocessing="auto",
        cross_validation=True,
        residual_analysis=True,
        feature_selection=False,
        **kwargs,
    ):
        super().__init__(analytical_intention=f"{model_type} regression analysis")
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

    def _apply_feature_selection(self, X, y, feature_names):
        """Run feature selection if enabled; return (X_processed, feature_names)."""
        if not self.feature_selection:
            return X, feature_names
        self.feature_selector_ = FeatureSelectionTransformer(
            **self.model_kwargs.get("feature_selection_params", {})
        )
        self.feature_selector_.fit(X, y, feature_names=feature_names)
        X_processed = self.feature_selector_.transform(X)
        selected_features = self.feature_selector_.get_result()["selected_features"]
        self.feature_selection_result_ = self.feature_selector_.get_result()
        logger.info(
            f"Feature selection completed - {len(selected_features)} features selected"
        )
        return X_processed, selected_features

    def _instantiate_regressor(self):
        """Instantiate and return the configured regressor."""
        if self.model_type == "linear":
            return LinearRegressionTransformer(
                **self.model_kwargs.get("model_params", {})
            )
        elif self.model_type in ["ridge", "lasso", "elastic_net"]:
            model_params = self.model_kwargs.get("model_params", {})
            model_params["method"] = self.model_type
            return RegularizedRegressionTransformer(**model_params)
        elif self.model_type == "logistic":
            return LogisticRegressionTransformer(
                **self.model_kwargs.get("model_params", {})
            )
        elif self.model_type == "polynomial":
            return PolynomialRegressionTransformer(
                **self.model_kwargs.get("model_params", {})
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _run_residual_analysis(self, X_processed, y):
        """Fit residual analyzer if enabled and model is not logistic."""
        if not self.residual_analysis or self.model_type == "logistic":
            return
        fitted_values = self.regressor_.predict(X_processed)
        residuals = y - fitted_values
        self.residual_analyzer_ = ResidualAnalysisTransformer(
            **self.model_kwargs.get("residual_params", {})
        )
        self.residual_analyzer_.fit(
            X_processed, y, residuals=residuals, fitted_values=fitted_values
        )
        self.residual_analysis_result_ = self.residual_analyzer_.get_result()
        logger.info("Residual analysis completed")

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
            X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            X_processed, feature_names = self._apply_feature_selection(
                X.copy(), y, feature_names
            )

            self.regressor_ = self._instantiate_regressor()
            self.regressor_.fit(
                X_processed, y, feature_names=feature_names, **fit_params
            )
            self.regression_result_ = self.regressor_.get_result()

            self._run_residual_analysis(X_processed, y)

            self._state = PipelineState.FITTED
            fit_time = time.time() - start_time
            logger.info(f"Regression modeling pipeline completed in {fit_time:.3f}s")

        except Exception as e:
            logger.error(f"Error in regression modeling pipeline: {e}")
            self._state = PipelineState.ERROR
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

    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        return f"{self.model_type}_regression"

    def _configure_analysis_pipeline(self):
        """Configure analysis steps based on intention and complexity level."""
        steps = [self.fit]
        if self.residual_analysis:
            steps.append(lambda X, y: self.residual_analyzer_)
        return steps

    def _execute_analysis_step(self, step, data, context):
        """Execute individual analysis step with error handling and metadata."""
        result = step(data, context) if callable(step) else step
        return result, {}

    def _execute_streaming_analysis(self, data):
        """Execute analysis with streaming support for large datasets."""
        # Regression modeling currently processes in-memory
        return self._execute_standard_analysis(data)

    def _execute_standard_analysis(self, data):
        """Execute analysis on full dataset in memory."""
        # Primary analysis is handled by the fit method
        return self.regression_result_, {}

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
            "model_type": self.model_type,
            "pipeline_config": {
                "preprocessing_level": self.preprocessing_level,
                "cross_validation": self.cross_validation,
                "residual_analysis": self.residual_analysis,
                "feature_selection": self.feature_selection,
            },
        }

        # Add regression results
        if self.regression_result_:
            results["regression_analysis"] = self.regression_result_.to_dict()

        # Add feature selection results
        if self.feature_selection_result_:
            results["feature_selection"] = self.feature_selection_result_

        # Add residual analysis results
        if self.residual_analysis_result_:
            results["residual_analysis"] = self.residual_analysis_result_.to_dict()

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
                "model_evaluation",
            ],
            suggested_compositions=[
                {
                    "tool": "statistical_analysis",
                    "purpose": "validate model assumptions",
                    "inputs": ["residuals", "feature_correlations"],
                },
                {
                    "tool": "visualization",
                    "purpose": "model diagnostics plots",
                    "inputs": ["fitted_vs_residuals", "qq_plot", "feature_importance"],
                },
            ],
            data_artifacts={
                "fitted_model": self.regressor_,
                "residuals": getattr(self.regression_result_, "residuals", None),
                "feature_importance": getattr(
                    self.regression_result_, "feature_importance", None
                ),
                "selected_features": (
                    self.feature_selection_result_.get("selected_features")
                    if self.feature_selection_result_
                    else None
                ),
            },
        )
