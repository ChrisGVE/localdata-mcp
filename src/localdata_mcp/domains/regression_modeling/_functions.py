"""
Regression & Modeling Domain - High-level convenience functions.

Provides simple entry points for common regression analysis tasks
without requiring direct transformer instantiation.
"""

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)

from ._base import logger
from ._pipeline import RegressionModelingPipeline
from ._residuals import ResidualAnalysisTransformer
from ._feature_selection import FeatureSelectionTransformer


# High-level convenience functions
def fit_regression_model(
    data,
    target_column,
    model_type="linear",
    feature_columns=None,
    cross_validation=True,
    residual_analysis=True,
    **kwargs,
):
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
        **kwargs,
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
        "r2_score": r2_score(y_test, y_pred_test),
        "mse": mean_squared_error(y_test, y_pred_test),
        "mae": mean_absolute_error(y_test, y_pred_test),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "explained_variance": explained_variance_score(y_test, y_pred_test),
    }

    evaluation = {
        "test_metrics": test_metrics,
        "test_predictions": y_pred_test.tolist(),
        "test_residuals": (y_test - y_pred_test).tolist(),
    }

    # Training set comparison if provided
    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train)
        train_metrics = {
            "r2_score": r2_score(y_train, y_pred_train),
            "mse": mean_squared_error(y_train, y_pred_train),
            "mae": mean_absolute_error(y_train, y_pred_train),
            "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        }

        evaluation["train_metrics"] = train_metrics
        evaluation["overfitting_check"] = {
            "r2_gap": train_metrics["r2_score"] - test_metrics["r2_score"],
            "mse_ratio": test_metrics["mse"] / train_metrics["mse"],
            "likely_overfitting": train_metrics["r2_score"] - test_metrics["r2_score"]
            > 0.1,
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


def select_features(X, y, method="model_based", feature_names=None, **kwargs):
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
    result["selected_data"] = selector.transform(X)

    return result
