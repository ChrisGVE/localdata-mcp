"""
Time Series Analysis - Forecast evaluation.

Contains the ForecastEvaluator class that provides comprehensive evaluation
metrics for comparing forecasted values against actual observations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ...pipeline.base import PipelineResult


class ForecastEvaluator(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for evaluating forecast accuracy.

    Provides comprehensive evaluation metrics for comparing forecasted values
    against actual observations, including MAE, MAPE, RMSE, and other
    time series specific metrics.

    Parameters:
    -----------
    metrics : list, default=['mae', 'mape', 'rmse', 'mase']
        List of metrics to compute
    seasonal_period : int, optional
        Seasonal period for MASE calculation
    """

    def __init__(self, metrics=None, seasonal_period=None):
        self.metrics = metrics or ["mae", "mape", "rmse", "mase"]
        self.seasonal_period = seasonal_period

    def fit(self, X, y=None):
        """ForecastEvaluator doesn't require fitting."""
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> PipelineResult:
        """This method is not used - use evaluate_forecast instead."""
        raise NotImplementedError("Use evaluate_forecast method instead")

    def evaluate_forecast(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
        historical_data: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate forecast accuracy using multiple metrics.

        Parameters:
        -----------
        actual : pd.Series or np.ndarray
            Actual observed values
        predicted : pd.Series or np.ndarray
            Predicted/forecasted values
        historical_data : pd.Series or np.ndarray, optional
            Historical data for MASE calculation

        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Convert to numpy arrays for calculation
        actual_array = actual.values if hasattr(actual, "values") else np.array(actual)
        predicted_array = (
            predicted.values if hasattr(predicted, "values") else np.array(predicted)
        )

        # Ensure same length
        min_length = min(len(actual_array), len(predicted_array))
        actual_array = actual_array[:min_length]
        predicted_array = predicted_array[:min_length]

        if len(actual_array) == 0:
            raise ValueError("No data points to evaluate")

        results = {}

        # Calculate requested metrics
        for metric in self.metrics:
            if metric == "mae":
                results["mae"] = self._calculate_mae(actual_array, predicted_array)
            elif metric == "mape":
                results["mape"] = self._calculate_mape(actual_array, predicted_array)
            elif metric == "rmse":
                results["rmse"] = self._calculate_rmse(actual_array, predicted_array)
            elif metric == "mase":
                results["mase"] = self._calculate_mase(
                    actual_array, predicted_array, historical_data
                )
            elif metric == "smape":
                results["smape"] = self._calculate_smape(actual_array, predicted_array)
            elif metric == "mse":
                results["mse"] = self._calculate_mse(actual_array, predicted_array)
            elif metric == "r2":
                results["r2"] = self._calculate_r2(actual_array, predicted_array)
            elif metric == "directional_accuracy":
                results["directional_accuracy"] = self._calculate_directional_accuracy(
                    actual_array, predicted_array
                )

        # Add additional summary statistics
        results["forecast_bias"] = np.mean(predicted_array - actual_array)
        results["mean_actual"] = np.mean(actual_array)
        results["mean_predicted"] = np.mean(predicted_array)
        results["std_actual"] = np.std(actual_array)
        results["std_predicted"] = np.std(predicted_array)
        results["n_observations"] = len(actual_array)

        return results

    def _calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.

        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values

        Returns:
        --------
        mae : float
            Mean Absolute Error
        """
        return float(np.mean(np.abs(actual - predicted)))

    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values

        Returns:
        --------
        mape : float
            Mean Absolute Percentage Error (as percentage)
        """
        # Handle zero values in actual
        mask = actual != 0
        if not np.any(mask):
            return float("inf")  # All actual values are zero

        mape_values = np.abs((actual[mask] - predicted[mask]) / actual[mask]) * 100
        return float(np.mean(mape_values))

    def _calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error.

        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values

        Returns:
        --------
        rmse : float
            Root Mean Square Error
        """
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))

    def _calculate_mse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Square Error.

        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values

        Returns:
        --------
        mse : float
            Mean Square Error
        """
        return float(np.mean((actual - predicted) ** 2))

    def _calculate_mase(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        historical_data: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate Mean Absolute Scaled Error.

        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
        historical_data : np.ndarray, optional
            Historical data for scaling factor calculation

        Returns:
        --------
        mase : float
            Mean Absolute Scaled Error
        """
        if historical_data is None:
            # Cannot calculate MASE without historical data
            return float("nan")

        historical_array = (
            historical_data.values
            if hasattr(historical_data, "values")
            else np.array(historical_data)
        )

        # Calculate seasonal naive forecast errors
        seasonal_period = self.seasonal_period or 1

        if len(historical_array) <= seasonal_period:
            # Use simple naive forecast (lag-1) if insufficient data for seasonal naive
            naive_errors = np.abs(np.diff(historical_array))
        else:
            # Use seasonal naive forecast
            naive_forecast = historical_array[:-seasonal_period]
            naive_actual = historical_array[seasonal_period:]
            naive_errors = np.abs(naive_actual - naive_forecast)

        if len(naive_errors) == 0 or np.mean(naive_errors) == 0:
            return float("inf")

        mae = np.mean(np.abs(actual - predicted))
        mean_naive_error = np.mean(naive_errors)

        return float(mae / mean_naive_error)

    def _calculate_smape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.

        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values

        Returns:
        --------
        smape : float
            Symmetric Mean Absolute Percentage Error (as percentage)
        """
        denominator = (np.abs(actual) + np.abs(predicted)) / 2.0

        # Handle zero denominators
        mask = denominator != 0
        if not np.any(mask):
            return 0.0  # Perfect forecast when both actual and predicted are zero

        smape_values = np.abs(actual[mask] - predicted[mask]) / denominator[mask] * 100
        return float(np.mean(smape_values))

    def _calculate_r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination).

        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values

        Returns:
        --------
        r2 : float
            R-squared value
        """
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return float(1 - (ss_res / ss_tot))

    def _calculate_directional_accuracy(
        self, actual: np.ndarray, predicted: np.ndarray
    ) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions).

        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values

        Returns:
        --------
        directional_accuracy : float
            Directional accuracy as percentage
        """
        if len(actual) < 2 or len(predicted) < 2:
            return float("nan")

        # Calculate direction changes
        actual_direction = np.diff(actual) >= 0
        predicted_direction = np.diff(predicted) >= 0

        # Calculate accuracy
        correct_directions = actual_direction == predicted_direction
        return float(np.mean(correct_directions) * 100)

    def create_evaluation_report(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
        historical_data: Optional[Union[pd.Series, np.ndarray]] = None,
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report.

        Parameters:
        -----------
        actual : pd.Series or np.ndarray
            Actual observed values
        predicted : pd.Series or np.ndarray
            Predicted/forecasted values
        historical_data : pd.Series or np.ndarray, optional
            Historical data for MASE calculation
        model_name : str, default='Model'
            Name of the model being evaluated

        Returns:
        --------
        report : dict
            Comprehensive evaluation report
        """
        metrics = self.evaluate_forecast(actual, predicted, historical_data)

        # Create performance categorization
        performance_category = self._categorize_performance(metrics)

        # Generate interpretation
        interpretation = self._generate_evaluation_interpretation(metrics, model_name)

        # Generate recommendations
        recommendations = self._generate_evaluation_recommendations(metrics)

        report = {
            "model_name": model_name,
            "evaluation_metrics": metrics,
            "performance_category": performance_category,
            "interpretation": interpretation,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def _categorize_performance(self, metrics: Dict[str, float]) -> str:
        """
        Categorize model performance based on metrics.

        Parameters:
        -----------
        metrics : dict
            Evaluation metrics

        Returns:
        --------
        category : str
            Performance category
        """
        # Use MAPE as primary metric for categorization
        mape = metrics.get("mape", float("inf"))

        if mape <= 5:
            return "excellent"
        elif mape <= 15:
            return "good"
        elif mape <= 25:
            return "moderate"
        else:
            return "poor"

    def _generate_evaluation_interpretation(
        self, metrics: Dict[str, float], model_name: str
    ) -> str:
        """
        Generate human-readable interpretation of evaluation results.

        Parameters:
        -----------
        metrics : dict
            Evaluation metrics
        model_name : str
            Name of the model

        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        mae = metrics.get("mae", 0)
        mape = metrics.get("mape", 0)
        rmse = metrics.get("rmse", 0)
        r2 = metrics.get("r2", 0)
        n_obs = metrics.get("n_observations", 0)

        interpretation = (
            f"{model_name} evaluation on {n_obs} observations: "
            f"MAE = {mae:.3f}, MAPE = {mape:.1f}%, RMSE = {rmse:.3f}. "
        )

        if r2 != 0:
            interpretation += f"Variance explained (R²) = {r2:.3f}. "

        # Add performance assessment
        if mape <= 10:
            interpretation += "Forecast accuracy is excellent."
        elif mape <= 20:
            interpretation += "Forecast accuracy is good."
        elif mape <= 30:
            interpretation += "Forecast accuracy is moderate."
        else:
            interpretation += "Forecast accuracy needs improvement."

        return interpretation

    def _generate_evaluation_recommendations(
        self, metrics: Dict[str, float]
    ) -> List[str]:
        """
        Generate recommendations based on evaluation metrics.

        Parameters:
        -----------
        metrics : dict
            Evaluation metrics

        Returns:
        --------
        recommendations : List[str]
            List of recommendations
        """
        recommendations = []

        mape = metrics.get("mape", 0)
        mae = metrics.get("mae", 0)
        rmse = metrics.get("rmse", 0)
        bias = abs(metrics.get("forecast_bias", 0))
        directional_accuracy = metrics.get("directional_accuracy", 0)

        if mape > 30:
            recommendations.append(
                "High forecast error - consider alternative models or additional features"
            )

        if rmse > mae * 2:
            recommendations.append(
                "High RMSE relative to MAE suggests presence of large forecast errors"
            )

        if bias > mae * 0.5:
            recommendations.append(
                "Significant forecast bias detected - model systematically over/under-predicts"
            )

        if directional_accuracy and directional_accuracy < 50:
            recommendations.append(
                "Poor directional accuracy - model struggles to predict trend changes"
            )

        if metrics.get("n_observations", 0) < 20:
            recommendations.append(
                "Limited evaluation data - collect more out-of-sample observations for robust evaluation"
            )

        if not recommendations:
            recommendations.append(
                "Forecast performance is satisfactory - continue monitoring"
            )

        return recommendations
