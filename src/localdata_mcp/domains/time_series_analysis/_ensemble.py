"""
Time Series Analysis - Ensemble forecasting.

Contains the EnsembleForecaster class that combines multiple forecasting
models (Exponential Smoothing, ARIMA) to create robust ensemble predictions.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ...pipeline.base import (
    CompositionMetadata,
    PipelineResult,
)
from ._base import TimeSeriesValidationError
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class EnsembleForecaster(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for ensemble forecasting methods.

    Combines multiple forecasting models (Exponential Smoothing, ARIMA)
    to create robust ensemble predictions. Uses weighted averaging or more
    sophisticated combination methods.

    Parameters:
    -----------
    forecast_steps : int, default=10
        Number of steps to forecast ahead
    confidence_level : float, default=0.95
        Confidence level for prediction intervals
    methods : list, default=['exponential_smoothing', 'arima']
        List of forecasting methods to combine
    weights : dict, optional
        Weights for each method (auto-computed if None)
    combination_method : str, default='weighted_average'
        Method for combining forecasts: 'weighted_average', 'median', 'best_performer'
    validation_split : float, default=0.2
        Fraction of data to use for model validation and weight optimization
    """

    def __init__(
        self,
        forecast_steps=10,
        confidence_level=0.95,
        methods=None,
        weights=None,
        combination_method="weighted_average",
        validation_split=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.forecast_steps = forecast_steps
        self.confidence_level = confidence_level
        self.methods = methods or ["exponential_smoothing", "arima"]
        self.weights = weights or {}
        self.combination_method = combination_method
        self.validation_split = validation_split

        # Model instances
        self.models_ = {}
        self.fitted_models_ = {}
        self.model_performance_ = {}
        self.computed_weights_ = {}
        self.training_data_ = None
        self.validation_data_ = None

    def _split_data(self, data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Split data into training and validation sets.

        Parameters:
        -----------
        data : pd.Series
            Full time series data

        Returns:
        --------
        train_data : pd.Series
            Training data
        val_data : pd.Series
            Validation data
        """
        split_point = int(len(data) * (1 - self.validation_split))
        train_data = data.iloc[:split_point]
        val_data = data.iloc[split_point:]
        return train_data, val_data

    def _initialize_models(self) -> Dict[str, "TimeSeriesTransformer"]:
        """
        Initialize the forecasting models for the ensemble.

        Returns:
        --------
        models : dict
            Dictionary of initialized model instances
        """
        # Lazy imports to avoid circular dependencies
        from ._arima import ARIMAForecastTransformer
        from ._auto_arima import AutoARIMATransformer
        from ._exponential import ExponentialSmoothingForecaster

        models = {}

        for method in self.methods:
            if method == "exponential_smoothing":
                models[method] = ExponentialSmoothingForecaster(
                    forecast_steps=(
                        len(self.validation_data_)
                        if self.validation_data_ is not None
                        else self.forecast_steps
                    ),
                    confidence_level=self.confidence_level,
                )

            elif method == "arima":
                models[method] = ARIMAForecastTransformer(
                    forecast_steps=(
                        len(self.validation_data_)
                        if self.validation_data_ is not None
                        else self.forecast_steps
                    ),
                    alpha=1.0 - self.confidence_level,
                )

            elif method == "auto_arima":
                models[method] = AutoARIMATransformer(
                    forecast_steps=(
                        len(self.validation_data_)
                        if self.validation_data_ is not None
                        else self.forecast_steps
                    ),
                    stepwise=True,  # Faster for ensemble
                )

        return models

    def _evaluate_model_performance(
        self,
        model: TimeSeriesTransformer,
        method: str,
        train_data: pd.DataFrame,
        val_data: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate model performance on validation data.

        Parameters:
        -----------
        model : TimeSeriesTransformer
            Fitted model to evaluate
        method : str
            Method name
        train_data : pd.DataFrame
            Training data
        val_data : pd.Series
            Validation data for evaluation

        Returns:
        --------
        performance : dict
            Performance metrics
        """
        try:
            # Fit model on training data
            model.fit(train_data)

            # Generate forecasts for validation period
            result = model.transform(train_data)

            # Extract forecast values
            if hasattr(result, "data") and "forecast_values" in result.data:
                forecast_dict = result.data["forecast_values"]
                if isinstance(forecast_dict, dict):
                    forecast_values = pd.Series(forecast_dict)
                else:
                    forecast_values = forecast_dict
            else:
                raise ValueError(f"Could not extract forecasts from {method} model")

            # Align forecast and actual values
            min_length = min(len(forecast_values), len(val_data))
            forecast_aligned = forecast_values.iloc[:min_length]
            actual_aligned = val_data.iloc[:min_length]

            # Calculate metrics
            mae = np.mean(np.abs(forecast_aligned - actual_aligned))
            mse = np.mean((forecast_aligned - actual_aligned) ** 2)
            rmse = np.sqrt(mse)

            # MAPE (handle zero values)
            mape_values = np.abs((actual_aligned - forecast_aligned) / actual_aligned)
            mape_values = mape_values[np.isfinite(mape_values)]  # Remove inf/nan
            mape = np.mean(mape_values) * 100 if len(mape_values) > 0 else float("inf")

            return {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "score": 1.0 / (1.0 + rmse),  # Higher is better
            }

        except Exception as e:
            logger.warning(f"Could not evaluate {method}: {e}")
            return {
                "mae": float("inf"),
                "mse": float("inf"),
                "rmse": float("inf"),
                "mape": float("inf"),
                "score": 0.0,
            }

    def _compute_weights(self) -> Dict[str, float]:
        """
        Compute weights for ensemble combination based on validation performance.

        Returns:
        --------
        weights : dict
            Computed weights for each method
        """
        if self.weights:
            # Use provided weights, normalize to sum to 1
            total_weight = sum(self.weights.values())
            return {k: v / total_weight for k, v in self.weights.items()}

        # Compute weights based on performance scores
        if not self.model_performance_:
            # Equal weights as fallback
            num_models = len(self.fitted_models_)
            return {method: 1.0 / num_models for method in self.fitted_models_}

        # Inverse error weighting (better performance = higher weight)
        scores = {
            method: perf["score"] for method, perf in self.model_performance_.items()
        }
        total_score = sum(scores.values())

        if total_score == 0:
            # Equal weights fallback
            num_models = len(scores)
            return {method: 1.0 / num_models for method in scores}

        return {method: score / total_score for method, score in scores.items()}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the ensemble forecasting models."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)

        logger = get_logger(__name__)

        try:
            # Extract univariate time series
            if isinstance(X, pd.DataFrame):
                if X.shape[1] == 1:
                    data = X.iloc[:, 0]
                else:
                    raise ValueError(
                        "Ensemble forecasting requires univariate time series data"
                    )
            elif y is not None:
                data = y
            else:
                raise ValueError("No target variable provided for ensemble forecasting")

            self.training_data_ = data

            # Split data for validation if we have enough data points
            if len(data) > 50 and self.validation_split > 0:
                train_data, val_data = self._split_data(data)
                self.validation_data_ = val_data
                logger.info(
                    f"Split data: {len(train_data)} training, {len(val_data)} validation"
                )
            else:
                train_data = data
                self.validation_data_ = None
                logger.info(
                    "Using full dataset for training (insufficient data for validation)"
                )

            # Initialize models
            self.models_ = self._initialize_models()
            logger.info(
                f"Initialized {len(self.models_)} models: {list(self.models_.keys())}"
            )

            # Fit models and evaluate performance
            train_df = pd.DataFrame({"value": train_data})

            for method, model in self.models_.items():
                try:
                    logger.info(f"Fitting {method} model...")

                    if self.validation_data_ is not None:
                        # Evaluate on validation set
                        performance = self._evaluate_model_performance(
                            model, method, train_df, self.validation_data_
                        )
                        self.model_performance_[method] = performance
                        logger.info(
                            f"{method} validation RMSE: {performance['rmse']:.4f}"
                        )
                    else:
                        # Fit on full data
                        model.fit(train_df)

                    self.fitted_models_[method] = model

                except Exception as e:
                    logger.warning(f"Failed to fit {method}: {e}")
                    continue

            if not self.fitted_models_:
                raise ValueError("No models could be fitted successfully")

            # Now fit models on full training data for final predictions
            full_train_df = pd.DataFrame({"value": self.training_data_})
            for method in self.fitted_models_:
                try:
                    # Update forecast steps for final fitting
                    self.fitted_models_[method].forecast_steps = self.forecast_steps
                    self.fitted_models_[method].fit(full_train_df)
                except Exception as e:
                    logger.warning(f"Failed to refit {method} on full data: {e}")

            # Compute ensemble weights
            self.computed_weights_ = self._compute_weights()
            logger.info(f"Computed weights: {self.computed_weights_}")

        except Exception as e:
            logger.error(f"Error in ensemble forecasting fit: {e}")
            raise TimeSeriesValidationError(
                f"Ensemble forecasting fit failed: {str(e)}"
            )

        return self

    def _collect_individual_forecasts(
        self, X: pd.DataFrame
    ) -> Tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame]]:
        """Collect forecasts and confidence intervals from each fitted model."""
        logger = get_logger(__name__)
        individual_forecasts: Dict[str, pd.Series] = {}
        individual_intervals: Dict[str, pd.DataFrame] = {}

        for method, model in self.fitted_models_.items():
            try:
                result = model.transform(X)
                if not (hasattr(result, "data") and "forecast_values" in result.data):
                    continue
                forecast_dict = result.data["forecast_values"]
                forecast_values = (
                    pd.Series(forecast_dict)
                    if isinstance(forecast_dict, dict)
                    else forecast_dict
                )
                individual_forecasts[method] = forecast_values

                if "confidence_intervals" in result.data:
                    intervals_dict = result.data["confidence_intervals"]
                    intervals = (
                        pd.DataFrame(intervals_dict)
                        if isinstance(intervals_dict, dict)
                        else intervals_dict
                    )
                    individual_intervals[method] = intervals
            except Exception as e:
                logger.warning(f"Failed to get forecast from {method}: {e}")

        return individual_forecasts, individual_intervals

    def _combine_forecasts(
        self, individual_forecasts: Dict[str, pd.Series]
    ) -> pd.Series:
        """Combine individual forecasts using the configured combination method."""
        if self.combination_method == "weighted_average":
            return self._weighted_average_combination(individual_forecasts)
        elif self.combination_method == "median":
            return self._median_combination(individual_forecasts)
        elif self.combination_method == "best_performer":
            return self._best_performer_combination(individual_forecasts)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    def _build_ensemble_result_data(
        self,
        ensemble_forecast: pd.Series,
        individual_forecasts: Dict[str, pd.Series],
        ensemble_intervals,
    ) -> Dict:
        """Build the result data dictionary for the ensemble forecast."""
        forecast_std = np.std([f.values for f in individual_forecasts.values()], axis=0)
        forecast_agreement = 1.0 - (
            np.mean(forecast_std) / np.mean(np.abs(ensemble_forecast))
        )
        return {
            "forecast_method": "Ensemble",
            "forecast_values": ensemble_forecast.to_dict(),
            "confidence_intervals": (
                ensemble_intervals.to_dict() if ensemble_intervals is not None else None
            ),
            "forecast_horizon": self.forecast_steps,
            "confidence_level": self.confidence_level,
            "ensemble_details": {
                "methods": list(self.fitted_models_.keys()),
                "weights": self.computed_weights_,
                "combination_method": self.combination_method,
                "individual_forecasts": {
                    m: f.to_dict() for m, f in individual_forecasts.items()
                },
            },
            "ensemble_statistics": {
                "forecast_agreement": float(forecast_agreement),
                "method_count": len(individual_forecasts),
                "forecast_std_dev": (
                    forecast_std.tolist()
                    if hasattr(forecast_std, "tolist")
                    else float(forecast_std)
                ),
            },
            "model_performance": self.model_performance_,
            "interpretation": self._generate_interpretation(
                ensemble_forecast, individual_forecasts
            ),
            "recommendations": self._generate_recommendations(),
        }

    def transform(self, X: pd.DataFrame) -> PipelineResult:
        """Generate ensemble forecasts using fitted models."""
        check_is_fitted(self, ["fitted_models_", "computed_weights_", "training_data_"])

        start_time = time.time()
        logger = get_logger(__name__)

        try:
            individual_forecasts, individual_intervals = (
                self._collect_individual_forecasts(X)
            )

            if not individual_forecasts:
                raise ValueError("No individual forecasts could be generated")

            ensemble_forecast = self._combine_forecasts(individual_forecasts)
            ensemble_intervals = self._combine_confidence_intervals(
                individual_intervals
            )
            result_data = self._build_ensemble_result_data(
                ensemble_forecast, individual_forecasts, ensemble_intervals
            )

            return PipelineResult(
                success=True,
                data=result_data,
                metadata={},
                execution_time_seconds=time.time() - start_time,
                memory_used_mb=0.0,
                pipeline_stage="forecast",
                composition_metadata=CompositionMetadata(
                    domain="time_series",
                    analysis_type="forecast",
                    result_type="predictions",
                    data_artifacts={
                        "forecast_steps": self.forecast_steps,
                        "methods": self.methods,
                        "combination_method": self.combination_method,
                    },
                ),
            )

        except Exception as e:
            logger.error(f"Error in ensemble forecasting transform: {e}")
            raise TimeSeriesValidationError(
                f"Ensemble forecasting transform failed: {str(e)}"
            )

    def _weighted_average_combination(
        self, forecasts: Dict[str, pd.Series]
    ) -> pd.Series:
        """
        Combine forecasts using weighted average.

        Parameters:
        -----------
        forecasts : dict
            Dictionary of individual forecasts

        Returns:
        --------
        combined_forecast : pd.Series
            Weighted average forecast
        """
        # Align all forecasts to same length
        min_length = min(len(f) for f in forecasts.values())
        aligned_forecasts = {
            method: f.iloc[:min_length] for method, f in forecasts.items()
        }

        # Calculate weighted average
        combined = None
        for method, forecast in aligned_forecasts.items():
            weight = self.computed_weights_.get(method, 0.0)
            if combined is None:
                combined = forecast * weight
            else:
                combined += forecast * weight

        return combined

    def _median_combination(self, forecasts: Dict[str, pd.Series]) -> pd.Series:
        """
        Combine forecasts using median.

        Parameters:
        -----------
        forecasts : dict
            Dictionary of individual forecasts

        Returns:
        --------
        combined_forecast : pd.Series
            Median forecast
        """
        # Align all forecasts to same length
        min_length = min(len(f) for f in forecasts.values())
        forecast_array = np.array(
            [f.iloc[:min_length].values for f in forecasts.values()]
        )

        # Calculate median across methods
        median_values = np.median(forecast_array, axis=0)

        # Use index from first forecast
        first_forecast = list(forecasts.values())[0]
        return pd.Series(median_values, index=first_forecast.index[:min_length])

    def _best_performer_combination(self, forecasts: Dict[str, pd.Series]) -> pd.Series:
        """
        Use forecast from best-performing model.

        Parameters:
        -----------
        forecasts : dict
            Dictionary of individual forecasts

        Returns:
        --------
        combined_forecast : pd.Series
            Best performer's forecast
        """
        if not self.model_performance_:
            # Fallback to first available forecast
            return list(forecasts.values())[0]

        # Find best performing method
        best_method = max(
            self.model_performance_.keys(),
            key=lambda x: self.model_performance_[x]["score"],
        )

        return forecasts.get(best_method, list(forecasts.values())[0])

    def _combine_confidence_intervals(
        self, intervals: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """
        Combine confidence intervals from multiple models.

        Parameters:
        -----------
        intervals : dict
            Dictionary of individual confidence intervals

        Returns:
        --------
        combined_intervals : pd.DataFrame or None
            Combined confidence intervals
        """
        if not intervals:
            return None

        try:
            # Align all intervals
            min_length = min(len(ci) for ci in intervals.values())
            aligned_intervals = {
                method: ci.iloc[:min_length] for method, ci in intervals.items()
            }

            # Use the widest intervals (most conservative approach)
            all_lowers = [
                ci["lower"] if "lower" in ci.columns else ci.iloc[:, 0]
                for ci in aligned_intervals.values()
            ]
            all_uppers = [
                ci["upper"] if "upper" in ci.columns else ci.iloc[:, 1]
                for ci in aligned_intervals.values()
            ]

            combined_lower = np.min(all_lowers, axis=0)
            combined_upper = np.max(all_uppers, axis=0)

            # Use index from first interval
            first_interval = list(intervals.values())[0]
            return pd.DataFrame(
                {"lower": combined_lower, "upper": combined_upper},
                index=first_interval.index[:min_length],
            )

        except Exception as e:
            logger.warning(f"Could not combine confidence intervals: {e}")
            return None

    def _generate_interpretation(
        self, ensemble_forecast: pd.Series, individual_forecasts: Dict[str, pd.Series]
    ) -> str:
        """
        Generate human-readable interpretation of ensemble forecast results.

        Parameters:
        -----------
        ensemble_forecast : pd.Series
            Combined ensemble forecast
        individual_forecasts : dict
            Dictionary of individual model forecasts

        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        avg_forecast = ensemble_forecast.mean()
        forecast_trend = (
            "increasing"
            if ensemble_forecast.iloc[-1] > ensemble_forecast.iloc[0]
            else "decreasing"
        )

        # Calculate forecast agreement
        forecast_std = np.std([f.values for f in individual_forecasts.values()], axis=0)
        avg_std = np.mean(forecast_std)
        agreement = (
            "high"
            if avg_std < abs(avg_forecast) * 0.1
            else ("moderate" if avg_std < abs(avg_forecast) * 0.2 else "low")
        )

        interpretation = (
            f"Ensemble of {len(individual_forecasts)} models predicts {forecast_trend} trend over {self.forecast_steps} periods. "
            f"Average predicted value: {avg_forecast:.2f}. "
            f"Model agreement: {agreement} (std dev: {avg_std:.2f}). "
            f"Best performing method: {max(self.model_performance_.keys(), key=lambda x: self.model_performance_[x]['score']) if self.model_performance_ else 'N/A'}."
        )

        return interpretation

    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on the ensemble model performance.

        Returns:
        --------
        recommendations : List[str]
            List of recommendations for the user
        """
        recommendations = []

        if len(self.fitted_models_) < 3:
            recommendations.append(
                "Consider adding more forecasting methods to the ensemble for better robustness"
            )

        if self.model_performance_:
            # Check performance spread
            scores = [perf["score"] for perf in self.model_performance_.values()]
            score_std = np.std(scores)
            if score_std > 0.3:
                recommendations.append(
                    "High performance variation between models - consider investigating data characteristics"
                )

            # Check if one model dominates
            max_weight = max(self.computed_weights_.values())
            if max_weight > 0.8:
                best_method = max(
                    self.computed_weights_.keys(),
                    key=lambda x: self.computed_weights_[x],
                )
                recommendations.append(
                    f"Single model ({best_method}) dominates ensemble - consider using it individually"
                )

        if len(self.training_data_) < 100:
            recommendations.append(
                "More historical data would improve ensemble forecast reliability"
            )

        recommendations.append(
            "Ensemble forecasting combines multiple methods to reduce individual model biases"
        )

        return recommendations
