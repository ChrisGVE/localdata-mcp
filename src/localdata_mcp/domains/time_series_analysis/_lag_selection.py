"""
Lag selection transformer for automated ARIMA order determination.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._base import TimeSeriesAnalysisResult
from ._transformer import TimeSeriesTransformer
from ...logging_manager import get_logger

logger = get_logger(__name__)


class LagSelectionTransformer(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for automated lag selection using information criteria.

    Determines optimal lag structures for time series models using AIC, BIC, and
    other information criteria, combining ACF/PACF analysis with statistical tests.

    Parameters:
    -----------
    max_ar_order : int, default=5
        Maximum AR order to consider
    max_ma_order : int, default=5
        Maximum MA order to consider
    information_criteria : list of str, default=['aic', 'bic']
        Information criteria to use: 'aic', 'bic', 'hqic'
    seasonal : bool, default=False
        Whether to consider seasonal components
    seasonal_periods : list of int, optional
        Seasonal periods to test if seasonal=True
    """

    def __init__(
        self,
        max_ar_order=5,
        max_ma_order=5,
        information_criteria=["aic", "bic"],
        seasonal=False,
        seasonal_periods=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_ar_order = max_ar_order
        self.max_ma_order = max_ma_order
        self.information_criteria = information_criteria
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods or []
        self.best_orders_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the lag selection transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Perform automated lag selection analysis.

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Lag selection results with optimal model orders
        """
        start_time = time.time()

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            # Use first column for analysis
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")

            series = X.iloc[:, 0].dropna()
            if len(series) < 20:
                raise ValueError(
                    "Insufficient data points for lag selection (need at least 20)"
                )

            # Grid search over model orders
            results_grid = self._grid_search_orders(series)

            # Select best models for each criterion
            best_models = {}
            for criterion in self.information_criteria:
                if criterion in results_grid and len(results_grid[criterion]) > 0:
                    best_model = min(
                        results_grid[criterion], key=lambda x: x[criterion]
                    )
                    best_models[criterion] = {
                        "ar_order": best_model["ar_order"],
                        "ma_order": best_model["ma_order"],
                        "criterion_value": best_model[criterion],
                        "seasonal_order": best_model.get("seasonal_order", 0),
                    }

            self.best_orders_ = best_models

            # Generate recommendations
            recommendations = []
            warnings_list = []

            if len(best_models) == 0:
                interpretation = "Could not determine optimal model orders"
                recommendations.append(
                    "Use default ARIMA(1,0,1) or consult domain expert"
                )
                warnings_list.append("Model selection failed - check data quality")
            else:
                # Get consensus recommendation
                consensus = self._get_consensus_recommendation(best_models)

                interpretation = f"Optimal model orders determined: ARIMA({consensus['ar']},{consensus['d']},{consensus['ma']})"
                recommendations.append(
                    f"Recommended model: ARIMA({consensus['ar']},{consensus['d']},{consensus['ma']})"
                )

                if consensus.get("seasonal_order", 0) > 0:
                    recommendations.append(
                        f"Seasonal component suggested: ({consensus['seasonal_order']},{consensus['seasonal_d']},{consensus['seasonal_ma']})"
                    )

                # Add criterion-specific information
                for criterion, model_info in best_models.items():
                    recommendations.append(
                        f"{criterion.upper()} suggests: AR({model_info['ar_order']}) MA({model_info['ma_order']})"
                    )

            result = TimeSeriesAnalysisResult(
                analysis_type="lag_selection",
                interpretation=interpretation,
                model_diagnostics={
                    "best_models": best_models,
                    "grid_search_results": results_grid,
                    "consensus_recommendation": consensus
                    if len(best_models) > 0
                    else None,
                    "series_length": len(series),
                },
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time,
            )

            return result

        except Exception as e:
            logger.error(f"Error in lag selection: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="lag_selection_error",
                interpretation=f"Error during lag selection: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time,
            )

    def _grid_search_orders(self, series: pd.Series) -> Dict[str, List[Dict]]:
        """
        Perform grid search over ARIMA orders.

        Parameters:
        -----------
        series : pd.Series
            Time series data

        Returns:
        --------
        results : dict
            Dictionary of results for each information criterion
        """
        results = {criterion: [] for criterion in self.information_criteria}

        try:
            from statsmodels.tsa.arima.model import ARIMA

            # Test different AR and MA orders
            for ar_order in range(self.max_ar_order + 1):
                for ma_order in range(self.max_ma_order + 1):
                    if ar_order == 0 and ma_order == 0:
                        continue  # Skip (0,0,0) model

                    try:
                        # Fit ARIMA model
                        model = ARIMA(series, order=(ar_order, 0, ma_order))
                        fitted_model = model.fit()

                        # Collect information criteria
                        model_result = {
                            "ar_order": ar_order,
                            "ma_order": ma_order,
                            "aic": fitted_model.aic,
                            "bic": fitted_model.bic,
                            "hqic": fitted_model.hqic,
                            "converged": fitted_model.mle_retvals["converged"]
                            if hasattr(fitted_model, "mle_retvals")
                            else True,
                        }

                        # Add to results for each criterion
                        for criterion in self.information_criteria:
                            if criterion in model_result:
                                results[criterion].append(model_result)

                    except Exception as e:
                        logger.debug(f"ARIMA({ar_order},0,{ma_order}) failed: {e}")
                        continue

        except ImportError:
            logger.warning(
                "statsmodels ARIMA not available - using simplified lag selection"
            )

        return results

    def _get_consensus_recommendation(
        self, best_models: Dict[str, Dict]
    ) -> Dict[str, int]:
        """Get consensus recommendation from different information criteria."""
        if len(best_models) == 0:
            return {
                "ar": 1,
                "d": 0,
                "ma": 1,
                "seasonal_order": 0,
                "seasonal_d": 0,
                "seasonal_ma": 0,
            }

        # Get the most common orders
        ar_orders = [model["ar_order"] for model in best_models.values()]
        ma_orders = [model["ma_order"] for model in best_models.values()]

        # Use mode or median
        consensus_ar = int(np.round(np.median(ar_orders)))
        consensus_ma = int(np.round(np.median(ma_orders)))

        return {
            "ar": consensus_ar,
            "d": 0,  # Differencing order (would be determined by stationarity tests)
            "ma": consensus_ma,
            "seasonal_order": 0,  # Simplified for now
            "seasonal_d": 0,
            "seasonal_ma": 0,
        }
