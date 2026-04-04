"""
Time Series Analysis - Multivariate base transformer.

Contains the MultivariateTimeSeriesTransformer base class that all multivariate
time series transformers (VAR, cointegration, Granger, impulse response) inherit from.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class MultivariateTimeSeriesTransformer(TimeSeriesTransformer):
    """
    Base transformer for multivariate time series analysis.

    This transformer provides common functionality for multivariate time series
    operations including data validation, preprocessing, and result formatting.
    It extends the TimeSeriesTransformer to handle multiple time series columns
    while maintaining streaming compatibility.

    Key Features:
    - Multivariate data validation and preprocessing
    - Streaming-compatible processing for large datasets
    - Comprehensive error handling for multivariate operations
    - Integration with existing pipeline infrastructure
    - Support for pandas MultiIndex and multiple columns

    Parameters:
    -----------
    min_series : int, default=2
        Minimum number of time series required
    max_series : int, default=None
        Maximum number of time series allowed (None for no limit)
    require_stationarity : bool, default=False
        Whether to require all series to be stationary
    """

    def __init__(
        self,
        min_series: int = 2,
        max_series: Optional[int] = None,
        require_stationarity: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_series = min_series
        self.max_series = max_series
        self.require_stationarity = require_stationarity
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the multivariate time series transformer.

        Validates the input data and marks the transformer as fitted.
        Subclasses perform their actual fitting inside ``_analysis_logic``.

        Parameters:
        -----------
        X : pd.DataFrame
            Input multivariate time series data
        y : pd.Series, optional
            Ignored, present for API compatibility

        Returns:
        --------
        self
        """
        self._validate_multivariate_data(X)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Transform multivariate time series data.

        Delegates to ``_analysis_logic`` which each subclass implements with
        its specific analysis (VAR, cointegration, Granger causality, etc.).

        Parameters:
        -----------
        X : pd.DataFrame
            Input multivariate time series data

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Analysis results
        """
        return self._analysis_logic(X)

    def _analysis_logic(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Core analysis logic to be implemented by subclasses.

        Parameters:
        -----------
        X : pd.DataFrame
            Input multivariate time series data

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Analysis results
        """
        raise NotImplementedError("Subclasses must implement _analysis_logic")

    def _validate_multivariate_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate multivariate time series data.

        Parameters:
        -----------
        X : pd.DataFrame
            Input multivariate time series data

        Returns:
        --------
        X_validated : pd.DataFrame
            Validated multivariate time series data

        Raises:
        -------
        TimeSeriesValidationError
            If data doesn't meet multivariate requirements
        """
        # First apply standard time series validation
        X, _ = self._validate_time_series(X)

        # Check number of series
        n_series = X.shape[1] if X.ndim > 1 else 1

        if n_series < self.min_series:
            raise TimeSeriesValidationError(
                f"Insufficient number of time series. Found {n_series}, "
                f"minimum required: {self.min_series}"
            )

        if self.max_series is not None and n_series > self.max_series:
            raise TimeSeriesValidationError(
                f"Too many time series. Found {n_series}, "
                f"maximum allowed: {self.max_series}"
            )

        # Check for sufficient observations
        min_obs = max(50, self.min_series * 10)  # Adaptive minimum
        if len(X) < min_obs:
            logger.warning(
                f"Limited observations ({len(X)}) for multivariate analysis. "
                f"Recommend at least {min_obs} observations."
            )

        # Check for stationarity if required
        if self.require_stationarity:
            self._check_multivariate_stationarity(X)

        # Check for multicollinearity
        self._check_multicollinearity(X)

        return X

    def _check_multivariate_stationarity(self, X: pd.DataFrame) -> None:
        """
        Check stationarity of all time series.

        Parameters:
        -----------
        X : pd.DataFrame
            Input multivariate time series data

        Raises:
        -------
        TimeSeriesValidationError
            If any series is non-stationary
        """
        non_stationary_series = []

        for col in X.columns:
            series = X[col].dropna()
            if len(series) < 10:
                continue

            try:
                # Use ADF test for stationarity
                adf_result = adfuller(series, autolag="AIC")
                if adf_result[1] > 0.05:  # p-value > 0.05
                    non_stationary_series.append(col)
            except Exception as e:
                logger.warning(f"Could not test stationarity for series {col}: {e}")

        if non_stationary_series:
            raise TimeSeriesValidationError(
                f"Non-stationary time series detected: {non_stationary_series}. "
                "Consider differencing or other transformations."
            )

    def _check_multicollinearity(
        self, X: pd.DataFrame, threshold: float = 0.95
    ) -> None:
        """
        Check for excessive multicollinearity between time series.

        Parameters:
        -----------
        X : pd.DataFrame
            Input multivariate time series data
        threshold : float, default=0.95
            Correlation threshold above which to warn
        """
        try:
            corr_matrix = X.corr().abs()

            # Find highly correlated pairs
            high_corr_pairs = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j and corr_matrix.loc[col1, col2] > threshold:
                        high_corr_pairs.append(
                            (col1, col2, corr_matrix.loc[col1, col2])
                        )

            if high_corr_pairs:
                warning_msg = "High correlation detected between time series: "
                for col1, col2, corr in high_corr_pairs:
                    warning_msg += f"\n  {col1} <-> {col2}: {corr:.3f}"
                logger.warning(
                    warning_msg + "\nThis may affect multivariate analysis results."
                )

        except Exception as e:
            logger.warning(f"Could not check multicollinearity: {e}")

    def _prepare_multivariate_result(
        self, analysis_type: str, **kwargs
    ) -> TimeSeriesAnalysisResult:
        """
        Prepare standardized result structure for multivariate analysis.

        Parameters:
        -----------
        analysis_type : str
            Type of multivariate analysis performed
        **kwargs
            Additional result parameters

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Standardized result structure
        """
        result = TimeSeriesAnalysisResult(analysis_type=analysis_type, **kwargs)

        return result

    def _calculate_data_quality_score(self, X: pd.DataFrame) -> float:
        """
        Calculate a data quality score for the multivariate time series.

        Parameters:
        -----------
        X : pd.DataFrame
            Input multivariate time series data

        Returns:
        --------
        score : float
            Quality score between 0.0 and 1.0
        """
        score = 1.0

        # Penalise for missing values
        missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        score -= missing_ratio * 0.4

        # Penalise for very short series
        if len(X) < 50:
            score -= 0.2
        elif len(X) < 100:
            score -= 0.1

        # Penalise for constant columns
        constant_cols = (X.std() == 0).sum()
        if constant_cols > 0:
            score -= 0.1 * (constant_cols / X.shape[1])

        return max(0.0, min(1.0, score))
