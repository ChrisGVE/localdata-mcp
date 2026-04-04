"""
Time Series Analysis Domain - Comprehensive time series analysis capabilities.

This package implements advanced time series analysis tools including forecasting,
decomposition, stationarity testing, and multivariate analysis.
"""

# Base types
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError

# Transformer ABC
from ._transformer import TimeSeriesTransformer

# Pipeline and utilities
from ._pipeline import (
    TimeSeriesPipeline,
    detect_time_series_gaps,
    infer_time_series_frequency,
    validate_datetime_index,
    validate_time_series_continuity,
)

# Preprocessing
from ._preprocessing import (
    TimeSeriesImputationTransformer,
    TimeSeriesResamplingTransformer,
)

# Feature extraction
from ._features import TimeSeriesFeatureExtractor

# Quality validation
from ._quality import TimeSeriesQualityValidator

# Stationarity testing
from ._stationarity import StationarityTestTransformer, UnitRootTestTransformer

# Autocorrelation analysis
from ._autocorrelation import AutocorrelationAnalysisTransformer
from ._partial_autocorrelation import PartialAutocorrelationAnalysisTransformer
from ._lag_selection import LagSelectionTransformer

# Decomposition
from ._decomposition import TimeSeriesDecompositionTransformer
from ._trend import TrendAnalysisTransformer

# Forecasting
from ._arima import ARIMAForecastTransformer
from ._sarima import SARIMAForecastTransformer
from ._auto_arima import AutoARIMATransformer
from ._advanced_forecasting import AdvancedForecastingTransformer
from ._exponential import ExponentialSmoothingForecaster

# Ensemble and evaluation
from ._ensemble import EnsembleForecaster
from ._evaluation import ForecastEvaluator

# Multivariate
from ._multivariate_base import MultivariateTimeSeriesTransformer
from ._var import VARModelForecaster
from ._cointegration import CointegrationAnalyzer
from ._granger import GrangerCausalityAnalyzer
from ._impulse_response import ImpulseResponseAnalyzer

# Anomaly detection
from ._changepoint import ChangePointDetector
from ._anomaly_detector import AnomalyDetector
from ._structural_break import StructuralBreakTester
from ._seasonal_anomaly import SeasonalAnomalyDetector

__all__ = [
    # Base
    "TimeSeriesAnalysisResult",
    "TimeSeriesValidationError",
    "TimeSeriesTransformer",
    # Pipeline
    "TimeSeriesPipeline",
    "detect_time_series_gaps",
    "infer_time_series_frequency",
    "validate_datetime_index",
    "validate_time_series_continuity",
    # Preprocessing
    "TimeSeriesResamplingTransformer",
    "TimeSeriesFeatureExtractor",
    "TimeSeriesImputationTransformer",
    "TimeSeriesQualityValidator",
    # Stationarity
    "StationarityTestTransformer",
    "UnitRootTestTransformer",
    # Autocorrelation
    "AutocorrelationAnalysisTransformer",
    "PartialAutocorrelationAnalysisTransformer",
    "LagSelectionTransformer",
    # Decomposition
    "TimeSeriesDecompositionTransformer",
    "TrendAnalysisTransformer",
    # Forecasting
    "ARIMAForecastTransformer",
    "SARIMAForecastTransformer",
    "AutoARIMATransformer",
    "AdvancedForecastingTransformer",
    "ExponentialSmoothingForecaster",
    # Ensemble
    "EnsembleForecaster",
    "ForecastEvaluator",
    # Multivariate
    "MultivariateTimeSeriesTransformer",
    "VARModelForecaster",
    "CointegrationAnalyzer",
    "GrangerCausalityAnalyzer",
    "ImpulseResponseAnalyzer",
    # Anomaly
    "ChangePointDetector",
    "AnomalyDetector",
    "StructuralBreakTester",
    "SeasonalAnomalyDetector",
]
