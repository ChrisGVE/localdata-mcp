"""
Time Series Analysis Domain - Comprehensive time series analysis capabilities.

This package implements advanced time series analysis tools including forecasting,
decomposition, stationarity testing, and multivariate analysis.
"""

from ._advanced_forecasting import AdvancedForecastingTransformer
from ._anomaly_detector import AnomalyDetector

# Forecasting
from ._arima import ARIMAForecastTransformer
from ._auto_arima import AutoARIMATransformer

# Autocorrelation analysis
from ._autocorrelation import AutocorrelationAnalysisTransformer

# Base types
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError

# Anomaly detection
from ._changepoint import ChangePointDetector
from ._cointegration import CointegrationAnalyzer

# Decomposition
from ._decomposition import TimeSeriesDecompositionTransformer

# Ensemble and evaluation
from ._ensemble import EnsembleForecaster
from ._evaluation import ForecastEvaluator
from ._exponential import ExponentialSmoothingForecaster

# Feature extraction
from ._features import TimeSeriesFeatureExtractor
from ._granger import GrangerCausalityAnalyzer
from ._impulse_response import ImpulseResponseAnalyzer
from ._lag_selection import LagSelectionTransformer

# Multivariate
from ._multivariate_base import MultivariateTimeSeriesTransformer
from ._partial_autocorrelation import PartialAutocorrelationAnalysisTransformer

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

# Quality validation
from ._quality import TimeSeriesQualityValidator
from ._sarima import SARIMAForecastTransformer
from ._seasonal_anomaly import SeasonalAnomalyDetector

# Stationarity testing
from ._stationarity import StationarityTestTransformer, UnitRootTestTransformer
from ._structural_break import StructuralBreakTester

# Transformer ABC
from ._transformer import TimeSeriesTransformer
from ._trend import TrendAnalysisTransformer
from ._var import VARModelForecaster

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
