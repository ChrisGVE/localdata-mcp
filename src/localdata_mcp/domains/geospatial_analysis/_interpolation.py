"""
Spatial interpolation (kriging, IDW, variogram modeling) for the geospatial analysis domain.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._base import GeospatialLibrary, _dependency_status
from ._data import SpatialPoint
from ._distance import SpatialDistanceCalculator

logger = get_logger(__name__)


@dataclass
class InterpolationResult:
    """Container for spatial interpolation results."""

    method: str
    interpolated_values: np.ndarray
    prediction_variance: Optional[np.ndarray] = None
    cross_validation_score: Optional[float] = None
    variogram_model: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class VariogramModel:
    """
    Variogram modeling for kriging interpolation.

    Provides variogram calculation, fitting, and model selection
    for spatial interpolation with fallback methods when
    scikit-gstat is not available.
    """

    VARIOGRAM_MODELS = {
        "spherical": lambda h, nugget, sill, range_param: (
            nugget
            + (sill - nugget) * (1.5 * h / range_param - 0.5 * (h / range_param) ** 3)
            if h <= range_param
            else sill
        ),
        "exponential": lambda h, nugget, sill, range_param: nugget
        + (sill - nugget) * (1 - np.exp(-3 * h / range_param)),
        "gaussian": lambda h, nugget, sill, range_param: nugget
        + (sill - nugget) * (1 - np.exp(-3 * (h / range_param) ** 2)),
        "linear": lambda h, nugget, sill, range_param: (
            nugget + (sill - nugget) * h / range_param if h <= range_param else sill
        ),
    }

    def __init__(self, model_type: str = "spherical"):
        """Initialize variogram model."""
        self.model_type = model_type
        self.parameters = None
        self.empirical_variogram = None
        self.scikit_gstat_available = _dependency_status.is_available(
            GeospatialLibrary.SCIKIT_GSTAT
        )

        if self.scikit_gstat_available:
            try:
                import skgstat as skg

                self.skg = skg
                logger.debug("scikit-gstat available for advanced variogram modeling")
            except ImportError:
                self.scikit_gstat_available = False
                logger.warning(
                    "scikit-gstat import failed, using fallback variogram methods"
                )

    def fit(
        self,
        points: List[Union[SpatialPoint, Tuple[float, float]]],
        values: np.ndarray,
        max_distance: Optional[float] = None,
        n_lags: int = 15,
    ) -> Dict[str, Any]:
        """Fit variogram model to data."""
        if self.scikit_gstat_available:
            return self._fit_scikit_gstat(points, values, max_distance, n_lags)
        else:
            self.is_fitted_ = True
            return self._fit_fallback(points, values, max_distance, n_lags)

    def _fit_scikit_gstat(self, points, values, max_distance, n_lags):
        """Fit variogram using scikit-gstat."""
        try:
            coords = []
            for point in points:
                if isinstance(point, tuple):
                    coords.append(point)
                elif isinstance(point, SpatialPoint):
                    coords.append((point.x, point.y))

            coords = np.array(coords)
            values = np.asarray(values)

            vario = self.skg.Variogram(
                coordinates=coords,
                values=values,
                model=self.model_type,
                maxlag=max_distance,
                n_lags=n_lags,
                normalize=False,
            )

            self.parameters = {
                "nugget": vario.parameters[0],
                "sill": vario.parameters[1],
                "range": vario.parameters[2],
                "model_type": self.model_type,
            }

            self.empirical_variogram = {
                "lags": vario.bins,
                "semivariance": vario.experimental,
                "counts": vario.bin_count,
            }

            return {
                "parameters": self.parameters,
                "empirical_variogram": self.empirical_variogram,
                "model_rmse": vario.rmse,
                "r_squared": getattr(vario, "r_squared", None),
                "method": "scikit_gstat",
            }

        except Exception as e:
            logger.warning(f"scikit-gstat variogram fitting failed: {e}")
            return self._fit_fallback(points, values, max_distance, n_lags)

    def _fit_fallback(self, points, values, max_distance, n_lags):
        """Fallback variogram fitting without scikit-gstat."""
        distance_calc = SpatialDistanceCalculator()
        coords = []
        for point in points:
            if isinstance(point, tuple):
                coords.append(point)
            elif isinstance(point, SpatialPoint):
                coords.append((point.x, point.y))

        values = np.asarray(values)

        distance_matrix = distance_calc.distance_matrix(coords, coords)

        if max_distance is None:
            max_distance = np.max(distance_matrix) / 2

        lag_width = max_distance / n_lags
        lags = np.arange(lag_width, max_distance + lag_width, lag_width)
        semivariances = []
        counts = []

        for i, lag in enumerate(lags):
            lag_min = i * lag_width
            lag_max = (i + 1) * lag_width

            mask = (distance_matrix >= lag_min) & (distance_matrix < lag_max)
            i_indices, j_indices = np.where(mask)

            if len(i_indices) > 0:
                squared_diffs = (values[i_indices] - values[j_indices]) ** 2
                semivariance = np.mean(squared_diffs) / 2
                semivariances.append(semivariance)
                counts.append(len(i_indices))
            else:
                semivariances.append(0)
                counts.append(0)

        self.empirical_variogram = {
            "lags": lags,
            "semivariance": np.array(semivariances),
            "counts": np.array(counts),
        }

        valid_idx = np.array(counts) > 0
        if np.any(valid_idx):
            nugget = min(semivariances) if semivariances else 0
            sill = max(semivariances) if semivariances else np.var(values)
            range_est = max_distance / 3

            self.parameters = {
                "nugget": nugget,
                "sill": sill,
                "range": range_est,
                "model_type": self.model_type,
            }
        else:
            self.parameters = {
                "nugget": 0,
                "sill": np.var(values),
                "range": max_distance / 3,
                "model_type": self.model_type,
            }

        return {
            "parameters": self.parameters,
            "empirical_variogram": self.empirical_variogram,
            "model_rmse": None,
            "r_squared": None,
            "method": "fallback",
        }

    def predict_semivariance(self, distances: np.ndarray) -> np.ndarray:
        """Predict semivariance at given distances using fitted model."""
        if self.parameters is None:
            raise ValueError("Variogram model must be fitted before prediction")

        distances = np.asarray(distances)
        model_func = self.VARIOGRAM_MODELS[self.model_type]

        vectorized_model = np.vectorize(
            lambda h: model_func(
                h,
                self.parameters["nugget"],
                self.parameters["sill"],
                self.parameters["range"],
            )
        )

        return vectorized_model(distances)
