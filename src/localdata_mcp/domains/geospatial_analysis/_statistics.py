"""
Spatial statistics analysis for the geospatial analysis domain.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._data import SpatialPoint
from ._distance import SpatialDistanceCalculator

logger = get_logger(__name__)


@dataclass
class SpatialStatisticsResult:
    """Container for spatial statistics results."""

    statistic: str
    value: float
    p_value: Optional[float] = None
    z_score: Optional[float] = None
    expected_value: Optional[float] = None
    variance: Optional[float] = None
    interpretation: Optional[str] = None
    significance_level: float = 0.05
    is_significant: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate significance after initialization."""
        if self.p_value is not None:
            self.is_significant = self.p_value < self.significance_level


class SpatialWeightsMatrix:
    """
    Spatial weights matrix for spatial statistics calculations.

    Provides various methods for constructing spatial weights matrices
    including distance-based, contiguity-based, and k-nearest neighbor weights.
    """

    def __init__(self, method: str = "distance", **kwargs):
        """
        Initialize spatial weights matrix.

        Parameters
        ----------
        method : str, default 'distance'
            Weighting method ('distance', 'knn', 'queen', 'rook').
        **kwargs : additional parameters
            Method-specific parameters.
        """
        self.method = method
        self.params = kwargs
        self.weights = None
        self.n_observations = 0

    def build_weights(
        self,
        points: List[Union[SpatialPoint, Tuple[float, float]]],
        values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build spatial weights matrix."""
        self._points = points
        self.n_observations = len(points)

        if self.method == "distance":
            self.weights = self._build_distance_weights(points)
        elif self.method == "knn":
            k = self.params.get("k", 8)
            self.weights = self._build_knn_weights(points, k)
        elif self.method == "inverse_distance":
            power = self.params.get("power", 1.0)
            self.weights = self._build_inverse_distance_weights(points, power)
        else:
            raise ValueError(f"Unsupported weights method: {self.method}")

        return self.weights

    def _build_distance_weights(self, points):
        """Build distance-based weights matrix."""
        distance_calc = SpatialDistanceCalculator()
        distance_matrix = distance_calc.distance_matrix(points, points)

        threshold = self.params.get(
            "threshold", np.percentile(distance_matrix[distance_matrix > 0], 25)
        )

        weights = np.where(distance_matrix <= threshold, 1.0, 0.0)
        np.fill_diagonal(weights, 0)

        row_sums = np.sum(weights, axis=1)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums[:, np.newaxis]

        return weights

    def _build_knn_weights(self, points, k):
        """Build k-nearest neighbor weights matrix."""
        distance_calc = SpatialDistanceCalculator()

        weights = np.zeros((len(points), len(points)))

        for i in range(len(points)):
            query_point = [points[i]]
            distances, indices = distance_calc.nearest_neighbors(
                query_point, points, k=k + 1
            )

            neighbor_indices = indices[0][1 : k + 1] if len(indices[0]) > 1 else []

            for j in neighbor_indices:
                weights[i, j] = 1.0

        row_sums = np.sum(weights, axis=1)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums[:, np.newaxis]

        return weights

    def create_knn_weights(self, k: int = 8) -> np.ndarray:
        """Create k-nearest neighbor weights matrix using stored points."""
        self.method = "knn"
        self.params["k"] = k
        if hasattr(self, "_points") and self._points is not None:
            return self.build_weights(self._points)
        raise ValueError(
            "No points available. Call build_weights first or initialize with points."
        )

    def _build_inverse_distance_weights(self, points, power):
        """Build inverse distance weights matrix."""
        distance_calc = SpatialDistanceCalculator()
        distance_matrix = distance_calc.distance_matrix(points, points)

        distance_matrix[distance_matrix == 0] = np.inf

        weights = 1.0 / (distance_matrix**power)
        weights[np.isinf(weights)] = 0

        row_sums = np.sum(weights, axis=1)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums[:, np.newaxis]

        return weights
