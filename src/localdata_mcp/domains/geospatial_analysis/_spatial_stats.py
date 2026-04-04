"""
Spatial statistics computations (Moran's I, Geary's C, LISA, Getis-Ord).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats as scipy_stats

from ...logging_manager import get_logger
from ._data import SpatialPoint
from ._distance import SpatialDistanceCalculator
from ._statistics import SpatialStatisticsResult, SpatialWeightsMatrix

logger = get_logger(__name__)


class SpatialStatistics:
    """
    Comprehensive spatial statistics analysis tools.

    Implements Moran's I, Geary's C, spatial clustering detection,
    and other spatial statistical measures with significance testing.
    """

    def __init__(self):
        """Initialize spatial statistics analyzer."""
        self.distance_calculator = SpatialDistanceCalculator()

    def morans_i(
        self,
        values: np.ndarray,
        points: List[Union[SpatialPoint, Tuple[float, float]]],
        weights_method: str = "knn",
        **weights_params,
    ) -> SpatialStatisticsResult:
        """Calculate Moran's I spatial autocorrelation statistic."""
        values = np.asarray(values)
        n = len(values)

        if n != len(points):
            raise ValueError("Number of values must match number of points")

        weights_matrix = SpatialWeightsMatrix(weights_method, **weights_params)
        W = weights_matrix.build_weights(points)

        y = values - np.mean(values)

        S0 = np.sum(W)
        if S0 == 0:
            return SpatialStatisticsResult(
                statistic="morans_i",
                value=0.0,
                interpretation="No spatial connectivity detected",
            )

        numerator = np.sum(np.outer(y, y) * W)
        denominator = np.sum(y * y)

        if denominator == 0:
            morans_i = 0.0
        else:
            morans_i = (n / S0) * (numerator / denominator)

        expected_i = -1.0 / (n - 1)

        S1 = 0.5 * np.sum((W + W.T) ** 2)
        S2 = np.sum(np.sum(W, axis=1) ** 2)

        b2 = n * np.sum(y**4) / (np.sum(y**2) ** 2)

        variance_i = ((n - 1) * S1 - 2 * S2 + S0**2) / (
            (n - 1) * (n - 2) * (n - 3) * S0**2
        )
        variance_i -= (
            (b2 - 3) * (n * S1 - S0**2) / ((n - 1) * (n - 2) * (n - 3) * S0**2)
        )
        variance_i += expected_i**2

        if variance_i > 0:
            z_score = (morans_i - expected_i) / np.sqrt(variance_i)
            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))
        else:
            z_score = None
            p_value = None

        if p_value is not None:
            if p_value < 0.05:
                if morans_i > expected_i:
                    interpretation = (
                        "Significant positive spatial autocorrelation (clustering)"
                    )
                else:
                    interpretation = (
                        "Significant negative spatial autocorrelation (dispersion)"
                    )
            else:
                interpretation = "No significant spatial autocorrelation"
        else:
            interpretation = "Unable to determine significance"

        return SpatialStatisticsResult(
            statistic="morans_i",
            value=morans_i,
            p_value=p_value,
            z_score=z_score,
            expected_value=expected_i,
            variance=variance_i,
            interpretation=interpretation,
            metadata={
                "n_observations": n,
                "sum_weights": S0,
                "weights_method": weights_method,
                "weights_params": weights_params,
            },
        )

    def gearys_c(
        self,
        values: np.ndarray,
        points: List[Union[SpatialPoint, Tuple[float, float]]],
        weights_method: str = "knn",
        **weights_params,
    ) -> SpatialStatisticsResult:
        """Calculate Geary's C spatial autocorrelation statistic."""
        values = np.asarray(values)
        n = len(values)

        if n != len(points):
            raise ValueError("Number of values must match number of points")

        weights_matrix = SpatialWeightsMatrix(weights_method, **weights_params)
        W = weights_matrix.build_weights(points)

        S0 = np.sum(W)
        if S0 == 0:
            return SpatialStatisticsResult(
                statistic="gearys_c",
                value=1.0,
                interpretation="No spatial connectivity detected",
            )

        numerator = 0.0
        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * (values[i] - values[j]) ** 2

        mean_val = np.mean(values)
        denominator = 2 * S0 * np.sum((values - mean_val) ** 2)

        if denominator == 0:
            gearys_c = 1.0
        else:
            gearys_c = ((n - 1) / S0) * (numerator / denominator)

        expected_c = 1.0

        S1 = 0.5 * np.sum((W + W.T) ** 2)
        S2 = np.sum(np.sum(W, axis=1) ** 2)

        variance_c = ((2 * S1 + S2) * (n - 1) - 4 * S0**2) / (2 * (n + 1) * S0**2)

        if variance_c > 0:
            z_score = (gearys_c - expected_c) / np.sqrt(variance_c)
            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))
        else:
            z_score = None
            p_value = None

        if p_value is not None:
            if p_value < 0.05:
                if gearys_c < expected_c:
                    interpretation = (
                        "Significant positive spatial autocorrelation (clustering)"
                    )
                else:
                    interpretation = (
                        "Significant negative spatial autocorrelation (dispersion)"
                    )
            else:
                interpretation = "No significant spatial autocorrelation"
        else:
            interpretation = "Unable to determine significance"

        return SpatialStatisticsResult(
            statistic="gearys_c",
            value=gearys_c,
            p_value=p_value,
            z_score=z_score,
            expected_value=expected_c,
            variance=variance_c,
            interpretation=interpretation,
            metadata={
                "n_observations": n,
                "sum_weights": S0,
                "weights_method": weights_method,
                "weights_params": weights_params,
            },
        )

    def local_morans_i(
        self,
        values: np.ndarray,
        points: List[Union[SpatialPoint, Tuple[float, float]]],
        weights_method: str = "knn",
        **weights_params,
    ) -> Dict[str, np.ndarray]:
        """Calculate Local Indicators of Spatial Association (LISA)."""
        values = np.asarray(values)
        n = len(values)

        weights_matrix = SpatialWeightsMatrix(weights_method, **weights_params)
        W = weights_matrix.build_weights(points)

        y = (values - np.mean(values)) / np.std(values)

        local_i = np.zeros(n)
        for i in range(n):
            neighbors = W[i, :] > 0
            if np.any(neighbors):
                local_i[i] = y[i] * np.sum(W[i, neighbors] * y[neighbors])

        cluster_types = np.full(n, "Not significant", dtype="<U20")
        mean_y = np.mean(y)

        for i in range(n):
            neighbors = W[i, :] > 0
            if np.any(neighbors):
                neighbor_mean = np.mean(y[neighbors])

                if y[i] > mean_y and neighbor_mean > mean_y:
                    cluster_types[i] = "High-High"
                elif y[i] < mean_y and neighbor_mean < mean_y:
                    cluster_types[i] = "Low-Low"
                elif y[i] > mean_y and neighbor_mean < mean_y:
                    cluster_types[i] = "High-Low"
                elif y[i] < mean_y and neighbor_mean > mean_y:
                    cluster_types[i] = "Low-High"

        return {
            "local_morans_i": local_i,
            "cluster_types": cluster_types,
            "standardized_values": y,
            "weights_matrix": W,
        }

    def spatial_clustering_analysis(
        self,
        values: np.ndarray,
        points: List[Union[SpatialPoint, Tuple[float, float]]],
        method: str = "hotspot",
        **params,
    ) -> Dict[str, Any]:
        """Perform spatial clustering analysis (hot spot analysis)."""
        if method == "hotspot":
            return self._getis_ord_gi(values, points, **params)
        elif method == "getis_ord":
            return self._getis_ord_gi(values, points, **params)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

    def _getis_ord_gi(
        self,
        values: np.ndarray,
        points: List[Union[SpatialPoint, Tuple[float, float]]],
        weights_method: str = "distance",
        **weights_params,
    ) -> Dict[str, Any]:
        """Calculate Getis-Ord Gi* statistic for hot spot analysis."""
        values = np.asarray(values)
        n = len(values)

        weights_matrix = SpatialWeightsMatrix(weights_method, **weights_params)
        W = weights_matrix.build_weights(points)

        np.fill_diagonal(W, 1.0)

        mean_val = np.mean(values)
        std_val = np.std(values)

        gi_star = np.zeros(n)
        z_scores = np.zeros(n)
        p_values = np.zeros(n)

        for i in range(n):
            wi_sum = np.sum(W[i, :])

            if wi_sum > 0:
                weighted_sum = np.sum(W[i, :] * values)
                gi_star[i] = weighted_sum

                expected_gi = wi_sum * mean_val

                wi_squared_sum = np.sum(W[i, :] ** 2)
                variance_gi = wi_sum * std_val**2 - (wi_sum * mean_val) ** 2 / (n - 1)
                variance_gi = variance_gi * (n - 1 - wi_squared_sum) / (n - 2)

                if variance_gi > 0:
                    z_scores[i] = (gi_star[i] - expected_gi) / np.sqrt(variance_gi)
                    p_values[i] = 2 * (1 - scipy_stats.norm.cdf(abs(z_scores[i])))

        significance_level = weights_params.get("significance_level", 0.05)
        hotspots = (z_scores > 0) & (p_values < significance_level)
        coldspots = (z_scores < 0) & (p_values < significance_level)

        cluster_labels = np.full(n, "Not significant", dtype="<U20")
        cluster_labels[hotspots] = "Hot spot"
        cluster_labels[coldspots] = "Cold spot"

        return {
            "gi_star": gi_star,
            "z_scores": z_scores,
            "p_values": p_values,
            "cluster_labels": cluster_labels,
            "hotspots": hotspots,
            "coldspots": coldspots,
            "n_hotspots": np.sum(hotspots),
            "n_coldspots": np.sum(coldspots),
            "significance_level": significance_level,
        }
