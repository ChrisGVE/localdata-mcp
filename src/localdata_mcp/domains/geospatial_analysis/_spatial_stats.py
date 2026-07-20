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


def _weight_sums(W: np.ndarray) -> Tuple[float, float, float]:
    """Return the three weight aggregates S0, S1 and S2 that the variance formulas need.

    These follow Cliff & Ord (1981), *Spatial Processes*, and are shared by both
    Moran's I and Geary's C:

    - ``S0 = sum_i sum_j w_ij``
    - ``S1 = 0.5 * sum_i sum_j (w_ij + w_ji)**2``
    - ``S2 = sum_i (w_i. + w_.i)**2`` — each row sum plus the *matching column sum*.

    S2 is the one that is easy to get wrong: using row sums alone understates it
    (for a row-standardised k-nearest-neighbour matrix, by roughly a factor of
    four), which shrinks Moran's variance and drives Geary's negative, leaving it
    with no p-value at all.
    """
    return (
        float(np.sum(W)),
        float(0.5 * np.sum((W + W.T) ** 2)),
        float(np.sum((np.sum(W, axis=1) + np.sum(W, axis=0)) ** 2)),
    )


class SpatialStatistics:
    """
    Comprehensive spatial statistics analysis tools.

    Implements Moran's I, Geary's C, spatial clustering detection,
    and other spatial statistical measures with significance testing.

    Significance is assessed under the normality assumption of Cliff & Ord
    (1981): the statistic is compared to its expectation using a closed-form
    variance, and the resulting z-score is read against a normal distribution.
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

        S0, S1, S2 = _weight_sums(W)
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

        # Var_N(I) = (n^2 S1 - n S2 + 3 S0^2) / (S0^2 (n^2 - 1)) - E[I]^2
        # (Cliff & Ord 1981, normality assumption).
        variance_i = (n * n * S1 - n * S2 + 3 * S0**2) / (
            S0**2 * (n * n - 1)
        ) - expected_i**2

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

        S0, S1, S2 = _weight_sums(W)
        if S0 == 0:
            return SpatialStatisticsResult(
                statistic="gearys_c",
                value=1.0,
                interpretation="No spatial connectivity detected",
            )

        # C = (n-1) / (2 S0) * sum_i sum_j w_ij (v_i - v_j)^2 / sum_i (v_i - vbar)^2.
        # The 2*S0 belongs to this one denominator; dividing by S0 again — once in
        # the leading factor and once here — deflated C by a factor of S0.
        numerator = float(np.sum(W * (values[:, None] - values[None, :]) ** 2))
        denominator = 2 * S0 * np.sum((values - np.mean(values)) ** 2)

        if denominator == 0:
            gearys_c = 1.0
        else:
            gearys_c = (n - 1) * (numerator / denominator)

        expected_c = 1.0

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
        """Calculate Getis-Ord Gi* statistic for hot spot analysis.

        Uses the standard form (Getis & Ord 1992; Ord & Getis 1995):

            Gi*(z) = (sum_j w_ij x_j - Xbar sum_j w_ij)
                     / (S * sqrt((n sum_j w_ij^2 - (sum_j w_ij)^2) / (n - 1)))

        with ``S = sqrt(sum_j x_j^2 / n - Xbar^2)``. The z-score *is* the
        statistic that gets tested; ``gi_star`` below reports the underlying
        neighbourhood share for interpretation.

        Weights are made binary before use. Gi* counts a point among its own
        neighbours (that is the star), and adding a self-weight of 1 to an
        already row-standardised matrix would weight the point eight times its
        own neighbours instead of equally.
        """
        values = np.asarray(values, dtype=float)
        n = len(values)

        weights_matrix = SpatialWeightsMatrix(weights_method, **weights_params)
        W = (weights_matrix.build_weights(points) > 0).astype(float)
        np.fill_diagonal(W, 1.0)

        mean_val = np.mean(values)
        # Population standard deviation about the mean, matching the formula's S.
        s_val = np.sqrt(np.mean(values**2) - mean_val**2)
        total = np.sum(values)

        gi_star = np.zeros(n)
        z_scores = np.zeros(n)
        p_values = np.ones(n)

        for i in range(n):
            wi_sum = np.sum(W[i, :])
            if wi_sum <= 0:
                continue

            weighted_sum = np.sum(W[i, :] * values)
            gi_star[i] = weighted_sum / total if total != 0 else 0.0

            wi_squared_sum = np.sum(W[i, :] ** 2)
            denominator = s_val * np.sqrt((n * wi_squared_sum - wi_sum**2) / (n - 1))

            if denominator > 0:
                z_scores[i] = (weighted_sum - mean_val * wi_sum) / denominator
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
