"""
Sklearn-compatible network transformer and high-level network functions.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._base import NetworkAnalysisType
from ._isochrone import IsochroneGenerator
from ._network import AccessibilityResult, IsochroneResult, RouteResult, SpatialNetwork
from ._network_analysis import AccessibilityAnalyzer, NetworkRouter

logger = get_logger(__name__)


class SpatialNetworkTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for network analysis operations."""

    def __init__(
        self,
        analysis_type: NetworkAnalysisType = NetworkAnalysisType.ACCESSIBILITY,
        network_data: Optional[Dict] = None,
        service_locations: List[Any] = None,
        max_travel_time: float = None,
    ):
        self.analysis_type = analysis_type
        self.network_data = network_data
        self.service_locations = service_locations or []
        self.max_travel_time = max_travel_time

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        self.network_ = SpatialNetwork()

        if self.network_data:
            self.network_.create_network_from_edges(
                self.network_data["nodes"], self.network_data["edges"]
            )
        else:
            if isinstance(X, pd.DataFrame):
                if "x" in X.columns and "y" in X.columns:
                    points = list(zip(X["x"], X["y"]))
                elif "longitude" in X.columns and "latitude" in X.columns:
                    points = list(zip(X["longitude"], X["latitude"]))
                else:
                    raise ValueError(
                        "DataFrame must contain 'x'/'y' or 'longitude'/'latitude' columns"
                    )
            else:
                points = [(row[0], row[1]) for row in X]

            self.network_.create_network_from_points(points, connection_method="knn")

        if self.analysis_type in [
            NetworkAnalysisType.SHORTEST_PATH,
            NetworkAnalysisType.ROUTING,
        ]:
            self.router_ = NetworkRouter(self.network_)
        elif self.analysis_type == NetworkAnalysisType.ACCESSIBILITY:
            self.accessibility_analyzer_ = AccessibilityAnalyzer(self.network_)
        elif self.analysis_type == NetworkAnalysisType.ISOCHRONE:
            self.isochrone_generator_ = IsochroneGenerator(self.network_)

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        check_is_fitted(self, "network_")

        if self.analysis_type == NetworkAnalysisType.ACCESSIBILITY:
            if not hasattr(self, "accessibility_analyzer_"):
                raise ValueError("Accessibility analyzer not initialized")

            demand_locations = (
                list(range(len(X)))
                if not isinstance(X, pd.DataFrame)
                else list(X.index)
            )

            result = self.accessibility_analyzer_.calculate_accessibility(
                self.service_locations,
                demand_locations,
                max_travel_time=self.max_travel_time,
            )

            if isinstance(X, pd.DataFrame):
                X_transformed = X.copy()
                X_transformed["accessibility_score"] = [
                    result.accessibility_scores.get(i, 0) for i in X.index
                ]
                X_transformed["travel_time"] = [
                    result.travel_times.get(i, float("inf")) for i in X.index
                ]
                X_transformed["is_reachable"] = [
                    i in result.reachable_locations for i in X.index
                ]
                return X_transformed
            else:
                accessibility_values = [
                    result.accessibility_scores.get(i, 0) for i in range(len(X))
                ]
                travel_times = [
                    result.travel_times.get(i, float("inf")) for i in range(len(X))
                ]
                return np.column_stack([X, accessibility_values, travel_times])

        elif self.analysis_type == NetworkAnalysisType.ISOCHRONE:
            if not hasattr(self, "isochrone_generator_"):
                raise ValueError("Isochrone generator not initialized")

            time_bands = (
                [self.max_travel_time] if self.max_travel_time else [500, 1000, 1500]
            )
            result = self.isochrone_generator_.generate_isochrones(
                self.service_locations, time_bands
            )

            summary_data = {
                "time_band": time_bands,
                "coverage_area": [
                    result.coverage_areas.get(band, 0) for band in time_bands
                ],
                "has_polygon": [
                    i < len(result.isochrone_polygons)
                    and result.isochrone_polygons[i] is not None
                    for i in range(len(time_bands))
                ],
            }

            return pd.DataFrame(summary_data)

        else:
            stats = self.network_.get_network_statistics()

            if isinstance(X, pd.DataFrame):
                X_transformed = X.copy()
                for key, value in stats.items():
                    X_transformed[f"network_{key}"] = value
                return X_transformed
            else:
                return X


# High-level convenience functions
def optimize_route(
    network_data: Dict,
    waypoints: List[Any],
    return_to_start: bool = False,
    optimization_method: str = "greedy",
) -> RouteResult:
    """Optimize route through multiple waypoints on a spatial network."""
    network = SpatialNetwork()
    network.create_network_from_edges(network_data["nodes"], network_data["edges"])

    router = NetworkRouter(network)
    return router.find_optimal_route(waypoints, return_to_start, optimization_method)


def optimize_routes(
    network_data: Dict,
    waypoint_sets: List[List[Any]],
    optimization_method: str = "greedy",
    return_to_start: bool = False,
) -> List[RouteResult]:
    """Optimize multiple route sets on a spatial network."""
    results = []

    for waypoints in waypoint_sets:
        try:
            result = optimize_route(
                network_data=network_data,
                waypoints=waypoints,
                return_to_start=return_to_start,
                optimization_method=optimization_method,
            )
            results.append(result)
        except Exception as e:
            logger.warning(f"Route optimization failed for waypoints {waypoints}: {e}")
            failed_result = RouteResult(
                route_path=[],
                route_coordinates=[],
                total_distance=float("inf"),
                total_time=None,
                waypoints=waypoints,
                path_geometry=None,
                execution_time=0.0,
            )
            results.append(failed_result)

    return results


def analyze_accessibility(
    network_data: Dict,
    service_locations: List[Any],
    demand_locations: List[Any],
    max_travel_time: float = None,
    impedance_function: str = "linear",
) -> AccessibilityResult:
    """Analyze spatial accessibility to services on a network."""
    network = SpatialNetwork()
    network.create_network_from_edges(network_data["nodes"], network_data["edges"])

    analyzer = AccessibilityAnalyzer(network)
    return analyzer.calculate_accessibility(
        service_locations, demand_locations, max_travel_time, impedance_function
    )


def generate_service_isochrones(
    network_data: Dict,
    service_locations: List[Any],
    time_bands: List[float],
    resolution: int = 50,
) -> IsochroneResult:
    """Generate isochrone polygons for service locations."""
    network = SpatialNetwork()
    network.create_network_from_edges(network_data["nodes"], network_data["edges"])

    generator = IsochroneGenerator(network)
    return generator.generate_isochrones(service_locations, time_bands, resolution)
