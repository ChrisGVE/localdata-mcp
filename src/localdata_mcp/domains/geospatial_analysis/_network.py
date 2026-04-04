"""
Spatial network analysis for the geospatial analysis domain.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._base import GeospatialLibrary, NetworkAnalysisType, _dependency_status
from ._distance import SpatialDistanceCalculator

logger = get_logger(__name__)


@dataclass
class RouteResult:
    """Results from a route optimization operation."""

    route_path: List[Any]
    route_coordinates: List[Tuple[float, float]]
    total_distance: float
    total_time: Optional[float]
    waypoints: List[Any]
    path_geometry: Optional[Any]
    execution_time: float

    def summary(self) -> Dict[str, Any]:
        return {
            "total_distance": self.total_distance,
            "total_time": self.total_time,
            "waypoints_count": len(self.waypoints),
            "path_length": len(self.route_path),
            "execution_time_seconds": self.execution_time,
        }


@dataclass
class AccessibilityResult:
    """Results from an accessibility analysis."""

    accessibility_scores: Dict[str, float]
    reachable_locations: List[Any]
    travel_times: Dict[str, float]
    service_coverage: Dict[str, Any]
    analysis_parameters: Dict[str, Any]
    execution_time: float

    def summary(self) -> Dict[str, Any]:
        return {
            "total_locations_analyzed": len(self.accessibility_scores),
            "reachable_count": len(self.reachable_locations),
            "average_accessibility": np.mean(list(self.accessibility_scores.values()))
            if self.accessibility_scores
            else 0,
            "max_travel_time": max(self.travel_times.values())
            if self.travel_times
            else 0,
            "analysis_parameters": self.analysis_parameters,
            "execution_time_seconds": self.execution_time,
        }


@dataclass
class IsochroneResult:
    """Results from an isochrone analysis."""

    isochrone_polygons: List[Any]
    time_bands: List[float]
    coverage_areas: Dict[float, float]
    population_coverage: Optional[Dict[float, int]]
    service_points: List[Any]
    execution_time: float

    def summary(self) -> Dict[str, Any]:
        return {
            "time_bands": self.time_bands,
            "isochrone_count": len(self.isochrone_polygons),
            "coverage_areas": self.coverage_areas,
            "population_coverage": self.population_coverage,
            "service_points_count": len(self.service_points),
            "execution_time_seconds": self.execution_time,
        }


class SpatialNetwork:
    """Core spatial network representation with networkx integration."""

    def __init__(self):
        from ._dependency import GeospatialDependencyChecker

        self.dependency_checker = GeospatialDependencyChecker()
        self.network = None
        self.node_coordinates = {}
        self.edge_geometries = {}

    def create_network_from_points(
        self,
        points: List[Tuple[float, float]],
        connection_threshold: float = None,
        connection_method: str = "distance",
    ) -> None:
        """Create a network from point coordinates."""
        if not _dependency_status.is_available(GeospatialLibrary.NETWORKX):
            raise ValueError("NetworkX is required for network analysis")

        import networkx as nx

        self.network = nx.Graph()

        for i, (x, y) in enumerate(points):
            self.network.add_node(i, x=x, y=y)
            self.node_coordinates[i] = (x, y)

        if connection_method == "distance" and connection_threshold:
            self._connect_by_distance(connection_threshold)
        elif connection_method == "knn":
            k = (
                min(5, len(points) - 1)
                if connection_threshold is None
                else int(connection_threshold)
            )
            self._connect_by_knn(k)
        elif connection_method == "delaunay":
            self._connect_by_delaunay()
        else:
            logger.warning(f"Unknown connection method: {connection_method}")

    def create_network_from_edges(self, nodes: List[Dict], edges: List[Dict]) -> None:
        """Create a network from explicit node and edge definitions."""
        if not _dependency_status.is_available(GeospatialLibrary.NETWORKX):
            raise ValueError("NetworkX is required for network analysis")

        import networkx as nx

        self.network = nx.Graph()

        for node in nodes:
            node_id = node["id"]
            x, y = node["x"], node["y"]
            self.network.add_node(
                node_id,
                x=x,
                y=y,
                **{k: v for k, v in node.items() if k not in ["id", "x", "y"]},
            )
            self.node_coordinates[node_id] = (x, y)

        for edge in edges:
            source, target = edge["source"], edge["target"]
            weight = edge.get("weight", 1.0)

            if (
                "weight" not in edge
                and source in self.node_coordinates
                and target in self.node_coordinates
            ):
                coord1, coord2 = (
                    self.node_coordinates[source],
                    self.node_coordinates[target],
                )
                weight = np.sqrt(
                    (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2
                )

            self.network.add_edge(
                source,
                target,
                weight=weight,
                **{
                    k: v
                    for k, v in edge.items()
                    if k not in ["source", "target", "weight"]
                },
            )

            if "geometry" in edge:
                self.edge_geometries[(source, target)] = edge["geometry"]

    def _connect_by_distance(self, threshold):
        """Connect nodes within distance threshold."""
        distance_calc = SpatialDistanceCalculator()

        node_list = list(self.network.nodes())
        for i, node1 in enumerate(node_list):
            coord1 = self.node_coordinates[node1]
            for node2 in node_list[i + 1 :]:
                coord2 = self.node_coordinates[node2]
                distance = distance_calc.euclidean_distance(coord1, coord2)
                if distance <= threshold:
                    self.network.add_edge(node1, node2, weight=distance)

    def _connect_by_knn(self, k):
        """Connect each node to k nearest neighbors."""
        distance_calc = SpatialDistanceCalculator()

        for node in self.network.nodes():
            coord = self.node_coordinates[node]
            distances = []
            for other_node in self.network.nodes():
                if other_node != node:
                    other_coord = self.node_coordinates[other_node]
                    distance = distance_calc.euclidean_distance(coord, other_coord)
                    distances.append((distance, other_node))

            distances.sort()
            for distance, neighbor in distances[:k]:
                if not self.network.has_edge(node, neighbor):
                    self.network.add_edge(node, neighbor, weight=distance)

    def _connect_by_delaunay(self):
        """Connect nodes using Delaunay triangulation."""
        if not _dependency_status.is_available(GeospatialLibrary.SCIPY):
            logger.warning("SciPy not available for Delaunay triangulation")
            return

        try:
            from scipy.spatial import Delaunay

            points = np.array(
                [self.node_coordinates[node] for node in self.network.nodes()]
            )
            node_list = list(self.network.nodes())

            tri = Delaunay(points)

            distance_calc = SpatialDistanceCalculator()

            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        node1, node2 = node_list[simplex[i]], node_list[simplex[j]]
                        if not self.network.has_edge(node1, node2):
                            coord1, coord2 = (
                                self.node_coordinates[node1],
                                self.node_coordinates[node2],
                            )
                            distance = distance_calc.euclidean_distance(coord1, coord2)
                            self.network.add_edge(node1, node2, weight=distance)

        except ImportError:
            logger.warning("Delaunay triangulation requires scipy")

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get basic network topology statistics."""
        if self.network is None:
            return {}

        import networkx as nx

        stats = {
            "num_nodes": self.network.number_of_nodes(),
            "num_edges": self.network.number_of_edges(),
            "density": nx.density(self.network),
            "is_connected": nx.is_connected(self.network),
        }

        if nx.is_connected(self.network):
            stats.update(
                {
                    "average_clustering": nx.average_clustering(self.network),
                    "average_path_length": nx.average_shortest_path_length(
                        self.network, weight="weight"
                    ),
                    "diameter": nx.diameter(self.network),
                }
            )
        else:
            stats["num_components"] = nx.number_connected_components(self.network)

        return stats
