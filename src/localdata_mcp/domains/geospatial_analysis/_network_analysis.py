"""
Network routing, accessibility, isochrone analysis, and network transformer.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._base import GeospatialLibrary, NetworkAnalysisType, _dependency_status
from ._distance import SpatialDistanceCalculator
from ._network import RouteResult, AccessibilityResult, IsochroneResult, SpatialNetwork

logger = get_logger(__name__)


class NetworkRouter:
    """Route optimization and path finding with spatial constraints."""

    def __init__(self, network: SpatialNetwork):
        self.network = network
        self.distance_calc = SpatialDistanceCalculator()

    def find_shortest_path(self, source, target, weight="weight") -> RouteResult:
        """Find shortest path between two nodes."""
        start_time = time.time()

        if self.network.network is None:
            raise ValueError("Network not initialized")

        if not _dependency_status.is_available(GeospatialLibrary.NETWORKX):
            raise ValueError("NetworkX is required for routing")

        import networkx as nx

        try:
            path = nx.shortest_path(self.network.network, source, target, weight=weight)
            path_length = nx.shortest_path_length(
                self.network.network, source, target, weight=weight
            )

            route_coordinates = [self.network.node_coordinates[node] for node in path]

            path_geometry = None
            if _dependency_status.is_available(GeospatialLibrary.SHAPELY):
                from shapely.geometry import LineString

                path_geometry = LineString(route_coordinates)

            execution_time = time.time() - start_time

            return RouteResult(
                route_path=path,
                route_coordinates=route_coordinates,
                total_distance=path_length,
                total_time=None,
                waypoints=[source, target],
                path_geometry=path_geometry,
                execution_time=execution_time,
            )

        except nx.NetworkXNoPath:
            execution_time = time.time() - start_time
            return RouteResult(
                route_path=[],
                route_coordinates=[],
                total_distance=float("inf"),
                total_time=None,
                waypoints=[source, target],
                path_geometry=None,
                execution_time=execution_time,
            )

    def find_optimal_route(
        self, waypoints, return_to_start=False, optimization_method="greedy"
    ) -> RouteResult:
        """Find optimal route through multiple waypoints."""
        start_time = time.time()

        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required")

        if optimization_method == "greedy":
            route_path, total_distance = self._greedy_route_optimization(
                waypoints, return_to_start
            )
        elif optimization_method == "nearest_neighbor":
            route_path, total_distance = self._nearest_neighbor_tsp(
                waypoints, return_to_start
            )
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")

        route_coordinates = [self.network.node_coordinates[node] for node in route_path]

        path_geometry = None
        if _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            from shapely.geometry import LineString

            path_geometry = LineString(route_coordinates)

        execution_time = time.time() - start_time

        return RouteResult(
            route_path=route_path,
            route_coordinates=route_coordinates,
            total_distance=total_distance,
            total_time=None,
            waypoints=waypoints,
            path_geometry=path_geometry,
            execution_time=execution_time,
        )

    def _greedy_route_optimization(self, waypoints, return_to_start):
        """Greedy optimization for route through waypoints."""
        import networkx as nx

        if len(waypoints) == 2:
            try:
                path = nx.shortest_path(
                    self.network.network, waypoints[0], waypoints[1], weight="weight"
                )
                distance = nx.shortest_path_length(
                    self.network.network, waypoints[0], waypoints[1], weight="weight"
                )

                if return_to_start and waypoints[0] != waypoints[1]:
                    return_path = nx.shortest_path(
                        self.network.network,
                        waypoints[1],
                        waypoints[0],
                        weight="weight",
                    )[1:]
                    return_distance = nx.shortest_path_length(
                        self.network.network,
                        waypoints[1],
                        waypoints[0],
                        weight="weight",
                    )
                    path.extend(return_path)
                    distance += return_distance

                return path, distance
            except nx.NetworkXNoPath:
                return waypoints, float("inf")

        full_path = []
        total_distance = 0

        for i in range(len(waypoints) - 1):
            try:
                segment = nx.shortest_path(
                    self.network.network,
                    waypoints[i],
                    waypoints[i + 1],
                    weight="weight",
                )
                segment_distance = nx.shortest_path_length(
                    self.network.network,
                    waypoints[i],
                    waypoints[i + 1],
                    weight="weight",
                )

                if i == 0:
                    full_path.extend(segment)
                else:
                    full_path.extend(segment[1:])

                total_distance += segment_distance

            except nx.NetworkXNoPath:
                return waypoints, float("inf")

        if return_to_start and len(waypoints) > 1:
            try:
                return_segment = nx.shortest_path(
                    self.network.network, waypoints[-1], waypoints[0], weight="weight"
                )[1:]
                return_distance = nx.shortest_path_length(
                    self.network.network, waypoints[-1], waypoints[0], weight="weight"
                )
                full_path.extend(return_segment)
                total_distance += return_distance
            except nx.NetworkXNoPath:
                return full_path, float("inf")

        return full_path, total_distance

    def _nearest_neighbor_tsp(self, waypoints, return_to_start):
        """Nearest neighbor heuristic for TSP-like routing."""
        import networkx as nx

        if len(waypoints) <= 2:
            return self._greedy_route_optimization(waypoints, return_to_start)

        distance_matrix = {}
        for i, wp1 in enumerate(waypoints):
            for j, wp2 in enumerate(waypoints):
                if i != j:
                    try:
                        dist = nx.shortest_path_length(
                            self.network.network, wp1, wp2, weight="weight"
                        )
                        distance_matrix[(wp1, wp2)] = dist
                    except nx.NetworkXNoPath:
                        distance_matrix[(wp1, wp2)] = float("inf")

        start_wp = waypoints[0]
        current_wp = start_wp
        remaining = set(waypoints[1:])
        route_order = [current_wp]
        total_distance = 0

        while remaining:
            nearest = min(
                remaining,
                key=lambda wp: distance_matrix.get((current_wp, wp), float("inf")),
            )
            distance = distance_matrix.get((current_wp, nearest), float("inf"))

            if distance == float("inf"):
                break

            route_order.append(nearest)
            total_distance += distance
            current_wp = nearest
            remaining.remove(nearest)

        if return_to_start and len(route_order) > 1:
            return_distance = distance_matrix.get((current_wp, start_wp), float("inf"))
            if return_distance != float("inf"):
                route_order.append(start_wp)
                total_distance += return_distance

        full_path = []
        for i in range(len(route_order) - 1):
            try:
                segment = nx.shortest_path(
                    self.network.network,
                    route_order[i],
                    route_order[i + 1],
                    weight="weight",
                )
                if i == 0:
                    full_path.extend(segment)
                else:
                    full_path.extend(segment[1:])
            except nx.NetworkXNoPath:
                full_path.extend([route_order[i], route_order[i + 1]])

        return full_path, total_distance


class AccessibilityAnalyzer:
    """Spatial accessibility analysis for service coverage and reachability."""

    def __init__(self, network: SpatialNetwork):
        self.network = network

    def calculate_accessibility(
        self,
        service_locations,
        demand_locations,
        max_travel_time=None,
        impedance_function="linear",
    ) -> AccessibilityResult:
        """Calculate accessibility scores for demand locations to services."""
        start_time = time.time()

        if self.network.network is None:
            raise ValueError("Network not initialized")

        import networkx as nx

        accessibility_scores = {}
        travel_times = {}
        reachable_locations = []

        for demand_node in demand_locations:
            accessibility_score = 0
            min_travel_time = float("inf")

            for service_node in service_locations:
                try:
                    travel_distance = nx.shortest_path_length(
                        self.network.network, demand_node, service_node, weight="weight"
                    )

                    if impedance_function == "linear":
                        impedance = (
                            1 / (1 + travel_distance) if travel_distance > 0 else 1
                        )
                    elif impedance_function == "exponential":
                        impedance = (
                            np.exp(-travel_distance / 1000)
                            if travel_distance > 0
                            else 1
                        )
                    elif impedance_function == "gaussian":
                        impedance = np.exp(-(travel_distance**2) / (2 * (500**2)))
                    else:
                        impedance = 1

                    accessibility_score += impedance
                    min_travel_time = min(min_travel_time, travel_distance)

                except nx.NetworkXNoPath:
                    continue

            accessibility_scores[demand_node] = accessibility_score
            travel_times[demand_node] = min_travel_time

            if max_travel_time is None or min_travel_time <= max_travel_time:
                reachable_locations.append(demand_node)

        total_demand = len(demand_locations)
        reachable_count = len(reachable_locations)
        coverage_percentage = (
            (reachable_count / total_demand * 100) if total_demand > 0 else 0
        )

        service_coverage = {
            "total_demand_locations": total_demand,
            "reachable_locations": reachable_count,
            "coverage_percentage": coverage_percentage,
            "average_accessibility": np.mean(list(accessibility_scores.values()))
            if accessibility_scores
            else 0,
        }

        analysis_parameters = {
            "service_count": len(service_locations),
            "demand_count": len(demand_locations),
            "max_travel_time": max_travel_time,
            "impedance_function": impedance_function,
        }

        execution_time = time.time() - start_time

        return AccessibilityResult(
            accessibility_scores=accessibility_scores,
            reachable_locations=reachable_locations,
            travel_times=travel_times,
            service_coverage=service_coverage,
            analysis_parameters=analysis_parameters,
            execution_time=execution_time,
        )


# IsochroneGenerator moved to _isochrone.py
