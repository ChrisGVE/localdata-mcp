"""
Spatial distance calculation and distance transformer for the geospatial analysis domain.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._coordinates import CoordinateReferenceSystem
from ._data import SpatialPoint

logger = get_logger(__name__)


class SpatialDistanceCalculator:
    """
    Comprehensive spatial distance calculation tools.

    Provides various distance metrics for spatial data including great circle
    distances, euclidean distances, and optimized batch calculations.
    """

    def __init__(self, default_crs: str = "EPSG:4326"):
        """
        Initialize distance calculator.

        Parameters
        ----------
        default_crs : str, default 'EPSG:4326'
            Default coordinate reference system for calculations.
        """
        self.default_crs = default_crs
        self.crs_handler = CoordinateReferenceSystem()

    def haversine_distance(
        self,
        lat1: Union[float, np.ndarray, Tuple[float, float]],
        lon1: Union[float, np.ndarray, Tuple[float, float], None] = None,
        lat2: Union[float, np.ndarray, None] = None,
        lon2: Union[float, np.ndarray, None] = None,
        unit: str = "km",
    ) -> Union[float, np.ndarray]:
        """
        Calculate great circle distance using haversine formula.

        Supports two calling conventions:
        - haversine_distance(point1_tuple, point2_tuple)
        - haversine_distance(lat1, lon1, lat2, lon2)
        """
        # Support tuple-based calling: haversine_distance((lat1,lon1), (lat2,lon2))
        if isinstance(lat1, tuple) and isinstance(lon1, tuple):
            point1, point2 = lat1, lon1
            lat1, lon1 = point1[0], point1[1]
            lat2, lon2 = point2[0], point2[1]

        lat1, lon1, lat2, lon2 = (
            np.asarray(lat1),
            np.asarray(lon1),
            np.asarray(lat2),
            np.asarray(lon2),
        )

        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        radius = {"km": 6371.0, "m": 6371000.0, "mi": 3959.0, "nmi": 3440.0}

        if unit not in radius:
            raise ValueError(f"Unsupported unit: {unit}. Use 'km', 'm', 'mi', or 'nmi'")

        distance = c * radius[unit]

        if distance.ndim == 0:
            return float(distance)
        return distance

    def euclidean_distance(
        self,
        x1: Union[float, np.ndarray, Tuple[float, float]],
        y1: Union[float, np.ndarray, None] = None,
        x2: Union[float, np.ndarray, Tuple[float, float], None] = None,
        y2: Union[float, np.ndarray, None] = None,
        z1: Optional[Union[float, np.ndarray]] = None,
        z2: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate euclidean distance between points.

        Supports two calling conventions:
        - euclidean_distance(point1_tuple, point2_tuple)
        - euclidean_distance(x1, y1, x2, y2)
        """
        if isinstance(x1, tuple) and isinstance(y1, tuple):
            point1, point2 = x1, y1
            x1, y1_val = point1[0], point1[1]
            x2, y2 = point2[0], point2[1]
            z1 = point1[2] if len(point1) > 2 else None
            z2 = point2[2] if len(point2) > 2 else None
            y1 = y1_val

        x1, y1, x2, y2 = np.asarray(x1), np.asarray(y1), np.asarray(x2), np.asarray(y2)

        dx = x2 - x1
        dy = y2 - y1

        if z1 is not None and z2 is not None:
            z1, z2 = np.asarray(z1), np.asarray(z2)
            dz = z2 - z1
            distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            distance = np.sqrt(dx * dx + dy * dy)

        if distance.ndim == 0:
            return float(distance)
        return distance

    def manhattan_distance(
        self,
        x1: Union[float, np.ndarray, Tuple[float, float]],
        y1: Union[float, np.ndarray, Tuple[float, float], None] = None,
        x2: Union[float, np.ndarray, None] = None,
        y2: Union[float, np.ndarray, None] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate Manhattan (taxicab) distance between points.

        Supports two calling conventions:
        - manhattan_distance(point1_tuple, point2_tuple)
        - manhattan_distance(x1, y1, x2, y2)
        """
        if isinstance(x1, tuple) and isinstance(y1, tuple):
            point1, point2 = x1, y1
            x1, y1_val = point1[0], point1[1]
            x2, y2 = point2[0], point2[1]
            y1 = y1_val

        x1, y1, x2, y2 = np.asarray(x1), np.asarray(y1), np.asarray(x2), np.asarray(y2)

        distance = np.abs(x2 - x1) + np.abs(y2 - y1)

        if distance.ndim == 0:
            return float(distance)
        return distance

    def calculate_distance(
        self,
        point1: Union[SpatialPoint, Tuple[float, float]],
        point2: Union[SpatialPoint, Tuple[float, float]],
        method: str = "auto",
        crs: Optional[str] = None,
    ) -> float:
        """Calculate distance between two spatial points using appropriate method."""
        if isinstance(point1, tuple):
            point1 = SpatialPoint(x=point1[0], y=point1[1], crs=crs)
        if isinstance(point2, tuple):
            point2 = SpatialPoint(x=point2[0], y=point2[1], crs=crs)

        effective_crs = crs or point1.crs or point2.crs or self.default_crs

        if method == "auto":
            if self.crs_handler._is_geographic_crs(effective_crs):
                method = "haversine"
            else:
                method = "euclidean"

        if method == "haversine":
            return self.haversine_distance(point1.y, point1.x, point2.y, point2.x)
        elif method == "euclidean":
            return self.euclidean_distance(point1.x, point1.y, point2.x, point2.y)
        elif method == "manhattan":
            return self.manhattan_distance(point1.x, point1.y, point2.x, point2.y)
        else:
            raise ValueError(f"Unsupported distance method: {method}")

    def distance_matrix(
        self,
        points1: List[Union[SpatialPoint, Tuple[float, float]]],
        points2: Optional[List[Union[SpatialPoint, Tuple[float, float]]]] = None,
        method: str = "auto",
        crs: Optional[str] = None,
        symmetric: bool = True,
    ) -> np.ndarray:
        """Calculate distance matrix between sets of points."""
        if points2 is None:
            points2 = points1
            is_self_distance = True
        else:
            is_self_distance = False

        coords1 = self._extract_coordinates(points1, crs)
        coords2 = self._extract_coordinates(points2, crs)

        effective_crs = crs or self.default_crs
        if method == "auto":
            if self.crs_handler._is_geographic_crs(effective_crs):
                method = "haversine"
            else:
                method = "euclidean"

        n1, n2 = len(coords1), len(coords2)
        distances = np.zeros((n1, n2))

        if method == "haversine":
            lat1 = coords1[:, 1:2]
            lon1 = coords1[:, 0:1]
            lat2 = coords2[:, 1].reshape(1, -1)
            lon2 = coords2[:, 0].reshape(1, -1)
            distances = self.haversine_distance(lat1, lon1, lat2, lon2)

        elif method == "euclidean":
            x1 = coords1[:, 0:1]
            y1 = coords1[:, 1:2]
            x2 = coords2[:, 0].reshape(1, -1)
            y2 = coords2[:, 1].reshape(1, -1)
            distances = self.euclidean_distance(x1, y1, x2, y2)

        elif method == "manhattan":
            x1 = coords1[:, 0:1]
            y1 = coords1[:, 1:2]
            x2 = coords2[:, 0].reshape(1, -1)
            y2 = coords2[:, 1].reshape(1, -1)
            distances = self.manhattan_distance(x1, y1, x2, y2)

        if is_self_distance and symmetric:
            i_upper = np.triu_indices(n1, k=1)
            distances[i_upper] = distances.T[i_upper]

        return distances

    def _extract_coordinates(
        self,
        points: List[Union[SpatialPoint, Tuple[float, float]]],
        crs: Optional[str] = None,
    ) -> np.ndarray:
        """Extract coordinate array from points list."""
        coords = []
        for point in points:
            if isinstance(point, tuple):
                coords.append(point)
            elif isinstance(point, SpatialPoint):
                coords.append((point.x, point.y))
            else:
                raise ValueError(f"Unsupported point type: {type(point)}")

        return np.array(coords)

    def nearest_neighbors(
        self,
        query_points: List[Union[SpatialPoint, Tuple[float, float]]],
        reference_points: List[Union[SpatialPoint, Tuple[float, float]]],
        k: int = 1,
        method: str = "auto",
        crs: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for each query point.

        Returns
        -------
        distances : ndarray
            Distances to k nearest neighbors for each query point.
        indices : ndarray
            Indices of k nearest neighbors for each query point.
        """
        dist_matrix = self.distance_matrix(
            query_points, reference_points, method=method, crs=crs
        )

        if k >= dist_matrix.shape[1]:
            k = dist_matrix.shape[1]

        indices = np.argpartition(dist_matrix, k - 1, axis=1)[:, :k]
        distances = np.take_along_axis(dist_matrix, indices, axis=1)

        sort_indices = np.argsort(distances, axis=1)
        distances = np.take_along_axis(distances, sort_indices, axis=1)
        indices = np.take_along_axis(indices, sort_indices, axis=1)

        return distances, indices
