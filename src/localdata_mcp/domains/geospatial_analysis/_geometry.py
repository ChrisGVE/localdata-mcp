"""
Geometric operations and spatial indexing for the geospatial analysis domain.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...logging_manager import get_logger
from ._base import GeospatialLibrary, _dependency_status
from ._data import SpatialPoint

logger = get_logger(__name__)


@dataclass
class GeometricResult:
    """Container for geometric operation results."""

    operation: str
    result: Any
    geometry_type: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class _GeometricOperationsBase:
    """
    Base geometric operations with Shapely integration.

    Provides point-in-polygon testing, buffer analysis, and convex hulls
    with fallback mechanisms when Shapely is not available.
    Extended by GeometricOperations in _geometry_advanced.py.
    """

    def __init__(self):
        """Initialize geometric operations handler."""
        self.shapely_available = _dependency_status.is_available(
            GeospatialLibrary.SHAPELY
        )
        self._geometry_cache = {}

        if self.shapely_available:
            try:
                import shapely.speedups
                from shapely import affinity, validation
                from shapely.geometry import (
                    LineString,
                    MultiPoint,
                    MultiPolygon,
                    Point,
                    Polygon,
                )
                from shapely.ops import cascaded_union, unary_union

                if shapely.speedups.available:
                    shapely.speedups.enable()

                self.shapely_geom = {
                    "Point": Point,
                    "Polygon": Polygon,
                    "LineString": LineString,
                    "MultiPoint": MultiPoint,
                    "MultiPolygon": MultiPolygon,
                }
                self.shapely_ops = {
                    "unary_union": unary_union,
                    "cascaded_union": (
                        cascaded_union
                        if hasattr(shapely.ops, "cascaded_union")
                        else unary_union
                    ),
                }
                self.shapely_affinity = affinity
                self.shapely_validation = validation

                logger.debug("Shapely available for geometric operations")
            except ImportError:
                self.shapely_available = False
                logger.warning(
                    "Shapely import failed, using fallback geometric operations"
                )

    def create_point(self, x: float, y: float, z: Optional[float] = None) -> Any:
        """Create a point geometry."""
        if self.shapely_available:
            if z is not None:
                return self.shapely_geom["Point"](x, y, z)
            return self.shapely_geom["Point"](x, y)
        else:
            return SpatialPoint(x=x, y=y, z=z)

    def create_polygon(self, coordinates: List[Tuple[float, float]]) -> Any:
        """Create a polygon geometry."""
        if self.shapely_available:
            return self.shapely_geom["Polygon"](coordinates)
        else:
            return {
                "type": "polygon",
                "coordinates": coordinates,
                "bounds": self._calculate_bounds(coordinates),
            }

    def point_in_polygon(
        self, point: Union[SpatialPoint, Tuple[float, float]], polygon: Any
    ) -> bool:
        """Test if point is inside polygon."""
        if isinstance(point, tuple):
            px, py = point
        elif isinstance(point, SpatialPoint):
            px, py = point.x, point.y
        else:
            raise ValueError("Point must be tuple or SpatialPoint")

        if self.shapely_available:
            if hasattr(polygon, "contains"):
                test_point = self.create_point(px, py)
                return polygon.contains(test_point)
            elif hasattr(polygon, "coords"):
                return False
            else:
                raise ValueError("Invalid polygon object for Shapely")
        else:
            if isinstance(polygon, dict) and polygon.get("type") == "polygon":
                coords = polygon["coordinates"]
                return self._point_in_polygon_raycast(px, py, coords)
            else:
                raise ValueError("Invalid polygon object for fallback method")

    def _point_in_polygon_raycast(
        self, x: float, y: float, polygon_coords: List[Tuple[float, float]]
    ) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon_coords)
        inside = False

        p1x, p1y = polygon_coords[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_coords[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def buffer_analysis(
        self, geometry: Any, distance: float, resolution: int = 16
    ) -> GeometricResult:
        """Create buffer around geometry."""
        if self.shapely_available:
            try:
                if isinstance(geometry, SpatialPoint):
                    geometry = self.create_point(geometry.x, geometry.y)

                buffered = geometry.buffer(distance, resolution=resolution)

                return GeometricResult(
                    operation="buffer",
                    result=buffered,
                    geometry_type=buffered.geom_type,
                    properties={
                        "original_area": getattr(geometry, "area", 0),
                        "buffer_area": buffered.area,
                        "buffer_distance": distance,
                        "resolution": resolution,
                    },
                )
            except Exception as e:
                return GeometricResult(
                    operation="buffer",
                    result=None,
                    warnings=[f"Shapely buffer failed: {e}"],
                )
        else:
            if isinstance(geometry, SpatialPoint):
                points = []
                for i in range(resolution):
                    angle = 2 * math.pi * i / resolution
                    x = geometry.x + distance * math.cos(angle)
                    y = geometry.y + distance * math.sin(angle)
                    points.append((x, y))
                points.append(points[0])

                buffer_poly = self.create_polygon(points)

                return GeometricResult(
                    operation="buffer",
                    result=buffer_poly,
                    geometry_type="polygon",
                    properties={
                        "buffer_distance": distance,
                        "resolution": resolution,
                        "approximated": True,
                    },
                    warnings=[
                        "Using approximate circular buffer (Shapely not available)"
                    ],
                )
            else:
                return GeometricResult(
                    operation="buffer",
                    result=None,
                    warnings=[
                        "Buffer analysis not available for this geometry type without Shapely"
                    ],
                )

    def convex_hull(
        self, points: List[Union[SpatialPoint, Tuple[float, float]]]
    ) -> GeometricResult:
        """Calculate convex hull of point set."""
        if not points:
            return GeometricResult(
                operation="convex_hull", result=None, warnings=["No points provided"]
            )

        if self.shapely_available:
            try:
                shapely_points = []
                for point in points:
                    if isinstance(point, tuple):
                        shapely_points.append(self.create_point(point[0], point[1]))
                    elif isinstance(point, SpatialPoint):
                        shapely_points.append(self.create_point(point.x, point.y))

                multipoint = self.shapely_geom["MultiPoint"](shapely_points)
                hull = multipoint.convex_hull

                return GeometricResult(
                    operation="convex_hull",
                    result=hull,
                    geometry_type=hull.geom_type,
                    properties={
                        "input_points": len(points),
                        "hull_area": getattr(hull, "area", 0),
                        "hull_length": getattr(hull, "length", 0),
                    },
                )
            except Exception as e:
                logger.warning(f"Shapely convex hull failed: {e}")

        coords = []
        for point in points:
            if isinstance(point, tuple):
                coords.append(point)
            elif isinstance(point, SpatialPoint):
                coords.append((point.x, point.y))

        hull_coords = self._graham_scan(coords)
        hull_polygon = self.create_polygon(hull_coords)

        return GeometricResult(
            operation="convex_hull",
            result=hull_polygon,
            geometry_type="polygon",
            properties={
                "input_points": len(points),
                "hull_vertices": len(hull_coords),
                "approximated": not self.shapely_available,
            },
            warnings=(
                ["Using Graham scan algorithm (Shapely not available)"]
                if not self.shapely_available
                else []
            ),
        )

    def _graham_scan(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Graham scan algorithm for convex hull."""

        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        points = sorted(set(points))
        if len(points) <= 3:
            return points + [points[0]] if points else []

        lower = []
        for p in points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        if hull:
            hull.append(hull[0])

        return hull

    def _calculate_bounds(
        self, coords: List[Tuple[float, float]]
    ) -> Tuple[float, float, float, float]:
        """Calculate bounding box from coordinate list."""
        if not coords:
            return (0, 0, 0, 0)

        xs, ys = zip(*coords)
        return (min(xs), min(ys), max(xs), max(ys))
