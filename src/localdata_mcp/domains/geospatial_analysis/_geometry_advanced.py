"""
Extended geometric operations: bounding box, intersection, union.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ...logging_manager import get_logger
from ._data import SpatialPoint
from ._geometry import GeometricResult, _GeometricOperationsBase

logger = get_logger(__name__)


class GeometricOperations(_GeometricOperationsBase):
    """
    Full geometric operations with Shapely integration.

    Extends _GeometricOperationsBase with bounding box, intersection,
    and union operations.
    """

    def bounding_box(
        self, geometry: Union[Any, List[Union[SpatialPoint, Tuple[float, float]]]]
    ) -> GeometricResult:
        """Calculate bounding box (minimum bounding rectangle)."""
        if self.shapely_available and hasattr(geometry, "bounds"):
            bounds = geometry.bounds
            minx, miny, maxx, maxy = bounds

            bbox_coords = [
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy),
                (minx, miny),
            ]
            bbox_polygon = self.create_polygon(bbox_coords)

            return GeometricResult(
                operation="bounding_box",
                result=bbox_polygon,
                geometry_type="polygon",
                properties={
                    "bounds": bounds,
                    "width": maxx - minx,
                    "height": maxy - miny,
                    "area": (maxx - minx) * (maxy - miny),
                    "center": ((minx + maxx) / 2, (miny + maxy) / 2),
                },
            )
        else:
            if isinstance(geometry, list):
                coords = []
                for point in geometry:
                    if isinstance(point, tuple):
                        coords.append(point)
                    elif isinstance(point, SpatialPoint):
                        coords.append((point.x, point.y))
            else:
                return GeometricResult(
                    operation="bounding_box",
                    result=None,
                    warnings=["Cannot calculate bounding box for this geometry type"],
                )

            if not coords:
                return GeometricResult(
                    operation="bounding_box",
                    result=None,
                    warnings=["No coordinates found"],
                )

            bounds = self._calculate_bounds(coords)
            minx, miny, maxx, maxy = bounds

            bbox_coords = [
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy),
                (minx, miny),
            ]
            bbox_polygon = self.create_polygon(bbox_coords)

            return GeometricResult(
                operation="bounding_box",
                result=bbox_polygon,
                geometry_type="polygon",
                properties={
                    "bounds": bounds,
                    "width": maxx - minx,
                    "height": maxy - miny,
                    "area": (maxx - minx) * (maxy - miny),
                    "center": ((minx + maxx) / 2, (miny + maxy) / 2),
                },
            )

    def intersection(self, geom1: Any, geom2: Any) -> GeometricResult:
        """Calculate intersection of two geometries."""
        if self.shapely_available:
            try:
                intersection = geom1.intersection(geom2)

                return GeometricResult(
                    operation="intersection",
                    result=intersection,
                    geometry_type=intersection.geom_type,
                    properties={
                        "is_empty": intersection.is_empty,
                        "area": getattr(intersection, "area", 0),
                        "length": getattr(intersection, "length", 0),
                    },
                )
            except Exception as e:
                return GeometricResult(
                    operation="intersection",
                    result=None,
                    warnings=[f"Shapely intersection failed: {e}"],
                )
        else:
            return GeometricResult(
                operation="intersection",
                result=None,
                warnings=["Intersection operation requires Shapely library"],
            )

    def union(self, geometries: List[Any]) -> GeometricResult:
        """Calculate union of multiple geometries."""
        if self.shapely_available:
            try:
                if len(geometries) == 1:
                    union_result = geometries[0]
                else:
                    union_result = self.shapely_ops["unary_union"](geometries)

                return GeometricResult(
                    operation="union",
                    result=union_result,
                    geometry_type=union_result.geom_type,
                    properties={
                        "input_count": len(geometries),
                        "area": getattr(union_result, "area", 0),
                        "length": getattr(union_result, "length", 0),
                    },
                )
            except Exception as e:
                return GeometricResult(
                    operation="union",
                    result=None,
                    warnings=[f"Shapely union failed: {e}"],
                )
        else:
            return GeometricResult(
                operation="union",
                result=None,
                warnings=["Union operation requires Shapely library"],
            )
