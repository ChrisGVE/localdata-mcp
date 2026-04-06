"""
Spatial data structures and validation for the geospatial analysis domain.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...logging_manager import get_logger
from ._base import GeospatialLibrary, _dependency_status

logger = get_logger(__name__)


@dataclass
class SpatialPoint:
    """Represents a spatial point with coordinates and optional attributes."""

    x: float
    y: float
    z: Optional[float] = None
    crs: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Synchronize properties and attributes fields."""
        if self.properties is not None and self.attributes is None:
            self.attributes = self.properties
        elif self.attributes is not None and self.properties is None:
            self.properties = self.attributes

    def to_tuple(self, include_z: bool = True) -> Tuple[float, ...]:
        """Convert point to coordinate tuple."""
        if include_z and self.z is not None:
            return (self.x, self.y, self.z)
        return (self.x, self.y)

    def distance_to(self, other: "SpatialPoint", method: str = "euclidean") -> float:
        """Calculate distance to another point."""
        if method == "euclidean":
            dx = self.x - other.x
            dy = self.y - other.y
            if self.z is not None and other.z is not None:
                dz = self.z - other.z
                return math.sqrt(dx * dx + dy * dy + dz * dz)
            return math.sqrt(dx * dx + dy * dy)
        elif method == "haversine":
            return _haversine_distance(self.y, self.x, other.y, other.x)
        else:
            raise ValueError(f"Unsupported distance method: {method}")


@dataclass
class SpatialDataFrame:
    """Wrapper for spatial data with optional geopandas integration."""

    data: pd.DataFrame
    geometry_column: str = "geometry"
    crs: Optional[str] = None
    _geopandas_df: Optional[Any] = None  # GeoPandas GeoDataFrame when available

    def __post_init__(self):
        """Initialize spatial data frame."""
        # Try to create geopandas GeoDataFrame if available
        if _dependency_status.is_available(GeospatialLibrary.GEOPANDAS):
            try:
                import geopandas as gpd

                if self.geometry_column in self.data.columns:
                    self._geopandas_df = gpd.GeoDataFrame(
                        self.data, geometry=self.geometry_column, crs=self.crs
                    )
            except Exception as e:
                logger.warning(f"Failed to create GeoDataFrame: {e}")

    @property
    def is_geopandas_enabled(self) -> bool:
        """Check if geopandas integration is active."""
        return self._geopandas_df is not None

    def get_points(self) -> List[SpatialPoint]:
        """Extract spatial points from the data."""
        points = []
        if self.is_geopandas_enabled:
            for idx, row in self._geopandas_df.iterrows():
                geom = row[self.geometry_column]
                if hasattr(geom, "x") and hasattr(geom, "y"):
                    attrs = {
                        col: row[col]
                        for col in self.data.columns
                        if col != self.geometry_column
                    }
                    points.append(
                        SpatialPoint(x=geom.x, y=geom.y, crs=self.crs, attributes=attrs)
                    )
        else:
            x_col = "x" if "x" in self.data.columns else "longitude"
            y_col = "y" if "y" in self.data.columns else "latitude"

            if x_col in self.data.columns and y_col in self.data.columns:
                for idx, row in self.data.iterrows():
                    attrs = {
                        col: row[col]
                        for col in self.data.columns
                        if col not in [x_col, y_col]
                    }
                    points.append(
                        SpatialPoint(
                            x=row[x_col], y=row[y_col], crs=self.crs, attributes=attrs
                        )
                    )
        return points

    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get spatial bounds (minx, miny, maxx, maxy)."""
        if self.is_geopandas_enabled:
            return tuple(self._geopandas_df.total_bounds)
        else:
            points = self.get_points()
            if not points:
                return None
            xs = [p.x for p in points]
            ys = [p.y for p in points]
            return (min(xs), min(ys), max(xs), max(ys))


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    r = 6371
    return c * r


class SpatialDataValidator:
    """Validates spatial data quality and consistency."""

    @staticmethod
    def validate_coordinates(points: List[SpatialPoint]) -> Dict[str, Any]:
        """Validate coordinate data quality."""
        results = {
            "total_points": len(points),
            "valid_points": 0,
            "invalid_coordinates": [],
            "coordinate_ranges": {},
            "duplicate_coordinates": 0,
            "warnings": [],
        }

        if not points:
            results["warnings"].append("No points provided for validation")
            return results

        coordinates = set()
        xs, ys = [], []

        for i, point in enumerate(points):
            if (
                isinstance(point.x, (int, float))
                and not math.isnan(point.x)
                and isinstance(point.y, (int, float))
                and not math.isnan(point.y)
            ):
                results["valid_points"] += 1

                coord_tuple = (point.x, point.y)
                if coord_tuple in coordinates:
                    results["duplicate_coordinates"] += 1
                else:
                    coordinates.add(coord_tuple)

                xs.append(point.x)
                ys.append(point.y)
            else:
                results["invalid_coordinates"].append(i)

        if xs and ys:
            results["coordinate_ranges"] = {
                "x_range": (min(xs), max(xs)),
                "y_range": (min(ys), max(ys)),
                "centroid": (sum(xs) / len(xs), sum(ys) / len(ys)),
            }

            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)

            if max(abs(min(xs)), abs(max(xs))) > 180:
                results["warnings"].append(
                    "X coordinates exceed typical longitude range (-180 to 180)"
                )
            if max(abs(min(ys)), abs(max(ys))) > 90:
                results["warnings"].append(
                    "Y coordinates exceed typical latitude range (-90 to 90)"
                )
            if x_range > 360 or y_range > 180:
                results["warnings"].append(
                    "Coordinate ranges suggest projected coordinates, not geographic"
                )

        return results

    @staticmethod
    def validate_crs(crs_string: str) -> Dict[str, Any]:
        """Validate coordinate reference system specification."""
        results = {
            "valid": False,
            "crs_type": None,
            "authority": None,
            "code": None,
            "warnings": [],
        }

        if not crs_string:
            results["warnings"].append("No CRS specified")
            return results

        if _dependency_status.is_available(GeospatialLibrary.PYPROJ):
            try:
                import pyproj

                crs = pyproj.CRS.from_string(crs_string)
                results["valid"] = True
                results["crs_type"] = crs.type_name
                if crs.to_authority():
                    results["authority"] = crs.to_authority()[0]
                    results["code"] = crs.to_authority()[1]
            except Exception as e:
                results["warnings"].append(f"Invalid CRS: {e}")
        else:
            common_crs = ["EPSG:4326", "EPSG:3857", "WGS84", "WGS:84"]
            if crs_string.upper() in [c.upper() for c in common_crs]:
                results["valid"] = True
                results["crs_type"] = (
                    "Geographic" if "4326" in crs_string else "Projected"
                )
            else:
                results["warnings"].append("Cannot validate CRS without pyproj library")

        return results
