"""
Coordinate reference system handling and transformations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._base import GeospatialLibrary, _dependency_status

logger = get_logger(__name__)


@dataclass
class CoordinateTransformation:
    """Container for coordinate transformation operations."""

    source_crs: str
    target_crs: str
    transformation_accuracy: Optional[float] = None
    transformation_method: Optional[str] = None
    is_valid: bool = True
    error_message: Optional[str] = None


class CoordinateReferenceSystem:
    """
    Coordinate Reference System handler with pyproj integration.

    Provides CRS validation, conversion, and transformation capabilities
    with fallback mechanisms when pyproj is not available.
    """

    # Common CRS definitions for fallback mode
    COMMON_CRS = {
        "EPSG:4326": {
            "name": "WGS 84",
            "type": "geographic",
            "unit": "degree",
            "authority": "EPSG",
            "code": "4326",
        },
        "EPSG:3857": {
            "name": "WGS 84 / Pseudo-Mercator",
            "type": "projected",
            "unit": "metre",
            "authority": "EPSG",
            "code": "3857",
        },
        "EPSG:4269": {
            "name": "NAD83",
            "type": "geographic",
            "unit": "degree",
            "authority": "EPSG",
            "code": "4269",
        },
        "EPSG:32633": {
            "name": "WGS 84 / UTM zone 33N",
            "type": "projected",
            "unit": "metre",
            "authority": "EPSG",
            "code": "32633",
        },
    }

    def __init__(self):
        """Initialize CRS handler."""
        self.pyproj_available = _dependency_status.is_available(
            GeospatialLibrary.PYPROJ
        )
        self._crs_cache = {}

        if self.pyproj_available:
            try:
                import pyproj

                self.pyproj = pyproj
                logger.debug("PyProj available for CRS operations")
            except ImportError:
                self.pyproj_available = False
                logger.warning("PyProj import failed, using fallback CRS handling")

    def validate_crs(self, crs_string: str) -> Dict[str, Any]:
        """
        Validate a coordinate reference system specification.

        Parameters
        ----------
        crs_string : str
            CRS specification (EPSG code, proj4 string, WKT, etc.)

        Returns
        -------
        validation_result : dict
            Validation results including validity, type, and metadata.
        """
        result = {
            "valid": False,
            "crs_string": crs_string,
            "crs_type": None,
            "authority": None,
            "code": None,
            "name": None,
            "unit": None,
            "area_of_use": None,
            "warnings": [],
        }

        if not crs_string:
            result["warnings"].append("Empty CRS specification")
            return result

        if self.pyproj_available:
            try:
                crs = self.pyproj.CRS.from_string(crs_string)
                result["valid"] = True
                result["crs_type"] = crs.type_name
                result["name"] = crs.name

                if crs.to_authority():
                    result["authority"] = crs.to_authority()[0]
                    result["code"] = crs.to_authority()[1]

                if hasattr(crs, "axis_info") and crs.axis_info:
                    result["unit"] = crs.axis_info[0].unit_name

                if hasattr(crs, "area_of_use") and crs.area_of_use:
                    result["area_of_use"] = {
                        "name": crs.area_of_use.name,
                        "bounds": [
                            crs.area_of_use.west,
                            crs.area_of_use.south,
                            crs.area_of_use.east,
                            crs.area_of_use.north,
                        ],
                    }

                self._crs_cache[crs_string] = crs

            except Exception as e:
                result["warnings"].append(f"PyProj CRS validation failed: {e}")
                result.update(self._fallback_crs_validation(crs_string))
        else:
            result.update(self._fallback_crs_validation(crs_string))

        return result

    def _fallback_crs_validation(self, crs_string: str) -> Dict[str, Any]:
        """Fallback CRS validation without pyproj."""
        result = {"valid": False, "warnings": []}

        crs_upper = crs_string.upper().strip()

        if crs_upper in self.COMMON_CRS:
            crs_info = self.COMMON_CRS[crs_upper]
            result.update(
                {
                    "valid": True,
                    "crs_type": crs_info["type"],
                    "authority": crs_info["authority"],
                    "code": crs_info["code"],
                    "name": crs_info["name"],
                    "unit": crs_info["unit"],
                }
            )
        elif any(common in crs_upper for common in ["WGS84", "WGS:84", "WGS 84"]):
            result.update(
                {
                    "valid": True,
                    "crs_type": "geographic",
                    "authority": "EPSG",
                    "code": "4326",
                    "name": "WGS 84",
                    "unit": "degree",
                }
            )
            result["warnings"].append("CRS assumed to be WGS84 (EPSG:4326)")
        else:
            result["warnings"].append("Cannot validate CRS without pyproj library")
            if "EPSG:" in crs_upper or "+proj=" in crs_upper:
                result["valid"] = True
                result["warnings"].append("CRS format recognized but not validated")

        return result

    def get_transformation(
        self, source_crs: str, target_crs: str
    ) -> CoordinateTransformation:
        """
        Create coordinate transformation between two CRS.

        Parameters
        ----------
        source_crs : str
            Source coordinate reference system.
        target_crs : str
            Target coordinate reference system.

        Returns
        -------
        transformation : CoordinateTransformation
            Transformation object with metadata.
        """
        transformation = CoordinateTransformation(
            source_crs=source_crs, target_crs=target_crs
        )

        if self.pyproj_available:
            try:
                source_validation = self.validate_crs(source_crs)
                target_validation = self.validate_crs(target_crs)

                if not source_validation["valid"]:
                    transformation.is_valid = False
                    transformation.error_message = f"Invalid source CRS: {source_crs}"
                    return transformation

                if not target_validation["valid"]:
                    transformation.is_valid = False
                    transformation.error_message = f"Invalid target CRS: {target_crs}"
                    return transformation

                source_crs_obj = self._crs_cache.get(
                    source_crs
                ) or self.pyproj.CRS.from_string(source_crs)
                target_crs_obj = self._crs_cache.get(
                    target_crs
                ) or self.pyproj.CRS.from_string(target_crs)

                transformer = self.pyproj.Transformer.from_crs(
                    source_crs_obj, target_crs_obj, always_xy=True
                )

                transformation.transformation_accuracy = getattr(
                    transformer, "accuracy", None
                )
                transformation.transformation_method = (
                    str(transformer.description)
                    if hasattr(transformer, "description")
                    else None
                )

                cache_key = f"{source_crs}_{target_crs}"
                if not hasattr(self, "_transformer_cache"):
                    self._transformer_cache = {}
                self._transformer_cache[cache_key] = transformer

            except Exception as e:
                transformation.is_valid = False
                transformation.error_message = f"Transformation creation failed: {e}"
        else:
            if source_crs == target_crs:
                transformation.transformation_method = "identity"
            elif self._is_geographic_crs(source_crs) and self._is_geographic_crs(
                target_crs
            ):
                transformation.transformation_method = "geographic_fallback"
            else:
                transformation.is_valid = False
                transformation.error_message = (
                    "Cannot create transformation without pyproj library"
                )

        return transformation

    def transform_coordinates(
        self,
        coordinates: Union[List[Tuple[float, float]], np.ndarray],
        source_crs: str,
        target_crs: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Transform coordinates between coordinate reference systems.

        Parameters
        ----------
        coordinates : list of tuples or ndarray
            Input coordinates as [(x, y), ...] or array.
        source_crs : str
            Source coordinate reference system.
        target_crs : str
            Target coordinate reference system.

        Returns
        -------
        transformed_coords : ndarray
            Transformed coordinates.
        warnings : list
            Any transformation warnings.
        """
        warnings_list = []

        if isinstance(coordinates, list):
            coords_array = np.array(coordinates)
        else:
            coords_array = np.asarray(coordinates)

        if coords_array.shape[1] < 2:
            raise ValueError("Coordinates must have at least 2 dimensions (x, y)")

        if source_crs == target_crs:
            warnings_list.append(
                "Source and target CRS are identical - no transformation needed"
            )
            return coords_array, warnings_list

        if self.pyproj_available:
            try:
                cache_key = f"{source_crs}_{target_crs}"
                if (
                    hasattr(self, "_transformer_cache")
                    and cache_key in self._transformer_cache
                ):
                    transformer = self._transformer_cache[cache_key]
                else:
                    transformation = self.get_transformation(source_crs, target_crs)
                    if not transformation.is_valid:
                        raise ValueError(transformation.error_message)
                    transformer = self._transformer_cache[cache_key]

                x_coords = coords_array[:, 0]
                y_coords = coords_array[:, 1]

                transformed_x, transformed_y = transformer.transform(x_coords, y_coords)

                result_coords = np.column_stack([transformed_x, transformed_y])

                if coords_array.shape[1] > 2:
                    result_coords = np.column_stack(
                        [result_coords, coords_array[:, 2:]]
                    )

                return result_coords, warnings_list

            except Exception as e:
                warnings_list.append(f"PyProj transformation failed: {e}")

        if self._is_geographic_crs(source_crs) and self._is_geographic_crs(target_crs):
            warnings_list.append(
                "Using fallback geographic transformation - may be inaccurate"
            )
            return coords_array, warnings_list
        else:
            raise ValueError(
                f"Cannot transform coordinates from {source_crs} to {target_crs} without pyproj library"
            )

    def _is_geographic_crs(self, crs_string: str) -> bool:
        """Check if CRS is geographic (latitude/longitude)."""
        crs_upper = crs_string.upper()
        geographic_indicators = ["4326", "WGS84", "WGS:84", "WGS 84", "4269", "NAD83"]
        return any(indicator in crs_upper for indicator in geographic_indicators)

    def get_crs_info(self, crs_string: str) -> Dict[str, Any]:
        """Get detailed information about a CRS."""
        validation = self.validate_crs(crs_string)

        info = {
            "crs_string": crs_string,
            "is_valid": validation["valid"],
            "name": validation.get("name", "Unknown"),
            "type": validation.get("crs_type", "Unknown"),
            "authority": validation.get("authority"),
            "code": validation.get("code"),
            "unit": validation.get("unit"),
            "area_of_use": validation.get("area_of_use"),
            "is_geographic": self._is_geographic_crs(crs_string),
            "warnings": validation.get("warnings", []),
        }

        return info

    def list_common_crs(self) -> Dict[str, Dict[str, str]]:
        """List commonly used coordinate reference systems."""
        return self.COMMON_CRS.copy()
