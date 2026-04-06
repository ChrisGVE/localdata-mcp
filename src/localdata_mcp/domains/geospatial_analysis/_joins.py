"""
Spatial join and overlay operations for the geospatial analysis domain.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._base import (
    GeospatialLibrary,
    OverlayOperation,
    SpatialJoinType,
    _dependency_status,
)
from ._dependency import GeospatialDependencyChecker

logger = get_logger(__name__)


@dataclass
class SpatialJoinResult:
    """Results from a spatial join operation."""

    joined_data: pd.DataFrame
    left_geometry_column: str
    right_geometry_column: str
    join_type: SpatialJoinType
    match_counts: Dict[str, int]
    execution_time: float

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the join result."""
        return {
            "total_rows": len(self.joined_data),
            "join_type": self.join_type.value,
            "match_counts": self.match_counts,
            "execution_time_seconds": self.execution_time,
            "columns": list(self.joined_data.columns),
        }


@dataclass
class OverlayResult:
    """Results from a spatial overlay operation."""

    result_geometries: List[Any]
    result_data: pd.DataFrame
    operation: OverlayOperation
    input_counts: Dict[str, int]
    output_count: int
    execution_time: float

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the overlay result."""
        return {
            "operation": self.operation.value,
            "input_counts": self.input_counts,
            "output_count": self.output_count,
            "execution_time_seconds": self.execution_time,
            "columns": (
                list(self.result_data.columns) if self.result_data is not None else []
            ),
        }


class SpatialJoinEngine:
    """Core engine for spatial join operations with support for multiple backends."""

    def __init__(self):
        """Initialize the spatial join engine."""
        self.dependency_checker = GeospatialDependencyChecker()

    def _prepare_geometries(
        self, data: pd.DataFrame, geometry_column: str
    ) -> List[Any]:
        """Prepare geometries for spatial operations."""
        if geometry_column not in data.columns:
            raise ValueError(f"Geometry column '{geometry_column}' not found in data")

        geometries = []
        for idx, row in data.iterrows():
            geom = row[geometry_column]
            if isinstance(geom, (tuple, list)) and len(geom) == 2:
                if _dependency_status.is_available(GeospatialLibrary.SHAPELY):
                    from shapely.geometry import Point

                    geometries.append(Point(geom[0], geom[1]))
                else:
                    geometries.append(geom)
            else:
                geometries.append(geom)

        return geometries

    def _spatial_predicate_check(self, geom1, geom2, join_type):
        """Check spatial relationship between two geometries."""
        if not _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            if join_type == SpatialJoinType.INTERSECTS:
                if isinstance(geom1, (tuple, list)) and isinstance(
                    geom2, (tuple, list)
                ):
                    return (
                        abs(geom1[0] - geom2[0]) < 1e-6
                        and abs(geom1[1] - geom2[1]) < 1e-6
                    )
            return False

        try:
            if join_type == SpatialJoinType.INTERSECTS:
                return geom1.intersects(geom2)
            elif join_type == SpatialJoinType.WITHIN:
                return geom1.within(geom2)
            elif join_type == SpatialJoinType.CONTAINS:
                return geom1.contains(geom2)
            elif join_type == SpatialJoinType.TOUCHES:
                return geom1.touches(geom2)
            elif join_type == SpatialJoinType.OVERLAPS:
                return geom1.overlaps(geom2)
            elif join_type == SpatialJoinType.CROSSES:
                return geom1.crosses(geom2)
            elif join_type == SpatialJoinType.DISJOINT:
                return geom1.disjoint(geom2)
            else:
                return False
        except Exception:
            return False

    def _find_nearest_geometry(self, target_geom, candidate_geometries):
        """Find the index of the nearest geometry."""
        if not _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            if isinstance(target_geom, (tuple, list)):
                min_dist = float("inf")
                min_idx = -1
                for i, candidate in enumerate(candidate_geometries):
                    if isinstance(candidate, (tuple, list)):
                        dist = np.sqrt(
                            (target_geom[0] - candidate[0]) ** 2
                            + (target_geom[1] - candidate[1]) ** 2
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = i
                return min_idx
            return -1

        min_dist = float("inf")
        min_idx = -1

        for i, candidate in enumerate(candidate_geometries):
            try:
                dist = target_geom.distance(candidate)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            except Exception:
                continue

        return min_idx

    def spatial_join(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_geometry_col: str = "geometry",
        right_geometry_col: str = "geometry",
        join_type: SpatialJoinType = SpatialJoinType.INTERSECTS,
        how: str = "inner",
    ) -> SpatialJoinResult:
        """Perform spatial join between two datasets."""
        start_time = time.time()

        left_geometries = self._prepare_geometries(left_df, left_geometry_col)
        right_geometries = self._prepare_geometries(right_df, right_geometry_col)

        joined_rows = []
        match_counts = {"matches": 0, "no_matches": 0}

        for left_idx, left_geom in enumerate(left_geometries):
            matches_found = []

            if join_type == SpatialJoinType.NEAREST:
                nearest_idx = self._find_nearest_geometry(left_geom, right_geometries)
                if nearest_idx >= 0:
                    matches_found = [nearest_idx]
            else:
                for right_idx, right_geom in enumerate(right_geometries):
                    if self._spatial_predicate_check(left_geom, right_geom, join_type):
                        matches_found.append(right_idx)

            if matches_found:
                match_counts["matches"] += len(matches_found)
                for right_idx in matches_found:
                    joined_row = {}
                    for col in left_df.columns:
                        if col != left_geometry_col:
                            joined_row[f"{col}_left"] = left_df.iloc[left_idx][col]
                        else:
                            joined_row[left_geometry_col] = left_df.iloc[left_idx][col]

                    for col in right_df.columns:
                        if col != right_geometry_col:
                            joined_row[f"{col}_right"] = right_df.iloc[right_idx][col]
                        elif col != left_geometry_col:
                            joined_row[f"{right_geometry_col}_right"] = right_df.iloc[
                                right_idx
                            ][col]

                    joined_rows.append(joined_row)
            else:
                match_counts["no_matches"] += 1
                if how == "left":
                    joined_row = {}
                    for col in left_df.columns:
                        if col != left_geometry_col:
                            joined_row[f"{col}_left"] = left_df.iloc[left_idx][col]
                        else:
                            joined_row[left_geometry_col] = left_df.iloc[left_idx][col]

                    for col in right_df.columns:
                        if col != right_geometry_col:
                            joined_row[f"{col}_right"] = None
                        elif col != left_geometry_col:
                            joined_row[f"{right_geometry_col}_right"] = None

                    joined_rows.append(joined_row)

        if how == "right":
            matched_right_indices = set()
            for left_idx, left_geom in enumerate(left_geometries):
                for right_idx, right_geom in enumerate(right_geometries):
                    if self._spatial_predicate_check(left_geom, right_geom, join_type):
                        matched_right_indices.add(right_idx)

            for right_idx in range(len(right_geometries)):
                if right_idx not in matched_right_indices:
                    joined_row = {}
                    for col in left_df.columns:
                        if col != left_geometry_col:
                            joined_row[f"{col}_left"] = None
                        else:
                            joined_row[left_geometry_col] = None

                    for col in right_df.columns:
                        if col != right_geometry_col:
                            joined_row[f"{col}_right"] = right_df.iloc[right_idx][col]
                        else:
                            joined_row[right_geometry_col] = right_df.iloc[right_idx][
                                col
                            ]

                    joined_rows.append(joined_row)

        joined_df = pd.DataFrame(joined_rows) if joined_rows else pd.DataFrame()

        execution_time = time.time() - start_time

        return SpatialJoinResult(
            joined_data=joined_df,
            left_geometry_column=left_geometry_col,
            right_geometry_column=right_geometry_col,
            join_type=join_type,
            match_counts=match_counts,
            execution_time=execution_time,
        )
