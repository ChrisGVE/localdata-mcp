"""
Spatial overlay, aggregation, and join/overlay transformers.
"""

import time
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
from ._joins import OverlayResult, SpatialJoinEngine, SpatialJoinResult

logger = get_logger(__name__)


class SpatialOverlayEngine:
    """Core engine for spatial overlay operations."""

    def __init__(self):
        """Initialize the spatial overlay engine."""
        self.dependency_checker = GeospatialDependencyChecker()

    def _perform_geometric_operation(self, geom1, geom2, operation):
        """Perform geometric operation between two geometries."""
        if not _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            logger.warning(
                f"Shapely not available, overlay operation {operation.value} not supported"
            )
            return None

        try:
            if operation == OverlayOperation.INTERSECTION:
                return geom1.intersection(geom2)
            elif operation == OverlayOperation.UNION:
                return geom1.union(geom2)
            elif operation == OverlayOperation.DIFFERENCE:
                return geom1.difference(geom2)
            elif operation == OverlayOperation.SYMMETRIC_DIFFERENCE:
                return geom1.symmetric_difference(geom2)
            else:
                return None
        except Exception as e:
            logger.warning(f"Geometric operation failed: {e}")
            return None

    def spatial_overlay(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        operation: OverlayOperation,
        left_geometry_col: str = "geometry",
        right_geometry_col: str = "geometry",
        keep_geom_type: bool = True,
    ) -> OverlayResult:
        """Perform spatial overlay operation between two datasets."""
        start_time = time.time()

        if not _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            raise ValueError("Shapely is required for spatial overlay operations")

        left_geometries = self._prepare_geometries(left_df, left_geometry_col)
        right_geometries = self._prepare_geometries(right_df, right_geometry_col)

        result_geometries = []
        result_rows = []

        input_counts = {"left_features": len(left_df), "right_features": len(right_df)}

        for left_idx, left_geom in enumerate(left_geometries):
            for right_idx, right_geom in enumerate(right_geometries):
                result_geom = self._perform_geometric_operation(
                    left_geom, right_geom, operation
                )

                if result_geom is not None and not result_geom.is_empty:
                    if keep_geom_type:
                        left_geom_type = type(left_geom).__name__
                        result_geom_type = type(result_geom).__name__
                        if left_geom_type != result_geom_type:
                            continue

                    result_geometries.append(result_geom)

                    result_row = {"geometry": result_geom}

                    for col in left_df.columns:
                        if col != left_geometry_col:
                            result_row[f"{col}_1"] = left_df.iloc[left_idx][col]

                    for col in right_df.columns:
                        if col != right_geometry_col:
                            result_row[f"{col}_2"] = right_df.iloc[right_idx][col]

                    result_rows.append(result_row)

        result_df = pd.DataFrame(result_rows) if result_rows else pd.DataFrame()

        execution_time = time.time() - start_time

        return OverlayResult(
            result_geometries=result_geometries,
            result_data=result_df,
            operation=operation,
            input_counts=input_counts,
            output_count=len(result_geometries),
            execution_time=execution_time,
        )

    def _prepare_geometries(self, data, geometry_column):
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


class SpatialAggregator:
    """Tools for spatial aggregation operations."""

    def __init__(self):
        """Initialize the spatial aggregator."""
        self.join_engine = SpatialJoinEngine()

    def aggregate_by_geometry(
        self,
        point_df: pd.DataFrame,
        polygon_df: pd.DataFrame,
        value_column: str,
        aggregation_functions: List[str] = ["mean", "sum", "count"],
        point_geometry_col: str = "geometry",
        polygon_geometry_col: str = "geometry",
    ) -> pd.DataFrame:
        """Aggregate point values within polygons."""
        join_result = self.join_engine.spatial_join(
            point_df,
            polygon_df,
            left_geometry_col=point_geometry_col,
            right_geometry_col=polygon_geometry_col,
            join_type=SpatialJoinType.WITHIN,
            how="inner",
        )

        if join_result.joined_data.empty:
            result_df = polygon_df.copy()
            for func in aggregation_functions:
                result_df[f"{value_column}_{func}"] = np.nan
            return result_df

        polygon_cols = [
            col for col in join_result.joined_data.columns if col.endswith("_right")
        ]

        if not polygon_cols:
            raise ValueError("No polygon columns found in join result")

        agg_dict = {}
        value_col_left = f"{value_column}_left"

        if value_col_left not in join_result.joined_data.columns:
            raise ValueError(f"Value column '{value_column}' not found in joined data")

        for func in aggregation_functions:
            if func in ["mean", "sum", "std", "min", "max", "median"]:
                agg_dict[f"{value_column}_{func}"] = (value_col_left, func)
            elif func == "count":
                agg_dict[f"{value_column}_count"] = (value_col_left, "count")
            else:
                logger.warning(f"Aggregation function '{func}' not supported")

        grouped = join_result.joined_data.groupby(polygon_cols).agg(agg_dict)
        grouped.columns = [col[0] for col in grouped.columns]
        grouped = grouped.reset_index()

        polygon_mapping = {}
        for col in polygon_df.columns:
            suffixed_col = f"{col}_right"
            if suffixed_col in polygon_cols:
                polygon_mapping[suffixed_col] = col

        grouped = grouped.rename(columns=polygon_mapping)

        result_df = polygon_df.merge(
            grouped, on=list(polygon_mapping.values()), how="left"
        )

        for func in aggregation_functions:
            col_name = f"{value_column}_{func}"
            if col_name in result_df.columns:
                if func == "count":
                    result_df[col_name] = result_df[col_name].fillna(0)
                else:
                    result_df[col_name] = result_df[col_name].fillna(np.nan)

        return result_df


class SpatialJoinTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for spatial join operations."""

    def __init__(
        self,
        right_data: pd.DataFrame,
        left_geometry_col: str = "geometry",
        right_geometry_col: str = "geometry",
        join_type: SpatialJoinType = SpatialJoinType.INTERSECTS,
        how: str = "inner",
    ):
        self.right_data = right_data
        self.left_geometry_col = left_geometry_col
        self.right_geometry_col = right_geometry_col
        self.join_type = join_type
        self.how = how

    def fit(self, X, y=None):
        self.join_engine_ = SpatialJoinEngine()
        return self

    def transform(self, X):
        check_is_fitted(self, "join_engine_")

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "Spatial join requires DataFrame input with geometry column"
            )

        result = self.join_engine_.spatial_join(
            X,
            self.right_data,
            left_geometry_col=self.left_geometry_col,
            right_geometry_col=self.right_geometry_col,
            join_type=self.join_type,
            how=self.how,
        )

        return result.joined_data


class SpatialOverlayTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for spatial overlay operations."""

    def __init__(
        self,
        right_data: pd.DataFrame,
        operation: OverlayOperation,
        left_geometry_col: str = "geometry",
        right_geometry_col: str = "geometry",
        keep_geom_type: bool = True,
    ):
        self.right_data = right_data
        self.operation = operation
        self.left_geometry_col = left_geometry_col
        self.right_geometry_col = right_geometry_col
        self.keep_geom_type = keep_geom_type

    def fit(self, X, y=None):
        self.overlay_engine_ = SpatialOverlayEngine()
        return self

    def transform(self, X):
        check_is_fitted(self, "overlay_engine_")

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "Spatial overlay requires DataFrame input with geometry column"
            )

        result = self.overlay_engine_.spatial_overlay(
            X,
            self.right_data,
            operation=self.operation,
            left_geometry_col=self.left_geometry_col,
            right_geometry_col=self.right_geometry_col,
            keep_geom_type=self.keep_geom_type,
        )

        return result.result_data


# High-level convenience functions
def perform_spatial_join(
    left_data,
    right_data,
    join_type="intersects",
    left_geometry_col="geometry",
    right_geometry_col="geometry",
    how="inner",
) -> SpatialJoinResult:
    """Perform spatial join between two datasets."""
    join_type_enum = SpatialJoinType(join_type)
    engine = SpatialJoinEngine()

    return engine.spatial_join(
        left_data,
        right_data,
        left_geometry_col=left_geometry_col,
        right_geometry_col=right_geometry_col,
        join_type=join_type_enum,
        how=how,
    )


def perform_spatial_overlay(
    left_data,
    right_data,
    operation="intersection",
    left_geometry_col="geometry",
    right_geometry_col="geometry",
    keep_geom_type=True,
) -> OverlayResult:
    """Perform spatial overlay operation between two datasets."""
    operation_enum = OverlayOperation(operation)
    engine = SpatialOverlayEngine()

    return engine.spatial_overlay(
        left_data,
        right_data,
        operation=operation_enum,
        left_geometry_col=left_geometry_col,
        right_geometry_col=right_geometry_col,
        keep_geom_type=keep_geom_type,
    )


def aggregate_points_in_polygons(
    point_data,
    polygon_data,
    value_column,
    aggregation_functions=["mean", "sum", "count"],
    point_geometry_col="geometry",
    polygon_geometry_col="geometry",
) -> pd.DataFrame:
    """Aggregate point values within polygon boundaries."""
    aggregator = SpatialAggregator()

    return aggregator.aggregate_by_geometry(
        point_data,
        polygon_data,
        value_column=value_column,
        aggregation_functions=aggregation_functions,
        point_geometry_col=point_geometry_col,
        polygon_geometry_col=polygon_geometry_col,
    )
