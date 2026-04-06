"""
Sklearn-compatible geometry transformer for the geospatial analysis domain.
"""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._data import SpatialPoint
from ._geometry_advanced import GeometricOperations

logger = get_logger(__name__)


class SpatialGeometryTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for spatial geometric operations.

    This transformer performs geometric operations on spatial data
    and can be used within sklearn pipelines for spatial feature engineering.
    """

    def __init__(
        self,
        operations: List[str] = ["buffer", "convex_hull"],
        buffer_distance: float = 1.0,
        coordinate_columns: Optional[List[str]] = None,
    ):
        """
        Initialize spatial geometry transformer.

        Parameters
        ----------
        operations : list of str, default ['buffer', 'convex_hull']
            Geometric operations to perform.
        buffer_distance : float, default 1.0
            Distance for buffer operations.
        coordinate_columns : list of str, optional
            Names of coordinate columns.
        """
        self.operations = operations
        self.buffer_distance = buffer_distance
        self.coordinate_columns = coordinate_columns or ["x", "y"]

        self.geometric_ops_ = None
        self.feature_names_out_ = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Any = None
    ) -> "SpatialGeometryTransformer":
        """
        Fit the transformer.

        Parameters
        ----------
        X : DataFrame or array-like
            Input spatial data.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.geometric_ops_ = GeometricOperations()

        self.feature_names_out_ = []
        for op in self.operations:
            if op == "buffer":
                self.feature_names_out_.extend([f"buffer_{self.buffer_distance}_area"])
            elif op == "convex_hull":
                self.feature_names_out_.extend(["convex_hull_area"])
            elif op == "bounding_box":
                self.feature_names_out_.extend(
                    ["bbox_area", "bbox_width", "bbox_height"]
                )

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data with geometric operations.

        Parameters
        ----------
        X : DataFrame or array-like
            Input spatial data.

        Returns
        -------
        X_transformed : DataFrame or array-like
            Data with geometric features added.
        """
        check_is_fitted(self, "geometric_ops_")

        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
                points = [SpatialPoint(x=row[0], y=row[1]) for row in coords]
            else:
                raise ValueError(
                    f"Coordinate columns {self.coordinate_columns} not found"
                )
        else:
            coords = np.asarray(X)
            points = [SpatialPoint(x=row[0], y=row[1]) for row in coords]

        features = []

        for op in self.operations:
            if op == "buffer":
                buffer_areas = []
                for point in points:
                    result = self.geometric_ops_.buffer_analysis(
                        point, self.buffer_distance
                    )
                    if result.result and hasattr(result.result, "area"):
                        buffer_areas.append(result.result.area)
                    elif result.properties.get("approximated"):
                        buffer_areas.append(np.pi * self.buffer_distance**2)
                    else:
                        buffer_areas.append(0)
                features.append(buffer_areas)

            elif op == "convex_hull":
                hull_result = self.geometric_ops_.convex_hull(points)
                hull_area = hull_result.properties.get("hull_area", 0)
                features.append([hull_area] * len(points))

            elif op == "bounding_box":
                bbox_result = self.geometric_ops_.bounding_box(points)
                bbox_props = bbox_result.properties
                bbox_area = bbox_props.get("area", 0)
                bbox_width = bbox_props.get("width", 0)
                bbox_height = bbox_props.get("height", 0)

                features.extend(
                    [
                        [bbox_area] * len(points),
                        [bbox_width] * len(points),
                        [bbox_height] * len(points),
                    ]
                )

        feature_array = (
            np.column_stack(features) if features else np.empty((len(points), 0))
        )

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for i, col_name in enumerate(self.feature_names_out_):
                if i < feature_array.shape[1]:
                    X_transformed[col_name] = feature_array[:, i]
            return X_transformed
        else:
            return feature_array

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get output feature names."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
