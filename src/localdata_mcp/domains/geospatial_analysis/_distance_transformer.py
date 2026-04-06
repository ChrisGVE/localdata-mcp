"""
Sklearn-compatible distance transformer for the geospatial analysis domain.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._data import SpatialPoint
from ._distance import SpatialDistanceCalculator

logger = get_logger(__name__)


class SpatialDistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for spatial distance calculations.

    This transformer calculates distances from points to reference locations
    and can be used within sklearn pipelines for spatial feature engineering.
    """

    def __init__(
        self,
        reference_points: Optional[
            List[Union[SpatialPoint, Tuple[float, float]]]
        ] = None,
        method: str = "auto",
        crs: Optional[str] = None,
        coordinate_columns: Optional[List[str]] = None,
        output_columns: Optional[List[str]] = None,
        k_nearest: Optional[int] = None,
        distance_metrics: Optional[List[str]] = None,
        output_format: Optional[str] = None,
    ):
        """
        Initialize spatial distance transformer.

        Parameters
        ----------
        reference_points : list, optional
            Reference points to calculate distances to. If None, uses centroids.
        method : str, default 'auto'
            Distance calculation method.
        crs : str, optional
            Coordinate reference system.
        coordinate_columns : list of str, optional
            Names of coordinate columns. If None, assumes ['x', 'y'].
        output_columns : list of str, optional
            Names for output distance columns.
        k_nearest : int, optional
            If specified, calculates distances to k nearest reference points.
        distance_metrics : list of str, optional
            Distance metrics to compute (e.g., ['euclidean', 'haversine']).
            If provided, overrides method parameter with the first metric.
        output_format : str, optional
            Output format ('matrix', 'dataframe'). Default is None.
        """
        self.reference_points = reference_points
        if distance_metrics:
            self.method = distance_metrics[0]
        else:
            self.method = method
        self.crs = crs
        self.coordinate_columns = coordinate_columns or ["x", "y"]
        self.output_columns = output_columns
        self.k_nearest = k_nearest
        self.distance_metrics = distance_metrics
        self.output_format = output_format

        self.distance_calculator_ = None
        self.fitted_reference_points_ = None
        self.feature_names_out_ = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Any = None
    ) -> "SpatialDistanceTransformer":
        """
        Fit the transformer by determining reference points.

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
        logger.info("Fitting spatial distance transformer")

        self.distance_calculator_ = SpatialDistanceCalculator(self.crs or "EPSG:4326")

        if self.reference_points is not None:
            self.fitted_reference_points_ = self.reference_points
            logger.info(
                f"Using {len(self.fitted_reference_points_)} provided reference points"
            )
        else:
            if isinstance(X, pd.DataFrame):
                if all(col in X.columns for col in self.coordinate_columns):
                    coords = X[self.coordinate_columns].values
                else:
                    raise ValueError(
                        f"Coordinate columns {self.coordinate_columns} not found"
                    )
            else:
                coords = np.asarray(X)

            if coords.shape[1] >= 2:
                centroid = np.mean(coords, axis=0)
                self.fitted_reference_points_ = [(centroid[0], centroid[1])]
                logger.info("Using data centroid as reference point")
            else:
                raise ValueError(
                    "Input data must have at least 2 coordinate dimensions"
                )

        n_ref = len(self.fitted_reference_points_)
        if self.k_nearest:
            n_features = min(self.k_nearest, n_ref)
        else:
            n_features = n_ref

        if self.output_columns:
            if len(self.output_columns) != n_features:
                raise ValueError(
                    f"Number of output columns ({len(self.output_columns)}) "
                    f"must match number of features ({n_features})"
                )
            self.feature_names_out_ = self.output_columns
        else:
            if self.k_nearest:
                self.feature_names_out_ = [
                    f"distance_to_nearest_{i + 1}" for i in range(n_features)
                ]
            else:
                self.feature_names_out_ = [
                    f"distance_to_ref_{i}" for i in range(n_features)
                ]

        logger.info(
            f"Configured {n_features} distance features: {self.feature_names_out_}"
        )

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data by calculating distances to reference points.

        Parameters
        ----------
        X : DataFrame or array-like
            Input spatial data.

        Returns
        -------
        X_transformed : DataFrame or array-like
            Data with distance features added.
        """
        check_is_fitted(self, "fitted_reference_points_")

        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
                query_points = [(row[0], row[1]) for row in coords]
            else:
                raise ValueError(
                    f"Coordinate columns {self.coordinate_columns} not found"
                )
        else:
            coords = np.asarray(X)
            query_points = [(row[0], row[1]) for row in coords]

        if self.k_nearest:
            distances, indices = self.distance_calculator_.nearest_neighbors(
                query_points,
                self.fitted_reference_points_,
                k=self.k_nearest,
                method=self.method,
                crs=self.crs,
            )
            distance_features = distances
        else:
            distance_matrix = self.distance_calculator_.distance_matrix(
                query_points,
                self.fitted_reference_points_,
                method=self.method,
                crs=self.crs,
                symmetric=False,
            )
            distance_features = distance_matrix

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for i, col_name in enumerate(self.feature_names_out_):
                if distance_features.ndim == 1:
                    X_transformed[col_name] = distance_features
                else:
                    X_transformed[col_name] = distance_features[:, i]
            return X_transformed
        else:
            return distance_features

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get output feature names."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
