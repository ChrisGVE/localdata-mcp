"""
Sklearn-compatible autocorrelation transformer for the geospatial analysis domain.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._spatial_stats import SpatialStatistics

logger = get_logger(__name__)


class SpatialAutocorrelationTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for spatial autocorrelation analysis.

    This transformer calculates spatial autocorrelation statistics and
    can be used within sklearn pipelines for spatial feature engineering.
    """

    def __init__(
        self,
        statistics: List[str] = ["morans_i", "gearys_c"],
        weights_method: str = "knn",
        coordinate_columns: Optional[List[str]] = None,
        value_column: str = "value",
        **weights_params,
    ):
        """
        Initialize spatial autocorrelation transformer.

        Parameters
        ----------
        statistics : list of str, default ['morans_i', 'gearys_c']
            Spatial statistics to calculate.
        weights_method : str, default 'knn'
            Spatial weights method.
        coordinate_columns : list of str, optional
            Names of coordinate columns.
        value_column : str, default 'value'
            Name of value column for analysis.
        **weights_params : additional parameters
            Parameters for spatial weights construction.
        """
        self.statistics = statistics
        self.weights_method = weights_method
        self.coordinate_columns = coordinate_columns or ["x", "y"]
        self.value_column = value_column
        self.weights_params = weights_params

        self.spatial_stats_ = None
        self.feature_names_out_ = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Any = None
    ) -> "SpatialAutocorrelationTransformer":
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
        self.spatial_stats_ = SpatialStatistics()

        self.feature_names_out_ = []
        for stat in self.statistics:
            if stat == "morans_i":
                self.feature_names_out_.extend(
                    ["morans_i", "morans_i_pvalue", "morans_i_zscore"]
                )
            elif stat == "gearys_c":
                self.feature_names_out_.extend(
                    ["gearys_c", "gearys_c_pvalue", "gearys_c_zscore"]
                )

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data with spatial autocorrelation statistics.

        Parameters
        ----------
        X : DataFrame or array-like
            Input spatial data.

        Returns
        -------
        X_transformed : DataFrame or array-like
            Data with spatial autocorrelation features added.
        """
        check_is_fitted(self, "spatial_stats_")

        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
                points = [(row[0], row[1]) for row in coords]
            else:
                raise ValueError(
                    f"Coordinate columns {self.coordinate_columns} not found"
                )

            if self.value_column in X.columns:
                values = X[self.value_column].values
            else:
                raise ValueError(f"Value column {self.value_column} not found")
        else:
            coords = X[:, :2]
            points = [(row[0], row[1]) for row in coords]
            values = X[:, 2] if X.shape[1] > 2 else np.ones(len(points))

        features = []

        for stat in self.statistics:
            if stat == "morans_i":
                result = self.spatial_stats_.morans_i(
                    values, points, self.weights_method, **self.weights_params
                )
                features.extend(
                    [result.value, result.p_value or np.nan, result.z_score or np.nan]
                )

            elif stat == "gearys_c":
                result = self.spatial_stats_.gearys_c(
                    values, points, self.weights_method, **self.weights_params
                )
                features.extend(
                    [result.value, result.p_value or np.nan, result.z_score or np.nan]
                )

        n_rows = len(points)
        feature_matrix = np.array([features] * n_rows)

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for i, col_name in enumerate(self.feature_names_out_):
                X_transformed[col_name] = feature_matrix[:, i]
            return X_transformed
        else:
            return np.column_stack([X, feature_matrix])

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get output feature names."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
