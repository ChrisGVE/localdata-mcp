"""
Sklearn-compatible coordinate transformer for the geospatial analysis domain.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._coordinates import CoordinateReferenceSystem

logger = get_logger(__name__)


class SpatialCoordinateTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for coordinate reference system transformations.

    This transformer handles CRS transformations within the sklearn pipeline framework,
    with support for both pyproj-based and fallback transformation methods.
    """

    def __init__(
        self,
        target_crs: str = "EPSG:4326",
        source_crs: Optional[str] = None,
        validate_crs: bool = True,
        coordinate_columns: Optional[List[str]] = None,
    ):
        """
        Initialize coordinate transformer.

        Parameters
        ----------
        target_crs : str, default 'EPSG:4326'
            Target coordinate reference system.
        source_crs : str, optional
            Source CRS. If None, will attempt to detect from data.
        validate_crs : bool, default True
            Whether to validate CRS specifications.
        coordinate_columns : list of str, optional
            Names of coordinate columns. If None, assumes ['x', 'y'].
        """
        self.target_crs = target_crs
        self.source_crs = source_crs
        self.validate_crs = validate_crs
        self.coordinate_columns = coordinate_columns or ["x", "y"]

        self.crs_handler_ = None
        self.transformation_ = None
        self.fitted_source_crs_ = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Any = None
    ) -> "SpatialCoordinateTransformer":
        """
        Fit the transformer by validating CRS and preparing transformation.

        Parameters
        ----------
        X : DataFrame or array-like
            Input data with spatial coordinates.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        logger.info("Fitting spatial coordinate transformer")

        self.crs_handler_ = CoordinateReferenceSystem()

        if self.source_crs is None:
            if isinstance(X, pd.DataFrame) and "crs" in X.columns:
                unique_crs = X["crs"].dropna().unique()
                if len(unique_crs) == 1:
                    self.fitted_source_crs_ = unique_crs[0]
                    logger.info(
                        f"Detected source CRS from data: {self.fitted_source_crs_}"
                    )
                elif len(unique_crs) > 1:
                    raise ValueError(f"Multiple CRS found in data: {unique_crs}")
                else:
                    raise ValueError(
                        "No CRS information found in data and source_crs not specified"
                    )
            else:
                self.fitted_source_crs_ = "EPSG:4326"
                logger.warning("No CRS specified, assuming WGS84 (EPSG:4326)")
        else:
            self.fitted_source_crs_ = self.source_crs

        if self.validate_crs:
            source_validation = self.crs_handler_.validate_crs(self.fitted_source_crs_)
            target_validation = self.crs_handler_.validate_crs(self.target_crs)

            if not source_validation["valid"]:
                raise ValueError(f"Invalid source CRS: {self.fitted_source_crs_}")
            if not target_validation["valid"]:
                raise ValueError(f"Invalid target CRS: {self.target_crs}")

            for warning in source_validation.get("warnings", []):
                logger.warning(f"Source CRS warning: {warning}")
            for warning in target_validation.get("warnings", []):
                logger.warning(f"Target CRS warning: {warning}")

        self.transformation_ = self.crs_handler_.get_transformation(
            self.fitted_source_crs_, self.target_crs
        )

        if not self.transformation_.is_valid:
            raise ValueError(
                f"Cannot create transformation: {self.transformation_.error_message}"
            )

        logger.info(
            f"Prepared CRS transformation: {self.fitted_source_crs_} -> {self.target_crs}"
        )

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform coordinates to target CRS.

        Parameters
        ----------
        X : DataFrame or array-like
            Input data with spatial coordinates.

        Returns
        -------
        X_transformed : DataFrame or array-like
            Data with transformed coordinates.
        """
        check_is_fitted(self, "transformation_")

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()

            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
            else:
                raise ValueError(
                    f"Coordinate columns {self.coordinate_columns} not found in DataFrame"
                )

            transformed_coords, warnings = self.crs_handler_.transform_coordinates(
                coords, self.fitted_source_crs_, self.target_crs
            )

            for i, col in enumerate(self.coordinate_columns):
                if i < transformed_coords.shape[1]:
                    X_transformed[col] = transformed_coords[:, i]

            if "crs" in X_transformed.columns:
                X_transformed["crs"] = self.target_crs

            for warning in warnings:
                logger.warning(f"Coordinate transformation warning: {warning}")

            return X_transformed

        else:
            coords = np.asarray(X)
            transformed_coords, warnings = self.crs_handler_.transform_coordinates(
                coords, self.fitted_source_crs_, self.target_crs
            )

            for warning in warnings:
                logger.warning(f"Coordinate transformation warning: {warning}")

            return transformed_coords

    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about the fitted transformation."""
        check_is_fitted(self, "transformation_")

        return {
            "source_crs": self.fitted_source_crs_,
            "target_crs": self.target_crs,
            "transformation_valid": self.transformation_.is_valid,
            "transformation_accuracy": self.transformation_.transformation_accuracy,
            "transformation_method": self.transformation_.transformation_method,
            "coordinate_columns": self.coordinate_columns,
        }
