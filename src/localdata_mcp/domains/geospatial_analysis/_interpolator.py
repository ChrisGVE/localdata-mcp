"""
Spatial interpolator and interpolation transformer for the geospatial analysis domain.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._data import SpatialPoint
from ._distance import SpatialDistanceCalculator
from ._interpolation import InterpolationResult, VariogramModel

logger = get_logger(__name__)


class SpatialInterpolator:
    """
    Comprehensive spatial interpolation tools.

    Implements kriging interpolation with variogram modeling,
    inverse distance weighting, and other spatial interpolation methods.
    """

    def __init__(self):
        """Initialize spatial interpolator."""
        self.distance_calculator = SpatialDistanceCalculator()
        self.last_variogram_model = None

    def ordinary_kriging(
        self,
        train_points: List[Union[SpatialPoint, Tuple[float, float]]],
        train_values: np.ndarray,
        predict_points: List[Union[SpatialPoint, Tuple[float, float]]],
        variogram_model: str = "spherical",
        **variogram_params,
    ) -> InterpolationResult:
        """Perform ordinary kriging interpolation."""
        train_values = np.asarray(train_values)
        warnings = []

        try:
            vario_model = VariogramModel(variogram_model)
            vario_info = vario_model.fit(train_points, train_values, **variogram_params)
            self.last_variogram_model = vario_model

            n_train = len(train_points)
            n_predict = len(predict_points)

            train_coords = self._extract_coordinates(train_points)
            predict_coords = self._extract_coordinates(predict_points)

            train_distances = self.distance_calculator.distance_matrix(
                train_points, train_points
            )

            pred_to_train_distances = self.distance_calculator.distance_matrix(
                predict_points, train_points, symmetric=False
            )

            C = self._distances_to_covariances(train_distances, vario_model)

            C_extended = np.zeros((n_train + 1, n_train + 1))
            C_extended[:n_train, :n_train] = C
            C_extended[n_train, :n_train] = 1
            C_extended[:n_train, n_train] = 1
            C_extended[n_train, n_train] = 0

            predictions = np.zeros(n_predict)
            variances = np.zeros(n_predict)

            for i in range(n_predict):
                c = self._distances_to_covariances(
                    pred_to_train_distances[i], vario_model
                )

                rhs = np.zeros(n_train + 1)
                rhs[:n_train] = c
                rhs[n_train] = 1

                try:
                    weights = np.linalg.solve(C_extended, rhs)
                    predictions[i] = np.dot(weights[:n_train], train_values)
                    variances[i] = (
                        vario_model.parameters["sill"]
                        - np.dot(weights[:n_train], c)
                        - weights[n_train]
                    )

                except np.linalg.LinAlgError:
                    warnings.append(f"Singular kriging system at prediction point {i}")
                    distances = pred_to_train_distances[i]
                    weights = 1 / (distances + 1e-10)
                    weights /= np.sum(weights)
                    predictions[i] = np.dot(weights, train_values)
                    variances[i] = np.nan

            return InterpolationResult(
                method="ordinary_kriging",
                interpolated_values=predictions,
                prediction_variance=variances,
                variogram_model=vario_info,
                metadata={
                    "n_training_points": n_train,
                    "n_prediction_points": n_predict,
                    "variogram_method": vario_info.get("method", "unknown"),
                },
                warnings=warnings,
            )

        except Exception as e:
            warnings.append(f"Kriging failed: {e}")
            logger.warning(f"Ordinary kriging failed, falling back to IDW: {e}")
            return self.inverse_distance_weighting(
                train_points, train_values, predict_points
            )

    def inverse_distance_weighting(
        self,
        train_points: List[Union[SpatialPoint, Tuple[float, float]]],
        train_values: np.ndarray,
        predict_points: List[Union[SpatialPoint, Tuple[float, float]]],
        power: float = 2.0,
        max_distance: Optional[float] = None,
    ) -> InterpolationResult:
        """Perform inverse distance weighting interpolation."""
        train_values = np.asarray(train_values)

        distance_matrix = self.distance_calculator.distance_matrix(
            predict_points, train_points, symmetric=False
        )

        if max_distance is not None:
            distance_matrix = np.where(
                distance_matrix > max_distance, np.inf, distance_matrix
            )

        weights = 1.0 / (distance_matrix + 1e-10) ** power

        exact_matches = distance_matrix < 1e-10

        predictions = np.zeros(len(predict_points))

        for i in range(len(predict_points)):
            if np.any(exact_matches[i]):
                exact_idx = np.where(exact_matches[i])[0][0]
                predictions[i] = train_values[exact_idx]
            else:
                valid_weights = weights[i][np.isfinite(weights[i])]
                valid_values = train_values[np.isfinite(weights[i])]

                if len(valid_weights) > 0:
                    weight_sum = np.sum(valid_weights)
                    if weight_sum > 0:
                        predictions[i] = (
                            np.sum(valid_weights * valid_values) / weight_sum
                        )
                    else:
                        predictions[i] = np.mean(train_values)
                else:
                    predictions[i] = np.mean(train_values)

        return InterpolationResult(
            method="inverse_distance_weighting",
            interpolated_values=predictions,
            metadata={
                "power": power,
                "max_distance": max_distance,
                "n_training_points": len(train_points),
                "n_prediction_points": len(predict_points),
            },
        )

    def cross_validate_kriging(
        self,
        points: List[Union[SpatialPoint, Tuple[float, float]]],
        values: np.ndarray,
        n_folds: int = 5,
        variogram_model: str = "spherical",
    ) -> Dict[str, float]:
        """Perform cross-validation for kriging parameters."""
        values = np.asarray(values)
        n_points = len(points)

        fold_size = n_points // n_folds
        indices = np.arange(n_points)
        np.random.shuffle(indices)

        predictions = np.zeros(n_points)
        fold_scores = []

        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_points

            test_indices = indices[start_idx:end_idx]
            train_indices = np.setdiff1d(indices, test_indices)

            train_points = [points[i] for i in train_indices]
            train_values = values[train_indices]
            test_points = [points[i] for i in test_indices]
            test_values = values[test_indices]

            result = self.ordinary_kriging(
                train_points, train_values, test_points, variogram_model
            )

            predictions[test_indices] = result.interpolated_values

            fold_rmse = np.sqrt(
                np.mean((result.interpolated_values - test_values) ** 2)
            )
            fold_scores.append(fold_rmse)

        rmse = np.sqrt(np.mean((predictions - values) ** 2))
        mae = np.mean(np.abs(predictions - values))
        r2 = 1 - np.sum((values - predictions) ** 2) / np.sum(
            (values - np.mean(values)) ** 2
        )

        return {
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "fold_rmse_mean": np.mean(fold_scores),
            "fold_rmse_std": np.std(fold_scores),
            "n_folds": n_folds,
        }

    def _extract_coordinates(self, points):
        """Extract coordinate array from points list."""
        coords = []
        for point in points:
            if isinstance(point, tuple):
                coords.append(point)
            elif isinstance(point, SpatialPoint):
                coords.append((point.x, point.y))
        return np.array(coords)

    def _distances_to_covariances(self, distances, variogram_model):
        """Convert distances to covariances using variogram model."""
        semivariances = variogram_model.predict_semivariance(distances)
        covariances = variogram_model.parameters["sill"] - semivariances
        return covariances


class SpatialInterpolationTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for spatial interpolation.

    This transformer performs spatial interpolation on test data based on
    training data and can be used within sklearn pipelines.
    """

    def __init__(
        self,
        method: str = "kriging",
        coordinate_columns: Optional[List[str]] = None,
        value_column: str = "value",
        **method_params,
    ):
        """Initialize spatial interpolation transformer."""
        self.method = method
        self.coordinate_columns = coordinate_columns or ["x", "y"]
        self.value_column = value_column
        self.method_params = method_params

        self.interpolator_ = None
        self.train_points_ = None
        self.train_values_ = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Any = None
    ) -> "SpatialInterpolationTransformer":
        """Fit the interpolation model."""
        self.interpolator_ = SpatialInterpolator()

        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
                self.train_points_ = [(row[0], row[1]) for row in coords]
            else:
                raise ValueError(
                    f"Coordinate columns {self.coordinate_columns} not found"
                )

            if self.value_column in X.columns:
                self.train_values_ = X[self.value_column].values
            else:
                raise ValueError(f"Value column {self.value_column} not found")
        else:
            coords = X[:, :2]
            self.train_points_ = [(row[0], row[1]) for row in coords]
            self.train_values_ = (
                X[:, 2] if X.shape[1] > 2 else np.ones(len(self.train_points_))
            )

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Interpolate values at new locations."""
        check_is_fitted(self, "train_points_")

        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
                predict_points = [(row[0], row[1]) for row in coords]
            else:
                raise ValueError(
                    f"Coordinate columns {self.coordinate_columns} not found"
                )
        else:
            coords = X[:, :2]
            predict_points = [(row[0], row[1]) for row in coords]

        if self.method == "kriging":
            result = self.interpolator_.ordinary_kriging(
                self.train_points_,
                self.train_values_,
                predict_points,
                **self.method_params,
            )
        elif self.method == "idw":
            result = self.interpolator_.inverse_distance_weighting(
                self.train_points_,
                self.train_values_,
                predict_points,
                **self.method_params,
            )
        else:
            raise ValueError(f"Unsupported interpolation method: {self.method}")

        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            X_transformed[f"{self.value_column}_interpolated"] = (
                result.interpolated_values
            )
            if result.prediction_variance is not None:
                X_transformed[f"{self.value_column}_variance"] = (
                    result.prediction_variance
                )
            return X_transformed
        else:
            if result.prediction_variance is not None:
                return np.column_stack(
                    [X, result.interpolated_values, result.prediction_variance]
                )
            else:
                return np.column_stack([X, result.interpolated_values])
