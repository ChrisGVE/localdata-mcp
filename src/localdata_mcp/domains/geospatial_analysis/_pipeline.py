"""
Geospatial analysis pipeline and high-level convenience functions.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ...logging_manager import get_logger
from ._autocorrelation_transformer import SpatialAutocorrelationTransformer
from ._base import NetworkAnalysisType
from ._data import SpatialPoint
from ._distance import SpatialDistanceCalculator
from ._distance_transformer import SpatialDistanceTransformer
from ._geometry_transformer import SpatialGeometryTransformer
from ._interpolator import SpatialInterpolationTransformer
from ._network import SpatialNetwork
from ._network_transformer import SpatialNetworkTransformer
from ._spatial_stats import SpatialStatistics
from ._statistics import SpatialWeightsMatrix

logger = get_logger(__name__)


class GeospatialAnalysisPipeline:
    """Unified pipeline for comprehensive geospatial analysis workflows."""

    def __init__(
        self,
        coordinate_columns: List[str] = ["x", "y"],
        value_column: str = None,
        network_data: Dict = None,
    ):
        self.coordinate_columns = coordinate_columns
        self.value_column = value_column
        self.network_data = network_data
        self.transformers = []
        self.results = {}

    def add_autocorrelation_analysis(self, method="moran", k_neighbors=8):
        """Add spatial autocorrelation analysis to pipeline."""
        if self.value_column:
            transformer = SpatialAutocorrelationTransformer(
                coordinate_columns=self.coordinate_columns,
                value_column=self.value_column,
                test_type=method,
                k_neighbors=k_neighbors,
            )
            self.transformers.append(("autocorrelation", transformer))
        else:
            logger.warning("No value column specified for autocorrelation analysis")

    def add_distance_analysis(self, distance_metrics=["euclidean"]):
        """Add spatial distance analysis to pipeline."""
        transformer = SpatialDistanceTransformer(
            coordinate_columns=self.coordinate_columns,
            distance_metrics=distance_metrics,
            output_format="dataframe",
        )
        self.transformers.append(("distance", transformer))

    def add_geometry_analysis(self, operations=["convex_hull", "bounding_box"]):
        """Add geometric analysis to pipeline."""
        transformer = SpatialGeometryTransformer(
            coordinate_columns=self.coordinate_columns, operations=operations
        )
        self.transformers.append(("geometry", transformer))

    def add_interpolation_analysis(self, method="kriging"):
        """Add spatial interpolation to pipeline."""
        if self.value_column:
            transformer = SpatialInterpolationTransformer(
                coordinate_columns=self.coordinate_columns,
                value_column=self.value_column,
                method=method,
            )
            self.transformers.append(("interpolation", transformer))
        else:
            logger.warning("No value column specified for interpolation analysis")

    def add_network_analysis(
        self, analysis_type="accessibility", service_locations=None
    ):
        """Add network analysis to pipeline."""
        if self.network_data:
            transformer = SpatialNetworkTransformer(
                analysis_type=NetworkAnalysisType(analysis_type),
                network_data=self.network_data,
                service_locations=service_locations or [],
            )
            self.transformers.append(("network", transformer))
        else:
            logger.warning("No network data specified for network analysis")

    def fit_transform(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Execute the complete geospatial analysis pipeline."""
        results = {}
        current_data = data.copy()

        for name, transformer in self.transformers:
            try:
                logger.info(f"Executing {name} analysis...")
                transformer.fit(current_data)
                result = transformer.transform(current_data)
                results[name] = result

                if isinstance(result, pd.DataFrame) and len(result) == len(
                    current_data
                ):
                    new_columns = [
                        col for col in result.columns if col not in current_data.columns
                    ]
                    for col in new_columns:
                        current_data[col] = result[col]

            except Exception as e:
                logger.error(f"Pipeline step {name} failed: {e}")
                results[name] = pd.DataFrame()

        self.results = results
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary of pipeline execution."""
        summary = {
            "steps_executed": len(self.results),
            "successful_steps": sum(
                1 for result in self.results.values() if not result.empty
            ),
            "failed_steps": sum(1 for result in self.results.values() if result.empty),
            "total_features_created": sum(
                len(result.columns)
                for result in self.results.values()
                if isinstance(result, pd.DataFrame)
            ),
            "analysis_types": list(self.results.keys()),
        }

        for name, result in self.results.items():
            if isinstance(result, pd.DataFrame) and not result.empty:
                summary[f"{name}_columns"] = list(result.columns)
                summary[f"{name}_rows"] = len(result)

        return summary


# High-level convenience functions


def analyze_spatial_autocorrelation(
    data: pd.DataFrame,
    value_column: str,
    coordinate_columns: List[str] = ["x", "y"],
    method: str = "moran",
    k_neighbors: int = 8,
) -> Dict[str, Any]:
    """Analyze spatial autocorrelation in dataset.

    Returns Moran's I or Geary's C for ``value_column`` together with the
    significance test that decides whether nearby observations really do
    resemble each other more than distant ones.
    """
    missing = [c for c in [*coordinate_columns, value_column] if c not in data.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found. Available columns: {list(data.columns)}"
        )

    points = list(
        zip(
            data[coordinate_columns[0]].to_numpy(),
            data[coordinate_columns[1]].to_numpy(),
        )
    )
    values = data[value_column].to_numpy(dtype=float)

    stats = SpatialStatistics()
    if method in ("moran", "morans_i"):
        result = stats.morans_i(values, points, weights_method="knn", k=k_neighbors)
    elif method in ("geary", "gearys_c"):
        result = stats.gearys_c(values, points, weights_method="knn", k=k_neighbors)
    else:
        raise ValueError(
            f"Unsupported autocorrelation method '{method}'. "
            "Use 'moran' (Moran's I) or 'geary' (Geary's C)."
        )

    return {
        "method": method,
        "statistic": result.statistic,
        "value": result.value,
        "p_value": result.p_value,
        "expected_value": result.expected_value,
        "variance": result.variance,
        "z_score": result.z_score,
        "is_significant": result.is_significant,
        "interpretation": result.interpretation,
        "n_observations": len(values),
        "k_neighbors": k_neighbors,
    }


def perform_spatial_clustering(
    data: pd.DataFrame,
    coordinate_columns: List[str] = ["x", "y"],
    method: str = "hotspot",
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Identify spatial clusters and hot spots in data.

    Runs a Getis-Ord Gi* hot-spot analysis over the first numeric non-coordinate
    column and returns the input frame with the per-point statistic, its p-value
    and a ``cluster_id`` of 1 (hot spot), -1 (cold spot) or 0 (not significant).
    """
    missing = [c for c in coordinate_columns if c not in data.columns]
    if missing:
        raise ValueError(
            f"Coordinate column(s) {missing} not found. "
            f"Available columns: {list(data.columns)}"
        )

    coordinates = list(
        zip(
            data[coordinate_columns[0]].to_numpy(),
            data[coordinate_columns[1]].to_numpy(),
        )
    )

    value_columns = [
        col
        for col in data.columns
        if col not in coordinate_columns
        and pd.api.types.is_numeric_dtype(data[col])
        and not pd.api.types.is_bool_dtype(data[col])
    ]

    if not value_columns:
        raise ValueError(
            "Spatial clustering needs a numeric value column besides the "
            f"coordinates {coordinate_columns}. Available columns: {list(data.columns)}"
        )

    values = data[value_columns[0]].to_numpy(dtype=float)

    # `method` selects the clustering statistic; 'hotspot' and 'getis_ord' are
    # the two names for Gi*. An unknown name raises rather than silently
    # producing a hot-spot analysis the caller did not ask for.
    hotspot = SpatialStatistics().spatial_clustering_analysis(
        values,
        coordinates,
        method=method,
        weights_method="knn",
        k=8,
        significance_level=significance_level,
    )

    result_data = data.copy()
    result_data["gi_star_statistic"] = hotspot["gi_star"]
    result_data["gi_star_z_score"] = hotspot["z_scores"]
    result_data["gi_star_p_value"] = hotspot["p_values"]
    result_data["is_hotspot"] = hotspot["hotspots"]
    result_data["is_coldspot"] = hotspot["coldspots"]
    result_data["cluster_label"] = hotspot["cluster_labels"]

    cluster_ids = np.zeros(len(data), dtype=int)
    cluster_ids[hotspot["hotspots"]] = 1
    cluster_ids[hotspot["coldspots"]] = -1
    result_data["cluster_id"] = cluster_ids

    return result_data


def calculate_spatial_distance(
    data1: pd.DataFrame,
    data2: pd.DataFrame = None,
    coordinate_columns: List[str] = ["x", "y"],
    distance_type: str = "euclidean",
    output_format: str = "matrix",
) -> Union[np.ndarray, pd.DataFrame]:
    """Calculate spatial distances between points.

    With ``data2`` omitted the points in ``data1`` are compared against
    themselves, so ``output_format='matrix'`` yields the full symmetric pairwise
    matrix. Previously the one-frame path delegated to a transformer whose
    output columns are named ``distance_to_ref_*``, looked for columns named
    after the *metric* instead, matched none, and returned an empty array.
    """

    def _coords(frame: pd.DataFrame):
        missing = [c for c in coordinate_columns if c not in frame.columns]
        if missing:
            raise ValueError(
                f"Coordinate column(s) {missing} not found. "
                f"Available columns: {list(frame.columns)}"
            )
        return list(
            zip(
                frame[coordinate_columns[0]].to_numpy(),
                frame[coordinate_columns[1]].to_numpy(),
            )
        )

    coords1 = _coords(data1)
    coords2 = coords1 if data2 is None else _coords(data2)

    distances = SpatialDistanceCalculator().distance_matrix(
        coords1, coords2, method=distance_type
    )

    if output_format == "pairs":
        return pd.DataFrame(
            [
                {"point1_index": i, "point2_index": j, "distance": distances[i, j]}
                for i in range(len(coords1))
                for j in range(len(coords2))
            ]
        )

    return distances
