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
    """Analyze spatial autocorrelation in dataset."""
    transformer = SpatialAutocorrelationTransformer(
        coordinate_columns=coordinate_columns,
        value_column=value_column,
        test_type=method,
        k_neighbors=k_neighbors,
    )

    transformer.fit(data)
    result = transformer.transform(data)

    if hasattr(transformer, "statistics_"):
        return {
            "method": method,
            "statistic": transformer.statistics_.statistic,
            "p_value": transformer.statistics_.p_value,
            "expected_value": transformer.statistics_.expected_value,
            "variance": transformer.statistics_.variance,
            "z_score": transformer.statistics_.z_score,
            "interpretation": (
                "significant spatial autocorrelation"
                if transformer.statistics_.p_value < 0.05
                else "no significant spatial autocorrelation"
            ),
            "local_statistics": (
                transformer.statistics_.local_statistics
                if hasattr(transformer.statistics_, "local_statistics")
                else None
            ),
        }
    else:
        return {"error": "Analysis failed", "method": method}


def perform_spatial_clustering(
    data: pd.DataFrame,
    coordinate_columns: List[str] = ["x", "y"],
    method: str = "hotspot",
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Identify spatial clusters and hot spots in data."""
    stats = SpatialStatistics()

    coordinates = [
        (row[coordinate_columns[0]], row[coordinate_columns[1]])
        for _, row in data.iterrows()
    ]
    spatial_weights = SpatialWeightsMatrix(method="knn", k=8)
    weights_matrix = spatial_weights.build_weights(coordinates)

    value_columns = [
        col
        for col in data.columns
        if col not in coordinate_columns and data[col].dtype in [np.float64, np.int64]
    ]

    if not value_columns:
        logger.warning("No numeric value columns found for clustering analysis")
        result_data = data.copy()
        result_data["cluster_id"] = 0
        result_data["is_hotspot"] = False
        return result_data

    value_column = value_columns[0]
    values = data[value_column].values

    try:
        hotspot_result = stats.getis_ord_gi_star(coordinates, values, weights_matrix)

        result_data = data.copy()
        result_data["gi_star_statistic"] = hotspot_result.gi_star_values
        result_data["gi_star_p_value"] = hotspot_result.p_values
        result_data["is_hotspot"] = hotspot_result.p_values < significance_level
        result_data["is_coldspot"] = (hotspot_result.gi_star_values < 0) & (
            hotspot_result.p_values < significance_level
        )

        cluster_ids = np.zeros(len(data))
        cluster_ids[result_data["is_hotspot"]] = 1
        cluster_ids[result_data["is_coldspot"]] = -1
        result_data["cluster_id"] = cluster_ids

        return result_data

    except Exception as e:
        logger.warning(f"Spatial clustering failed: {e}")
        result_data = data.copy()
        result_data["cluster_id"] = 0
        result_data["is_hotspot"] = False
        return result_data


def calculate_spatial_distance(
    data1: pd.DataFrame,
    data2: pd.DataFrame = None,
    coordinate_columns: List[str] = ["x", "y"],
    distance_type: str = "euclidean",
    output_format: str = "matrix",
) -> Union[np.ndarray, pd.DataFrame]:
    """Calculate spatial distances between points."""
    if data2 is not None:
        coords1 = [
            (row[coordinate_columns[0]], row[coordinate_columns[1]])
            for _, row in data1.iterrows()
        ]
        coords2 = [
            (row[coordinate_columns[0]], row[coordinate_columns[1]])
            for _, row in data2.iterrows()
        ]

        distance_calc = SpatialDistanceCalculator()
        distances = distance_calc.distance_matrix(
            coords1, coords2, method=distance_type
        )

        if output_format == "matrix":
            return distances
        elif output_format == "pairs":
            pairs = []
            for i, coord1 in enumerate(coords1):
                for j, coord2 in enumerate(coords2):
                    pairs.append(
                        {
                            "point1_index": i,
                            "point2_index": j,
                            "distance": distances[i, j],
                        }
                    )
            return pd.DataFrame(pairs)
        else:
            return distances
    else:
        distance_transformer = SpatialDistanceTransformer(
            method=distance_type,
            coordinate_columns=coordinate_columns,
            output_format="dataframe",
        )
        distance_transformer.fit(data1)
        result = distance_transformer.transform(data1)

        if output_format == "matrix":
            distance_columns = [
                col for col in result.columns if col.startswith(distance_type)
            ]
            if distance_columns:
                return result[distance_columns].values
            else:
                return np.array([])
        else:
            return result
