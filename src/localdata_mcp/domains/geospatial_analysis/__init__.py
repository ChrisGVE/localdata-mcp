"""
Geospatial Analysis Domain - Comprehensive spatial analysis capabilities.

This package implements advanced geospatial analysis tools including coordinate systems,
spatial statistics, geometric operations, and network analysis using optional
integration with geopandas, shapely, pyproj, and networkx.

Key Features:
- Coordinate Reference System (CRS) handling and transformations
- Spatial statistics (Moran's I, Geary's C, spatial clustering)
- Geometric operations (point-in-polygon, buffers, spatial joins)
- Network analysis with spatial constraints
- Spatial interpolation using kriging
- Full sklearn pipeline compatibility
- Core geospatial dependencies (geopandas, shapely, pyproj, fiona) always available
- Comprehensive spatial validation and error handling
"""

from ._base import (
    GeospatialLibrary,
    DependencyStatus,
    SpatialJoinType,
    OverlayOperation,
    NetworkAnalysisType,
    _dependency_status,
)

from ._data import (
    SpatialPoint,
    SpatialDataFrame,
    SpatialDataValidator,
    _haversine_distance,
)

from ._dependency import (
    GeospatialDependencyChecker,
    get_dependency_status,
    check_geospatial_capabilities,
    create_spatial_dataframe,
)

from ._processor import (
    SpatialAnalysisResult,
    SpatialDataProcessor,
)

from ._coordinates import (
    CoordinateTransformation,
    CoordinateReferenceSystem,
)

from ._coord_transformer import (
    SpatialCoordinateTransformer,
)

from ._distance import (
    SpatialDistanceCalculator,
)

from ._distance_transformer import (
    SpatialDistanceTransformer,
)

from ._geometry import (
    GeometricResult,
)

from ._geometry_advanced import (
    GeometricOperations,
)

from ._spatial_indexer import (
    SpatialIndexer,
)

from ._geometry_transformer import (
    SpatialGeometryTransformer,
)

from ._statistics import (
    SpatialStatisticsResult,
    SpatialWeightsMatrix,
)

from ._spatial_stats import (
    SpatialStatistics,
)

from ._autocorrelation_transformer import (
    SpatialAutocorrelationTransformer,
)

from ._interpolation import (
    InterpolationResult,
    VariogramModel,
)

from ._interpolator import (
    SpatialInterpolator,
    SpatialInterpolationTransformer,
)

from ._joins import (
    SpatialJoinResult,
    SpatialJoinEngine,
)

from ._overlay import (
    OverlayResult,
    SpatialOverlayEngine,
    SpatialAggregator,
    SpatialJoinTransformer,
    SpatialOverlayTransformer,
    perform_spatial_join,
    perform_spatial_overlay,
    aggregate_points_in_polygons,
)

from ._network import (
    RouteResult,
    AccessibilityResult,
    IsochroneResult,
    SpatialNetwork,
)

from ._network_analysis import (
    NetworkRouter,
    AccessibilityAnalyzer,
)

from ._isochrone import (
    IsochroneGenerator,
)

from ._network_transformer import (
    SpatialNetworkTransformer,
    optimize_route,
    optimize_routes,
    analyze_accessibility,
    generate_service_isochrones,
)

from ._pipeline import (
    GeospatialAnalysisPipeline,
    analyze_spatial_autocorrelation,
    perform_spatial_clustering,
    calculate_spatial_distance,
)

from ...logging_manager import get_logger

logger = get_logger(__name__)

# Initialize dependency checking on module import
logger.info("Initializing geospatial analysis module")
logger.info(
    f"Available geospatial libraries: "
    f"{[lib.value for lib, available in _dependency_status.available_libraries.items() if available]}"
)

if not _dependency_status.has_core_geospatial():
    logger.warning(
        "Core geospatial libraries (geopandas, shapely) reported unavailable, "
        "but they should be installed as required dependencies."
    )

__all__ = [
    # Enums
    "GeospatialLibrary",
    "SpatialJoinType",
    "OverlayOperation",
    "NetworkAnalysisType",
    # Data structures
    "DependencyStatus",
    "SpatialPoint",
    "SpatialDataFrame",
    "SpatialDataValidator",
    "CoordinateTransformation",
    "SpatialAnalysisResult",
    "GeometricResult",
    "SpatialStatisticsResult",
    "InterpolationResult",
    "SpatialJoinResult",
    "OverlayResult",
    "RouteResult",
    "AccessibilityResult",
    "IsochroneResult",
    # Core classes
    "CoordinateReferenceSystem",
    "SpatialDistanceCalculator",
    "GeometricOperations",
    "SpatialIndexer",
    "SpatialWeightsMatrix",
    "SpatialStatistics",
    "VariogramModel",
    "SpatialInterpolator",
    "SpatialDataProcessor",
    # Engines
    "SpatialJoinEngine",
    "SpatialOverlayEngine",
    "SpatialAggregator",
    "SpatialNetwork",
    "NetworkRouter",
    "AccessibilityAnalyzer",
    "IsochroneGenerator",
    # Transformers
    "GeospatialDependencyChecker",
    "SpatialCoordinateTransformer",
    "SpatialDistanceTransformer",
    "SpatialGeometryTransformer",
    "SpatialAutocorrelationTransformer",
    "SpatialInterpolationTransformer",
    "SpatialJoinTransformer",
    "SpatialOverlayTransformer",
    "SpatialNetworkTransformer",
    # Pipeline
    "GeospatialAnalysisPipeline",
    # High-level functions
    "get_dependency_status",
    "check_geospatial_capabilities",
    "create_spatial_dataframe",
    "analyze_spatial_autocorrelation",
    "perform_spatial_clustering",
    "calculate_spatial_distance",
    "optimize_route",
    "optimize_routes",
    "analyze_accessibility",
    "generate_service_isochrones",
    "perform_spatial_join",
    "perform_spatial_overlay",
    "aggregate_points_in_polygons",
    # Internal (used by other modules)
    "_haversine_distance",
    "_dependency_status",
]
