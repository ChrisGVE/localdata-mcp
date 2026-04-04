"""
Dependency checking and management for the geospatial analysis domain.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from ._base import GeospatialLibrary, _dependency_status
from ._data import SpatialDataFrame

logger = get_logger(__name__)


class GeospatialDependencyChecker(BaseEstimator, TransformerMixin):
    """
    Pipeline component for checking and managing geospatial library dependencies.

    This component provides runtime dependency checking and graceful degradation
    for optional geospatial libraries, enabling the pipeline to work with
    varying levels of functionality based on available dependencies.
    """

    def __init__(
        self,
        required_libraries: Optional[List[GeospatialLibrary]] = None,
        fail_on_missing: bool = False,
        enable_fallbacks: bool = True,
    ):
        """
        Initialize dependency checker.

        Parameters
        ----------
        required_libraries : list of GeospatialLibrary, optional
            Libraries that must be available. If None, uses core set.
        fail_on_missing : bool, default False
            Whether to raise error when required libraries are missing.
        enable_fallbacks : bool, default True
            Whether to enable fallback mechanisms when libraries are missing.
        """
        self.required_libraries = required_libraries or [
            GeospatialLibrary.GEOPANDAS,
            GeospatialLibrary.SHAPELY,
        ]
        self.fail_on_missing = fail_on_missing
        self.enable_fallbacks = enable_fallbacks

        self.dependency_status_ = None
        self.available_features_ = None

    def fit(self, X: Any = None, y: Any = None) -> "GeospatialDependencyChecker":
        """
        Check dependency status and determine available features.

        Parameters
        ----------
        X : ignored
            Not used, present for API consistency.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        logger.info("Checking geospatial library dependencies")

        self.dependency_status_ = _dependency_status

        self.available_features_ = self._determine_available_features()

        missing_required = [
            lib
            for lib in self.required_libraries
            if not self.dependency_status_.is_available(lib)
        ]

        if missing_required:
            missing_names = [lib.value for lib in missing_required]
            message = f"Missing required geospatial libraries: {missing_names}"

            if self.fail_on_missing:
                raise ImportError(message)
            else:
                logger.warning(message)
                if self.enable_fallbacks:
                    logger.info("Enabling fallback mechanisms for missing libraries")
                    self.dependency_status_.fallback_active = True

        logger.info(
            f"Available geospatial libraries: "
            f"{[lib.value for lib, available in self.dependency_status_.available_libraries.items() if available]}"
        )
        logger.info(f"Available features: {list(self.available_features_.keys())}")

        return self

    def transform(self, X: Any) -> Any:
        """
        Return the input data unchanged (pass-through transformer).

        When X is a DataFrame, returns it as-is to enable pipeline compatibility.
        Otherwise returns dependency information dict.

        Parameters
        ----------
        X : DataFrame or any
            Input data to pass through.

        Returns
        -------
        X : DataFrame or dict
            Input data unchanged if DataFrame, otherwise dependency info.
        """
        check_is_fitted(self, "dependency_status_")

        if isinstance(X, pd.DataFrame):
            return X

        return {
            "dependency_status": self.dependency_status_,
            "available_features": self.available_features_,
            "fallback_active": self.dependency_status_.fallback_active,
        }

    def _determine_available_features(self) -> Dict[str, bool]:
        """Determine which features are available based on dependencies."""
        features = {
            "coordinate_transformations": self.dependency_status_.is_available(
                GeospatialLibrary.PYPROJ
            ),
            "advanced_geometries": self.dependency_status_.is_available(
                GeospatialLibrary.SHAPELY
            ),
            "spatial_dataframes": self.dependency_status_.is_available(
                GeospatialLibrary.GEOPANDAS
            ),
            "network_analysis": self.dependency_status_.is_available(
                GeospatialLibrary.NETWORKX
            ),
            "spatial_statistics": True,  # Always available with scipy/sklearn
            "kriging_interpolation": self.dependency_status_.is_available(
                GeospatialLibrary.SCIKIT_GSTAT
            ),
            "raster_operations": self.dependency_status_.is_available(
                GeospatialLibrary.RASTERIO
            ),
            "vector_io": self.dependency_status_.is_available(GeospatialLibrary.FIONA),
            "basic_operations": True,  # Core operations always available
            "distance_calculations": True,  # Basic distance calculations always available
        }
        return features

    def get_capability_report(self) -> str:
        """Generate a human-readable capability report."""
        check_is_fitted(self, "dependency_status_")

        report_lines = ["Geospatial Analysis Capabilities Report", "=" * 40]

        report_lines.append("\nAvailable Libraries:")
        for lib, available in self.dependency_status_.available_libraries.items():
            status = "✓" if available else "✗"
            version = self.dependency_status_.versions.get(lib, "N/A")
            report_lines.append(f"  {status} {lib.value} ({version})")

        report_lines.append("\nAvailable Features:")
        for feature, available in self.available_features_.items():
            status = "✓" if available else "✗"
            report_lines.append(f"  {status} {feature.replace('_', ' ').title()}")

        if self.dependency_status_.fallback_active:
            report_lines.append(
                "\n⚠️  Fallback mode active - some functionality may be limited"
            )

        missing = self.dependency_status_.get_missing_libraries()
        if missing:
            report_lines.append(
                f"\nMissing Libraries: {[lib.value for lib in missing]}"
            )
            report_lines.append(
                "Install with: pip install "
                + " ".join(lib.value.replace("_", "-") for lib in missing)
            )

        return "\n".join(report_lines)


# Convenience functions
def get_dependency_status():
    """Get current geospatial dependency status."""
    return _dependency_status


def check_geospatial_capabilities() -> Dict[str, Any]:
    """Check and return geospatial analysis capabilities."""
    checker = GeospatialDependencyChecker()
    checker.fit()
    return checker.transform(None)


def create_spatial_dataframe(
    data: pd.DataFrame,
    geometry_column: str = "geometry",
    x_column: str = "x",
    y_column: str = "y",
    crs: Optional[str] = None,
) -> SpatialDataFrame:
    """
    Create a SpatialDataFrame from a regular DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with spatial information.
    geometry_column : str, default 'geometry'
        Name of geometry column if it exists.
    x_column : str, default 'x'
        Name of x-coordinate column.
    y_column : str, default 'y'
        Name of y-coordinate column.
    crs : str, optional
        Coordinate reference system.

    Returns
    -------
    SpatialDataFrame
        Spatial data frame with optional geopandas integration.
    """
    return SpatialDataFrame(data=data, geometry_column=geometry_column, crs=crs)
