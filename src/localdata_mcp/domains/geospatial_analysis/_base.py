"""
Base types, enums, and dataclasses for the geospatial analysis domain.
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ...logging_manager import get_logger

logger = get_logger(__name__)

# Suppress specific warnings for geospatial operations
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class GeospatialLibrary(Enum):
    """Enumeration of optional geospatial libraries."""

    GEOPANDAS = "geopandas"
    SHAPELY = "shapely"
    PYPROJ = "pyproj"
    NETWORKX = "networkx"
    SCIKIT_GSTAT = "scikit_gstat"
    RASTERIO = "rasterio"
    FIONA = "fiona"


class SpatialJoinType(Enum):
    """Types of spatial join operations."""

    INTERSECTS = "intersects"
    WITHIN = "within"
    CONTAINS = "contains"
    TOUCHES = "touches"
    OVERLAPS = "overlaps"
    CROSSES = "crosses"
    DISJOINT = "disjoint"
    NEAREST = "nearest"


class OverlayOperation(Enum):
    """Types of spatial overlay operations."""

    INTERSECTION = "intersection"
    UNION = "union"
    DIFFERENCE = "difference"
    SYMMETRIC_DIFFERENCE = "symmetric_difference"
    IDENTITY = "identity"


class NetworkAnalysisType(Enum):
    """Types of network analysis operations."""

    SHORTEST_PATH = "shortest_path"
    ROUTING = "routing"
    ACCESSIBILITY = "accessibility"
    ISOCHRONE = "isochrone"
    NETWORK_CONNECTIVITY = "connectivity"
    SERVICE_AREA = "service_area"


@dataclass
class DependencyStatus:
    """Status of geospatial library dependencies."""

    available_libraries: Dict[GeospatialLibrary, bool] = field(default_factory=dict)
    import_errors: Dict[GeospatialLibrary, str] = field(default_factory=dict)
    versions: Dict[GeospatialLibrary, Optional[str]] = field(default_factory=dict)
    fallback_active: bool = False

    def __post_init__(self):
        """Initialize dependency status on creation."""
        if not self.available_libraries:
            self._check_dependencies()

    def _check_dependencies(self):
        """Check availability of all geospatial libraries."""
        for lib in GeospatialLibrary:
            try:
                if lib == GeospatialLibrary.GEOPANDAS:
                    import geopandas as gpd

                    self.versions[lib] = gpd.__version__
                elif lib == GeospatialLibrary.SHAPELY:
                    import shapely

                    self.versions[lib] = shapely.__version__
                elif lib == GeospatialLibrary.PYPROJ:
                    import pyproj

                    self.versions[lib] = pyproj.__version__
                elif lib == GeospatialLibrary.NETWORKX:
                    import networkx as nx

                    self.versions[lib] = nx.__version__
                elif lib == GeospatialLibrary.SCIKIT_GSTAT:
                    import skgstat

                    self.versions[lib] = skgstat.__version__
                elif lib == GeospatialLibrary.RASTERIO:
                    import rasterio

                    self.versions[lib] = rasterio.__version__
                elif lib == GeospatialLibrary.FIONA:
                    import fiona

                    self.versions[lib] = fiona.__version__

                self.available_libraries[lib] = True
                logger.debug(f"Successfully imported {lib.value} v{self.versions[lib]}")

            except ImportError as e:
                self.available_libraries[lib] = False
                self.import_errors[lib] = str(e)
                self.versions[lib] = None
                logger.debug(f"Failed to import {lib.value}: {e}")

    def is_available(self, library: GeospatialLibrary) -> bool:
        """Check if a specific library is available."""
        return self.available_libraries.get(library, False)

    def get_missing_libraries(self) -> List[GeospatialLibrary]:
        """Get list of missing libraries."""
        return [
            lib for lib, available in self.available_libraries.items() if not available
        ]

    def has_core_geospatial(self) -> bool:
        """Check if core geospatial libraries are available."""
        return self.is_available(GeospatialLibrary.GEOPANDAS) and self.is_available(
            GeospatialLibrary.SHAPELY
        )


# Global dependency status checker
_dependency_status = DependencyStatus()
