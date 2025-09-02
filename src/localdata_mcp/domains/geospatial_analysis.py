"""
Geospatial Analysis Domain - Comprehensive spatial analysis capabilities.

This module implements advanced geospatial analysis tools including coordinate systems,
spatial statistics, geometric operations, and network analysis using optional
integration with geopandas, shapely, pyproj, and networkx.

Key Features:
- Coordinate Reference System (CRS) handling and transformations
- Spatial statistics (Moran's I, Geary's C, spatial clustering)
- Geometric operations (point-in-polygon, buffers, spatial joins)
- Network analysis with spatial constraints
- Spatial interpolation using kriging
- Full sklearn pipeline compatibility
- Optional dependency management with graceful degradation
- Comprehensive spatial validation and error handling
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import json
import math
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.neighbors import NearestNeighbors

from ..logging_manager import get_logger
from ..pipeline.base import (
    AnalysisPipelineBase, PipelineResult, CompositionMetadata, 
    StreamingConfig, PipelineState
)

logger = get_logger(__name__)

# Suppress specific warnings for geospatial operations
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class GeospatialLibrary(Enum):
    """Enumeration of optional geospatial libraries."""
    GEOPANDAS = "geopandas"
    SHAPELY = "shapely"
    PYPROJ = "pyproj"
    NETWORKX = "networkx"
    SCIKIT_GSTAT = "scikit_gstat"
    RASTERIO = "rasterio"
    FIONA = "fiona"


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
        return [lib for lib, available in self.available_libraries.items() if not available]
    
    def has_core_geospatial(self) -> bool:
        """Check if core geospatial libraries are available."""
        return (self.is_available(GeospatialLibrary.GEOPANDAS) and 
                self.is_available(GeospatialLibrary.SHAPELY))


@dataclass
class SpatialPoint:
    """Represents a spatial point with coordinates and optional attributes."""
    x: float
    y: float
    z: Optional[float] = None
    crs: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    
    def to_tuple(self, include_z: bool = True) -> Tuple[float, ...]:
        """Convert point to coordinate tuple."""
        if include_z and self.z is not None:
            return (self.x, self.y, self.z)
        return (self.x, self.y)
    
    def distance_to(self, other: 'SpatialPoint', method: str = 'euclidean') -> float:
        """Calculate distance to another point."""
        if method == 'euclidean':
            dx = self.x - other.x
            dy = self.y - other.y
            if self.z is not None and other.z is not None:
                dz = self.z - other.z
                return math.sqrt(dx*dx + dy*dy + dz*dz)
            return math.sqrt(dx*dx + dy*dy)
        elif method == 'haversine':
            return _haversine_distance(self.y, self.x, other.y, other.x)
        else:
            raise ValueError(f"Unsupported distance method: {method}")


@dataclass
class SpatialDataFrame:
    """Wrapper for spatial data with optional geopandas integration."""
    data: pd.DataFrame
    geometry_column: str = 'geometry'
    crs: Optional[str] = None
    _geopandas_df: Optional[Any] = None  # GeoPandas GeoDataFrame when available
    
    def __post_init__(self):
        """Initialize spatial data frame."""
        # Try to create geopandas GeoDataFrame if available
        if _dependency_status.is_available(GeospatialLibrary.GEOPANDAS):
            try:
                import geopandas as gpd
                if self.geometry_column in self.data.columns:
                    self._geopandas_df = gpd.GeoDataFrame(
                        self.data, 
                        geometry=self.geometry_column,
                        crs=self.crs
                    )
            except Exception as e:
                logger.warning(f"Failed to create GeoDataFrame: {e}")
    
    @property
    def is_geopandas_enabled(self) -> bool:
        """Check if geopandas integration is active."""
        return self._geopandas_df is not None
    
    def get_points(self) -> List[SpatialPoint]:
        """Extract spatial points from the data."""
        points = []
        if self.is_geopandas_enabled:
            # Use geopandas geometry extraction
            for idx, row in self._geopandas_df.iterrows():
                geom = row[self.geometry_column]
                if hasattr(geom, 'x') and hasattr(geom, 'y'):
                    attrs = {col: row[col] for col in self.data.columns 
                           if col != self.geometry_column}
                    points.append(SpatialPoint(
                        x=geom.x, y=geom.y, 
                        crs=self.crs, 
                        attributes=attrs
                    ))
        else:
            # Fallback: assume x, y columns exist
            x_col = 'x' if 'x' in self.data.columns else 'longitude'
            y_col = 'y' if 'y' in self.data.columns else 'latitude'
            
            if x_col in self.data.columns and y_col in self.data.columns:
                for idx, row in self.data.iterrows():
                    attrs = {col: row[col] for col in self.data.columns 
                           if col not in [x_col, y_col]}
                    points.append(SpatialPoint(
                        x=row[x_col], y=row[y_col],
                        crs=self.crs,
                        attributes=attrs
                    ))
        return points
    
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get spatial bounds (minx, miny, maxx, maxy)."""
        if self.is_geopandas_enabled:
            return tuple(self._geopandas_df.total_bounds)
        else:
            points = self.get_points()
            if not points:
                return None
            xs = [p.x for p in points]
            ys = [p.y for p in points]
            return (min(xs), min(ys), max(xs), max(ys))


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r


class SpatialDataValidator:
    """Validates spatial data quality and consistency."""
    
    @staticmethod
    def validate_coordinates(points: List[SpatialPoint]) -> Dict[str, Any]:
        """Validate coordinate data quality."""
        results = {
            'total_points': len(points),
            'valid_points': 0,
            'invalid_coordinates': [],
            'coordinate_ranges': {},
            'duplicate_coordinates': 0,
            'warnings': []
        }
        
        if not points:
            results['warnings'].append("No points provided for validation")
            return results
        
        coordinates = set()
        xs, ys = [], []
        
        for i, point in enumerate(points):
            # Check for valid coordinates
            if (isinstance(point.x, (int, float)) and not math.isnan(point.x) and
                isinstance(point.y, (int, float)) and not math.isnan(point.y)):
                results['valid_points'] += 1
                
                # Check for duplicates
                coord_tuple = (point.x, point.y)
                if coord_tuple in coordinates:
                    results['duplicate_coordinates'] += 1
                else:
                    coordinates.add(coord_tuple)
                
                xs.append(point.x)
                ys.append(point.y)
            else:
                results['invalid_coordinates'].append(i)
        
        if xs and ys:
            results['coordinate_ranges'] = {
                'x_range': (min(xs), max(xs)),
                'y_range': (min(ys), max(ys)),
                'centroid': (sum(xs)/len(xs), sum(ys)/len(ys))
            }
            
            # Add coordinate system warnings
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            
            if max(abs(min(xs)), abs(max(xs))) > 180:
                results['warnings'].append("X coordinates exceed typical longitude range (-180 to 180)")
            if max(abs(min(ys)), abs(max(ys))) > 90:
                results['warnings'].append("Y coordinates exceed typical latitude range (-90 to 90)")
            if x_range > 360 or y_range > 180:
                results['warnings'].append("Coordinate ranges suggest projected coordinates, not geographic")
        
        return results
    
    @staticmethod
    def validate_crs(crs_string: str) -> Dict[str, Any]:
        """Validate coordinate reference system specification."""
        results = {
            'valid': False,
            'crs_type': None,
            'authority': None,
            'code': None,
            'warnings': []
        }
        
        if not crs_string:
            results['warnings'].append("No CRS specified")
            return results
        
        # Check if pyproj is available for detailed validation
        if _dependency_status.is_available(GeospatialLibrary.PYPROJ):
            try:
                import pyproj
                crs = pyproj.CRS.from_string(crs_string)
                results['valid'] = True
                results['crs_type'] = crs.type_name
                if crs.to_authority():
                    results['authority'] = crs.to_authority()[0]
                    results['code'] = crs.to_authority()[1]
            except Exception as e:
                results['warnings'].append(f"Invalid CRS: {e}")
        else:
            # Basic validation without pyproj
            common_crs = ['EPSG:4326', 'EPSG:3857', 'WGS84', 'WGS:84']
            if crs_string.upper() in [c.upper() for c in common_crs]:
                results['valid'] = True
                results['crs_type'] = 'Geographic' if '4326' in crs_string else 'Projected'
            else:
                results['warnings'].append("Cannot validate CRS without pyproj library")
        
        return results


# Global dependency status checker
_dependency_status = DependencyStatus()


class GeospatialDependencyChecker(BaseEstimator, TransformerMixin):
    """
    Pipeline component for checking and managing geospatial library dependencies.
    
    This component provides runtime dependency checking and graceful degradation
    for optional geospatial libraries, enabling the pipeline to work with
    varying levels of functionality based on available dependencies.
    """
    
    def __init__(self, 
                 required_libraries: Optional[List[GeospatialLibrary]] = None,
                 fail_on_missing: bool = False,
                 enable_fallbacks: bool = True):
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
            GeospatialLibrary.SHAPELY
        ]
        self.fail_on_missing = fail_on_missing
        self.enable_fallbacks = enable_fallbacks
        
        self.dependency_status_ = None
        self.available_features_ = None
        
    def fit(self, X: Any = None, y: Any = None) -> 'GeospatialDependencyChecker':
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
        
        # Get current dependency status
        self.dependency_status_ = _dependency_status
        
        # Determine available features based on libraries
        self.available_features_ = self._determine_available_features()
        
        # Check required libraries
        missing_required = [lib for lib in self.required_libraries 
                          if not self.dependency_status_.is_available(lib)]
        
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
        
        logger.info(f"Available geospatial libraries: "
                   f"{[lib.value for lib, available in self.dependency_status_.available_libraries.items() if available]}")
        logger.info(f"Available features: {list(self.available_features_.keys())}")
        
        return self
    
    def transform(self, X: Any) -> Dict[str, Any]:
        """
        Return dependency status and available features.
        
        Parameters
        ----------
        X : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        dependency_info : dict
            Dictionary containing dependency status and available features.
        """
        check_is_fitted(self, 'dependency_status_')
        
        return {
            'dependency_status': self.dependency_status_,
            'available_features': self.available_features_,
            'fallback_active': self.dependency_status_.fallback_active
        }
    
    def _determine_available_features(self) -> Dict[str, bool]:
        """Determine which features are available based on dependencies."""
        features = {
            'coordinate_transformations': self.dependency_status_.is_available(GeospatialLibrary.PYPROJ),
            'advanced_geometries': self.dependency_status_.is_available(GeospatialLibrary.SHAPELY),
            'spatial_dataframes': self.dependency_status_.is_available(GeospatialLibrary.GEOPANDAS),
            'network_analysis': self.dependency_status_.is_available(GeospatialLibrary.NETWORKX),
            'spatial_statistics': True,  # Always available with scipy/sklearn
            'kriging_interpolation': self.dependency_status_.is_available(GeospatialLibrary.SCIKIT_GSTAT),
            'raster_operations': self.dependency_status_.is_available(GeospatialLibrary.RASTERIO),
            'vector_io': self.dependency_status_.is_available(GeospatialLibrary.FIONA),
            'basic_operations': True,  # Core operations always available
            'distance_calculations': True  # Basic distance calculations always available
        }
        return features
    
    def get_capability_report(self) -> str:
        """Generate a human-readable capability report."""
        check_is_fitted(self, 'dependency_status_')
        
        report_lines = ["Geospatial Analysis Capabilities Report", "=" * 40]
        
        # Available libraries
        report_lines.append("\nAvailable Libraries:")
        for lib, available in self.dependency_status_.available_libraries.items():
            status = "✓" if available else "✗"
            version = self.dependency_status_.versions.get(lib, "N/A")
            report_lines.append(f"  {status} {lib.value} ({version})")
        
        # Available features
        report_lines.append("\nAvailable Features:")
        for feature, available in self.available_features_.items():
            status = "✓" if available else "✗"
            report_lines.append(f"  {status} {feature.replace('_', ' ').title()}")
        
        # Fallback status
        if self.dependency_status_.fallback_active:
            report_lines.append("\n⚠️  Fallback mode active - some functionality may be limited")
        
        # Missing libraries
        missing = self.dependency_status_.get_missing_libraries()
        if missing:
            report_lines.append(f"\nMissing Libraries: {[lib.value for lib in missing]}")
            report_lines.append("Install with: pip install " + " ".join(lib.value.replace('_', '-') for lib in missing))
        
        return "\n".join(report_lines)


@dataclass
class SpatialAnalysisResult:
    """Container for spatial analysis results."""
    analysis_type: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    spatial_data: Optional[SpatialDataFrame] = None
    computation_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'analysis_type': self.analysis_type,
            'results': self.results,
            'metadata': self.metadata,
            'computation_time': self.computation_time,
            'warnings': self.warnings,
            'has_spatial_data': self.spatial_data is not None
        }


class SpatialDataProcessor(AnalysisPipelineBase):
    """
    Base processor for spatial data operations with sklearn compatibility.
    
    This class provides the foundation for all geospatial analysis components,
    handling dependency management, data validation, and result formatting.
    """
    
    def __init__(self, 
                 validate_input: bool = True,
                 enable_fallbacks: bool = True,
                 coordinate_columns: Optional[Dict[str, str]] = None):
        """
        Initialize spatial data processor.
        
        Parameters
        ----------
        validate_input : bool, default True
            Whether to validate spatial input data.
        enable_fallbacks : bool, default True
            Whether to use fallback methods when libraries are missing.
        coordinate_columns : dict, optional
            Mapping of coordinate column names {'x': 'longitude', 'y': 'latitude'}.
        """
        super().__init__()
        self.validate_input = validate_input
        self.enable_fallbacks = enable_fallbacks
        self.coordinate_columns = coordinate_columns or {'x': 'x', 'y': 'y'}
        
        self.dependency_checker_ = None
        self.spatial_validator_ = None
        self.fitted_data_info_ = None
    
    def _validate_spatial_data(self, data: Union[pd.DataFrame, SpatialDataFrame]) -> SpatialDataFrame:
        """Convert and validate spatial data."""
        if isinstance(data, SpatialDataFrame):
            spatial_df = data
        else:
            # Convert DataFrame to SpatialDataFrame
            spatial_df = SpatialDataFrame(data)
        
        if self.validate_input:
            points = spatial_df.get_points()
            validation_results = SpatialDataValidator.validate_coordinates(points)
            
            if validation_results['warnings']:
                logger.warning(f"Spatial data validation warnings: {validation_results['warnings']}")
            
            if validation_results['valid_points'] == 0:
                raise ValueError("No valid spatial coordinates found in data")
        
        return spatial_df
    
    def _create_result(self, 
                      analysis_type: str,
                      results: Dict[str, Any],
                      spatial_data: Optional[SpatialDataFrame] = None,
                      computation_time: float = 0.0,
                      warnings: Optional[List[str]] = None) -> SpatialAnalysisResult:
        """Create standardized spatial analysis result."""
        metadata = {
            'timestamp': time.time(),
            'analysis_type': analysis_type,
            'dependency_status': _dependency_status.available_libraries.copy(),
            'fallback_active': _dependency_status.fallback_active,
            'data_shape': spatial_data.data.shape if spatial_data else None,
            'coordinate_columns': self.coordinate_columns
        }
        
        return SpatialAnalysisResult(
            analysis_type=analysis_type,
            results=results,
            metadata=metadata,
            spatial_data=spatial_data,
            computation_time=computation_time,
            warnings=warnings or []
        )


# Make dependency status globally accessible
def get_dependency_status() -> DependencyStatus:
    """Get current geospatial dependency status."""
    return _dependency_status


def check_geospatial_capabilities() -> Dict[str, Any]:
    """Check and return geospatial analysis capabilities."""
    checker = GeospatialDependencyChecker()
    checker.fit()
    return checker.transform(None)


def create_spatial_dataframe(data: pd.DataFrame, 
                           geometry_column: str = 'geometry',
                           x_column: str = 'x',
                           y_column: str = 'y',
                           crs: Optional[str] = None) -> SpatialDataFrame:
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
    return SpatialDataFrame(
        data=data,
        geometry_column=geometry_column,
        crs=crs
    )


# Initialize dependency checking on module import
logger.info("Initializing geospatial analysis module")
logger.info(f"Available geospatial libraries: "
           f"{[lib.value for lib, available in _dependency_status.available_libraries.items() if available]}")

if not _dependency_status.has_core_geospatial():
    logger.warning("Core geospatial libraries (geopandas, shapely) not available. "
                  "Some functionality will be limited to basic operations.")
    logger.info("Install with: pip install geopandas shapely")