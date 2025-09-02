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
from scipy import stats
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


@dataclass
class CoordinateTransformation:
    """Container for coordinate transformation operations."""
    source_crs: str
    target_crs: str
    transformation_accuracy: Optional[float] = None
    transformation_method: Optional[str] = None
    is_valid: bool = True
    error_message: Optional[str] = None


class CoordinateReferenceSystem:
    """
    Coordinate Reference System handler with pyproj integration.
    
    Provides CRS validation, conversion, and transformation capabilities
    with fallback mechanisms when pyproj is not available.
    """
    
    # Common CRS definitions for fallback mode
    COMMON_CRS = {
        'EPSG:4326': {
            'name': 'WGS 84',
            'type': 'geographic',
            'unit': 'degree',
            'authority': 'EPSG',
            'code': '4326'
        },
        'EPSG:3857': {
            'name': 'WGS 84 / Pseudo-Mercator',
            'type': 'projected',
            'unit': 'metre',
            'authority': 'EPSG',
            'code': '3857'
        },
        'EPSG:4269': {
            'name': 'NAD83',
            'type': 'geographic',
            'unit': 'degree',
            'authority': 'EPSG',
            'code': '4269'
        },
        'EPSG:32633': {
            'name': 'WGS 84 / UTM zone 33N',
            'type': 'projected',
            'unit': 'metre',
            'authority': 'EPSG',
            'code': '32633'
        }
    }
    
    def __init__(self):
        """Initialize CRS handler."""
        self.pyproj_available = _dependency_status.is_available(GeospatialLibrary.PYPROJ)
        self._crs_cache = {}
        
        if self.pyproj_available:
            try:
                import pyproj
                self.pyproj = pyproj
                logger.debug("PyProj available for CRS operations")
            except ImportError:
                self.pyproj_available = False
                logger.warning("PyProj import failed, using fallback CRS handling")
    
    def validate_crs(self, crs_string: str) -> Dict[str, Any]:
        """
        Validate a coordinate reference system specification.
        
        Parameters
        ----------
        crs_string : str
            CRS specification (EPSG code, proj4 string, WKT, etc.)
            
        Returns
        -------
        validation_result : dict
            Validation results including validity, type, and metadata.
        """
        result = {
            'valid': False,
            'crs_string': crs_string,
            'crs_type': None,
            'authority': None,
            'code': None,
            'name': None,
            'unit': None,
            'area_of_use': None,
            'warnings': []
        }
        
        if not crs_string:
            result['warnings'].append("Empty CRS specification")
            return result
        
        if self.pyproj_available:
            try:
                crs = self.pyproj.CRS.from_string(crs_string)
                result['valid'] = True
                result['crs_type'] = crs.type_name
                result['name'] = crs.name
                
                # Get authority information
                if crs.to_authority():
                    result['authority'] = crs.to_authority()[0]
                    result['code'] = crs.to_authority()[1]
                
                # Get unit information
                if hasattr(crs, 'axis_info') and crs.axis_info:
                    result['unit'] = crs.axis_info[0].unit_name
                
                # Get area of use
                if hasattr(crs, 'area_of_use') and crs.area_of_use:
                    result['area_of_use'] = {
                        'name': crs.area_of_use.name,
                        'bounds': [crs.area_of_use.west, crs.area_of_use.south,
                                 crs.area_of_use.east, crs.area_of_use.north]
                    }
                
                self._crs_cache[crs_string] = crs
                
            except Exception as e:
                result['warnings'].append(f"PyProj CRS validation failed: {e}")
                # Try fallback validation
                result.update(self._fallback_crs_validation(crs_string))
        else:
            # Use fallback validation
            result.update(self._fallback_crs_validation(crs_string))
        
        return result
    
    def _fallback_crs_validation(self, crs_string: str) -> Dict[str, Any]:
        """Fallback CRS validation without pyproj."""
        result = {'valid': False, 'warnings': []}
        
        # Normalize CRS string
        crs_upper = crs_string.upper().strip()
        
        # Check against common CRS definitions
        if crs_upper in self.COMMON_CRS:
            crs_info = self.COMMON_CRS[crs_upper]
            result.update({
                'valid': True,
                'crs_type': crs_info['type'],
                'authority': crs_info['authority'],
                'code': crs_info['code'],
                'name': crs_info['name'],
                'unit': crs_info['unit']
            })
        elif any(common in crs_upper for common in ['WGS84', 'WGS:84', 'WGS 84']):
            result.update({
                'valid': True,
                'crs_type': 'geographic',
                'authority': 'EPSG',
                'code': '4326',
                'name': 'WGS 84',
                'unit': 'degree'
            })
            result['warnings'].append("CRS assumed to be WGS84 (EPSG:4326)")
        else:
            result['warnings'].append("Cannot validate CRS without pyproj library")
            # Still mark as potentially valid for basic operations
            if 'EPSG:' in crs_upper or '+proj=' in crs_upper:
                result['valid'] = True
                result['warnings'].append("CRS format recognized but not validated")
        
        return result
    
    def get_transformation(self, source_crs: str, target_crs: str) -> CoordinateTransformation:
        """
        Create coordinate transformation between two CRS.
        
        Parameters
        ----------
        source_crs : str
            Source coordinate reference system.
        target_crs : str  
            Target coordinate reference system.
            
        Returns
        -------
        transformation : CoordinateTransformation
            Transformation object with metadata.
        """
        transformation = CoordinateTransformation(
            source_crs=source_crs,
            target_crs=target_crs
        )
        
        if self.pyproj_available:
            try:
                # Validate both CRS
                source_validation = self.validate_crs(source_crs)
                target_validation = self.validate_crs(target_crs)
                
                if not source_validation['valid']:
                    transformation.is_valid = False
                    transformation.error_message = f"Invalid source CRS: {source_crs}"
                    return transformation
                
                if not target_validation['valid']:
                    transformation.is_valid = False
                    transformation.error_message = f"Invalid target CRS: {target_crs}"
                    return transformation
                
                # Create transformer
                source_crs_obj = self._crs_cache.get(source_crs) or self.pyproj.CRS.from_string(source_crs)
                target_crs_obj = self._crs_cache.get(target_crs) or self.pyproj.CRS.from_string(target_crs)
                
                transformer = self.pyproj.Transformer.from_crs(
                    source_crs_obj, target_crs_obj, always_xy=True
                )
                
                transformation.transformation_accuracy = getattr(transformer, 'accuracy', None)
                transformation.transformation_method = str(transformer.description) if hasattr(transformer, 'description') else None
                
                # Cache the transformer for reuse
                cache_key = f"{source_crs}_{target_crs}"
                if not hasattr(self, '_transformer_cache'):
                    self._transformer_cache = {}
                self._transformer_cache[cache_key] = transformer
                
            except Exception as e:
                transformation.is_valid = False
                transformation.error_message = f"Transformation creation failed: {e}"
        else:
            # Basic validation without pyproj
            if source_crs == target_crs:
                transformation.transformation_method = "identity"
            elif self._is_geographic_crs(source_crs) and self._is_geographic_crs(target_crs):
                transformation.transformation_method = "geographic_fallback"
            else:
                transformation.is_valid = False
                transformation.error_message = "Cannot create transformation without pyproj library"
        
        return transformation
    
    def transform_coordinates(self, 
                            coordinates: Union[List[Tuple[float, float]], np.ndarray],
                            source_crs: str,
                            target_crs: str) -> Tuple[np.ndarray, List[str]]:
        """
        Transform coordinates between coordinate reference systems.
        
        Parameters
        ----------
        coordinates : list of tuples or ndarray
            Input coordinates as [(x, y), ...] or array.
        source_crs : str
            Source coordinate reference system.
        target_crs : str
            Target coordinate reference system.
            
        Returns
        -------
        transformed_coords : ndarray
            Transformed coordinates.
        warnings : list
            Any transformation warnings.
        """
        warnings = []
        
        # Convert coordinates to numpy array
        if isinstance(coordinates, list):
            coords_array = np.array(coordinates)
        else:
            coords_array = np.asarray(coordinates)
        
        if coords_array.shape[1] < 2:
            raise ValueError("Coordinates must have at least 2 dimensions (x, y)")
        
        # Handle identity transformation
        if source_crs == target_crs:
            warnings.append("Source and target CRS are identical - no transformation needed")
            return coords_array, warnings
        
        if self.pyproj_available:
            try:
                # Get or create transformer
                cache_key = f"{source_crs}_{target_crs}"
                if hasattr(self, '_transformer_cache') and cache_key in self._transformer_cache:
                    transformer = self._transformer_cache[cache_key]
                else:
                    transformation = self.get_transformation(source_crs, target_crs)
                    if not transformation.is_valid:
                        raise ValueError(transformation.error_message)
                    transformer = self._transformer_cache[cache_key]
                
                # Transform coordinates
                x_coords = coords_array[:, 0]
                y_coords = coords_array[:, 1]
                
                transformed_x, transformed_y = transformer.transform(x_coords, y_coords)
                
                # Create result array
                result_coords = np.column_stack([transformed_x, transformed_y])
                
                # Add z-coordinates if present
                if coords_array.shape[1] > 2:
                    result_coords = np.column_stack([result_coords, coords_array[:, 2:]])
                
                return result_coords, warnings
                
            except Exception as e:
                warnings.append(f"PyProj transformation failed: {e}")
                # Fall through to fallback method
        
        # Fallback transformation method
        if self._is_geographic_crs(source_crs) and self._is_geographic_crs(target_crs):
            warnings.append("Using fallback geographic transformation - may be inaccurate")
            # For now, return original coordinates with warning
            # In a more complete implementation, we might do simple datum shifts
            return coords_array, warnings
        else:
            raise ValueError(f"Cannot transform coordinates from {source_crs} to {target_crs} without pyproj library")
    
    def _is_geographic_crs(self, crs_string: str) -> bool:
        """Check if CRS is geographic (latitude/longitude)."""
        crs_upper = crs_string.upper()
        geographic_indicators = ['4326', 'WGS84', 'WGS:84', 'WGS 84', '4269', 'NAD83']
        return any(indicator in crs_upper for indicator in geographic_indicators)
    
    def get_crs_info(self, crs_string: str) -> Dict[str, Any]:
        """Get detailed information about a CRS."""
        validation = self.validate_crs(crs_string)
        
        info = {
            'crs_string': crs_string,
            'is_valid': validation['valid'],
            'name': validation.get('name', 'Unknown'),
            'type': validation.get('crs_type', 'Unknown'),
            'authority': validation.get('authority'),
            'code': validation.get('code'),
            'unit': validation.get('unit'),
            'area_of_use': validation.get('area_of_use'),
            'is_geographic': self._is_geographic_crs(crs_string),
            'warnings': validation.get('warnings', [])
        }
        
        return info
    
    def list_common_crs(self) -> Dict[str, Dict[str, str]]:
        """List commonly used coordinate reference systems."""
        return self.COMMON_CRS.copy()


class SpatialCoordinateTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for coordinate reference system transformations.
    
    This transformer handles CRS transformations within the sklearn pipeline framework,
    with support for both pyproj-based and fallback transformation methods.
    """
    
    def __init__(self, 
                 target_crs: str = 'EPSG:4326',
                 source_crs: Optional[str] = None,
                 validate_crs: bool = True,
                 coordinate_columns: Optional[List[str]] = None):
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
        self.coordinate_columns = coordinate_columns or ['x', 'y']
        
        self.crs_handler_ = None
        self.transformation_ = None
        self.fitted_source_crs_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> 'SpatialCoordinateTransformer':
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
        
        # Initialize CRS handler
        self.crs_handler_ = CoordinateReferenceSystem()
        
        # Determine source CRS
        if self.source_crs is None:
            # Try to detect CRS from data
            if isinstance(X, pd.DataFrame) and 'crs' in X.columns:
                unique_crs = X['crs'].dropna().unique()
                if len(unique_crs) == 1:
                    self.fitted_source_crs_ = unique_crs[0]
                    logger.info(f"Detected source CRS from data: {self.fitted_source_crs_}")
                elif len(unique_crs) > 1:
                    raise ValueError(f"Multiple CRS found in data: {unique_crs}")
                else:
                    raise ValueError("No CRS information found in data and source_crs not specified")
            else:
                # Assume WGS84 for latitude/longitude data
                self.fitted_source_crs_ = 'EPSG:4326'
                logger.warning("No CRS specified, assuming WGS84 (EPSG:4326)")
        else:
            self.fitted_source_crs_ = self.source_crs
        
        # Validate CRS if requested
        if self.validate_crs:
            source_validation = self.crs_handler_.validate_crs(self.fitted_source_crs_)
            target_validation = self.crs_handler_.validate_crs(self.target_crs)
            
            if not source_validation['valid']:
                raise ValueError(f"Invalid source CRS: {self.fitted_source_crs_}")
            if not target_validation['valid']:
                raise ValueError(f"Invalid target CRS: {self.target_crs}")
            
            for warning in source_validation.get('warnings', []):
                logger.warning(f"Source CRS warning: {warning}")
            for warning in target_validation.get('warnings', []):
                logger.warning(f"Target CRS warning: {warning}")
        
        # Prepare transformation
        self.transformation_ = self.crs_handler_.get_transformation(
            self.fitted_source_crs_, self.target_crs
        )
        
        if not self.transformation_.is_valid:
            raise ValueError(f"Cannot create transformation: {self.transformation_.error_message}")
        
        logger.info(f"Prepared CRS transformation: {self.fitted_source_crs_} -> {self.target_crs}")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
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
        check_is_fitted(self, 'transformation_')
        
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            
            # Extract coordinates
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
            else:
                raise ValueError(f"Coordinate columns {self.coordinate_columns} not found in DataFrame")
            
            # Transform coordinates
            transformed_coords, warnings = self.crs_handler_.transform_coordinates(
                coords, self.fitted_source_crs_, self.target_crs
            )
            
            # Update DataFrame with transformed coordinates
            for i, col in enumerate(self.coordinate_columns):
                if i < transformed_coords.shape[1]:
                    X_transformed[col] = transformed_coords[:, i]
            
            # Update CRS information if present
            if 'crs' in X_transformed.columns:
                X_transformed['crs'] = self.target_crs
            
            # Log any warnings
            for warning in warnings:
                logger.warning(f"Coordinate transformation warning: {warning}")
            
            return X_transformed
            
        else:
            # Handle array input
            coords = np.asarray(X)
            transformed_coords, warnings = self.crs_handler_.transform_coordinates(
                coords, self.fitted_source_crs_, self.target_crs
            )
            
            for warning in warnings:
                logger.warning(f"Coordinate transformation warning: {warning}")
            
            return transformed_coords
    
    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about the fitted transformation."""
        check_is_fitted(self, 'transformation_')
        
        return {
            'source_crs': self.fitted_source_crs_,
            'target_crs': self.target_crs,
            'transformation_valid': self.transformation_.is_valid,
            'transformation_accuracy': self.transformation_.transformation_accuracy,
            'transformation_method': self.transformation_.transformation_method,
            'coordinate_columns': self.coordinate_columns
        }


class SpatialDistanceCalculator:
    """
    Comprehensive spatial distance calculation tools.
    
    Provides various distance metrics for spatial data including great circle
    distances, euclidean distances, and optimized batch calculations.
    """
    
    def __init__(self, default_crs: str = 'EPSG:4326'):
        """
        Initialize distance calculator.
        
        Parameters
        ----------
        default_crs : str, default 'EPSG:4326'
            Default coordinate reference system for calculations.
        """
        self.default_crs = default_crs
        self.crs_handler = CoordinateReferenceSystem()
        
    def haversine_distance(self, 
                          lat1: Union[float, np.ndarray], 
                          lon1: Union[float, np.ndarray],
                          lat2: Union[float, np.ndarray], 
                          lon2: Union[float, np.ndarray],
                          unit: str = 'km') -> Union[float, np.ndarray]:
        """
        Calculate great circle distance using haversine formula.
        
        Parameters
        ----------
        lat1, lon1 : float or array-like
            Latitude and longitude of first point(s) in decimal degrees.
        lat2, lon2 : float or array-like
            Latitude and longitude of second point(s) in decimal degrees.
        unit : str, default 'km'
            Distance unit ('km', 'm', 'mi', 'nmi').
            
        Returns
        -------
        distance : float or array
            Great circle distance(s) in specified unit.
        """
        # Convert to numpy arrays for vectorization
        lat1, lon1, lat2, lon2 = np.asarray(lat1), np.asarray(lon1), np.asarray(lat2), np.asarray(lon2)
        
        # Convert decimal degrees to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to avoid numerical errors
        
        # Earth radius in different units
        radius = {
            'km': 6371.0,
            'm': 6371000.0,
            'mi': 3959.0,
            'nmi': 3440.0
        }
        
        if unit not in radius:
            raise ValueError(f"Unsupported unit: {unit}. Use 'km', 'm', 'mi', or 'nmi'")
        
        distance = c * radius[unit]
        
        # Return scalar if input was scalar
        if distance.ndim == 0:
            return float(distance)
        return distance
    
    def euclidean_distance(self, 
                          x1: Union[float, np.ndarray], 
                          y1: Union[float, np.ndarray],
                          x2: Union[float, np.ndarray], 
                          y2: Union[float, np.ndarray],
                          z1: Optional[Union[float, np.ndarray]] = None,
                          z2: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
        """
        Calculate euclidean distance between points.
        
        Parameters
        ----------
        x1, y1 : float or array-like
            X and Y coordinates of first point(s).
        x2, y2 : float or array-like
            X and Y coordinates of second point(s).
        z1, z2 : float or array-like, optional
            Z coordinates for 3D distance calculation.
            
        Returns
        -------
        distance : float or array
            Euclidean distance(s).
        """
        x1, y1, x2, y2 = np.asarray(x1), np.asarray(y1), np.asarray(x2), np.asarray(y2)
        
        dx = x2 - x1
        dy = y2 - y1
        
        if z1 is not None and z2 is not None:
            z1, z2 = np.asarray(z1), np.asarray(z2)
            dz = z2 - z1
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        else:
            distance = np.sqrt(dx*dx + dy*dy)
        
        # Return scalar if input was scalar
        if distance.ndim == 0:
            return float(distance)
        return distance
    
    def manhattan_distance(self, 
                          x1: Union[float, np.ndarray], 
                          y1: Union[float, np.ndarray],
                          x2: Union[float, np.ndarray], 
                          y2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Manhattan (taxicab) distance between points.
        
        Parameters
        ----------
        x1, y1 : float or array-like
            X and Y coordinates of first point(s).
        x2, y2 : float or array-like
            X and Y coordinates of second point(s).
            
        Returns
        -------
        distance : float or array
            Manhattan distance(s).
        """
        x1, y1, x2, y2 = np.asarray(x1), np.asarray(y1), np.asarray(x2), np.asarray(y2)
        
        distance = np.abs(x2 - x1) + np.abs(y2 - y1)
        
        # Return scalar if input was scalar
        if distance.ndim == 0:
            return float(distance)
        return distance
    
    def calculate_distance(self,
                          point1: Union[SpatialPoint, Tuple[float, float]],
                          point2: Union[SpatialPoint, Tuple[float, float]],
                          method: str = 'auto',
                          crs: Optional[str] = None) -> float:
        """
        Calculate distance between two spatial points using appropriate method.
        
        Parameters
        ----------
        point1, point2 : SpatialPoint or tuple
            Spatial points to calculate distance between.
        method : str, default 'auto'
            Distance calculation method ('auto', 'haversine', 'euclidean', 'manhattan').
        crs : str, optional
            Coordinate reference system. If None, uses default or point CRS.
            
        Returns
        -------
        distance : float
            Distance between points.
        """
        # Convert tuples to SpatialPoint objects
        if isinstance(point1, tuple):
            point1 = SpatialPoint(x=point1[0], y=point1[1], crs=crs)
        if isinstance(point2, tuple):
            point2 = SpatialPoint(x=point2[0], y=point2[1], crs=crs)
        
        # Determine CRS
        effective_crs = crs or point1.crs or point2.crs or self.default_crs
        
        # Auto-select method based on CRS
        if method == 'auto':
            if self.crs_handler._is_geographic_crs(effective_crs):
                method = 'haversine'
            else:
                method = 'euclidean'
        
        # Calculate distance
        if method == 'haversine':
            return self.haversine_distance(point1.y, point1.x, point2.y, point2.x)
        elif method == 'euclidean':
            return self.euclidean_distance(point1.x, point1.y, point2.x, point2.y)
        elif method == 'manhattan':
            return self.manhattan_distance(point1.x, point1.y, point2.x, point2.y)
        else:
            raise ValueError(f"Unsupported distance method: {method}")
    
    def distance_matrix(self,
                       points1: List[Union[SpatialPoint, Tuple[float, float]]],
                       points2: Optional[List[Union[SpatialPoint, Tuple[float, float]]]] = None,
                       method: str = 'auto',
                       crs: Optional[str] = None,
                       symmetric: bool = True) -> np.ndarray:
        """
        Calculate distance matrix between sets of points.
        
        Parameters
        ----------
        points1 : list of SpatialPoint or tuples
            First set of points.
        points2 : list of SpatialPoint or tuples, optional
            Second set of points. If None, uses points1 for symmetric matrix.
        method : str, default 'auto'
            Distance calculation method.
        crs : str, optional
            Coordinate reference system.
        symmetric : bool, default True
            Whether to optimize for symmetric matrix calculation.
            
        Returns
        -------
        distance_matrix : ndarray
            Distance matrix of shape (len(points1), len(points2)).
        """
        # Use points1 for both sets if points2 not provided
        if points2 is None:
            points2 = points1
            is_self_distance = True
        else:
            is_self_distance = False
        
        # Convert to coordinate arrays
        coords1 = self._extract_coordinates(points1, crs)
        coords2 = self._extract_coordinates(points2, crs)
        
        # Determine method
        effective_crs = crs or self.default_crs
        if method == 'auto':
            if self.crs_handler._is_geographic_crs(effective_crs):
                method = 'haversine'
            else:
                method = 'euclidean'
        
        # Calculate distance matrix
        n1, n2 = len(coords1), len(coords2)
        distances = np.zeros((n1, n2))
        
        if method == 'haversine':
            # Vectorized haversine calculation
            lat1 = coords1[:, 1:2]  # Shape (n1, 1)
            lon1 = coords1[:, 0:1]  # Shape (n1, 1)
            lat2 = coords2[:, 1].reshape(1, -1)  # Shape (1, n2)
            lon2 = coords2[:, 0].reshape(1, -1)  # Shape (1, n2)
            
            distances = self.haversine_distance(lat1, lon1, lat2, lon2)
            
        elif method == 'euclidean':
            # Vectorized euclidean calculation
            x1 = coords1[:, 0:1]  # Shape (n1, 1)
            y1 = coords1[:, 1:2]  # Shape (n1, 1)
            x2 = coords2[:, 0].reshape(1, -1)  # Shape (1, n2)
            y2 = coords2[:, 1].reshape(1, -1)  # Shape (1, n2)
            
            distances = self.euclidean_distance(x1, y1, x2, y2)
            
        elif method == 'manhattan':
            # Vectorized manhattan calculation
            x1 = coords1[:, 0:1]
            y1 = coords1[:, 1:2]
            x2 = coords2[:, 0].reshape(1, -1)
            y2 = coords2[:, 1].reshape(1, -1)
            
            distances = self.manhattan_distance(x1, y1, x2, y2)
        
        # Optimize symmetric case
        if is_self_distance and symmetric:
            # Fill upper triangle from lower triangle
            i_upper = np.triu_indices(n1, k=1)
            distances[i_upper] = distances.T[i_upper]
        
        return distances
    
    def _extract_coordinates(self, 
                           points: List[Union[SpatialPoint, Tuple[float, float]]],
                           crs: Optional[str] = None) -> np.ndarray:
        """Extract coordinate array from points list."""
        coords = []
        for point in points:
            if isinstance(point, tuple):
                coords.append(point)
            elif isinstance(point, SpatialPoint):
                coords.append((point.x, point.y))
            else:
                raise ValueError(f"Unsupported point type: {type(point)}")
        
        return np.array(coords)
    
    def nearest_neighbors(self,
                         query_points: List[Union[SpatialPoint, Tuple[float, float]]],
                         reference_points: List[Union[SpatialPoint, Tuple[float, float]]],
                         k: int = 1,
                         method: str = 'auto',
                         crs: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for each query point.
        
        Parameters
        ----------
        query_points : list
            Points to find neighbors for.
        reference_points : list
            Points to search within.
        k : int, default 1
            Number of nearest neighbors to find.
        method : str, default 'auto'
            Distance calculation method.
        crs : str, optional
            Coordinate reference system.
            
        Returns
        -------
        distances : ndarray
            Distances to k nearest neighbors for each query point.
        indices : ndarray
            Indices of k nearest neighbors for each query point.
        """
        # Calculate distance matrix
        dist_matrix = self.distance_matrix(
            query_points, reference_points, method=method, crs=crs
        )
        
        # Find k nearest neighbors
        if k >= dist_matrix.shape[1]:
            # If k is larger than available points, return all
            k = dist_matrix.shape[1]
        
        # Get indices of k smallest distances for each query point
        indices = np.argpartition(dist_matrix, k-1, axis=1)[:, :k]
        
        # Get corresponding distances
        distances = np.take_along_axis(dist_matrix, indices, axis=1)
        
        # Sort by distance
        sort_indices = np.argsort(distances, axis=1)
        distances = np.take_along_axis(distances, sort_indices, axis=1)
        indices = np.take_along_axis(indices, sort_indices, axis=1)
        
        return distances, indices


class SpatialDistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for spatial distance calculations.
    
    This transformer calculates distances from points to reference locations
    and can be used within sklearn pipelines for spatial feature engineering.
    """
    
    def __init__(self,
                 reference_points: Optional[List[Union[SpatialPoint, Tuple[float, float]]]] = None,
                 method: str = 'auto',
                 crs: Optional[str] = None,
                 coordinate_columns: Optional[List[str]] = None,
                 output_columns: Optional[List[str]] = None,
                 k_nearest: Optional[int] = None):
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
        """
        self.reference_points = reference_points
        self.method = method
        self.crs = crs
        self.coordinate_columns = coordinate_columns or ['x', 'y']
        self.output_columns = output_columns
        self.k_nearest = k_nearest
        
        self.distance_calculator_ = None
        self.fitted_reference_points_ = None
        self.feature_names_out_ = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> 'SpatialDistanceTransformer':
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
        
        self.distance_calculator_ = SpatialDistanceCalculator(self.crs or 'EPSG:4326')
        
        # Determine reference points
        if self.reference_points is not None:
            self.fitted_reference_points_ = self.reference_points
            logger.info(f"Using {len(self.fitted_reference_points_)} provided reference points")
        else:
            # Extract reference points from training data
            if isinstance(X, pd.DataFrame):
                if all(col in X.columns for col in self.coordinate_columns):
                    coords = X[self.coordinate_columns].values
                else:
                    raise ValueError(f"Coordinate columns {self.coordinate_columns} not found")
            else:
                coords = np.asarray(X)
            
            # Use data centroids as reference points
            if coords.shape[1] >= 2:
                centroid = np.mean(coords, axis=0)
                self.fitted_reference_points_ = [(centroid[0], centroid[1])]
                logger.info("Using data centroid as reference point")
            else:
                raise ValueError("Input data must have at least 2 coordinate dimensions")
        
        # Determine output feature names
        n_ref = len(self.fitted_reference_points_)
        if self.k_nearest:
            n_features = min(self.k_nearest, n_ref)
        else:
            n_features = n_ref
        
        if self.output_columns:
            if len(self.output_columns) != n_features:
                raise ValueError(f"Number of output columns ({len(self.output_columns)}) "
                               f"must match number of features ({n_features})")
            self.feature_names_out_ = self.output_columns
        else:
            if self.k_nearest:
                self.feature_names_out_ = [f'distance_to_nearest_{i+1}' for i in range(n_features)]
            else:
                self.feature_names_out_ = [f'distance_to_ref_{i}' for i in range(n_features)]
        
        logger.info(f"Configured {n_features} distance features: {self.feature_names_out_}")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
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
        check_is_fitted(self, 'fitted_reference_points_')
        
        # Extract coordinates
        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
                query_points = [(row[0], row[1]) for row in coords]
            else:
                raise ValueError(f"Coordinate columns {self.coordinate_columns} not found")
        else:
            coords = np.asarray(X)
            query_points = [(row[0], row[1]) for row in coords]
        
        # Calculate distances
        if self.k_nearest:
            # Find k nearest reference points
            distances, indices = self.distance_calculator_.nearest_neighbors(
                query_points, self.fitted_reference_points_, 
                k=self.k_nearest, method=self.method, crs=self.crs
            )
            distance_features = distances
        else:
            # Calculate distances to all reference points
            distance_matrix = self.distance_calculator_.distance_matrix(
                query_points, self.fitted_reference_points_,
                method=self.method, crs=self.crs, symmetric=False
            )
            distance_features = distance_matrix
        
        # Create output
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for i, col_name in enumerate(self.feature_names_out_):
                if distance_features.ndim == 1:
                    X_transformed[col_name] = distance_features
                else:
                    X_transformed[col_name] = distance_features[:, i]
            return X_transformed
        else:
            # For array input, return distance features only
            return distance_features
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names."""
        check_is_fitted(self, 'feature_names_out_')
        return self.feature_names_out_.copy()


@dataclass
class GeometricResult:
    """Container for geometric operation results."""
    operation: str
    result: Any
    geometry_type: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class GeometricOperations:
    """
    Comprehensive geometric operations with Shapely integration.
    
    Provides point-in-polygon testing, buffer analysis, convex hulls,
    bounding boxes, and other spatial geometric computations with
    fallback mechanisms when Shapely is not available.
    """
    
    def __init__(self):
        """Initialize geometric operations handler."""
        self.shapely_available = _dependency_status.is_available(GeospatialLibrary.SHAPELY)
        self._geometry_cache = {}
        
        if self.shapely_available:
            try:
                from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiPolygon
                from shapely.ops import cascaded_union, unary_union
                from shapely import affinity, validation
                import shapely.speedups
                
                # Enable speedups if available
                if shapely.speedups.available:
                    shapely.speedups.enable()
                    
                self.shapely_geom = {
                    'Point': Point,
                    'Polygon': Polygon,
                    'LineString': LineString,
                    'MultiPoint': MultiPoint,
                    'MultiPolygon': MultiPolygon
                }
                self.shapely_ops = {
                    'unary_union': unary_union,
                    'cascaded_union': cascaded_union if hasattr(shapely.ops, 'cascaded_union') else unary_union
                }
                self.shapely_affinity = affinity
                self.shapely_validation = validation
                
                logger.debug("Shapely available for geometric operations")
            except ImportError:
                self.shapely_available = False
                logger.warning("Shapely import failed, using fallback geometric operations")
    
    def create_point(self, x: float, y: float, z: Optional[float] = None) -> Any:
        """
        Create a point geometry.
        
        Parameters
        ----------
        x, y : float
            Point coordinates.
        z : float, optional
            Z coordinate for 3D point.
            
        Returns
        -------
        point : Shapely Point or SpatialPoint
            Point geometry object.
        """
        if self.shapely_available:
            if z is not None:
                return self.shapely_geom['Point'](x, y, z)
            return self.shapely_geom['Point'](x, y)
        else:
            return SpatialPoint(x=x, y=y, z=z)
    
    def create_polygon(self, coordinates: List[Tuple[float, float]]) -> Any:
        """
        Create a polygon geometry.
        
        Parameters
        ----------
        coordinates : list of tuples
            Polygon boundary coordinates.
            
        Returns
        -------
        polygon : Shapely Polygon or dict
            Polygon geometry object.
        """
        if self.shapely_available:
            return self.shapely_geom['Polygon'](coordinates)
        else:
            # Fallback: store as coordinate list with basic properties
            return {
                'type': 'polygon',
                'coordinates': coordinates,
                'bounds': self._calculate_bounds(coordinates)
            }
    
    def point_in_polygon(self, 
                        point: Union[SpatialPoint, Tuple[float, float]], 
                        polygon: Any) -> bool:
        """
        Test if point is inside polygon.
        
        Parameters
        ----------
        point : SpatialPoint or tuple
            Point to test.
        polygon : Shapely Polygon or dict
            Polygon to test against.
            
        Returns
        -------
        inside : bool
            True if point is inside polygon.
        """
        # Convert point to coordinates
        if isinstance(point, tuple):
            px, py = point
        elif isinstance(point, SpatialPoint):
            px, py = point.x, point.y
        else:
            raise ValueError("Point must be tuple or SpatialPoint")
        
        if self.shapely_available:
            # Use Shapely for accurate point-in-polygon test
            if hasattr(polygon, 'contains'):
                test_point = self.create_point(px, py)
                return polygon.contains(test_point)
            elif hasattr(polygon, 'coords'):
                # LineString or similar
                return False
            else:
                raise ValueError("Invalid polygon object for Shapely")
        else:
            # Fallback: ray casting algorithm
            if isinstance(polygon, dict) and polygon.get('type') == 'polygon':
                coords = polygon['coordinates']
                return self._point_in_polygon_raycast(px, py, coords)
            else:
                raise ValueError("Invalid polygon object for fallback method")
    
    def _point_in_polygon_raycast(self, x: float, y: float, 
                                 polygon_coords: List[Tuple[float, float]]) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon_coords)
        inside = False
        
        p1x, p1y = polygon_coords[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_coords[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def buffer_analysis(self, 
                       geometry: Any, 
                       distance: float,
                       resolution: int = 16) -> GeometricResult:
        """
        Create buffer around geometry.
        
        Parameters
        ----------
        geometry : Shapely geometry or SpatialPoint
            Input geometry.
        distance : float
            Buffer distance.
        resolution : int, default 16
            Number of points in circular arcs.
            
        Returns
        -------
        result : GeometricResult
            Buffer geometry and properties.
        """
        if self.shapely_available:
            try:
                if isinstance(geometry, SpatialPoint):
                    geometry = self.create_point(geometry.x, geometry.y)
                
                buffered = geometry.buffer(distance, resolution=resolution)
                
                return GeometricResult(
                    operation='buffer',
                    result=buffered,
                    geometry_type=buffered.geom_type,
                    properties={
                        'original_area': getattr(geometry, 'area', 0),
                        'buffer_area': buffered.area,
                        'buffer_distance': distance,
                        'resolution': resolution
                    }
                )
            except Exception as e:
                return GeometricResult(
                    operation='buffer',
                    result=None,
                    warnings=[f"Shapely buffer failed: {e}"]
                )
        else:
            # Fallback: approximate circular buffer for points
            if isinstance(geometry, SpatialPoint):
                # Create approximate circular polygon
                import math
                points = []
                for i in range(resolution):
                    angle = 2 * math.pi * i / resolution
                    x = geometry.x + distance * math.cos(angle)
                    y = geometry.y + distance * math.sin(angle)
                    points.append((x, y))
                points.append(points[0])  # Close the polygon
                
                buffer_poly = self.create_polygon(points)
                
                return GeometricResult(
                    operation='buffer',
                    result=buffer_poly,
                    geometry_type='polygon',
                    properties={
                        'buffer_distance': distance,
                        'resolution': resolution,
                        'approximated': True
                    },
                    warnings=['Using approximate circular buffer (Shapely not available)']
                )
            else:
                return GeometricResult(
                    operation='buffer',
                    result=None,
                    warnings=['Buffer analysis not available for this geometry type without Shapely']
                )
    
    def convex_hull(self, points: List[Union[SpatialPoint, Tuple[float, float]]]) -> GeometricResult:
        """
        Calculate convex hull of point set.
        
        Parameters
        ----------
        points : list of SpatialPoint or tuples
            Input points.
            
        Returns
        -------
        result : GeometricResult
            Convex hull geometry and properties.
        """
        if not points:
            return GeometricResult(
                operation='convex_hull',
                result=None,
                warnings=['No points provided']
            )
        
        if self.shapely_available:
            try:
                # Convert to Shapely points
                shapely_points = []
                for point in points:
                    if isinstance(point, tuple):
                        shapely_points.append(self.create_point(point[0], point[1]))
                    elif isinstance(point, SpatialPoint):
                        shapely_points.append(self.create_point(point.x, point.y))
                
                multipoint = self.shapely_geom['MultiPoint'](shapely_points)
                hull = multipoint.convex_hull
                
                return GeometricResult(
                    operation='convex_hull',
                    result=hull,
                    geometry_type=hull.geom_type,
                    properties={
                        'input_points': len(points),
                        'hull_area': getattr(hull, 'area', 0),
                        'hull_length': getattr(hull, 'length', 0)
                    }
                )
            except Exception as e:
                # Fall through to fallback method
                logger.warning(f"Shapely convex hull failed: {e}")
        
        # Fallback: Graham scan algorithm
        coords = []
        for point in points:
            if isinstance(point, tuple):
                coords.append(point)
            elif isinstance(point, SpatialPoint):
                coords.append((point.x, point.y))
        
        hull_coords = self._graham_scan(coords)
        hull_polygon = self.create_polygon(hull_coords)
        
        return GeometricResult(
            operation='convex_hull',
            result=hull_polygon,
            geometry_type='polygon',
            properties={
                'input_points': len(points),
                'hull_vertices': len(hull_coords),
                'approximated': not self.shapely_available
            },
            warnings=['Using Graham scan algorithm (Shapely not available)'] if not self.shapely_available else []
        )
    
    def _graham_scan(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Graham scan algorithm for convex hull."""
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        points = sorted(set(points))
        if len(points) <= 3:
            return points + [points[0]] if points else []
        
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        # Remove last point of each half because it's repeated
        hull = lower[:-1] + upper[:-1]
        if hull:
            hull.append(hull[0])  # Close the polygon
        
        return hull
    
    def bounding_box(self, 
                    geometry: Union[Any, List[Union[SpatialPoint, Tuple[float, float]]]]) -> GeometricResult:
        """
        Calculate bounding box (minimum bounding rectangle).
        
        Parameters
        ----------
        geometry : Shapely geometry or list of points
            Input geometry or point set.
            
        Returns
        -------
        result : GeometricResult
            Bounding box geometry and properties.
        """
        if self.shapely_available and hasattr(geometry, 'bounds'):
            # Use Shapely bounds
            bounds = geometry.bounds
            minx, miny, maxx, maxy = bounds
            
            # Create bounding box polygon
            bbox_coords = [
                (minx, miny), (maxx, miny), 
                (maxx, maxy), (minx, maxy), 
                (minx, miny)
            ]
            bbox_polygon = self.create_polygon(bbox_coords)
            
            return GeometricResult(
                operation='bounding_box',
                result=bbox_polygon,
                geometry_type='polygon',
                properties={
                    'bounds': bounds,
                    'width': maxx - minx,
                    'height': maxy - miny,
                    'area': (maxx - minx) * (maxy - miny),
                    'center': ((minx + maxx) / 2, (miny + maxy) / 2)
                }
            )
        else:
            # Handle list of points or fallback
            if isinstance(geometry, list):
                coords = []
                for point in geometry:
                    if isinstance(point, tuple):
                        coords.append(point)
                    elif isinstance(point, SpatialPoint):
                        coords.append((point.x, point.y))
            else:
                return GeometricResult(
                    operation='bounding_box',
                    result=None,
                    warnings=['Cannot calculate bounding box for this geometry type']
                )
            
            if not coords:
                return GeometricResult(
                    operation='bounding_box',
                    result=None,
                    warnings=['No coordinates found']
                )
            
            bounds = self._calculate_bounds(coords)
            minx, miny, maxx, maxy = bounds
            
            bbox_coords = [
                (minx, miny), (maxx, miny), 
                (maxx, maxy), (minx, maxy), 
                (minx, miny)
            ]
            bbox_polygon = self.create_polygon(bbox_coords)
            
            return GeometricResult(
                operation='bounding_box',
                result=bbox_polygon,
                geometry_type='polygon',
                properties={
                    'bounds': bounds,
                    'width': maxx - minx,
                    'height': maxy - miny,
                    'area': (maxx - minx) * (maxy - miny),
                    'center': ((minx + maxx) / 2, (miny + maxy) / 2)
                }
            )
    
    def _calculate_bounds(self, coords: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """Calculate bounding box from coordinate list."""
        if not coords:
            return (0, 0, 0, 0)
        
        xs, ys = zip(*coords)
        return (min(xs), min(ys), max(xs), max(ys))
    
    def intersection(self, geom1: Any, geom2: Any) -> GeometricResult:
        """
        Calculate intersection of two geometries.
        
        Parameters
        ----------
        geom1, geom2 : Shapely geometries
            Input geometries.
            
        Returns
        -------
        result : GeometricResult
            Intersection geometry and properties.
        """
        if self.shapely_available:
            try:
                intersection = geom1.intersection(geom2)
                
                return GeometricResult(
                    operation='intersection',
                    result=intersection,
                    geometry_type=intersection.geom_type,
                    properties={
                        'is_empty': intersection.is_empty,
                        'area': getattr(intersection, 'area', 0),
                        'length': getattr(intersection, 'length', 0)
                    }
                )
            except Exception as e:
                return GeometricResult(
                    operation='intersection',
                    result=None,
                    warnings=[f"Shapely intersection failed: {e}"]
                )
        else:
            return GeometricResult(
                operation='intersection',
                result=None,
                warnings=['Intersection operation requires Shapely library']
            )
    
    def union(self, geometries: List[Any]) -> GeometricResult:
        """
        Calculate union of multiple geometries.
        
        Parameters
        ----------
        geometries : list of Shapely geometries
            Input geometries.
            
        Returns
        -------
        result : GeometricResult
            Union geometry and properties.
        """
        if self.shapely_available:
            try:
                if len(geometries) == 1:
                    union_result = geometries[0]
                else:
                    union_result = self.shapely_ops['unary_union'](geometries)
                
                return GeometricResult(
                    operation='union',
                    result=union_result,
                    geometry_type=union_result.geom_type,
                    properties={
                        'input_count': len(geometries),
                        'area': getattr(union_result, 'area', 0),
                        'length': getattr(union_result, 'length', 0)
                    }
                )
            except Exception as e:
                return GeometricResult(
                    operation='union',
                    result=None,
                    warnings=[f"Shapely union failed: {e}"]
                )
        else:
            return GeometricResult(
                operation='union',
                result=None,
                warnings=['Union operation requires Shapely library']
            )


class SpatialIndexer:
    """
    Spatial indexing for performance optimization.
    
    Provides spatial indexing capabilities using R-tree or grid-based methods
    to accelerate spatial queries on large datasets.
    """
    
    def __init__(self, method: str = 'grid'):
        """
        Initialize spatial indexer.
        
        Parameters
        ----------
        method : str, default 'grid'
            Indexing method ('grid', 'quadtree', 'rtree').
        """
        self.method = method
        self.index = None
        self.geometries = {}
        self.bounds = None
        
        # Try to use rtree if available
        if method == 'rtree':
            try:
                from rtree import index as rtree_index
                self.rtree_index = rtree_index
                self.rtree_available = True
            except ImportError:
                logger.warning("R-tree not available, falling back to grid indexing")
                self.method = 'grid'
                self.rtree_available = False
        else:
            self.rtree_available = False
    
    def build_index(self, geometries: Dict[int, Any]):
        """
        Build spatial index from geometries.
        
        Parameters
        ----------
        geometries : dict
            Dictionary mapping IDs to geometry objects.
        """
        self.geometries = geometries.copy()
        
        if self.method == 'rtree' and self.rtree_available:
            self._build_rtree_index()
        elif self.method == 'grid':
            self._build_grid_index()
        else:
            raise ValueError(f"Unsupported indexing method: {self.method}")
    
    def _build_rtree_index(self):
        """Build R-tree spatial index."""
        self.index = self.rtree_index.Index()
        
        for geom_id, geometry in self.geometries.items():
            if hasattr(geometry, 'bounds'):
                bounds = geometry.bounds
                self.index.insert(geom_id, bounds)
    
    def _build_grid_index(self):
        """Build grid-based spatial index."""
        if not self.geometries:
            return
        
        # Calculate overall bounds
        all_bounds = []
        for geometry in self.geometries.values():
            if hasattr(geometry, 'bounds'):
                all_bounds.append(geometry.bounds)
            elif isinstance(geometry, SpatialPoint):
                all_bounds.append((geometry.x, geometry.y, geometry.x, geometry.y))
        
        if not all_bounds:
            return
        
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        
        self.bounds = (minx, miny, maxx, maxy)
        
        # Create grid
        grid_size = int(np.sqrt(len(self.geometries))) + 1
        self.grid_size = max(grid_size, 10)
        self.cell_width = (maxx - minx) / self.grid_size
        self.cell_height = (maxy - miny) / self.grid_size
        
        # Initialize grid
        self.index = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.index[(i, j)] = []
        
        # Populate grid
        for geom_id, geometry in self.geometries.items():
            bounds = self._get_geometry_bounds(geometry)
            if bounds:
                cells = self._get_intersecting_cells(bounds)
                for cell in cells:
                    if cell in self.index:
                        self.index[cell].append(geom_id)
    
    def _get_geometry_bounds(self, geometry):
        """Get bounds for any geometry type."""
        if hasattr(geometry, 'bounds'):
            return geometry.bounds
        elif isinstance(geometry, SpatialPoint):
            return (geometry.x, geometry.y, geometry.x, geometry.y)
        elif isinstance(geometry, dict) and 'bounds' in geometry:
            return geometry['bounds']
        return None
    
    def _get_intersecting_cells(self, bounds):
        """Get grid cells that intersect with bounds."""
        if not self.bounds:
            return []
        
        minx, miny, maxx, maxy = bounds
        base_minx, base_miny, base_maxx, base_maxy = self.bounds
        
        # Calculate cell ranges
        min_i = max(0, int((minx - base_minx) / self.cell_width))
        max_i = min(self.grid_size - 1, int((maxx - base_minx) / self.cell_width))
        min_j = max(0, int((miny - base_miny) / self.cell_height))
        max_j = min(self.grid_size - 1, int((maxy - base_miny) / self.cell_height))
        
        cells = []
        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                cells.append((i, j))
        
        return cells
    
    def query(self, bounds: Tuple[float, float, float, float]) -> List[int]:
        """
        Query spatial index for intersecting geometries.
        
        Parameters
        ----------
        bounds : tuple
            Query bounds (minx, miny, maxx, maxy).
            
        Returns
        -------
        geometry_ids : list
            IDs of potentially intersecting geometries.
        """
        if self.index is None:
            return list(self.geometries.keys())
        
        if self.method == 'rtree' and self.rtree_available:
            return list(self.index.intersection(bounds))
        elif self.method == 'grid':
            cells = self._get_intersecting_cells(bounds)
            candidates = set()
            for cell in cells:
                if cell in self.index:
                    candidates.update(self.index[cell])
            return list(candidates)
        else:
            return list(self.geometries.keys())


class SpatialGeometryTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for spatial geometric operations.
    
    This transformer performs geometric operations on spatial data
    and can be used within sklearn pipelines for spatial feature engineering.
    """
    
    def __init__(self,
                 operations: List[str] = ['buffer', 'convex_hull'],
                 buffer_distance: float = 1.0,
                 coordinate_columns: Optional[List[str]] = None):
        """
        Initialize spatial geometry transformer.
        
        Parameters
        ----------
        operations : list of str, default ['buffer', 'convex_hull']
            Geometric operations to perform.
        buffer_distance : float, default 1.0
            Distance for buffer operations.
        coordinate_columns : list of str, optional
            Names of coordinate columns.
        """
        self.operations = operations
        self.buffer_distance = buffer_distance
        self.coordinate_columns = coordinate_columns or ['x', 'y']
        
        self.geometric_ops_ = None
        self.feature_names_out_ = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> 'SpatialGeometryTransformer':
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
        self.geometric_ops_ = GeometricOperations()
        
        # Determine output feature names
        self.feature_names_out_ = []
        for op in self.operations:
            if op == 'buffer':
                self.feature_names_out_.extend([f'buffer_{self.buffer_distance}_area'])
            elif op == 'convex_hull':
                self.feature_names_out_.extend(['convex_hull_area'])
            elif op == 'bounding_box':
                self.feature_names_out_.extend(['bbox_area', 'bbox_width', 'bbox_height'])
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data with geometric operations.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Input spatial data.
            
        Returns
        -------
        X_transformed : DataFrame or array-like
            Data with geometric features added.
        """
        check_is_fitted(self, 'geometric_ops_')
        
        # Extract coordinates
        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
                points = [SpatialPoint(x=row[0], y=row[1]) for row in coords]
            else:
                raise ValueError(f"Coordinate columns {self.coordinate_columns} not found")
        else:
            coords = np.asarray(X)
            points = [SpatialPoint(x=row[0], y=row[1]) for row in coords]
        
        # Perform geometric operations
        features = []
        
        for op in self.operations:
            if op == 'buffer':
                # Individual point buffers
                buffer_areas = []
                for point in points:
                    result = self.geometric_ops_.buffer_analysis(point, self.buffer_distance)
                    if result.result and hasattr(result.result, 'area'):
                        buffer_areas.append(result.result.area)
                    elif result.properties.get('approximated'):
                        # Approximate area for circular buffer
                        buffer_areas.append(np.pi * self.buffer_distance ** 2)
                    else:
                        buffer_areas.append(0)
                features.append(buffer_areas)
            
            elif op == 'convex_hull':
                # Convex hull for all points
                hull_result = self.geometric_ops_.convex_hull(points)
                hull_area = hull_result.properties.get('hull_area', 0)
                features.append([hull_area] * len(points))
            
            elif op == 'bounding_box':
                # Bounding box for all points
                bbox_result = self.geometric_ops_.bounding_box(points)
                bbox_props = bbox_result.properties
                bbox_area = bbox_props.get('area', 0)
                bbox_width = bbox_props.get('width', 0)
                bbox_height = bbox_props.get('height', 0)
                
                features.extend([
                    [bbox_area] * len(points),
                    [bbox_width] * len(points),
                    [bbox_height] * len(points)
                ])
        
        # Create output
        feature_array = np.column_stack(features) if features else np.empty((len(points), 0))
        
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for i, col_name in enumerate(self.feature_names_out_):
                if i < feature_array.shape[1]:
                    X_transformed[col_name] = feature_array[:, i]
            return X_transformed
        else:
            return feature_array
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names."""
        check_is_fitted(self, 'feature_names_out_')
        return self.feature_names_out_.copy()


@dataclass
class SpatialStatisticsResult:
    """Container for spatial statistics results."""
    statistic: str
    value: float
    p_value: Optional[float] = None
    z_score: Optional[float] = None
    expected_value: Optional[float] = None
    variance: Optional[float] = None
    interpretation: Optional[str] = None
    significance_level: float = 0.05
    is_significant: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate significance after initialization."""
        if self.p_value is not None:
            self.is_significant = self.p_value < self.significance_level


class SpatialWeightsMatrix:
    """
    Spatial weights matrix for spatial statistics calculations.
    
    Provides various methods for constructing spatial weights matrices
    including distance-based, contiguity-based, and k-nearest neighbor weights.
    """
    
    def __init__(self, method: str = 'distance', **kwargs):
        """
        Initialize spatial weights matrix.
        
        Parameters
        ----------
        method : str, default 'distance'
            Weighting method ('distance', 'knn', 'queen', 'rook').
        **kwargs : additional parameters
            Method-specific parameters.
        """
        self.method = method
        self.params = kwargs
        self.weights = None
        self.n_observations = 0
        
    def build_weights(self, 
                     points: List[Union[SpatialPoint, Tuple[float, float]]],
                     values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build spatial weights matrix.
        
        Parameters
        ----------
        points : list of SpatialPoint or tuples
            Spatial locations.
        values : array-like, optional
            Associated values (for some weight methods).
            
        Returns
        -------
        weights : ndarray
            Spatial weights matrix.
        """
        self.n_observations = len(points)
        
        if self.method == 'distance':
            self.weights = self._build_distance_weights(points)
        elif self.method == 'knn':
            k = self.params.get('k', 8)
            self.weights = self._build_knn_weights(points, k)
        elif self.method == 'inverse_distance':
            power = self.params.get('power', 1.0)
            self.weights = self._build_inverse_distance_weights(points, power)
        else:
            raise ValueError(f"Unsupported weights method: {self.method}")
        
        return self.weights
    
    def _build_distance_weights(self, points: List[Union[SpatialPoint, Tuple[float, float]]]) -> np.ndarray:
        """Build distance-based weights matrix."""
        distance_calc = SpatialDistanceCalculator()
        distance_matrix = distance_calc.distance_matrix(points, points)
        
        # Use threshold distance for connectivity
        threshold = self.params.get('threshold', np.percentile(distance_matrix[distance_matrix > 0], 25))
        
        weights = np.where(distance_matrix <= threshold, 1.0, 0.0)
        np.fill_diagonal(weights, 0)  # No self-weights
        
        # Row-normalize weights
        row_sums = np.sum(weights, axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights = weights / row_sums[:, np.newaxis]
        
        return weights
    
    def _build_knn_weights(self, points: List[Union[SpatialPoint, Tuple[float, float]]], k: int) -> np.ndarray:
        """Build k-nearest neighbor weights matrix."""
        distance_calc = SpatialDistanceCalculator()
        
        weights = np.zeros((len(points), len(points)))
        
        for i in range(len(points)):
            query_point = [points[i]]
            distances, indices = distance_calc.nearest_neighbors(
                query_point, points, k=k+1  # +1 to exclude self
            )
            
            # Skip the first neighbor (self)
            neighbor_indices = indices[0][1:k+1] if len(indices[0]) > 1 else []
            
            for j in neighbor_indices:
                weights[i, j] = 1.0
        
        # Row-normalize weights
        row_sums = np.sum(weights, axis=1)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums[:, np.newaxis]
        
        return weights
    
    def _build_inverse_distance_weights(self, 
                                       points: List[Union[SpatialPoint, Tuple[float, float]]], 
                                       power: float) -> np.ndarray:
        """Build inverse distance weights matrix."""
        distance_calc = SpatialDistanceCalculator()
        distance_matrix = distance_calc.distance_matrix(points, points)
        
        # Avoid division by zero
        distance_matrix[distance_matrix == 0] = np.inf
        
        # Inverse distance weighting
        weights = 1.0 / (distance_matrix ** power)
        weights[np.isinf(weights)] = 0  # Set diagonal to 0
        
        # Row-normalize weights
        row_sums = np.sum(weights, axis=1)
        row_sums[row_sums == 0] = 1
        weights = weights / row_sums[:, np.newaxis]
        
        return weights


class SpatialStatistics:
    """
    Comprehensive spatial statistics analysis tools.
    
    Implements Moran's I, Geary's C, spatial clustering detection,
    and other spatial statistical measures with significance testing.
    """
    
    def __init__(self):
        """Initialize spatial statistics analyzer."""
        self.distance_calculator = SpatialDistanceCalculator()
        
    def morans_i(self, 
                values: np.ndarray,
                points: List[Union[SpatialPoint, Tuple[float, float]]],
                weights_method: str = 'knn',
                **weights_params) -> SpatialStatisticsResult:
        """
        Calculate Moran's I spatial autocorrelation statistic.
        
        Parameters
        ----------
        values : array-like
            Attribute values at each location.
        points : list of SpatialPoint or tuples
            Spatial locations.
        weights_method : str, default 'knn'
            Spatial weights method.
        **weights_params : additional parameters
            Parameters for weights matrix construction.
            
        Returns
        -------
        result : SpatialStatisticsResult
            Moran's I statistic and significance test results.
        """
        values = np.asarray(values)
        n = len(values)
        
        if n != len(points):
            raise ValueError("Number of values must match number of points")
        
        # Build spatial weights matrix
        weights_matrix = SpatialWeightsMatrix(weights_method, **weights_params)
        W = weights_matrix.build_weights(points)
        
        # Calculate Moran's I
        y = values - np.mean(values)  # Center the values
        
        # Sum of all weights
        S0 = np.sum(W)
        if S0 == 0:
            return SpatialStatisticsResult(
                statistic="morans_i",
                value=0.0,
                interpretation="No spatial connectivity detected"
            )
        
        # Moran's I calculation
        numerator = np.sum(np.outer(y, y) * W)
        denominator = np.sum(y * y)
        
        if denominator == 0:
            morans_i = 0.0
        else:
            morans_i = (n / S0) * (numerator / denominator)
        
        # Expected value and variance for significance testing
        expected_i = -1.0 / (n - 1)
        
        # Calculate variance (simplified version)
        S1 = 0.5 * np.sum((W + W.T) ** 2)
        S2 = np.sum(np.sum(W, axis=1) ** 2)
        
        b2 = n * np.sum(y ** 4) / (np.sum(y ** 2) ** 2)
        
        variance_i = ((n - 1) * S1 - 2 * S2 + S0 ** 2) / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2)
        variance_i -= (b2 - 3) * (n * S1 - S0 ** 2) / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2)
        variance_i += expected_i ** 2
        
        # Z-score and p-value
        if variance_i > 0:
            z_score = (morans_i - expected_i) / np.sqrt(variance_i)
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = None
            p_value = None
        
        # Interpretation
        if p_value is not None:
            if p_value < 0.05:
                if morans_i > expected_i:
                    interpretation = "Significant positive spatial autocorrelation (clustering)"
                else:
                    interpretation = "Significant negative spatial autocorrelation (dispersion)"
            else:
                interpretation = "No significant spatial autocorrelation"
        else:
            interpretation = "Unable to determine significance"
        
        return SpatialStatisticsResult(
            statistic="morans_i",
            value=morans_i,
            p_value=p_value,
            z_score=z_score,
            expected_value=expected_i,
            variance=variance_i,
            interpretation=interpretation,
            metadata={
                'n_observations': n,
                'sum_weights': S0,
                'weights_method': weights_method,
                'weights_params': weights_params
            }
        )
    
    def gearys_c(self, 
                values: np.ndarray,
                points: List[Union[SpatialPoint, Tuple[float, float]]],
                weights_method: str = 'knn',
                **weights_params) -> SpatialStatisticsResult:
        """
        Calculate Geary's C spatial autocorrelation statistic.
        
        Parameters
        ----------
        values : array-like
            Attribute values at each location.
        points : list of SpatialPoint or tuples
            Spatial locations.
        weights_method : str, default 'knn'
            Spatial weights method.
        **weights_params : additional parameters
            Parameters for weights matrix construction.
            
        Returns
        -------
        result : SpatialStatisticsResult
            Geary's C statistic and significance test results.
        """
        values = np.asarray(values)
        n = len(values)
        
        if n != len(points):
            raise ValueError("Number of values must match number of points")
        
        # Build spatial weights matrix
        weights_matrix = SpatialWeightsMatrix(weights_method, **weights_params)
        W = weights_matrix.build_weights(points)
        
        # Calculate Geary's C
        S0 = np.sum(W)
        if S0 == 0:
            return SpatialStatisticsResult(
                statistic="gearys_c",
                value=1.0,
                interpretation="No spatial connectivity detected"
            )
        
        # Numerator: sum of squared differences weighted by spatial weights
        numerator = 0.0
        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * (values[i] - values[j]) ** 2
        
        # Denominator: variance term
        mean_val = np.mean(values)
        denominator = 2 * S0 * np.sum((values - mean_val) ** 2)
        
        if denominator == 0:
            gearys_c = 1.0
        else:
            gearys_c = ((n - 1) / S0) * (numerator / denominator)
        
        # Expected value
        expected_c = 1.0
        
        # Variance calculation (simplified)
        S1 = 0.5 * np.sum((W + W.T) ** 2)
        S2 = np.sum(np.sum(W, axis=1) ** 2)
        
        variance_c = ((2 * S1 + S2) * (n - 1) - 4 * S0 ** 2) / (2 * (n + 1) * S0 ** 2)
        
        # Z-score and p-value
        if variance_c > 0:
            z_score = (gearys_c - expected_c) / np.sqrt(variance_c)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = None
            p_value = None
        
        # Interpretation
        if p_value is not None:
            if p_value < 0.05:
                if gearys_c < expected_c:
                    interpretation = "Significant positive spatial autocorrelation (clustering)"
                else:
                    interpretation = "Significant negative spatial autocorrelation (dispersion)"
            else:
                interpretation = "No significant spatial autocorrelation"
        else:
            interpretation = "Unable to determine significance"
        
        return SpatialStatisticsResult(
            statistic="gearys_c",
            value=gearys_c,
            p_value=p_value,
            z_score=z_score,
            expected_value=expected_c,
            variance=variance_c,
            interpretation=interpretation,
            metadata={
                'n_observations': n,
                'sum_weights': S0,
                'weights_method': weights_method,
                'weights_params': weights_params
            }
        )
    
    def local_morans_i(self, 
                      values: np.ndarray,
                      points: List[Union[SpatialPoint, Tuple[float, float]]],
                      weights_method: str = 'knn',
                      **weights_params) -> Dict[str, np.ndarray]:
        """
        Calculate Local Indicators of Spatial Association (LISA) using local Moran's I.
        
        Parameters
        ----------
        values : array-like
            Attribute values at each location.
        points : list of SpatialPoint or tuples
            Spatial locations.
        weights_method : str, default 'knn'
            Spatial weights method.
        **weights_params : additional parameters
            Parameters for weights matrix construction.
            
        Returns
        -------
        lisa_results : dict
            Dictionary containing local Moran's I values, p-values, and cluster types.
        """
        values = np.asarray(values)
        n = len(values)
        
        # Build spatial weights matrix
        weights_matrix = SpatialWeightsMatrix(weights_method, **weights_params)
        W = weights_matrix.build_weights(points)
        
        # Standardize values
        y = (values - np.mean(values)) / np.std(values)
        
        # Calculate local Moran's I for each location
        local_i = np.zeros(n)
        for i in range(n):
            neighbors = W[i, :] > 0
            if np.any(neighbors):
                local_i[i] = y[i] * np.sum(W[i, neighbors] * y[neighbors])
        
        # Determine cluster types
        cluster_types = np.full(n, 'Not significant', dtype='<U20')
        mean_y = np.mean(y)
        
        for i in range(n):
            neighbors = W[i, :] > 0
            if np.any(neighbors):
                neighbor_mean = np.mean(y[neighbors])
                
                if y[i] > mean_y and neighbor_mean > mean_y:
                    cluster_types[i] = 'High-High'
                elif y[i] < mean_y and neighbor_mean < mean_y:
                    cluster_types[i] = 'Low-Low'
                elif y[i] > mean_y and neighbor_mean < mean_y:
                    cluster_types[i] = 'High-Low'
                elif y[i] < mean_y and neighbor_mean > mean_y:
                    cluster_types[i] = 'Low-High'
        
        return {
            'local_morans_i': local_i,
            'cluster_types': cluster_types,
            'standardized_values': y,
            'weights_matrix': W
        }
    
    def spatial_clustering_analysis(self,
                                  values: np.ndarray,
                                  points: List[Union[SpatialPoint, Tuple[float, float]]],
                                  method: str = 'hotspot',
                                  **params) -> Dict[str, Any]:
        """
        Perform spatial clustering analysis (hot spot analysis).
        
        Parameters
        ----------
        values : array-like
            Attribute values at each location.
        points : list of SpatialPoint or tuples
            Spatial locations.
        method : str, default 'hotspot'
            Clustering method ('hotspot', 'getis_ord').
        **params : additional parameters
            Method-specific parameters.
            
        Returns
        -------
        clustering_results : dict
            Clustering analysis results.
        """
        if method == 'hotspot':
            return self._getis_ord_gi(values, points, **params)
        elif method == 'getis_ord':
            return self._getis_ord_gi(values, points, **params)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    
    def _getis_ord_gi(self,
                     values: np.ndarray,
                     points: List[Union[SpatialPoint, Tuple[float, float]]],
                     weights_method: str = 'distance',
                     **weights_params) -> Dict[str, Any]:
        """Calculate Getis-Ord Gi* statistic for hot spot analysis."""
        values = np.asarray(values)
        n = len(values)
        
        # Build spatial weights matrix (include self-weights for Gi*)
        weights_matrix = SpatialWeightsMatrix(weights_method, **weights_params)
        W = weights_matrix.build_weights(points)
        
        # Add self-weights (diagonal = 1 for Gi*)
        np.fill_diagonal(W, 1.0)
        
        # Calculate Gi* for each location
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        gi_star = np.zeros(n)
        z_scores = np.zeros(n)
        p_values = np.zeros(n)
        
        for i in range(n):
            # Sum of weights for location i
            wi_sum = np.sum(W[i, :])
            
            if wi_sum > 0:
                # Gi* calculation
                weighted_sum = np.sum(W[i, :] * values)
                gi_star[i] = weighted_sum
                
                # Expected value and variance
                expected_gi = wi_sum * mean_val
                
                # Variance calculation
                wi_squared_sum = np.sum(W[i, :] ** 2)
                variance_gi = wi_sum * std_val ** 2 - (wi_sum * mean_val) ** 2 / (n - 1)
                variance_gi = variance_gi * (n - 1 - wi_squared_sum) / (n - 2)
                
                # Z-score
                if variance_gi > 0:
                    z_scores[i] = (gi_star[i] - expected_gi) / np.sqrt(variance_gi)
                    p_values[i] = 2 * (1 - stats.norm.cdf(abs(z_scores[i])))
        
        # Classify hot spots and cold spots
        significance_level = params.get('significance_level', 0.05)
        hotspots = (z_scores > 0) & (p_values < significance_level)
        coldspots = (z_scores < 0) & (p_values < significance_level)
        
        cluster_labels = np.full(n, 'Not significant', dtype='<U20')
        cluster_labels[hotspots] = 'Hot spot'
        cluster_labels[coldspots] = 'Cold spot'
        
        return {
            'gi_star': gi_star,
            'z_scores': z_scores,
            'p_values': p_values,
            'cluster_labels': cluster_labels,
            'hotspots': hotspots,
            'coldspots': coldspots,
            'n_hotspots': np.sum(hotspots),
            'n_coldspots': np.sum(coldspots),
            'significance_level': significance_level
        }


class SpatialAutocorrelationTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for spatial autocorrelation analysis.
    
    This transformer calculates spatial autocorrelation statistics and
    can be used within sklearn pipelines for spatial feature engineering.
    """
    
    def __init__(self,
                 statistics: List[str] = ['morans_i', 'gearys_c'],
                 weights_method: str = 'knn',
                 coordinate_columns: Optional[List[str]] = None,
                 value_column: str = 'value',
                 **weights_params):
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
        self.coordinate_columns = coordinate_columns or ['x', 'y']
        self.value_column = value_column
        self.weights_params = weights_params
        
        self.spatial_stats_ = None
        self.feature_names_out_ = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> 'SpatialAutocorrelationTransformer':
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
        
        # Determine output feature names
        self.feature_names_out_ = []
        for stat in self.statistics:
            if stat == 'morans_i':
                self.feature_names_out_.extend([
                    'morans_i', 'morans_i_pvalue', 'morans_i_zscore'
                ])
            elif stat == 'gearys_c':
                self.feature_names_out_.extend([
                    'gearys_c', 'gearys_c_pvalue', 'gearys_c_zscore'
                ])
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
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
        check_is_fitted(self, 'spatial_stats_')
        
        # Extract coordinates and values
        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
                points = [(row[0], row[1]) for row in coords]
            else:
                raise ValueError(f"Coordinate columns {self.coordinate_columns} not found")
            
            if self.value_column in X.columns:
                values = X[self.value_column].values
            else:
                raise ValueError(f"Value column {self.value_column} not found")
        else:
            # Assume array format: [x, y, value, ...]
            coords = X[:, :2]
            points = [(row[0], row[1]) for row in coords]
            values = X[:, 2] if X.shape[1] > 2 else np.ones(len(points))
        
        # Calculate spatial statistics
        features = []
        
        for stat in self.statistics:
            if stat == 'morans_i':
                result = self.spatial_stats_.morans_i(
                    values, points, self.weights_method, **self.weights_params
                )
                features.extend([
                    result.value,
                    result.p_value or np.nan,
                    result.z_score or np.nan
                ])
            
            elif stat == 'gearys_c':
                result = self.spatial_stats_.gearys_c(
                    values, points, self.weights_method, **self.weights_params
                )
                features.extend([
                    result.value,
                    result.p_value or np.nan,
                    result.z_score or np.nan
                ])
        
        # Create output - these are global statistics, so replicate for all rows
        n_rows = len(points)
        feature_matrix = np.array([features] * n_rows)
        
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for i, col_name in enumerate(self.feature_names_out_):
                X_transformed[col_name] = feature_matrix[:, i]
            return X_transformed
        else:
            return np.column_stack([X, feature_matrix])
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names."""
        check_is_fitted(self, 'feature_names_out_')
        return self.feature_names_out_.copy()


@dataclass
class InterpolationResult:
    """Container for spatial interpolation results."""
    method: str
    interpolated_values: np.ndarray
    prediction_variance: Optional[np.ndarray] = None
    cross_validation_score: Optional[float] = None
    variogram_model: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class VariogramModel:
    """
    Variogram modeling for kriging interpolation.
    
    Provides variogram calculation, fitting, and model selection
    for spatial interpolation with fallback methods when
    scikit-gstat is not available.
    """
    
    # Available variogram models
    VARIOGRAM_MODELS = {
        'spherical': lambda h, nugget, sill, range_param: nugget + (sill - nugget) * (
            1.5 * h / range_param - 0.5 * (h / range_param) ** 3
        ) if h <= range_param else sill,
        'exponential': lambda h, nugget, sill, range_param: nugget + (sill - nugget) * (
            1 - np.exp(-3 * h / range_param)
        ),
        'gaussian': lambda h, nugget, sill, range_param: nugget + (sill - nugget) * (
            1 - np.exp(-3 * (h / range_param) ** 2)
        ),
        'linear': lambda h, nugget, sill, range_param: nugget + (sill - nugget) * h / range_param if h <= range_param else sill
    }
    
    def __init__(self, model_type: str = 'spherical'):
        """
        Initialize variogram model.
        
        Parameters
        ----------
        model_type : str, default 'spherical'
            Type of variogram model ('spherical', 'exponential', 'gaussian', 'linear').
        """
        self.model_type = model_type
        self.parameters = None
        self.empirical_variogram = None
        self.scikit_gstat_available = _dependency_status.is_available(GeospatialLibrary.SCIKIT_GSTAT)
        
        if self.scikit_gstat_available:
            try:
                import skgstat as skg
                self.skg = skg
                logger.debug("scikit-gstat available for advanced variogram modeling")
            except ImportError:
                self.scikit_gstat_available = False
                logger.warning("scikit-gstat import failed, using fallback variogram methods")
    
    def fit(self, 
           points: List[Union[SpatialPoint, Tuple[float, float]]],
           values: np.ndarray,
           max_distance: Optional[float] = None,
           n_lags: int = 15) -> Dict[str, Any]:
        """
        Fit variogram model to data.
        
        Parameters
        ----------
        points : list of SpatialPoint or tuples
            Spatial locations.
        values : array-like
            Values at each location.
        max_distance : float, optional
            Maximum distance for variogram calculation.
        n_lags : int, default 15
            Number of distance lags for variogram.
            
        Returns
        -------
        variogram_info : dict
            Variogram model parameters and fit information.
        """
        if self.scikit_gstat_available:
            return self._fit_scikit_gstat(points, values, max_distance, n_lags)
        else:
            return self._fit_fallback(points, values, max_distance, n_lags)
    
    def _fit_scikit_gstat(self, points, values, max_distance, n_lags):
        """Fit variogram using scikit-gstat."""
        try:
            # Convert points to coordinate array
            coords = []
            for point in points:
                if isinstance(point, tuple):
                    coords.append(point)
                elif isinstance(point, SpatialPoint):
                    coords.append((point.x, point.y))
            
            coords = np.array(coords)
            values = np.asarray(values)
            
            # Create variogram
            vario = self.skg.Variogram(
                coordinates=coords,
                values=values,
                model=self.model_type,
                maxlag=max_distance,
                n_lags=n_lags,
                normalize=False
            )
            
            # Fit the model
            self.parameters = {
                'nugget': vario.parameters[0],
                'sill': vario.parameters[1],
                'range': vario.parameters[2],
                'model_type': self.model_type
            }
            
            self.empirical_variogram = {
                'lags': vario.bins,
                'semivariance': vario.experimental,
                'counts': vario.bin_count
            }
            
            return {
                'parameters': self.parameters,
                'empirical_variogram': self.empirical_variogram,
                'model_rmse': vario.rmse,
                'r_squared': getattr(vario, 'r_squared', None),
                'method': 'scikit_gstat'
            }
            
        except Exception as e:
            logger.warning(f"scikit-gstat variogram fitting failed: {e}")
            return self._fit_fallback(points, values, max_distance, n_lags)
    
    def _fit_fallback(self, points, values, max_distance, n_lags):
        """Fallback variogram fitting without scikit-gstat."""
        # Convert points and calculate distances
        distance_calc = SpatialDistanceCalculator()
        coords = []
        for point in points:
            if isinstance(point, tuple):
                coords.append(point)
            elif isinstance(point, SpatialPoint):
                coords.append((point.x, point.y))
        
        values = np.asarray(values)
        
        # Calculate distance matrix
        distance_matrix = distance_calc.distance_matrix(coords, coords)
        
        # Determine max distance if not provided
        if max_distance is None:
            max_distance = np.max(distance_matrix) / 2
        
        # Calculate empirical variogram
        lag_width = max_distance / n_lags
        lags = np.arange(lag_width, max_distance + lag_width, lag_width)
        semivariances = []
        counts = []
        
        for i, lag in enumerate(lags):
            lag_min = i * lag_width
            lag_max = (i + 1) * lag_width
            
            # Find pairs within this lag distance
            mask = (distance_matrix >= lag_min) & (distance_matrix < lag_max)
            i_indices, j_indices = np.where(mask)
            
            if len(i_indices) > 0:
                # Calculate semivariance for this lag
                squared_diffs = (values[i_indices] - values[j_indices]) ** 2
                semivariance = np.mean(squared_diffs) / 2
                semivariances.append(semivariance)
                counts.append(len(i_indices))
            else:
                semivariances.append(0)
                counts.append(0)
        
        self.empirical_variogram = {
            'lags': lags,
            'semivariance': np.array(semivariances),
            'counts': np.array(counts)
        }
        
        # Simple parameter estimation
        valid_idx = np.array(counts) > 0
        if np.any(valid_idx):
            nugget = min(semivariances) if semivariances else 0
            sill = max(semivariances) if semivariances else np.var(values)
            # Estimate range as distance where semivariance reaches ~95% of sill
            range_est = max_distance / 3  # Simple heuristic
            
            self.parameters = {
                'nugget': nugget,
                'sill': sill,
                'range': range_est,
                'model_type': self.model_type
            }
        else:
            # Default parameters if no valid lags
            self.parameters = {
                'nugget': 0,
                'sill': np.var(values),
                'range': max_distance / 3,
                'model_type': self.model_type
            }
        
        return {
            'parameters': self.parameters,
            'empirical_variogram': self.empirical_variogram,
            'model_rmse': None,
            'r_squared': None,
            'method': 'fallback'
        }
    
    def predict_semivariance(self, distances: np.ndarray) -> np.ndarray:
        """
        Predict semivariance at given distances using fitted model.
        
        Parameters
        ----------
        distances : array-like
            Distances at which to predict semivariance.
            
        Returns
        -------
        semivariances : ndarray
            Predicted semivariances.
        """
        if self.parameters is None:
            raise ValueError("Variogram model must be fitted before prediction")
        
        distances = np.asarray(distances)
        model_func = self.VARIOGRAM_MODELS[self.model_type]
        
        # Vectorize the model function
        vectorized_model = np.vectorize(
            lambda h: model_func(h, 
                                self.parameters['nugget'],
                                self.parameters['sill'],
                                self.parameters['range'])
        )
        
        return vectorized_model(distances)


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
        
    def ordinary_kriging(self,
                        train_points: List[Union[SpatialPoint, Tuple[float, float]]],
                        train_values: np.ndarray,
                        predict_points: List[Union[SpatialPoint, Tuple[float, float]]],
                        variogram_model: str = 'spherical',
                        **variogram_params) -> InterpolationResult:
        """
        Perform ordinary kriging interpolation.
        
        Parameters
        ----------
        train_points : list of SpatialPoint or tuples
            Known data locations.
        train_values : array-like
            Known values at training locations.
        predict_points : list of SpatialPoint or tuples
            Locations where values should be predicted.
        variogram_model : str, default 'spherical'
            Variogram model type.
        **variogram_params : additional parameters
            Parameters for variogram fitting.
            
        Returns
        -------
        result : InterpolationResult
            Interpolation results with predictions and variance.
        """
        train_values = np.asarray(train_values)
        warnings = []
        
        try:
            # Fit variogram model
            vario_model = VariogramModel(variogram_model)
            vario_info = vario_model.fit(train_points, train_values, **variogram_params)
            self.last_variogram_model = vario_model
            
            # Perform kriging
            n_train = len(train_points)
            n_predict = len(predict_points)
            
            # Calculate distance matrices
            train_coords = self._extract_coordinates(train_points)
            predict_coords = self._extract_coordinates(predict_points)
            
            # Distance matrix between training points
            train_distances = self.distance_calculator.distance_matrix(train_points, train_points)
            
            # Distance matrix from prediction to training points
            pred_to_train_distances = self.distance_calculator.distance_matrix(
                predict_points, train_points, symmetric=False
            )
            
            # Build kriging system
            # Covariance matrix (training points)
            C = self._distances_to_covariances(train_distances, vario_model)
            
            # Add Lagrange multiplier row/column
            C_extended = np.zeros((n_train + 1, n_train + 1))
            C_extended[:n_train, :n_train] = C
            C_extended[n_train, :n_train] = 1
            C_extended[:n_train, n_train] = 1
            C_extended[n_train, n_train] = 0
            
            # Solve kriging system for each prediction point
            predictions = np.zeros(n_predict)
            variances = np.zeros(n_predict)
            
            for i in range(n_predict):
                # Covariance vector (prediction to training points)
                c = self._distances_to_covariances(pred_to_train_distances[i], vario_model)
                
                # Extended right-hand side
                rhs = np.zeros(n_train + 1)
                rhs[:n_train] = c
                rhs[n_train] = 1
                
                # Solve system
                try:
                    weights = np.linalg.solve(C_extended, rhs)
                    
                    # Prediction
                    predictions[i] = np.dot(weights[:n_train], train_values)
                    
                    # Prediction variance
                    variances[i] = vario_model.parameters['sill'] - np.dot(weights[:n_train], c) - weights[n_train]
                    
                except np.linalg.LinAlgError:
                    warnings.append(f"Singular kriging system at prediction point {i}")
                    # Fallback to inverse distance weighting
                    distances = pred_to_train_distances[i]
                    weights = 1 / (distances + 1e-10)  # Avoid division by zero
                    weights /= np.sum(weights)
                    predictions[i] = np.dot(weights, train_values)
                    variances[i] = np.nan
            
            return InterpolationResult(
                method='ordinary_kriging',
                interpolated_values=predictions,
                prediction_variance=variances,
                variogram_model=vario_info,
                metadata={
                    'n_training_points': n_train,
                    'n_prediction_points': n_predict,
                    'variogram_method': vario_info.get('method', 'unknown')
                },
                warnings=warnings
            )
            
        except Exception as e:
            warnings.append(f"Kriging failed: {e}")
            logger.warning(f"Ordinary kriging failed, falling back to IDW: {e}")
            return self.inverse_distance_weighting(train_points, train_values, predict_points)
    
    def inverse_distance_weighting(self,
                                  train_points: List[Union[SpatialPoint, Tuple[float, float]]],
                                  train_values: np.ndarray,
                                  predict_points: List[Union[SpatialPoint, Tuple[float, float]]],
                                  power: float = 2.0,
                                  max_distance: Optional[float] = None) -> InterpolationResult:
        """
        Perform inverse distance weighting interpolation.
        
        Parameters
        ----------
        train_points : list of SpatialPoint or tuples
            Known data locations.
        train_values : array-like
            Known values at training locations.
        predict_points : list of SpatialPoint or tuples
            Locations where values should be predicted.
        power : float, default 2.0
            Power parameter for distance weighting.
        max_distance : float, optional
            Maximum distance for considering points.
            
        Returns
        -------
        result : InterpolationResult
            Interpolation results.
        """
        train_values = np.asarray(train_values)
        
        # Calculate distances from prediction to training points
        distance_matrix = self.distance_calculator.distance_matrix(
            predict_points, train_points, symmetric=False
        )
        
        # Apply maximum distance constraint
        if max_distance is not None:
            distance_matrix = np.where(distance_matrix > max_distance, np.inf, distance_matrix)
        
        # Calculate weights (inverse distance)
        # Add small epsilon to avoid division by zero
        weights = 1.0 / (distance_matrix + 1e-10) ** power
        
        # Handle points that are exactly at training locations
        exact_matches = distance_matrix < 1e-10
        
        predictions = np.zeros(len(predict_points))
        
        for i in range(len(predict_points)):
            if np.any(exact_matches[i]):
                # Use exact value if prediction point matches training point
                exact_idx = np.where(exact_matches[i])[0][0]
                predictions[i] = train_values[exact_idx]
            else:
                # Normalize weights and calculate weighted average
                valid_weights = weights[i][np.isfinite(weights[i])]
                valid_values = train_values[np.isfinite(weights[i])]
                
                if len(valid_weights) > 0:
                    weight_sum = np.sum(valid_weights)
                    if weight_sum > 0:
                        predictions[i] = np.sum(valid_weights * valid_values) / weight_sum
                    else:
                        predictions[i] = np.mean(train_values)
                else:
                    predictions[i] = np.mean(train_values)
        
        return InterpolationResult(
            method='inverse_distance_weighting',
            interpolated_values=predictions,
            metadata={
                'power': power,
                'max_distance': max_distance,
                'n_training_points': len(train_points),
                'n_prediction_points': len(predict_points)
            }
        )
    
    def cross_validate_kriging(self,
                              points: List[Union[SpatialPoint, Tuple[float, float]]],
                              values: np.ndarray,
                              n_folds: int = 5,
                              variogram_model: str = 'spherical') -> Dict[str, float]:
        """
        Perform cross-validation for kriging parameters.
        
        Parameters
        ----------
        points : list of SpatialPoint or tuples
            Data locations.
        values : array-like
            Values at each location.
        n_folds : int, default 5
            Number of cross-validation folds.
        variogram_model : str, default 'spherical'
            Variogram model type.
            
        Returns
        -------
        cv_results : dict
            Cross-validation metrics.
        """
        values = np.asarray(values)
        n_points = len(points)
        
        # Create cross-validation folds
        fold_size = n_points // n_folds
        indices = np.arange(n_points)
        np.random.shuffle(indices)
        
        predictions = np.zeros(n_points)
        fold_scores = []
        
        for fold in range(n_folds):
            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_points
            
            test_indices = indices[start_idx:end_idx]
            train_indices = np.setdiff1d(indices, test_indices)
            
            train_points = [points[i] for i in train_indices]
            train_values = values[train_indices]
            test_points = [points[i] for i in test_indices]
            test_values = values[test_indices]
            
            # Perform kriging
            result = self.ordinary_kriging(train_points, train_values, test_points, variogram_model)
            
            # Store predictions
            predictions[test_indices] = result.interpolated_values
            
            # Calculate fold score (RMSE)
            fold_rmse = np.sqrt(np.mean((result.interpolated_values - test_values) ** 2))
            fold_scores.append(fold_rmse)
        
        # Calculate overall metrics
        rmse = np.sqrt(np.mean((predictions - values) ** 2))
        mae = np.mean(np.abs(predictions - values))
        r2 = 1 - np.sum((values - predictions) ** 2) / np.sum((values - np.mean(values)) ** 2)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'fold_rmse_mean': np.mean(fold_scores),
            'fold_rmse_std': np.std(fold_scores),
            'n_folds': n_folds
        }
    
    def _extract_coordinates(self, points: List[Union[SpatialPoint, Tuple[float, float]]]) -> np.ndarray:
        """Extract coordinate array from points list."""
        coords = []
        for point in points:
            if isinstance(point, tuple):
                coords.append(point)
            elif isinstance(point, SpatialPoint):
                coords.append((point.x, point.y))
        return np.array(coords)
    
    def _distances_to_covariances(self, distances: np.ndarray, variogram_model: VariogramModel) -> np.ndarray:
        """Convert distances to covariances using variogram model."""
        # Covariance = Sill - Semivariance
        semivariances = variogram_model.predict_semivariance(distances)
        covariances = variogram_model.parameters['sill'] - semivariances
        return covariances


class SpatialInterpolationTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for spatial interpolation.
    
    This transformer performs spatial interpolation on test data based on
    training data and can be used within sklearn pipelines.
    """
    
    def __init__(self,
                 method: str = 'kriging',
                 coordinate_columns: Optional[List[str]] = None,
                 value_column: str = 'value',
                 **method_params):
        """
        Initialize spatial interpolation transformer.
        
        Parameters
        ----------
        method : str, default 'kriging'
            Interpolation method ('kriging', 'idw').
        coordinate_columns : list of str, optional
            Names of coordinate columns.
        value_column : str, default 'value'
            Name of value column.
        **method_params : additional parameters
            Method-specific parameters.
        """
        self.method = method
        self.coordinate_columns = coordinate_columns or ['x', 'y']
        self.value_column = value_column
        self.method_params = method_params
        
        self.interpolator_ = None
        self.train_points_ = None
        self.train_values_ = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> 'SpatialInterpolationTransformer':
        """
        Fit the interpolation model.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Training spatial data.
        y : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.interpolator_ = SpatialInterpolator()
        
        # Extract training data
        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
                self.train_points_ = [(row[0], row[1]) for row in coords]
            else:
                raise ValueError(f"Coordinate columns {self.coordinate_columns} not found")
            
            if self.value_column in X.columns:
                self.train_values_ = X[self.value_column].values
            else:
                raise ValueError(f"Value column {self.value_column} not found")
        else:
            # Assume array format: [x, y, value, ...]
            coords = X[:, :2]
            self.train_points_ = [(row[0], row[1]) for row in coords]
            self.train_values_ = X[:, 2] if X.shape[1] > 2 else np.ones(len(self.train_points_))
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Interpolate values at new locations.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Test locations for interpolation.
            
        Returns
        -------
        X_transformed : DataFrame or array-like
            Data with interpolated values added.
        """
        check_is_fitted(self, 'train_points_')
        
        # Extract prediction points
        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in self.coordinate_columns):
                coords = X[self.coordinate_columns].values
                predict_points = [(row[0], row[1]) for row in coords]
            else:
                raise ValueError(f"Coordinate columns {self.coordinate_columns} not found")
        else:
            coords = X[:, :2]
            predict_points = [(row[0], row[1]) for row in coords]
        
        # Perform interpolation
        if self.method == 'kriging':
            result = self.interpolator_.ordinary_kriging(
                self.train_points_, self.train_values_, predict_points, **self.method_params
            )
        elif self.method == 'idw':
            result = self.interpolator_.inverse_distance_weighting(
                self.train_points_, self.train_values_, predict_points, **self.method_params
            )
        else:
            raise ValueError(f"Unsupported interpolation method: {self.method}")
        
        # Create output
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            X_transformed[f'{self.value_column}_interpolated'] = result.interpolated_values
            if result.prediction_variance is not None:
                X_transformed[f'{self.value_column}_variance'] = result.prediction_variance
            return X_transformed
        else:
            if result.prediction_variance is not None:
                return np.column_stack([X, result.interpolated_values, result.prediction_variance])
            else:
                return np.column_stack([X, result.interpolated_values])


# =============================================================================
# Spatial Joins and Overlay Operations
# =============================================================================

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


@dataclass
class SpatialJoinResult:
    """Results from a spatial join operation."""
    joined_data: pd.DataFrame
    left_geometry_column: str
    right_geometry_column: str
    join_type: SpatialJoinType
    match_counts: Dict[str, int]
    execution_time: float
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the join result."""
        return {
            'total_rows': len(self.joined_data),
            'join_type': self.join_type.value,
            'match_counts': self.match_counts,
            'execution_time_seconds': self.execution_time,
            'columns': list(self.joined_data.columns)
        }


@dataclass 
class OverlayResult:
    """Results from a spatial overlay operation."""
    result_geometries: List[Any]
    result_data: pd.DataFrame
    operation: OverlayOperation
    input_counts: Dict[str, int]
    output_count: int
    execution_time: float
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the overlay result."""
        return {
            'operation': self.operation.value,
            'input_counts': self.input_counts,
            'output_count': self.output_count,
            'execution_time_seconds': self.execution_time,
            'columns': list(self.result_data.columns) if self.result_data is not None else []
        }


class SpatialJoinEngine:
    """
    Core engine for spatial join operations with support for multiple backends.
    """
    
    def __init__(self):
        """Initialize the spatial join engine."""
        self.dependency_checker = GeospatialDependencyChecker()
        
    def _prepare_geometries(self, data: pd.DataFrame, geometry_column: str) -> List[Any]:
        """Prepare geometries for spatial operations."""
        if geometry_column not in data.columns:
            raise ValueError(f"Geometry column '{geometry_column}' not found in data")
            
        geometries = []
        for idx, row in data.iterrows():
            geom = row[geometry_column]
            if isinstance(geom, (tuple, list)) and len(geom) == 2:
                # Convert point coordinates to geometry object
                if _dependency_status.is_available(GeospatialLibrary.SHAPELY):
                    from shapely.geometry import Point
                    geometries.append(Point(geom[0], geom[1]))
                else:
                    geometries.append(geom)
            else:
                geometries.append(geom)
                
        return geometries
        
    def _spatial_predicate_check(self, geom1: Any, geom2: Any, join_type: SpatialJoinType) -> bool:
        """Check spatial relationship between two geometries."""
        if not _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            # Fallback to basic point operations
            if join_type == SpatialJoinType.INTERSECTS:
                if isinstance(geom1, (tuple, list)) and isinstance(geom2, (tuple, list)):
                    return abs(geom1[0] - geom2[0]) < 1e-6 and abs(geom1[1] - geom2[1]) < 1e-6
            return False
            
        # Use Shapely for full geometric operations
        try:
            if join_type == SpatialJoinType.INTERSECTS:
                return geom1.intersects(geom2)
            elif join_type == SpatialJoinType.WITHIN:
                return geom1.within(geom2)
            elif join_type == SpatialJoinType.CONTAINS:
                return geom1.contains(geom2)
            elif join_type == SpatialJoinType.TOUCHES:
                return geom1.touches(geom2)
            elif join_type == SpatialJoinType.OVERLAPS:
                return geom1.overlaps(geom2)
            elif join_type == SpatialJoinType.CROSSES:
                return geom1.crosses(geom2)
            elif join_type == SpatialJoinType.DISJOINT:
                return geom1.disjoint(geom2)
            else:
                return False
        except Exception:
            return False
            
    def _find_nearest_geometry(self, target_geom: Any, candidate_geometries: List[Any]) -> int:
        """Find the index of the nearest geometry."""
        if not _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            # Basic distance for point geometries
            if isinstance(target_geom, (tuple, list)):
                min_dist = float('inf')
                min_idx = -1
                for i, candidate in enumerate(candidate_geometries):
                    if isinstance(candidate, (tuple, list)):
                        dist = np.sqrt((target_geom[0] - candidate[0])**2 + (target_geom[1] - candidate[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = i
                return min_idx
            return -1
            
        # Use Shapely for accurate distance calculations
        min_dist = float('inf')
        min_idx = -1
        
        for i, candidate in enumerate(candidate_geometries):
            try:
                dist = target_geom.distance(candidate)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            except Exception:
                continue
                
        return min_idx
        
    def spatial_join(self, 
                    left_df: pd.DataFrame, 
                    right_df: pd.DataFrame,
                    left_geometry_col: str = 'geometry',
                    right_geometry_col: str = 'geometry',
                    join_type: SpatialJoinType = SpatialJoinType.INTERSECTS,
                    how: str = 'inner') -> SpatialJoinResult:
        """
        Perform spatial join between two datasets.
        
        Parameters
        ----------
        left_df : DataFrame
            Left dataset with geometry column.
        right_df : DataFrame
            Right dataset with geometry column.
        left_geometry_col : str
            Name of geometry column in left dataset.
        right_geometry_col : str
            Name of geometry column in right dataset.
        join_type : SpatialJoinType
            Type of spatial relationship to test.
        how : str
            Type of join ('inner', 'left', 'right').
            
        Returns
        -------
        SpatialJoinResult
            Results of the spatial join operation.
        """
        start_time = time.time()
        
        # Prepare geometries
        left_geometries = self._prepare_geometries(left_df, left_geometry_col)
        right_geometries = self._prepare_geometries(right_df, right_geometry_col)
        
        # Initialize result storage
        joined_rows = []
        match_counts = {'matches': 0, 'no_matches': 0}
        
        # Perform spatial join
        for left_idx, left_geom in enumerate(left_geometries):
            matches_found = []
            
            if join_type == SpatialJoinType.NEAREST:
                # Find single nearest neighbor
                nearest_idx = self._find_nearest_geometry(left_geom, right_geometries)
                if nearest_idx >= 0:
                    matches_found = [nearest_idx]
            else:
                # Find all geometries that satisfy the spatial predicate
                for right_idx, right_geom in enumerate(right_geometries):
                    if self._spatial_predicate_check(left_geom, right_geom, join_type):
                        matches_found.append(right_idx)
                        
            # Create joined records
            if matches_found:
                match_counts['matches'] += len(matches_found)
                for right_idx in matches_found:
                    joined_row = {}
                    # Add left data with suffix
                    for col in left_df.columns:
                        if col != left_geometry_col:
                            joined_row[f"{col}_left"] = left_df.iloc[left_idx][col]
                        else:
                            joined_row[left_geometry_col] = left_df.iloc[left_idx][col]
                            
                    # Add right data with suffix
                    for col in right_df.columns:
                        if col != right_geometry_col:
                            joined_row[f"{col}_right"] = right_df.iloc[right_idx][col]
                        elif col != left_geometry_col:  # Avoid duplicate geometry columns
                            joined_row[f"{right_geometry_col}_right"] = right_df.iloc[right_idx][col]
                            
                    joined_rows.append(joined_row)
            else:
                match_counts['no_matches'] += 1
                if how == 'left':
                    # Include left record with null right values
                    joined_row = {}
                    for col in left_df.columns:
                        if col != left_geometry_col:
                            joined_row[f"{col}_left"] = left_df.iloc[left_idx][col]
                        else:
                            joined_row[left_geometry_col] = left_df.iloc[left_idx][col]
                            
                    for col in right_df.columns:
                        if col != right_geometry_col:
                            joined_row[f"{col}_right"] = None
                        elif col != left_geometry_col:
                            joined_row[f"{right_geometry_col}_right"] = None
                            
                    joined_rows.append(joined_row)
                    
        # Handle right join case
        if how == 'right':
            # Find unmatched right geometries
            matched_right_indices = set()
            for left_idx, left_geom in enumerate(left_geometries):
                for right_idx, right_geom in enumerate(right_geometries):
                    if self._spatial_predicate_check(left_geom, right_geom, join_type):
                        matched_right_indices.add(right_idx)
                        
            # Add unmatched right records
            for right_idx in range(len(right_geometries)):
                if right_idx not in matched_right_indices:
                    joined_row = {}
                    for col in left_df.columns:
                        if col != left_geometry_col:
                            joined_row[f"{col}_left"] = None
                        else:
                            joined_row[left_geometry_col] = None
                            
                    for col in right_df.columns:
                        if col != right_geometry_col:
                            joined_row[f"{col}_right"] = right_df.iloc[right_idx][col]
                        else:
                            joined_row[right_geometry_col] = right_df.iloc[right_idx][col]
                            
                    joined_rows.append(joined_row)
        
        # Create result DataFrame
        joined_df = pd.DataFrame(joined_rows) if joined_rows else pd.DataFrame()
        
        execution_time = time.time() - start_time
        
        return SpatialJoinResult(
            joined_data=joined_df,
            left_geometry_column=left_geometry_col,
            right_geometry_column=right_geometry_col,
            join_type=join_type,
            match_counts=match_counts,
            execution_time=execution_time
        )


class SpatialOverlayEngine:
    """
    Core engine for spatial overlay operations.
    """
    
    def __init__(self):
        """Initialize the spatial overlay engine."""
        self.dependency_checker = GeospatialDependencyChecker()
        
    def _perform_geometric_operation(self, geom1: Any, geom2: Any, operation: OverlayOperation) -> Any:
        """Perform geometric operation between two geometries."""
        if not _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            logger.warning(f"Shapely not available, overlay operation {operation.value} not supported")
            return None
            
        try:
            if operation == OverlayOperation.INTERSECTION:
                return geom1.intersection(geom2)
            elif operation == OverlayOperation.UNION:
                return geom1.union(geom2)
            elif operation == OverlayOperation.DIFFERENCE:
                return geom1.difference(geom2)
            elif operation == OverlayOperation.SYMMETRIC_DIFFERENCE:
                return geom1.symmetric_difference(geom2)
            else:
                return None
        except Exception as e:
            logger.warning(f"Geometric operation failed: {e}")
            return None
            
    def spatial_overlay(self,
                       left_df: pd.DataFrame,
                       right_df: pd.DataFrame,
                       operation: OverlayOperation,
                       left_geometry_col: str = 'geometry',
                       right_geometry_col: str = 'geometry',
                       keep_geom_type: bool = True) -> OverlayResult:
        """
        Perform spatial overlay operation between two datasets.
        
        Parameters
        ----------
        left_df : DataFrame
            Left dataset with geometry column.
        right_df : DataFrame
            Right dataset with geometry column.
        operation : OverlayOperation
            Type of overlay operation to perform.
        left_geometry_col : str
            Name of geometry column in left dataset.
        right_geometry_col : str
            Name of geometry column in right dataset.
        keep_geom_type : bool
            Whether to filter results to maintain geometry type.
            
        Returns
        -------
        OverlayResult
            Results of the spatial overlay operation.
        """
        start_time = time.time()
        
        if not _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            raise ValueError("Shapely is required for spatial overlay operations")
            
        # Prepare input data
        left_geometries = self._prepare_geometries(left_df, left_geometry_col)
        right_geometries = self._prepare_geometries(right_df, right_geometry_col)
        
        result_geometries = []
        result_rows = []
        
        input_counts = {
            'left_features': len(left_df),
            'right_features': len(right_df)
        }
        
        # Perform overlay operation
        for left_idx, left_geom in enumerate(left_geometries):
            for right_idx, right_geom in enumerate(right_geometries):
                result_geom = self._perform_geometric_operation(left_geom, right_geom, operation)
                
                if result_geom is not None and not result_geom.is_empty:
                    # Filter by geometry type if requested
                    if keep_geom_type:
                        left_geom_type = type(left_geom).__name__
                        result_geom_type = type(result_geom).__name__
                        if left_geom_type != result_geom_type:
                            continue
                    
                    result_geometries.append(result_geom)
                    
                    # Combine attributes from both datasets
                    result_row = {'geometry': result_geom}
                    
                    # Add left attributes
                    for col in left_df.columns:
                        if col != left_geometry_col:
                            result_row[f"{col}_1"] = left_df.iloc[left_idx][col]
                            
                    # Add right attributes  
                    for col in right_df.columns:
                        if col != right_geometry_col:
                            result_row[f"{col}_2"] = right_df.iloc[right_idx][col]
                            
                    result_rows.append(result_row)
        
        # Create result DataFrame
        result_df = pd.DataFrame(result_rows) if result_rows else pd.DataFrame()
        
        execution_time = time.time() - start_time
        
        return OverlayResult(
            result_geometries=result_geometries,
            result_data=result_df,
            operation=operation,
            input_counts=input_counts,
            output_count=len(result_geometries),
            execution_time=execution_time
        )
    
    def _prepare_geometries(self, data: pd.DataFrame, geometry_column: str) -> List[Any]:
        """Prepare geometries for spatial operations."""
        if geometry_column not in data.columns:
            raise ValueError(f"Geometry column '{geometry_column}' not found in data")
            
        geometries = []
        for idx, row in data.iterrows():
            geom = row[geometry_column]
            if isinstance(geom, (tuple, list)) and len(geom) == 2:
                # Convert point coordinates to geometry object
                if _dependency_status.is_available(GeospatialLibrary.SHAPELY):
                    from shapely.geometry import Point
                    geometries.append(Point(geom[0], geom[1]))
                else:
                    geometries.append(geom)
            else:
                geometries.append(geom)
                
        return geometries


class SpatialAggregator:
    """
    Tools for spatial aggregation operations.
    """
    
    def __init__(self):
        """Initialize the spatial aggregator."""
        self.join_engine = SpatialJoinEngine()
        
    def aggregate_by_geometry(self,
                             point_df: pd.DataFrame,
                             polygon_df: pd.DataFrame,
                             value_column: str,
                             aggregation_functions: List[str] = ['mean', 'sum', 'count'],
                             point_geometry_col: str = 'geometry',
                             polygon_geometry_col: str = 'geometry') -> pd.DataFrame:
        """
        Aggregate point values within polygons.
        
        Parameters
        ----------
        point_df : DataFrame
            Point dataset with values to aggregate.
        polygon_df : DataFrame
            Polygon dataset for aggregation boundaries.
        value_column : str
            Column in point data to aggregate.
        aggregation_functions : list
            List of aggregation functions to apply.
        point_geometry_col : str
            Name of geometry column in point dataset.
        polygon_geometry_col : str
            Name of geometry column in polygon dataset.
            
        Returns
        -------
        DataFrame
            Polygon data with aggregated values.
        """
        # Perform spatial join to assign points to polygons
        join_result = self.join_engine.spatial_join(
            point_df, polygon_df,
            left_geometry_col=point_geometry_col,
            right_geometry_col=polygon_geometry_col,
            join_type=SpatialJoinType.WITHIN,
            how='inner'
        )
        
        if join_result.joined_data.empty:
            # No points within polygons, return empty result
            result_df = polygon_df.copy()
            for func in aggregation_functions:
                result_df[f"{value_column}_{func}"] = np.nan
            return result_df
            
        # Group by polygon attributes and aggregate
        # Identify polygon columns (those with '_right' suffix)
        polygon_cols = [col for col in join_result.joined_data.columns if col.endswith('_right')]
        
        if not polygon_cols:
            raise ValueError("No polygon columns found in join result")
            
        # Create aggregation dictionary
        agg_dict = {}
        value_col_left = f"{value_column}_left"
        
        if value_col_left not in join_result.joined_data.columns:
            raise ValueError(f"Value column '{value_column}' not found in joined data")
            
        for func in aggregation_functions:
            if func in ['mean', 'sum', 'std', 'min', 'max', 'median']:
                agg_dict[f"{value_column}_{func}"] = (value_col_left, func)
            elif func == 'count':
                agg_dict[f"{value_column}_count"] = (value_col_left, 'count')
            else:
                logger.warning(f"Aggregation function '{func}' not supported")
                
        # Perform aggregation
        grouped = join_result.joined_data.groupby(polygon_cols).agg(agg_dict)
        grouped.columns = [col[0] for col in grouped.columns]
        grouped = grouped.reset_index()
        
        # Merge back with original polygon data
        # Create mapping from original to suffixed columns
        polygon_mapping = {}
        for col in polygon_df.columns:
            suffixed_col = f"{col}_right"
            if suffixed_col in polygon_cols:
                polygon_mapping[suffixed_col] = col
                
        # Rename columns back to original names
        grouped = grouped.rename(columns=polygon_mapping)
        
        # Merge with original polygon data to preserve all polygons
        result_df = polygon_df.merge(grouped, on=list(polygon_mapping.values()), how='left')
        
        # Fill NaN values for polygons with no points
        for func in aggregation_functions:
            col_name = f"{value_column}_{func}"
            if col_name in result_df.columns:
                if func == 'count':
                    result_df[col_name] = result_df[col_name].fillna(0)
                else:
                    result_df[col_name] = result_df[col_name].fillna(np.nan)
                    
        return result_df


class SpatialJoinTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for spatial join operations.
    """
    
    def __init__(self,
                 right_data: pd.DataFrame,
                 left_geometry_col: str = 'geometry',
                 right_geometry_col: str = 'geometry',
                 join_type: SpatialJoinType = SpatialJoinType.INTERSECTS,
                 how: str = 'inner'):
        """
        Initialize spatial join transformer.
        
        Parameters
        ----------
        right_data : DataFrame
            Right dataset to join with.
        left_geometry_col : str
            Name of geometry column in left dataset.
        right_geometry_col : str
            Name of geometry column in right dataset.
        join_type : SpatialJoinType
            Type of spatial relationship to test.
        how : str
            Type of join ('inner', 'left', 'right').
        """
        self.right_data = right_data
        self.left_geometry_col = left_geometry_col
        self.right_geometry_col = right_geometry_col
        self.join_type = join_type
        self.how = how
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit the spatial join transformer.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Left dataset for spatial join.
        y : array-like, optional
            Target values (ignored).
            
        Returns
        -------
        self : SpatialJoinTransformer
            Fitted transformer.
        """
        self.join_engine_ = SpatialJoinEngine()
        return self
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Perform spatial join transformation.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Left dataset for spatial join.
            
        Returns
        -------
        X_transformed : DataFrame or array-like
            Joined dataset.
        """
        check_is_fitted(self, 'join_engine_')
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Spatial join requires DataFrame input with geometry column")
            
        result = self.join_engine_.spatial_join(
            X, self.right_data,
            left_geometry_col=self.left_geometry_col,
            right_geometry_col=self.right_geometry_col,
            join_type=self.join_type,
            how=self.how
        )
        
        return result.joined_data


class SpatialOverlayTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for spatial overlay operations.
    """
    
    def __init__(self,
                 right_data: pd.DataFrame,
                 operation: OverlayOperation,
                 left_geometry_col: str = 'geometry',
                 right_geometry_col: str = 'geometry',
                 keep_geom_type: bool = True):
        """
        Initialize spatial overlay transformer.
        
        Parameters
        ----------
        right_data : DataFrame
            Right dataset for overlay operation.
        operation : OverlayOperation
            Type of overlay operation to perform.
        left_geometry_col : str
            Name of geometry column in left dataset.
        right_geometry_col : str
            Name of geometry column in right dataset.
        keep_geom_type : bool
            Whether to filter results to maintain geometry type.
        """
        self.right_data = right_data
        self.operation = operation
        self.left_geometry_col = left_geometry_col
        self.right_geometry_col = right_geometry_col
        self.keep_geom_type = keep_geom_type
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit the spatial overlay transformer.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Left dataset for spatial overlay.
        y : array-like, optional
            Target values (ignored).
            
        Returns
        -------
        self : SpatialOverlayTransformer
            Fitted transformer.
        """
        self.overlay_engine_ = SpatialOverlayEngine()
        return self
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Perform spatial overlay transformation.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Left dataset for spatial overlay.
            
        Returns
        -------
        X_transformed : DataFrame or array-like
            Overlay result dataset.
        """
        check_is_fitted(self, 'overlay_engine_')
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Spatial overlay requires DataFrame input with geometry column")
            
        result = self.overlay_engine_.spatial_overlay(
            X, self.right_data,
            operation=self.operation,
            left_geometry_col=self.left_geometry_col,
            right_geometry_col=self.right_geometry_col,
            keep_geom_type=self.keep_geom_type
        )
        
        return result.result_data


# High-level convenience functions
def perform_spatial_join(left_data: pd.DataFrame,
                        right_data: pd.DataFrame,
                        join_type: str = 'intersects',
                        left_geometry_col: str = 'geometry',
                        right_geometry_col: str = 'geometry',
                        how: str = 'inner') -> SpatialJoinResult:
    """
    Perform spatial join between two datasets.
    
    Parameters
    ----------
    left_data : DataFrame
        Left dataset with geometry column.
    right_data : DataFrame
        Right dataset with geometry column.
    join_type : str
        Type of spatial relationship ('intersects', 'within', 'contains', etc.).
    left_geometry_col : str
        Name of geometry column in left dataset.
    right_geometry_col : str
        Name of geometry column in right dataset.
    how : str
        Type of join ('inner', 'left', 'right').
        
    Returns
    -------
    SpatialJoinResult
        Results of the spatial join operation.
    """
    join_type_enum = SpatialJoinType(join_type)
    engine = SpatialJoinEngine()
    
    return engine.spatial_join(
        left_data, right_data,
        left_geometry_col=left_geometry_col,
        right_geometry_col=right_geometry_col,
        join_type=join_type_enum,
        how=how
    )


def perform_spatial_overlay(left_data: pd.DataFrame,
                           right_data: pd.DataFrame,
                           operation: str = 'intersection',
                           left_geometry_col: str = 'geometry',
                           right_geometry_col: str = 'geometry',
                           keep_geom_type: bool = True) -> OverlayResult:
    """
    Perform spatial overlay operation between two datasets.
    
    Parameters
    ----------
    left_data : DataFrame
        Left dataset with geometry column.
    right_data : DataFrame
        Right dataset with geometry column.
    operation : str
        Type of overlay operation ('intersection', 'union', 'difference', etc.).
    left_geometry_col : str
        Name of geometry column in left dataset.
    right_geometry_col : str
        Name of geometry column in right dataset.
    keep_geom_type : bool
        Whether to filter results to maintain geometry type.
        
    Returns
    -------
    OverlayResult
        Results of the spatial overlay operation.
    """
    operation_enum = OverlayOperation(operation)
    engine = SpatialOverlayEngine()
    
    return engine.spatial_overlay(
        left_data, right_data,
        operation=operation_enum,
        left_geometry_col=left_geometry_col,
        right_geometry_col=right_geometry_col,
        keep_geom_type=keep_geom_type
    )


def aggregate_points_in_polygons(point_data: pd.DataFrame,
                                polygon_data: pd.DataFrame,
                                value_column: str,
                                aggregation_functions: List[str] = ['mean', 'sum', 'count'],
                                point_geometry_col: str = 'geometry',
                                polygon_geometry_col: str = 'geometry') -> pd.DataFrame:
    """
    Aggregate point values within polygon boundaries.
    
    Parameters
    ----------
    point_data : DataFrame
        Point dataset with values to aggregate.
    polygon_data : DataFrame
        Polygon dataset for aggregation boundaries.
    value_column : str
        Column in point data to aggregate.
    aggregation_functions : list
        List of aggregation functions to apply.
    point_geometry_col : str
        Name of geometry column in point dataset.
    polygon_geometry_col : str
        Name of geometry column in polygon dataset.
        
    Returns
    -------
    DataFrame
        Polygon data with aggregated values.
    """
    aggregator = SpatialAggregator()
    
    return aggregator.aggregate_by_geometry(
        point_data, polygon_data,
        value_column=value_column,
        aggregation_functions=aggregation_functions,
        point_geometry_col=point_geometry_col,
        polygon_geometry_col=polygon_geometry_col
    )


# =============================================================================
# Network Analysis with Spatial Constraints
# =============================================================================

class NetworkAnalysisType(Enum):
    """Types of network analysis operations."""
    SHORTEST_PATH = "shortest_path"
    ROUTING = "routing"
    ACCESSIBILITY = "accessibility"
    ISOCHRONE = "isochrone"
    NETWORK_CONNECTIVITY = "connectivity"
    SERVICE_AREA = "service_area"


@dataclass
class RouteResult:
    """Results from a route optimization operation."""
    route_path: List[Any]
    route_coordinates: List[Tuple[float, float]]
    total_distance: float
    total_time: Optional[float]
    waypoints: List[Any]
    path_geometry: Optional[Any]
    execution_time: float
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the route result."""
        return {
            'total_distance': self.total_distance,
            'total_time': self.total_time,
            'waypoints_count': len(self.waypoints),
            'path_length': len(self.route_path),
            'execution_time_seconds': self.execution_time
        }


@dataclass
class AccessibilityResult:
    """Results from an accessibility analysis."""
    accessibility_scores: Dict[str, float]
    reachable_locations: List[Any]
    travel_times: Dict[str, float]
    service_coverage: Dict[str, Any]
    analysis_parameters: Dict[str, Any]
    execution_time: float
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the accessibility result."""
        return {
            'total_locations_analyzed': len(self.accessibility_scores),
            'reachable_count': len(self.reachable_locations),
            'average_accessibility': np.mean(list(self.accessibility_scores.values())) if self.accessibility_scores else 0,
            'max_travel_time': max(self.travel_times.values()) if self.travel_times else 0,
            'analysis_parameters': self.analysis_parameters,
            'execution_time_seconds': self.execution_time
        }


@dataclass
class IsochroneResult:
    """Results from an isochrone analysis."""
    isochrone_polygons: List[Any]
    time_bands: List[float]
    coverage_areas: Dict[float, float]
    population_coverage: Optional[Dict[float, int]]
    service_points: List[Any]
    execution_time: float
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the isochrone result."""
        return {
            'time_bands': self.time_bands,
            'isochrone_count': len(self.isochrone_polygons),
            'coverage_areas': self.coverage_areas,
            'population_coverage': self.population_coverage,
            'service_points_count': len(self.service_points),
            'execution_time_seconds': self.execution_time
        }


class SpatialNetwork:
    """
    Core spatial network representation with networkx integration.
    """
    
    def __init__(self):
        """Initialize the spatial network."""
        self.dependency_checker = GeospatialDependencyChecker()
        self.network = None
        self.node_coordinates = {}
        self.edge_geometries = {}
        
    def create_network_from_points(self, 
                                  points: List[Tuple[float, float]], 
                                  connection_threshold: float = None,
                                  connection_method: str = 'distance') -> None:
        """
        Create a network from point coordinates.
        
        Parameters
        ----------
        points : list
            List of (x, y) coordinate tuples.
        connection_threshold : float, optional
            Maximum distance for automatic connections.
        connection_method : str
            Method for connecting points ('distance', 'knn', 'delaunay').
        """
        if not _dependency_status.is_available(GeospatialLibrary.NETWORKX):
            raise ValueError("NetworkX is required for network analysis")
            
        import networkx as nx
        
        self.network = nx.Graph()
        
        # Add nodes with coordinates
        for i, (x, y) in enumerate(points):
            self.network.add_node(i, x=x, y=y)
            self.node_coordinates[i] = (x, y)
            
        # Create connections based on method
        if connection_method == 'distance' and connection_threshold:
            self._connect_by_distance(connection_threshold)
        elif connection_method == 'knn':
            k = min(5, len(points) - 1) if connection_threshold is None else int(connection_threshold)
            self._connect_by_knn(k)
        elif connection_method == 'delaunay':
            self._connect_by_delaunay()
        else:
            logger.warning(f"Unknown connection method: {connection_method}")
            
    def create_network_from_edges(self, 
                                 nodes: List[Dict], 
                                 edges: List[Dict]) -> None:
        """
        Create a network from explicit node and edge definitions.
        
        Parameters
        ----------
        nodes : list
            List of node dictionaries with 'id', 'x', 'y' keys.
        edges : list
            List of edge dictionaries with 'source', 'target', optional 'weight', 'geometry' keys.
        """
        if not _dependency_status.is_available(GeospatialLibrary.NETWORKX):
            raise ValueError("NetworkX is required for network analysis")
            
        import networkx as nx
        
        self.network = nx.Graph()
        
        # Add nodes
        for node in nodes:
            node_id = node['id']
            x, y = node['x'], node['y']
            self.network.add_node(node_id, x=x, y=y, **{k: v for k, v in node.items() if k not in ['id', 'x', 'y']})
            self.node_coordinates[node_id] = (x, y)
            
        # Add edges
        for edge in edges:
            source, target = edge['source'], edge['target']
            weight = edge.get('weight', 1.0)
            
            # Calculate distance if not provided
            if 'weight' not in edge and source in self.node_coordinates and target in self.node_coordinates:
                coord1, coord2 = self.node_coordinates[source], self.node_coordinates[target]
                weight = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                
            self.network.add_edge(source, target, weight=weight, **{k: v for k, v in edge.items() if k not in ['source', 'target', 'weight']})
            
            # Store edge geometry if provided
            if 'geometry' in edge:
                self.edge_geometries[(source, target)] = edge['geometry']
                
    def _connect_by_distance(self, threshold: float) -> None:
        """Connect nodes within distance threshold."""
        distance_calc = SpatialDistanceCalculator()
        
        node_list = list(self.network.nodes())
        for i, node1 in enumerate(node_list):
            coord1 = self.node_coordinates[node1]
            for node2 in node_list[i+1:]:
                coord2 = self.node_coordinates[node2]
                
                distance = distance_calc.euclidean_distance(coord1, coord2)
                if distance <= threshold:
                    self.network.add_edge(node1, node2, weight=distance)
                    
    def _connect_by_knn(self, k: int) -> None:
        """Connect each node to k nearest neighbors."""
        distance_calc = SpatialDistanceCalculator()
        
        for node in self.network.nodes():
            coord = self.node_coordinates[node]
            
            # Calculate distances to all other nodes
            distances = []
            for other_node in self.network.nodes():
                if other_node != node:
                    other_coord = self.node_coordinates[other_node]
                    distance = distance_calc.euclidean_distance(coord, other_coord)
                    distances.append((distance, other_node))
                    
            # Connect to k nearest neighbors
            distances.sort()
            for distance, neighbor in distances[:k]:
                if not self.network.has_edge(node, neighbor):
                    self.network.add_edge(node, neighbor, weight=distance)
                    
    def _connect_by_delaunay(self) -> None:
        """Connect nodes using Delaunay triangulation."""
        if not _dependency_status.is_available(GeospatialLibrary.SCIPY):
            logger.warning("SciPy not available for Delaunay triangulation")
            return
            
        try:
            from scipy.spatial import Delaunay
            
            points = np.array([self.node_coordinates[node] for node in self.network.nodes()])
            node_list = list(self.network.nodes())
            
            tri = Delaunay(points)
            
            distance_calc = SpatialDistanceCalculator()
            
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        node1, node2 = node_list[simplex[i]], node_list[simplex[j]]
                        if not self.network.has_edge(node1, node2):
                            coord1, coord2 = self.node_coordinates[node1], self.node_coordinates[node2]
                            distance = distance_calc.euclidean_distance(coord1, coord2)
                            self.network.add_edge(node1, node2, weight=distance)
                            
        except ImportError:
            logger.warning("Delaunay triangulation requires scipy")
            
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get basic network topology statistics."""
        if self.network is None:
            return {}
            
        import networkx as nx
        
        stats = {
            'num_nodes': self.network.number_of_nodes(),
            'num_edges': self.network.number_of_edges(),
            'density': nx.density(self.network),
            'is_connected': nx.is_connected(self.network),
        }
        
        if nx.is_connected(self.network):
            stats.update({
                'average_clustering': nx.average_clustering(self.network),
                'average_path_length': nx.average_shortest_path_length(self.network, weight='weight'),
                'diameter': nx.diameter(self.network)
            })
        else:
            stats['num_components'] = nx.number_connected_components(self.network)
            
        return stats


class NetworkRouter:
    """
    Route optimization and path finding with spatial constraints.
    """
    
    def __init__(self, network: SpatialNetwork):
        """Initialize the network router."""
        self.network = network
        self.distance_calc = SpatialDistanceCalculator()
        
    def find_shortest_path(self, 
                          source: Any, 
                          target: Any, 
                          weight: str = 'weight') -> RouteResult:
        """
        Find shortest path between two nodes.
        
        Parameters
        ----------
        source : Any
            Source node identifier.
        target : Any
            Target node identifier.
        weight : str
            Edge attribute to use as weight.
            
        Returns
        -------
        RouteResult
            Route optimization results.
        """
        start_time = time.time()
        
        if self.network.network is None:
            raise ValueError("Network not initialized")
            
        if not _dependency_status.is_available(GeospatialLibrary.NETWORKX):
            raise ValueError("NetworkX is required for routing")
            
        import networkx as nx
        
        try:
            path = nx.shortest_path(self.network.network, source, target, weight=weight)
            path_length = nx.shortest_path_length(self.network.network, source, target, weight=weight)
            
            # Extract route coordinates
            route_coordinates = [self.network.node_coordinates[node] for node in path]
            
            # Create path geometry if Shapely available
            path_geometry = None
            if _dependency_status.is_available(GeospatialLibrary.SHAPELY):
                from shapely.geometry import LineString
                path_geometry = LineString(route_coordinates)
                
            execution_time = time.time() - start_time
            
            return RouteResult(
                route_path=path,
                route_coordinates=route_coordinates,
                total_distance=path_length,
                total_time=None,  # Could be calculated if time weights available
                waypoints=[source, target],
                path_geometry=path_geometry,
                execution_time=execution_time
            )
            
        except nx.NetworkXNoPath:
            execution_time = time.time() - start_time
            return RouteResult(
                route_path=[],
                route_coordinates=[],
                total_distance=float('inf'),
                total_time=None,
                waypoints=[source, target],
                path_geometry=None,
                execution_time=execution_time
            )
            
    def find_optimal_route(self, 
                          waypoints: List[Any], 
                          return_to_start: bool = False,
                          optimization_method: str = 'greedy') -> RouteResult:
        """
        Find optimal route through multiple waypoints.
        
        Parameters
        ----------
        waypoints : list
            List of waypoint node identifiers.
        return_to_start : bool
            Whether to return to starting waypoint.
        optimization_method : str
            Method for route optimization ('greedy', 'nearest_neighbor').
            
        Returns
        -------
        RouteResult
            Optimized route results.
        """
        start_time = time.time()
        
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required")
            
        if optimization_method == 'greedy':
            route_path, total_distance = self._greedy_route_optimization(waypoints, return_to_start)
        elif optimization_method == 'nearest_neighbor':
            route_path, total_distance = self._nearest_neighbor_tsp(waypoints, return_to_start)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
            
        # Extract route coordinates
        route_coordinates = [self.network.node_coordinates[node] for node in route_path]
        
        # Create path geometry if Shapely available
        path_geometry = None
        if _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            from shapely.geometry import LineString
            path_geometry = LineString(route_coordinates)
            
        execution_time = time.time() - start_time
        
        return RouteResult(
            route_path=route_path,
            route_coordinates=route_coordinates,
            total_distance=total_distance,
            total_time=None,
            waypoints=waypoints,
            path_geometry=path_geometry,
            execution_time=execution_time
        )
        
    def _greedy_route_optimization(self, waypoints: List[Any], return_to_start: bool) -> Tuple[List[Any], float]:
        """Greedy optimization for route through waypoints."""
        import networkx as nx
        
        if len(waypoints) == 2:
            # Simple shortest path for 2 waypoints
            try:
                path = nx.shortest_path(self.network.network, waypoints[0], waypoints[1], weight='weight')
                distance = nx.shortest_path_length(self.network.network, waypoints[0], waypoints[1], weight='weight')
                
                if return_to_start and waypoints[0] != waypoints[1]:
                    return_path = nx.shortest_path(self.network.network, waypoints[1], waypoints[0], weight='weight')[1:]
                    return_distance = nx.shortest_path_length(self.network.network, waypoints[1], waypoints[0], weight='weight')
                    path.extend(return_path)
                    distance += return_distance
                    
                return path, distance
            except nx.NetworkXNoPath:
                return waypoints, float('inf')
        
        # For multiple waypoints, connect them sequentially
        full_path = []
        total_distance = 0
        
        for i in range(len(waypoints) - 1):
            try:
                segment = nx.shortest_path(self.network.network, waypoints[i], waypoints[i+1], weight='weight')
                segment_distance = nx.shortest_path_length(self.network.network, waypoints[i], waypoints[i+1], weight='weight')
                
                if i == 0:
                    full_path.extend(segment)
                else:
                    full_path.extend(segment[1:])  # Skip duplicate node
                    
                total_distance += segment_distance
                
            except nx.NetworkXNoPath:
                return waypoints, float('inf')
                
        # Return to start if requested
        if return_to_start and len(waypoints) > 1:
            try:
                return_segment = nx.shortest_path(self.network.network, waypoints[-1], waypoints[0], weight='weight')[1:]
                return_distance = nx.shortest_path_length(self.network.network, waypoints[-1], waypoints[0], weight='weight')
                full_path.extend(return_segment)
                total_distance += return_distance
            except nx.NetworkXNoPath:
                return full_path, float('inf')
                
        return full_path, total_distance
        
    def _nearest_neighbor_tsp(self, waypoints: List[Any], return_to_start: bool) -> Tuple[List[Any], float]:
        """Nearest neighbor heuristic for TSP-like routing."""
        import networkx as nx
        
        if len(waypoints) <= 2:
            return self._greedy_route_optimization(waypoints, return_to_start)
            
        # Build distance matrix between waypoints
        distance_matrix = {}
        for i, wp1 in enumerate(waypoints):
            for j, wp2 in enumerate(waypoints):
                if i != j:
                    try:
                        dist = nx.shortest_path_length(self.network.network, wp1, wp2, weight='weight')
                        distance_matrix[(wp1, wp2)] = dist
                    except nx.NetworkXNoPath:
                        distance_matrix[(wp1, wp2)] = float('inf')
                        
        # Nearest neighbor algorithm
        start_wp = waypoints[0]
        current_wp = start_wp
        remaining = set(waypoints[1:])
        route_order = [current_wp]
        total_distance = 0
        
        while remaining:
            nearest = min(remaining, key=lambda wp: distance_matrix.get((current_wp, wp), float('inf')))
            distance = distance_matrix.get((current_wp, nearest), float('inf'))
            
            if distance == float('inf'):
                break
                
            route_order.append(nearest)
            total_distance += distance
            current_wp = nearest
            remaining.remove(nearest)
            
        # Add return to start if requested
        if return_to_start and len(route_order) > 1:
            return_distance = distance_matrix.get((current_wp, start_wp), float('inf'))
            if return_distance != float('inf'):
                route_order.append(start_wp)
                total_distance += return_distance
                
        # Convert waypoint route to full node path
        full_path = []
        for i in range(len(route_order) - 1):
            try:
                segment = nx.shortest_path(self.network.network, route_order[i], route_order[i+1], weight='weight')
                if i == 0:
                    full_path.extend(segment)
                else:
                    full_path.extend(segment[1:])
            except nx.NetworkXNoPath:
                full_path.extend([route_order[i], route_order[i+1]])
                
        return full_path, total_distance


class AccessibilityAnalyzer:
    """
    Spatial accessibility analysis for service coverage and reachability.
    """
    
    def __init__(self, network: SpatialNetwork):
        """Initialize the accessibility analyzer."""
        self.network = network
        
    def calculate_accessibility(self, 
                              service_locations: List[Any], 
                              demand_locations: List[Any],
                              max_travel_time: float = None,
                              impedance_function: str = 'linear') -> AccessibilityResult:
        """
        Calculate accessibility scores for demand locations to services.
        
        Parameters
        ----------
        service_locations : list
            List of service node identifiers.
        demand_locations : list
            List of demand node identifiers.
        max_travel_time : float, optional
            Maximum travel time/distance threshold.
        impedance_function : str
            Function for distance decay ('linear', 'exponential', 'gaussian').
            
        Returns
        -------
        AccessibilityResult
            Accessibility analysis results.
        """
        start_time = time.time()
        
        if self.network.network is None:
            raise ValueError("Network not initialized")
            
        import networkx as nx
        
        accessibility_scores = {}
        travel_times = {}
        reachable_locations = []
        
        # Calculate accessibility for each demand location
        for demand_node in demand_locations:
            accessibility_score = 0
            min_travel_time = float('inf')
            
            for service_node in service_locations:
                try:
                    travel_distance = nx.shortest_path_length(
                        self.network.network, demand_node, service_node, weight='weight'
                    )
                    
                    # Apply distance decay function
                    if impedance_function == 'linear':
                        impedance = 1 / (1 + travel_distance) if travel_distance > 0 else 1
                    elif impedance_function == 'exponential':
                        impedance = np.exp(-travel_distance / 1000) if travel_distance > 0 else 1  # Decay parameter
                    elif impedance_function == 'gaussian':
                        impedance = np.exp(-(travel_distance ** 2) / (2 * (500 ** 2)))  # Gaussian decay
                    else:
                        impedance = 1
                        
                    accessibility_score += impedance
                    min_travel_time = min(min_travel_time, travel_distance)
                    
                except nx.NetworkXNoPath:
                    continue
                    
            accessibility_scores[demand_node] = accessibility_score
            travel_times[demand_node] = min_travel_time
            
            # Check reachability threshold
            if max_travel_time is None or min_travel_time <= max_travel_time:
                reachable_locations.append(demand_node)
                
        # Calculate service coverage statistics
        total_demand = len(demand_locations)
        reachable_count = len(reachable_locations)
        coverage_percentage = (reachable_count / total_demand * 100) if total_demand > 0 else 0
        
        service_coverage = {
            'total_demand_locations': total_demand,
            'reachable_locations': reachable_count,
            'coverage_percentage': coverage_percentage,
            'average_accessibility': np.mean(list(accessibility_scores.values())) if accessibility_scores else 0
        }
        
        analysis_parameters = {
            'service_count': len(service_locations),
            'demand_count': len(demand_locations),
            'max_travel_time': max_travel_time,
            'impedance_function': impedance_function
        }
        
        execution_time = time.time() - start_time
        
        return AccessibilityResult(
            accessibility_scores=accessibility_scores,
            reachable_locations=reachable_locations,
            travel_times=travel_times,
            service_coverage=service_coverage,
            analysis_parameters=analysis_parameters,
            execution_time=execution_time
        )


class IsochroneGenerator:
    """
    Generate isochrones (equal-time/distance polygons) for service areas.
    """
    
    def __init__(self, network: SpatialNetwork):
        """Initialize the isochrone generator."""
        self.network = network
        
    def generate_isochrones(self, 
                          service_locations: List[Any], 
                          time_bands: List[float],
                          resolution: int = 50) -> IsochroneResult:
        """
        Generate isochrone polygons for service locations.
        
        Parameters
        ----------
        service_locations : list
            List of service node identifiers.
        time_bands : list
            List of travel time thresholds for isochrone bands.
        resolution : int
            Resolution for isochrone polygon generation.
            
        Returns
        -------
        IsochroneResult
            Isochrone analysis results.
        """
        start_time = time.time()
        
        if not _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            logger.warning("Shapely not available for isochrone polygon generation")
            return IsochroneResult(
                isochrone_polygons=[],
                time_bands=time_bands,
                coverage_areas={},
                population_coverage=None,
                service_points=service_locations,
                execution_time=time.time() - start_time
            )
            
        import networkx as nx
        from shapely.geometry import Point, Polygon
        from shapely.ops import unary_union
        
        isochrone_polygons = []
        coverage_areas = {}
        
        # Generate isochrones for each time band
        for time_threshold in sorted(time_bands):
            band_polygons = []
            
            # Generate isochrone for each service location
            for service_node in service_locations:
                reachable_nodes = []
                
                # Find all nodes reachable within time threshold
                try:
                    path_lengths = nx.single_source_shortest_path_length(
                        self.network.network, service_node, cutoff=time_threshold, weight='weight'
                    )
                    reachable_nodes = list(path_lengths.keys())
                except:
                    continue
                    
                if len(reachable_nodes) < 3:  # Need at least 3 points for polygon
                    continue
                    
                # Extract coordinates of reachable nodes
                reachable_coords = [self.network.node_coordinates[node] for node in reachable_nodes]
                
                # Create convex hull as simple isochrone approximation
                if len(reachable_coords) >= 3:
                    try:
                        points = [Point(coord) for coord in reachable_coords]
                        # Create buffer around points to form isochrone
                        buffer_distance = time_threshold / 10  # Simple heuristic
                        buffered_points = [point.buffer(buffer_distance) for point in points]
                        isochrone_polygon = unary_union(buffered_points)
                        band_polygons.append(isochrone_polygon)
                    except Exception as e:
                        logger.warning(f"Failed to create isochrone polygon: {e}")
                        continue
                        
            # Combine polygons for this time band
            if band_polygons:
                combined_polygon = unary_union(band_polygons)
                isochrone_polygons.append(combined_polygon)
                coverage_areas[time_threshold] = combined_polygon.area
            else:
                isochrone_polygons.append(None)
                coverage_areas[time_threshold] = 0
                
        execution_time = time.time() - start_time
        
        return IsochroneResult(
            isochrone_polygons=isochrone_polygons,
            time_bands=time_bands,
            coverage_areas=coverage_areas,
            population_coverage=None,  # Would need population data
            service_points=service_locations,
            execution_time=execution_time
        )


class SpatialNetworkTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for network analysis operations.
    """
    
    def __init__(self,
                 analysis_type: NetworkAnalysisType = NetworkAnalysisType.ACCESSIBILITY,
                 network_data: Optional[Dict] = None,
                 service_locations: List[Any] = None,
                 max_travel_time: float = None):
        """
        Initialize spatial network transformer.
        
        Parameters
        ----------
        analysis_type : NetworkAnalysisType
            Type of network analysis to perform.
        network_data : dict, optional
            Network definition with 'nodes' and 'edges' keys.
        service_locations : list, optional
            Service node identifiers for accessibility/isochrone analysis.
        max_travel_time : float, optional
            Maximum travel time threshold.
        """
        self.analysis_type = analysis_type
        self.network_data = network_data
        self.service_locations = service_locations or []
        self.max_travel_time = max_travel_time
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit the spatial network transformer.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Input data containing node coordinates or network definition.
        y : array-like, optional
            Target values (ignored).
            
        Returns
        -------
        self : SpatialNetworkTransformer
            Fitted transformer.
        """
        # Initialize network
        self.network_ = SpatialNetwork()
        
        if self.network_data:
            # Use provided network data
            self.network_.create_network_from_edges(
                self.network_data['nodes'], 
                self.network_data['edges']
            )
        else:
            # Create network from input coordinates
            if isinstance(X, pd.DataFrame):
                if 'x' in X.columns and 'y' in X.columns:
                    points = list(zip(X['x'], X['y']))
                elif 'longitude' in X.columns and 'latitude' in X.columns:
                    points = list(zip(X['longitude'], X['latitude']))
                else:
                    raise ValueError("DataFrame must contain 'x'/'y' or 'longitude'/'latitude' columns")
            else:
                points = [(row[0], row[1]) for row in X]
                
            self.network_.create_network_from_points(points, connection_method='knn')
            
        # Initialize analyzers based on analysis type
        if self.analysis_type in [NetworkAnalysisType.SHORTEST_PATH, NetworkAnalysisType.ROUTING]:
            self.router_ = NetworkRouter(self.network_)
        elif self.analysis_type == NetworkAnalysisType.ACCESSIBILITY:
            self.accessibility_analyzer_ = AccessibilityAnalyzer(self.network_)
        elif self.analysis_type == NetworkAnalysisType.ISOCHRONE:
            self.isochrone_generator_ = IsochroneGenerator(self.network_)
            
        return self
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Perform network analysis transformation.
        
        Parameters
        ----------
        X : DataFrame or array-like
            Input data for analysis.
            
        Returns
        -------
        X_transformed : DataFrame or array-like
            Analysis results.
        """
        check_is_fitted(self, 'network_')
        
        if self.analysis_type == NetworkAnalysisType.ACCESSIBILITY:
            if not hasattr(self, 'accessibility_analyzer_'):
                raise ValueError("Accessibility analyzer not initialized")
                
            # Use all nodes as demand locations if not specified
            demand_locations = list(range(len(X))) if not isinstance(X, pd.DataFrame) else list(X.index)
            
            result = self.accessibility_analyzer_.calculate_accessibility(
                self.service_locations, 
                demand_locations,
                max_travel_time=self.max_travel_time
            )
            
            # Create output DataFrame
            if isinstance(X, pd.DataFrame):
                X_transformed = X.copy()
                X_transformed['accessibility_score'] = [result.accessibility_scores.get(i, 0) for i in X.index]
                X_transformed['travel_time'] = [result.travel_times.get(i, float('inf')) for i in X.index]
                X_transformed['is_reachable'] = [i in result.reachable_locations for i in X.index]
                return X_transformed
            else:
                accessibility_values = [result.accessibility_scores.get(i, 0) for i in range(len(X))]
                travel_times = [result.travel_times.get(i, float('inf')) for i in range(len(X))]
                return np.column_stack([X, accessibility_values, travel_times])
                
        elif self.analysis_type == NetworkAnalysisType.ISOCHRONE:
            if not hasattr(self, 'isochrone_generator_'):
                raise ValueError("Isochrone generator not initialized")
                
            time_bands = [self.max_travel_time] if self.max_travel_time else [500, 1000, 1500]
            result = self.isochrone_generator_.generate_isochrones(
                self.service_locations, 
                time_bands
            )
            
            # Return isochrone summary as DataFrame
            summary_data = {
                'time_band': time_bands,
                'coverage_area': [result.coverage_areas.get(band, 0) for band in time_bands],
                'has_polygon': [i < len(result.isochrone_polygons) and result.isochrone_polygons[i] is not None 
                               for i in range(len(time_bands))]
            }
            
            return pd.DataFrame(summary_data)
            
        else:
            # For other analysis types, return network statistics
            stats = self.network_.get_network_statistics()
            
            if isinstance(X, pd.DataFrame):
                X_transformed = X.copy()
                for key, value in stats.items():
                    X_transformed[f'network_{key}'] = value
                return X_transformed
            else:
                return X  # Return unchanged for non-accessibility analysis
                

# High-level convenience functions
def optimize_route(network_data: Dict,
                  waypoints: List[Any],
                  return_to_start: bool = False,
                  optimization_method: str = 'greedy') -> RouteResult:
    """
    Optimize route through multiple waypoints on a spatial network.
    
    Parameters
    ----------
    network_data : dict
        Network definition with 'nodes' and 'edges' keys.
    waypoints : list
        List of waypoint node identifiers.
    return_to_start : bool
        Whether to return to starting waypoint.
    optimization_method : str
        Method for route optimization.
        
    Returns
    -------
    RouteResult
        Optimized route results.
    """
    network = SpatialNetwork()
    network.create_network_from_edges(network_data['nodes'], network_data['edges'])
    
    router = NetworkRouter(network)
    return router.find_optimal_route(waypoints, return_to_start, optimization_method)


def analyze_accessibility(network_data: Dict,
                         service_locations: List[Any],
                         demand_locations: List[Any],
                         max_travel_time: float = None,
                         impedance_function: str = 'linear') -> AccessibilityResult:
    """
    Analyze spatial accessibility to services on a network.
    
    Parameters
    ----------
    network_data : dict
        Network definition with 'nodes' and 'edges' keys.
    service_locations : list
        List of service node identifiers.
    demand_locations : list
        List of demand node identifiers.
    max_travel_time : float, optional
        Maximum travel time threshold.
    impedance_function : str
        Function for distance decay.
        
    Returns
    -------
    AccessibilityResult
        Accessibility analysis results.
    """
    network = SpatialNetwork()
    network.create_network_from_edges(network_data['nodes'], network_data['edges'])
    
    analyzer = AccessibilityAnalyzer(network)
    return analyzer.calculate_accessibility(
        service_locations, demand_locations, max_travel_time, impedance_function
    )


def generate_service_isochrones(network_data: Dict,
                               service_locations: List[Any],
                               time_bands: List[float],
                               resolution: int = 50) -> IsochroneResult:
    """
    Generate isochrone polygons for service locations.
    
    Parameters
    ----------
    network_data : dict
        Network definition with 'nodes' and 'edges' keys.
    service_locations : list
        List of service node identifiers.
    time_bands : list
        List of travel time thresholds.
    resolution : int
        Resolution for polygon generation.
        
    Returns
    -------
    IsochroneResult
        Isochrone analysis results.
    """
    network = SpatialNetwork()
    network.create_network_from_edges(network_data['nodes'], network_data['edges'])
    
    generator = IsochroneGenerator(network)
    return generator.generate_isochrones(service_locations, time_bands, resolution)


# Initialize dependency checking on module import
logger.info("Initializing geospatial analysis module")
logger.info(f"Available geospatial libraries: "
           f"{[lib.value for lib, available in _dependency_status.available_libraries.items() if available]}")

if not _dependency_status.has_core_geospatial():
    logger.warning("Core geospatial libraries (geopandas, shapely) not available. "
                  "Some functionality will be limited to basic operations.")
    logger.info("Install with: pip install geopandas shapely")