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


# Initialize dependency checking on module import
logger.info("Initializing geospatial analysis module")
logger.info(f"Available geospatial libraries: "
           f"{[lib.value for lib, available in _dependency_status.available_libraries.items() if available]}")

if not _dependency_status.has_core_geospatial():
    logger.warning("Core geospatial libraries (geopandas, shapely) not available. "
                  "Some functionality will be limited to basic operations.")
    logger.info("Install with: pip install geopandas shapely")