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


# Initialize dependency checking on module import
logger.info("Initializing geospatial analysis module")
logger.info(f"Available geospatial libraries: "
           f"{[lib.value for lib, available in _dependency_status.available_libraries.items() if available]}")

if not _dependency_status.has_core_geospatial():
    logger.warning("Core geospatial libraries (geopandas, shapely) not available. "
                  "Some functionality will be limited to basic operations.")
    logger.info("Install with: pip install geopandas shapely")