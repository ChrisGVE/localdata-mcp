"""
Geospatial Integration Adapter - LocalData MCP v2.0

Specialized adapter for geospatial libraries integration:
- geopandas: Geospatial dataframes and operations
- shapely: Geometric operations and spatial analysis  
- rasterio: Raster data processing
- pyproj: Coordinate reference system transformations
- folium: Interactive mapping and visualization

Key Integration Challenges:
- GeoDataFrame vs DataFrame format differences
- Shapely geometry objects and spatial indexing
- Coordinate Reference System (CRS) handling
- Streaming compatibility for large geospatial datasets
- Memory-efficient raster processing
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Import base integration architecture
from library_integration_shims import (
    BaseLibraryAdapter,
    LibraryCategory, 
    LibraryDependency,
    IntegrationStrategy,
    IntegrationMetadata,
    LibraryIntegrationResult,
    requires_library,
    CompositionError
)

logger = logging.getLogger(__name__)


# ============================================================================
# Geospatial-Specific Data Structures
# ============================================================================

@dataclass
class GeospatialContext:
    """Context for geospatial operations."""
    crs: Optional[str] = None  # Coordinate Reference System
    geometry_column: str = 'geometry'
    bbox: Optional[Tuple[float, float, float, float]] = None  # Bounding box
    spatial_index_available: bool = False
    raster_profile: Optional[Dict[str, Any]] = None


@dataclass
class GeospatialMetadata(IntegrationMetadata):
    """Extended metadata for geospatial operations."""
    geometry_type: Optional[str] = None
    crs_original: Optional[str] = None
    crs_transformed: Optional[str] = None
    spatial_operations: List[str] = None
    raster_bands: Optional[int] = None
    raster_dtype: Optional[str] = None
    
    def __post_init__(self):
        if self.spatial_operations is None:
            self.spatial_operations = []


# ============================================================================
# Geospatial Adapter Implementation
# ============================================================================

class GeospatialAdapter(BaseLibraryAdapter):
    """
    Integration adapter for geospatial analysis libraries.
    
    Handles:
    - GeoDataFrame creation and conversion
    - Spatial operations with streaming support
    - CRS transformations and projections
    - Geometric analysis and spatial joins
    - Raster data processing with memory management
    """
    
    def __init__(self):
        dependencies = [
            LibraryDependency(
                name="geopandas",
                import_path="geopandas",
                min_version="0.10.0",
                sklearn_equivalent="sklearn.cluster.DBSCAN",  # Spatial clustering fallback
                installation_hint="pip install geopandas"
            ),
            LibraryDependency(
                name="shapely",
                import_path="shapely.geometry",
                min_version="1.8.0", 
                installation_hint="pip install shapely"
            ),
            LibraryDependency(
                name="rasterio",
                import_path="rasterio",
                sklearn_equivalent="numpy",  # Array processing fallback
                installation_hint="pip install rasterio"
            ),
            LibraryDependency(
                name="pyproj",
                import_path="pyproj",
                installation_hint="pip install pyproj"
            ),
            LibraryDependency(
                name="folium",
                import_path="folium",
                installation_hint="pip install folium"
            )
        ]
        
        super().__init__(LibraryCategory.GEOSPATIAL, dependencies)
        self.default_crs = "EPSG:4326"  # WGS84 
    
    def get_supported_functions(self) -> Dict[str, Callable]:
        """Return supported geospatial functions."""
        return {
            # Data Creation and Conversion
            'create_geodataframe': self.create_geodataframe,
            'points_from_coordinates': self.points_from_coordinates,
            'dataframe_to_geodataframe': self.dataframe_to_geodataframe,
            
            # Spatial Operations
            'spatial_join': self.spatial_join,
            'buffer_geometries': self.buffer_geometries,
            'calculate_distances': self.calculate_distances,
            'spatial_intersection': self.spatial_intersection,
            'spatial_clustering': self.spatial_clustering,
            
            # Coordinate Operations
            'transform_crs': self.transform_crs,
            'calculate_centroids': self.calculate_centroids,
            'calculate_area': self.calculate_area,
            
            # Raster Operations
            'read_raster': self.read_raster,
            'raster_stats': self.raster_stats,
            'raster_to_points': self.raster_to_points,
            
            # Visualization
            'create_map': self.create_interactive_map,
            'add_markers': self.add_markers_to_map
        }
    
    def adapt_function_call(self, 
                          function_name: str,
                          data: Any,
                          parameters: Dict[str, Any]) -> Tuple[Any, GeospatialMetadata]:
        """Adapt function call to geospatial library APIs."""
        
        if function_name not in self.get_supported_functions():
            raise CompositionError(
                f"Unsupported geospatial function: {function_name}",
                error_type="unsupported_function"
            )
        
        func = self.get_supported_functions()[function_name]
        
        # Convert input data to appropriate format
        converted_data, input_transformations = self.convert_input_data(data, "geopandas.GeoDataFrame")
        
        # Execute the function
        try:
            result = func(converted_data, **parameters)
            
            # Convert result to standard format if needed
            output_result, output_transformations = self.convert_output_data(result)
            
            # Create metadata
            metadata = GeospatialMetadata(
                library_used="geopandas" if self.is_library_available("geopandas") else "fallback",
                integration_strategy=IntegrationStrategy.SKLEARN_WRAPPER,
                data_transformations=input_transformations + output_transformations,
                streaming_compatible=True,
                original_parameters=parameters
            )
            
            return output_result, metadata
            
        except Exception as e:
            return self._handle_geospatial_error(function_name, data, parameters, e)
    
    # ========================================================================
    # Data Creation and Conversion Functions
    # ========================================================================
    
    @requires_library("geopandas")
    def create_geodataframe(self, data: pd.DataFrame, **params) -> Any:
        """Create GeoDataFrame from pandas DataFrame with coordinate columns."""
        geopandas = self.get_library("geopandas")
        
        lon_col = params.get('longitude_column', 'longitude')
        lat_col = params.get('latitude_column', 'latitude')  
        crs = params.get('crs', self.default_crs)
        
        if lon_col not in data.columns or lat_col not in data.columns:
            raise ValueError(f"Required coordinate columns {lon_col}, {lat_col} not found")
        
        # Create point geometries
        geometry = geopandas.points_from_xy(data[lon_col], data[lat_col])
        
        # Create GeoDataFrame
        gdf = geopandas.GeoDataFrame(data, geometry=geometry, crs=crs)
        
        return gdf
    
    @requires_library("geopandas")
    def points_from_coordinates(self, data: pd.DataFrame, **params) -> Any:
        """Create point geometries from coordinate columns."""
        geopandas = self.get_library("geopandas")
        
        lon_col = params.get('longitude_column', 'longitude')
        lat_col = params.get('latitude_column', 'latitude')
        
        geometry = geopandas.points_from_xy(data[lon_col], data[lat_col])
        return geometry
    
    def dataframe_to_geodataframe(self, data: pd.DataFrame, **params) -> Any:
        """Convert DataFrame to GeoDataFrame with fallback."""
        if self.is_library_available("geopandas"):
            return self.create_geodataframe(data, **params)
        else:
            # Fallback: Add coordinate-based clustering
            return self._coordinate_clustering_fallback(data, **params)
    
    # ========================================================================
    # Spatial Operations 
    # ========================================================================
    
    @requires_library("geopandas") 
    def spatial_join(self, left_gdf: Any, right_gdf: Any, **params) -> Any:
        """Perform spatial join between two GeoDataFrames."""
        how = params.get('how', 'left')
        predicate = params.get('predicate', 'intersects')
        
        return left_gdf.sjoin(right_gdf, how=how, predicate=predicate)
    
    @requires_library("geopandas")
    def buffer_geometries(self, gdf: Any, **params) -> Any:
        """Create buffer around geometries."""
        distance = params.get('distance', 1000)  # meters
        resolution = params.get('resolution', 16)
        
        gdf = gdf.copy()
        gdf['geometry'] = gdf['geometry'].buffer(distance, resolution=resolution)
        return gdf
    
    @requires_library("geopandas")
    def calculate_distances(self, gdf: Any, **params) -> Any:
        """Calculate distances between geometries."""
        target_geometry = params.get('target_geometry')
        method = params.get('method', 'centroid')
        
        if target_geometry is None:
            # Calculate pairwise distances to centroids
            centroids = gdf.geometry.centroid
            distances = centroids.distance(centroids.iloc[0])
        else:
            distances = gdf.geometry.distance(target_geometry)
        
        result_gdf = gdf.copy()
        result_gdf['distance'] = distances
        return result_gdf
    
    @requires_library("geopandas")
    def spatial_intersection(self, left_gdf: Any, right_gdf: Any, **params) -> Any:
        """Find spatial intersections between geometries."""
        return left_gdf.overlay(right_gdf, how='intersection')
    
    def spatial_clustering(self, data: Any, **params) -> Any:
        """Spatial clustering with geospatial or sklearn fallback."""
        if self.is_library_available("geopandas") and hasattr(data, 'geometry'):
            return self._geopandas_spatial_clustering(data, **params)
        else:
            return self._coordinate_clustering_fallback(data, **params)
    
    # ========================================================================
    # Coordinate Reference System Operations
    # ========================================================================
    
    @requires_library("geopandas")
    def transform_crs(self, gdf: Any, **params) -> Any:
        """Transform coordinate reference system."""
        target_crs = params.get('target_crs', 'EPSG:3857')  # Web Mercator default
        
        if gdf.crs is None:
            gdf.crs = self.default_crs
        
        return gdf.to_crs(target_crs)
    
    @requires_library("geopandas")
    def calculate_centroids(self, gdf: Any, **params) -> Any:
        """Calculate centroids of geometries."""
        result_gdf = gdf.copy()
        result_gdf['geometry'] = gdf.geometry.centroid
        return result_gdf
    
    @requires_library("geopandas") 
    def calculate_area(self, gdf: Any, **params) -> Any:
        """Calculate area of geometries."""
        units = params.get('units', 'square_meters')
        
        # Ensure we're in a projected CRS for area calculation
        if gdf.crs and gdf.crs.is_geographic:
            # Transform to appropriate UTM or local projected CRS
            projected_gdf = gdf.to_crs('+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +datum=WGS84')
            areas = projected_gdf.geometry.area
        else:
            areas = gdf.geometry.area
        
        result_gdf = gdf.copy()
        result_gdf['area'] = areas
        return result_gdf
    
    # ========================================================================
    # Raster Operations
    # ========================================================================
    
    @requires_library("rasterio")
    def read_raster(self, file_path: Union[str, Path], **params) -> Any:
        """Read raster data with memory-efficient processing."""
        rasterio = self.get_library("rasterio")
        
        chunk_size = params.get('chunk_size', 1024)
        band = params.get('band', 1)
        
        with rasterio.open(file_path) as src:
            # Read with chunking for memory efficiency
            data = src.read(band, 
                          window=rasterio.windows.Window(0, 0, chunk_size, chunk_size))
            profile = src.profile
        
        return {
            'data': data,
            'profile': profile,
            'crs': profile.get('crs'),
            'transform': profile.get('transform')
        }
    
    @requires_library("rasterio")
    def raster_stats(self, raster_data: Dict[str, Any], **params) -> Any:
        """Calculate statistics for raster data."""
        data = raster_data['data']
        
        stats = {
            'mean': np.nanmean(data),
            'std': np.nanstd(data),
            'min': np.nanmin(data),
            'max': np.nanmax(data),
            'count': np.count_nonzero(~np.isnan(data)),
            'nodata_count': np.count_nonzero(np.isnan(data))
        }
        
        return pd.DataFrame([stats])
    
    @requires_library("rasterio", "geopandas") 
    def raster_to_points(self, raster_data: Dict[str, Any], **params) -> Any:
        """Convert raster data to point GeoDataFrame."""
        geopandas = self.get_library("geopandas")
        
        data = raster_data['data']
        transform = raster_data['transform']
        crs = raster_data['crs']
        
        # Get coordinates for each pixel
        rows, cols = np.where(~np.isnan(data))
        values = data[rows, cols]
        
        # Transform pixel coordinates to geographic coordinates
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        
        # Create GeoDataFrame
        points_gdf = geopandas.GeoDataFrame({
            'value': values,
            'geometry': geopandas.points_from_xy(xs, ys)
        }, crs=crs)
        
        return points_gdf
    
    # ========================================================================
    # Visualization Functions
    # ========================================================================
    
    @requires_library("folium")
    def create_interactive_map(self, data: Any, **params) -> Any:
        """Create interactive folium map."""
        folium = self.get_library("folium")
        
        # Default map parameters
        center_lat = params.get('center_lat', data.geometry.centroid.y.mean())
        center_lon = params.get('center_lon', data.geometry.centroid.x.mean())
        zoom_start = params.get('zoom_start', 10)
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start
        )
        
        # Add geometries to map
        for idx, row in data.iterrows():
            if hasattr(row.geometry, 'coords'):  # Point geometry
                folium.Marker(
                    [row.geometry.y, row.geometry.x],
                    popup=f"Point {idx}"
                ).add_to(m)
        
        return m
    
    @requires_library("folium")
    def add_markers_to_map(self, map_obj: Any, data: Any, **params) -> Any:
        """Add markers to existing folium map."""
        popup_column = params.get('popup_column', None)
        
        for idx, row in data.iterrows():
            popup_text = row[popup_column] if popup_column else f"Point {idx}"
            
            folium.Marker(
                [row.geometry.y, row.geometry.x],
                popup=str(popup_text)
            ).add_to(map_obj)
        
        return map_obj
    
    # ========================================================================
    # Fallback Implementations
    # ========================================================================
    
    def _coordinate_clustering_fallback(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Fallback spatial clustering using sklearn when geopandas unavailable."""
        from sklearn.cluster import DBSCAN
        
        lon_col = params.get('longitude_column', 'longitude')
        lat_col = params.get('latitude_column', 'latitude')
        eps = params.get('eps', 0.01)  # degrees
        min_samples = params.get('min_samples', 5)
        
        if lon_col not in data.columns or lat_col not in data.columns:
            raise ValueError(f"Coordinate columns {lon_col}, {lat_col} required for fallback clustering")
        
        # Perform DBSCAN clustering on coordinates
        coordinates = data[[lon_col, lat_col]].values
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
        
        result = data.copy()
        result['spatial_cluster'] = clustering.labels_
        result['is_core_point'] = clustering.labels_ != -1
        
        return result
    
    def _geopandas_spatial_clustering(self, gdf: Any, **params) -> Any:
        """Spatial clustering using geopandas geometric operations."""
        eps = params.get('eps', 1000)  # meters
        min_samples = params.get('min_samples', 5)
        
        # Convert to projected CRS for distance-based clustering
        if gdf.crs and gdf.crs.is_geographic:
            gdf_projected = gdf.to_crs('+proj=aea +lat_1=29.5 +lat_2=45.5')
        else:
            gdf_projected = gdf
        
        # Extract coordinates for sklearn clustering
        centroids = gdf_projected.geometry.centroid
        coordinates = np.column_stack([centroids.x, centroids.y])
        
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
        
        result = gdf.copy()
        result['spatial_cluster'] = clustering.labels_
        return result
    
    def _handle_geospatial_error(self, 
                                function_name: str,
                                data: Any,
                                parameters: Dict[str, Any],
                                error: Exception) -> Tuple[Any, GeospatialMetadata]:
        """Handle errors in geospatial operations with intelligent fallbacks."""
        
        # Log the error
        logger.error(f"Geospatial operation {function_name} failed: {error}")
        
        # Attempt fallback strategies
        if "geopandas" in str(error) and function_name == "spatial_clustering":
            # Use sklearn fallback for spatial clustering
            fallback_result = self._coordinate_clustering_fallback(data, **parameters)
            
            metadata = GeospatialMetadata(
                library_used="sklearn_fallback",
                integration_strategy=IntegrationStrategy.FALLBACK_CHAIN,
                fallback_used=True,
                original_parameters=parameters
            )
            
            return fallback_result, metadata
        
        # If no fallback available, raise the original error
        raise CompositionError(
            f"Geospatial operation {function_name} failed: {error}",
            error_type="geospatial_operation_failed"
        )


if __name__ == "__main__":
    # Example usage
    adapter = GeospatialAdapter()
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'name': ['Location A', 'Location B', 'Location C'],
        'longitude': [-74.0060, -118.2437, -87.6298],
        'latitude': [40.7128, 34.0522, 41.8781]
    })
    
    try:
        # Create GeoDataFrame
        result, metadata = adapter.adapt_function_call(
            'create_geodataframe', 
            sample_data, 
            {'longitude_column': 'longitude', 'latitude_column': 'latitude'}
        )
        print(f"Created GeoDataFrame with {len(result)} points")
        print(f"Library used: {metadata.library_used}")
        
    except Exception as e:
        print(f"Geospatial adapter test failed: {e}")