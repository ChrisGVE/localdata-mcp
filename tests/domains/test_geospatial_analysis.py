"""
Tests for geospatial analysis domain functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from localdata_mcp.domains.geospatial_analysis import (
    GeospatialDependencyChecker,
    SpatialPoint,
    SpatialDataFrame,
    SpatialDistanceCalculator,
    SpatialDistanceTransformer,
    SpatialGeometryTransformer,
    SpatialAutocorrelationTransformer,
    SpatialInterpolationTransformer,
    SpatialJoinEngine,
    SpatialOverlayEngine,
    SpatialNetwork,
    NetworkRouter,
    AccessibilityAnalyzer,
    GeospatialAnalysisPipeline,
    analyze_spatial_autocorrelation,
    perform_spatial_clustering,
    calculate_spatial_distance,
    GeospatialLibrary,
    SpatialJoinType,
    OverlayOperation,
    NetworkAnalysisType
)


class TestGeospatialDependencyChecker:
    """Test geospatial dependency management."""
    
    def test_initialization(self):
        """Test dependency checker initialization."""
        checker = GeospatialDependencyChecker()
        assert checker is not None
        
    def test_fit_transform(self):
        """Test basic fit_transform operation."""
        checker = GeospatialDependencyChecker()
        
        # Test with DataFrame
        data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6],
            'value': [10, 20, 30]
        })
        
        checker.fit(data)
        result = checker.transform(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)


class TestSpatialDataStructures:
    """Test spatial data structure classes."""
    
    def test_spatial_point(self):
        """Test SpatialPoint creation and properties."""
        point = SpatialPoint(x=1.0, y=2.0, properties={'name': 'test'})
        
        assert point.x == 1.0
        assert point.y == 2.0
        assert point.properties['name'] == 'test'
        
    def test_spatial_dataframe(self):
        """Test SpatialDataFrame wrapper."""
        data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6],
            'value': [10, 20, 30]
        })
        
        spatial_df = SpatialDataFrame(data, geometry_column='geometry')
        assert spatial_df.data is not None
        assert spatial_df.geometry_column == 'geometry'


class TestSpatialDistanceCalculator:
    """Test spatial distance calculation methods."""
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        calc = SpatialDistanceCalculator()
        
        point1 = (0, 0)
        point2 = (3, 4)
        
        distance = calc.euclidean_distance(point1, point2)
        assert distance == 5.0  # 3-4-5 triangle
        
    def test_haversine_distance(self):
        """Test Haversine distance calculation."""
        calc = SpatialDistanceCalculator()
        
        # London to Paris (approximate)
        london = (51.5074, -0.1278)
        paris = (48.8566, 2.3522)
        
        distance = calc.haversine_distance(london, paris)
        assert distance > 300  # Should be ~344 km
        assert distance < 400
        
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        calc = SpatialDistanceCalculator()
        
        point1 = (0, 0)
        point2 = (3, 4)
        
        distance = calc.manhattan_distance(point1, point2)
        assert distance == 7.0  # |3-0| + |4-0| = 7


class TestSpatialTransformers:
    """Test sklearn-compatible spatial transformers."""
    
    def test_distance_transformer(self):
        """Test spatial distance transformer."""
        data = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0, 1, 2],
            'value': [10, 20, 30]
        })
        
        transformer = SpatialDistanceTransformer(
            coordinate_columns=['x', 'y'],
            distance_metrics=['euclidean']
        )
        
        transformer.fit(data)
        result = transformer.transform(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        
    def test_geometry_transformer(self):
        """Test spatial geometry transformer."""
        data = pd.DataFrame({
            'x': [0, 1, 2, 0],
            'y': [0, 0, 1, 1],
            'value': [10, 20, 30, 40]
        })
        
        transformer = SpatialGeometryTransformer(
            coordinate_columns=['x', 'y'],
            operations=['bounding_box', 'convex_hull']
        )
        
        transformer.fit(data)
        result = transformer.transform(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)


class TestSpatialJoinEngine:
    """Test spatial join operations."""
    
    def test_spatial_join_basic(self):
        """Test basic spatial join functionality."""
        engine = SpatialJoinEngine()
        
        left_data = pd.DataFrame({
            'geometry': [(0, 0), (1, 1), (2, 2)],
            'left_value': [10, 20, 30]
        })
        
        right_data = pd.DataFrame({
            'geometry': [(0.1, 0.1), (1.1, 1.1), (3, 3)],
            'right_value': [100, 200, 300]
        })
        
        # Test with basic intersection (will fall back to basic operations)
        result = engine.spatial_join(
            left_data, right_data,
            left_geometry_col='geometry',
            right_geometry_col='geometry',
            join_type=SpatialJoinType.INTERSECTS
        )
        
        assert result.joined_data is not None
        assert isinstance(result.match_counts, dict)


class TestSpatialNetwork:
    """Test spatial network analysis."""
    
    def test_network_creation_from_points(self):
        """Test network creation from point coordinates."""
        network = SpatialNetwork()
        
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        # This will fail without NetworkX, but we can test the structure
        with pytest.raises(ValueError, match="NetworkX is required"):
            network.create_network_from_points(points, connection_method='knn')
            
    def test_network_creation_from_edges(self):
        """Test network creation from explicit edges."""
        network = SpatialNetwork()
        
        nodes = [
            {'id': 'A', 'x': 0, 'y': 0},
            {'id': 'B', 'x': 1, 'y': 0},
            {'id': 'C', 'x': 1, 'y': 1}
        ]
        
        edges = [
            {'source': 'A', 'target': 'B', 'weight': 1.0},
            {'source': 'B', 'target': 'C', 'weight': 1.0}
        ]
        
        # This will fail without NetworkX, but we can test the structure
        with pytest.raises(ValueError, match="NetworkX is required"):
            network.create_network_from_edges(nodes, edges)


class TestHighLevelFunctions:
    """Test high-level geospatial analysis functions."""
    
    def test_analyze_spatial_autocorrelation(self):
        """Test spatial autocorrelation analysis."""
        data = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 0, 0, 0, 0],
            'value': [10, 15, 20, 25, 30]  # Linear trend
        })
        
        result = analyze_spatial_autocorrelation(
            data, 
            value_column='value',
            coordinate_columns=['x', 'y'],
            method='moran'
        )
        
        assert isinstance(result, dict)
        assert 'method' in result
        assert result['method'] == 'moran'
        
    def test_perform_spatial_clustering(self):
        """Test spatial clustering analysis."""
        data = pd.DataFrame({
            'x': [0, 1, 10, 11, 20, 21],
            'y': [0, 1, 10, 11, 20, 21],
            'value': [100, 110, 50, 60, 200, 210]  # Three clusters
        })
        
        result = perform_spatial_clustering(
            data,
            coordinate_columns=['x', 'y'],
            method='hotspot'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'cluster_id' in result.columns
        assert 'is_hotspot' in result.columns
        assert len(result) == len(data)
        
    def test_calculate_spatial_distance(self):
        """Test spatial distance calculation."""
        data1 = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0, 0, 0]
        })
        
        data2 = pd.DataFrame({
            'x': [0, 1],
            'y': [1, 1]
        })
        
        # Test distance matrix between two datasets
        result = calculate_spatial_distance(
            data1, data2,
            coordinate_columns=['x', 'y'],
            distance_type='euclidean',
            output_format='matrix'
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)  # 3 points in data1, 2 points in data2


class TestGeospatialAnalysisPipeline:
    """Test unified geospatial analysis pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline creation and configuration."""
        pipeline = GeospatialAnalysisPipeline(
            coordinate_columns=['x', 'y'],
            value_column='value'
        )
        
        assert pipeline.coordinate_columns == ['x', 'y']
        assert pipeline.value_column == 'value'
        assert len(pipeline.transformers) == 0
        
    def test_pipeline_add_analysis_steps(self):
        """Test adding analysis steps to pipeline."""
        pipeline = GeospatialAnalysisPipeline(
            coordinate_columns=['x', 'y'],
            value_column='value'
        )
        
        pipeline.add_distance_analysis(['euclidean'])
        pipeline.add_geometry_analysis(['bounding_box'])
        pipeline.add_autocorrelation_analysis('moran')
        
        assert len(pipeline.transformers) == 3
        assert any(name == 'distance' for name, _ in pipeline.transformers)
        assert any(name == 'geometry' for name, _ in pipeline.transformers)
        assert any(name == 'autocorrelation' for name, _ in pipeline.transformers)
        
    def test_pipeline_execution(self):
        """Test pipeline execution."""
        data = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4],
            'value': [10, 20, 30, 40, 50]
        })
        
        pipeline = GeospatialAnalysisPipeline(
            coordinate_columns=['x', 'y'],
            value_column='value'
        )
        
        pipeline.add_distance_analysis(['euclidean'])
        pipeline.add_geometry_analysis(['bounding_box'])
        
        results = pipeline.fit_transform(data)
        
        assert isinstance(results, dict)
        assert len(results) == 2  # Two analysis steps
        
        summary = pipeline.get_summary()
        assert isinstance(summary, dict)
        assert 'steps_executed' in summary
        assert summary['steps_executed'] == 2


class TestGeospatialEnums:
    """Test geospatial enumeration classes."""
    
    def test_geospatial_library_enum(self):
        """Test GeospatialLibrary enum."""
        assert GeospatialLibrary.GEOPANDAS.value == "geopandas"
        assert GeospatialLibrary.SHAPELY.value == "shapely"
        assert GeospatialLibrary.PYPROJ.value == "pyproj"
        assert GeospatialLibrary.NETWORKX.value == "networkx"
        
    def test_spatial_join_type_enum(self):
        """Test SpatialJoinType enum."""
        assert SpatialJoinType.INTERSECTS.value == "intersects"
        assert SpatialJoinType.WITHIN.value == "within"
        assert SpatialJoinType.CONTAINS.value == "contains"
        assert SpatialJoinType.NEAREST.value == "nearest"
        
    def test_overlay_operation_enum(self):
        """Test OverlayOperation enum."""
        assert OverlayOperation.INTERSECTION.value == "intersection"
        assert OverlayOperation.UNION.value == "union"
        assert OverlayOperation.DIFFERENCE.value == "difference"
        
    def test_network_analysis_type_enum(self):
        """Test NetworkAnalysisType enum."""
        assert NetworkAnalysisType.SHORTEST_PATH.value == "shortest_path"
        assert NetworkAnalysisType.ACCESSIBILITY.value == "accessibility"
        assert NetworkAnalysisType.ISOCHRONE.value == "isochrone"


class TestIntegrationWithOtherDomains:
    """Test integration with other analysis domains."""
    
    def test_pipeline_with_statistical_data(self):
        """Test geospatial pipeline with statistical analysis data."""
        # Create spatial data with statistical properties
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.normal(0, 10, 100),
            'y': np.random.normal(0, 10, 100),
            'value': np.random.normal(50, 15, 100)
        })
        
        pipeline = GeospatialAnalysisPipeline(
            coordinate_columns=['x', 'y'],
            value_column='value'
        )
        
        pipeline.add_distance_analysis(['euclidean'])
        pipeline.add_autocorrelation_analysis('moran')
        
        results = pipeline.fit_transform(data)
        
        assert len(results) >= 1  # At least distance analysis should work
        assert isinstance(results, dict)
        
        # Verify that results can be used for further analysis
        for name, result in results.items():
            if isinstance(result, pd.DataFrame) and not result.empty:
                assert len(result) == len(data)