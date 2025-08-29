"""Tests for analytical format support (Parquet, Feather, Arrow) in localdata-mcp."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from localdata_mcp import DatabaseManager


class TestAnalyticalFormats:
    """Test analytical format support (Parquet, Feather, Arrow)"""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh DatabaseManager instance for testing."""
        return DatabaseManager()
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing analytical formats."""
        return pd.DataFrame({
            'id': range(1000),
            'name': [f'Item_{i}' for i in range(1000)],
            'value': [i * 1.5 for i in range(1000)],
            'category': [f'Cat_{i % 5}' for i in range(1000)],
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1H'),
            'is_active': [i % 2 == 0 for i in range(1000)]
        })
    
    @patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True)
    def test_parquet_file_connection(self, manager, sample_dataframe):
        """Test connecting to Parquet files."""
        # Create temporary Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save DataFrame as Parquet
            sample_dataframe.to_parquet(temp_path, engine='pyarrow')
            
            # Connect to Parquet file
            result = manager.connect_database("parquet_test", "parquet", temp_path)
            assert "Successfully connected" in result
            assert "parquet_test" in manager.connections
            
            # Test querying the data
            query_result = manager.execute_query("parquet_test", "SELECT COUNT(*) as total FROM data_table")
            assert "1000" in query_result
            
            # Test complex query
            query_result = manager.execute_query("parquet_test", "SELECT category, COUNT(*) as count FROM data_table GROUP BY category")
            assert "Cat_0" in query_result
            assert "200" in query_result  # Each category should have 200 items
            
            # Clean up
            manager.disconnect_database("parquet_test")
        
        except ImportError:
            pytest.skip("pyarrow not available for Parquet testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True)
    def test_feather_file_connection(self, manager, sample_dataframe):
        """Test connecting to Feather files."""
        # Create temporary Feather file
        with tempfile.NamedTemporaryFile(suffix='.feather', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save DataFrame as Feather
            sample_dataframe.to_feather(temp_path)
            
            # Connect to Feather file
            result = manager.connect_database("feather_test", "feather", temp_path)
            assert "Successfully connected" in result
            assert "feather_test" in manager.connections
            
            # Test querying the data
            query_result = manager.execute_query("feather_test", "SELECT COUNT(*) as total FROM data_table")
            assert "1000" in query_result
            
            # Test date filtering
            query_result = manager.execute_query("feather_test", "SELECT COUNT(*) as count FROM data_table WHERE timestamp >= '2023-01-01'")
            assert "1000" in query_result
            
            # Clean up
            manager.disconnect_database("feather_test")
        
        except ImportError:
            pytest.skip("pyarrow not available for Feather testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True) 
    def test_arrow_file_connection(self, manager, sample_dataframe):
        """Test connecting to Arrow IPC files."""
        # Create temporary Arrow file  
        with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save DataFrame as Arrow (using Feather format which is Arrow IPC)
            sample_dataframe.to_feather(temp_path)
            
            # Connect to Arrow file
            result = manager.connect_database("arrow_test", "arrow", temp_path)
            assert "Successfully connected" in result
            assert "arrow_test" in manager.connections
            
            # Test querying the data
            query_result = manager.execute_query("arrow_test", "SELECT COUNT(*) as total FROM data_table")
            assert "1000" in query_result
            
            # Test aggregation query
            query_result = manager.execute_query("arrow_test", "SELECT AVG(value) as avg_value FROM data_table")
            # Average of 0*1.5 to 999*1.5 should be around 748.5
            assert "748" in query_result or "749" in query_result
            
            # Clean up  
            manager.disconnect_database("arrow_test")
        
        except ImportError:
            pytest.skip("pyarrow not available for Arrow testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_pyarrow_missing_parquet(self, manager):
        """Test Parquet file handling when pyarrow is not available."""
        with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', False):
            with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
                # Try to load as Parquet without pyarrow
                with pytest.raises(ValueError, match="pyarrow library is required for Parquet"):
                    manager._create_engine_from_file(temp_file.name, "parquet")
    
    def test_pyarrow_missing_feather(self, manager):
        """Test Feather file handling when pyarrow is not available."""
        with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', False):
            with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
                # Try to load as Feather without pyarrow
                with pytest.raises(ValueError, match="pyarrow library is required for Feather"):
                    manager._create_engine_from_file(temp_file.name, "feather")
    
    def test_pyarrow_missing_arrow(self, manager):
        """Test Arrow file handling when pyarrow is not available."""
        with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', False):
            with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
                # Try to load as Arrow without pyarrow
                with pytest.raises(ValueError, match="pyarrow library is required for Arrow"):
                    manager._create_engine_from_file(temp_file.name, "arrow")
    
    @patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True)
    def test_large_analytical_file_handling(self, manager):
        """Test handling of large analytical format files."""
        # Create larger DataFrame for testing
        large_df = pd.DataFrame({
            'id': range(50000),
            'data': [f'data_{i}' for i in range(50000)],
            'value': [i * 0.1 for i in range(50000)]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save large DataFrame as Parquet
            large_df.to_parquet(temp_path, engine='pyarrow')
            
            # Test file size detection
            is_large = manager._is_large_file(temp_path, threshold_mb=1)  # 1MB threshold
            # This file might or might not be large depending on compression
            
            # Connect to large file
            result = manager.connect_database("large_parquet", "parquet", temp_path)
            assert "Successfully connected" in result
            
            # Test querying - should handle large files appropriately
            query_result = manager.execute_query_json("large_parquet", "SELECT COUNT(*) as total FROM data_table")
            assert "50000" in query_result
            
            manager.disconnect_database("large_parquet")
        
        except ImportError:
            pytest.skip("pyarrow not available for large file testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True)
    def test_analytical_format_data_types(self, manager):
        """Test that analytical formats preserve data types correctly."""
        # Create DataFrame with various data types
        typed_df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True],
            'datetime_col': pd.date_range('2023-01-01', periods=5)
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save and load via Parquet
            typed_df.to_parquet(temp_path, engine='pyarrow')
            
            result = manager.connect_database("typed_parquet", "parquet", temp_path)
            assert "Successfully connected" in result
            
            # Test that we can query with type-specific operations
            query_result = manager.execute_query("typed_parquet", "SELECT AVG(float_col) as avg_float FROM data_table")
            assert "3.3" in query_result
            
            # Test boolean operations
            query_result = manager.execute_query("typed_parquet", "SELECT COUNT(*) as true_count FROM data_table WHERE bool_col = 1")
            assert "3" in query_result  # Should be 3 True values
            
            manager.disconnect_database("typed_parquet")
        
        except ImportError:
            pytest.skip("pyarrow not available for data type testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True)
    def test_analytical_format_performance(self, manager, sample_dataframe):
        """Test performance characteristics of analytical formats."""
        parquet_path = None
        feather_path = None
        
        try:
            # Create temporary files for different formats
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as pf:
                parquet_path = pf.name
            with tempfile.NamedTemporaryFile(suffix='.feather', delete=False) as ff:
                feather_path = ff.name
            
            # Save in both formats
            sample_dataframe.to_parquet(parquet_path, engine='pyarrow')
            sample_dataframe.to_feather(feather_path)
            
            # Test connection times and query performance
            import time
            
            # Test Parquet
            start_time = time.time()
            result = manager.connect_database("perf_parquet", "parquet", parquet_path)
            parquet_connect_time = time.time() - start_time
            assert "Successfully connected" in result
            
            start_time = time.time()  
            query_result = manager.execute_query("perf_parquet", "SELECT COUNT(*) FROM data_table")
            parquet_query_time = time.time() - start_time
            assert "1000" in query_result
            
            # Test Feather
            start_time = time.time()
            result = manager.connect_database("perf_feather", "feather", feather_path)
            feather_connect_time = time.time() - start_time
            assert "Successfully connected" in result
            
            start_time = time.time()
            query_result = manager.execute_query("perf_feather", "SELECT COUNT(*) FROM data_table")  
            feather_query_time = time.time() - start_time
            assert "1000" in query_result
            
            # Both should complete reasonably quickly (less than 5 seconds each)
            assert parquet_connect_time < 5.0
            assert parquet_query_time < 5.0
            assert feather_connect_time < 5.0
            assert feather_query_time < 5.0
            
            # Clean up connections
            manager.disconnect_database("perf_parquet")
            manager.disconnect_database("perf_feather")
        
        except ImportError:
            pytest.skip("pyarrow not available for performance testing")
        finally:
            # Clean up files
            for path in [parquet_path, feather_path]:
                if path and os.path.exists(path):
                    os.unlink(path)
    
    @patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True)
    def test_corrupted_analytical_files(self, manager):
        """Test handling of corrupted analytical format files."""
        # Create corrupted Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False, mode='w') as temp_file:
            temp_file.write("This is not valid Parquet data")
            temp_path = temp_file.name
        
        try:
            # Attempt to connect to corrupted file
            result = manager.connect_database("corrupted_parquet", "parquet", temp_path)
            assert "Failed to connect" in result or "error" in result.lower()
        
        except ImportError:
            pytest.skip("pyarrow not available for corruption testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAnalyticalFormatIntegration:
    """Test integration of analytical formats with MCP tools."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh DatabaseManager instance for testing."""
        return DatabaseManager()
    
    @patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True)
    def test_describe_database_analytical(self, manager):
        """Test describe_database tool with analytical formats."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10.5, 20.7, 30.2]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            test_df.to_parquet(temp_path, engine='pyarrow')
            
            # Connect and describe
            manager.connect_database("test_parquet", "parquet", temp_path)
            description = manager.describe_database("test_parquet")
            
            # Should contain database metadata
            assert "test_parquet" in description
            assert "data_table" in description
            assert "columns" in description
            
            manager.disconnect_database("test_parquet")
        
        except ImportError:
            pytest.skip("pyarrow not available for integration testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True)
    def test_query_buffering_analytical(self, manager):
        """Test query buffering with large analytical format results.""" 
        # Create DataFrame with >100 rows to trigger buffering
        large_df = pd.DataFrame({
            'id': range(150),
            'data': [f'item_{i}' for i in range(150)]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            large_df.to_parquet(temp_path, engine='pyarrow')
            
            manager.connect_database("buffer_test", "parquet", temp_path)
            
            # Query should trigger buffering for large results
            result = manager.execute_query_json("buffer_test", "SELECT * FROM data_table")
            
            # Should contain metadata about buffering
            assert "query_id" in result
            assert "total_rows" in result
            assert "150" in result  # Total rows
            
            manager.disconnect_database("buffer_test")
        
        except ImportError:
            pytest.skip("pyarrow not available for buffering testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])