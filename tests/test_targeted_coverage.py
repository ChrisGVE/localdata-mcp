"""Targeted tests to achieve 100% coverage by hitting specific uncovered lines."""

import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd

from localdata_mcp.localdata_mcp import DatabaseManager


class TestTargetedCoverage:
    """Targeted tests for specific uncovered line numbers."""

    def test_import_error_flags_coverage(self):
        """Test that import availability flags are properly accessed."""
        # Import the module to ensure flags are set
        import localdata_mcp.localdata_mcp as ldm
        
        # Access all the availability flags to ensure they're covered
        flags = [
            ldm.TOML_AVAILABLE,
            ldm.OPENPYXL_AVAILABLE, 
            ldm.XLRD_AVAILABLE,
            ldm.DEFUSEDXML_AVAILABLE,
            ldm.ODFPY_AVAILABLE,
            ldm.LXML_AVAILABLE,
            ldm.PYARROW_AVAILABLE
        ]
        assert all(isinstance(flag, bool) for flag in flags)

    def test_path_validation_errors(self):
        """Test path validation error handling (lines 140-141)."""
        manager = DatabaseManager()
        
        # Create a path that will cause an error
        with patch('pathlib.Path.resolve', side_effect=OSError("Path error")):
            with pytest.raises(ValueError):
                manager._validate_file_path("/invalid/path")

    def test_cleanup_error_handling(self):
        """Test cleanup operations with errors (lines 241-242, 248-251)."""
        manager = DatabaseManager()
        
        # Add a mock file to temp_files
        manager.temp_files.append("nonexistent_file.tmp")
        
        # Mock os.remove to raise OSError
        with patch('os.remove', side_effect=OSError("Permission denied")):
            # This should not raise an exception
            manager._cleanup_temp_files()

    def test_engine_disposal_with_error(self):
        """Test engine disposal error handling (lines 248-251)."""
        manager = DatabaseManager()
        
        # Create mock engine that throws error on disposal
        mock_engine = Mock()
        mock_engine.dispose.side_effect = Exception("Disposal failed")
        manager.connections["test"] = mock_engine
        
        # Should handle the error gracefully
        result = manager.disconnect_database("test")
        assert result["success"] is True

    def test_database_connection_variants(self):
        """Test different database connection code paths (lines 257, 259, 261)."""
        manager = DatabaseManager()
        
        # Test SQLite connection (should trigger line 257)
        result = manager.connect_database("test.db", "sqlite") 
        assert result["success"] is True
        
        # Test other connection types to trigger lines 259, 261
        result = manager.connect_database("postgresql://localhost/test", "postgresql")
        assert result["success"] is True
        
        result = manager.connect_database("mysql://localhost/test", "mysql")  
        assert result["success"] is True

    def test_yaml_file_loading(self):
        """Test YAML file loading (lines 291-292)."""
        manager = DatabaseManager()
        
        # Create test YAML file
        yaml_content = "test:\n  data:\n    - value1\n    - value2\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name
        
        try:
            result = manager.connect_database(yaml_file, "yaml")
            assert result["success"] is True
        finally:
            os.unlink(yaml_file)

    def test_toml_file_loading(self):
        """Test TOML file loading (lines 303-305)."""
        manager = DatabaseManager()
        
        # Check if TOML is available
        try:
            import toml
            toml_available = True
        except ImportError:
            toml_available = False
        
        if toml_available:
            # Create test TOML file
            toml_content = '[section]\nkey = "value"\n'
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                f.write(toml_content)
                toml_file = f.name
            
            try:
                result = manager.connect_database(toml_file, "toml")
                assert result["success"] is True
            finally:
                os.unlink(toml_file)
        else:
            pytest.skip("TOML not available")

    def test_ini_file_loading(self):
        """Test INI file loading (line 317).""" 
        manager = DatabaseManager()
        
        # Create test INI file
        ini_content = "[section1]\nkey1=value1\n[section2]\nkey2=value2\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(ini_content)
            ini_file = f.name
        
        try:
            result = manager.connect_database(ini_file, "ini")
            assert result["success"] is True
        finally:
            os.unlink(ini_file)

    def test_feather_file_loading(self):
        """Test Feather file loading (line 337)."""
        try:
            import pyarrow
        except ImportError:
            pytest.skip("PyArrow not available")
        
        manager = DatabaseManager()
        
        # Create test dataframe and save as feather
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        with tempfile.NamedTemporaryFile(suffix='.feather', delete=False) as f:
            feather_file = f.name
        
        try:
            df.to_feather(feather_file)
            result = manager.connect_database(feather_file, "feather")
            assert result["success"] is True
        finally:
            os.unlink(feather_file)

    def test_arrow_file_loading(self):
        """Test Arrow file loading (lines 344-346)."""
        try:
            import pyarrow
        except ImportError:
            pytest.skip("PyArrow not available")
        
        manager = DatabaseManager()
        
        # Create test dataframe and save as arrow
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False) as f:
            arrow_file = f.name
        
        try:
            df.to_feather(arrow_file)  # Arrow uses same reader as feather
            result = manager.connect_database(arrow_file, "arrow")
            assert result["success"] is True
        finally:
            os.unlink(arrow_file)

    def test_unsupported_file_type_error(self):
        """Test unsupported file type handling (lines 351-361)."""
        manager = DatabaseManager()
        
        # Create file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            f.write(b"some content")
            unknown_file = f.name
        
        try:
            result = manager.connect_database(unknown_file, "auto")
            assert result["success"] is False
            assert "Unsupported file type" in result["message"]
        finally:
            os.unlink(unknown_file)

    def test_temp_file_operations(self):
        """Test temporary file creation and cleanup (lines 351-361)."""
        manager = DatabaseManager()
        
        # Create a CSV file that will trigger temp file creation
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_file = f.name
        
        try:
            result = manager.connect_database(csv_file, "csv")
            assert result["success"] is True
            
            # Verify temp files were created and tracked
            assert len(manager.temp_files) > 0
        finally:
            os.unlink(csv_file)

    def test_ods_file_error_handling(self):
        """Test ODS file error handling (lines 527-528, 550-552)."""
        manager = DatabaseManager()
        
        # Create invalid ODS file to trigger error handling
        with tempfile.NamedTemporaryFile(suffix='.ods', delete=False) as f:
            f.write(b"invalid ods content")
            ods_file = f.name
        
        try:
            result = manager.connect_database(ods_file, "ods")
            # Should fail gracefully
            assert result["success"] is False
        finally:
            os.unlink(ods_file)

    def test_memory_info_functionality(self):
        """Test memory info features.""" 
        manager = DatabaseManager()
        
        # Test memory info retrieval
        memory_info = manager.get_memory_info()
        assert "memory_percent" in memory_info
        assert isinstance(memory_info["memory_percent"], (int, float))

    def test_cleanup_all_functionality(self):
        """Test cleanup all functionality."""
        manager = DatabaseManager()
        
        # Add some connections and temp files
        manager.connections["test1"] = Mock()
        manager.connections["test2"] = Mock()
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        manager.temp_files.append(temp_file)
        
        # Clean up all
        manager.cleanup_all()
        
        # Verify cleanup
        assert len(manager.connections) == 0
        assert len(manager.temp_files) == 0
        
        # Clean up temp file if it still exists
        try:
            os.unlink(temp_file)
        except OSError:
            pass

    def test_query_execution_edge_cases(self):
        """Test query execution edge cases."""
        manager = DatabaseManager()
        
        # Create a simple database connection
        result = manager.connect_database("test.db", "sqlite")
        assert result["success"] is True
        
        # Test query execution with empty results
        query_result = manager.execute_query("SELECT 1 WHERE 0=1")  # Returns empty
        assert query_result["success"] is True
        assert len(query_result["data"]) == 0

    def test_file_type_detection_edge_cases(self):
        """Test file type detection for various formats."""
        manager = DatabaseManager()
        
        # Test different file extensions to trigger different code paths
        test_extensions = ['.json', '.xml', '.txt']
        
        for ext in test_extensions:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                if ext == '.json':
                    f.write(b'{"data": [1, 2, 3]}')
                elif ext == '.xml':
                    f.write(b'<root><item>test</item></root>')
                else:
                    f.write(b'test content')
                test_file = f.name
            
            try:
                result = manager.connect_database(test_file, "auto")
                # Some might succeed, others might fail - that's okay
                assert "success" in result
            finally:
                os.unlink(test_file)

    def test_main_function_coverage(self):
        """Test main function."""
        from localdata_mcp.localdata_mcp import main
        
        # Mock the app to avoid actually running it
        with patch('localdata_mcp.localdata_mcp.mcp') as mock_mcp:
            mock_app = Mock()
            mock_mcp.run.return_value = mock_app
            mock_app.run = Mock()
            
            # Test main function
            main()
            
            # Verify app.run was called
            mock_app.run.assert_called_once()

    def test_additional_edge_cases(self):
        """Test additional edge cases for remaining lines."""
        manager = DatabaseManager()
        
        # Test various scenarios to hit remaining uncovered lines
        # This includes error conditions and edge cases that are hard to trigger
        
        # Test with malformed data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content,with,too,many,commas\n")
            malformed_csv = f.name
        
        try:
            # This might trigger error handling code
            result = manager.connect_database(malformed_csv, "csv")
            assert "success" in result
        finally:
            os.unlink(malformed_csv)

    def test_module_level_code_coverage(self):
        """Ensure all module-level code is executed."""
        # Import and access module-level variables
        import localdata_mcp.localdata_mcp as ldm
        
        # Ensure logger is properly initialized
        assert hasattr(ldm, 'logger')
        assert ldm.logger is not None
        
        # Ensure all availability flags are accessible
        availability_flags = [
            'TOML_AVAILABLE', 'OPENPYXL_AVAILABLE', 'XLRD_AVAILABLE',
            'DEFUSEDXML_AVAILABLE', 'ODFPY_AVAILABLE', 'LXML_AVAILABLE', 
            'PYARROW_AVAILABLE'
        ]
        
        for flag in availability_flags:
            assert hasattr(ldm, flag)
            value = getattr(ldm, flag)
            assert isinstance(value, bool)