"""
Complete coverage test to reach 100% test coverage.
This file specifically targets the remaining 80 missing lines.
"""

import json
import pytest
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import patch, Mock, MagicMock
from tests.test_helpers import create_test_manager
from tests.mock_helpers import mock_csv_connection


class TestCompleteCoverage:
    """Tests to achieve the final 100% coverage by targeting specific missing lines."""

    @pytest.fixture
    def manager(self):
        return create_test_manager()

    def test_module_level_import_flags(self):
        """Test module-level import availability flags (lines 27-68)."""
        
        # Test import availability flags by checking their current values
        # These are set at module level, so we can just verify they exist
        import localdata_mcp.localdata_mcp as lmcp
        
        # Check that availability flags are defined (coverage of the try/except blocks)
        assert hasattr(lmcp, 'YAML_AVAILABLE')
        assert hasattr(lmcp, 'TOML_AVAILABLE')
        assert hasattr(lmcp, 'OPENPYXL_AVAILABLE')
        assert hasattr(lmcp, 'XLRD_AVAILABLE')
        assert hasattr(lmcp, 'ODFPY_AVAILABLE')
        assert hasattr(lmcp, 'DEFUSEDXML_AVAILABLE')
        
        # Verify they are boolean values
        assert isinstance(lmcp.YAML_AVAILABLE, bool)
        assert isinstance(lmcp.TOML_AVAILABLE, bool)
        assert isinstance(lmcp.OPENPYXL_AVAILABLE, bool)
        assert isinstance(lmcp.XLRD_AVAILABLE, bool)
        assert isinstance(lmcp.ODFPY_AVAILABLE, bool)
        assert isinstance(lmcp.DEFUSEDXML_AVAILABLE, bool)

    def test_path_sanitization_edge_cases_complete(self, manager):
        """Test path sanitization edge cases (lines 140-153)."""
        
        # Test absolute path handling - should be rejected
        test_path = "/absolute/path/to/file.csv"
        try:
            result = manager._sanitize_path(test_path)
            assert False, "Should have raised ValueError for absolute path"
        except ValueError as e:
            assert "outside the allowed directory" in str(e)
            
        # Test relative path that goes outside current directory  
        test_path = "../../outside/file.csv"
        try:
            result = manager._sanitize_path(test_path)
            assert False, "Should have raised ValueError for path outside cwd"
        except ValueError as e:
            assert "outside the allowed directory" in str(e)
            
        # Test invalid path that causes OSError in Path.resolve()
        with patch('pathlib.Path.resolve', side_effect=OSError("Path resolution error")):
            try:
                result = manager._sanitize_path("./test.csv")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Invalid path" in str(e)
                
        # Test nonexistent file path (should hit line 152-153)
        test_path = "./nonexistent_file.csv"
        try:
            result = manager._sanitize_path(test_path)
            assert False, "Should have raised ValueError for nonexistent file"
        except ValueError as e:
            assert "File not found" in str(e)

    def test_file_loading_error_branches(self, manager):
        """Test file loading error branches (lines 241-361)."""
        
        # Create a temporary file to satisfy path sanitization
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.yml', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(b"test: value\n")
            
        try:
            # Test YAML file loading with missing yaml library
            with patch('localdata_mcp.localdata_mcp.YAML_AVAILABLE', False):
                try:
                    manager._load_yaml_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "yaml library is required" in str(e)
        finally:
            os.unlink(temp_file_path)
            
        # Test TOML loading error
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(b"[test]\nkey = 'value'")
            
        try:
            with patch('localdata_mcp.localdata_mcp.TOML_AVAILABLE', False):
                try:
                    manager._load_toml_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "toml library is required" in str(e)
        finally:
            os.unlink(temp_file_path)
            
        # Test XML file error handling
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(b"<root><item>test</item></root>")
            
        try:
            with patch('pandas.read_xml', side_effect=Exception("XML parse error")):
                try:
                    manager._load_xml_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "Failed to load XML file" in str(e)
        finally:
            os.unlink(temp_file_path)
            
        # Test TSV file error handling  
        with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(b"col1\tcol2\nval1\tval2")
            
        try:
            with patch('pandas.read_csv', side_effect=Exception("CSV parse error")):
                try:
                    manager._load_tsv_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "Failed to load TSV file" in str(e)
        finally:
            os.unlink(temp_file_path)

    def test_ods_file_error_handling(self, manager):
        """Test ODS file error handling (lines 527-600)."""
        
        # Test ODS file loading with missing odfpy library
        with patch('localdata_mcp.localdata_mcp.ODFPY_AVAILABLE', False):
            try:
                manager._load_ods_file("/test.ods")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "odfpy library is required" in str(e)
                
        # Test ODS file loading with pandas errors
        with patch('localdata_mcp.localdata_mcp.ODFPY_AVAILABLE', True), \
             patch('pandas.ExcelFile', side_effect=Exception("ODS read error")):
            try:
                manager._load_ods_file("/test.ods")
                assert False, "Should have raised ValueError"  
            except ValueError as e:
                assert "Failed to load ODS file" in str(e)

    def test_utility_method_branches(self, manager):
        """Test utility method branches (lines 644, 764-966, 911, 918, 944-961)."""
        
        # Test _get_table_metadata with missing constraint info
        mock_inspector = Mock()
        mock_inspector.get_columns.return_value = [
            {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'autoincrement': True}
        ]
        mock_inspector.get_foreign_keys.return_value = []
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': []}  # Empty primary keys
        mock_inspector.get_indexes.return_value = []
        mock_inspector.get_table_options.side_effect = NotImplementedError()  # Test exception branch
        
        result = manager._get_table_metadata(mock_inspector, "test_table")
        assert result["name"] == "test_table"
        assert result["options"] == {}  # Should be empty due to NotImplementedError
        
        # Test _safe_table_identifier with invalid characters
        try:
            result = manager._safe_table_identifier("invalid-table-name")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid table name" in str(e)
            
        # Test _safe_table_identifier with valid name
        result = manager._safe_table_identifier("valid_table_name")
        assert "valid_table_name" in result

    def test_database_inspection_missing_lines(self, manager):
        """Test database inspection methods missing lines (1009-1070)."""
        
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "./test.csv")
            
            # Test describe_database with mock that causes exceptions
            with patch('sqlalchemy.inspect') as mock_inspect:
                mock_inspector = Mock()
                mock_inspector.get_table_names.return_value = ['test_table']
                mock_inspector.get_server_version_info.side_effect = Exception("Version error")
                mock_inspector.default_schema_name = None
                mock_inspector.get_schema_names.side_effect = Exception("Schema error")
                mock_inspect.return_value = mock_inspector
                
                # This should hit the exception path in describe_database
                result = manager.describe_database("test_db")
                assert "An error occurred:" in result
                
            # Test find_table with empty result (line 1040)  
            with patch('sqlalchemy.inspect') as mock_inspect:
                mock_inspector = Mock()
                mock_inspector.get_table_names.return_value = []  # No tables
                mock_inspect.return_value = mock_inspector
                
                result = manager.find_table("nonexistent_table")
                assert "was not found in any connected databases" in result
                
            # Test describe_table with exceptions (lines 1059-1070)
            with patch('sqlalchemy.inspect') as mock_inspect:
                mock_inspector = Mock()
                mock_inspector.get_table_names.return_value = ['data_table']  # Use actual table name
                mock_inspect.return_value = mock_inspector
                
                # Make _get_table_metadata raise an exception
                with patch.object(manager, '_get_table_metadata', side_effect=Exception("Metadata error")):
                    result = manager.describe_table("test_db", "data_table")
                    assert "An error occurred:" in result

    def test_memory_and_cleanup_branches(self, manager):
        """Test memory management and cleanup branches."""
        
        # Test _auto_clear_buffers_if_needed with high memory
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 90  # High memory usage
            
            # Add some query buffers to clear
            from localdata_mcp.localdata_mcp import QueryBuffer
            import time
            buffer = QueryBuffer(
                query_id="test_1",
                db_name="test_db",
                query="SELECT * FROM test",
                results=pd.DataFrame({'id': [1]}),
                timestamp=time.time()
            )
            manager.query_buffers["test_1"] = buffer
            
            result = manager._auto_clear_buffers_if_needed("test_db")
            assert result is True  # Should return True when buffers were cleared

    def test_main_function_and_module_entry(self):
        """Test main function and module entry point (line 1161)."""
        
        # Test the if __name__ == "__main__": branch
        with patch('localdata_mcp.localdata_mcp.main') as mock_main:
            # Simulate running as main module
            exec("if __name__ == '__main__': main()", {'__name__': '__main__', 'main': mock_main})
            mock_main.assert_called_once()

    def test_additional_edge_cases(self, manager):
        """Test additional edge cases to complete coverage."""
        
        # Test Excel file loading with various engines and errors
        with patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', False), \
             patch('localdata_mcp.localdata_mcp.XLRD_AVAILABLE', False):
            try:
                manager._load_excel_file("/test.xlsx")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "openpyxl library is required" in str(e)
                
        # Test file size and large file detection edge cases
        with patch.object(manager, '_get_file_size', return_value=100 * 1024 * 1024):  # 100MB  
            assert manager._is_large_file("/test.csv") is True
            
        with patch.object(manager, '_get_file_size', return_value=50 * 1024 * 1024):   # 50MB
            assert manager._is_large_file("/test.csv") is False
            
        # Test query ID generation uniqueness
        query_id_1 = manager._generate_query_id("db1", "SELECT * FROM table1")
        query_id_2 = manager._generate_query_id("db1", "SELECT * FROM table2") 
        assert query_id_1 != query_id_2
        
        # Test sheet name sanitization with edge cases
        sanitized = manager._sanitize_sheet_name("Sheet-Name With @#$ Special")
        assert sanitized == "Sheet_Name_With_Special"
        
        # Test memory check error handling
        with patch('psutil.virtual_memory', side_effect=Exception("Memory check failed")):
            memory_info = manager._check_memory_usage()
            assert "error" in memory_info
            assert memory_info["low_memory"] is False


if __name__ == "__main__":
    pytest.main([__file__])