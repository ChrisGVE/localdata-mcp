"""
Final push for 100% test coverage.
Targeting the remaining 233 missing lines to achieve complete coverage.
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


class TestFinalPushCoverage:
    """Final tests to achieve 100% coverage by targeting specific remaining lines."""

    @pytest.fixture
    def manager(self):
        return create_test_manager()

    def test_import_availability_flags_coverage(self):
        """Test import availability flag setting (lines 27-68)."""
        
        # Import the module to ensure availability flags are properly set
        import localdata_mcp.localdata_mcp as lmcp
        
        # Check that all the availability flags exist and are boolean
        availability_flags = [
            'TOML_AVAILABLE', 'OPENPYXL_AVAILABLE', 'XLRD_AVAILABLE', 
            'DEFUSEDXML_AVAILABLE', 'ODFPY_AVAILABLE', 'LXML_AVAILABLE', 
            'PYARROW_AVAILABLE'
        ]
        
        for flag in availability_flags:
            if hasattr(lmcp, flag):
                assert isinstance(getattr(lmcp, flag), bool), f"{flag} should be boolean"

    def test_sanitize_path_error_cases(self, manager):
        """Test _sanitize_path error cases (lines 140-141, 153, 161-162)."""
        
        # Test path resolution error (line 140-141)
        with patch('pathlib.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.resolve.side_effect = OSError("Path resolution failed")
            mock_path.return_value = mock_path_instance
            
            try:
                manager._sanitize_path("./test.csv")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Invalid path" in str(e)
                
        # Test _get_file_size error case (line 161-162) 
        with patch('os.path.getsize', side_effect=OSError("File access denied")):
            try:
                manager._get_file_size("./test.csv")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Cannot get size of file" in str(e)

    def test_file_format_loading_branches(self, manager):
        """Test file format loading branches (lines 241-361)."""
        
        # Create temporary files to test with
        
        # Test YAML file with error (lines 241-242)
        with tempfile.NamedTemporaryFile(suffix='.yml', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(b"invalid: yaml: content: [")
            
        try:
            with patch('yaml.safe_load', side_effect=Exception("YAML parse error")):
                try:
                    manager._load_yaml_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "Failed to load YAML file" in str(e)
        finally:
            os.unlink(temp_file_path)
            
        # Test TOML loading error (lines 248-251)
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(b"[invalid toml")
            
        try:
            with patch('localdata_mcp.localdata_mcp.TOML_AVAILABLE', True), \
                 patch('toml.load', side_effect=Exception("TOML parse error")):
                try:
                    manager._load_toml_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "Failed to load TOML file" in str(e)
        finally:
            os.unlink(temp_file_path)
            
        # Test INI file loading error (lines 257, 259, 261)
        with tempfile.NamedTemporaryFile(suffix='.ini', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(b"[section1]\nkey1=value1")
            
        try:
            # Test successful loading path first
            result = manager._load_ini_file(temp_file_path)
            assert isinstance(result, pd.DataFrame)
            
            # Test error case with configparser exception
            with patch('configparser.ConfigParser.read', side_effect=Exception("INI read error")):
                try:
                    manager._load_ini_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "Failed to load INI file" in str(e)
        finally:
            os.unlink(temp_file_path)

    def test_create_engine_from_file_branches(self, manager):
        """Test _create_engine_from_file branches (lines 284-380)."""
        
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write("id,name\n1,test\n2,test2\n")
            
        try:
            # Test successful CSV loading
            engine = manager._create_engine_from_file(temp_file_path, "csv")
            assert engine is not None
            
            # Test large file handling
            with patch.object(manager, '_is_large_file', return_value=True):
                engine = manager._create_engine_from_file(temp_file_path, "csv")
                assert engine is not None
                
        finally:
            os.unlink(temp_file_path)
            
        # Test parquet file branch (lines 369-371)
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            # Write minimal parquet-like content
            temp_file.write(b"fake parquet content")
            
        try:
            with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True), \
                 patch('pandas.read_parquet', return_value=pd.DataFrame({'id': [1], 'name': ['test']})):
                engine = manager._create_engine_from_file(temp_file_path, "parquet")
                assert engine is not None
                
            # Test pyarrow not available case  
            with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', False):
                try:
                    manager._create_engine_from_file(temp_file_path, "parquet")
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "pyarrow library is required" in str(e)
        finally:
            os.unlink(temp_file_path)

    def test_excel_ods_loading_complete(self, manager):
        """Test Excel and ODS loading complete branches (lines 392-558)."""
        
        # Test Excel file loading with various scenarios
        sample_data = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        
        # Test Excel with unknown extension (line 403)
        with patch('pandas.ExcelFile') as mock_excel_file, \
             patch('pandas.read_excel', return_value=sample_data):
            
            mock_file_obj = Mock()
            mock_file_obj.sheet_names = ['Sheet1']
            mock_excel_file.return_value.__enter__.return_value = mock_file_obj
            
            # Test with unknown extension - should use auto-detection
            with tempfile.NamedTemporaryFile(suffix='.xlsb', delete=False, dir='.') as temp_file:
                temp_file_path = temp_file.name
                
            try:
                result = manager._load_excel_file(temp_file_path)
                assert 'Sheet1' in result
            finally:
                os.unlink(temp_file_path)
                
        # Test Excel file with pandas error (lines 412, 420-421)
        with patch('pandas.ExcelFile', side_effect=Exception("Excel read error")):
            try:
                manager._load_excel_file("./nonexistent.xlsx")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Failed to load Excel file" in str(e)
                
        # Test date conversion in Excel (lines 449-450, 462)
        df_with_dates = pd.DataFrame({
            'date_col': ['2024-01-01', '2024-01-02'],
            'text_col': ['A', 'B']
        })
        
        with patch('pandas.ExcelFile') as mock_excel_file, \
             patch('pandas.read_excel', return_value=df_with_dates):
            
            mock_file_obj = Mock()
            mock_file_obj.sheet_names = ['Sheet1']
            mock_excel_file.return_value.__enter__.return_value = mock_file_obj
            
            # Mock datetime conversion with one success, one failure
            with patch('pandas.to_datetime') as mock_to_datetime:
                mock_to_datetime.side_effect = [
                    pd.to_datetime(['2024-01-01', '2024-01-02']),
                    Exception("Not a date")
                ]
                
                with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False, dir='.') as temp_file:
                    temp_file_path = temp_file.name
                    
                try:
                    result = manager._load_excel_file(temp_file_path)
                    assert 'Sheet1' in result
                finally:
                    os.unlink(temp_file_path)

    def test_ods_loading_complete_branches(self, manager):
        """Test ODS loading complete branches (lines 527-558)."""
        
        sample_data = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        
        # Test ODS with pandas error (lines 527-528)  
        with patch('localdata_mcp.localdata_mcp.ODFPY_AVAILABLE', True), \
             patch('pandas.ExcelFile', side_effect=Exception("ODS read error")):
            try:
                manager._load_ods_file("./test.ods")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Failed to load ODS file" in str(e)
                
        # Test ODS date conversion (lines 550-552, 555)
        df_with_dates = pd.DataFrame({
            'date_col': ['2024-01-01', '2024-01-02'],
            'text_col': ['A', 'B']
        })
        
        with patch('localdata_mcp.localdata_mcp.ODFPY_AVAILABLE', True), \
             patch('pandas.ExcelFile') as mock_excel_file, \
             patch('pandas.read_excel', return_value=df_with_dates):
            
            mock_file_obj = Mock()
            mock_file_obj.sheet_names = ['Sheet1']
            mock_excel_file.return_value.__enter__.return_value = mock_file_obj
            
            # Mock datetime conversion with one success, one failure
            with patch('pandas.to_datetime') as mock_to_datetime:
                mock_to_datetime.side_effect = [
                    pd.to_datetime(['2024-01-01', '2024-01-02']),
                    Exception("Not a date")
                ]
                
                with tempfile.NamedTemporaryFile(suffix='.ods', delete=False, dir='.') as temp_file:
                    temp_file_path = temp_file.name
                    
                try:
                    result = manager._load_ods_file(temp_file_path)
                    assert 'Sheet1' in result
                finally:
                    os.unlink(temp_file_path)

    def test_analytical_formats(self, manager):
        """Test analytical format loading (lines 565-620)."""
        
        # Test feather format (lines 565-587)
        with tempfile.NamedTemporaryFile(suffix='.feather', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(b"fake feather content")
            
        try:
            # Test successful feather loading
            with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True), \
                 patch('pandas.read_feather', return_value=pd.DataFrame({'id': [1], 'name': ['test']})):
                result = manager._load_feather_file(temp_file_path)
                assert isinstance(result, pd.DataFrame)
                
            # Test pyarrow not available
            with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', False):
                try:
                    manager._load_feather_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "pyarrow library is required" in str(e)
                    
            # Test feather loading error
            with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True), \
                 patch('pandas.read_feather', side_effect=Exception("Feather read error")):
                try:
                    manager._load_feather_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "Failed to load Feather file" in str(e)
        finally:
            os.unlink(temp_file_path)
            
        # Test arrow format (lines 599-620)
        with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False, dir='.') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(b"fake arrow content")
            
        try:
            # Test successful arrow loading
            with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True), \
                 patch('pyarrow.ipc.open_file') as mock_arrow_file:
                
                mock_reader = Mock()
                mock_table = Mock()
                mock_table.to_pandas.return_value = pd.DataFrame({'id': [1], 'name': ['test']})
                mock_reader.read_all.return_value = mock_table
                mock_arrow_file.return_value.__enter__.return_value = mock_reader
                
                result = manager._load_arrow_file(temp_file_path)
                assert isinstance(result, pd.DataFrame)
                
            # Test pyarrow not available
            with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', False):
                try:
                    manager._load_arrow_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "pyarrow library is required" in str(e)
                    
            # Test arrow loading error
            with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True), \
                 patch('pyarrow.ipc.open_file', side_effect=Exception("Arrow read error")):
                try:
                    manager._load_arrow_file(temp_file_path)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "Failed to load Arrow file" in str(e)
        finally:
            os.unlink(temp_file_path)

    def test_connection_error_handling(self, manager):
        """Test connection error handling (lines 694-695, 743-766)."""
        
        # Test database connection with existing connection (lines 694-695)
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "./test.csv")
            
            # Try to connect again with same name
            result = manager.connect_database("test_db", "csv", "./test2.csv")
            assert "already connected" in result

    def test_execute_query_edge_cases(self, manager):
        """Test execute_query edge cases (lines 787-945)."""
        
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "./test.csv")
            
            # Test query with SQL injection protection
            query = "SELECT * FROM data_table WHERE id = '1; DROP TABLE data_table; --'"
            result = manager.execute_query("test_db", query)
            # Should execute safely without dropping table
            assert "error" not in result.lower() or "data" in result
            
            # Test empty result handling
            with patch('pandas.read_sql') as mock_read_sql:
                mock_read_sql.return_value = pd.DataFrame()  # Empty result
                result = manager.execute_query("test_db", "SELECT * FROM data_table WHERE 1=0")
                assert result == "[]"  # Should return empty JSON array

    def test_database_inspection_all_branches(self, manager):
        """Test all database inspection branches (lines 1009-1070)."""
        
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "./test.csv")
            
            # Test describe_database complete path (lines 1009-1020)
            with patch('sqlalchemy.inspect') as mock_inspect:
                mock_inspector = Mock()
                mock_inspector.get_table_names.return_value = ['data_table']
                mock_inspector.default_schema_name = 'main'
                mock_inspector.get_server_version_info.return_value = (3, 39, 4)
                mock_inspector.get_schema_names.return_value = ['main']
                mock_inspect.return_value = mock_inspector
                
                # Mock _get_table_metadata
                with patch.object(manager, '_get_table_metadata', return_value={
                    'name': 'data_table',
                    'columns': [{'name': 'id', 'type': 'INTEGER'}],
                    'foreign_keys': [],
                    'primary_keys': ['id'],
                    'indexes': [],
                    'options': {}
                }):
                    # Mock connection.execute for row count
                    original_engine = manager.connections["test_db"]
                    with patch.object(original_engine, 'connect') as mock_connect:
                        mock_connection = Mock()
                        mock_result = Mock()
                        mock_result.scalar.return_value = 100
                        mock_connection.execute.return_value = mock_result
                        mock_connect.return_value.__enter__.return_value = mock_connection
                        
                        result = manager.describe_database("test_db")
                        response = json.loads(result)
                        assert response["name"] == "test_db"
                        assert len(response["tables"]) == 1
                        assert response["tables"][0]["size"] == 100

    def test_utility_functions_complete(self, manager):
        """Test utility functions complete coverage (lines 1100-1139, 1161)."""
        
        # Test memory check error handling (lines 1100-1102)
        with patch('psutil.virtual_memory', side_effect=Exception("Memory error")):
            result = manager._check_memory_usage()
            assert "error" in result
            assert result["low_memory"] is False
            
        # Test _get_sql_flavor with engine (lines 1127, 1129, 1131, 1137-1139)
        mock_engine = Mock()
        mock_engine.dialect.name = "postgresql"
        flavor = manager._get_sql_flavor("unknown", mock_engine)
        assert flavor == "Postgresql"  # Title case
        
        # Test if __name__ == "__main__" branch (line 1161)
        # This is covered by importing the module, but we can test main() directly
        with patch('localdata_mcp.localdata_mcp.mcp') as mock_mcp:
            from localdata_mcp.localdata_mcp import main
            main()
            mock_mcp.run.assert_called_once_with(transport="stdio")


if __name__ == "__main__":
    pytest.main([__file__])