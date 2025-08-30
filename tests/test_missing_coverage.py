"""
Targeted tests for missing coverage areas to reach 100% coverage.
This file specifically tests code paths that aren't covered by existing tests.
"""

import json
import pytest
import pandas as pd
import tempfile
import configparser
from unittest.mock import patch, Mock, MagicMock, mock_open
from tests.test_helpers import create_test_manager
from tests.mock_helpers import mock_csv_connection, mock_database_inspection


class TestMissingCoverage:
    """Tests for specific missing coverage areas."""

    @pytest.fixture
    def manager(self):
        return create_test_manager()

    def test_import_error_handling(self, manager):
        """Test import error handling for optional libraries."""
        
        # Test TOML not available
        with patch('localdata_mcp.localdata_mcp.TOML_AVAILABLE', False):
            with mock_csv_connection():
                # Should get error when trying to connect to TOML file without library
                result = manager.connect_database("toml_db", "toml", "/tmp/test.toml") 
                assert "toml library is required" in result

        # Test OPENPYXL not available  
        with patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', False):
            with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value='/tmp/test.xlsx'), \
                 patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
                 patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False):
                result = manager.connect_database("excel_db", "excel", "/tmp/test.xlsx")
                assert "openpyxl library is required" in result

        # Test XLRD not available
        with patch('localdata_mcp.localdata_mcp.XLRD_AVAILABLE', False):
            with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value='/tmp/test.xls'), \
                 patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
                 patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False):
                result = manager.connect_database("excel_db", "excel", "/tmp/test.xls")
                assert "xlrd library is required" in result

        # Test ODFPY not available
        with patch('localdata_mcp.localdata_mcp.ODFPY_AVAILABLE', False):
            with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value='/tmp/test.ods'), \
                 patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
                 patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False):
                result = manager.connect_database("ods_db", "ods", "/tmp/test.ods")
                assert "odfpy library is required" in result

        # Test PYARROW not available
        with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', False):
            for file_type in ['parquet', 'feather', 'arrow']:
                with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value=f'/tmp/test.{file_type}'), \
                     patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
                     patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False):
                    result = manager.connect_database(f"{file_type}_db", file_type, f"/tmp/test.{file_type}")
                    assert "pyarrow library is required" in result

        # Test LXML not available  
        with patch('localdata_mcp.localdata_mcp.LXML_AVAILABLE', False):
            with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value='/tmp/test.xml'), \
                 patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
                 patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False), \
                 patch('pandas.read_xml', side_effect=Exception("lxml not found")):
                result = manager.connect_database("xml_db", "xml", "/tmp/test.xml")
                assert "lxml library is required" in result

    def test_path_sanitization_edge_cases(self, manager):
        """Test path sanitization edge cases."""
        
        # Test invalid path (OSError)
        with patch('pathlib.Path.resolve', side_effect=OSError("Invalid path")):
            try:
                manager._sanitize_path("/invalid/path")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Invalid path" in str(e)

        # Test path outside allowed directory
        with patch('pathlib.Path.resolve') as mock_resolve, \
             patch('pathlib.Path.relative_to', side_effect=ValueError("Outside directory")):
            mock_path = Mock()
            mock_resolve.return_value = mock_path
            try:
                manager._sanitize_path("/outside/path")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "outside the allowed directory" in str(e)

        # Test file not found
        with patch('pathlib.Path.resolve') as mock_resolve, \
             patch('pathlib.Path.relative_to'), \
             patch('pathlib.Path.is_file', return_value=False):
            mock_path = Mock()
            mock_resolve.return_value = mock_path
            try:
                manager._sanitize_path("/nonexistent/file")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "File not found" in str(e)

        # Test successful path sanitization
        with patch('pathlib.Path.resolve') as mock_resolve, \
             patch('pathlib.Path.relative_to'), \
             patch('pathlib.Path.is_file', return_value=True):
            mock_path = Mock()
            mock_path.__str__ = Mock(return_value="/safe/path/file.csv")
            mock_resolve.return_value = mock_path
            result = manager._sanitize_path("./file.csv")
            assert result == "/safe/path/file.csv"

    def test_file_size_operations(self, manager):
        """Test file size detection and large file handling."""
        
        # Test file size error
        with patch('os.path.getsize', side_effect=OSError("Permission denied")):
            try:
                manager._get_file_size("/protected/file")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Cannot get size" in str(e)

        # Test large file detection with different thresholds
        with patch('os.path.getsize', return_value=200 * 1024 * 1024):  # 200MB
            assert manager._is_large_file("/large/file", 100) is True  # 100MB threshold
            assert manager._is_large_file("/large/file", 300) is False  # 300MB threshold

    def test_excel_file_loading_details(self, manager):
        """Test detailed Excel file loading scenarios."""
        
        # Test Excel file with unknown extension
        test_df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        
        with patch('pathlib.Path.suffix', '.unknown'), \
             patch('pandas.ExcelFile') as mock_excel_file, \
             patch('pandas.read_excel', return_value=test_df):
            
            mock_file_obj = Mock()
            mock_file_obj.sheet_names = ['Sheet1']
            mock_excel_file.return_value.__enter__.return_value = mock_file_obj
            
            result = manager._load_excel_file("/test/file.unknown")
            assert 'Sheet1' in result

        # Test Excel file with empty sheet
        empty_df = pd.DataFrame()
        test_df = pd.DataFrame({'id': [1], 'name': ['A']})
        
        with patch('pandas.ExcelFile') as mock_excel_file, \
             patch('pandas.read_excel', side_effect=[empty_df, test_df]):
            
            mock_file_obj = Mock()
            mock_file_obj.sheet_names = ['EmptySheet', 'DataSheet']
            mock_excel_file.return_value.__enter__.return_value = mock_file_obj
            
            result = manager._load_excel_file("/test/file.xlsx")
            # Empty sheet should be skipped, only DataSheet should remain
            assert 'DataSheet' in result
            assert len(result) == 1

        # Test Excel file with sheet loading error
        with patch('pandas.ExcelFile') as mock_excel_file, \
             patch('pandas.read_excel', side_effect=Exception("Sheet corrupted")):
            
            mock_file_obj = Mock()
            mock_file_obj.sheet_names = ['BadSheet']
            mock_excel_file.return_value.__enter__.return_value = mock_file_obj
            
            try:
                manager._load_excel_file("/test/file.xlsx")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "contains no readable data" in str(e)

        # Test Excel file datetime handling
        test_df = pd.DataFrame({
            'date_col': ['2024-01-01', '2024-01-02'],
            'text_col': ['A', 'B'],
            'invalid_date': ['not_a_date', 'also_not_a_date']
        })
        
        with patch('pandas.ExcelFile') as mock_excel_file, \
             patch('pandas.read_excel', return_value=test_df), \
             patch('pandas.to_datetime') as mock_to_datetime:
            
            # Mock datetime conversion - first call succeeds, others fail
            mock_to_datetime.side_effect = [
                pd.to_datetime(['2024-01-01', '2024-01-02']),  # date_col succeeds
                Exception("Invalid date"),  # text_col fails
                Exception("Invalid date")   # invalid_date fails
            ]
            
            mock_file_obj = Mock()
            mock_file_obj.sheet_names = ['Sheet1']
            mock_excel_file.return_value.__enter__.return_value = mock_file_obj
            
            result = manager._load_excel_file("/test/file.xlsx")
            assert 'Sheet1' in result

    def test_ods_file_loading_details(self, manager):
        """Test ODS file loading edge cases."""
        
        # Test ODS with empty sheet
        empty_df = pd.DataFrame()
        test_df = pd.DataFrame({'metric': ['cpu'], 'value': [75.0]})
        
        with patch('pandas.ExcelFile') as mock_excel_file, \
             patch('pandas.read_excel', side_effect=[empty_df, test_df]):
            
            mock_file_obj = Mock()
            mock_file_obj.sheet_names = ['EmptyData', 'RealData']
            mock_excel_file.return_value.__enter__.return_value = mock_file_obj
            
            result = manager._load_ods_file("/test/file.ods")
            assert 'RealData' in result
            assert len(result) == 1

        # Test ODS with sheet error
        with patch('pandas.ExcelFile') as mock_excel_file, \
             patch('pandas.read_excel', side_effect=Exception("ODS error")):
            
            mock_file_obj = Mock()
            mock_file_obj.sheet_names = ['BadData']
            mock_excel_file.return_value.__enter__.return_value = mock_file_obj
            
            try:
                manager._load_ods_file("/test/file.ods")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "contains no readable data" in str(e)

    def test_xml_loading_edge_cases(self, manager):
        """Test XML file loading edge cases."""
        
        # Test empty XML file
        with patch('pandas.read_xml', return_value=pd.DataFrame()):
            try:
                manager._load_xml_file("/empty.xml")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "contains no data" in str(e)

        # Test successful XML loading
        test_df = pd.DataFrame({'id': [1, 2], 'name-with-dash': ['A', 'B']})
        with patch('pandas.read_xml', return_value=test_df):
            result = manager._load_xml_file("/test.xml")
            # Check that column names are cleaned
            assert 'name_with_dash' in result.columns

    def test_ini_file_loading_edge_cases(self, manager):
        """Test INI file loading scenarios."""
        
        # Test INI with only DEFAULT section
        mock_config = Mock()
        mock_config.sections.return_value = []
        mock_config.defaults.return_value = {'key1': 'value1', 'key2': 'value2'}
        
        with patch('configparser.ConfigParser', return_value=mock_config):
            result = manager._load_ini_file("/test.ini")
            assert len(result) == 2
            assert all(result['section'] == 'DEFAULT')

        # Test completely empty INI file
        mock_config = Mock()
        mock_config.sections.return_value = []
        mock_config.defaults.return_value = {}
        
        with patch('configparser.ConfigParser', return_value=mock_config):
            try:
                manager._load_ini_file("/empty.ini")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "contains no configuration data" in str(e)

    def test_csv_parsing_fallback(self, manager):
        """Test CSV parsing with fallback scenarios."""
        
        # Test CSV with parser error, fallback to no header
        fallback_df = pd.DataFrame({0: [1, 2], 1: ['A', 'B']})
        
        with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value='/tmp/test.csv'), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False), \
             patch('pandas.read_csv', side_effect=[pd.errors.ParserError("Bad CSV"), fallback_df]), \
             patch('sqlalchemy.create_engine') as mock_engine, \
             patch('pandas.DataFrame.to_sql'):
            
            mock_engine.return_value.dialect.name = 'sqlite'
            result = manager.connect_database("csv_db", "csv", "/tmp/test.csv")
            response = json.loads(result)
            assert response["success"] is True

        # Test TSV with parser error, fallback to no header  
        with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value='/tmp/test.tsv'), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False), \
             patch('pandas.read_csv', side_effect=[pd.errors.ParserError("Bad TSV"), fallback_df]), \
             patch('sqlalchemy.create_engine') as mock_engine, \
             patch('pandas.DataFrame.to_sql'):
            
            mock_engine.return_value.dialect.name = 'sqlite'
            result = manager.connect_database("tsv_db", "tsv", "/tmp/test.tsv")  
            response = json.loads(result)
            assert response["success"] is True

    def test_database_metadata_edge_cases(self, manager):
        """Test database metadata handling edge cases."""
        
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "/tmp/test.csv")
            
            # Test table options not implemented (SQLite case)
            with patch('sqlalchemy.inspect') as mock_inspect:
                mock_inspector = Mock()
                mock_inspector.get_columns.return_value = [
                    {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'autoincrement': True, 'default': None}
                ]
                mock_inspector.get_foreign_keys.return_value = [
                    {'name': 'fk1', 'constrained_columns': ['id'], 'referred_table': 'other', 'referred_columns': ['id']}
                ]
                mock_inspector.get_pk_constraint.return_value = {'constrained_columns': ['id']}
                mock_inspector.get_indexes.return_value = [
                    {'name': 'idx1', 'column_names': ['id'], 'unique': True}
                ]
                mock_inspector.get_table_options.side_effect = NotImplementedError("Not supported")
                mock_inspect.return_value = mock_inspector
                
                metadata = manager._get_table_metadata(mock_inspector, "test_table")
                assert metadata['options'] == {}
                assert len(metadata['columns']) == 1
                assert metadata['columns'][0]['primary_key'] is True
                assert metadata['columns'][0]['foreign_key']['referred_table'] == 'other'

    def test_safe_table_identifier_validation(self, manager):
        """Test SQL table identifier validation."""
        
        # Test invalid table names
        invalid_names = [
            "123invalid",  # starts with number
            "table-with-dash",  # contains dash
            "table with space",  # contains space
            "table!@#",  # special characters
            "",  # empty string
        ]
        
        for invalid_name in invalid_names:
            try:
                manager._safe_table_identifier(invalid_name)
                assert False, f"Should have rejected invalid name: {invalid_name}"
            except ValueError as e:
                assert "Invalid table name" in str(e)

        # Test valid table names
        valid_names = ["valid_table", "table1", "_private_table", "table_123"]
        for valid_name in valid_names:
            result = manager._safe_table_identifier(valid_name)
            assert valid_name in result

    def test_cleanup_operations_detailed(self, manager):
        """Test detailed cleanup operations."""
        
        # Test temp file cleanup with existing files
        with patch('os.path.exists', return_value=True), \
             patch('os.unlink') as mock_unlink:
            
            manager.temp_files = ['/tmp/file1.sqlite', '/tmp/file2.sqlite']
            manager._cleanup_all()
            assert mock_unlink.call_count == 2
            assert len(manager.temp_files) == 0

        # Test temp file cleanup with missing files (OSError ignored)
        with patch('os.path.exists', return_value=True), \
             patch('os.unlink', side_effect=OSError("File busy")):
            
            manager.temp_files = ['/tmp/busy_file.sqlite']
            manager._cleanup_all()  # Should not raise exception
            assert len(manager.temp_files) == 0

        # Test connection cleanup with dispose errors (ignored)
        mock_engine1 = Mock()
        mock_engine1.dispose.side_effect = Exception("Cleanup error")
        mock_engine2 = Mock()
        
        manager.connections = {'db1': mock_engine1, 'db2': mock_engine2}
        manager.db_types = {'db1': 'csv', 'db2': 'json'}
        
        manager._cleanup_all()  # Should not raise exception
        assert len(manager.connections) == 0
        assert len(manager.db_types) == 0

    def test_memory_usage_error_handling(self, manager):
        """Test memory usage checking error handling."""
        
        # Test psutil error
        with patch('psutil.virtual_memory', side_effect=Exception("psutil error")):
            result = manager._check_memory_usage()
            assert "error" in result
            assert result["low_memory"] is False

    def test_sql_flavor_detection_comprehensive(self, manager):
        """Test SQL flavor detection for all database types."""
        
        # Test all supported database types
        test_cases = [
            ("sqlite", None, "SQLite"),
            ("postgresql", None, "PostgreSQL"), 
            ("mysql", None, "MySQL"),
            ("csv", None, "SQLite"),
            ("json", None, "SQLite"),
            ("excel", None, "SQLite"),
        ]
        
        for db_type, engine, expected_flavor in test_cases:
            result = manager._get_sql_flavor(db_type, engine)
            assert result == expected_flavor

        # Test with engine dialect
        mock_engine = Mock()
        mock_engine.dialect.name = 'oracle'
        result = manager._get_sql_flavor("unknown", mock_engine)
        assert result == "Oracle"

        # Test unknown without engine
        result = manager._get_sql_flavor("unknown", None)
        assert result == "Unknown"


if __name__ == "__main__":
    pytest.main([__file__])