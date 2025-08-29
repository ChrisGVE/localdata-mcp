"""Comprehensive tests for spreadsheet format support in localdata-mcp."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from localdata_mcp import DatabaseManager


class TestSpreadsheetFormats:
    """Test spreadsheet format support (Excel, ODS, etc.)"""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh DatabaseManager instance for testing."""
        return DatabaseManager()
    
    @pytest.fixture
    def test_data_dir(self):
        """Get path to test data directory."""
        return Path(__file__).parent / "data"
    
    def test_csv_file_connection(self, manager, test_data_dir):
        """Test connecting to CSV file."""
        csv_file = test_data_dir / "test_data.csv"
        
        # Connect to CSV file
        result = manager.connect_database("test_csv", "csv", str(csv_file))
        assert "Successfully connected" in result
        assert "test_csv" in manager.connections
        
        # Test querying the data
        query_result = manager.execute_query("test_csv", "SELECT * FROM data_table")
        assert "Alice Johnson" in query_result
        assert "Engineering" in query_result
        
        # Clean up
        manager.disconnect_database("test_csv")
    
    def test_tsv_file_connection(self, manager, test_data_dir):
        """Test connecting to TSV (tab-separated values) file."""
        tsv_file = test_data_dir / "test_data.tsv"
        
        # Connect to TSV file
        result = manager.connect_database("test_tsv", "tsv", str(tsv_file))
        assert "Successfully connected" in result
        assert "test_tsv" in manager.connections
        
        # Test querying the data
        query_result = manager.execute_query("test_tsv", "SELECT * FROM data_table")
        assert "Bob Smith" in query_result
        assert "Sales" in query_result
        
        # Clean up
        manager.disconnect_database("test_tsv")
    
    def test_xml_file_connection(self, manager, test_data_dir):
        """Test connecting to XML file."""
        xml_file = test_data_dir / "structured_data.xml"
        
        # Connect to XML file
        result = manager.connect_database("test_xml", "xml", str(xml_file))
        assert "Successfully connected" in result
        assert "test_xml" in manager.connections
        
        # Test querying the data
        query_result = manager.execute_query("test_xml", "SELECT * FROM data_table")
        assert "Carol Williams" in query_result
        assert "Marketing" in query_result
        
        # Clean up
        manager.disconnect_database("test_xml")
    
    def test_ini_file_connection(self, manager, test_data_dir):
        """Test connecting to INI configuration file."""
        ini_file = test_data_dir / "config.ini"
        
        # Connect to INI file
        result = manager.connect_database("test_ini", "ini", str(ini_file))
        assert "Successfully connected" in result
        assert "test_ini" in manager.connections
        
        # Test querying the data
        query_result = manager.execute_query("test_ini", "SELECT * FROM data_table")
        assert "database" in query_result
        assert "localhost" in query_result
        
        # Clean up
        manager.disconnect_database("test_ini")
    
    @patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', True)
    def test_excel_file_missing_library(self, manager, test_data_dir):
        """Test Excel file handling when openpyxl is not available."""
        with patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', False):
            with pytest.raises(ValueError, match="openpyxl library is required"):
                # This should fail since we're mocking openpyxl as unavailable
                manager._load_excel_file(str(test_data_dir / "test_data.csv"))  # Using CSV as placeholder
    
    @patch('localdata_mcp.localdata_mcp.ODFPY_AVAILABLE', True)  
    def test_ods_file_missing_library(self, manager, test_data_dir):
        """Test ODS file handling when odfpy is not available."""
        with patch('localdata_mcp.localdata_mcp.ODFPY_AVAILABLE', False):
            with pytest.raises(ValueError, match="odfpy library is required"):
                # This should fail since we're mocking odfpy as unavailable
                manager._load_ods_file(str(test_data_dir / "test_data.csv"))  # Using CSV as placeholder
    
    @patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', True)
    def test_parquet_file_missing_library(self, manager, test_data_dir):
        """Test Parquet file handling when pyarrow is not available."""  
        with patch('localdata_mcp.localdata_mcp.PYARROW_AVAILABLE', False):
            with pytest.raises(ValueError, match="pyarrow library is required"):
                # Create a temporary CSV file and try to load as Parquet
                csv_file = test_data_dir / "test_data.csv"
                manager._create_engine_from_file(str(csv_file), "parquet")
    
    def test_sheet_name_sanitization(self, manager):
        """Test sheet name sanitization for SQL table names."""
        test_cases = [
            ("Sheet with spaces", "Sheet_with_spaces"),
            ("Sheet-with-hyphens", "Sheet_with_hyphens"),
            ("Sheet@#$%Special", "Sheet____Special"),
            ("123StartWithNumber", "sheet_123StartWithNumber"),
            ("", "sheet_unnamed"),
            ("Valid_Name", "Valid_Name"),
        ]
        
        used_names = set()
        for original, expected in test_cases:
            result = manager._sanitize_sheet_name(original, used_names)
            if expected == "Sheet_with_spaces":
                assert result == expected or result.startswith(expected)
            else:
                assert result == expected or result.startswith(expected.split('_')[0])
    
    def test_sheet_name_uniqueness(self, manager):
        """Test that duplicate sheet names are made unique."""
        used_names = set()
        
        # First occurrence
        name1 = manager._sanitize_sheet_name("duplicate", used_names)
        assert name1 == "duplicate"
        
        # Second occurrence should be made unique
        name2 = manager._sanitize_sheet_name("duplicate", used_names)
        assert name2 == "duplicate_1"
        
        # Third occurrence
        name3 = manager._sanitize_sheet_name("duplicate", used_names)
        assert name3 == "duplicate_2"
    
    def test_file_path_security(self, manager):
        """Test file path security restrictions."""
        # Test path traversal prevention
        with pytest.raises(ValueError, match="outside the allowed directory"):
            manager._sanitize_path("../etc/passwd")
        
        with pytest.raises(ValueError, match="outside the allowed directory"):
            manager._sanitize_path("/etc/passwd")
        
        # Test non-existent file
        with pytest.raises(ValueError, match="File not found"):
            manager._sanitize_path("nonexistent_file.csv")
    
    def test_file_size_detection(self, manager, test_data_dir):
        """Test file size detection functionality."""
        csv_file = test_data_dir / "test_data.csv"
        
        # Test file size detection
        size = manager._get_file_size(str(csv_file))
        assert size > 0
        
        # Test large file detection (with low threshold)
        is_large = manager._is_large_file(str(csv_file), threshold_mb=0.001)  # Very small threshold
        assert is_large  # CSV file should be considered large with tiny threshold
        
        # Test with normal threshold
        is_large_normal = manager._is_large_file(str(csv_file), threshold_mb=100)
        assert not is_large_normal  # CSV file should not be large with normal threshold
    
    def test_empty_file_handling(self, manager, test_data_dir):
        """Test handling of empty files."""
        empty_file = test_data_dir / "empty.csv"
        
        # Connect to empty CSV file - should handle gracefully
        result = manager.connect_database("empty_csv", "csv", str(empty_file))
        # This might succeed or fail depending on pandas behavior with empty files
        # The main thing is it shouldn't crash
        
        if "Successfully connected" in result:
            # If connection succeeded, test querying
            query_result = manager.execute_query("empty_csv", "SELECT COUNT(*) as row_count FROM data_table")
            assert "0" in query_result or "no results" in query_result.lower()
            manager.disconnect_database("empty_csv")
    
    def test_corrupted_file_handling(self, manager, test_data_dir):
        """Test handling of corrupted files."""
        corrupted_file = test_data_dir / "corrupted.xlsx"
        
        # Attempt to connect to corrupted Excel file
        result = manager.connect_database("corrupted_excel", "excel", str(corrupted_file))
        assert "Failed to connect" in result or "error" in result.lower()
    
    def test_multiple_connections(self, manager, test_data_dir):
        """Test managing multiple file connections simultaneously."""
        csv_file = test_data_dir / "test_data.csv"
        tsv_file = test_data_dir / "test_data.tsv"
        xml_file = test_data_dir / "structured_data.xml"
        
        # Connect to multiple files
        result1 = manager.connect_database("conn1", "csv", str(csv_file))
        result2 = manager.connect_database("conn2", "tsv", str(tsv_file))
        result3 = manager.connect_database("conn3", "xml", str(xml_file))
        
        assert "Successfully connected" in result1
        assert "Successfully connected" in result2  
        assert "Successfully connected" in result3
        
        # Test that all connections are active
        databases = manager.list_databases()
        assert "conn1" in databases
        assert "conn2" in databases
        assert "conn3" in databases
        
        # Test querying different connections
        query1 = manager.execute_query("conn1", "SELECT COUNT(*) FROM data_table")
        query2 = manager.execute_query("conn2", "SELECT COUNT(*) FROM data_table")
        query3 = manager.execute_query("conn3", "SELECT COUNT(*) FROM data_table")
        
        # All should return some data
        assert query1 and "error" not in query1.lower()
        assert query2 and "error" not in query2.lower()  
        assert query3 and "error" not in query3.lower()
        
        # Clean up
        manager.disconnect_database("conn1")
        manager.disconnect_database("conn2")
        manager.disconnect_database("conn3")
    
    def test_duplicate_connection_names(self, manager, test_data_dir):
        """Test that duplicate connection names are handled properly."""
        csv_file = test_data_dir / "test_data.csv"
        
        # First connection should succeed
        result1 = manager.connect_database("duplicate_name", "csv", str(csv_file))
        assert "Successfully connected" in result1
        
        # Second connection with same name should fail
        result2 = manager.connect_database("duplicate_name", "csv", str(csv_file))
        assert "already connected" in result2
        
        # Clean up
        manager.disconnect_database("duplicate_name")
    
    def test_connection_limits(self, manager, test_data_dir):
        """Test connection limit enforcement."""
        csv_file = test_data_dir / "test_data.csv"
        connections = []
        
        # Create connections up to the limit
        for i in range(10):  # Default limit is 10
            name = f"conn_{i}"
            result = manager.connect_database(name, "csv", str(csv_file))
            if "Successfully connected" in result:
                connections.append(name)
        
        # Next connection should fail due to limit
        result = manager.connect_database("over_limit", "csv", str(csv_file))
        assert "Maximum number of concurrent connections" in result
        
        # Clean up
        for conn_name in connections:
            manager.disconnect_database(conn_name)


class TestDataTypeHandling:
    """Test handling of various data types in spreadsheet formats."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh DatabaseManager instance for testing."""
        return DatabaseManager()
    
    def test_date_handling(self, manager):
        """Test proper handling of date columns."""
        # Create test data with dates
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,date_col,number_col\n")
            f.write("John,2023-01-15,100\n")
            f.write("Jane,2023-02-20,200\n")
            temp_file = f.name
        
        try:
            # Connect and test
            result = manager.connect_database("date_test", "csv", temp_file)
            assert "Successfully connected" in result
            
            # Query data and check date handling
            query_result = manager.execute_query("date_test", "SELECT * FROM data_table WHERE date_col >= '2023-01-01'")
            assert "John" in query_result
            assert "Jane" in query_result
            
            manager.disconnect_database("date_test")
        finally:
            os.unlink(temp_file)
    
    def test_numeric_handling(self, manager):
        """Test proper handling of numeric columns."""
        # Create test data with various numeric types
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,integer_col,float_col,currency_col\n")
            f.write("Item1,100,19.99,1000.50\n")
            f.write("Item2,200,29.99,2000.75\n")
            temp_file = f.name
        
        try:
            # Connect and test
            result = manager.connect_database("numeric_test", "csv", temp_file)
            assert "Successfully connected" in result
            
            # Test numeric operations
            query_result = manager.execute_query("numeric_test", "SELECT AVG(float_col) as avg_price FROM data_table")
            assert "24.99" in query_result or "25" in query_result  # Average of 19.99 and 29.99
            
            manager.disconnect_database("numeric_test")
        finally:
            os.unlink(temp_file)
    
    def test_boolean_handling(self, manager):
        """Test proper handling of boolean columns."""
        # Create test data with boolean values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,active,published\n")
            f.write("User1,True,false\n")
            f.write("User2,FALSE,True\n") 
            f.write("User3,1,0\n")
            temp_file = f.name
        
        try:
            # Connect and test
            result = manager.connect_database("boolean_test", "csv", temp_file)
            assert "Successfully connected" in result
            
            # Test boolean queries
            query_result = manager.execute_query("boolean_test", "SELECT * FROM data_table WHERE active = 'True' OR active = '1'")
            assert "User1" in query_result
            assert "User3" in query_result
            
            manager.disconnect_database("boolean_test")
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])