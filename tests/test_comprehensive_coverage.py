"""
Comprehensive test coverage for all 9 LocalData MCP Server tools.
This test file systematically tests all tools to achieve 100% coverage.
"""

import json
import pytest
import pandas as pd
from unittest.mock import patch, Mock, MagicMock
from tests.test_helpers import create_test_manager
from tests.mock_helpers import (
    mock_csv_connection, mock_json_connection, mock_excel_connection,
    mock_database_query, mock_large_dataset_query, mock_database_inspection,
    mock_memory_pressure, mock_normal_memory, mock_corrupted_file
)


class TestComprehensiveCoverage:
    """Comprehensive tests for all 9 tools to maximize coverage."""

    @pytest.fixture
    def manager(self):
        return create_test_manager()

    def test_connect_database_all_formats(self, manager):
        """Test connect_database with all supported file formats."""
        
        # Test CSV
        with mock_csv_connection():
            result = manager.connect_database("csv_db", "csv", "/tmp/test.csv")
            response = json.loads(result)
            assert response["success"] is True
            assert response["connection_info"]["db_type"] == "csv"
        
        # Test JSON
        with mock_json_connection():
            result = manager.connect_database("json_db", "json", "/tmp/test.json")
            response = json.loads(result)
            assert response["success"] is True
            assert response["connection_info"]["db_type"] == "json"
        
        # Test Excel
        with mock_excel_connection():
            result = manager.connect_database("excel_db", "excel", "/tmp/test.xlsx")
            response = json.loads(result)
            assert response["success"] is True
            assert response["connection_info"]["db_type"] == "excel"

    def test_connect_database_error_conditions(self, manager):
        """Test connect_database error handling."""
        
        # Test unsupported database type
        result = manager.connect_database("bad_db", "unsupported_type", "/tmp/test.bad")
        assert "Unsupported db_type" in result
        
        # Test duplicate connection name
        with mock_csv_connection():
            manager.connect_database("dup_db", "csv", "/tmp/test.csv")
            result = manager.connect_database("dup_db", "csv", "/tmp/test2.csv")
            assert "already connected" in result

    def test_connect_database_with_special_formats(self, manager):
        """Test connecting to YAML, TOML, XML, INI, TSV files."""
        
        # Test YAML
        sample_yaml_data = pd.DataFrame([
            {'app': 'test', 'version': '1.0'},
            {'database': 'postgres', 'port': 5432}
        ])
        
        with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value='/tmp/test.yml'), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False), \
             patch('yaml.safe_load', return_value={'app': 'test', 'version': '1.0'}), \
             patch('pandas.json_normalize', return_value=sample_yaml_data), \
             patch('sqlalchemy.create_engine') as mock_engine, \
             patch('pandas.DataFrame.to_sql'):
            
            mock_engine.return_value.dialect.name = 'sqlite'
            result = manager.connect_database("yaml_db", "yaml", "/tmp/test.yml")
            response = json.loads(result)
            assert response["success"] is True

        # Test TOML
        with patch('localdata_mcp.localdata_mcp.TOML_AVAILABLE', True), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value='/tmp/test.toml'), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False), \
             patch('toml.load', return_value={'project': 'test', 'version': '0.1.0'}), \
             patch('pandas.json_normalize', return_value=sample_yaml_data), \
             patch('sqlalchemy.create_engine') as mock_engine, \
             patch('pandas.DataFrame.to_sql'):
            
            mock_engine.return_value.dialect.name = 'sqlite'
            result = manager.connect_database("toml_db", "toml", "/tmp/test.toml")
            response = json.loads(result)
            assert response["success"] is True

        # Test XML
        xml_data = pd.DataFrame([
            {'id': 1, 'name': 'Product A', 'price': 10.99},
            {'id': 2, 'name': 'Product B', 'price': 25.50}
        ])
        
        with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value='/tmp/test.xml'), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False), \
             patch('pandas.read_xml', return_value=xml_data), \
             patch('sqlalchemy.create_engine') as mock_engine, \
             patch('pandas.DataFrame.to_sql'):
            
            mock_engine.return_value.dialect.name = 'sqlite'
            result = manager.connect_database("xml_db", "xml", "/tmp/test.xml")
            response = json.loads(result)
            assert response["success"] is True

        # Test INI
        ini_data = pd.DataFrame([
            {'section': 'database', 'key': 'host', 'value': 'localhost'},
            {'section': 'database', 'key': 'port', 'value': '5432'}
        ])
        
        with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path', return_value='/tmp/test.ini'), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=1024), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file', return_value=False), \
             patch('localdata_mcp.localdata_mcp.DatabaseManager._load_ini_file', return_value=ini_data), \
             patch('sqlalchemy.create_engine') as mock_engine, \
             patch('pandas.DataFrame.to_sql'):
            
            mock_engine.return_value.dialect.name = 'sqlite'
            result = manager.connect_database("ini_db", "ini", "/tmp/test.ini")
            response = json.loads(result)
            assert response["success"] is True

    def test_execute_query_comprehensive(self, manager):
        """Test execute_query with various scenarios."""
        
        # First connect a database
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "/tmp/test.csv")
            
            # Test small query (no chunking)
            small_result = pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['A', 'B', 'C']
            })
            
            with mock_database_query(small_result):
                result = manager.execute_query("test_db", "SELECT * FROM data_table LIMIT 3")
                response = json.loads(result)
                assert "data" in response
                assert response["metadata"]["chunked"] is False
                assert len(response["data"]) == 3
            
            # Test large query (with chunking)
            with mock_large_dataset_query(200):  # 200 rows
                result = manager.execute_query("test_db", "SELECT * FROM data_table")
                response = json.loads(result)
                assert response["metadata"]["chunked"] is True
                assert response["metadata"]["total_rows"] == 200
                assert "query_id" in response["metadata"]
                
                # Test next_chunk functionality
                query_id = response["metadata"]["query_id"]
                next_result = manager.next_chunk(query_id, 11, "10")
                next_response = json.loads(next_result)
                assert len(next_response["data"]) == 10
                assert next_response["metadata"]["showing_rows"] == "11-20"

    def test_execute_query_error_conditions(self, manager):
        """Test execute_query error handling."""
        
        # Test with nonexistent database
        result = manager.execute_query("nonexistent_db", "SELECT * FROM test")
        assert "not connected" in result
        
        # Test empty result
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "/tmp/test.csv")
            
            empty_df = pd.DataFrame()
            with mock_database_query(empty_df):
                result = manager.execute_query("test_db", "SELECT * FROM data_table WHERE 1=0")
                response = json.loads(result)
                assert response == []

    def test_execute_query_memory_management(self, manager):
        """Test execute_query memory management features."""
        
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "/tmp/test.csv")
            
            # Test with high memory pressure
            with mock_memory_pressure(), mock_large_dataset_query(150):
                result = manager.execute_query("test_db", "SELECT * FROM data_table")
                response = json.loads(result)
                # Should still work but with memory warnings
                assert "memory_info" in response["metadata"]
                assert response["metadata"]["memory_info"]["low_memory"] is True

    def test_next_chunk_comprehensive(self, manager):
        """Test next_chunk with various scenarios."""
        
        # Set up a database with buffered query
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "/tmp/test.csv")
            
            with mock_large_dataset_query(150):
                # Create a large query result first
                result = manager.execute_query("test_db", "SELECT * FROM data_table")
                response = json.loads(result)
                query_id = response["metadata"]["query_id"]
                
                # Test getting next chunk
                next_result = manager.next_chunk(query_id, 11, "20")
                next_response = json.loads(next_result)
                assert len(next_response["data"]) == 20
                
                # Test getting all remaining
                all_result = manager.next_chunk(query_id, 31, "all")
                all_response = json.loads(all_result)
                assert len(all_response["data"]) == 120  # 150 - 30 already shown
                
                # Test invalid query ID
                invalid_result = manager.next_chunk("invalid_query_id", 1, "10")
                assert "not found" in invalid_result
                
                # Test invalid start row
                invalid_start = manager.next_chunk(query_id, 0, "10")  # start_row must be >= 1
                assert "must be between 1 and" in invalid_start
                
                # Test invalid chunk size
                invalid_chunk = manager.next_chunk(query_id, 1, "invalid")
                assert "must be a positive integer" in invalid_chunk

    def test_list_databases_comprehensive(self, manager):
        """Test list_databases in various states."""
        
        # Test empty state
        result = manager.list_databases()
        response = json.loads(result)
        assert response["message"] == "No databases are currently connected."
        assert response["databases"] == []
        
        # Test with multiple connections
        with mock_csv_connection():
            manager.connect_database("csv_db", "csv", "/tmp/test.csv")
            
        with mock_json_connection():
            manager.connect_database("json_db", "json", "/tmp/test.json")
            
        result = manager.list_databases()
        response = json.loads(result)
        assert response["total_connections"] == 2
        assert len(response["databases"]) == 2
        
        db_names = [db["name"] for db in response["databases"]]
        assert "csv_db" in db_names
        assert "json_db" in db_names

    def test_describe_database_comprehensive(self, manager):
        """Test describe_database functionality."""
        
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "/tmp/test.csv")
            
            with mock_database_inspection():
                result = manager.describe_database("test_db")
                response = json.loads(result)
                
                assert response["name"] == "test_db"
                assert "dialect" in response
                assert "tables" in response
                assert len(response["tables"]) > 0
                
                # Test nonexistent database
                error_result = manager.describe_database("nonexistent_db")
                assert "not connected" in error_result

    def test_find_table_comprehensive(self, manager):
        """Test find_table functionality."""
        
        with mock_csv_connection():
            manager.connect_database("db1", "csv", "/tmp/test1.csv")
            
        with mock_json_connection():
            manager.connect_database("db2", "json", "/tmp/test2.json")
        
        with mock_database_inspection():
            # Test finding existing table
            result = manager.find_table("data_table")
            response = json.loads(result)
            assert isinstance(response, list)
            assert len(response) >= 1  # Should find the table in at least one database
            
            # Test finding nonexistent table
            result = manager.find_table("nonexistent_table")
            assert "was not found" in result

    def test_describe_table_comprehensive(self, manager):
        """Test describe_table functionality."""
        
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "/tmp/test.csv")
            
            with mock_database_inspection():
                result = manager.describe_table("test_db", "data_table")
                response = json.loads(result)
                
                assert response["name"] == "data_table"
                assert "columns" in response
                assert "size" in response
                
                # Test nonexistent table
                error_result = manager.describe_table("test_db", "nonexistent_table")
                assert "does not exist" in error_result

    def test_get_query_history(self, manager):
        """Test get_query_history functionality."""
        
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "/tmp/test.csv")
            
            # Initially no history
            result = manager.get_query_history("test_db")
            assert "No query history found" in result
            
            # Execute some queries to build history
            with mock_database_query():
                manager.execute_query("test_db", "SELECT * FROM data_table")
                manager.execute_query("test_db", "SELECT COUNT(*) FROM data_table")
                
                # Check history
                result = manager.get_query_history("test_db")
                assert "SELECT * FROM data_table" in result
                assert "SELECT COUNT(*) FROM data_table" in result
                
            # Test nonexistent database
            result = manager.get_query_history("nonexistent_db")
            assert result == "No query history found for database 'nonexistent_db'."

    def test_disconnect_database_comprehensive(self, manager):
        """Test disconnect_database functionality."""
        
        with mock_csv_connection():
            manager.connect_database("test_db", "csv", "/tmp/test.csv")
            
            # Verify connection exists
            assert "test_db" in manager.connections
            
            # Disconnect
            result = manager.disconnect_database("test_db")
            assert "Successfully disconnected" in result
            assert "test_db" not in manager.connections
            
            # Try to disconnect again
            result = manager.disconnect_database("test_db")
            assert "not connected" in result

    def test_memory_and_performance_features(self, manager):
        """Test memory monitoring and performance features."""
        
        # Test memory checking methods
        with mock_normal_memory():
            memory_info = manager._check_memory_usage()
            assert memory_info["used_percent"] == 50.0
            assert memory_info["low_memory"] is False
        
        with mock_memory_pressure():
            memory_info = manager._check_memory_usage()
            assert memory_info["used_percent"] == 90.0
            assert memory_info["low_memory"] is True

    def test_file_handling_edge_cases(self, manager):
        """Test various file handling edge cases."""
        
        # Test large file detection
        with patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=200 * 1024 * 1024):  # 200MB
            assert manager._is_large_file("/tmp/large_file.csv") is True
        
        with patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size', return_value=50 * 1024 * 1024):  # 50MB
            assert manager._is_large_file("/tmp/small_file.csv") is False

    def test_utility_methods(self, manager):
        """Test various utility methods for coverage."""
        
        # Test query ID generation
        query_id = manager._generate_query_id("test_db", "SELECT * FROM test")
        assert "test_db" in query_id
        assert len(query_id.split('_')) >= 3  # db_name can have underscores, followed by timestamp_hash
        
        # Test sheet name sanitization
        sanitized = manager._sanitize_sheet_name("Sheet With Spaces & Special-Chars")
        assert sanitized == "Sheet_With_Spaces_Special_Chars"
        
        # Test with used names
        used_names = set()
        name1 = manager._sanitize_sheet_name("test", used_names)
        name2 = manager._sanitize_sheet_name("test", used_names)  # Should get unique suffix
        assert name1 != name2
        assert name1 == "test"
        assert name2 == "test_1"
        
        # Test SQL flavor detection
        mock_engine = Mock()
        mock_engine.dialect.name = 'postgresql'
        flavor = manager._get_sql_flavor("postgresql", mock_engine)
        assert flavor == "PostgreSQL"
        
        # Test safe table identifier
        safe_name = manager._safe_table_identifier("valid_table_name")
        assert "valid_table_name" in safe_name

    def test_cleanup_operations(self, manager):
        """Test cleanup and resource management."""
        
        # Test buffer cleanup
        manager._cleanup_expired_buffers()  # Should not error on empty buffers
        
        # Test auto-cleanup functionality
        manager.last_cleanup = 0  # Force cleanup on next call
        manager._cleanup_expired_buffers()
        
        # Test all cleanup
        manager._cleanup_all()


if __name__ == "__main__":
    pytest.main([__file__])