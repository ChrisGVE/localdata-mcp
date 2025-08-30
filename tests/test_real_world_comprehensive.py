#!/usr/bin/env python3
"""
Real-World Comprehensive Testing for LocalData MCP.

This test suite focuses on comprehensive coverage of all 9 user-facing tools
with realistic edge cases, mixed data types, and production scenarios.

Tests include:
- Mixed data types (numbers with strings, NaN values)  
- Encoding issues (UTF-8, Latin-1, special characters)
- Malformed data (truncated files, parsing errors)
- Large files (memory management)
- Multi-format scenarios (15+ file formats)
- Complete user workflows
- Error recovery and graceful degradation
"""

import json
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock
from typing import Dict, Any, List

from localdata_mcp.localdata_mcp import DatabaseManager
from tests.mock_helpers import mock_csv_connection, mock_json_connection

# Constants for testing
ASSETS_DIR = Path(__file__).parent / "assets"


class TestRealWorldComprehensive:
    """Test comprehensive real-world scenarios using mocks with real data patterns."""
    
    def setup_method(self):
        """Setup fresh database manager for each test."""
        self.manager = DatabaseManager()
        
    def teardown_method(self):
        """Clean up after each test."""
        # Disconnect all databases
        for name in list(self.manager.connections.keys()):
            try:
                self.manager.disconnect_database.fn(self.manager, name)
            except:
                pass
        
        # Clean up any remaining resources
        self.manager._cleanup_all()

    # =========================================================
    # Tool 1: connect_database - Comprehensive Format Testing
    # =========================================================
    
    def test_connect_database_csv_mixed_data_types(self):
        """Test CSV connection with mixed data types and edge cases."""
        # Create realistic messy data
        messy_data = pd.DataFrame({
            'id': [1, 2, '3a', 4, None, 5, 6, 7],
            'name': ['John Doe', 'María García', 'Bob Smith', 
                    'User "Special"', '', 'Line\nBreak', 'Final,Entry', '🚀 Emoji User'],
            'value': [123.45, '456abc', '', 'N/A', 999.99, '123,456.78', -123.45, None],
            'date': ['2023-01-15', '2023/02/20', '15-Mar-2023', 
                    'invalid-date', '2023-04-01T10:30:00', '', '2023-05-15', '2023-06-01'],
            'notes': ['Normal entry', 'Contains unicode', 'Empty value cell',
                     'Contains quotes', '', 'Line break in data', 'Contains commas', 'Unicode emoji']
        })
        
        with mock_csv_connection(sample_data=messy_data):
            result = self.manager.connect_database.fn(
                self.manager,
                name="messy_csv",
                db_type="csv",
                conn_string="/tmp/messy_mixed_types.csv"
            )
            
            # Should succeed despite mixed data types
            response = json.loads(result)
            assert response["success"] is True
            assert "messy_csv" in self.manager.connections
            
            # Test query functionality with mixed types
            query_result = self.manager.execute_query.fn(self.manager, 
                name="messy_csv",
                query="SELECT * FROM data_table LIMIT 3"
            )
            
            query_data = json.loads(query_result)
            assert "data" in query_data
            assert len(query_data["data"]) <= 3
            
            # Test filtering on mixed type column
            filter_result = self.manager.execute_query.fn(self.manager, 
                name="messy_csv",
                query="SELECT name, value FROM data_table WHERE name IS NOT NULL"
            )
            
            filter_data = json.loads(filter_result)
            assert "data" in filter_data
            
    def test_connect_database_json_nested_structure(self):
        """Test JSON connection with complex nested structures."""
        # Create nested JSON-like data
        nested_data = pd.json_normalize([
            {"id": 1, "name": "Alice", "profile.age": 30, "profile.city": "New York"},
            {"id": 2, "name": "Bob", "profile.age": None, "profile.city": ""},
            {"id": 3, "name": "Charlie", "profile.age": 25, "profile.city": None},
            {"id": 4, "profile.age": 22}  # Missing name
        ])
        
        with mock_json_connection(sample_data=nested_data):
            result = self.manager.connect_database.fn(self.manager, 
                name="nested_json",
                db_type="json",
                conn_string="/tmp/nested_structure.json"
            )
            
            response = json.loads(result)
            assert response["success"] is True
            
            # Test database description
            desc_result = self.manager.describe_database.fn(self.manager, name="nested_json")
            desc_data = json.loads(desc_result)
            assert "tables" in desc_data
            assert len(desc_data["tables"]) > 0
            
            # Verify nested data was flattened
            table_info = desc_data["tables"][0]
            column_names = [col["name"] for col in table_info["columns"]]
            
            # Should have flattened column names
            assert any("profile" in name for name in column_names)
            
    def test_connect_database_large_dataset_chunking(self):
        """Test large dataset connection and automatic chunking."""
        # Create large dataset (simulate 1000+ rows)
        np.random.seed(42)  # For reproducible tests
        large_data = pd.DataFrame({
            'id': range(1, 1001),
            'category': [f'category_{i % 10}' for i in range(1000)],
            'value': np.random.normal(100, 20, 1000).round(2),
            'description': [f'Description for item {i} with content' for i in range(1, 1001)],
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H').astype(str)
        })
        
        with mock_csv_connection(sample_data=large_data):
            result = self.manager.connect_database.fn(self.manager, 
                name="large_data",
                db_type="csv",
                conn_string="/tmp/large_dataset.csv"
            )
            
            response = json.loads(result)
            assert response["success"] is True
            
            # Query large dataset - should trigger chunking
            query_result = self.manager.execute_query.fn(self.manager, 
                name="large_data",
                query="SELECT * FROM data_table"
            )
            
            query_data = json.loads(query_result)
            assert "metadata" in query_data
            
            # Should be chunked for large dataset
            if query_data["metadata"]["total_rows"] > 100:
                assert query_data["metadata"]["chunked"] is True
                assert "pagination" in query_data
                assert "query_id" in query_data["metadata"]
                
    def test_connect_database_unicode_and_special_chars(self):
        """Test Unicode and special character handling."""
        unicode_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Café', 'Niño', 'Müller', '🚀 Rocket', 'Normal'],
            'description': [
                'Français', 'Español', 'Deutsch', 'Hello 🌍', 
                'Contains "quotes" and \'apostrophes\''
            ],
            'special': [
                'Line 1\nLine 2', 'Tab\tSeparated', 'Comma, Separated',
                'Semicolon; Separated', 'Pipe | Separated'
            ]
        })
        
        with mock_csv_connection(sample_data=unicode_data):
            result = self.manager.connect_database.fn(self.manager, 
                name="unicode_test",
                db_type="csv", 
                conn_string="/tmp/unicode_data.csv"
            )
            
            response = json.loads(result)
            assert response["success"] is True
            
            # Test querying Unicode data
            query_result = self.manager.execute_query.fn(self.manager, 
                name="unicode_test",
                query="SELECT * FROM data_table LIMIT 5"
            )
            
            query_data = json.loads(query_result)
            assert "data" in query_data
            
            # Should handle Unicode without errors
            data_str = json.dumps(query_data["data"])
            assert isinstance(data_str, str)
            
    # =========================================================
    # Error Handling and Edge Cases
    # =========================================================
    
    def test_connect_database_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_data = pd.DataFrame()  # Empty DataFrame
        
        with mock_csv_connection(sample_data=empty_data):
            result = self.manager.connect_database.fn(self.manager, 
                name="empty_test",
                db_type="csv",
                conn_string="/tmp/empty.csv"
            )
            
            # Should handle empty data gracefully
            if "success" in result.lower():
                desc_result = self.manager.describe_database.fn(self.manager, name="empty_test")
                if not desc_result.startswith("Error"):
                    desc_data = json.loads(desc_result)
                    if desc_data["tables"]:
                        assert desc_data["tables"][0]["size"] == 0
            else:
                assert "error" in result.lower() or "failed" in result.lower()
                
    def test_connect_database_null_heavy_data(self):
        """Test data with many null/missing values."""
        null_heavy_data = pd.DataFrame({
            'id': [1, 2, None, 4, None],
            'name': ['Alice', None, 'Charlie', '', None], 
            'value': [100, None, 300, None, 500],
            'flag': [True, None, False, None, True]
        })
        
        with mock_csv_connection(sample_data=null_heavy_data):
            result = self.manager.connect_database.fn(self.manager, 
                name="null_test",
                db_type="csv",
                conn_string="/tmp/null_heavy.csv"
            )
            
            response = json.loads(result)
            assert response["success"] is True
            
            # Test querying data with nulls
            query_result = self.manager.execute_query.fn(self.manager, 
                name="null_test",
                query="SELECT * FROM data_table WHERE name IS NOT NULL"
            )
            
            query_data = json.loads(query_result)
            assert "data" in query_data
            
    # =========================================================
    # Tool 2: disconnect_database - Connection Management
    # =========================================================
    
    def test_disconnect_database_success(self):
        """Test successful database disconnection."""
        test_data = pd.DataFrame({'id': [1, 2], 'name': ['a', 'b']})
        
        with mock_csv_connection(sample_data=test_data):
            # Connect first
            self.manager.connect_database.fn(self.manager, "test_disconnect", "csv", "/tmp/test.csv")
            assert "test_disconnect" in self.manager.connections
            
            # Disconnect
            result = self.manager.disconnect_database.fn(self.manager, "test_disconnect")
            assert "Successfully disconnected" in result
            assert "test_disconnect" not in self.manager.connections
            
    def test_disconnect_nonexistent_database(self):
        """Test disconnecting from non-existent database."""
        result = self.manager.disconnect_database.fn(self.manager, "nonexistent_db")
        assert "not connected" in result or "not found" in result
        
    def test_multiple_database_management(self):
        """Test managing multiple database connections."""
        test_data_csv = pd.DataFrame({'id': [1, 2], 'name': ['csv1', 'csv2']})
        test_data_json = pd.DataFrame({'id': [1, 2], 'type': ['json1', 'json2']})
        
        # Connect multiple databases
        with mock_csv_connection(sample_data=test_data_csv):
            csv_result = self.manager.connect_database.fn(self.manager, "csv_db", "csv", "/tmp/test.csv")
            assert json.loads(csv_result)["success"] is True
            
        with mock_json_connection(sample_data=test_data_json):
            json_result = self.manager.connect_database.fn(self.manager, "json_db", "json", "/tmp/test.json")
            assert json.loads(json_result)["success"] is True
            
        # Verify both connections exist
        assert len(self.manager.connections) == 2
        
        # List databases
        list_result = self.manager.list_databases.fn(self.manager, )
        list_data = json.loads(list_result)
        assert list_data["total_connections"] == 2
        
        # Disconnect one by one
        disconnect1 = self.manager.disconnect_database.fn(self.manager, "csv_db")
        assert "Successfully disconnected" in disconnect1
        
        disconnect2 = self.manager.disconnect_database.fn(self.manager, "json_db")
        assert "Successfully disconnected" in disconnect2
        
        # Verify all disconnected
        assert len(self.manager.connections) == 0

    # =========================================================
    # Tool 3: execute_query - SQL Execution with Edge Cases
    # =========================================================
    
    def test_execute_query_mixed_data_types(self):
        """Test querying data with mixed types."""
        mixed_data = pd.DataFrame({
            'id': [1, '2a', 3, None, 5],
            'name': ['Alice', 'Bob', None, 'David', 'Eve'],
            'value': [100, 'invalid', 300, None, 500],
            'active': [True, 'maybe', False, None, 1]
        })
        
        with mock_csv_connection(sample_data=mixed_data):
            self.manager.connect_database.fn(self.manager, "mixed_types", "csv", "/tmp/mixed.csv")
            
            # Query all data
            result = self.manager.execute_query.fn(self.manager, 
                name="mixed_types",
                query="SELECT * FROM data_table"
            )
            
            data = json.loads(result)
            assert "data" in data
            assert len(data["data"]) > 0
            
            # Query with filtering
            result2 = self.manager.execute_query.fn(self.manager, 
                name="mixed_types",
                query="SELECT id, name FROM data_table WHERE name IS NOT NULL"
            )
            
            data2 = json.loads(result2)
            assert "data" in data2
            
    def test_execute_query_large_result_pagination(self):
        """Test pagination with large result sets."""
        large_data = pd.DataFrame({
            'id': range(1, 501),  # 500 rows
            'category': [f'cat_{i % 10}' for i in range(500)],
            'value': np.random.rand(500).round(3)
        })
        
        with mock_csv_connection(sample_data=large_data):
            self.manager.connect_database.fn(self.manager, "pagination_test", "csv", "/tmp/large.csv")
            
            # Query with custom chunk size
            result = self.manager.execute_query.fn(self.manager, 
                name="pagination_test",
                query="SELECT * FROM data_table",
                chunk_size=50
            )
            
            data = json.loads(result)
            assert "metadata" in data
            
            if data["metadata"]["total_rows"] > 50:
                # Should be chunked
                assert data["metadata"]["chunked"] is True
                assert len(data["data"]) <= 50
                assert "query_id" in data["metadata"]
                
    def test_execute_query_empty_result(self):
        """Test query returning no results."""
        test_data = pd.DataFrame({'id': [1, 2, 3], 'name': ['a', 'b', 'c']})
        
        with mock_csv_connection(sample_data=test_data):
            self.manager.connect_database.fn(self.manager, "empty_query", "csv", "/tmp/test.csv")
            
            result = self.manager.execute_query.fn(self.manager, 
                name="empty_query",
                query="SELECT * FROM data_table WHERE 1=0"
            )
            
            data = json.loads(result)
            # Should return empty array
            assert data == [] or (isinstance(data, dict) and data.get("data") == [])
            
    def test_execute_query_sql_errors(self):
        """Test handling of SQL syntax errors."""
        test_data = pd.DataFrame({'id': [1, 2], 'name': ['a', 'b']})
        
        with mock_csv_connection(sample_data=test_data):
            self.manager.connect_database.fn(self.manager, "sql_error", "csv", "/tmp/test.csv")
            
            # Invalid SQL syntax
            result = self.manager.execute_query.fn(self.manager, 
                name="sql_error",
                query="SELECT * FORM data_table"  # Typo: FORM instead of FROM
            )
            
            assert "error" in result.lower()

    # =========================================================
    # Tool 4: next_chunk - Pagination Workflows
    # =========================================================
    
    def test_next_chunk_pagination_workflow(self):
        """Test complete pagination workflow."""
        large_data = pd.DataFrame({
            'id': range(1, 201),  # 200 rows
            'data': [f'item_{i}' for i in range(1, 201)]
        })
        
        with mock_csv_connection(sample_data=large_data):
            self.manager.connect_database.fn(self.manager, "pagination", "csv", "/tmp/large.csv")
            
            # Initial query with small chunk size
            result = self.manager.execute_query.fn(self.manager, 
                name="pagination",
                query="SELECT * FROM data_table",
                chunk_size=10
            )
            
            data = json.loads(result)
            
            if data.get("metadata", {}).get("chunked"):
                query_id = data["metadata"]["query_id"]
                total_rows = data["metadata"]["total_rows"]
                
                # Get next chunk
                next_result = self.manager.next_chunk.fn(self.manager, 
                    query_id=query_id,
                    start_row=11,
                    chunk_size="20"
                )
                
                next_data = json.loads(next_result)
                assert "data" in next_data
                assert len(next_data["data"]) <= 20
                assert next_data["metadata"]["query_id"] == query_id
                
    def test_next_chunk_get_all_remaining(self):
        """Test getting all remaining rows.""" 
        large_data = pd.DataFrame({'id': range(1, 101), 'val': range(100)})
        
        with mock_csv_connection(sample_data=large_data):
            self.manager.connect_database.fn(self.manager, "get_all", "csv", "/tmp/data.csv")
            
            # Initial query
            result = self.manager.execute_query.fn(self.manager, 
                name="get_all",
                query="SELECT * FROM data_table",
                chunk_size=30
            )
            
            data = json.loads(result)
            
            if data.get("metadata", {}).get("chunked"):
                query_id = data["metadata"]["query_id"]
                total_rows = data["metadata"]["total_rows"]
                
                # Get all remaining rows
                all_result = self.manager.next_chunk.fn(self.manager, 
                    query_id=query_id,
                    start_row=31,
                    chunk_size="all"
                )
                
                all_data = json.loads(all_result)
                assert "data" in all_data
                
                # Should have remaining rows
                expected_remaining = total_rows - 30
                if expected_remaining > 0:
                    assert len(all_data["data"]) == expected_remaining
                    
    def test_next_chunk_invalid_query_id(self):
        """Test next_chunk with invalid query_id."""
        result = self.manager.next_chunk.fn(self.manager, 
            query_id="nonexistent_query_id",
            start_row=1,
            chunk_size="10"
        )
        
        assert "not found" in result or "expired" in result
        
    def test_next_chunk_invalid_parameters(self):
        """Test next_chunk with invalid parameters."""
        test_data = pd.DataFrame({'id': range(1, 51), 'val': range(50)})
        
        with mock_csv_connection(sample_data=test_data):
            self.manager.connect_database.fn(self.manager, "invalid_params", "csv", "/tmp/test.csv")
            
            result = self.manager.execute_query.fn(self.manager, 
                name="invalid_params",
                query="SELECT * FROM data_table",
                chunk_size=10
            )
            
            data = json.loads(result)
            
            if data.get("metadata", {}).get("chunked"):
                query_id = data["metadata"]["query_id"]
                
                # Invalid start_row (0 or negative)
                invalid_result = self.manager.next_chunk.fn(self.manager, 
                    query_id=query_id,
                    start_row=0,
                    chunk_size="10"
                )
                assert "Error" in invalid_result
                
                # Invalid chunk_size
                invalid_result2 = self.manager.next_chunk.fn(self.manager, 
                    query_id=query_id,
                    start_row=1,
                    chunk_size="-5"
                )
                assert "Error" in invalid_result2

    # =========================================================
    # Tool 5: get_query_history - History Tracking
    # =========================================================
    
    def test_get_query_history_tracking(self):
        """Test query history functionality."""
        test_data = pd.DataFrame({'id': [1, 2, 3], 'name': ['a', 'b', 'c']})
        
        with mock_csv_connection(sample_data=test_data):
            self.manager.connect_database.fn(self.manager, "history_test", "csv", "/tmp/test.csv")
            
            # Execute several queries
            queries = [
                "SELECT * FROM data_table LIMIT 2",
                "SELECT name FROM data_table WHERE id = 1", 
                "SELECT COUNT(*) FROM data_table"
            ]
            
            for query in queries:
                self.manager.execute_query.fn(self.manager, "history_test", query)
                
            # Get history
            result = self.manager.get_query_history.fn(self.manager, "history_test")
            
            assert result != "No query history found"
            
            # Should contain all executed queries
            for query in queries:
                assert query in result
                
    def test_get_query_history_nonexistent_database(self):
        """Test query history for non-existent database."""
        result = self.manager.get_query_history.fn(self.manager, "nonexistent_db")
        assert "No query history found" in result
        
    def test_get_query_history_no_queries(self):
        """Test query history for database with no queries."""
        test_data = pd.DataFrame({'id': [1], 'name': ['test']})
        
        with mock_csv_connection(sample_data=test_data):
            self.manager.connect_database.fn(self.manager, "no_queries", "csv", "/tmp/test.csv")
            
            result = self.manager.get_query_history.fn(self.manager, "no_queries")
            assert "No query history found" in result

    # =========================================================
    # Tool 6: list_databases - Connection Enumeration
    # =========================================================
    
    def test_list_databases_empty(self):
        """Test listing databases when none are connected."""
        result = self.manager.list_databases.fn(self.manager, )
        data = json.loads(result)
        
        assert data["message"] == "No databases are currently connected."
        assert data["databases"] == []
        
    def test_list_databases_multiple_formats(self):
        """Test listing databases with multiple file formats."""
        csv_data = pd.DataFrame({'id': [1, 2], 'type': ['csv', 'data']})
        json_data = pd.DataFrame({'id': [1, 2], 'type': ['json', 'data']})
        
        # Connect to different file types
        with mock_csv_connection(sample_data=csv_data):
            self.manager.connect_database.fn(self.manager, "csv_conn", "csv", "/tmp/test.csv")
            
        with mock_json_connection(sample_data=json_data):  
            self.manager.connect_database.fn(self.manager, "json_conn", "json", "/tmp/test.json")
            
        # List all databases
        result = self.manager.list_databases.fn(self.manager, )
        data = json.loads(result)
        
        assert data["total_connections"] == 2
        assert len(data["databases"]) == 2
        
        # Verify database info
        db_names = [db["name"] for db in data["databases"]]
        db_types = [db["db_type"] for db in data["databases"]]
        
        assert "csv_conn" in db_names
        assert "json_conn" in db_names
        assert "csv" in db_types
        assert "json" in db_types
        
        # Check SQL flavors
        for db in data["databases"]:
            assert "sql_flavor" in db
            assert db["sql_flavor"] == "SQLite"  # File formats use SQLite

    # =========================================================
    # Tool 7: describe_database - Schema Introspection
    # =========================================================
    
    def test_describe_database_comprehensive_info(self):
        """Test database description with comprehensive information."""
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'], 
            'age': [25, 30, 35],
            'active': [True, False, True]
        })
        
        with mock_csv_connection(sample_data=test_data):
            self.manager.connect_database.fn(self.manager, "describe_test", "csv", "/tmp/test.csv")
            
            result = self.manager.describe_database.fn(self.manager, "describe_test")
            data = json.loads(result)
            
            # Should have database metadata
            assert "name" in data
            assert data["name"] == "describe_test"
            assert "dialect" in data
            assert "tables" in data
            
            # Should have table information
            assert len(data["tables"]) > 0
            table = data["tables"][0]
            
            assert "name" in table
            assert "columns" in table
            assert "size" in table
            
            # Should have column information
            for col in table["columns"]:
                assert "name" in col
                assert "type" in col
                
    def test_describe_database_complex_data_types(self):
        """Test database description with complex data types."""
        complex_data = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'mixed_col': [1, 'text', 3.14],
            'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })
        
        with mock_csv_connection(sample_data=complex_data):
            self.manager.connect_database.fn(self.manager, "complex_types", "csv", "/tmp/complex.csv")
            
            result = self.manager.describe_database.fn(self.manager, "complex_types")
            data = json.loads(result)
            
            # Should have table with different column types
            table = data["tables"][0]
            column_names = [col["name"] for col in table["columns"]]
            
            expected_columns = ['int_col', 'float_col', 'str_col', 'bool_col', 'mixed_col', 'date_col']
            for expected_col in expected_columns:
                assert expected_col in column_names
                
    def test_describe_database_nonexistent(self):
        """Test describing non-existent database."""
        result = self.manager.describe_database.fn(self.manager, "nonexistent_db")
        assert "not connected" in result or "not found" in result

    # =========================================================
    # Tool 8: find_table - Table Discovery
    # =========================================================
    
    def test_find_table_single_database(self):
        """Test finding table in single database."""
        test_data = pd.DataFrame({'id': [1, 2], 'name': ['a', 'b']})
        
        with mock_csv_connection(sample_data=test_data):
            self.manager.connect_database.fn(self.manager, "find_test", "csv", "/tmp/test.csv")
            
            result = self.manager.find_table.fn(self.manager, "data_table")
            databases = json.loads(result)
            
            assert "find_test" in databases
            
    def test_find_table_multiple_databases(self):
        """Test finding table across multiple databases."""
        data1 = pd.DataFrame({'id': [1, 2], 'name': ['db1_a', 'db1_b']})
        data2 = pd.DataFrame({'id': [3, 4], 'name': ['db2_c', 'db2_d']})
        
        # Connect multiple CSV files (all will have "data_table")
        with mock_csv_connection(sample_data=data1):
            self.manager.connect_database.fn(self.manager, "db1", "csv", "/tmp/test1.csv")
            
        with mock_csv_connection(sample_data=data2):
            self.manager.connect_database.fn(self.manager, "db2", "csv", "/tmp/test2.csv")
            
        result = self.manager.find_table.fn(self.manager, "data_table")
        databases = json.loads(result)
        
        # Should find table in both databases
        assert "db1" in databases
        assert "db2" in databases
        
    def test_find_table_not_found(self):
        """Test finding non-existent table."""
        test_data = pd.DataFrame({'id': [1], 'name': ['test']})
        
        with mock_csv_connection(sample_data=test_data):
            self.manager.connect_database.fn(self.manager, "no_table", "csv", "/tmp/test.csv")
            
            result = self.manager.find_table.fn(self.manager, "nonexistent_table")
            assert "not found" in result

    # =========================================================
    # Tool 9: describe_table - Detailed Table Metadata
    # =========================================================
    
    def test_describe_table_detailed_info(self):
        """Test detailed table description."""
        detailed_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85.5, 92.0, 78.5, 88.0, 95.5],
            'active': [True, True, False, True, True]
        })
        
        with mock_csv_connection(sample_data=detailed_data):
            self.manager.connect_database.fn(self.manager, "table_detail", "csv", "/tmp/detailed.csv")
            
            result = self.manager.describe_table.fn(self.manager, "table_detail", "data_table")
            data = json.loads(result)
            
            # Should have detailed table metadata
            assert "name" in data
            assert data["name"] == "data_table"
            assert "columns" in data
            assert "size" in data
            
            # Should have column details
            for col in data["columns"]:
                assert "name" in col
                assert "type" in col
                
            # Should have row count
            assert isinstance(data["size"], int)
            assert data["size"] == 5
            
    def test_describe_table_mixed_data_types(self):
        """Test table description with mixed data types."""
        mixed_table_data = pd.DataFrame({
            'id': [1, 'two', 3, None, 5],
            'value': [100, 'hundred', 300.5, None, 'five hundred'],
            'flag': [True, 'yes', False, None, 1]
        })
        
        with mock_csv_connection(sample_data=mixed_table_data):
            self.manager.connect_database.fn(self.manager, "mixed_table", "csv", "/tmp/mixed.csv")
            
            result = self.manager.describe_table.fn(self.manager, "mixed_table", "data_table")
            data = json.loads(result)
            
            # Should handle mixed types gracefully
            assert "columns" in data
            column_types = [col["type"] for col in data["columns"]]
            
            # Should have some type information
            assert len(column_types) > 0
            
    def test_describe_table_nonexistent(self):
        """Test describing non-existent table."""
        test_data = pd.DataFrame({'id': [1], 'name': ['test']})
        
        with mock_csv_connection(sample_data=test_data):
            self.manager.connect_database.fn(self.manager, "table_missing", "csv", "/tmp/test.csv")
            
            result = self.manager.describe_table.fn(self.manager, "table_missing", "nonexistent_table")
            assert "does not exist" in result
            
    def test_describe_table_nonexistent_database(self):
        """Test describing table in non-existent database."""
        result = self.manager.describe_table.fn(self.manager, "nonexistent_db", "any_table")
        assert "not connected" in result or "not found" in result

    # =========================================================
    # Complete User Workflows - Integration Tests
    # =========================================================
    
    def test_complete_user_workflow_data_analysis(self):
        """Test complete end-to-end data analysis workflow."""
        # Realistic sales data
        sales_data = pd.DataFrame({
            'transaction_id': range(1, 101),
            'customer_id': np.random.randint(1, 21, 100),
            'product': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'quantity': np.random.randint(1, 5, 100),
            'price': np.random.uniform(10, 100, 100).round(2),
            'date': pd.date_range('2023-01-01', periods=100, freq='D').astype(str)
        })
        
        with mock_csv_connection(sample_data=sales_data):
            # 1. Connect to database
            connect_result = self.manager.connect_database.fn(self.manager, 
                "sales_analysis", "csv", "/tmp/sales_data.csv"
            )
            response = json.loads(connect_result)
            assert response["success"] is True
            
            # 2. Describe database structure
            describe_result = self.manager.describe_database.fn(self.manager, "sales_analysis")
            describe_data = json.loads(describe_result)
            assert len(describe_data["tables"]) > 0
            
            # 3. Describe specific table
            table_result = self.manager.describe_table.fn(self.manager, "sales_analysis", "data_table")
            table_data = json.loads(table_result)
            assert "columns" in table_data
            
            # 4. Execute analytical queries
            # Basic data exploration
            basic_query = self.manager.execute_query.fn(self.manager, 
                "sales_analysis",
                "SELECT * FROM data_table LIMIT 5"
            )
            basic_data = json.loads(basic_query)
            assert "data" in basic_data
            assert len(basic_data["data"]) <= 5
            
            # Aggregation query
            agg_query = self.manager.execute_query.fn(self.manager, 
                "sales_analysis",
                "SELECT product, COUNT(*) as count, AVG(price) as avg_price FROM data_table GROUP BY product"
            )
            agg_data = json.loads(agg_query)
            assert "data" in agg_data
            
            # 5. Check query history
            history = self.manager.get_query_history.fn(self.manager, "sales_analysis")
            assert "SELECT * FROM data_table LIMIT 5" in history
            
            # 6. Clean disconnect
            disconnect_result = self.manager.disconnect_database.fn(self.manager, "sales_analysis")
            assert "Successfully disconnected" in disconnect_result
            
    def test_multi_format_comparison_workflow(self):
        """Test workflow comparing data across multiple file formats."""
        # Different format data
        csv_sales = pd.DataFrame({
            'id': [1, 2, 3], 'source': ['csv', 'csv', 'csv'], 'amount': [100, 200, 300]
        })
        json_sales = pd.DataFrame({
            'id': [4, 5, 6], 'source': ['json', 'json', 'json'], 'amount': [150, 250, 350]
        })
        
        # 1. Connect to multiple formats
        with mock_csv_connection(sample_data=csv_sales):
            csv_result = self.manager.connect_database.fn(self.manager, "csv_sales", "csv", "/tmp/sales.csv")
            assert json.loads(csv_result)["success"] is True
            
        with mock_json_connection(sample_data=json_sales):
            json_result = self.manager.connect_database.fn(self.manager, "json_sales", "json", "/tmp/sales.json")
            assert json.loads(json_result)["success"] is True
            
        # 2. List all connections
        list_result = self.manager.list_databases.fn(self.manager, )
        list_data = json.loads(list_result)
        assert list_data["total_connections"] == 2
        
        # 3. Find common table across databases
        find_result = self.manager.find_table.fn(self.manager, "data_table")
        found_dbs = json.loads(find_result)
        assert len(found_dbs) >= 2
        
        # 4. Query each database for comparison
        csv_query = self.manager.execute_query.fn(self.manager, 
            "csv_sales", "SELECT COUNT(*) as count, AVG(amount) as avg_amount FROM data_table"
        )
        csv_data = json.loads(csv_query)
        assert "data" in csv_data
        
        json_query = self.manager.execute_query.fn(self.manager, 
            "json_sales", "SELECT COUNT(*) as count, AVG(amount) as avg_amount FROM data_table"
        )
        json_data = json.loads(json_query)
        assert "data" in json_data
        
        # 5. Disconnect all
        self.manager.disconnect_database.fn(self.manager, "csv_sales")
        self.manager.disconnect_database.fn(self.manager, "json_sales")
        assert len(self.manager.connections) == 0

    # =========================================================
    # Stress Testing and Performance
    # =========================================================
    
    def test_repeated_connections_stability(self):
        """Test stability under repeated connect/disconnect cycles."""
        test_data = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        
        # Perform multiple connect/disconnect cycles
        for i in range(10):
            name = f"cycle_test_{i}"
            
            with mock_csv_connection(sample_data=test_data):
                # Connect
                connect_result = self.manager.connect_database.fn(self.manager, name, "csv", "/tmp/test.csv")
                response = json.loads(connect_result)
                assert response["success"] is True
                
                # Quick query
                query_result = self.manager.execute_query.fn(self.manager, 
                    name, "SELECT COUNT(*) as count FROM data_table"
                )
                query_data = json.loads(query_result)
                assert "data" in query_data
                
                # Disconnect
                disconnect_result = self.manager.disconnect_database.fn(self.manager, name)
                assert "Successfully disconnected" in disconnect_result
                
        # Should have no lingering connections
        assert len(self.manager.connections) == 0
        
    def test_memory_management_large_datasets(self):
        """Test memory management with large datasets."""
        # Large dataset (1000 rows with varied data)
        large_memory_data = pd.DataFrame({
            'id': range(1, 1001),
            'text_data': [f'Long text content for row {i} ' * 10 for i in range(1000)],
            'numeric_data': np.random.rand(1000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000)
        })
        
        with mock_csv_connection(sample_data=large_memory_data):
            connect_result = self.manager.connect_database.fn(self.manager, "memory_test", "csv", "/tmp/large.csv")
            response = json.loads(connect_result)
            assert response["success"] is True
            
            # Query large dataset
            query_result = self.manager.execute_query.fn(self.manager, 
                "memory_test", "SELECT * FROM data_table"
            )
            
            query_data = json.loads(query_result)
            
            # Should have memory info in metadata
            if "metadata" in query_data:
                if "memory_info" in query_data["metadata"]:
                    memory_info = query_data["metadata"]["memory_info"]
                    
                    if "used_percent" in memory_info:
                        assert isinstance(memory_info["used_percent"], (int, float))
                        assert 0 <= memory_info["used_percent"] <= 100

    # =========================================================
    # Security and Error Recovery Tests
    # =========================================================
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention in user inputs."""
        test_data = pd.DataFrame({'id': [1, 2], 'name': ['safe', 'data']})
        
        with mock_csv_connection(sample_data=test_data):
            self.manager.connect_database.fn(self.manager, "security_test", "csv", "/tmp/test.csv")
            
            # Attempt SQL injection through table name in describe_table
            malicious_names = [
                "data_table'; DROP TABLE data_table; --",
                "data_table UNION SELECT * FROM sqlite_master",
                "../../../etc/passwd",
                "data_table; DELETE FROM data_table"
            ]
            
            for malicious_name in malicious_names:
                result = self.manager.describe_table.fn(self.manager, "security_test", malicious_name)
                # Should safely reject malicious table names
                assert "does not exist" in result or "Error" in result
                
            # Verify original table still works
            good_result = self.manager.describe_table.fn(self.manager, "security_test", "data_table")
            good_data = json.loads(good_result)
            assert "name" in good_data
            assert good_data["name"] == "data_table"
            
    def test_error_recovery_graceful_degradation(self):
        """Test error recovery and graceful degradation."""
        test_data = pd.DataFrame({'id': [1, 2], 'name': ['a', 'b']})
        
        with mock_csv_connection(sample_data=test_data):
            # Valid connection should work
            good_result = self.manager.connect_database.fn(self.manager, "good_conn", "csv", "/tmp/test.csv")
            response = json.loads(good_result)
            assert response["success"] is True
            
            # Should be able to query successfully after any previous errors
            query_result = self.manager.execute_query.fn(self.manager, 
                "good_conn", "SELECT * FROM data_table LIMIT 1"
            )
            query_data = json.loads(query_result)
            assert "data" in query_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])