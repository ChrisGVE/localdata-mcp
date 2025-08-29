"""Comprehensive tests for the streamlined LocalData MCP Server architecture.

Tests the 9 core tools with enhanced functionality:
1. connect_database (with SQL flavor detection)
2. disconnect_database  
3. execute_query (with chunking and memory management)
4. next_chunk (new pagination tool)
5. list_databases (with SQL flavor info)
6. describe_database
7. find_table
8. describe_table
9. get_query_history
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from localdata_mcp import DatabaseManager


class TestStreamlinedArchitecture:
    """Test the 9 core tools of the streamlined architecture."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh DatabaseManager instance for testing."""
        return DatabaseManager()
    
    @pytest.fixture
    def test_csv_file(self):
        """Create a test CSV file."""
        df = pd.DataFrame({
            'id': range(1, 151),  # 150 rows to test chunking
            'name': [f'user_{i}' for i in range(1, 151)],
            'value': [i * 10 for i in range(1, 151)]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            return f.name
    
    @pytest.fixture  
    def test_json_file(self):
        """Create a test JSON file."""
        data = [
            {'category': 'A', 'count': 10, 'active': True},
            {'category': 'B', 'count': 20, 'active': False},
            {'category': 'C', 'count': 15, 'active': True}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            return f.name


class TestConnectDatabase:
    """Test the enhanced connect_database tool with SQL flavor detection."""
    
    @pytest.fixture
    def manager(self):
        return DatabaseManager()
    
    def test_connect_csv_with_sql_flavor(self, manager, test_csv_file):
        """Test connecting to CSV with SQL flavor detection."""
        result = manager.connect_database("test_csv", "csv", test_csv_file)
        
        # Parse JSON response
        response = json.loads(result)
        
        assert response["success"] is True
        assert response["message"] == "Successfully connected to database 'test_csv'"
        assert response["connection_info"]["name"] == "test_csv"
        assert response["connection_info"]["db_type"] == "csv"
        assert response["connection_info"]["sql_flavor"] == "SQLite"
        assert response["connection_info"]["total_connections"] == 1
    
    def test_connect_json_with_sql_flavor(self, manager, test_json_file):
        """Test connecting to JSON with SQL flavor detection."""
        result = manager.connect_database("test_json", "json", test_json_file)
        
        response = json.loads(result)
        
        assert response["success"] is True
        assert response["connection_info"]["sql_flavor"] == "SQLite"
        assert response["connection_info"]["db_type"] == "json"
    
    def test_connect_duplicate_name(self, manager, test_csv_file):
        """Test connecting with duplicate database name."""
        # First connection should succeed
        result1 = manager.connect_database("test_db", "csv", test_csv_file)
        response1 = json.loads(result1)
        assert response1["success"] is True
        
        # Second connection with same name should fail
        result2 = manager.connect_database("test_db", "csv", test_csv_file)
        assert "already connected" in result2
    
    def test_connect_with_sheet_name(self, manager):
        """Test connecting to Excel file with specific sheet name."""
        # Create a test Excel file with multiple sheets
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            excel_path = f.name
        
        try:
            # Create Excel file with multiple sheets
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
                df2 = pd.DataFrame({'colA': [5, 6], 'colB': [7, 8]})
                df1.to_sheet(writer, sheet_name='Sheet1', index=False)
                df2.to_sheet(writer, sheet_name='Sheet2', index=False)
            
            result = manager.connect_database("excel_test", "excel", excel_path, sheet_name="Sheet1")
            
            if "Successfully connected" in result:
                response = json.loads(result)
                assert response["success"] is True
                assert response["connection_info"]["sql_flavor"] == "SQLite"
        
        finally:
            if os.path.exists(excel_path):
                os.unlink(excel_path)


class TestExecuteQueryWithMemoryManagement:
    """Test the enhanced execute_query tool with chunking and memory management."""
    
    @pytest.fixture
    def manager(self):
        return DatabaseManager()
    
    @pytest.fixture
    def connected_manager(self, manager, test_csv_file):
        """Manager with a connected CSV database."""
        manager.connect_database("test_db", "csv", test_csv_file)
        return manager
    
    def test_execute_small_query_no_chunking(self, connected_manager):
        """Test query with small result set (no chunking)."""
        result = connected_manager.execute_query("test_db", "SELECT * FROM data_table LIMIT 5")
        
        response = json.loads(result)
        
        assert "metadata" in response
        assert response["metadata"]["total_rows"] == 5
        assert response["metadata"]["chunked"] is False
        assert "memory_info" in response["metadata"]
        assert len(response["data"]) == 5
        
        # Should contain id, name, value columns
        first_row = response["data"][0]
        assert "id" in first_row
        assert "name" in first_row
        assert "value" in first_row
    
    def test_execute_large_query_with_chunking(self, connected_manager):
        """Test query with large result set (automatic chunking)."""
        result = connected_manager.execute_query("test_db", "SELECT * FROM data_table")
        
        response = json.loads(result)
        
        assert "metadata" in response
        assert response["metadata"]["total_rows"] == 150  # Our test data has 150 rows
        assert response["metadata"]["chunked"] is True
        assert "query_id" in response["metadata"]
        assert "pagination" in response
        
        # Should show first chunk (default 10 rows for large results)
        assert len(response["data"]) == 10
        assert response["metadata"]["showing_rows"] == "1-10"
        
        # Check pagination instructions
        pagination = response["pagination"]
        assert "use_next_chunk" in pagination
        assert "get_all_remaining" in pagination
        assert response["metadata"]["query_id"] in pagination["use_next_chunk"]
    
    def test_execute_query_custom_chunk_size(self, connected_manager):
        """Test query with custom chunk size."""
        result = connected_manager.execute_query("test_db", "SELECT * FROM data_table", chunk_size=25)
        
        response = json.loads(result)
        
        assert response["metadata"]["chunked"] is True
        assert len(response["data"]) == 25
        assert response["metadata"]["showing_rows"] == "1-25"
        assert response["metadata"]["chunk_size"] == 25
    
    @patch('localdata_mcp.localdata_mcp.psutil.virtual_memory')
    def test_execute_query_with_low_memory(self, mock_memory, connected_manager):
        """Test query execution with low memory condition."""
        # Mock high memory usage
        mock_memory.return_value = MagicMock(
            total=8 * 1024**3,  # 8GB
            available=1 * 1024**3,  # 1GB available
            percent=87.5  # High usage
        )
        
        result = connected_manager.execute_query("test_db", "SELECT COUNT(*) FROM data_table")
        
        response = json.loads(result)
        
        assert "memory_info" in response["metadata"]
        assert response["metadata"]["memory_info"]["low_memory"] is True
        assert response["metadata"]["memory_info"]["used_percent"] == 87.5
    
    def test_execute_query_empty_result(self, connected_manager):
        """Test query that returns no results."""
        result = connected_manager.execute_query("test_db", "SELECT * FROM data_table WHERE id > 1000")
        
        # Should return empty JSON array
        response = json.loads(result)
        assert response == []
    
    def test_execute_query_nonexistent_database(self, manager):
        """Test query on non-existent database."""
        result = manager.execute_query("nonexistent", "SELECT * FROM table")
        
        assert "not connected" in result
        assert "Use 'connect_database' first" in result


class TestNextChunk:
    """Test the new next_chunk pagination tool."""
    
    @pytest.fixture
    def manager_with_buffered_query(self, manager, test_csv_file):
        """Manager with a large query result buffered."""
        manager.connect_database("test_db", "csv", test_csv_file)
        # Execute large query to create buffer
        result = manager.execute_query("test_db", "SELECT * FROM data_table")
        response = json.loads(result)
        
        # Return manager and query_id
        return manager, response["metadata"]["query_id"]
    
    def test_next_chunk_basic(self, manager_with_buffered_query):
        """Test basic next_chunk functionality."""
        manager, query_id = manager_with_buffered_query
        
        # Get next chunk starting from row 11
        result = manager.next_chunk(query_id, start_row=11, chunk_size="10")
        
        response = json.loads(result)
        
        assert "metadata" in response
        assert response["metadata"]["query_id"] == query_id
        assert response["metadata"]["total_rows"] == 150
        assert response["metadata"]["showing_rows"] == "11-20"
        assert len(response["data"]) == 10
        
        # Check data is correct (should be rows 11-20)
        first_row = response["data"][0]
        assert first_row["id"] == 11  # 11th row has id=11
    
    def test_next_chunk_get_all_remaining(self, manager_with_buffered_query):
        """Test getting all remaining rows."""
        manager, query_id = manager_with_buffered_query
        
        result = manager.next_chunk(query_id, start_row=11, chunk_size="all")
        
        response = json.loads(result)
        
        assert len(response["data"]) == 140  # Rows 11-150 = 140 rows
        assert response["metadata"]["showing_rows"] == "11-150"
        
        # Should not have pagination since all remaining rows returned
        assert "pagination" not in response
    
    def test_next_chunk_pagination_continues(self, manager_with_buffered_query):
        """Test that pagination instructions are provided when more rows available."""
        manager, query_id = manager_with_buffered_query
        
        result = manager.next_chunk(query_id, start_row=11, chunk_size="50")
        
        response = json.loads(result)
        
        assert len(response["data"]) == 50
        assert response["metadata"]["showing_rows"] == "11-60"
        
        # Should have pagination for remaining rows
        assert "pagination" in response
        pagination = response["pagination"]
        assert "next_100" in pagination
        assert "get_all_remaining" in pagination
        assert "start_row=61" in pagination["next_100"]
    
    def test_next_chunk_invalid_query_id(self, manager):
        """Test next_chunk with invalid query ID."""
        result = manager.next_chunk("invalid_query_id", start_row=1, chunk_size="10")
        
        assert "not found" in result
        assert "expired" in result or "cleared" in result
    
    def test_next_chunk_invalid_start_row(self, manager_with_buffered_query):
        """Test next_chunk with invalid start row."""
        manager, query_id = manager_with_buffered_query
        
        # Test start_row too high
        result = manager.next_chunk(query_id, start_row=200, chunk_size="10")
        assert "start_row must be between 1 and 150" in result
        
        # Test start_row too low
        result = manager.next_chunk(query_id, start_row=0, chunk_size="10")
        assert "start_row must be between 1 and 150" in result
    
    def test_next_chunk_invalid_chunk_size(self, manager_with_buffered_query):
        """Test next_chunk with invalid chunk size."""
        manager, query_id = manager_with_buffered_query
        
        result = manager.next_chunk(query_id, start_row=1, chunk_size="invalid")
        assert "chunk_size must be a positive integer" in result
        
        result = manager.next_chunk(query_id, start_row=1, chunk_size="-5")
        assert "chunk_size must be a positive integer" in result
    
    def test_next_chunk_no_more_rows(self, manager_with_buffered_query):
        """Test next_chunk when no more rows available."""
        manager, query_id = manager_with_buffered_query
        
        # Get from beyond the last row
        result = manager.next_chunk(query_id, start_row=151, chunk_size="10")
        
        response = json.loads(result)
        assert response["metadata"]["message"] == "No more rows available"
        assert response["data"] == []


class TestListDatabasesWithFlavor:
    """Test the enhanced list_databases tool with SQL flavor information."""
    
    @pytest.fixture
    def manager(self):
        return DatabaseManager()
    
    def test_list_databases_empty(self, manager):
        """Test listing databases when none are connected."""
        result = manager.list_databases()
        
        response = json.loads(result)
        
        assert response["message"] == "No databases are currently connected."
        assert response["databases"] == []
    
    def test_list_databases_with_connections(self, manager, test_csv_file, test_json_file):
        """Test listing databases with multiple connections."""
        # Connect to multiple databases
        manager.connect_database("csv_db", "csv", test_csv_file)
        manager.connect_database("json_db", "json", test_json_file)
        
        result = manager.list_databases()
        
        response = json.loads(result)
        
        assert response["total_connections"] == 2
        assert len(response["databases"]) == 2
        
        # Check database info includes SQL flavor
        db_names = [db["name"] for db in response["databases"]]
        assert "csv_db" in db_names
        assert "json_db" in db_names
        
        for db in response["databases"]:
            assert "name" in db
            assert "db_type" in db
            assert "sql_flavor" in db
            assert db["sql_flavor"] == "SQLite"  # File formats use SQLite


class TestDescribeDatabase:
    """Test the describe_database tool."""
    
    @pytest.fixture
    def connected_manager(self, manager, test_csv_file):
        manager.connect_database("test_db", "csv", test_csv_file)
        return manager
    
    def test_describe_database_basic(self, connected_manager):
        """Test basic database description."""
        result = connected_manager.describe_database("test_db")
        
        response = json.loads(result)
        
        assert response["name"] == "test_db"
        assert "dialect" in response
        assert "tables" in response
        
        # Should have our data_table
        table_names = [table["name"] for table in response["tables"]]
        assert "data_table" in table_names
        
        # Find our table and verify structure
        data_table = next(table for table in response["tables"] if table["name"] == "data_table")
        assert data_table["size"] == 150  # Our test data has 150 rows
        
        # Should have id, name, value columns
        col_names = [col["name"] for col in data_table["columns"]]
        assert "id" in col_names
        assert "name" in col_names
        assert "value" in col_names


class TestFindTable:
    """Test the find_table tool."""
    
    @pytest.fixture
    def manager_with_multiple_dbs(self, manager, test_csv_file, test_json_file):
        manager.connect_database("csv_db", "csv", test_csv_file)
        manager.connect_database("json_db", "json", test_json_file)
        return manager
    
    def test_find_table_exists(self, manager_with_multiple_dbs):
        """Test finding a table that exists."""
        result = manager_with_multiple_dbs.find_table("data_table")
        
        databases = json.loads(result)
        
        # data_table should exist in both databases
        assert "csv_db" in databases
        assert "json_db" in databases
    
    def test_find_table_not_exists(self, manager_with_multiple_dbs):
        """Test finding a table that doesn't exist."""
        result = manager_with_multiple_dbs.find_table("nonexistent_table")
        
        assert "was not found" in result


class TestDescribeTable:
    """Test the describe_table tool."""
    
    @pytest.fixture
    def connected_manager(self, manager, test_csv_file):
        manager.connect_database("test_db", "csv", test_csv_file)
        return manager
    
    def test_describe_table_basic(self, connected_manager):
        """Test basic table description."""
        result = connected_manager.describe_table("test_db", "data_table")
        
        response = json.loads(result)
        
        assert response["name"] == "data_table"
        assert response["size"] == 150
        assert "columns" in response
        
        # Should have our expected columns
        col_names = [col["name"] for col in response["columns"]]
        assert "id" in col_names
        assert "name" in col_names  
        assert "value" in col_names
        
        # Check column types
        for col in response["columns"]:
            assert "name" in col
            assert "type" in col
    
    def test_describe_nonexistent_table(self, connected_manager):
        """Test describing a table that doesn't exist."""
        result = connected_manager.describe_table("test_db", "nonexistent_table")
        
        assert "does not exist" in result


class TestGetQueryHistory:
    """Test the get_query_history tool."""
    
    @pytest.fixture
    def connected_manager_with_history(self, manager, test_csv_file):
        manager.connect_database("test_db", "csv", test_csv_file)
        
        # Execute some queries to build history
        manager.execute_query("test_db", "SELECT COUNT(*) FROM data_table")
        manager.execute_query("test_db", "SELECT * FROM data_table LIMIT 5")
        manager.execute_query("test_db", "SELECT name FROM data_table WHERE id < 10")
        
        return manager
    
    def test_get_query_history_with_queries(self, connected_manager_with_history):
        """Test getting query history when queries have been executed."""
        result = connected_manager_with_history.get_query_history("test_db")
        
        # Should contain our executed queries
        assert "SELECT COUNT(*) FROM data_table" in result
        assert "SELECT * FROM data_table LIMIT 5" in result
        assert "SELECT name FROM data_table WHERE id < 10" in result
        
        # Should be separated by newlines
        lines = result.split('\n')
        assert len(lines) == 3
    
    def test_get_query_history_empty(self, manager, test_csv_file):
        """Test getting query history when no queries executed."""
        manager.connect_database("test_db", "csv", test_csv_file)
        
        result = manager.get_query_history("test_db")
        
        assert "No query history" in result
    
    def test_get_query_history_nonexistent_db(self, manager):
        """Test getting query history for non-existent database."""
        result = manager.get_query_history("nonexistent_db")
        
        assert "No query history found" in result


class TestDisconnectDatabase:
    """Test the disconnect_database tool."""
    
    @pytest.fixture
    def connected_manager(self, manager, test_csv_file):
        manager.connect_database("test_db", "csv", test_csv_file)
        return manager
    
    def test_disconnect_database_success(self, connected_manager):
        """Test successful database disconnection."""
        result = connected_manager.disconnect_database("test_db")
        
        assert "Successfully disconnected from database 'test_db'" in result
        
        # Database should no longer be in connections
        assert "test_db" not in connected_manager.connections
        assert "test_db" not in connected_manager.db_types
        assert "test_db" not in connected_manager.query_history
    
    def test_disconnect_nonexistent_database(self, manager):
        """Test disconnecting from non-existent database."""
        result = manager.disconnect_database("nonexistent_db")
        
        assert "not connected" in result


class TestMemoryManagement:
    """Test memory management features across tools."""
    
    @pytest.fixture
    def manager(self):
        return DatabaseManager()
    
    @patch('localdata_mcp.localdata_mcp.psutil.virtual_memory')
    def test_auto_buffer_clearing_on_low_memory(self, mock_memory, manager, test_csv_file):
        """Test automatic buffer clearing when memory is low."""
        # Mock high memory usage
        mock_memory.return_value = MagicMock(
            total=8 * 1024**3,
            available=1 * 1024**3,
            percent=90  # Very high usage
        )
        
        manager.connect_database("test_db", "csv", test_csv_file)
        
        # Create some buffers first
        result1 = manager.execute_query("test_db", "SELECT * FROM data_table")
        query_id1 = json.loads(result1)["metadata"]["query_id"]
        
        # Verify buffer exists
        assert query_id1 in manager.query_buffers
        
        # Execute another query - should trigger auto-clearing
        result2 = manager.execute_query("test_db", "SELECT * FROM data_table WHERE id > 50")
        
        # Original buffer should be cleared due to low memory
        assert query_id1 not in manager.query_buffers
    
    def test_memory_info_in_responses(self, manager, test_csv_file):
        """Test that memory info is included in query responses."""
        manager.connect_database("test_db", "csv", test_csv_file)
        
        result = manager.execute_query("test_db", "SELECT COUNT(*) FROM data_table")
        response = json.loads(result)
        
        assert "memory_info" in response["metadata"]
        memory_info = response["metadata"]["memory_info"]
        
        # Should have memory statistics
        assert "total_gb" in memory_info or "error" in memory_info
        if "total_gb" in memory_info:
            assert "available_gb" in memory_info
            assert "used_percent" in memory_info
            assert "low_memory" in memory_info


class TestErrorHandling:
    """Test error handling across all tools."""
    
    @pytest.fixture
    def manager(self):
        return DatabaseManager()
    
    def test_sql_injection_prevention(self, manager, test_csv_file):
        """Test SQL injection prevention."""
        manager.connect_database("test_db", "csv", test_csv_file) 
        
        # Test malicious queries
        malicious_queries = [
            "SELECT * FROM data_table; DROP TABLE data_table; --",
            "SELECT * FROM data_table UNION SELECT * FROM sqlite_master",
            "'; DELETE FROM data_table; --"
        ]
        
        for query in malicious_queries:
            result = manager.execute_query("test_db", query)
            # Should either fail safely or return controlled results
            assert "error" in result.lower() or isinstance(json.loads(result), (dict, list))
    
    def test_file_path_security(self, manager):
        """Test file path security validation."""
        dangerous_paths = [
            "../etc/passwd",
            "/etc/passwd", 
            "../../secrets.txt",
            "/dev/null"
        ]
        
        for path in dangerous_paths:
            result = manager.connect_database("test", "csv", path)
            assert "Failed to connect" in result or "outside the allowed directory" in result
    
    def test_concurrent_access_safety(self, manager, test_csv_file):
        """Test thread safety for concurrent operations."""
        import threading
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                conn_name = f"worker_{worker_id}"
                result = manager.connect_database(conn_name, "csv", test_csv_file)
                results.append(result)
                
                if "Successfully connected" in result:
                    query_result = manager.execute_query(conn_name, "SELECT COUNT(*) FROM data_table")
                    results.append(query_result)
                    manager.disconnect_database(conn_name)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access gracefully
        assert len(errors) == 0 or all("connection limit" in error.lower() for error in errors)


class TestIntegrationScenarios:
    """Test complete workflow scenarios using multiple tools."""
    
    @pytest.fixture
    def manager(self):
        return DatabaseManager()
    
    def test_complete_data_analysis_workflow(self, manager, test_csv_file):
        """Test a complete data analysis workflow."""
        # Step 1: Connect to database
        result = manager.connect_database("analysis_db", "csv", test_csv_file)
        response = json.loads(result)
        assert response["success"] is True
        
        # Step 2: List databases to confirm connection
        list_result = manager.list_databases()
        list_response = json.loads(list_result)
        assert list_response["total_connections"] == 1
        
        # Step 3: Describe database to understand schema
        describe_result = manager.describe_database("analysis_db")
        describe_response = json.loads(describe_result)
        assert len(describe_response["tables"]) >= 1
        
        # Step 4: Find our table
        find_result = manager.find_table("data_table")
        assert "analysis_db" in json.loads(find_result)
        
        # Step 5: Describe the table structure
        table_result = manager.describe_table("analysis_db", "data_table")
        table_response = json.loads(table_result)
        assert table_response["size"] == 150
        
        # Step 6: Execute analysis queries
        count_result = manager.execute_query("analysis_db", "SELECT COUNT(*) FROM data_table")
        count_response = json.loads(count_result)
        assert count_response["data"][0]["COUNT(*)"] == 150
        
        # Step 7: Execute large query with chunking
        large_result = manager.execute_query("analysis_db", "SELECT * FROM data_table")
        large_response = json.loads(large_result)
        assert large_response["metadata"]["chunked"] is True
        query_id = large_response["metadata"]["query_id"]
        
        # Step 8: Use pagination to get more data
        chunk_result = manager.next_chunk(query_id, start_row=11, chunk_size="20")
        chunk_response = json.loads(chunk_result)
        assert len(chunk_response["data"]) == 20
        
        # Step 9: Check query history
        history_result = manager.get_query_history("analysis_db")
        assert "SELECT COUNT(*)" in history_result
        assert "SELECT * FROM data_table" in history_result
        
        # Step 10: Disconnect cleanly
        disconnect_result = manager.disconnect_database("analysis_db")
        assert "Successfully disconnected" in disconnect_result
    
    def test_multiple_databases_workflow(self, manager, test_csv_file, test_json_file):
        """Test workflow with multiple databases."""
        # Connect to multiple sources
        manager.connect_database("csv_source", "csv", test_csv_file)
        manager.connect_database("json_source", "json", test_json_file)
        
        # List all connections
        list_result = manager.list_databases()
        list_response = json.loads(list_result)
        assert list_response["total_connections"] == 2
        
        # Query both databases
        csv_result = manager.execute_query("csv_source", "SELECT COUNT(*) as csv_count FROM data_table")
        json_result = manager.execute_query("json_source", "SELECT COUNT(*) as json_count FROM data_table")
        
        csv_response = json.loads(csv_result)
        json_response = json.loads(json_result)
        
        assert csv_response["data"][0]["csv_count"] == 150
        assert json_response["data"][0]["json_count"] == 3
        
        # Clean up both connections
        manager.disconnect_database("csv_source")
        manager.disconnect_database("json_source")
        
        # Verify all disconnected
        final_list = manager.list_databases()
        final_response = json.loads(final_list)
        assert final_response["databases"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])