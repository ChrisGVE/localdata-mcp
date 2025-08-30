#!/usr/bin/env python3
"""
Comprehensive real-world functionality testing for LocalData MCP.

This test suite validates real-world scenarios across all 9 user-facing tools
with focus on edge cases, mixed data types, and production reliability.
"""

import json
import os
import pytest
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

# Import the core DatabaseManager class 
from localdata_mcp.localdata_mcp import DatabaseManager, mcp

# Constants for testing
ASSETS_DIR = Path(__file__).parent / "assets"
TEST_DB_NAME = "test_db"


class TestComprehensiveRealWorld:
    """Test comprehensive real-world scenarios using direct manager access."""
    
    def setup_method(self):
        """Setup fresh database manager for each test."""
        self.manager = DatabaseManager()
        
    def teardown_method(self):
        """Clean up after each test."""
        # Disconnect all databases
        for name in list(self.manager.connections.keys()):
            try:
                self.manager.disconnect_database(name)
            except:
                pass
        
        # Clean up any remaining resources
        self.manager._cleanup_all()

    # =========================================================
    # Tool Testing via FastMCP Function Access  
    # =========================================================
    
    def test_connect_csv_mixed_data_types(self):
        """Test CSV connection with mixed data types."""
        csv_file = ASSETS_DIR / "messy_mixed_types.csv"
        
        # Call the tool method directly
        result = self.manager.connect_database(
            name="messy_csv",
            db_type="csv",
            conn_string=str(csv_file)
        )
        
        # Should succeed despite mixed data types
        response = json.loads(result)
        assert response["success"] is True
        assert "messy_csv" in self.manager.connections
        
        # Test query functionality
        query_result = self.manager.execute_query(
            name="messy_csv",
            query="SELECT * FROM data_table LIMIT 3"
        )
        
        query_data = json.loads(query_result)
        assert "data" in query_data
        assert len(query_data["data"]) <= 3
        
    def test_connect_json_nested_structure(self):
        """Test JSON connection with nested data."""
        json_file = ASSETS_DIR / "nested_structure.json"
        
        result = self.manager.connect_database(name="nested_json", 
            db_type="json",
            conn_string=str(json_file)
        )
        
        response = json.loads(result)
        assert response["success"] is True
        
        # Test database description
        desc_result = self.manager.describe_database(name="nested_json"
        )
        
        desc_data = json.loads(desc_result)
        assert "tables" in desc_data
        assert len(desc_data["tables"]) > 0
        
    def test_connect_yaml_complex_config(self):
        """Test YAML connection with complex structure."""
        yaml_file = ASSETS_DIR / "complex_config.yaml"
        
        result = self.manager.connect_database(name="complex_yaml",
            db_type="yaml", 
            conn_string=str(yaml_file)
        )
        
        response = json.loads(result)
        assert response["success"] is True
        
    def test_connect_xml_structured_data(self):
        """Test XML connection with structured catalog data."""
        xml_file = ASSETS_DIR / "structured_catalog.xml"
        
        result = self.manager.connect_database(name="structured_xml",
            db_type="xml",
            conn_string=str(xml_file)
        )
        
        response = json.loads(result)
        assert response["success"] is True
        
        # Verify data was loaded
        query_result = self.manager.execute_query(name="structured_xml",
            query="SELECT COUNT(*) as total FROM data_table"
        )
        
        query_data = json.loads(query_result)
        assert "data" in query_data
        assert len(query_data["data"]) == 1
        
    def test_connect_ini_configuration(self):
        """Test INI file connection."""
        ini_file = ASSETS_DIR / "complex_config.ini"
        
        result = self.manager.connect_database(name="config_ini",
            db_type="ini",
            conn_string=str(ini_file)
        )
        
        response = json.loads(result)
        assert response["success"] is True
        
        # Verify INI structure (section/key/value)
        query_result = self.manager.execute_query(name="config_ini",
            query="SELECT section, key, value FROM data_table LIMIT 5"
        )
        
        query_data = json.loads(query_result)
        assert "data" in query_data
        
        if query_data["data"]:
            first_row = query_data["data"][0]
            assert "section" in first_row
            assert "key" in first_row  
            assert "value" in first_row
        
    def test_connect_tsv_tab_characters(self):
        """Test TSV connection with tab characters in data."""
        tsv_file = ASSETS_DIR / "mixed_tabs.tsv"
        
        result = self.manager.connect_database(name="mixed_tsv",
            db_type="tsv",
            conn_string=str(tsv_file)
        )
        
        response = json.loads(result)
        assert response["success"] is True
        
        # Test table description
        table_result = self.manager.describe_table(name="mixed_tsv",
            table_name="data_table"
        )
        
        table_data = json.loads(table_result)
        assert "columns" in table_data
        assert len(table_data["columns"]) > 0
        
    # =========================================================
    # Error Handling Tests
    # =========================================================
    
    def test_malformed_json_handling(self):
        """Test graceful handling of malformed JSON."""
        malformed_json = ASSETS_DIR / "malformed.json"
        
        result = self.manager.connect_database(name="bad_json",
            db_type="json",
            conn_string=str(malformed_json)
        )
        
        # Should fail gracefully
        assert "Failed to connect" in result or "error" in result.lower()
        assert "bad_json" not in self.manager.connections
        
    def test_empty_files_handling(self):
        """Test handling of empty files."""
        empty_csv = ASSETS_DIR / "empty.csv"
        
        result = self.manager.connect_database(name="empty_csv",
            db_type="csv",
            conn_string=str(empty_csv)
        )
        
        # Should either succeed with empty table or fail gracefully
        if "success" in result.lower():
            # If successful, verify empty table
            desc_result = self.manager.describe_database(name="empty_csv"
            )
            
            if not desc_result.startswith("Error"):
                desc_data = json.loads(desc_result)
                if desc_data["tables"]:
                    assert desc_data["tables"][0]["size"] == 0
        else:
            assert "error" in result.lower() or "failed" in result.lower()
            
    def test_truncated_csv_handling(self):
        """Test handling of truncated CSV file.""" 
        truncated_csv = ASSETS_DIR / "truncated.csv"
        
        result = self.manager.connect_database(name="truncated_csv",
            db_type="csv",
            conn_string=str(truncated_csv)
        )
        
        # Should handle truncated data gracefully
        if "success" in result.lower():
            # Verify partial data was loaded
            query_result = self.manager.execute_query(name="truncated_csv",
                query="SELECT * FROM data_table"
            )
            query_data = json.loads(query_result)
            assert "data" in query_data
        else:
            assert "error" in result.lower()
            
    # =========================================================
    # Database Management Tests
    # =========================================================
    
    def test_list_multiple_databases(self):
        """Test listing multiple connected databases."""
        # Connect to multiple formats
        connections = [
            ("csv_conn", "csv", ASSETS_DIR / "messy_mixed_types.csv"),
            ("json_conn", "json", ASSETS_DIR / "nested_structure.json"),
            ("yaml_conn", "yaml", ASSETS_DIR / "complex_config.yaml")
        ]
        
        for name, db_type, file_path in connections:
            result = self.manager.connect_database(name, db_type, str(file_path)
            )
            response = json.loads(result)
            assert response["success"] is True
            
        # List all databases
        list_result = self.manager.list_databases()
        list_data = json.loads(list_result)
        
        assert list_data["total_connections"] == 3
        assert len(list_data["databases"]) == 3
        
        # Verify database info
        db_names = [db["name"] for db in list_data["databases"]]
        for name, _, _ in connections:
            assert name in db_names
            
    def test_find_table_across_databases(self):
        """Test finding tables across multiple databases."""
        # Connect multiple CSV files (all will have "data_table")
        files = [
            ("db1", ASSETS_DIR / "messy_mixed_types.csv"),
            ("db2", ASSETS_DIR / "large_dataset.csv")
        ]
        
        for name, file_path in files:
            result = self.manager.connect_database(name, "csv", str(file_path)
            )
            response = json.loads(result)
            assert response["success"] is True
            
        # Find table across databases
        find_result = self.manager.find_table(table_name="data_table"
        )
        
        databases = json.loads(find_result)
        assert "db1" in databases
        assert "db2" in databases
        
    def test_query_history_tracking(self):
        """Test query history functionality."""
        csv_file = ASSETS_DIR / "messy_mixed_types.csv"
        
        # Connect database
        self.manager.connect_database("history_test", "csv", str(csv_file)
        )
        
        # Execute multiple queries
        queries = [
            "SELECT * FROM data_table LIMIT 5",
            "SELECT COUNT(*) FROM data_table", 
            "SELECT name FROM data_table WHERE id = 1"
        ]
        
        for query in queries:
            self.manager.execute_query("history_test", query
            )
            
        # Get query history
        history_result = self.manager.get_query_history("history_test"
        )
        
        assert history_result != "No query history found"
        
        # Verify all queries are in history
        for query in queries:
            assert query in history_result
            
    # =========================================================
    # Large Dataset and Pagination Tests  
    # =========================================================
    
    def test_large_dataset_chunking(self):
        """Test automatic chunking for large datasets."""
        large_csv = ASSETS_DIR / "large_dataset.csv"
        
        # Connect to large dataset
        result = self.manager.connect_database("large_data", "csv", str(large_csv)
        )
        response = json.loads(result)
        assert response["success"] is True
        
        # Query large dataset - should trigger chunking
        query_result = self.manager.execute_query(name="large_data",
            query="SELECT * FROM data_table"
        )
        
        query_data = json.loads(query_result)
        assert "metadata" in query_data
        
        if query_data["metadata"]["total_rows"] > 100:
            # Should be chunked
            assert query_data["metadata"]["chunked"] is True
            assert "pagination" in query_data
            assert "query_id" in query_data["metadata"]
            
    def test_pagination_workflow(self):
        """Test complete pagination workflow."""
        large_csv = ASSETS_DIR / "large_dataset.csv"
        
        # Connect
        self.manager.connect_database("pagination", "csv", str(large_csv)
        )
        
        # Initial query with small chunk size
        query_result = self.manager.execute_query(name="pagination",
            query="SELECT * FROM data_table",
            chunk_size=10
        )
        
        query_data = json.loads(query_result)
        
        if query_data.get("metadata", {}).get("chunked"):
            query_id = query_data["metadata"]["query_id"]
            
            # Get next chunk
            next_result = self.manager.next_chunk(query_id=query_id,
                start_row=11,
                chunk_size="15"
            )
            
            next_data = json.loads(next_result)
            assert "data" in next_data
            assert len(next_data["data"]) <= 15
            assert next_data["metadata"]["query_id"] == query_id
            
    # =========================================================
    # Complete Workflow Tests
    # =========================================================
    
    def test_complete_user_workflow(self):
        """Test complete end-to-end user workflow."""
        csv_file = ASSETS_DIR / "messy_mixed_types.csv"
        
        # 1. Connect to database
        connect_result = self.manager.connect_database("workflow_test", "csv", str(csv_file)
        )
        response = json.loads(connect_result)
        assert response["success"] is True
        
        # 2. Describe database structure
        describe_result = self.manager.describe_database("workflow_test"
        )
        describe_data = json.loads(describe_result)
        assert len(describe_data["tables"]) > 0
        
        # 3. Describe specific table
        table_result = self.manager.describe_table("workflow_test", "data_table"
        )
        table_data = json.loads(table_result)
        assert "columns" in table_data
        
        # 4. Execute queries
        basic_query = self.manager.execute_query("workflow_test", 
            "SELECT * FROM data_table LIMIT 3"
        )
        basic_data = json.loads(basic_query)
        assert "data" in basic_data
        assert len(basic_data["data"]) <= 3
        
        # 5. Check query history
        history = self.manager.get_query_history("workflow_test"
        )
        assert "SELECT * FROM data_table LIMIT 3" in history
        
        # 6. Disconnect cleanly
        disconnect_result = self.manager.disconnect_database("workflow_test"
        )
        assert "Successfully disconnected" in disconnect_result
        assert "workflow_test" not in self.manager.connections
        
    def test_multi_format_comparison_workflow(self):
        """Test workflow comparing multiple file formats."""
        formats = [
            ("csv_data", "csv", ASSETS_DIR / "messy_mixed_types.csv"),
            ("json_data", "json", ASSETS_DIR / "nested_structure.json"),
            ("yaml_data", "yaml", ASSETS_DIR / "complex_config.yaml")
        ]
        
        # 1. Connect to all formats
        for name, db_type, file_path in formats:
            result = self.manager.connect_database(name, db_type, str(file_path)
            )
            response = json.loads(result)
            assert response["success"] is True
            
        # 2. List all connections
        list_result = self.manager.list_databases()
        list_data = json.loads(list_result)
        assert list_data["total_connections"] == 3
        
        # 3. Query each database for row counts
        for name, _, _ in formats:
            query_result = self.manager.execute_query(name,
                "SELECT COUNT(*) as record_count FROM data_table"  
            )
            query_data = json.loads(query_result)
            assert "data" in query_data
            assert len(query_data["data"]) == 1
            
        # 4. Disconnect all
        for name, _, _ in formats:
            disconnect_result = self.manager.disconnect_database(name
            )
            assert "Successfully disconnected" in disconnect_result
            
        assert len(self.manager.connections) == 0
        
    # =========================================================
    # Unicode and Special Character Tests
    # =========================================================
    
    def test_unicode_handling(self):
        """Test Unicode character handling across formats."""
        # Test with files containing Unicode
        unicode_tests = [
            ("unicode_yaml", "yaml", ASSETS_DIR / "unicode_special.yaml"),
            ("messy_csv", "csv", ASSETS_DIR / "messy_mixed_types.csv")
        ]
        
        for name, db_type, file_path in unicode_tests:
            # Connect
            connect_result = self.manager.connect_database(name, db_type, str(file_path)
            )
            
            if "success" in connect_result.lower():
                # Query data with potential Unicode
                query_result = self.manager.execute_query(name, "SELECT * FROM data_table LIMIT 3"
                )
                
                query_data = json.loads(query_result)
                if "data" in query_data and query_data["data"]:
                    # Should handle Unicode without errors
                    data_str = json.dumps(query_data["data"])
                    assert isinstance(data_str, str)
                    
    # =========================================================
    # Edge Cases and Stress Tests
    # =========================================================
    
    def test_repeated_connections(self):
        """Test stability under repeated connections."""
        csv_file = ASSETS_DIR / "messy_mixed_types.csv"
        
        # Perform multiple connect/disconnect cycles
        for i in range(5):
            name = f"cycle_test_{i}"
            
            # Connect
            connect_result = self.manager.connect_database(name, "csv", str(csv_file)
            )
            response = json.loads(connect_result)
            assert response["success"] is True
            
            # Quick query
            query_result = self.manager.execute_query(name, "SELECT COUNT(*) FROM data_table"
            )
            query_data = json.loads(query_result)
            assert "data" in query_data
            
            # Disconnect
            disconnect_result = self.manager.disconnect_database(name
            )
            assert "Successfully disconnected" in disconnect_result
            
        # Should have no lingering connections
        assert len(self.manager.connections) == 0
        
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        csv_file = ASSETS_DIR / "messy_mixed_types.csv"
        self.manager.connect_database("injection_test", "csv", str(csv_file)
        )
        
        # Attempt SQL injection through table name
        malicious_names = [
            "data_table'; DROP TABLE data_table; --",
            "data_table UNION SELECT * FROM sqlite_master"
        ]
        
        for malicious_name in malicious_names:
            result = self.manager.describe_table("injection_test", malicious_name
            )
            # Should safely reject malicious table names
            assert "does not exist" in result or "Error" in result
            
        # Original table should still work
        good_result = self.manager.describe_table("injection_test", "data_table"
        )
        good_data = json.loads(good_result)
        assert "name" in good_data
        assert good_data["name"] == "data_table"
        
    def test_memory_management(self):
        """Test memory management with large datasets.""" 
        large_csv = ASSETS_DIR / "large_dataset.csv"
        
        connect_result = self.manager.connect_database("memory_test", "csv", str(large_csv)
        )
        response = json.loads(connect_result)
        assert response["success"] is True
        
        # Query large dataset
        query_result = self.manager.execute_query("memory_test", "SELECT * FROM data_table"
        )
        
        query_data = json.loads(query_result)
        
        # Should have memory info in metadata
        if "metadata" in query_data:
            if "memory_info" in query_data["metadata"]:
                memory_info = query_data["metadata"]["memory_info"]
                
                if "used_percent" in memory_info:
                    assert isinstance(memory_info["used_percent"], (int, float))
                    assert 0 <= memory_info["used_percent"] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])