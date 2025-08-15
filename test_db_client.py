#!/usr/bin/env python3
"""
Basic test script for db-client-mcp security improvements
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import tempfile
import time
from pathlib import Path

# Add src to path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from db_client_mcp.db_client_mcp import DatabaseManager

def create_test_data():
    """Create test data files and databases for testing."""
    # Create a small CSV file
    small_data = pd.DataFrame({
        'id': range(1, 6),
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32]
    })
    small_data.to_csv('test_small.csv', index=False)
    
    # Create a large CSV file (101+ rows to test buffering)
    large_data = pd.DataFrame({
        'id': range(1, 151),
        'value': [f'value_{i}' for i in range(1, 151)],
        'category': ['A' if i % 3 == 0 else 'B' if i % 3 == 1 else 'C' for i in range(1, 151)]
    })
    large_data.to_csv('test_large.csv', index=False)
    
    # Create a SQLite database
    conn = sqlite3.connect('test.db')
    small_data.to_sql('users', conn, index=False, if_exists='replace')
    conn.close()
    
    print("Created test files: test_small.csv, test_large.csv, test.db")

def test_basic_functionality():
    """Test basic functionality to ensure existing API still works."""
    manager = DatabaseManager()
    
    print("=== Testing Basic Functionality ===")
    
    # Test connecting to SQLite database
    result = manager.connect_database("test_db", "sqlite", "test.db")
    print(f"Connect SQLite: {result}")
    assert "Successfully connected" in result
    
    # Test connecting to CSV file (small)
    result = manager.connect_database("small_csv", "csv", "test_small.csv")
    print(f"Connect small CSV: {result}")
    assert "Successfully connected" in result
    
    # Test connecting to large CSV file
    result = manager.connect_database("large_csv", "csv", "test_large.csv")
    print(f"Connect large CSV: {result}")
    assert "Successfully connected" in result
    
    # Test list databases
    result = manager.list_databases()
    print(f"List databases: {result}")
    databases = json.loads(result)
    assert "test_db" in databases
    assert "small_csv" in databases
    assert "large_csv" in databases
    
    print("✅ Basic functionality tests passed!")

def test_security_features():
    """Test security features."""
    manager = DatabaseManager()
    
    print("\\n=== Testing Security Features ===")
    
    # Test path security - should reject parent directory access
    result = manager.connect_database("bad", "csv", "../etc/passwd")
    print(f"Parent directory access: {result}")
    assert "outside the allowed directory" in result
    
    # Test connection limiting by trying to create 11 connections
    connections_created = 0
    for i in range(12):
        result = manager.connect_database(f"test_{i}", "sqlite", "test.db")
        if "Successfully connected" in result:
            connections_created += 1
        elif "Maximum number of concurrent connections" in result:
            print(f"Hit connection limit after {connections_created} connections: {result}")
            break
    
    assert connections_created <= 10, f"Should not exceed 10 connections, got {connections_created}"
    
    print("✅ Security features tests passed!")

def test_query_functionality():
    """Test query functionality including buffering."""
    manager = DatabaseManager()
    
    print("\\n=== Testing Query Functionality ===")
    
    # Connect to test databases
    manager.connect_database("test_db", "sqlite", "test.db")
    manager.connect_database("large_csv", "csv", "test_large.csv")
    
    # Test small query (should work normally)
    result = manager.execute_query("test_db", "SELECT * FROM users")
    print("Small markdown query result length:", len(result))
    assert "Alice" in result or "id" in result
    
    # Test small JSON query (should work normally)  
    result = manager.execute_query_json("test_db", "SELECT * FROM users")
    print("Small JSON query result type:", type(result))
    data = json.loads(result)
    assert isinstance(data, list)
    assert len(data) == 5
    
    # Test large query with markdown (should return error)
    result = manager.execute_query("large_csv", "SELECT * FROM data_table")
    print("Large markdown query result:", result[:100] + "...")
    assert "exceeds the 100-row limit" in result
    
    # Test large query with JSON (should return buffered result)
    result = manager.execute_query_json("large_csv", "SELECT * FROM data_table")
    print("Large JSON query result type:", type(result))
    data = json.loads(result)
    assert "metadata" in data
    assert "data" in data
    assert "next_options" in data
    assert data["metadata"]["total_rows"] == 150
    assert len(data["data"]) == 10  # First 10 rows
    
    query_id = data["metadata"]["query_id"]
    print(f"Generated query ID: {query_id}")
    
    # Test buffered query tools
    result = manager.get_buffered_query_info(query_id)
    buffer_info = json.loads(result)
    print("Buffer info total_rows:", buffer_info["total_rows"])
    assert buffer_info["total_rows"] == 150
    
    # Test chunk retrieval
    result = manager.get_query_chunk(query_id, 11, "50")
    chunk_data = json.loads(result)
    print("Chunk retrieval showing rows:", chunk_data["metadata"]["showing_rows"])
    assert chunk_data["metadata"]["showing_rows"] == "11-60"
    assert len(chunk_data["data"]) == 50
    
    # Test clearing buffer
    result = manager.clear_query_buffer(query_id)
    print("Clear buffer result:", result)
    assert "Successfully cleared" in result
    
    print("✅ Query functionality tests passed!")

def test_sql_injection_prevention():
    """Test SQL injection prevention."""
    manager = DatabaseManager()
    
    print("\\n=== Testing SQL Injection Prevention ===")
    
    manager.connect_database("test_db", "sqlite", "test.db")
    
    # Test safe table name handling
    result = manager.get_table_sample("test_db", "users", 3)
    print("Safe table query result length:", len(result))
    assert "Alice" in result or "id" in result
    
    # Test invalid table name (should be rejected)
    result = manager.get_table_sample("test_db", "users; DROP TABLE users; --", 3)
    print("Malicious table name result:", result)
    assert "Invalid table name" in result or "does not exist" in result
    
    print("✅ SQL injection prevention tests passed!")

def cleanup_test_files():
    """Clean up test files."""
    for filename in ['test_small.csv', 'test_large.csv', 'test.db']:
        try:
            os.unlink(filename)
        except FileNotFoundError:
            pass
    print("Cleaned up test files")

if __name__ == "__main__":
    try:
        create_test_data()
        test_basic_functionality()
        test_security_features()
        test_query_functionality()
        test_sql_injection_prevention()
        print("\\n🎉 ALL TESTS PASSED! 🎉")
    except Exception as e:
        print(f"\\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_test_files()