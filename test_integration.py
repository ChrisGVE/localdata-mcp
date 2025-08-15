#!/usr/bin/env python3
"""
Integration test for db-client-mcp to verify end-to-end functionality
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import tempfile

# Add src to path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_database():
    """Create a test SQLite database with sample data."""
    # Create test database
    conn = sqlite3.connect('test_integration.db')
    
    # Create tables with various sizes
    small_data = pd.DataFrame({
        'id': range(1, 6),
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32]
    })
    small_data.to_sql('users', conn, index=False, if_exists='replace')
    
    # Large table for testing buffering
    large_data = pd.DataFrame({
        'id': range(1, 151),
        'product': [f'Product_{i}' for i in range(1, 151)],
        'price': [10.0 + i * 0.5 for i in range(1, 151)]
    })
    large_data.to_sql('products', conn, index=False, if_exists='replace')
    
    conn.close()
    print("Created test database: test_integration.db")

def test_basic_integration():
    """Test basic integration by accessing tools through manager._get_tool_implementation()"""
    print("\n=== Testing Basic Integration ===")
    
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager
        
        # Create manager instance
        manager = DatabaseManager()
        
        # Test connecting to database by calling the underlying method
        # Since the methods are decorated with @mcp.tool, we need to access the actual function
        
        # Get the actual function from the tool decoration
        connect_func = manager.connect_database.__wrapped__
        
        # Test connection to SQLite
        result = connect_func(manager, "test_db", "sqlite", "test_integration.db")
        print(f"Connection result: {result}")
        assert "Successfully connected" in result
        
        # Test listing databases
        list_func = manager.list_databases.__wrapped__
        result = list_func(manager)
        print(f"List databases: {result}")
        databases = json.loads(result)
        assert "test_db" in databases
        
        # Test small query (should work normally)
        query_func = manager.execute_query.__wrapped__
        result = query_func(manager, "test_db", "SELECT * FROM users")
        print(f"Small query result length: {len(result)}")
        assert "Alice" in result or "id" in result
        assert "exceeds the 100-row limit" not in result
        
        # Test large query (should get error for markdown)
        result = query_func(manager, "test_db", "SELECT * FROM products")
        print(f"Large query result: {result[:100]}...")
        assert "exceeds the 100-row limit" in result
        
        # Test large query with JSON (should get buffered result)
        json_func = manager.execute_query_json.__wrapped__
        result = json_func(manager, "test_db", "SELECT * FROM products")
        data = json.loads(result)
        
        if isinstance(data, dict) and "metadata" in data:
            print(f"Buffered result - total_rows: {data['metadata']['total_rows']}")
            assert data["metadata"]["total_rows"] == 150
            assert len(data["data"]) == 10  # First 10 rows
            assert "query_id" in data["metadata"]
            
            query_id = data["metadata"]["query_id"]
            
            # Test buffer tools
            info_func = manager.get_buffered_query_info.__wrapped__
            info_result = info_func(manager, query_id)
            info_data = json.loads(info_result)
            print(f"Buffer info total_rows: {info_data['total_rows']}")
            assert info_data["total_rows"] == 150
            
            # Test chunk retrieval
            chunk_func = manager.get_query_chunk.__wrapped__
            chunk_result = chunk_func(manager, query_id, 11, "20")
            chunk_data = json.loads(chunk_result)
            print(f"Chunk rows: {chunk_data['metadata']['showing_rows']}")
            assert chunk_data["metadata"]["showing_rows"] == "11-30"
            assert len(chunk_data["data"]) == 20
            
        else:
            print("❌ Expected buffered result but got regular JSON array")
            return False
        
        # Test SQL injection prevention
        sample_func = manager.get_table_sample.__wrapped__
        result = sample_func(manager, "test_db", "users; DROP TABLE users; --", 5)
        print(f"SQL injection test: {result[:50]}...")
        assert "Invalid table name" in result
        
        print("✅ Basic integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_security_features():
    """Test security features."""
    print("\n=== Testing Security Features ===")
    
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager
        
        manager = DatabaseManager()
        connect_func = manager.connect_database.__wrapped__
        
        # Test path security
        result = connect_func(manager, "bad", "csv", "../etc/passwd")
        print(f"Path security test: {result[:50]}...")
        assert "outside the allowed directory" in result
        
        # Test connection limiting by creating multiple connections
        connection_count = 0
        for i in range(12):
            result = connect_func(manager, f"conn_{i}", "sqlite", "test_integration.db")
            if "Successfully connected" in result:
                connection_count += 1
            elif "Maximum number of concurrent connections" in result:
                print(f"Connection limit reached at {connection_count}: {result[:50]}...")
                break
        
        assert connection_count <= 10, f"Should not exceed 10 connections, got {connection_count}"
        
        print("✅ Security features tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False

def cleanup():
    """Clean up test files."""
    try:
        os.unlink('test_integration.db')
        print("Cleaned up test files")
    except FileNotFoundError:
        pass

def main():
    """Main test function."""
    print("🧪 Running db-client-mcp Integration Tests")
    print("=" * 60)
    
    try:
        create_test_database()
        
        tests_passed = 0
        total_tests = 2
        
        if test_basic_integration():
            tests_passed += 1
        
        if test_security_features():
            tests_passed += 1
        
        print(f"\n{'='*60}")
        print(f"Integration Test Results: {tests_passed}/{total_tests} passed")
        
        if tests_passed == total_tests:
            print("🎉 ALL INTEGRATION TESTS PASSED! 🎉")
            return True
        else:
            print(f"❌ {total_tests - tests_passed} integration tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False
    finally:
        cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)