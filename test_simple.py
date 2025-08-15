#!/usr/bin/env python3
"""
Simple test script to verify core functionality of db-client-mcp
"""

import os
import sys
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path

# Add src to path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that we can import the module."""
    print("=== Testing Imports ===")
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager, QueryBuffer
        print("✅ Successfully imported DatabaseManager and QueryBuffer")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_path_security():
    """Test path security functions."""
    print("\n=== Testing Path Security ===")
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager
        
        manager = DatabaseManager()
        
        # Create a test file in current directory
        with open('test_file.txt', 'w') as f:
            f.write('test')
        
        # Test valid path
        try:
            result = manager._sanitize_path('test_file.txt')
            print(f"✅ Valid path accepted: {result}")
        except Exception as e:
            print(f"❌ Valid path rejected: {e}")
            return False
        
        # Test invalid path (parent directory)
        try:
            result = manager._sanitize_path('../etc/passwd')
            print(f"❌ Invalid path accepted: {result}")
            return False
        except ValueError as e:
            print(f"✅ Invalid path correctly rejected: {e}")
        
        # Clean up
        os.unlink('test_file.txt')
        
        return True
    except Exception as e:
        print(f"❌ Path security test failed: {e}")
        return False

def test_file_size_detection():
    """Test file size detection."""
    print("\n=== Testing File Size Detection ===")
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager
        
        manager = DatabaseManager()
        
        # Create a small file
        with open('small_file.txt', 'w') as f:
            f.write('test')
        
        # Test file size detection
        size = manager._get_file_size('small_file.txt')
        print(f"✅ File size detected: {size} bytes")
        
        is_large = manager._is_large_file('small_file.txt', threshold_mb=1)
        print(f"✅ Large file detection: {is_large} (should be False)")
        
        # Clean up
        os.unlink('small_file.txt')
        
        return True
    except Exception as e:
        print(f"❌ File size detection test failed: {e}")
        return False

def test_query_id_generation():
    """Test query ID generation."""
    print("\n=== Testing Query ID Generation ===")
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager
        
        manager = DatabaseManager()
        
        # Generate some query IDs
        query_id1 = manager._generate_query_id("test_db", "SELECT * FROM users")
        query_id2 = manager._generate_query_id("test_db", "SELECT * FROM products")
        
        print(f"✅ Query ID 1: {query_id1}")
        print(f"✅ Query ID 2: {query_id2}")
        
        # Check format - query ID should end with _{timestamp}_{4char_hash}
        parts1 = query_id1.split('_')
        assert len(parts1) >= 3, f"Query ID should have at least 3 parts, got {len(parts1)}"
        assert query_id1.startswith("test_db"), f"Should start with 'test_db', got {query_id1}"
        assert parts1[-2].isdigit(), f"Second-to-last part should be timestamp, got {parts1[-2]}"
        assert len(parts1[-1]) == 4, f"Last part should be 4 chars, got {len(parts1[-1])}"
        
        print("✅ Query ID format is correct")
        
        return True
    except Exception as e:
        print(f"❌ Query ID generation test failed: {e}")
        return False

def test_safe_table_identifier():
    """Test safe table identifier generation."""
    print("\n=== Testing Safe Table Identifiers ===")
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager
        
        manager = DatabaseManager()
        
        # Test valid table name
        safe_name = manager._safe_table_identifier("users")
        print(f"✅ Safe identifier for 'users': {safe_name}")
        
        # Test invalid table name
        try:
            safe_name = manager._safe_table_identifier("users; DROP TABLE users; --")
            print(f"❌ Malicious table name accepted: {safe_name}")
            return False
        except ValueError as e:
            print(f"✅ Malicious table name correctly rejected: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Safe table identifier test failed: {e}")
        return False

def test_connection_management():
    """Test connection management initialization."""
    print("\n=== Testing Connection Management ===")
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager
        
        manager = DatabaseManager()
        
        # Check initialization
        assert hasattr(manager, 'connection_semaphore'), "Should have connection_semaphore"
        assert hasattr(manager, 'connection_lock'), "Should have connection_lock"
        assert hasattr(manager, 'connection_count'), "Should have connection_count"
        assert hasattr(manager, 'query_buffers'), "Should have query_buffers"
        assert hasattr(manager, 'temp_files'), "Should have temp_files"
        
        print("✅ Connection management properly initialized")
        
        # Test semaphore initial value
        initial_count = manager.connection_semaphore._value
        print(f"✅ Semaphore initial value: {initial_count} (should be 10)")
        assert initial_count == 10, f"Semaphore should start at 10, got {initial_count}"
        
        return True
    except Exception as e:
        print(f"❌ Connection management test failed: {e}")
        return False

def test_query_buffer_functionality():
    """Test query buffer data structure."""
    print("\n=== Testing Query Buffer ===")
    try:
        from db_client_mcp.db_client_mcp import QueryBuffer
        import pandas as pd
        import time
        
        # Create test DataFrame
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        
        # Create QueryBuffer
        buffer = QueryBuffer(
            query_id="test_123_abcd",
            db_name="test_db",
            query="SELECT * FROM test",
            results=df,
            timestamp=time.time()
        )
        
        print(f"✅ QueryBuffer created: {buffer.query_id}")
        assert buffer.db_name == "test_db"
        assert len(buffer.results) == 3
        
        return True
    except Exception as e:
        print(f"❌ Query buffer test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("🔧 Running db-client-mcp Security Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_path_security,
        test_file_size_detection,
        test_query_id_generation,
        test_safe_table_identifier,
        test_connection_management,
        test_query_buffer_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)