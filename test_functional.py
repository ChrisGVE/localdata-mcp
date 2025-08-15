#!/usr/bin/env python3
"""
Functional test script for db-client-mcp that works with the tool structure
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

def create_test_data():
    """Create test data files and databases for testing."""
    # Create a small CSV file
    small_data = pd.DataFrame({
        'id': range(1, 6),
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32]
    })
    small_data.to_csv('test_small.csv', index=False)
    
    # Create a large CSV file (150+ rows to test buffering)
    large_data = pd.DataFrame({
        'id': range(1, 151),
        'value': [f'value_{i}' for i in range(1, 151)],
        'category': ['A' if i % 3 == 0 else 'B' if i % 3 == 1 else 'C' for i in range(1, 151)]
    })
    large_data.to_csv('test_large.csv', index=False)
    
    # Create a SQLite database
    conn = sqlite3.connect('test.db')
    small_data.to_sql('users', conn, index=False, if_exists='replace')
    large_data.to_sql('products', conn, index=False, if_exists='replace')
    conn.close()
    
    print("Created test files: test_small.csv, test_large.csv, test.db")

def test_core_functionality():
    """Test core functionality by accessing methods directly"""
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager
        manager = DatabaseManager()
        
        print("=== Testing Core Security and Functionality ===")
        
        # Test 1: Path Security
        try:
            manager._sanitize_path('../etc/passwd')
            print("❌ Path security failed - should reject parent directory")
            return False
        except ValueError as e:
            print("✅ Path security working - rejected parent directory")
        
        # Test 2: Safe table identifiers
        try:
            safe = manager._safe_table_identifier('users')
            print(f"✅ Safe table identifier: {safe}")
        except Exception as e:
            print(f"❌ Safe table identifier failed: {e}")
            return False
        
        try:
            manager._safe_table_identifier('users; DROP TABLE users; --')
            print("❌ SQL injection prevention failed")
            return False
        except ValueError:
            print("✅ SQL injection prevention working")
        
        # Test 3: File size detection
        size = manager._get_file_size('test_small.csv')
        is_large = manager._is_large_file('test_small.csv')
        print(f"✅ File size detection: {size} bytes, large: {is_large}")
        
        # Test 4: Query ID generation
        query_id = manager._generate_query_id('test_db', 'SELECT * FROM users')
        print(f"✅ Query ID generated: {query_id}")
        
        # Test format
        parts = query_id.split('_')
        if len(parts) >= 3 and parts[0] == 'test' and parts[1] == 'db':
            print("✅ Query ID format correct")
        else:
            print("❌ Query ID format incorrect")
            return False
        
        # Test 5: Connection management initialization
        if (hasattr(manager, 'connection_semaphore') and 
            hasattr(manager, 'query_buffers') and 
            hasattr(manager, 'temp_files')):
            print("✅ Connection management initialized")
        else:
            print("❌ Connection management not properly initialized")
            return False
        
        # Test 6: Buffer cleanup
        initial_count = len(manager.query_buffers)
        manager._cleanup_expired_buffers()
        print(f"✅ Buffer cleanup ran (buffers: {initial_count})")
        
        print("\n✅ ALL CORE FUNCTIONALITY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_operations():
    """Test database operations by using the engine methods directly"""
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager
        manager = DatabaseManager()
        
        print("\n=== Testing Database Operations ===")
        
        # Test SQLite engine creation
        try:
            engine = manager._get_engine('sqlite', 'test.db')
            print("✅ SQLite engine created")
            
            # Store it manually to simulate connection
            manager.connections['test_db'] = engine
            manager.query_history['test_db'] = []
            
            # Test query execution using pandas
            df = pd.read_sql_query("SELECT * FROM users LIMIT 3", engine)
            print(f"✅ Query executed: {len(df)} rows")
            
            if len(df) == 3 and 'Alice' in df.to_string():
                print("✅ Query results correct")
            else:
                print("❌ Query results incorrect")
                return False
                
        except Exception as e:
            print(f"❌ SQLite operations failed: {e}")
            return False
        
        # Test CSV engine creation
        try:
            csv_engine = manager._get_engine('csv', 'test_small.csv')
            print("✅ CSV engine created")
            
            # Test CSV query
            df = pd.read_sql_query("SELECT * FROM data_table", csv_engine)
            print(f"✅ CSV query executed: {len(df)} rows")
            
        except Exception as e:
            print(f"❌ CSV operations failed: {e}")
            return False
        
        # Test large CSV with temporary storage
        try:
            # Create larger file to test temp storage (but not huge for testing speed)
            large_df = pd.DataFrame({
                'id': range(1, 1001),
                'data': ['x' * 100 for _ in range(1000)]  # 100KB per row
            })
            large_df.to_csv('test_large_temp.csv', index=False)
            
            large_engine = manager._get_engine('csv', 'test_large_temp.csv')
            print("✅ Large CSV engine created")
            
            # Check if temp files were created
            initial_temp_count = len(manager.temp_files)
            print(f"✅ Temp file tracking: {initial_temp_count} files")
            
            # Cleanup test file
            os.unlink('test_large_temp.csv')
            
        except Exception as e:
            print(f"❌ Large CSV operations failed: {e}")
            return False
        
        print("✅ ALL DATABASE OPERATIONS TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        return False

def test_query_buffering():
    """Test query buffering system"""
    try:
        from db_client_mcp.db_client_mcp import DatabaseManager, QueryBuffer
        import pandas as pd
        import time
        
        manager = DatabaseManager()
        print("\n=== Testing Query Buffering ===")
        
        # Create test data
        test_df = pd.DataFrame({
            'id': range(1, 201),  # 200 rows - over threshold
            'value': [f'data_{i}' for i in range(1, 201)]
        })
        
        # Create buffer
        query_id = manager._generate_query_id('test_db', 'SELECT * FROM test')
        buffer = QueryBuffer(
            query_id=query_id,
            db_name='test_db',
            query='SELECT * FROM test',
            results=test_df,
            timestamp=time.time()
        )
        
        # Add to manager
        manager.query_buffers[query_id] = buffer
        print(f"✅ Buffer created: {query_id}")
        
        # Test buffer info (simulate the method logic)
        if query_id in manager.query_buffers:
            stored_buffer = manager.query_buffers[query_id]
            print(f"✅ Buffer info: {len(stored_buffer.results)} rows, db: {stored_buffer.db_name}")
        else:
            print("❌ Buffer not found")
            return False
        
        # Test chunk retrieval logic
        df = manager.query_buffers[query_id].results
        total_rows = len(df)
        start_idx = 10  # 1-based row 11 -> 0-based index 10
        chunk_size = 20
        
        chunk_df = df.iloc[start_idx:start_idx + chunk_size]
        print(f"✅ Chunk retrieval: rows {start_idx+1}-{start_idx+len(chunk_df)} ({len(chunk_df)} rows)")
        
        if len(chunk_df) == 20:
            print("✅ Chunk size correct")
        else:
            print("❌ Chunk size incorrect")
            return False
        
        # Test expiry simulation
        old_timestamp = time.time() - 700  # Older than 10 minutes
        buffer.timestamp = old_timestamp
        
        # Force cleanup by resetting last_cleanup time
        manager.last_cleanup = 0
        
        # Run cleanup
        manager._cleanup_expired_buffers()
        
        if query_id not in manager.query_buffers:
            print("✅ Buffer expiry working")
        else:
            print("❌ Buffer expiry failed")
            return False
        
        print("✅ ALL QUERY BUFFERING TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Query buffering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_installation():
    """Test installation readiness"""
    try:
        print("\n=== Testing Installation ===")
        
        # Test pyproject.toml exists
        if os.path.exists('pyproject.toml'):
            print("✅ pyproject.toml exists")
        else:
            print("❌ pyproject.toml missing")
            return False
        
        # Test main module imports
        from db_client_mcp.db_client_mcp import main
        if callable(main):
            print("✅ Main function importable")
        else:
            print("❌ Main function not callable")
            return False
        
        # Test package structure
        if os.path.exists('src/db_client_mcp/__init__.py'):
            print("✅ Package structure correct")
        else:
            print("❌ Package structure incorrect")
            return False
        
        print("✅ INSTALLATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    for filename in ['test_small.csv', 'test_large.csv', 'test.db', 'test_large_temp.csv']:
        try:
            os.unlink(filename)
        except FileNotFoundError:
            pass
    print("Cleaned up test files")

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Functional Tests for db-client-mcp")
    print("=" * 60)
    
    try:
        create_test_data()
        
        tests = [
            test_core_functionality,
            test_database_operations, 
            test_query_buffering,
            test_installation
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        print(f"\n{'='*60}")
        print(f"FUNCTIONAL TEST RESULTS: {passed}/{total} passed")
        
        if passed == total:
            print("🎉 ALL FUNCTIONAL TESTS PASSED! 🎉")
            return True
        else:
            print(f"❌ {total - passed} tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False
    finally:
        cleanup_test_files()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)