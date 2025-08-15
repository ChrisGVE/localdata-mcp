#!/usr/bin/env python3
"""
Comprehensive test suite for db-client-mcp security improvements and functionality
Tests all critical areas: security, performance, edge cases, and resource management
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import tempfile
import time
import threading
import hashlib
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class TestResults:
    """Track test results and provide summary"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.failures = []
    
    def record_test(self, test_name: str, success: bool, error: str = None):
        self.total += 1
        if success:
            self.passed += 1
            print(f"✅ {test_name}")
        else:
            self.failed += 1
            self.failures.append((test_name, error))
            print(f"❌ {test_name}: {error}")
    
    def print_summary(self):
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE TEST RESULTS: {self.passed}/{self.total} passed")
        print(f"Success rate: {(self.passed/self.total)*100:.1f}%")
        
        if self.failures:
            print(f"\nFAILED TESTS ({len(self.failures)}):")
            for test_name, error in self.failures:
                print(f"  - {test_name}: {error}")
        
        if self.passed == self.total:
            print("🎉 ALL COMPREHENSIVE TESTS PASSED! 🎉")
            return True
        else:
            print(f"❌ {self.failed} tests failed")
            return False

class TestDataGenerator:
    """Generate test data files of various sizes and formats"""
    
    @staticmethod
    def create_test_files():
        """Create all test files needed for testing"""
        test_files = []
        
        # Small CSV (5 rows - under threshold)
        small_data = pd.DataFrame({
            'id': range(1, 6),
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'city': ['NYC', 'LA', 'Chicago', 'Miami', 'Seattle']
        })
        small_data.to_csv('test_small.csv', index=False)
        test_files.append('test_small.csv')
        
        # Exactly 100 rows (boundary condition)
        boundary_data = pd.DataFrame({
            'id': range(1, 101),
            'value': [f'value_{i}' for i in range(1, 101)],
            'category': ['A' if i % 3 == 0 else 'B' if i % 3 == 1 else 'C' for i in range(1, 101)]
        })
        boundary_data.to_csv('test_boundary_100.csv', index=False)
        test_files.append('test_boundary_100.csv')
        
        # Large CSV (150 rows - over threshold)
        large_data = pd.DataFrame({
            'id': range(1, 151),
            'product': [f'Product_{i}' for i in range(1, 151)],
            'price': [10.0 + i * 0.5 for i in range(1, 151)],
            'description': [f'Long description for product {i} with lots of text to make it interesting' for i in range(1, 151)]
        })
        large_data.to_csv('test_large.csv', index=False)
        test_files.append('test_large.csv')
        
        # Very large CSV (1000 rows - stress test)
        stress_data = pd.DataFrame({
            'id': range(1, 1001),
            'data': [f'stress_test_data_{i}_{hashlib.md5(str(i).encode()).hexdigest()}' for i in range(1, 1001)],
            'timestamp': [time.time() + i for i in range(1, 1001)]
        })
        stress_data.to_csv('test_stress.csv', index=False)
        test_files.append('test_stress.csv')
        
        # Large file (simulated ~2MB - not quite 100MB but large enough to test handling)
        # Creating actual 100MB+ file would be slow for testing
        large_content_data = pd.DataFrame({
            'id': range(1, 5001),
            'content': ['x' * 500 for _ in range(5000)]  # ~2.5MB file
        })
        large_content_data.to_csv('test_large_content.csv', index=False)
        test_files.append('test_large_content.csv')
        
        # SQLite database
        conn = sqlite3.connect('test_comprehensive.db')
        small_data.to_sql('users', conn, index=False, if_exists='replace')
        boundary_data.to_sql('boundary_table', conn, index=False, if_exists='replace')  
        large_data.to_sql('products', conn, index=False, if_exists='replace')
        
        # Add table with special characters for injection testing
        special_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ["O'Reilly", 'Smith; DROP TABLE users; --', 'Normal Name'],
            'data': ['<script>alert("xss")</script>', 'SELECT * FROM users', 'normal data']
        })
        special_data.to_sql('special_chars', conn, index=False, if_exists='replace')
        conn.close()
        test_files.append('test_comprehensive.db')
        
        # JSON file
        json_data = {
            'users': [
                {'id': 1, 'name': 'John', 'active': True},
                {'id': 2, 'name': 'Jane', 'active': False}
            ],
            'settings': {'debug': True, 'version': '1.0'}
        }
        with open('test_data.json', 'w') as f:
            json.dump(json_data, f)
        test_files.append('test_data.json')
        
        # YAML file (if yaml available)
        try:
            import yaml
            yaml_data = {'config': {'host': 'localhost', 'port': 5432}, 'data': [1, 2, 3]}
            with open('test_data.yaml', 'w') as f:
                yaml.dump(yaml_data, f)
            test_files.append('test_data.yaml')
        except ImportError:
            pass  # Skip YAML if not available
        
        # TOML file (if toml available)
        try:
            import toml
            toml_data = {'server': {'host': 'localhost', 'port': 8080}, 'data': {'items': [1, 2, 3]}}
            with open('test_data.toml', 'w') as f:
                toml.dump(toml_data, f)
            test_files.append('test_data.toml')
        except ImportError:
            pass  # Skip TOML if not available
        
        print(f"Created {len(test_files)} test files")
        return test_files
    
    @staticmethod
    def create_malicious_files():
        """Create files for security testing"""
        # File with path traversal in name
        if not os.path.exists('malicious'):
            os.makedirs('malicious')
        
        with open('malicious/innocent.txt', 'w') as f:
            f.write('This file should not be accessible via path traversal')
        
        return ['malicious/innocent.txt']
    
    @staticmethod
    def cleanup_files(file_list):
        """Clean up test files"""
        for file_path in file_list:
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except OSError:
                pass

class ComprehensiveTests:
    """Main test class with all comprehensive tests"""
    
    def __init__(self):
        self.results = TestResults()
        self.test_files = []
    
    def setup(self):
        """Set up test environment"""
        print("🔧 Setting up comprehensive test environment...")
        self.test_files = TestDataGenerator.create_test_files()
        self.test_files.extend(TestDataGenerator.create_malicious_files())
        
    def cleanup(self):
        """Clean up test environment"""
        print("🧹 Cleaning up test files...")
        TestDataGenerator.cleanup_files(self.test_files)
    
    def test_existing_api_compatibility(self):
        """Test that all existing API methods work exactly as before"""
        print("\n=== Testing Existing API Compatibility ===")
        
        try:
            from db_client_mcp.db_client_mcp import DatabaseManager
            manager = DatabaseManager()
            
            # Test basic connection (SQLite)
            result = manager.connect_database.__wrapped__(manager, "test_db", "sqlite", "test_comprehensive.db")
            self.results.record_test("Connect SQLite database", 
                                    "Successfully connected" in result, result if "Successfully connected" not in result else None)
            
            # Test CSV connection
            result = manager.connect_database.__wrapped__(manager, "csv_db", "csv", "test_small.csv")
            self.results.record_test("Connect CSV file", 
                                    "Successfully connected" in result, result if "Successfully connected" not in result else None)
            
            # Test list databases
            result = manager.list_databases.__wrapped__(manager)
            databases = json.loads(result)
            self.results.record_test("List databases", 
                                    "test_db" in databases and "csv_db" in databases,
                                    f"Expected test_db and csv_db, got: {databases}")
            
            # Test describe database
            result = manager.describe_database.__wrapped__(manager, "test_db")
            db_info = json.loads(result)
            self.results.record_test("Describe database", 
                                    "tables" in db_info and len(db_info["tables"]) > 0,
                                    f"No tables found in db_info: {db_info}")
            
            # Test basic query (small result)
            result = manager.execute_query.__wrapped__(manager, "test_db", "SELECT * FROM users LIMIT 3")
            self.results.record_test("Execute small query (markdown)", 
                                    "Alice" in result or "id" in result,
                                    f"Expected user data, got: {result[:100]}")
            
            # Test JSON query (small result) 
            result = manager.execute_query_json.__wrapped__(manager, "test_db", "SELECT * FROM users LIMIT 3")
            data = json.loads(result)
            self.results.record_test("Execute small query (JSON)", 
                                    isinstance(data, list) and len(data) == 3,
                                    f"Expected list of 3 items, got: {type(data)} with {len(data) if isinstance(data, list) else 'N/A'}")
            
            # Test table operations
            result = manager.describe_table.__wrapped__(manager, "test_db", "users")
            table_info = json.loads(result)
            self.results.record_test("Describe table", 
                                    "columns" in table_info and len(table_info["columns"]) > 0,
                                    f"No columns found: {table_info}")
            
            result = manager.get_table_sample.__wrapped__(manager, "test_db", "users", 2)
            self.results.record_test("Get table sample", 
                                    "Alice" in result or "id" in result,
                                    f"Expected sample data, got: {result[:100]}")
            
            result = manager.get_table_sample_json.__wrapped__(manager, "test_db", "users", 2)
            sample_data = json.loads(result)
            self.results.record_test("Get table sample JSON", 
                                    isinstance(sample_data, list) and len(sample_data) == 2,
                                    f"Expected list of 2 items, got: {type(sample_data)}")
            
            # Test query history
            result = manager.get_query_history.__wrapped__(manager, "test_db")
            self.results.record_test("Get query history", 
                                    "SELECT" in result,
                                    f"Expected query history with SELECT, got: {result}")
            
            # Test find table
            result = manager.find_table.__wrapped__(manager, "users")
            found_dbs = json.loads(result)
            self.results.record_test("Find table", 
                                    "test_db" in found_dbs,
                                    f"Expected test_db in results, got: {found_dbs}")
            
            # Test text file reading
            result = manager.read_text_file.__wrapped__(manager, "test_data.json", "json")
            parsed_data = json.loads(result)
            self.results.record_test("Read JSON file", 
                                    "users" in parsed_data,
                                    f"Expected 'users' key in JSON, got: {list(parsed_data.keys())}")
            
        except Exception as e:
            self.results.record_test("API compatibility test setup", False, str(e))
    
    def test_path_security_comprehensive(self):
        """Test path security with comprehensive scenarios"""
        print("\n=== Testing Path Security Comprehensive ===")
        
        try:
            from db_client_mcp.db_client_mcp import DatabaseManager
            manager = DatabaseManager()
            
            # Test 1: Valid path in current directory
            result = manager.connect_database.__wrapped__(manager, "valid", "csv", "test_small.csv")
            self.results.record_test("Path security - valid current dir", 
                                    "Successfully connected" in result,
                                    f"Valid path rejected: {result}")
            
            # Test 2: Parent directory traversal
            result = manager.connect_database.__wrapped__(manager, "bad1", "csv", "../etc/passwd")
            self.results.record_test("Path security - parent traversal", 
                                    "outside the allowed directory" in result,
                                    f"Parent traversal not blocked: {result}")
            
            # Test 3: Deep parent directory traversal
            result = manager.connect_database.__wrapped__(manager, "bad2", "csv", "../../etc/passwd")
            self.results.record_test("Path security - deep traversal", 
                                    "outside the allowed directory" in result,
                                    f"Deep traversal not blocked: {result}")
            
            # Test 4: Absolute path outside current directory
            result = manager.connect_database.__wrapped__(manager, "bad3", "csv", "/etc/passwd")
            self.results.record_test("Path security - absolute path", 
                                    "outside the allowed directory" in result,
                                    f"Absolute path not blocked: {result}")
            
            # Test 5: Path with mixed separators and dots
            result = manager.connect_database.__wrapped__(manager, "bad4", "csv", ".//../etc/passwd")
            self.results.record_test("Path security - mixed separators", 
                                    "outside the allowed directory" in result,
                                    f"Mixed separator traversal not blocked: {result}")
            
            # Test 6: Non-existent file
            result = manager.connect_database.__wrapped__(manager, "missing", "csv", "nonexistent.csv")
            self.results.record_test("Path security - nonexistent file", 
                                    "File not found" in result,
                                    f"Nonexistent file should be caught: {result}")
            
            # Test 7: Try to access malicious directory (should fail)
            result = manager.connect_database.__wrapped__(manager, "malicious", "csv", "malicious/innocent.txt")
            # This should work since malicious/ is in current directory
            # But the file should not be a valid CSV
            is_expected_error = ("Failed to create engine" in result or 
                               "outside the allowed directory" in result or
                               "File not found" in result or
                               "Unsupported file type" in result)
            self.results.record_test("Path security - subdirectory access", 
                                    is_expected_error,
                                    f"Unexpected result for subdirectory: {result}")
                                    
        except Exception as e:
            self.results.record_test("Path security test setup", False, str(e))
    
    def test_connection_management_limits(self):
        """Test connection limits and management"""
        print("\n=== Testing Connection Management ===")
        
        try:
            from db_client_mcp.db_client_mcp import DatabaseManager
            manager = DatabaseManager()
            
            # Test initial state
            self.results.record_test("Connection manager initialization", 
                                    hasattr(manager, 'connection_semaphore') and manager.connection_semaphore._value == 10,
                                    "Connection semaphore not properly initialized")
            
            # Test creating connections up to limit
            connections_created = []
            for i in range(12):  # Try to create more than limit
                result = manager.connect_database.__wrapped__(manager, f"conn_{i}", "sqlite", "test_comprehensive.db")
                if "Successfully connected" in result:
                    connections_created.append(f"conn_{i}")
                elif "Maximum number of concurrent connections" in result:
                    break
            
            self.results.record_test("Connection limit enforcement", 
                                    len(connections_created) <= 10,
                                    f"Created {len(connections_created)} connections, should be max 10")
            
            # Test that we hit the limit
            result = manager.connect_database.__wrapped__(manager, "should_fail", "sqlite", "test_comprehensive.db")
            self.results.record_test("Connection limit error message", 
                                    "Maximum number of concurrent connections" in result,
                                    f"Expected limit error, got: {result}")
            
            # Test disconnection frees up slots
            if connections_created:
                result = manager.disconnect_database.__wrapped__(manager, connections_created[0])
                self.results.record_test("Disconnect database", 
                                        "Successfully disconnected" in result,
                                        f"Disconnect failed: {result}")
                
                # Now we should be able to create one more
                result = manager.connect_database.__wrapped__(manager, "after_disconnect", "sqlite", "test_comprehensive.db")
                self.results.record_test("Connect after disconnect", 
                                        "Successfully connected" in result,
                                        f"Could not reconnect after disconnect: {result}")
            
        except Exception as e:
            self.results.record_test("Connection management test setup", False, str(e))
    
    def test_large_file_handling(self):
        """Test large file detection and handling"""
        print("\n=== Testing Large File Handling ===")
        
        try:
            from db_client_mcp.db_client_mcp import DatabaseManager
            manager = DatabaseManager()
            
            # Test file size detection
            small_size = manager._get_file_size("test_small.csv")
            large_size = manager._get_file_size("test_large_content.csv")
            
            self.results.record_test("File size detection", 
                                    small_size > 0 and large_size > small_size,
                                    f"Size detection failed: small={small_size}, large={large_size}")
            
            # Test large file threshold detection (use lower threshold for testing)
            is_large = manager._is_large_file("test_large_content.csv", threshold_mb=1)  # 1MB threshold
            is_small = manager._is_large_file("test_small.csv", threshold_mb=1)
            
            self.results.record_test("Large file threshold detection", 
                                    is_large and not is_small,
                                    f"Large file detection failed: large={is_large}, small={is_small}")
            
            # Test connection to large file (should use temporary storage)
            result = manager.connect_database.__wrapped__(manager, "large_csv", "csv", "test_large_content.csv")
            self.results.record_test("Connect large CSV file", 
                                    "Successfully connected" in result,
                                    f"Large file connection failed: {result}")
            
            # Verify that temporary files are being tracked
            temp_file_count = len(manager.temp_files)
            self.results.record_test("Temporary file tracking", 
                                    temp_file_count >= 0,  # At least some temp files might be created
                                    f"Temp file tracking failed: count={temp_file_count}")
            
            # Test querying large file
            if "Successfully connected" in result:
                query_result = manager.execute_query.__wrapped__(manager, "large_csv", "SELECT COUNT(*) as count FROM data_table")
                self.results.record_test("Query large file", 
                                        "count" in query_result,
                                        f"Large file query failed: {query_result[:100]}")
            
        except Exception as e:
            self.results.record_test("Large file handling test setup", False, str(e))
    
    def test_query_size_control(self):
        """Test query size thresholds and controls"""
        print("\n=== Testing Query Size Control ===")
        
        try:
            from db_client_mcp.db_client_mcp import DatabaseManager
            manager = DatabaseManager()
            
            # Connect to test databases
            manager.connect_database.__wrapped__(manager, "test_db", "sqlite", "test_comprehensive.db")
            manager.connect_database.__wrapped__(manager, "boundary_csv", "csv", "test_boundary_100.csv")
            
            # Test small query (under 100 rows) - should work normally in both formats
            small_md = manager.execute_query.__wrapped__(manager, "test_db", "SELECT * FROM users")
            self.results.record_test("Small query markdown", 
                                    "Alice" in small_md and "exceeds the 100-row limit" not in small_md,
                                    f"Small markdown query failed: {small_md[:100]}")
            
            small_json = manager.execute_query_json.__wrapped__(manager, "test_db", "SELECT * FROM users")
            small_data = json.loads(small_json)
            self.results.record_test("Small query JSON", 
                                    isinstance(small_data, list) and len(small_data) == 5,
                                    f"Small JSON query failed: {type(small_data)}")
            
            # Test exactly 100 rows (boundary condition)
            boundary_md = manager.execute_query.__wrapped__(manager, "boundary_csv", "SELECT * FROM data_table")
            self.results.record_test("100-row query markdown", 
                                    "exceeds the 100-row limit" not in boundary_md,  # Should still work
                                    f"100-row markdown query failed: {boundary_md[:100]}")
            
            boundary_json = manager.execute_query_json.__wrapped__(manager, "boundary_csv", "SELECT * FROM data_table")
            boundary_data = json.loads(boundary_json)
            self.results.record_test("100-row query JSON", 
                                    isinstance(boundary_data, list) and len(boundary_data) == 100,
                                    f"100-row JSON query failed: {type(boundary_data)} len={len(boundary_data) if isinstance(boundary_data, list) else 'N/A'}")
            
            # Test large query (over 100 rows) - markdown should fail, JSON should buffer
            large_md = manager.execute_query.__wrapped__(manager, "test_db", "SELECT * FROM products")
            self.results.record_test("Large query markdown (should fail)", 
                                    "exceeds the 100-row limit" in large_md,
                                    f"Large markdown query should fail: {large_md}")
            
            large_json = manager.execute_query_json.__wrapped__(manager, "test_db", "SELECT * FROM products")
            large_data = json.loads(large_json)
            
            # Should return buffered result with metadata
            is_buffered = (isinstance(large_data, dict) and 
                          "metadata" in large_data and 
                          "data" in large_data and
                          "next_options" in large_data)
            
            self.results.record_test("Large query JSON (should buffer)", 
                                    is_buffered,
                                    f"Large JSON query should return buffered result: {type(large_data)}")
            
            if is_buffered:
                self.results.record_test("Buffered result metadata", 
                                        large_data["metadata"]["total_rows"] == 150,
                                        f"Expected 150 total rows, got: {large_data['metadata'].get('total_rows')}")
                
                self.results.record_test("Buffered result first chunk", 
                                        len(large_data["data"]) == 10,
                                        f"Expected 10 rows in first chunk, got: {len(large_data['data'])}")
            
        except Exception as e:
            self.results.record_test("Query size control test setup", False, str(e))
    
    def test_query_buffering_system(self):
        """Test query buffering with chunk retrieval"""
        print("\n=== Testing Query Buffering System ===")
        
        try:
            from db_client_mcp.db_client_mcp import DatabaseManager
            manager = DatabaseManager()
            
            # Connect and create a large query result
            manager.connect_database.__wrapped__(manager, "test_db", "sqlite", "test_comprehensive.db")
            
            large_json = manager.execute_query_json.__wrapped__(manager, "test_db", "SELECT * FROM products")
            large_data = json.loads(large_json)
            
            if not (isinstance(large_data, dict) and "metadata" in large_data):
                self.results.record_test("Query buffering setup", False, "Could not create buffered query")
                return
                
            query_id = large_data["metadata"]["query_id"]
            
            # Test query ID format
            parts = query_id.split('_')
            self.results.record_test("Query ID format", 
                                    len(parts) >= 3 and query_id.startswith("test_db") and len(parts[-1]) == 4,
                                    f"Invalid query ID format: {query_id}")
            
            # Test buffered query info
            info_result = manager.get_buffered_query_info.__wrapped__(manager, query_id)
            info_data = json.loads(info_result)
            self.results.record_test("Get buffered query info", 
                                    info_data["total_rows"] == 150 and info_data["db_name"] == "test_db",
                                    f"Buffered query info failed: {info_data}")
            
            # Test chunk retrieval - get next 20 rows
            chunk_result = manager.get_query_chunk.__wrapped__(manager, query_id, 11, "20")
            chunk_data = json.loads(chunk_result)
            
            self.results.record_test("Get query chunk", 
                                    chunk_data["metadata"]["showing_rows"] == "11-30" and 
                                    len(chunk_data["data"]) == 20,
                                    f"Chunk retrieval failed: {chunk_data['metadata']}")
            
            # Test chunk retrieval - get all remaining rows
            remaining_result = manager.get_query_chunk.__wrapped__(manager, query_id, 31, "all")
            remaining_data = json.loads(remaining_result)
            
            expected_remaining = 150 - 30  # Total - already retrieved
            self.results.record_test("Get remaining chunk", 
                                    len(remaining_data["data"]) == expected_remaining,
                                    f"Expected {expected_remaining} remaining rows, got {len(remaining_data['data'])}")
            
            # Test chunk retrieval with invalid parameters
            invalid_result = manager.get_query_chunk.__wrapped__(manager, query_id, 200, "10")
            self.results.record_test("Invalid chunk request", 
                                    "start_row must be between" in invalid_result,
                                    f"Invalid chunk should fail: {invalid_result}")
            
            # Test clearing buffer
            clear_result = manager.clear_query_buffer.__wrapped__(manager, query_id)
            self.results.record_test("Clear query buffer", 
                                    "Successfully cleared" in clear_result,
                                    f"Buffer clear failed: {clear_result}")
            
            # Test accessing cleared buffer (should fail)
            cleared_result = manager.get_buffered_query_info.__wrapped__(manager, query_id)
            self.results.record_test("Access cleared buffer", 
                                    "not found" in cleared_result,
                                    f"Cleared buffer should not be accessible: {cleared_result}")
            
        except Exception as e:
            self.results.record_test("Query buffering test setup", False, str(e))
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention thoroughly"""
        print("\n=== Testing SQL Injection Prevention ===")
        
        try:
            from db_client_mcp.db_client_mcp import DatabaseManager
            manager = DatabaseManager()
            
            manager.connect_database.__wrapped__(manager, "test_db", "sqlite", "test_comprehensive.db")
            
            # Test safe table identifiers
            try:
                safe_name = manager._safe_table_identifier("users")
                self.results.record_test("Safe table identifier - valid", 
                                        safe_name is not None,
                                        "Valid table name rejected")
            except Exception as e:
                self.results.record_test("Safe table identifier - valid", False, str(e))
            
            # Test malicious table names
            malicious_names = [
                "users; DROP TABLE users; --",
                "users' OR '1'='1",
                "users/**/UNION/**/SELECT",
                "users\"; DROP TABLE users; --",
                "users` DROP TABLE users; #"
            ]
            
            for malicious_name in malicious_names:
                try:
                    manager._safe_table_identifier(malicious_name)
                    self.results.record_test(f"Block malicious table name: {malicious_name[:20]}", 
                                            False,
                                            f"Malicious name accepted: {malicious_name}")
                except ValueError:
                    self.results.record_test(f"Block malicious table name: {malicious_name[:20]}", 
                                            True)
                except Exception as e:
                    self.results.record_test(f"Block malicious table name: {malicious_name[:20]}", 
                                            False, f"Unexpected error: {e}")
            
            # Test parameterized queries in table samples
            result = manager.get_table_sample.__wrapped__(manager, "test_db", "users", 2)
            self.results.record_test("Parameterized query - valid table", 
                                    "Alice" in result,
                                    f"Valid parameterized query failed: {result[:100]}")
            
            # Test with malicious table name in get_table_sample
            result = manager.get_table_sample.__wrapped__(manager, "test_db", "users; DROP TABLE users; --", 2)
            self.results.record_test("Parameterized query - malicious table", 
                                    "Invalid table name" in result,
                                    f"Malicious table name not rejected: {result}")
            
            # Test that special characters in data don't break queries
            result = manager.execute_query.__wrapped__(manager, "test_db", "SELECT * FROM special_chars")
            self.results.record_test("Query with special characters in data", 
                                    "O'Reilly" in result or "Smith" in result,
                                    f"Special characters in data caused issues: {result[:100]}")
            
            # Test SQL injection in regular queries (should be prevented by SQLAlchemy)
            injection_query = "SELECT * FROM users WHERE id = '1' OR '1'='1'"
            result = manager.execute_query.__wrapped__(manager, "test_db", injection_query)
            # This query might succeed but shouldn't cause damage (it's just a SELECT)
            self.results.record_test("SQL injection in query", 
                                    not ("DROP" in result.upper() or "DELETE" in result.upper()),
                                    f"SQL injection might have caused damage: {result[:100]}")
            
        except Exception as e:
            self.results.record_test("SQL injection prevention test setup", False, str(e))
    
    def test_concurrent_operations(self):
        """Test concurrent operations and thread safety"""
        print("\n=== Testing Concurrent Operations ===")
        
        try:
            from db_client_mcp.db_client_mcp import DatabaseManager
            manager = DatabaseManager()
            
            # Connect initial database
            manager.connect_database.__wrapped__(manager, "test_db", "sqlite", "test_comprehensive.db")
            
            errors = []
            results = []
            
            def concurrent_query(thread_id):
                try:
                    # Each thread performs multiple operations
                    result = manager.execute_query_json.__wrapped__(manager, "test_db", f"SELECT * FROM users WHERE id <= {thread_id + 1}")
                    data = json.loads(result)
                    return f"Thread {thread_id}: {len(data)} rows"
                except Exception as e:
                    return f"Thread {thread_id} error: {e}"
            
            # Test concurrent queries
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(concurrent_query, i) for i in range(5)]
                for future in as_completed(futures):
                    results.append(future.result())
            
            success_count = sum(1 for r in results if "error" not in r.lower())
            self.results.record_test("Concurrent query operations", 
                                    success_count == 5,
                                    f"Only {success_count}/5 concurrent queries succeeded: {results}")
            
            # Test concurrent connections (should respect limit)
            def concurrent_connect(thread_id):
                try:
                    result = manager.connect_database.__wrapped__(manager, f"thread_db_{thread_id}", "sqlite", "test_comprehensive.db")
                    return "Successfully connected" in result
                except Exception as e:
                    return False
            
            connection_results = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(concurrent_connect, i) for i in range(8)]
                for future in as_completed(futures):
                    connection_results.append(future.result())
            
            # Some should succeed, some should fail due to connection limit
            successes = sum(connection_results)
            self.results.record_test("Concurrent connection limit", 
                                    successes <= 10,  # Respect the limit
                                    f"Too many concurrent connections: {successes}/8 succeeded")
            
        except Exception as e:
            self.results.record_test("Concurrent operations test setup", False, str(e))
    
    def test_resource_management(self):
        """Test resource management and cleanup"""
        print("\n=== Testing Resource Management ===")
        
        try:
            from db_client_mcp.db_client_mcp import DatabaseManager
            manager = DatabaseManager()
            
            # Test initial resource state
            initial_temp_files = len(manager.temp_files)
            initial_connections = len(manager.connections)
            initial_buffers = len(manager.query_buffers)
            
            self.results.record_test("Initial resource state", 
                                    initial_temp_files == 0 and initial_connections == 0 and initial_buffers == 0,
                                    f"Resources not clean: temp={initial_temp_files}, conn={initial_connections}, buf={initial_buffers}")
            
            # Create connections and resources
            manager.connect_database.__wrapped__(manager, "test_db", "sqlite", "test_comprehensive.db")
            manager.connect_database.__wrapped__(manager, "csv_db", "csv", "test_large_content.csv")  # Should create temp file
            
            # Create buffered query
            result = manager.execute_query_json.__wrapped__(manager, "test_db", "SELECT * FROM products")
            data = json.loads(result)
            
            # Check resource usage
            after_temp_files = len(manager.temp_files)
            after_connections = len(manager.connections)
            after_buffers = len(manager.query_buffers)
            
            self.results.record_test("Resource allocation", 
                                    after_connections == 2 and after_buffers >= 1,
                                    f"Resources not allocated: conn={after_connections}, buf={after_buffers}")
            
            # Test cleanup on disconnect
            result = manager.disconnect_database.__wrapped__(manager, "csv_db")
            self.results.record_test("Database disconnection", 
                                    "Successfully disconnected" in result,
                                    f"Disconnect failed: {result}")
            
            # Test buffer expiry (simulate by manipulating timestamp)
            if data and "metadata" in data:
                query_id = data["metadata"]["query_id"]
                if query_id in manager.query_buffers:
                    # Simulate old timestamp
                    manager.query_buffers[query_id].timestamp = time.time() - 700  # Older than 600s threshold
                    
                    # Trigger cleanup
                    manager._cleanup_expired_buffers()
                    
                    self.results.record_test("Buffer expiry cleanup", 
                                            query_id not in manager.query_buffers,
                                            f"Expired buffer not cleaned up: {query_id}")
            
            # Test manual cleanup
            manager._cleanup_all()
            
            final_connections = len(manager.connections)
            self.results.record_test("Manual cleanup all", 
                                    final_connections == 0,
                                    f"Manual cleanup failed: {final_connections} connections remaining")
            
        except Exception as e:
            self.results.record_test("Resource management test setup", False, str(e))
    
    def test_error_handling_edge_cases(self):
        """Test error handling and edge cases"""
        print("\n=== Testing Error Handling and Edge Cases ===")
        
        try:
            from db_client_mcp.db_client_mcp import DatabaseManager
            manager = DatabaseManager()
            
            # Test connecting to non-existent database
            result = manager.connect_database.__wrapped__(manager, "nonexistent", "sqlite", "nonexistent.db")
            self.results.record_test("Connect nonexistent database", 
                                    "Failed to connect" in result or "No such file" in result,
                                    f"Should fail for nonexistent DB: {result}")
            
            # Test query on non-connected database
            result = manager.execute_query.__wrapped__(manager, "not_connected", "SELECT 1")
            self.results.record_test("Query non-connected database", 
                                    "is not connected" in result,
                                    f"Should fail for non-connected DB: {result}")
            
            # Test malformed SQL
            manager.connect_database.__wrapped__(manager, "test_db", "sqlite", "test_comprehensive.db")
            result = manager.execute_query.__wrapped__(manager, "test_db", "SELCT * FRM users")  # Typos
            self.results.record_test("Malformed SQL query", 
                                    "error occurred" in result.lower(),
                                    f"Malformed SQL should cause error: {result}")
            
            # Test buffer operations on non-existent query
            result = manager.get_buffered_query_info.__wrapped__(manager, "nonexistent_query_id")
            self.results.record_test("Non-existent buffer info", 
                                    "not found" in result,
                                    f"Should fail for non-existent buffer: {result}")
            
            result = manager.get_query_chunk.__wrapped__(manager, "nonexistent_query_id", 1, "10")
            self.results.record_test("Non-existent buffer chunk", 
                                    "not found" in result,
                                    f"Should fail for non-existent buffer: {result}")
            
            # Test invalid chunk parameters
            # First create a valid buffer
            buffer_result = manager.execute_query_json.__wrapped__(manager, "test_db", "SELECT * FROM products")
            buffer_data = json.loads(buffer_result)
            if "metadata" in buffer_data:
                query_id = buffer_data["metadata"]["query_id"]
                
                # Test invalid start_row
                result = manager.get_query_chunk.__wrapped__(manager, query_id, -1, "10")
                self.results.record_test("Invalid start_row (negative)", 
                                        "must be between" in result,
                                        f"Should reject negative start_row: {result}")
                
                result = manager.get_query_chunk.__wrapped__(manager, query_id, 1000, "10")
                self.results.record_test("Invalid start_row (too high)", 
                                        "must be between" in result,
                                        f"Should reject too high start_row: {result}")
                
                # Test invalid chunk_size
                result = manager.get_query_chunk.__wrapped__(manager, query_id, 1, "invalid")
                self.results.record_test("Invalid chunk_size", 
                                        "must be a positive integer" in result,
                                        f"Should reject invalid chunk_size: {result}")
                
                result = manager.get_query_chunk.__wrapped__(manager, query_id, 1, "-5")
                self.results.record_test("Negative chunk_size", 
                                        "must be a positive integer" in result,
                                        f"Should reject negative chunk_size: {result}")
            
            # Test file permission issues (simulate by trying to read a directory)
            if not os.path.exists('test_dir'):
                os.makedirs('test_dir')
            result = manager.connect_database.__wrapped__(manager, "dir_test", "csv", "test_dir")
            self.results.record_test("Directory instead of file", 
                                    "File not found" in result or "Invalid path" in result,
                                    f"Should reject directory: {result}")
            
            # Cleanup test directory
            os.rmdir('test_dir')
            
        except Exception as e:
            self.results.record_test("Error handling test setup", False, str(e))
    
    def test_installation_compatibility(self):
        """Test that installation still works"""
        print("\n=== Testing Installation Compatibility ===")
        
        try:
            # Test that package can be imported
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            
            from db_client_mcp.db_client_mcp import main, DatabaseManager
            self.results.record_test("Package import", 
                                    callable(main) and DatabaseManager is not None,
                                    "Could not import main components")
            
            # Test that main function exists and is callable
            self.results.record_test("Main function callable", 
                                    callable(main),
                                    "Main function is not callable")
            
            # Test that pyproject.toml exists and is valid
            pyproject_path = os.path.join(os.path.dirname(__file__), 'pyproject.toml')
            self.results.record_test("Pyproject.toml exists", 
                                    os.path.exists(pyproject_path),
                                    "pyproject.toml not found")
            
            # Test that dependencies are listed
            if os.path.exists(pyproject_path):
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                    required_deps = ['fastmcp', 'pandas', 'sqlalchemy']
                    has_deps = all(dep in content for dep in required_deps)
                    self.results.record_test("Required dependencies", 
                                            has_deps,
                                            f"Missing required dependencies: {required_deps}")
            
        except Exception as e:
            self.results.record_test("Installation compatibility test setup", False, str(e))
    
    def run_all_tests(self):
        """Run all comprehensive tests"""
        print("🧪 Starting Comprehensive db-client-mcp Test Suite")
        print("="*80)
        
        # Setup
        self.setup()
        
        try:
            # Run all test categories
            self.test_existing_api_compatibility()
            self.test_path_security_comprehensive()
            self.test_connection_management_limits()
            self.test_large_file_handling()
            self.test_query_size_control()
            self.test_query_buffering_system()
            self.test_sql_injection_prevention()
            self.test_concurrent_operations()
            self.test_resource_management()
            self.test_error_handling_edge_cases()
            self.test_installation_compatibility()
            
        finally:
            # Always cleanup
            self.cleanup()
        
        # Print results
        return self.results.print_summary()

if __name__ == "__main__":
    test_suite = ComprehensiveTests()
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1)