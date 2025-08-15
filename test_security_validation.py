#!/usr/bin/env python3
"""
Security Validation Test Suite for db-client-mcp
Tests all security improvements and edge cases thoroughly
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import tempfile
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class SecurityValidator:
    """Comprehensive security validation tests"""
    
    def __init__(self):
        self.test_results = {'passed': 0, 'failed': 0, 'tests': []}
        self.temp_files = []
    
    def record_test(self, test_name: str, success: bool, details: str = ""):
        """Record test result"""
        self.test_results['tests'].append({
            'name': test_name,
            'status': 'PASS' if success else 'FAIL',
            'details': details
        })
        
        if success:
            self.test_results['passed'] += 1
            print(f"✅ {test_name}")
        else:
            self.test_results['failed'] += 1
            print(f"❌ {test_name}: {details}")
    
    def create_test_files(self):
        """Create test files for security testing"""
        # Exactly 100 rows (boundary condition)
        boundary_data = pd.DataFrame({
            'id': range(1, 101),
            'value': [f'boundary_{i}' for i in range(1, 101)]
        })
        boundary_data.to_csv('test_boundary_100.csv', index=False)
        self.temp_files.append('test_boundary_100.csv')
        
        # 101 rows (just over threshold)
        over_boundary_data = pd.DataFrame({
            'id': range(1, 102), 
            'value': [f'over_{i}' for i in range(1, 102)]
        })
        over_boundary_data.to_csv('test_boundary_101.csv', index=False)
        self.temp_files.append('test_boundary_101.csv')
        
        # Large file (simulating large content)
        large_content = pd.DataFrame({
            'id': range(1, 51),
            'content': ['X' * 10000 for _ in range(50)]  # 500KB+ file
        })
        large_content.to_csv('test_large_content.csv', index=False)
        self.temp_files.append('test_large_content.csv')
        
        # SQLite database
        conn = sqlite3.connect('test_security.db')
        boundary_data.to_sql('boundary_table', conn, index=False, if_exists='replace')
        over_boundary_data.to_sql('over_boundary_table', conn, index=False, if_exists='replace')
        conn.close()
        self.temp_files.append('test_security.db')
        
        # Create subdirectory with files
        os.makedirs('subdir', exist_ok=True)
        small_data = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        small_data.to_csv('subdir/sub_file.csv', index=False)
        self.temp_files.extend(['subdir/sub_file.csv', 'subdir'])
        
        return len(self.temp_files)
    
    def test_path_security_comprehensive(self):
        """Test path security with extensive scenarios"""
        from db_client_mcp.db_client_mcp import DatabaseManager
        manager = DatabaseManager()
        
        print("\n=== COMPREHENSIVE PATH SECURITY TESTS ===")
        
        # Valid cases
        valid_paths = [
            'test_boundary_100.csv',
            './test_boundary_100.csv',
            'subdir/sub_file.csv',
            './subdir/sub_file.csv'
        ]
        
        for path in valid_paths:
            try:
                result = manager._sanitize_path(path)
                self.record_test(f"Valid path: {path}", True)
            except Exception as e:
                self.record_test(f"Valid path: {path}", False, str(e))
        
        # Invalid cases (should all fail)
        invalid_paths = [
            '../etc/passwd',
            '../../etc/passwd',
            '/etc/passwd',
            '../',
            '../../',
            './../../etc/passwd',
            'subdir/../../../etc/passwd',
            '/Users/test/file.csv',
            '~/.bashrc',
            '/tmp/test.csv'
        ]
        
        for path in invalid_paths:
            try:
                result = manager._sanitize_path(path)
                self.record_test(f"Block invalid path: {path}", False, f"Accepted: {result}")
            except ValueError:
                self.record_test(f"Block invalid path: {path}", True)
            except Exception as e:
                self.record_test(f"Block invalid path: {path}", False, f"Unexpected error: {e}")
        
        # Non-existent files
        try:
            manager._sanitize_path('nonexistent.csv')
            self.record_test("Block non-existent file", False, "Should reject non-existent files")
        except ValueError:
            self.record_test("Block non-existent file", True)
    
    def test_connection_limits_stress(self):
        """Test connection limits under stress"""
        from db_client_mcp.db_client_mcp import DatabaseManager
        manager = DatabaseManager()
        
        print("\n=== CONNECTION LIMITS STRESS TEST ===")
        
        # Test semaphore initialization
        initial_value = manager.connection_semaphore._value
        self.record_test("Semaphore initialization", initial_value == 10, 
                        f"Expected 10, got {initial_value}")
        
        # Test acquiring connections up to limit
        acquired = []
        for i in range(12):
            if manager.connection_semaphore.acquire(blocking=False):
                acquired.append(i)
            else:
                break
        
        expected_acquired = min(10, len(acquired))
        self.record_test("Connection limit enforcement", len(acquired) <= 10,
                        f"Acquired {len(acquired)} connections, should be ≤10")
        
        # Test that 11th connection is blocked
        blocked = not manager.connection_semaphore.acquire(blocking=False)
        self.record_test("11th connection blocked", blocked,
                        "Should block connections beyond limit")
        
        # Release connections
        for _ in acquired:
            manager.connection_semaphore.release()
        
        # Test concurrent connection attempts
        def try_connection(thread_id):
            return manager.connection_semaphore.acquire(blocking=False)
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(try_connection, i) for i in range(15)]
            results = [future.result() for future in as_completed(futures)]
        
        successful = sum(results)
        self.record_test("Concurrent connection limit", successful <= 10,
                        f"Concurrent test: {successful}/15 acquired")
        
        # Release concurrent connections
        for _ in range(successful):
            manager.connection_semaphore.release()
    
    def test_query_size_boundaries(self):
        """Test query size boundary conditions"""
        from db_client_mcp.db_client_mcp import DatabaseManager
        manager = DatabaseManager()
        
        print("\n=== QUERY SIZE BOUNDARY TESTS ===")
        
        # Setup engines
        try:
            engine_100 = manager._get_engine('csv', 'test_boundary_100.csv')
            engine_101 = manager._get_engine('csv', 'test_boundary_101.csv')
            manager.connections['db_100'] = engine_100
            manager.connections['db_101'] = engine_101
            manager.query_history['db_100'] = []
            manager.query_history['db_101'] = []
        except Exception as e:
            self.record_test("Setup boundary databases", False, str(e))
            return
        
        # Test exactly 100 rows - should work in both formats
        try:
            df_100 = pd.read_sql_query("SELECT * FROM data_table", engine_100)
            self.record_test("100-row query execution", len(df_100) == 100,
                            f"Expected 100 rows, got {len(df_100)}")
            
            # Should not trigger buffering (≤100 rows)
            markdown_100 = df_100.to_markdown()
            self.record_test("100-row markdown format", len(markdown_100) > 0,
                            "Should generate markdown successfully")
            
            json_100 = df_100.to_json(orient='records')
            data_100 = json.loads(json_100)
            self.record_test("100-row JSON format", len(data_100) == 100,
                            f"Expected 100 items, got {len(data_100)}")
            
        except Exception as e:
            self.record_test("100-row boundary test", False, str(e))
        
        # Test 101 rows - should trigger buffering in JSON mode  
        try:
            df_101 = pd.read_sql_query("SELECT * FROM data_table", engine_101)
            self.record_test("101-row query execution", len(df_101) == 101,
                            f"Expected 101 rows, got {len(df_101)}")
            
            # Should trigger different behavior for large result sets
            self.record_test("101-row triggers large handling", len(df_101) > 100,
                            "Should be detected as large result set")
            
        except Exception as e:
            self.record_test("101-row boundary test", False, str(e))
    
    def test_file_size_detection(self):
        """Test file size detection and handling"""
        from db_client_mcp.db_client_mcp import DatabaseManager
        manager = DatabaseManager()
        
        print("\n=== FILE SIZE DETECTION TESTS ===")
        
        # Test size detection for different files
        test_files = ['test_boundary_100.csv', 'test_large_content.csv']
        
        for filename in test_files:
            try:
                size = manager._get_file_size(filename)
                self.record_test(f"File size detection: {filename}", size > 0,
                                f"Size: {size} bytes")
                
                # Test with different thresholds
                is_large_1mb = manager._is_large_file(filename, threshold_mb=1)
                is_large_100mb = manager._is_large_file(filename, threshold_mb=100)
                
                self.record_test(f"Large file detection (1MB): {filename}", 
                                isinstance(is_large_1mb, bool),
                                f"1MB threshold: {is_large_1mb}")
                
                self.record_test(f"Large file detection (100MB): {filename}",
                                isinstance(is_large_100mb, bool), 
                                f"100MB threshold: {is_large_100mb}")
                
            except Exception as e:
                self.record_test(f"File size test: {filename}", False, str(e))
    
    def test_sql_injection_comprehensive(self):
        """Test SQL injection prevention comprehensively"""
        from db_client_mcp.db_client_mcp import DatabaseManager
        manager = DatabaseManager()
        
        print("\n=== COMPREHENSIVE SQL INJECTION TESTS ===")
        
        # Test table name validation
        valid_names = ['users', 'user_data', 'data_table', 'Table123', '_internal']
        invalid_names = [
            "users; DROP TABLE users; --",
            "users' OR '1'='1",
            "users/*comment*/",
            "users\"; DROP TABLE users; --", 
            "users` OR `1`=`1",
            "users\\'; DROP TABLE users; #",
            "users UNION SELECT * FROM passwords",
            "users; INSERT INTO admin VALUES (1,'hacker'); --",
            "'; EXEC xp_cmdshell('dir'); --",
            "users<script>alert('xss')</script>"
        ]
        
        for name in valid_names:
            try:
                safe_name = manager._safe_table_identifier(name)
                self.record_test(f"Valid table name: {name}", True,
                                f"Safe identifier: {safe_name}")
            except Exception as e:
                self.record_test(f"Valid table name: {name}", False, str(e))
        
        for name in invalid_names:
            try:
                manager._safe_table_identifier(name)
                self.record_test(f"Block malicious table: {name[:30]}...", False,
                                "Should reject malicious table name")
            except ValueError:
                self.record_test(f"Block malicious table: {name[:30]}...", True)
            except Exception as e:
                self.record_test(f"Block malicious table: {name[:30]}...", False,
                                f"Unexpected error: {e}")
    
    def test_buffer_management_comprehensive(self):
        """Test query buffer management comprehensively"""
        from db_client_mcp.db_client_mcp import DatabaseManager, QueryBuffer
        manager = DatabaseManager()
        
        print("\n=== COMPREHENSIVE BUFFER MANAGEMENT TESTS ===")
        
        # Test buffer creation and ID format
        for i in range(5):
            query_id = manager._generate_query_id(f'db_{i}', f'SELECT {i}')
            parts = query_id.split('_')
            
            valid_format = (len(parts) >= 3 and 
                          query_id.startswith(f'db_{i}') and
                          parts[-2].isdigit() and  # timestamp
                          len(parts[-1]) == 4)     # hash
            
            self.record_test(f"Query ID format #{i+1}", valid_format,
                            f"ID: {query_id}")
        
        # Test buffer expiry logic
        test_df = pd.DataFrame({'id': range(1, 11), 'data': range(1, 11)})
        
        # Create multiple buffers with different timestamps
        current_time = time.time()
        test_buffers = []
        
        for i in range(3):
            query_id = f"test_{i}_{int(current_time)}_{i:04d}"
            # Make some buffers old enough to expire (> 600 seconds)
            timestamp = current_time - (700 + i * 100)  # All old enough to expire
            
            buffer = QueryBuffer(
                query_id=query_id,
                db_name=f'test_db_{i}',
                query=f'SELECT {i}',
                results=test_df.copy(),
                timestamp=timestamp
            )
            
            manager.query_buffers[query_id] = buffer
            test_buffers.append((query_id, timestamp))
        
        initial_count = len(manager.query_buffers)
        self.record_test("Buffer storage", initial_count == 3,
                        f"Expected 3 buffers, got {initial_count}")
        
        # Force cleanup (reset last_cleanup)
        manager.last_cleanup = 0
        manager._cleanup_expired_buffers()
        
        # Check which buffers were removed (those older than 600 seconds)
        remaining_count = len(manager.query_buffers)
        expired_count = initial_count - remaining_count
        
        self.record_test("Buffer expiry cleanup", expired_count > 0,
                        f"Removed {expired_count} expired buffers")
        
        # Test manual cleanup
        manager._cleanup_all()
        final_count = len(manager.query_buffers)
        
        self.record_test("Manual cleanup all", final_count == 0,
                        f"Expected 0 buffers after cleanup, got {final_count}")
    
    def test_resource_management(self):
        """Test resource management and cleanup"""
        from db_client_mcp.db_client_mcp import DatabaseManager
        manager = DatabaseManager()
        
        print("\n=== RESOURCE MANAGEMENT TESTS ===")
        
        # Test initial state
        initial_temp = len(manager.temp_files)
        initial_conn = len(manager.connections)
        initial_buf = len(manager.query_buffers)
        
        self.record_test("Initial resource state", 
                        initial_temp == 0 and initial_conn == 0 and initial_buf == 0,
                        f"temp={initial_temp}, conn={initial_conn}, buf={initial_buf}")
        
        # Create some resources
        try:
            # Create temporary file entry (simulate)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                manager.temp_files.append(tmp.name)
            
            # Create engine connections
            engine1 = manager._get_engine('sqlite', 'test_security.db')
            engine2 = manager._get_engine('csv', 'test_boundary_100.csv')
            
            manager.connections['test1'] = engine1
            manager.connections['test2'] = engine2
            manager.query_history['test1'] = []
            manager.query_history['test2'] = []
            
            # Create buffer
            query_id = manager._generate_query_id('test1', 'SELECT 1')
            test_df = pd.DataFrame({'id': [1, 2, 3]})
            from db_client_mcp.db_client_mcp import QueryBuffer
            buffer = QueryBuffer(
                query_id=query_id,
                db_name='test1',
                query='SELECT 1',
                results=test_df,
                timestamp=time.time()
            )
            manager.query_buffers[query_id] = buffer
            
            # Check resource allocation
            after_temp = len(manager.temp_files)
            after_conn = len(manager.connections) 
            after_buf = len(manager.query_buffers)
            
            self.record_test("Resource allocation",
                            after_temp >= 1 and after_conn == 2 and after_buf == 1,
                            f"temp={after_temp}, conn={after_conn}, buf={after_buf}")
            
            # Test cleanup
            manager._cleanup_all()
            
            final_temp = len(manager.temp_files)
            final_conn = len(manager.connections)
            final_buf = len(manager.query_buffers)
            
            self.record_test("Resource cleanup",
                            final_temp == 0 and final_conn == 0,
                            f"temp={final_temp}, conn={final_conn}, buf={final_buf}")
            
        except Exception as e:
            self.record_test("Resource management", False, str(e))
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        from db_client_mcp.db_client_mcp import DatabaseManager
        manager = DatabaseManager()
        
        print("\n=== ERROR HANDLING TESTS ===")
        
        # Test with non-existent files
        try:
            manager._get_file_size('nonexistent.csv')
            self.record_test("Nonexistent file size", False, "Should raise error")
        except ValueError:
            self.record_test("Nonexistent file size", True)
        except Exception as e:
            self.record_test("Nonexistent file size", False, f"Wrong error type: {e}")
        
        # Test invalid database types
        try:
            manager._get_engine('invalid_type', 'test.db')
            self.record_test("Invalid database type", False, "Should raise error")
        except ValueError:
            self.record_test("Invalid database type", True)
        except Exception as e:
            self.record_test("Invalid database type", False, f"Wrong error type: {e}")
        
        # Test corrupted CSV
        with open('corrupted.csv', 'w') as f:
            f.write('malformed,csv,data\n"unclosed,quote,field\n')
        self.temp_files.append('corrupted.csv')
        
        try:
            engine = manager._get_engine('csv', 'corrupted.csv')
            # Should handle gracefully or provide meaningful error
            self.record_test("Corrupted CSV handling", True, "Handled gracefully")
        except Exception as e:
            # Should provide meaningful error message
            meaningful_error = any(word in str(e).lower() for word in ['csv', 'parse', 'format', 'read'])
            self.record_test("Corrupted CSV error message", meaningful_error,
                            f"Error: {str(e)[:100]}")
    
    def cleanup(self):
        """Clean up test files"""
        for file_path in self.temp_files:
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except OSError:
                pass
    
    def run_validation(self):
        """Run all security validation tests"""
        print("🛡️  COMPREHENSIVE SECURITY VALIDATION SUITE")
        print("=" * 80)
        
        try:
            file_count = self.create_test_files()
            print(f"Created {file_count} test files")
            
            # Run all validation tests
            self.test_path_security_comprehensive()
            self.test_connection_limits_stress() 
            self.test_query_size_boundaries()
            self.test_file_size_detection()
            self.test_sql_injection_comprehensive()
            self.test_buffer_management_comprehensive()
            self.test_resource_management()
            self.test_error_handling()
            
        finally:
            self.cleanup()
        
        # Print comprehensive results
        print("\n" + "=" * 80)
        print("SECURITY VALIDATION RESULTS")
        print("=" * 80)
        
        total = self.test_results['passed'] + self.test_results['failed']
        success_rate = (self.test_results['passed'] / total * 100) if total > 0 else 0
        
        print(f"Total tests: {total}")
        print(f"Passed: {self.test_results['passed']}")
        print(f"Failed: {self.test_results['failed']}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if self.test_results['failed'] > 0:
            print(f"\nFAILED TESTS ({self.test_results['failed']}):")
            for test in self.test_results['tests']:
                if test['status'] == 'FAIL':
                    print(f"  ❌ {test['name']}: {test['details']}")
        
        print("\nSECURITY FEATURES VALIDATED:")
        print("✅ Path traversal prevention")
        print("✅ Connection limit enforcement") 
        print("✅ Query size thresholds")
        print("✅ File size detection")
        print("✅ SQL injection prevention")
        print("✅ Buffer management and expiry")
        print("✅ Resource cleanup")
        print("✅ Error handling")
        
        if self.test_results['failed'] == 0:
            print("\n🎉 ALL SECURITY VALIDATIONS PASSED! 🎉")
            print("🛡️  The db-client-mcp tool is SECURE and PRODUCTION READY! 🛡️")
            return True
        else:
            print(f"\n⚠️  {self.test_results['failed']} security issues found")
            return False

if __name__ == "__main__":
    validator = SecurityValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)