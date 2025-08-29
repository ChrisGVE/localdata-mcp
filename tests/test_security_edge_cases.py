"""Security and edge case tests for localdata-mcp spreadsheet support."""

import os
import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from localdata_mcp import DatabaseManager


class TestSecurityFeatures:
    """Test security features and protections."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh DatabaseManager instance for testing."""
        return DatabaseManager()
    
    @pytest.fixture
    def test_data_dir(self):
        """Get path to test data directory."""
        return Path(__file__).parent / "data"
    
    def test_path_traversal_prevention(self, manager):
        """Test prevention of path traversal attacks."""
        dangerous_paths = [
            "../etc/passwd",
            "../../etc/shadow", 
            "/etc/passwd",
            "/usr/bin/python",
            "..\\windows\\system32\\config\\sam",  # Windows paths
            "C:\\windows\\system32\\config\\sam",
            "../../../../proc/version",
            "/dev/null",
            "/tmp/../etc/passwd"
        ]
        
        for path in dangerous_paths:
            with pytest.raises(ValueError, match="outside the allowed directory|File not found"):
                manager._sanitize_path(path)
    
    def test_symlink_handling(self, manager):
        """Test handling of symbolic links.""" 
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write("col1,col2\nval1,val2\n")
            temp_path = temp_file.name
        
        # Create a symbolic link
        link_path = temp_path + ".link"
        
        try:
            os.symlink(temp_path, link_path)
            
            # Should be able to access via symlink if it's in allowed directory
            result = manager._sanitize_path(link_path)
            assert result  # Should not raise exception
            
        except OSError:
            # Symlink creation might fail on some systems
            pytest.skip("Cannot create symlinks on this system")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(link_path):
                os.unlink(link_path)
    
    def test_file_size_limits(self, manager):
        """Test file size limit enforcement."""
        # Create a relatively small file and test size detection
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            # Write a small amount of data
            for i in range(100):
                temp_file.write(f"row_{i},value_{i}\n")
            temp_path = temp_file.name
        
        try:
            # Test with very low threshold to trigger "large file" handling
            is_large = manager._is_large_file(temp_path, threshold_mb=0.001)  # 1KB threshold
            assert is_large
            
            # Test with normal threshold
            is_large_normal = manager._is_large_file(temp_path, threshold_mb=100)
            assert not is_large_normal
            
            # Test file size detection
            size = manager._get_file_size(temp_path)
            assert size > 0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_connection_limits(self, manager):
        """Test connection limit enforcement."""
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write("col1,col2\nval1,val2\n")
            temp_path = temp_file.name
        
        connections = []
        try:
            # Create connections up to the limit (default is 10)
            for i in range(10):
                conn_name = f"conn_{i}"
                result = manager.connect_database(conn_name, "csv", temp_path)
                if "Successfully connected" in result:
                    connections.append(conn_name)
            
            # Next connection should fail
            result = manager.connect_database("over_limit", "csv", temp_path)
            assert "Maximum number of concurrent connections" in result or "connection limit" in result.lower()
            
        finally:
            # Clean up connections
            for conn_name in connections:
                try:
                    manager.disconnect_database(conn_name)
                except:
                    pass
            
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_malicious_xml_handling(self, manager, test_data_dir):
        """Test handling of malicious XML files."""
        malicious_xml = test_data_dir / "malicious.xml"
        
        # Attempt to connect to malicious XML
        result = manager.connect_database("malicious", "xml", str(malicious_xml))
        
        # Should either fail safely or handle without exposing sensitive data
        if "Successfully connected" in result:
            # If connection succeeded, ensure no sensitive data is exposed
            query_result = manager.execute_query("malicious", "SELECT * FROM data_table")
            # Should not contain contents of /etc/passwd
            assert "root:" not in query_result
            assert "/bin/bash" not in query_result
            manager.disconnect_database("malicious")
        else:
            # Connection failure is acceptable for malicious files
            assert "error" in result.lower() or "failed" in result.lower()
    
    def test_sql_injection_prevention(self, manager):
        """Test SQL injection prevention in table names."""
        # Test invalid table names that could be SQL injection attempts
        invalid_names = [
            "users; DROP TABLE users; --",
            "'; DELETE FROM data; --", 
            "UNION SELECT * FROM sqlite_master",
            "users/*comment*/",
            "users UNION SELECT password FROM accounts",
            "users'; DROP TABLE sqlite_master; --"
        ]
        
        for table_name in invalid_names:
            with pytest.raises(ValueError, match="Invalid table name"):
                manager._safe_table_identifier(table_name)
    
    def test_resource_cleanup(self, manager):
        """Test proper cleanup of resources."""
        temp_files = []
        
        try:
            # Create multiple connections with temporary files
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                    temp_file.write("col1,col2\nval1,val2\n")
                    temp_path = temp_file.name
                    temp_files.append(temp_path)
                
                conn_name = f"cleanup_test_{i}"
                result = manager.connect_database(conn_name, "csv", temp_path)
                assert "Successfully connected" in result
            
            # Test cleanup on exit
            manager._cleanup_all()
            
            # All connections should be cleared
            assert len(manager.connections) == 0
            
        finally:
            # Manual cleanup
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_concurrent_access_safety(self, manager):
        """Test thread safety for concurrent access."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write("id,value\n")
            for i in range(100):
                temp_file.write(f"{i},value_{i}\n")
            temp_path = temp_file.name
        
        results = []
        errors = []
        
        def worker(worker_id):
            """Worker function for concurrent testing."""
            try:
                conn_name = f"worker_{worker_id}"
                result = manager.connect_database(conn_name, "csv", temp_path)
                results.append(result)
                
                if "Successfully connected" in result:
                    # Perform some queries
                    query_result = manager.execute_query(conn_name, "SELECT COUNT(*) FROM data_table")
                    results.append(query_result)
                    
                    manager.disconnect_database(conn_name)
                    
            except Exception as e:
                errors.append(str(e))
        
        try:
            # Start multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check results - should handle concurrent access gracefully
            assert len(errors) == 0 or all("connection limit" in error.lower() for error in errors)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestEdgeCases:
    """Test various edge cases and error conditions."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh DatabaseManager instance for testing."""
        return DatabaseManager()
    
    @pytest.fixture
    def test_data_dir(self):
        """Get path to test data directory."""
        return Path(__file__).parent / "data"
    
    def test_empty_file_handling(self, manager, test_data_dir):
        """Test handling of empty files."""
        empty_file = test_data_dir / "empty.csv"
        
        result = manager.connect_database("empty_test", "csv", str(empty_file))
        
        # Should handle empty files gracefully (might succeed or fail)
        if "Successfully connected" in result:
            # If connected, querying should work without crashing
            query_result = manager.execute_query("empty_test", "SELECT COUNT(*) FROM data_table")
            # Should return 0 or handle empty result gracefully
            manager.disconnect_database("empty_test")
    
    def test_corrupted_file_handling(self, manager, test_data_dir):
        """Test handling of corrupted files."""
        corrupted_file = test_data_dir / "corrupted.xlsx"
        
        # Should fail gracefully for corrupted Excel file
        result = manager.connect_database("corrupted_test", "excel", str(corrupted_file))
        assert "Failed to connect" in result or "error" in result.lower()
    
    def test_unsupported_file_types(self, manager):
        """Test handling of unsupported file types."""
        # Create a file with unsupported extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.unknown', delete=False) as temp_file:
            temp_file.write("some content")
            temp_path = temp_file.name
        
        try:
            # Should fail for unsupported type
            with pytest.raises(ValueError, match="Unsupported db_type"):
                manager._get_engine("unknown_type", temp_path)
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_memory_pressure_handling(self, manager):
        """Test handling of memory pressure with large datasets."""
        # Create a larger dataset to test memory handling
        large_data = []
        for i in range(10000):
            large_data.append(f"{i},item_{i},{i*1.5},{i%10}\n")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write("id,name,value,category\n")
            temp_file.writelines(large_data)
            temp_path = temp_file.name
        
        try:
            # Should handle large files appropriately
            result = manager.connect_database("large_test", "csv", temp_path)
            assert "Successfully connected" in result
            
            # Query should work
            query_result = manager.execute_query("large_test", "SELECT COUNT(*) FROM data_table")
            assert "10000" in query_result
            
            manager.disconnect_database("large_test")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_unicode_handling(self, manager):
        """Test handling of Unicode characters in data."""
        unicode_data = pd.DataFrame({
            'name': ['José', 'München', '北京', '🌟Star', 'Café'],
            'description': ['español', 'Deutsch', '中文', 'emoji', 'français'],
            'value': [1, 2, 3, 4, 5]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
            unicode_data.to_csv(temp_file.name, index=False, encoding='utf-8')
            temp_path = temp_file.name
        
        try:
            result = manager.connect_database("unicode_test", "csv", temp_path)
            assert "Successfully connected" in result
            
            # Query should handle Unicode properly
            query_result = manager.execute_query("unicode_test", "SELECT * FROM data_table WHERE name = 'José'")
            assert "José" in query_result or "Jos" in query_result  # Might be encoded differently
            
            manager.disconnect_database("unicode_test")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_special_column_names(self, manager):
        """Test handling of special characters in column names."""
        special_df = pd.DataFrame({
            'Column With Spaces': [1, 2, 3],
            'Column-With-Hyphens': ['a', 'b', 'c'],
            'Column.With.Dots': [1.1, 2.2, 3.3],
            'Column@Special#Chars': ['x', 'y', 'z']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            special_df.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name
        
        try:
            result = manager.connect_database("special_cols", "csv", temp_path)
            assert "Successfully connected" in result
            
            # Should be able to query with cleaned column names
            query_result = manager.execute_query("special_cols", "SELECT COUNT(*) FROM data_table")
            assert "3" in query_result
            
            # Test describe table to see column names
            description = manager.describe_table("special_cols", "data_table")
            # Column names should be present in some form
            assert "Column" in description
            
            manager.disconnect_database("special_cols")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_query_buffer_expiry(self, manager):
        """Test query buffer expiration functionality."""
        # Create test data
        test_df = pd.DataFrame({
            'id': range(200),  # >100 rows to trigger buffering
            'data': [f'item_{i}' for i in range(200)]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            test_df.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name
        
        try:
            result = manager.connect_database("buffer_test", "csv", temp_path)
            assert "Successfully connected" in result
            
            # Execute query that should trigger buffering
            query_result = manager.execute_query("buffer_test", "SELECT * FROM data_table")
            
            # Should contain query_id for buffering
            assert "query_id" in query_result
            
            # Extract query_id
            import json
            result_data = json.loads(query_result)
            query_id = result_data["metadata"]["query_id"]
            
            # Test manual cleanup (simulating expiry)
            with manager.query_buffer_lock:
                if query_id in manager.query_buffers:
                    del manager.query_buffers[query_id]
            
            # Try to access expired buffer
            chunk_result = manager.next_chunk(query_id, 1, "10")
            assert "not found" in chunk_result or "expired" in chunk_result
            
            manager.disconnect_database("buffer_test")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_malformed_data_handling(self, manager):
        """Test handling of malformed data in files."""
        # Create CSV with malformed data
        malformed_content = """name,age,salary
John,25,50000
Jane,thirty,60000
Bob,35,
Alice,,70000
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write(malformed_content)
            temp_path = temp_file.name
        
        try:
            result = manager.connect_database("malformed_test", "csv", temp_path)
            # Should handle malformed data gracefully
            
            if "Successfully connected" in result:
                # Query should work despite malformed data
                query_result = manager.execute_query("malformed_test", "SELECT COUNT(*) FROM data_table")
                # Should count all rows, even those with missing/invalid data
                assert "4" in query_result or "5" in query_result
                
                manager.disconnect_database("malformed_test")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_extremely_long_file_paths(self, manager):
        """Test handling of very long file paths."""
        # Create a nested directory structure
        base_dir = tempfile.mkdtemp()
        long_path_parts = ['very'] * 50 + ['long'] * 50 + ['path'] * 50  # Very long path
        
        try:
            long_dir = base_dir
            for part in long_path_parts[:20]:  # Limit to reasonable length
                long_dir = os.path.join(long_dir, part)
                os.makedirs(long_dir, exist_ok=True)
            
            # Create file in deep directory
            long_file_path = os.path.join(long_dir, "test.csv")
            with open(long_file_path, 'w') as f:
                f.write("col1,col2\nval1,val2\n")
            
            # Should handle long paths (might fail due to OS limits, which is acceptable)
            try:
                result = manager._sanitize_path(long_file_path)
                # If successful, should work normally
                assert result
            except (OSError, ValueError):
                # Long path errors are acceptable
                pass
                
        finally:
            # Clean up - use os.walk to handle deep directory
            import shutil
            try:
                shutil.rmtree(base_dir)
            except OSError:
                pass  # Cleanup failure is not critical for test


class TestPerformance:
    """Test performance characteristics and limits."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh DatabaseManager instance for testing."""
        return DatabaseManager()
    
    def test_connection_performance(self, manager):
        """Test connection performance with various file sizes."""
        file_sizes = [100, 1000, 5000]  # Number of rows
        
        for size in file_sizes:
            # Create test data
            test_data = []
            for i in range(size):
                test_data.append(f"{i},item_{i},{i*1.5}\n")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_file.write("id,name,value\n")
                temp_file.writelines(test_data)
                temp_path = temp_file.name
            
            try:
                # Measure connection time
                start_time = time.time()
                result = manager.connect_database(f"perf_test_{size}", "csv", temp_path)
                connection_time = time.time() - start_time
                
                assert "Successfully connected" in result
                
                # Connection should complete within reasonable time
                assert connection_time < 10.0  # 10 seconds max
                
                # Test query performance
                start_time = time.time()
                query_result = manager.execute_query(f"perf_test_{size}", "SELECT COUNT(*) FROM data_table")
                query_time = time.time() - start_time
                
                assert str(size) in query_result
                assert query_time < 5.0  # 5 seconds max for query
                
                manager.disconnect_database(f"perf_test_{size}")
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def test_memory_usage_monitoring(self, manager):
        """Test memory usage with various operations."""
        # Create moderately large dataset
        large_data = []
        for i in range(5000):
            large_data.append(f"{i},{'x'*100},{i*1.5}\n")  # Longer strings to use more memory
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_file.write("id,description,value\n")
            temp_file.writelines(large_data)
            temp_path = temp_file.name
        
        try:
            # Connect and perform operations
            result = manager.connect_database("memory_test", "csv", temp_path)
            assert "Successfully connected" in result
            
            # Perform various operations that use memory
            operations = [
                "SELECT COUNT(*) FROM data_table",
                "SELECT * FROM data_table LIMIT 10",
                "SELECT AVG(value) FROM data_table",
                "SELECT id, description FROM data_table WHERE value > 1000"
            ]
            
            for query in operations:
                query_result = manager.execute_query("memory_test", query)
                # Should complete without memory errors
                assert query_result is not None
                assert "error" not in query_result.lower() or "memory" not in query_result.lower()
            
            manager.disconnect_database("memory_test")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])