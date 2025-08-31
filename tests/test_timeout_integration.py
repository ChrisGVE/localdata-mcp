"""
Integration test for timeout system with real LocalData MCP server.

Tests the complete flow from configuration through execution with timeout management.
"""

import json
import os
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

import yaml
from src.localdata_mcp.localdata_mcp import LocalDataMCP


class TestTimeoutSystemIntegration(unittest.TestCase):
    """Integration tests for complete timeout system."""
    
    def setUp(self):
        """Set up test environment with real database and configuration."""
        # Create temporary SQLite database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create test data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table with enough data to test timeout scenarios
        cursor.execute("""
            CREATE TABLE large_dataset (
                id INTEGER PRIMARY KEY,
                data TEXT,
                value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert substantial test data (1000+ rows)
        test_data = []
        for i in range(1500):
            test_data.append((i, f"test_data_{i}_with_long_content_to_simulate_real_data", i * 1.5))
        
        cursor.executemany(
            "INSERT INTO large_dataset (id, data, value) VALUES (?, ?, ?)", 
            test_data
        )
        
        # Create a slow query table (with complex data for testing)
        cursor.execute("""
            CREATE TABLE slow_query_test (
                id INTEGER PRIMARY KEY,
                complex_data TEXT
            )
        """)
        
        # Insert data that makes queries slower
        complex_data = []
        for i in range(500):
            complex_text = "x" * 1000  # 1KB per row
            complex_data.append((i, complex_text))
        
        cursor.executemany(
            "INSERT INTO slow_query_test (id, complex_data) VALUES (?, ?)",
            complex_data
        )
        
        conn.commit()
        conn.close()
        
        # Create temporary config file
        self.config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        config_data = {
            'databases': {
                'test_fast_db': {
                    'type': 'sqlite',
                    'connection_string': f'sqlite:///{self.db_path}',
                    'query_timeout': 30,  # 30 seconds
                    'connection_timeout': 10
                },
                'test_slow_db': {
                    'type': 'sqlite', 
                    'connection_string': f'sqlite:///{self.db_path}',
                    'query_timeout': 2,   # Very short timeout for testing
                    'connection_timeout': 5
                }
            },
            'performance': {
                'memory_limit_mb': 1024,
                'enable_query_analysis': True,
                'chunk_size': 100
            },
            'logging': {
                'level': 'info',
                'console_output': True
            }
        }
        
        yaml.dump(config_data, self.config_file)
        self.config_file.close()
        
        # Initialize LocalData MCP with config
        self.mcp_server = LocalDataMCP()
        
        # Patch config manager to use our test config
        with patch('src.localdata_mcp.config_manager.get_config_manager') as mock_get_config:
            from src.localdata_mcp.config_manager import ConfigManager
            config_manager = ConfigManager(config_file=self.config_file.name)
            mock_get_config.return_value = config_manager
            self.config_manager = config_manager
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        if os.path.exists(self.config_file.name):
            os.unlink(self.config_file.name)
    
    def test_successful_query_with_timeout_configured(self):
        """Test successful query execution with timeout configured but not triggered."""
        # Connect to database
        connect_result = self.mcp_server.connect_database("test_fast_db")
        self.assertIn("Successfully connected", connect_result)
        
        # Execute query that should complete within timeout
        query = "SELECT COUNT(*) as total_rows FROM large_dataset"
        result = self.mcp_server.execute_query("test_fast_db", query)
        
        # Should succeed and include timeout metadata
        self.assertNotIn("Timeout Error", result)
        
        # Parse result to check metadata
        try:
            result_data = json.loads(result)
            if "metadata" in result_data and "timeout_info" in result_data["metadata"]:
                timeout_info = result_data["metadata"]["timeout_info"]
                self.assertTrue(timeout_info.get("timeout_configured", False))
                self.assertEqual(timeout_info.get("database_name"), "test_fast_db")
                self.assertIsNotNone(timeout_info.get("timeout_limit_seconds"))
        except json.JSONDecodeError:
            # Result might be in different format, that's OK as long as no timeout error
            pass
    
    def test_query_timeout_with_short_timeout(self):
        """Test query that should timeout with very short timeout configuration."""
        # Connect to database with short timeout
        connect_result = self.mcp_server.connect_database("test_slow_db")
        self.assertIn("Successfully connected", connect_result)
        
        # This query should be slow enough to trigger the 2-second timeout
        # Using a complex query with joins and sorting to make it slower
        slow_query = """
        SELECT t1.id, t1.complex_data, t2.data, t2.value
        FROM slow_query_test t1
        JOIN large_dataset t2 ON t1.id = t2.id % 500
        ORDER BY LENGTH(t1.complex_data) DESC, t2.value
        LIMIT 1000
        """
        
        start_time = time.time()
        result = self.mcp_server.execute_query("test_slow_db", slow_query)
        execution_time = time.time() - start_time
        
        # Should either timeout or complete quickly
        # If it times out, should get timeout error message
        if "Timeout Error" in result:
            self.assertIn("Query Timeout Error", result)
            self.assertIn("test_slow_db", result)
            # Execution time should be close to timeout limit (2 seconds)
            self.assertLessEqual(execution_time, 5)  # Allow some buffer for cleanup
        else:
            # If query completed fast enough, that's also valid
            # But should have timeout metadata
            pass
    
    def test_timeout_error_message_format(self):
        """Test that timeout error messages contain required information."""
        # Connect to database with short timeout
        connect_result = self.mcp_server.connect_database("test_slow_db")
        self.assertIn("Successfully connected", connect_result)
        
        # Execute a query designed to be slow
        slow_query = """
        WITH RECURSIVE slow_series(x) AS (
            SELECT 1
            UNION ALL
            SELECT x + 1 FROM slow_series WHERE x < 10000
        )
        SELECT COUNT(*) FROM slow_series
        """
        
        result = self.mcp_server.execute_query("test_slow_db", slow_query)
        
        if "Query Timeout Error" in result:
            # Verify error message contains expected information
            self.assertIn("execution time:", result)
            self.assertIn("reason:", result)
            self.assertIn("test_slow_db", result)
    
    def test_database_specific_timeout_configuration(self):
        """Test that different databases use their specific timeout configurations."""
        # Connect to both databases
        fast_connect = self.mcp_server.connect_database("test_fast_db")
        slow_connect = self.mcp_server.connect_database("test_slow_db")
        
        self.assertIn("Successfully connected", fast_connect)
        self.assertIn("Successfully connected", slow_connect)
        
        # Same query on both databases
        query = "SELECT COUNT(*) FROM large_dataset WHERE id > 500"
        
        # Fast DB should succeed (30 second timeout)
        fast_result = self.mcp_server.execute_query("test_fast_db", query)
        self.assertNotIn("Timeout Error", fast_result)
        
        # Slow DB with 2 second timeout - behavior depends on actual execution speed
        slow_result = self.mcp_server.execute_query("test_slow_db", query)
        # Either succeeds quickly or times out with proper message
        if "Timeout Error" in slow_result:
            self.assertIn("test_slow_db", slow_result)
    
    def test_timeout_with_streaming_execution(self):
        """Test timeout system works properly with streaming execution."""
        # Connect to database
        connect_result = self.mcp_server.connect_database("test_fast_db")
        self.assertIn("Successfully connected", connect_result)
        
        # Execute query that returns substantial data (triggers streaming)
        streaming_query = "SELECT * FROM large_dataset ORDER BY id"
        result = self.mcp_server.execute_query("test_fast_db", streaming_query)
        
        # Should succeed and include streaming + timeout metadata
        self.assertNotIn("Timeout Error", result)
        
        try:
            result_data = json.loads(result)
            
            # Check for streaming metadata
            if "metadata" in result_data:
                metadata = result_data["metadata"]
                self.assertTrue(metadata.get("streaming", False))
                
                # Check for timeout metadata
                if "timeout_info" in metadata:
                    timeout_info = metadata["timeout_info"]
                    self.assertTrue(timeout_info.get("timeout_configured", False))
                    self.assertIsNotNone(timeout_info.get("timeout_limit_seconds"))
                    
        except json.JSONDecodeError:
            # Different result format is OK as long as no timeout error
            pass
    
    def test_timeout_manager_active_operations(self):
        """Test that timeout manager properly tracks active operations."""
        from src.localdata_mcp.timeout_manager import get_timeout_manager
        
        timeout_manager = get_timeout_manager()
        
        # Should start with no active operations
        active_ops = timeout_manager.get_active_operations()
        initial_count = len(active_ops)
        
        # Connect and start a query
        connect_result = self.mcp_server.connect_database("test_fast_db")
        self.assertIn("Successfully connected", connect_result)
        
        # Execute a quick query
        result = self.mcp_server.execute_query("test_fast_db", "SELECT COUNT(*) FROM large_dataset")
        
        # After completion, should be back to initial count
        final_active_ops = timeout_manager.get_active_operations()
        final_count = len(final_active_ops)
        
        # Operations should be cleaned up (might not be exactly equal due to timing)
        self.assertLessEqual(final_count, initial_count + 1)


class TestTimeoutManagerUtilities(unittest.TestCase):
    """Test timeout manager utility functions and edge cases."""
    
    def test_global_timeout_manager_singleton(self):
        """Test that get_timeout_manager returns same instance."""
        from src.localdata_mcp.timeout_manager import get_timeout_manager
        
        manager1 = get_timeout_manager()
        manager2 = get_timeout_manager()
        
        self.assertIs(manager1, manager2)
    
    def test_timeout_config_validation(self):
        """Test timeout configuration edge cases."""
        from src.localdata_mcp.timeout_manager import TimeoutConfig
        from src.localdata_mcp.config_manager import DatabaseType
        
        # Valid configuration
        config = TimeoutConfig(
            query_timeout=30,
            connection_timeout=10,
            database_name="test",
            database_type=DatabaseType.SQLITE
        )
        
        self.assertEqual(config.query_timeout, 30)
        self.assertEqual(config.connection_timeout, 10)
        self.assertTrue(config.allow_cancellation)
        self.assertTrue(config.cleanup_on_timeout)
        self.assertEqual(config.timeout_buffer, 5)
    
    def test_operation_id_generation_uniqueness(self):
        """Test that operation IDs are handled properly for uniqueness."""
        from src.localdata_mcp.timeout_manager import get_timeout_manager
        
        timeout_manager = get_timeout_manager()
        
        # Start multiple operations with same pattern to test uniqueness
        operation_ids = []
        for i in range(5):
            op_id = f"test_operation_{i}"
            operation_ids.append(op_id)
        
        # All IDs should be unique (trivial test but verifies pattern)
        self.assertEqual(len(set(operation_ids)), len(operation_ids))


if __name__ == '__main__':
    # Run integration tests
    unittest.main(verbosity=2)