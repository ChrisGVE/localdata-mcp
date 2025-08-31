"""
Comprehensive tests for the QueryTimeoutManager system.

Tests cover timeout configuration, graceful cancellation, database-specific
timeout handling, and integration with the streaming pipeline.
"""

import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

# Import the modules we're testing
from src.localdata_mcp.timeout_manager import (
    QueryTimeoutManager, TimeoutConfig, TimeoutReason, QueryTimeoutError,
    get_timeout_manager, with_timeout
)
from src.localdata_mcp.config_manager import (
    ConfigManager, DatabaseConfig, DatabaseType
)


class TestQueryTimeoutManager(unittest.TestCase):
    """Test cases for QueryTimeoutManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.timeout_manager = QueryTimeoutManager()
        
        # Create a mock config manager with test database configurations
        self.mock_config_manager = Mock(spec=ConfigManager)
        
        # Test database configurations
        self.test_db_config = DatabaseConfig(
            name="test_db",
            type=DatabaseType.SQLITE,
            connection_string="sqlite:///test.db",
            query_timeout=30,
            connection_timeout=10
        )
        
        self.slow_db_config = DatabaseConfig(
            name="slow_db", 
            type=DatabaseType.POSTGRESQL,
            connection_string="postgresql://user:pass@remote:5432/db",
            query_timeout=300,  # 5 minutes for remote DB
            connection_timeout=60
        )
        
        # Mock the config manager to return our test configs
        self.mock_config_manager.get_database_config.side_effect = lambda name: {
            "test_db": self.test_db_config,
            "slow_db": self.slow_db_config
        }.get(name)
        
        # Patch the get_config_manager function
        patcher = patch('src.localdata_mcp.timeout_manager.get_config_manager')
        self.mock_get_config_manager = patcher.start()
        self.mock_get_config_manager.return_value = self.mock_config_manager
        self.addCleanup(patcher.stop)
    
    def test_get_timeout_config_existing_database(self):
        """Test getting timeout config for existing database."""
        config = self.timeout_manager.get_timeout_config("test_db")
        
        self.assertIsInstance(config, TimeoutConfig)
        self.assertEqual(config.database_name, "test_db")
        self.assertEqual(config.database_type, DatabaseType.SQLITE)
        self.assertEqual(config.query_timeout, 30)
        self.assertEqual(config.connection_timeout, 10)
        self.assertTrue(config.allow_cancellation)  # SQLite supports limited cancellation
    
    def test_get_timeout_config_missing_database(self):
        """Test getting timeout config for non-existent database uses defaults."""
        config = self.timeout_manager.get_timeout_config("missing_db")
        
        self.assertIsInstance(config, TimeoutConfig)
        self.assertEqual(config.database_name, "missing_db")
        self.assertEqual(config.database_type, DatabaseType.SQLITE)  # Default
        self.assertEqual(config.query_timeout, 300)  # Default 5 minutes
        self.assertEqual(config.connection_timeout, 30)  # Default 30 seconds
    
    def test_supports_cancellation_network_databases(self):
        """Test that network databases support cancellation."""
        network_types = [
            DatabaseType.POSTGRESQL, DatabaseType.MYSQL, DatabaseType.REDIS,
            DatabaseType.ELASTICSEARCH, DatabaseType.MONGODB, DatabaseType.INFLUXDB,
            DatabaseType.NEO4J, DatabaseType.COUCHDB
        ]
        
        for db_type in network_types:
            with self.subTest(db_type=db_type):
                supports = self.timeout_manager._supports_cancellation(db_type)
                self.assertTrue(supports, f"{db_type} should support cancellation")
    
    def test_supports_cancellation_file_databases(self):
        """Test that file-based databases don't support cancellation."""
        file_types = [
            DatabaseType.CSV, DatabaseType.JSON, DatabaseType.YAML,
            DatabaseType.EXCEL, DatabaseType.XML, DatabaseType.PARQUET
        ]
        
        for db_type in file_types:
            with self.subTest(db_type=db_type):
                supports = self.timeout_manager._supports_cancellation(db_type)
                self.assertFalse(supports, f"{db_type} should not support cancellation")
    
    def test_timeout_context_successful_execution(self):
        """Test timeout context with successful operation."""
        operation_id = "test_op_success"
        timeout_config = TimeoutConfig(
            query_timeout=10,
            connection_timeout=5,
            database_name="test_db",
            database_type=DatabaseType.SQLITE
        )
        
        cleanup_called = []
        
        def cleanup_func():
            cleanup_called.append(True)
        
        with self.timeout_manager.timeout_context(operation_id, timeout_config, cleanup_func) as context:
            # Simulate successful operation
            self.assertFalse(context['cancelled'])
            time.sleep(0.1)  # Small delay to simulate work
            self.assertFalse(self.timeout_manager.is_cancelled(operation_id))
        
        # Cleanup should not be called on successful execution
        self.assertEqual(len(cleanup_called), 0)
    
    def test_timeout_context_timeout_occurs(self):
        """Test timeout context when timeout occurs."""
        operation_id = "test_op_timeout"
        timeout_config = TimeoutConfig(
            query_timeout=1,  # Very short timeout
            connection_timeout=5,
            database_name="test_db",
            database_type=DatabaseType.SQLITE,
            cleanup_on_timeout=True
        )
        
        cleanup_called = []
        
        def cleanup_func():
            cleanup_called.append(True)
        
        with self.timeout_manager.timeout_context(operation_id, timeout_config, cleanup_func) as context:
            # Simulate long-running operation
            time.sleep(1.5)  # Wait longer than timeout
            
            # Should be cancelled after timeout
            self.assertTrue(self.timeout_manager.is_cancelled(operation_id))
        
        # Cleanup should be called on timeout
        self.assertEqual(len(cleanup_called), 1)
    
    def test_manual_cancellation(self):
        """Test manual operation cancellation."""
        operation_id = "test_op_manual_cancel"
        timeout_config = TimeoutConfig(
            query_timeout=60,  # Long timeout
            connection_timeout=5,
            database_name="test_db",
            database_type=DatabaseType.SQLITE
        )
        
        cleanup_called = []
        
        def cleanup_func():
            cleanup_called.append(True)
        
        with self.timeout_manager.timeout_context(operation_id, timeout_config, cleanup_func) as context:
            # Start operation
            self.assertFalse(self.timeout_manager.is_cancelled(operation_id))
            
            # Manually cancel it
            success = self.timeout_manager.cancel_operation(operation_id, TimeoutReason.MANUAL_CANCEL)
            self.assertTrue(success)
            
            # Should now be cancelled
            self.assertTrue(self.timeout_manager.is_cancelled(operation_id))
        
        # Cleanup should be called on manual cancellation
        self.assertEqual(len(cleanup_called), 1)
    
    def test_get_active_operations(self):
        """Test getting active operations information."""
        operation_id = "test_active_ops"
        timeout_config = TimeoutConfig(
            query_timeout=30,
            connection_timeout=5,
            database_name="test_db",
            database_type=DatabaseType.SQLITE
        )
        
        with self.timeout_manager.timeout_context(operation_id, timeout_config) as context:
            active_ops = self.timeout_manager.get_active_operations()
            
            self.assertIn(operation_id, active_ops)
            op_info = active_ops[operation_id]
            
            self.assertEqual(op_info['database_name'], 'test_db')
            self.assertEqual(op_info['database_type'], 'sqlite')
            self.assertEqual(op_info['timeout_limit'], 30)
            self.assertFalse(op_info['cancelled'])
            self.assertTrue(op_info['supports_cancellation'])
            self.assertGreater(op_info['time_remaining'], 0)
    
    def test_create_timeout_error(self):
        """Test timeout error creation."""
        operation_id = "test_error_creation"
        timeout_config = TimeoutConfig(
            query_timeout=30,
            connection_timeout=5,
            database_name="test_db",
            database_type=DatabaseType.SQLITE
        )
        
        with self.timeout_manager.timeout_context(operation_id, timeout_config):
            time.sleep(0.1)  # Small delay
            
            error = self.timeout_manager.create_timeout_error(
                operation_id, TimeoutReason.USER_TIMEOUT
            )
            
            self.assertIsInstance(error, QueryTimeoutError)
            self.assertEqual(error.timeout_reason, TimeoutReason.USER_TIMEOUT)
            self.assertEqual(error.database_name, "test_db")
            self.assertGreater(error.execution_time, 0)
            self.assertIn("Query timed out", error.message)


class TestTimeoutIntegration(unittest.TestCase):
    """Integration tests for timeout system with streaming executor."""
    
    def setUp(self):
        """Set up integration test environment."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create test data in SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create test table with enough data to test timeout
        cursor.execute("""
            CREATE TABLE test_data (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value INTEGER
            )
        """)
        
        # Insert test data
        test_data = [(i, f"name_{i}", i * 10) for i in range(1000)]
        cursor.executemany("INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)", test_data)
        
        conn.commit()
        conn.close()
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    @patch('src.localdata_mcp.config_manager.get_config_manager')
    def test_streaming_executor_with_timeout(self, mock_get_config):
        """Test streaming executor integration with timeout manager."""
        # Mock config manager
        mock_config_manager = Mock()
        test_db_config = DatabaseConfig(
            name="test_db",
            type=DatabaseType.SQLITE,
            connection_string=f"sqlite:///{self.db_path}",
            query_timeout=5,  # 5 second timeout
            connection_timeout=10
        )
        mock_config_manager.get_database_config.return_value = test_db_config
        mock_get_config.return_value = mock_config_manager
        
        # Import and test streaming executor
        from src.localdata_mcp.streaming_executor import StreamingQueryExecutor, create_streaming_source
        from sqlalchemy import create_engine
        
        # Create engine and streaming components
        engine = create_engine(f"sqlite:///{self.db_path}")
        query = "SELECT * FROM test_data ORDER BY id"
        
        # Create streaming source
        streaming_source = create_streaming_source(engine, query)
        
        # Create streaming executor
        executor = StreamingQueryExecutor()
        
        # Execute with timeout (should succeed as query is fast)
        query_id = "test_timeout_integration"
        try:
            first_chunk, metadata = executor.execute_streaming(
                streaming_source, 
                query_id, 
                initial_chunk_size=100,
                database_name="test_db"
            )
            
            # Verify timeout metadata is included
            self.assertIn("timeout_info", metadata)
            timeout_info = metadata["timeout_info"]
            self.assertTrue(timeout_info["timeout_configured"])
            self.assertEqual(timeout_info["timeout_limit_seconds"], 5)
            self.assertEqual(timeout_info["database_name"], "test_db")
            
            # Verify we got data
            self.assertFalse(first_chunk.empty)
            
        except Exception as e:
            self.fail(f"Streaming execution with timeout failed: {e}")
    
    def test_with_timeout_decorator(self):
        """Test the with_timeout decorator functionality."""
        
        @with_timeout("test_db", cleanup_func=lambda: None)
        def mock_database_operation():
            """Mock database operation for testing."""
            time.sleep(0.1)
            return "success"
        
        # Mock config manager to return short timeout
        with patch('src.localdata_mcp.timeout_manager.get_config_manager') as mock_get_config:
            mock_config_manager = Mock()
            test_db_config = DatabaseConfig(
                name="test_db",
                type=DatabaseType.SQLITE,
                connection_string="sqlite:///test.db",
                query_timeout=5,
                connection_timeout=10
            )
            mock_config_manager.get_database_config.return_value = test_db_config
            mock_get_config.return_value = mock_config_manager
            
            # Should execute successfully
            result = mock_database_operation()
            self.assertEqual(result, "success")


class TestTimeoutConfiguration(unittest.TestCase):
    """Test timeout configuration scenarios."""
    
    def test_per_database_timeout_configuration(self):
        """Test that different databases can have different timeout configurations."""
        with patch('src.localdata_mcp.timeout_manager.get_config_manager') as mock_get_config:
            mock_config_manager = Mock()
            
            # Configure different timeouts for different database types
            configs = {
                "local_sqlite": DatabaseConfig(
                    name="local_sqlite",
                    type=DatabaseType.SQLITE,
                    connection_string="sqlite:///local.db",
                    query_timeout=10,  # Short timeout for local
                    connection_timeout=5
                ),
                "remote_postgres": DatabaseConfig(
                    name="remote_postgres",
                    type=DatabaseType.POSTGRESQL,
                    connection_string="postgresql://user:pass@remote:5432/db",
                    query_timeout=300,  # Long timeout for remote
                    connection_timeout=60
                ),
                "csv_file": DatabaseConfig(
                    name="csv_file",
                    type=DatabaseType.CSV,
                    connection_string="/path/to/file.csv",
                    query_timeout=60,  # Medium timeout for file
                    connection_timeout=10
                )
            }
            
            mock_config_manager.get_database_config.side_effect = lambda name: configs.get(name)
            mock_get_config.return_value = mock_config_manager
            
            timeout_manager = QueryTimeoutManager()
            
            # Test local database - short timeout, supports cancellation
            local_config = timeout_manager.get_timeout_config("local_sqlite")
            self.assertEqual(local_config.query_timeout, 10)
            self.assertTrue(local_config.allow_cancellation)
            
            # Test remote database - long timeout, supports cancellation
            remote_config = timeout_manager.get_timeout_config("remote_postgres")
            self.assertEqual(remote_config.query_timeout, 300)
            self.assertTrue(remote_config.allow_cancellation)
            
            # Test file database - medium timeout, no cancellation support
            file_config = timeout_manager.get_timeout_config("csv_file")
            self.assertEqual(file_config.query_timeout, 60)
            self.assertFalse(file_config.allow_cancellation)


class TestTimeoutErrorScenarios(unittest.TestCase):
    """Test various timeout error scenarios."""
    
    def test_different_timeout_reasons(self):
        """Test different timeout reason messages."""
        timeout_manager = QueryTimeoutManager()
        
        test_cases = [
            (TimeoutReason.USER_TIMEOUT, "Query timed out"),
            (TimeoutReason.DATABASE_TIMEOUT, "Database-specific timeout exceeded"),
            (TimeoutReason.MEMORY_PRESSURE, "Query cancelled due to memory pressure"),
            (TimeoutReason.MANUAL_CANCEL, "Query cancelled (manual_cancel)")
        ]
        
        for reason, expected_text in test_cases:
            with self.subTest(reason=reason):
                error = QueryTimeoutError(
                    message=f"Test message for {reason.value}",
                    timeout_reason=reason,
                    execution_time=10.5,
                    database_name="test_db"
                )
                
                self.assertEqual(error.timeout_reason, reason)
                self.assertEqual(error.execution_time, 10.5)
                self.assertEqual(error.database_name, "test_db")
    
    def test_concurrent_operations(self):
        """Test timeout manager with multiple concurrent operations."""
        timeout_manager = QueryTimeoutManager()
        
        # Mock config
        with patch('src.localdata_mcp.timeout_manager.get_config_manager') as mock_get_config:
            mock_config_manager = Mock()
            test_db_config = DatabaseConfig(
                name="test_db",
                type=DatabaseType.SQLITE,
                connection_string="sqlite:///test.db",
                query_timeout=2,
                connection_timeout=5
            )
            mock_config_manager.get_database_config.return_value = test_db_config
            mock_get_config.return_value = mock_config_manager
            
            # Start multiple operations
            operation_ids = ["op1", "op2", "op3"]
            contexts = []
            
            timeout_config = timeout_manager.get_timeout_config("test_db")
            
            # Start all operations
            for op_id in operation_ids:
                context = timeout_manager.timeout_context(op_id, timeout_config)
                contexts.append((op_id, context.__enter__()))
            
            # Check all operations are active
            active_ops = timeout_manager.get_active_operations()
            for op_id in operation_ids:
                self.assertIn(op_id, active_ops)
            
            # Cancel one operation manually
            success = timeout_manager.cancel_operation("op2")
            self.assertTrue(success)
            
            # Wait for timeout on remaining operations
            time.sleep(2.5)
            
            # Check states
            self.assertFalse(timeout_manager.is_cancelled("op1"))  # Should be timed out
            self.assertTrue(timeout_manager.is_cancelled("op2"))   # Manually cancelled
            self.assertFalse(timeout_manager.is_cancelled("op3"))  # Should be timed out
            
            # Clean up contexts
            for op_id, context in contexts:
                try:
                    context.__exit__(None, None, None)
                except:
                    pass


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)