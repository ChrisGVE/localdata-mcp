"""Tests for Enhanced Connection Manager.

Test suite covering connection pooling, health monitoring, resource management,
and integration with configuration and timeout systems.
"""

import os
import pytest
import tempfile
import threading
import time
from unittest.mock import Mock, patch, MagicMock

from sqlalchemy import create_engine

from src.localdata_mcp.connection_manager import (
    EnhancedConnectionManager,
    ConnectionState,
    ResourceType,
    ConnectionMetrics,
    HealthCheckResult,
    ResourceLimit,
    get_enhanced_connection_manager
)
from src.localdata_mcp.config_manager import DatabaseConfig, DatabaseType


class TestEnhancedConnectionManager:
    """Test suite for EnhancedConnectionManager."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test databases
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test.db")
        
        # Create test database configuration
        self.test_config = DatabaseConfig(
            name="test_db",
            type=DatabaseType.SQLITE,
            connection_string=self.test_db_path,
            enabled=True,
            max_connections=5,
            connection_timeout=30,
            query_timeout=60,
            tags=["test", "sqlite"],
            metadata={"description": "Test database"}
        )
        
        # Mock configuration manager
        self.mock_config_manager = Mock()
        self.mock_config_manager.get_database_config.return_value = self.test_config
        self.mock_config_manager.get_database_configs.return_value = {"test_db": self.test_config}
        
        # Mock performance config
        self.mock_perf_config = Mock()
        self.mock_perf_config.memory_limit_mb = 1024
        self.mock_perf_config.memory_warning_threshold = 0.8
        self.mock_config_manager.get_performance_config.return_value = self.mock_perf_config

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_initialization(self, mock_timeout_mgr, mock_config_mgr):
        """Test connection manager initialization."""
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        
        assert manager._config_manager is not None
        assert manager._timeout_manager is not None
        assert isinstance(manager._engines, dict)
        assert isinstance(manager._metrics, dict)
        assert manager._monitoring_active is True

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_database_initialization(self, mock_timeout_mgr, mock_config_mgr):
        """Test database initialization with configuration."""
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        success = manager.initialize_database("test_db", self.test_config)
        
        assert success is True
        assert "test_db" in manager._engines
        assert "test_db" in manager._db_configs
        assert "test_db" in manager._metrics
        assert "test_db" in manager._health_status

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_engine_retrieval(self, mock_timeout_mgr, mock_config_mgr):
        """Test engine retrieval and lazy initialization."""
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        
        # Test non-existent database
        engine = manager.get_engine("nonexistent")
        assert engine is None
        
        # Test existing database
        manager.initialize_database("test_db", self.test_config)
        engine = manager.get_engine("test_db")
        assert engine is not None

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_connection_info(self, mock_timeout_mgr, mock_config_mgr):
        """Test connection information retrieval."""
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        manager.initialize_database("test_db", self.test_config)
        
        info = manager.get_connection_info("test_db")
        
        assert info is not None
        assert info["name"] == "test_db"
        assert info["type"] == "sqlite"
        assert info["enabled"] is True
        assert "test" in info["tags"]
        assert "connection_config" in info
        assert "metrics" in info
        assert "health" in info
        assert "resource_limits" in info

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_tag_filtering(self, mock_timeout_mgr, mock_config_mgr):
        """Test database filtering by tags."""
        # Create multiple database configs
        config2 = DatabaseConfig(
            name="test_db2",
            type=DatabaseType.SQLITE,
            connection_string=":memory:",
            tags=["production", "sqlite"]
        )
        
        configs = {"test_db": self.test_config, "test_db2": config2}
        self.mock_config_manager.get_database_configs.return_value = configs
        
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        
        # Get databases by tag
        test_dbs = manager.get_databases_by_tag("test")
        assert "test_db" in test_dbs
        assert "test_db2" not in test_dbs
        
        production_dbs = manager.get_databases_by_tag("production")
        assert "test_db2" in production_dbs
        assert "test_db" not in production_dbs

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_managed_query_execution(self, mock_timeout_mgr, mock_config_mgr):
        """Test managed query execution with resource tracking."""
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        manager.initialize_database("test_db", self.test_config)
        
        # Test successful query execution
        with manager.managed_query_execution("test_db", "test_query_1") as context:
            assert context["database_name"] == "test_db"
            assert context["query_id"] == "test_query_1"
            assert "start_time" in context
            
            # Simulate some work
            time.sleep(0.1)
        
        # Check metrics were updated
        metrics = manager._metrics["test_db"]
        assert metrics.total_queries == 1
        assert metrics.successful_queries == 1
        assert metrics.failed_queries == 0
        assert metrics.average_query_time > 0

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_query_failure_tracking(self, mock_timeout_mgr, mock_config_mgr):
        """Test query failure tracking and metrics."""
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        manager.initialize_database("test_db", self.test_config)
        
        # Test failed query execution
        with pytest.raises(ValueError):
            with manager.managed_query_execution("test_db", "test_query_fail") as context:
                raise ValueError("Simulated query failure")
        
        # Check metrics were updated
        metrics = manager._metrics["test_db"]
        assert metrics.total_queries == 1
        assert metrics.successful_queries == 0
        assert metrics.failed_queries == 1
        assert metrics.last_error == "Simulated query failure"

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_health_check(self, mock_timeout_mgr, mock_config_mgr):
        """Test database health checking."""
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        manager.initialize_database("test_db", self.test_config)
        
        # Trigger manual health check
        result = manager.trigger_health_check("test_db")
        
        assert result is not None
        assert isinstance(result, HealthCheckResult)
        assert result.response_time_ms >= 0
        assert result.state in ConnectionState

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_resource_limits(self, mock_timeout_mgr, mock_config_mgr):
        """Test resource limit enforcement."""
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        manager.initialize_database("test_db", self.test_config)
        
        # Get resource status
        status = manager.get_resource_status("test_db")
        
        assert ResourceType.MEMORY.value in status
        assert ResourceType.CONNECTIONS.value in status
        assert ResourceType.QUERY_TIME.value in status
        assert ResourceType.ERROR_RATE.value in status
        
        # Check resource limit structure
        memory_status = status[ResourceType.MEMORY.value]
        assert "current_value" in memory_status
        assert "max_value" in memory_status
        assert "warning_threshold" in memory_status
        assert "is_warning" in memory_status
        assert "is_exceeded" in memory_status

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_database_closure(self, mock_timeout_mgr, mock_config_mgr):
        """Test database connection closure and cleanup."""
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        manager.initialize_database("test_db", self.test_config)
        
        # Verify database is initialized
        assert "test_db" in manager._engines
        
        # Close database
        success = manager.close_database("test_db")
        assert success is True
        
        # Verify cleanup
        assert "test_db" not in manager._engines
        assert "test_db" not in manager._db_configs
        assert "test_db" not in manager._metrics

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_concurrent_access(self, mock_timeout_mgr, mock_config_mgr):
        """Test concurrent access to connection manager."""
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        manager.initialize_database("test_db", self.test_config)
        
        results = []
        errors = []
        
        def worker():
            try:
                with manager.managed_query_execution("test_db") as context:
                    time.sleep(0.05)  # Simulate work
                    results.append(context["query_id"])
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0
        assert len(results) == 5
        assert len(set(results)) == 5  # All unique query IDs

    def test_connection_metrics(self):
        """Test connection metrics calculations."""
        metrics = ConnectionMetrics()
        
        # Test initial state
        assert metrics.error_rate == 0.0
        assert metrics.success_rate == 100.0
        
        # Add some queries
        metrics.total_queries = 10
        metrics.successful_queries = 8
        metrics.failed_queries = 2
        
        assert metrics.error_rate == 20.0
        assert metrics.success_rate == 80.0

    def test_resource_limit_thresholds(self):
        """Test resource limit threshold checking."""
        limit = ResourceLimit(
            resource_type=ResourceType.MEMORY,
            max_value=1000.0,
            warning_threshold=800.0,
            current_value=750.0
        )
        
        assert not limit.is_warning
        assert not limit.is_exceeded
        
        # Set to warning level
        limit.current_value = 850.0
        assert limit.is_warning
        assert not limit.is_exceeded
        
        # Set to exceeded level
        limit.current_value = 1100.0
        assert limit.is_warning
        assert limit.is_exceeded

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_postgresql_engine_creation(self, mock_timeout_mgr, mock_config_mgr):
        """Test PostgreSQL engine creation with connection pooling."""
        pg_config = DatabaseConfig(
            name="postgres_db",
            type=DatabaseType.POSTGRESQL,
            connection_string="postgresql://user:pass@localhost/db",
            max_connections=20
        )
        
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        
        # This will fail to connect but should create the engine
        with patch('sqlalchemy.create_engine') as mock_create:
            mock_engine = Mock()
            mock_create.return_value = mock_engine
            
            engine = manager._create_enhanced_engine("postgres_db", pg_config)
            
            # Verify create_engine was called with pooling arguments
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert 'poolclass' in kwargs
            assert 'pool_size' in kwargs
            assert kwargs['pool_size'] == 10  # min(20, 10)

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_list_databases_with_tags(self, mock_timeout_mgr, mock_config_mgr):
        """Test listing databases with tag filtering."""
        # Create multiple database configs with different tags
        config1 = DatabaseConfig(name="db1", type=DatabaseType.SQLITE, 
                                connection_string=":memory:", tags=["test", "sqlite"])
        config2 = DatabaseConfig(name="db2", type=DatabaseType.SQLITE,
                                connection_string=":memory:", tags=["production", "sqlite"])
        config3 = DatabaseConfig(name="db3", type=DatabaseType.SQLITE,
                                connection_string=":memory:", tags=["test", "development"])
        
        configs = {"db1": config1, "db2": config2, "db3": config3}
        self.mock_config_manager.get_database_configs.return_value = configs
        
        mock_config_mgr.return_value = self.mock_config_manager
        mock_timeout_mgr.return_value = Mock()
        
        manager = EnhancedConnectionManager()
        
        # Test filtering by single tag
        test_dbs = manager.list_databases(include_tags=["test"])
        assert len(test_dbs) == 2  # db1 and db3
        
        prod_dbs = manager.list_databases(include_tags=["production"])
        assert len(prod_dbs) == 1  # db2 only
        
        # Test filtering by multiple tags (OR logic)
        mixed_dbs = manager.list_databases(include_tags=["test", "production"])
        assert len(mixed_dbs) == 3  # All databases


class TestGlobalConnectionManager:
    """Test global connection manager functions."""

    def test_singleton_behavior(self):
        """Test that global connection manager maintains singleton behavior."""
        # Clear any existing instance
        import src.localdata_mcp.connection_manager as cm_module
        cm_module._enhanced_connection_manager = None
        
        # Get two instances
        manager1 = get_enhanced_connection_manager()
        manager2 = get_enhanced_connection_manager()
        
        # Should be the same instance
        assert manager1 is manager2

    @patch('src.localdata_mcp.connection_manager.get_config_manager')
    @patch('src.localdata_mcp.connection_manager.get_timeout_manager')
    def test_manager_replacement(self, mock_timeout_mgr, mock_config_mgr):
        """Test replacing global connection manager."""
        from src.localdata_mcp.connection_manager import initialize_enhanced_connection_manager
        
        mock_config_mgr.return_value = Mock()
        mock_timeout_mgr.return_value = Mock()
        
        # Initialize first manager
        manager1 = initialize_enhanced_connection_manager()
        
        # Initialize second manager - should replace first
        manager2 = initialize_enhanced_connection_manager()
        
        assert manager1 is not manager2