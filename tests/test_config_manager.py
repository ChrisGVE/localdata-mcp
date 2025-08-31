"""Tests for configuration management system."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest
import yaml

from localdata_mcp.config_manager import (
    ConfigManager,
    DatabaseConfig,
    LoggingConfig,
    PerformanceConfig,
    DatabaseType,
    LogLevel,
    get_config_manager,
    initialize_config
)


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass validation."""
    
    def test_valid_database_config(self):
        """Test creation of valid database configuration."""
        config = DatabaseConfig(
            name="test_db",
            type=DatabaseType.POSTGRESQL,
            connection_string="postgresql://user:pass@localhost:5432/db"
        )
        assert config.name == "test_db"
        assert config.type == DatabaseType.POSTGRESQL
        assert config.enabled is True
        assert config.max_connections == 10
        assert config.connection_timeout == 30
        assert config.query_timeout == 300
        assert config.tags == []
        assert config.metadata == {}

    def test_invalid_max_connections(self):
        """Test validation of max_connections field."""
        with pytest.raises(ValueError, match="max_connections must be positive"):
            DatabaseConfig(
                name="test_db",
                type=DatabaseType.SQLITE,
                connection_string="/path/to/db.sqlite",
                max_connections=0
            )

    def test_invalid_timeouts(self):
        """Test validation of timeout fields."""
        with pytest.raises(ValueError, match="connection_timeout must be positive"):
            DatabaseConfig(
                name="test_db",
                type=DatabaseType.SQLITE,
                connection_string="/path/to/db.sqlite",
                connection_timeout=-1
            )
        
        with pytest.raises(ValueError, match="query_timeout must be positive"):
            DatabaseConfig(
                name="test_db",
                type=DatabaseType.SQLITE,
                connection_string="/path/to/db.sqlite",
                query_timeout=0
            )


class TestLoggingConfig:
    """Test LoggingConfig dataclass validation."""
    
    def test_valid_logging_config(self):
        """Test creation of valid logging configuration."""
        config = LoggingConfig(
            level=LogLevel.DEBUG,
            file_path="/var/log/localdata.log"
        )
        assert config.level == LogLevel.DEBUG
        assert config.file_path == "/var/log/localdata.log"
        assert config.console_output is True
        assert config.max_file_size == 10 * 1024 * 1024
        assert config.backup_count == 5

    def test_invalid_file_size(self):
        """Test validation of max_file_size field."""
        with pytest.raises(ValueError, match="max_file_size must be positive"):
            LoggingConfig(max_file_size=-1)

    def test_invalid_backup_count(self):
        """Test validation of backup_count field."""
        with pytest.raises(ValueError, match="backup_count must be non-negative"):
            LoggingConfig(backup_count=-1)


class TestPerformanceConfig:
    """Test PerformanceConfig dataclass validation."""
    
    def test_valid_performance_config(self):
        """Test creation of valid performance configuration."""
        config = PerformanceConfig(
            memory_limit_mb=4096,
            chunk_size=200,
            memory_warning_threshold=0.9
        )
        assert config.memory_limit_mb == 4096
        assert config.chunk_size == 200
        assert config.memory_warning_threshold == 0.9

    def test_invalid_memory_limit(self):
        """Test validation of memory_limit_mb field."""
        with pytest.raises(ValueError, match="memory_limit_mb must be positive"):
            PerformanceConfig(memory_limit_mb=0)

    def test_invalid_chunk_size(self):
        """Test validation of chunk_size field."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            PerformanceConfig(chunk_size=-1)

    def test_invalid_memory_threshold(self):
        """Test validation of memory_warning_threshold field."""
        with pytest.raises(ValueError, match="memory_warning_threshold must be between 0 and 1"):
            PerformanceConfig(memory_warning_threshold=1.5)


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear environment variables
        self.old_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith('LOCALDATA_') or key.startswith('POSTGRES_') or key.startswith('MYSQL_'):
                del os.environ[key]

    def teardown_method(self):
        """Clean up test environment."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.old_env)

    def test_default_configuration(self):
        """Test default configuration values."""
        config_manager = ConfigManager()
        
        # Test default logging config
        logging_config = config_manager.get_logging_config()
        assert logging_config.level == LogLevel.INFO
        assert logging_config.console_output is True
        
        # Test default performance config
        perf_config = config_manager.get_performance_config()
        assert perf_config.memory_limit_mb == 2048
        assert perf_config.query_buffer_timeout == 600
        assert perf_config.max_concurrent_connections == 10
        assert perf_config.chunk_size == 100
        assert perf_config.enable_query_analysis is True
        
        # Test default database configs (should be empty)
        db_configs = config_manager.get_database_configs()
        assert db_configs == {}

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        # Set legacy environment variables
        os.environ['POSTGRES_HOST'] = 'localhost'
        os.environ['POSTGRES_PORT'] = '5432'
        os.environ['POSTGRES_USER'] = 'testuser'
        os.environ['POSTGRES_PASSWORD'] = 'testpass'
        os.environ['POSTGRES_DB'] = 'testdb'
        
        # Set logging environment variables
        os.environ['LOCALDATA_LOG_LEVEL'] = 'debug'
        os.environ['LOCALDATA_LOG_FILE'] = '/tmp/test.log'
        
        # Set performance environment variables
        os.environ['LOCALDATA_MEMORY_LIMIT_MB'] = '4096'
        
        config_manager = ConfigManager()
        
        # Test database configuration from legacy env vars
        db_configs = config_manager.get_database_configs()
        assert 'default_postgresql' in db_configs
        
        pg_config = db_configs['default_postgresql']
        assert pg_config.type == DatabaseType.POSTGRESQL
        assert 'testuser:testpass@localhost:5432/testdb' in pg_config.connection_string
        
        # Test logging configuration
        logging_config = config_manager.get_logging_config()
        assert logging_config.level == LogLevel.DEBUG
        assert logging_config.file_path == '/tmp/test.log'
        
        # Test performance configuration
        perf_config = config_manager.get_performance_config()
        assert perf_config.memory_limit_mb == 4096

    def test_modern_environment_variables(self):
        """Test loading configuration from modern prefixed environment variables."""
        # Set modern database environment variables
        os.environ['LOCALDATA_DB_MYDB_TYPE'] = 'sqlite'
        os.environ['LOCALDATA_DB_MYDB_CONNECTION_STRING'] = '/path/to/test.db'
        os.environ['LOCALDATA_DB_MYDB_ENABLED'] = 'true'
        os.environ['LOCALDATA_DB_MYDB_MAX_CONNECTIONS'] = '5'
        
        config_manager = ConfigManager()
        
        # Test database configuration from modern env vars
        db_configs = config_manager.get_database_configs()
        assert 'mydb' in db_configs
        
        db_config = db_configs['mydb']
        assert db_config.type == DatabaseType.SQLITE
        assert db_config.connection_string == '/path/to/test.db'
        assert db_config.enabled is True
        assert db_config.max_connections == 5

    def test_yaml_configuration_loading(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
        databases:
          analytics_db:
            type: postgresql
            connection_string: postgresql://user:pass@localhost:5432/analytics
            enabled: true
            max_connections: 15
            tags: ["analytics", "production"]
          cache_db:
            type: redis
            connection_string: redis://localhost:6379/0
            enabled: true
            
        logging:
          level: warning
          file_path: /var/log/localdata.log
          console_output: false
          
        performance:
          memory_limit_mb: 4096
          chunk_size: 200
          enable_query_analysis: false
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            
            # Test database configurations
            db_configs = config_manager.get_database_configs()
            assert len(db_configs) == 2
            
            # Test analytics_db
            assert 'analytics_db' in db_configs
            analytics_config = db_configs['analytics_db']
            assert analytics_config.type == DatabaseType.POSTGRESQL
            assert analytics_config.max_connections == 15
            assert analytics_config.tags == ["analytics", "production"]
            
            # Test cache_db
            assert 'cache_db' in db_configs
            cache_config = db_configs['cache_db']
            assert cache_config.type == DatabaseType.REDIS
            
            # Test logging configuration
            logging_config = config_manager.get_logging_config()
            assert logging_config.level == LogLevel.WARNING
            assert logging_config.file_path == "/var/log/localdata.log"
            assert logging_config.console_output is False
            
            # Test performance configuration
            perf_config = config_manager.get_performance_config()
            assert perf_config.memory_limit_mb == 4096
            assert perf_config.chunk_size == 200
            assert perf_config.enable_query_analysis is False
            
        finally:
            os.unlink(config_file)

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in YAML files."""
        # Set environment variables for substitution
        os.environ['DB_HOST'] = 'prod-server'
        os.environ['DB_PORT'] = '5432'
        os.environ['DB_NAME'] = 'production_db'
        os.environ['LOG_LEVEL'] = 'error'
        
        yaml_content = """
        databases:
          prod_db:
            type: postgresql
            connection_string: postgresql://user:pass@${DB_HOST}:${DB_PORT}/${DB_NAME}
            
        logging:
          level: ${LOG_LEVEL}
          file_path: ${LOG_PATH:/tmp/default.log}
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            
            # Test database configuration with substitution
            db_configs = config_manager.get_database_configs()
            prod_config = db_configs['prod_db']
            assert 'prod-server:5432/production_db' in prod_config.connection_string
            
            # Test logging configuration with substitution
            logging_config = config_manager.get_logging_config()
            assert logging_config.level == LogLevel.ERROR
            # Test default value substitution (LOG_PATH not set, should use default)
            assert logging_config.file_path == "/tmp/default.log"
            
        finally:
            os.unlink(config_file)

    def test_configuration_precedence(self):
        """Test configuration precedence: env vars > YAML > defaults."""
        # Create YAML file with base configuration
        yaml_content = """
        databases:
          test_db:
            type: sqlite
            connection_string: /yaml/path/test.db
            max_connections: 5
            
        logging:
          level: info
          
        performance:
          memory_limit_mb: 1024
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name
        
        try:
            # Set environment variables that should override YAML
            os.environ['LOCALDATA_DB_TEST_DB_CONNECTION_STRING'] = '/env/path/test.db'
            os.environ['LOCALDATA_DB_TEST_DB_MAX_CONNECTIONS'] = '10'
            os.environ['LOCALDATA_LOG_LEVEL'] = 'debug'
            os.environ['LOCALDATA_MEMORY_LIMIT_MB'] = '2048'
            
            config_manager = ConfigManager(config_file=config_file)
            
            # Test that environment variables override YAML
            db_configs = config_manager.get_database_configs()
            test_config = db_configs['test_db']
            assert test_config.connection_string == '/env/path/test.db'  # From env
            assert test_config.max_connections == 10  # From env
            assert test_config.type == DatabaseType.SQLITE  # From YAML (not overridden)
            
            # Test logging override
            logging_config = config_manager.get_logging_config()
            assert logging_config.level == LogLevel.DEBUG  # From env
            
            # Test performance override
            perf_config = config_manager.get_performance_config()
            assert perf_config.memory_limit_mb == 2048  # From env
            
        finally:
            os.unlink(config_file)

    def test_config_file_discovery(self):
        """Test configuration file discovery in default locations."""
        yaml_content = """
        databases:
          discovered_db:
            type: sqlite
            connection_string: /discovered/test.db
        """
        
        # Test current directory discovery
        current_dir_config = Path('./localdata.yaml')
        try:
            with open(current_dir_config, 'w') as f:
                f.write(yaml_content)
            
            config_manager = ConfigManager()  # Should discover ./localdata.yaml
            
            db_configs = config_manager.get_database_configs()
            assert 'discovered_db' in db_configs
            discovered_config = db_configs['discovered_db']
            assert discovered_config.connection_string == '/discovered/test.db'
            
        finally:
            if current_dir_config.exists():
                current_dir_config.unlink()

    @patch('os.path.getmtime')
    def test_hot_reload_capability(self, mock_getmtime):
        """Test hot-reload capability for configuration changes."""
        yaml_content = """
        databases:
          reload_db:
            type: sqlite
            connection_string: /original/test.db
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name
        
        try:
            # Set initial modification time
            initial_mtime = 1000.0
            mock_getmtime.return_value = initial_mtime
            
            config_manager = ConfigManager(config_file=config_file)
            
            # Test initial configuration
            db_configs = config_manager.get_database_configs()
            reload_config = db_configs['reload_db']
            assert reload_config.connection_string == '/original/test.db'
            
            # Simulate file modification
            mock_getmtime.return_value = initial_mtime + 100.0  # File modified
            
            # Test change detection
            assert config_manager.has_config_changed() is True
            
            # Update file content and reload
            updated_content = """
            databases:
              reload_db:
                type: sqlite
                connection_string: /updated/test.db
            """
            
            with open(config_file, 'w') as f:
                f.write(updated_content)
            
            config_manager.reload_config()
            
            # Test updated configuration
            db_configs = config_manager.get_database_configs()
            reload_config = db_configs['reload_db']
            assert reload_config.connection_string == '/updated/test.db'
            
        finally:
            os.unlink(config_file)

    def test_invalid_database_type_validation(self):
        """Test validation of invalid database types."""
        yaml_content = """
        databases:
          invalid_db:
            type: unsupported_type
            connection_string: /test/path
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name
        
        try:
            # Should not raise exception but log warning
            config_manager = ConfigManager(config_file=config_file)
            
            # Invalid database should not be included in configs
            db_configs = config_manager.get_database_configs()
            assert 'invalid_db' not in db_configs
            
        finally:
            os.unlink(config_file)

    def test_missing_required_fields_validation(self):
        """Test validation when required fields are missing."""
        yaml_content = """
        databases:
          incomplete_db:
            type: sqlite
            # Missing connection_string
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name
        
        try:
            # Should not raise exception but log warning
            config_manager = ConfigManager(config_file=config_file)
            
            # Incomplete database should not be included in configs
            db_configs = config_manager.get_database_configs()
            assert 'incomplete_db' not in db_configs
            
        finally:
            os.unlink(config_file)

    def test_multi_database_configuration(self):
        """Test loading multiple database configurations with different types."""
        yaml_content = """
        databases:
          primary_pg:
            type: postgresql
            connection_string: postgresql://user:pass@localhost:5432/primary
            tags: ["primary", "production"]
            metadata:
              region: us-east-1
              backup_schedule: "daily"
              
          cache_redis:
            type: redis
            connection_string: redis://localhost:6379/0
            tags: ["cache", "memory"]
            max_connections: 20
            
          analytics_duck:
            type: duckdb
            connection_string: /data/analytics.duckdb
            tags: ["analytics", "reporting"]
            enabled: false
            
          files_csv:
            type: csv
            connection_string: /data/exports/data.csv
            tags: ["file", "export"]
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file=config_file)
            
            db_configs = config_manager.get_database_configs()
            assert len(db_configs) == 4
            
            # Test PostgreSQL config
            pg_config = db_configs['primary_pg']
            assert pg_config.type == DatabaseType.POSTGRESQL
            assert pg_config.tags == ["primary", "production"]
            assert pg_config.metadata["region"] == "us-east-1"
            assert pg_config.metadata["backup_schedule"] == "daily"
            
            # Test Redis config
            redis_config = db_configs['cache_redis']
            assert redis_config.type == DatabaseType.REDIS
            assert redis_config.max_connections == 20
            assert redis_config.tags == ["cache", "memory"]
            
            # Test DuckDB config (disabled)
            duck_config = db_configs['analytics_duck']
            assert duck_config.type == DatabaseType.DUCKDB
            assert duck_config.enabled is False
            assert duck_config.tags == ["analytics", "reporting"]
            
            # Test CSV config
            csv_config = db_configs['files_csv']
            assert csv_config.type == DatabaseType.CSV
            assert csv_config.tags == ["file", "export"]
            
        finally:
            os.unlink(config_file)


class TestGlobalConfigManager:
    """Test global configuration manager functions."""
    
    def test_get_config_manager_singleton(self):
        """Test that get_config_manager returns singleton instance."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2

    def test_initialize_config_override(self):
        """Test that initialize_config creates new instance."""
        original_manager = get_config_manager()
        
        yaml_content = """
        databases:
          test_init:
            type: sqlite
            connection_string: /init/test.db
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_file = f.name
        
        try:
            new_manager = initialize_config(config_file=config_file, auto_reload=True)
            
            assert new_manager is not original_manager
            assert get_config_manager() is new_manager
            
            # Test that new configuration is loaded
            db_configs = new_manager.get_database_configs()
            assert 'test_init' in db_configs
            
        finally:
            os.unlink(config_file)