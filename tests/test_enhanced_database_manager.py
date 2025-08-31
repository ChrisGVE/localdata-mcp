"""Tests for the enhanced database manager with integrated error handling."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call

from src.localdata_mcp.enhanced_database_manager import (
    EnhancedDatabaseManager, get_enhanced_database_manager, initialize_enhanced_database_manager
)
from src.localdata_mcp.error_handler import (
    DatabaseConnectionError, QueryExecutionError, SecurityViolationError,
    QueryTimeoutError, CircuitState
)


class TestEnhancedDatabaseManager:
    """Test enhanced database manager functionality."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        with patch('src.localdata_mcp.enhanced_database_manager.get_error_handler') as mock_error_handler, \
             patch('src.localdata_mcp.enhanced_database_manager.get_security_manager') as mock_security, \
             patch('src.localdata_mcp.enhanced_database_manager.get_enhanced_connection_manager') as mock_conn, \
             patch('src.localdata_mcp.enhanced_database_manager.get_timeout_manager') as mock_timeout, \
             patch('src.localdata_mcp.enhanced_database_manager.StreamingQueryExecutor') as mock_streaming:
            
            # Setup mock returns
            mock_error_handler_instance = Mock()
            mock_error_handler.return_value = mock_error_handler_instance
            
            mock_security_instance = Mock()
            mock_security.return_value = mock_security_instance
            
            mock_conn_instance = Mock()
            mock_conn.return_value = mock_conn_instance
            
            mock_timeout_instance = Mock()
            mock_timeout.return_value = mock_timeout_instance
            
            mock_streaming_instance = Mock()
            mock_streaming.return_value = mock_streaming_instance
            
            # Setup circuit breaker mocks
            mock_circuit_breaker = Mock()
            mock_circuit_breaker.is_request_allowed.return_value = True
            mock_circuit_breaker.get_state_info.return_value = {
                'name': 'test_breaker',
                'state': 'closed',
                'stats': {'total_requests': 0, 'failed_requests': 0}
            }
            mock_error_handler_instance.get_circuit_breaker.return_value = mock_circuit_breaker
            
            yield {
                'error_handler': mock_error_handler_instance,
                'security_manager': mock_security_instance,
                'connection_manager': mock_conn_instance,
                'timeout_manager': mock_timeout_instance,
                'streaming_executor': mock_streaming_instance,
                'circuit_breaker': mock_circuit_breaker
            }
    
    def test_initialization(self, mock_components):
        """Test enhanced database manager initialization."""
        manager = EnhancedDatabaseManager()
        
        assert manager.error_handler is not None
        assert manager.security_manager is not None
        assert manager.connection_manager is not None
        assert manager.timeout_manager is not None
        assert manager.streaming_executor is not None
        
        # Verify circuit breakers were created
        error_handler = mock_components['error_handler']
        assert error_handler.get_circuit_breaker.call_count >= 3  # connection, query, security
    
    def test_enhanced_query_execution_success(self, mock_components):
        """Test successful query execution with error handling."""
        manager = EnhancedDatabaseManager()
        
        # Setup mocks
        security_manager = mock_components['security_manager']
        security_manager.validate_query_security.return_value = (True, None, {'validated': True})
        
        connection_manager = mock_components['connection_manager']
        mock_engine = Mock()
        connection_manager.get_engine.return_value = mock_engine
        
        # Test the context manager
        with manager.enhanced_query_execution("test_db", "SELECT 1", "conn_123") as context:
            assert context['engine'] == mock_engine
            assert context['database_name'] == "test_db"
            assert context['query'] == "SELECT 1"
            assert context['connection_id'] == "conn_123"
            assert 'security_metadata' in context
        
        # Verify security validation was called
        security_manager.validate_query_security.assert_called_once_with(
            "SELECT 1", "test_db", "conn_123"
        )
        
        # Verify connection was obtained
        connection_manager.get_engine.assert_called_once_with("test_db")
    
    def test_enhanced_query_execution_security_failure(self, mock_components):
        """Test query execution with security validation failure."""
        manager = EnhancedDatabaseManager()
        
        # Setup security validation to fail
        security_manager = mock_components['security_manager']
        security_manager.validate_query_security.return_value = (
            False, "SQL injection detected", {'threat_level': 'high'}
        )
        
        # Should raise SecurityViolationError
        with pytest.raises(SecurityViolationError) as exc_info:
            with manager.enhanced_query_execution("test_db", "SELECT * FROM users; DROP TABLE users;"):
                pass
        
        assert "SQL injection detected" in str(exc_info.value)
        assert exc_info.value.database_name == "test_db"
    
    def test_enhanced_query_execution_connection_failure(self, mock_components):
        """Test query execution with connection failure."""
        manager = EnhancedDatabaseManager()
        
        # Setup mocks
        security_manager = mock_components['security_manager']
        security_manager.validate_query_security.return_value = (True, None, {})
        
        connection_manager = mock_components['connection_manager']
        connection_manager.get_engine.return_value = None  # Simulate connection failure
        
        # Should raise DatabaseConnectionError
        with pytest.raises(DatabaseConnectionError) as exc_info:
            with manager.enhanced_query_execution("test_db", "SELECT 1"):
                pass
        
        assert "Unable to get database engine" in str(exc_info.value)
        assert exc_info.value.database_name == "test_db"
    
    def test_enhanced_query_execution_with_error_recovery(self, mock_components):
        """Test query execution with error recovery."""
        manager = EnhancedDatabaseManager()
        
        # Setup security validation to succeed
        security_manager = mock_components['security_manager']
        security_manager.validate_query_security.return_value = (True, None, {})
        
        # Setup connection to succeed
        connection_manager = mock_components['connection_manager']
        mock_engine = Mock()
        connection_manager.get_engine.return_value = mock_engine
        
        # Setup error handler to recover from errors
        error_handler = mock_components['error_handler']
        error_handler.handle_error.return_value = (True, "recovered_result", None)
        
        # Create a mock that raises an error then recovers
        def mock_context_manager():
            class MockContext:
                def __enter__(self):
                    raise QueryExecutionError("Temporary failure")
                def __exit__(self, *args):
                    pass
            return MockContext()
        
        # Patch the timeout manager to raise an error
        timeout_manager = mock_components['timeout_manager']
        timeout_manager.managed_timeout = mock_context_manager
        
        # Should handle the error and potentially recover
        with manager.enhanced_query_execution("test_db", "SELECT 1") as context:
            # If recovery succeeds, context should contain recovery result
            if 'recovery_result' in context:
                assert context['recovery_result'] == "recovered_result"
    
    @patch('pandas.DataFrame')
    @patch('src.localdata_mcp.enhanced_database_manager.create_streaming_source')
    def test_execute_query_with_error_handling_success(self, mock_create_source, mock_dataframe, mock_components):
        """Test successful query execution with comprehensive error handling."""
        manager = EnhancedDatabaseManager()
        
        # Setup mocks
        security_manager = mock_components['security_manager']
        security_manager.validate_query_security.return_value = (True, None, {'validated': True})
        
        connection_manager = mock_components['connection_manager']
        mock_engine = Mock()
        connection_manager.get_engine.return_value = mock_engine
        
        timeout_manager = mock_components['timeout_manager']
        timeout_manager.get_query_timeout.return_value = 30.0
        
        # Setup streaming executor
        streaming_executor = mock_components['streaming_executor']
        mock_dataframe_instance = Mock()
        mock_dataframe_instance.empty = False
        mock_dataframe_instance.to_dict.return_value = [{'col1': 'value1', 'col2': 'value2'}]
        
        streaming_executor.execute_streaming.return_value = (
            mock_dataframe_instance,
            {'has_more_data': False, 'total_rows': 1}
        )
        
        # Setup timeout context manager
        timeout_manager.managed_timeout.return_value.__enter__ = Mock()
        timeout_manager.managed_timeout.return_value.__exit__ = Mock(return_value=False)
        
        # Execute query
        result = manager.execute_query_with_error_handling(
            database_name="test_db",
            query="SELECT col1, col2 FROM test_table",
            connection_id="conn_123"
        )
        
        # Verify successful response
        assert result['success'] is True
        assert 'data' in result
        assert 'metadata' in result
        assert result['metadata']['database_name'] == "test_db"
        assert 'error_handling' in result['metadata']
        
        # Verify security validation was called
        security_manager.validate_query_security.assert_called()
        
        # Verify streaming execution was called
        streaming_executor.execute_streaming.assert_called()
    
    def test_execute_query_with_error_handling_failure(self, mock_components):
        """Test query execution failure with error handling."""
        manager = EnhancedDatabaseManager()
        
        # Setup security validation to fail
        security_manager = mock_components['security_manager']
        security_manager.validate_query_security.side_effect = SecurityViolationError(
            "SQL injection detected",
            query="SELECT * FROM users; DROP TABLE users;"
        )
        
        # Setup error handler
        error_handler = mock_components['error_handler']
        error_handler.handle_error.return_value = (False, None, SecurityViolationError("SQL injection detected"))
        
        # Execute query
        result = manager.execute_query_with_error_handling(
            database_name="test_db",
            query="SELECT * FROM users; DROP TABLE users;"
        )
        
        # Verify error response
        assert result['success'] is False
        assert 'error' in result
        assert result['error']['category'] == 'security_violation'
        assert "SQL injection detected" in result['error']['message']
        assert result['metadata']['database_name'] == "test_db"
    
    def test_connect_database_with_error_handling_success(self, mock_components):
        """Test successful database connection with error handling."""
        manager = EnhancedDatabaseManager()
        
        # Setup connection manager
        connection_manager = mock_components['connection_manager']
        connection_manager.initialize_database.return_value = True
        connection_manager.get_connection_info.return_value = {
            'name': 'test_db',
            'type': 'postgresql',
            'health': {'is_healthy': True}
        }
        
        # Connect database
        result = manager.connect_database_with_error_handling(
            name="test_db",
            db_type="postgresql", 
            conn_string="postgresql://user:pass@localhost/db"
        )
        
        # Verify successful response
        assert result['success'] is True
        assert 'connection_info' in result
        assert result['connection_info']['name'] == 'test_db'
        assert result['metadata']['database_name'] == 'test_db'
        
        # Verify connection manager was called
        connection_manager.initialize_database.assert_called_once_with("test_db")
        connection_manager.get_connection_info.assert_called_once_with("test_db")
    
    def test_connect_database_with_error_handling_failure(self, mock_components):
        """Test database connection failure with error handling."""
        manager = EnhancedDatabaseManager()
        
        # Setup connection manager to fail
        connection_manager = mock_components['connection_manager']
        connection_manager.initialize_database.return_value = False
        
        # Setup error handler
        error_handler = mock_components['error_handler']
        mock_error = DatabaseConnectionError("Failed to initialize database connection 'test_db'")
        error_handler.handle_error.return_value = (False, None, mock_error)
        
        # Connect database
        result = manager.connect_database_with_error_handling(
            name="test_db",
            db_type="postgresql",
            conn_string="invalid_connection_string"
        )
        
        # Verify error response
        assert result['success'] is False
        assert 'error' in result
        assert result['error']['category'] == 'connection'
        assert "Failed to initialize" in result['error']['message']
    
    def test_get_health_status(self, mock_components):
        """Test health status reporting."""
        manager = EnhancedDatabaseManager()
        
        # Setup mock health data
        error_handler = mock_components['error_handler']
        error_handler.get_system_health.return_value = {
            'overall_health': {'health_score': 85.0, 'status': 'healthy'},
            'error_statistics': {'total_errors': 5}
        }
        
        connection_manager = mock_components['connection_manager'] 
        connection_manager.list_databases.return_value = [
            {'name': 'db1', 'health': {'is_healthy': True}},
            {'name': 'db2', 'health': {'is_healthy': False}}
        ]
        
        # Get health status
        health = manager.get_health_status()
        
        # Verify health response structure
        assert 'overall_status' in health
        assert 'overall_health_score' in health
        assert 'components' in health
        assert 'recommendations' in health
        assert 'timestamp' in health
        
        # Verify components are included
        components = health['components']
        assert 'error_handler' in components
        assert 'circuit_breakers' in components
        assert 'connection_manager' in components
        
        # Verify connection manager health
        conn_health = components['connection_manager']
        assert conn_health['total_databases'] == 2
        assert conn_health['healthy_connections'] == 1
        assert conn_health['health_rate'] == 50.0
    
    def test_circuit_breaker_integration(self, mock_components):
        """Test circuit breaker integration in operations."""
        manager = EnhancedDatabaseManager()
        
        # Get circuit breaker and simulate it being open
        circuit_breaker = mock_components['circuit_breaker']
        circuit_breaker.is_request_allowed.return_value = False
        circuit_breaker.get_state_info.return_value = {
            'name': 'test_breaker',
            'state': 'open',
            'stats': {'failed_requests': 10}
        }
        
        # Setup mocks that wouldn't normally be reached due to circuit breaker
        security_manager = mock_components['security_manager']
        security_manager.validate_query_security.return_value = (True, None, {})
        
        # Attempt query execution - should fail due to open circuit breaker
        with pytest.raises(DatabaseConnectionError) as exc_info:
            with manager.enhanced_query_execution("test_db", "SELECT 1"):
                pass
        
        assert "Circuit breaker" in str(exc_info.value)
        assert "is open" in str(exc_info.value)
        
        # Security validation should not have been called due to circuit breaker
        security_manager.validate_query_security.assert_not_called()
    
    def test_reset_circuit_breakers(self, mock_components):
        """Test circuit breaker reset functionality."""
        manager = EnhancedDatabaseManager()
        
        # Reset circuit breakers
        result = manager.reset_circuit_breakers()
        
        # Verify successful reset
        assert result['success'] is True
        assert "reset" in result['message'].lower()
        assert 'circuit_breakers' in result
        
        # Verify all circuit breakers were reset
        circuit_breaker = mock_components['circuit_breaker']
        assert circuit_breaker.reset.call_count == 3  # connection, query, security
    
    def test_global_enhanced_database_manager_functions(self, mock_components):
        """Test global enhanced database manager accessor functions."""
        # Get global manager
        manager1 = get_enhanced_database_manager()
        manager2 = get_enhanced_database_manager()
        
        # Should return same instance
        assert manager1 is manager2
        
        # Initialize new manager
        manager3 = initialize_enhanced_database_manager()
        manager4 = get_enhanced_database_manager()
        
        # Should now return the new instance
        assert manager3 is manager4
        assert manager3 is not manager1


class TestEnhancedDatabaseManagerIntegration:
    """Integration tests with more realistic scenarios."""
    
    @pytest.fixture
    def realistic_mocks(self):
        """Create more realistic mocks for integration testing."""
        with patch.multiple(
            'src.localdata_mcp.enhanced_database_manager',
            get_error_handler=Mock(),
            get_security_manager=Mock(),
            get_enhanced_connection_manager=Mock(),
            get_timeout_manager=Mock(),
            StreamingQueryExecutor=Mock()
        ) as mocks:
            
            # Setup realistic error handler
            error_handler = Mock()
            error_handler.get_system_health.return_value = {
                'overall_health': {'health_score': 95.0, 'status': 'healthy'},
                'circuit_breakers': {'total_breakers': 3, 'states': {'closed': 3}},
                'error_statistics': {'total_errors': 2},
                'recovery_statistics': {'success_rate': 80.0}
            }
            
            # Setup circuit breakers
            circuit_breaker = Mock()
            circuit_breaker.is_request_allowed.return_value = True
            circuit_breaker.get_state_info.return_value = {
                'name': 'test_breaker',
                'state': 'closed',
                'stats': {'total_requests': 100, 'successful_requests': 95, 'failed_requests': 5}
            }
            error_handler.get_circuit_breaker.return_value = circuit_breaker
            
            mocks['get_error_handler'].return_value = error_handler
            
            yield mocks
    
    def test_complete_query_execution_flow(self, realistic_mocks):
        """Test complete query execution flow with realistic components."""
        manager = EnhancedDatabaseManager()
        
        # Setup complete flow mocks
        security_manager = realistic_mocks['get_security_manager'].return_value
        security_manager.validate_query_security.return_value = (
            True, None, {'validated': True, 'complexity': 'low'}
        )
        
        connection_manager = realistic_mocks['get_enhanced_connection_manager'].return_value
        mock_engine = Mock()
        connection_manager.get_engine.return_value = mock_engine
        
        timeout_manager = realistic_mocks['get_timeout_manager'].return_value
        timeout_manager.get_query_timeout.return_value = 30.0
        timeout_manager.managed_timeout.return_value.__enter__ = Mock()
        timeout_manager.managed_timeout.return_value.__exit__ = Mock(return_value=False)
        
        streaming_executor = realistic_mocks['StreamingQueryExecutor'].return_value
        mock_df = Mock()
        mock_df.empty = False
        mock_df.to_dict.return_value = [
            {'id': 1, 'name': 'Alice'},
            {'id': 2, 'name': 'Bob'}
        ]
        streaming_executor.execute_streaming.return_value = (
            mock_df,
            {'has_more_data': False, 'total_rows': 2, 'execution_time': 0.5}
        )
        
        # Execute query
        result = manager.execute_query_with_error_handling(
            database_name="production_db",
            query="SELECT id, name FROM users LIMIT 10",
            connection_id="web_app_001",
            enable_analysis=True,
            chunk_size=100
        )
        
        # Verify complete successful flow
        assert result['success'] is True
        assert len(result['data']) == 2
        assert result['data'][0]['name'] == 'Alice'
        assert result['metadata']['database_name'] == 'production_db'
        assert result['metadata']['rows_returned'] == 2
        
        # Verify all components were called appropriately
        security_manager.validate_query_security.assert_called_once()
        connection_manager.get_engine.assert_called_once_with('production_db')
        timeout_manager.managed_timeout.assert_called_once()
        streaming_executor.execute_streaming.assert_called_once()
    
    def test_error_cascade_and_recovery(self, realistic_mocks):
        """Test error cascading through components and recovery attempts."""
        manager = EnhancedDatabaseManager()
        
        # Setup initial failure in connection
        connection_manager = realistic_mocks['get_enhanced_connection_manager'].return_value
        connection_manager.get_engine.side_effect = [
            None,  # First attempt fails
            Mock()  # Second attempt (after recovery) succeeds
        ]
        
        # Setup error handler for recovery
        error_handler = realistic_mocks['get_error_handler'].return_value
        
        # First call handles the connection error
        error_handler.handle_error.side_effect = [
            (False, None, DatabaseConnectionError("Connection failed")),  # Recovery fails
            (True, "connection_recovered", None)  # Second recovery succeeds
        ]
        
        # Execute query - should handle the connection failure
        result = manager.execute_query_with_error_handling(
            database_name="flaky_db",
            query="SELECT COUNT(*) FROM metrics"
        )
        
        # Should result in error since recovery failed
        assert result['success'] is False
        assert result['error']['category'] == 'connection'
        
        # Verify error handler was called
        error_handler.handle_error.assert_called()
    
    def test_health_monitoring_with_degraded_components(self, realistic_mocks):
        """Test health monitoring when components are degraded."""
        manager = EnhancedDatabaseManager()
        
        # Setup degraded system health
        error_handler = realistic_mocks['get_error_handler'].return_value
        error_handler.get_system_health.return_value = {
            'overall_health': {
                'health_score': 65.0,
                'status': 'degraded',
                'recommendations': ['High error rate detected']
            },
            'circuit_breakers': {
                'total_breakers': 3,
                'states': {'closed': 2, 'open': 1}
            },
            'error_statistics': {'total_errors': 25, 'error_rate_per_minute': 2.5}
        }
        
        # Setup connection manager with mixed health
        connection_manager = realistic_mocks['get_enhanced_connection_manager'].return_value
        connection_manager.list_databases.return_value = [
            {'name': 'primary_db', 'health': {'is_healthy': True}},
            {'name': 'secondary_db', 'health': {'is_healthy': False}},
            {'name': 'analytics_db', 'health': {'is_healthy': True}}
        ]
        
        # Get health status
        health = manager.get_health_status()
        
        # Verify degraded status is reflected
        assert health['overall_status'] in ['degraded', 'unhealthy']
        assert health['overall_health_score'] < 90.0
        
        # Verify recommendations are provided
        assert len(health['recommendations']) > 0
        
        # Verify component health details
        conn_health = health['components']['connection_manager']
        assert conn_health['total_databases'] == 3
        assert conn_health['healthy_connections'] == 2
        assert conn_health['health_rate'] < 100.0


if __name__ == "__main__":
    pytest.main([__file__])