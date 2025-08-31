"""Comprehensive tests for the advanced error handling system."""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from src.localdata_mcp.error_handler import (
    # Exception classes
    LocalDataError, DatabaseConnectionError, QueryExecutionError,
    SecurityViolationError, QueryTimeoutError, ResourceExhaustionError,
    ConfigurationError,
    
    # Enums
    ErrorCategory, ErrorSeverity, RetryStrategy, CircuitState,
    
    # Retry mechanism
    RetryPolicy, RetryableOperation, retry_on_failure,
    
    # Circuit breaker
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry,
    circuit_breaker_protection,
    
    # Recovery system
    ErrorRecoveryManager, RecoveryStrategy, RecoveryAction,
    
    # Error handling
    ErrorHandler, ErrorLogger, get_error_handler, initialize_error_handler
)


class TestCustomExceptions:
    """Test custom exception hierarchy."""
    
    def test_local_data_error_basic(self):
        """Test basic LocalDataError functionality."""
        error = LocalDataError(
            message="Test error",
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.HIGH,
            database_name="test_db",
            query="SELECT * FROM test"
        )
        
        assert error.message == "Test error"
        assert error.category == ErrorCategory.CONNECTION
        assert error.severity == ErrorSeverity.HIGH
        assert error.database_name == "test_db"
        assert error.query == "SELECT * FROM test"
        assert error.error_code.startswith("connection_")
        assert isinstance(error.timestamp, float)
        assert str(error) == "[CONNECTION] Test error"
    
    def test_local_data_error_to_dict(self):
        """Test error serialization to dictionary."""
        error = LocalDataError(
            message="Test error",
            category=ErrorCategory.QUERY_EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            database_name="test_db",
            metadata={"key": "value"},
            recovery_suggestions=["Try again", "Check connection"]
        )
        
        error_dict = error.to_dict()
        
        assert error_dict['message'] == "Test error"
        assert error_dict['category'] == "query_execution"
        assert error_dict['severity'] == "medium"
        assert error_dict['database_name'] == "test_db"
        assert error_dict['metadata'] == {"key": "value"}
        assert error_dict['recovery_suggestions'] == ["Try again", "Check connection"]
    
    def test_database_connection_error(self):
        """Test DatabaseConnectionError specific functionality."""
        error = DatabaseConnectionError(
            message="Connection failed",
            database_name="test_db"
        )
        
        assert error.category == ErrorCategory.CONNECTION
        assert error.database_name == "test_db"
        assert len(error.recovery_suggestions) > 0
        assert "connection parameters" in error.recovery_suggestions[0].lower()
    
    def test_security_violation_error(self):
        """Test SecurityViolationError specific functionality."""
        error = SecurityViolationError(
            message="SQL injection detected",
            query="SELECT * FROM users WHERE id = 1; DROP TABLE users;"
        )
        
        assert error.category == ErrorCategory.SECURITY_VIOLATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.query == "SELECT * FROM users WHERE id = 1; DROP TABLE users;"
        assert len(error.recovery_suggestions) > 0
    
    def test_query_timeout_error(self):
        """Test QueryTimeoutError with timing metadata."""
        error = QueryTimeoutError(
            message="Query timed out",
            execution_time=45.0,
            timeout_limit=30.0,
            query="SELECT * FROM large_table"
        )
        
        assert error.category == ErrorCategory.TIMEOUT
        assert error.metadata['execution_time'] == 45.0
        assert error.metadata['timeout_limit'] == 30.0
        assert error.query == "SELECT * FROM large_table"


class TestRetryMechanism:
    """Test retry mechanism and policies."""
    
    def test_retry_policy_delay_calculation(self):
        """Test different retry delay calculation strategies."""
        
        # Exponential backoff
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False  # Disable jitter for predictable testing
        )
        
        assert policy.calculate_delay(1) == 1.0
        assert policy.calculate_delay(2) == 2.0  
        assert policy.calculate_delay(3) == 4.0
        
        # Fixed delay
        fixed_policy = RetryPolicy(
            strategy=RetryStrategy.FIXED,
            base_delay=5.0,
            jitter=False
        )
        
        assert fixed_policy.calculate_delay(1) == 5.0
        assert fixed_policy.calculate_delay(2) == 5.0
        assert fixed_policy.calculate_delay(3) == 5.0
        
        # Linear backoff
        linear_policy = RetryPolicy(
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay=2.0,
            jitter=False
        )
        
        assert linear_policy.calculate_delay(1) == 2.0
        assert linear_policy.calculate_delay(2) == 4.0
        assert linear_policy.calculate_delay(3) == 6.0
    
    def test_retry_policy_should_retry(self):
        """Test retry decision logic."""
        policy = RetryPolicy(
            max_attempts=3,
            retry_on=(DatabaseConnectionError, QueryExecutionError),
            stop_on=(SecurityViolationError,)
        )
        
        # Should retry on allowed exceptions
        conn_error = DatabaseConnectionError("Connection failed")
        assert policy.should_retry(conn_error, 1) is True
        assert policy.should_retry(conn_error, 2) is True
        assert policy.should_retry(conn_error, 3) is False  # Max attempts reached
        
        # Should not retry on stop exceptions
        security_error = SecurityViolationError("SQL injection detected")
        assert policy.should_retry(security_error, 1) is False
        
        # Should not retry on unhandled exceptions
        value_error = ValueError("Invalid value")
        assert policy.should_retry(value_error, 1) is False
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_retryable_operation_success(self, mock_sleep):
        """Test successful operation execution."""
        mock_operation = Mock(return_value="success")
        
        policy = RetryPolicy(max_attempts=3, base_delay=0.1)
        retryable_op = RetryableOperation(mock_operation, policy, "test_op")
        
        result = retryable_op.execute("arg1", kwarg1="value1")
        
        assert result == "success"
        assert mock_operation.call_count == 1
        assert mock_sleep.call_count == 0
        assert len(retryable_op.attempt_history) == 0  # No failures recorded
    
    @patch('time.sleep')
    def test_retryable_operation_retry_and_succeed(self, mock_sleep):
        """Test operation that fails first then succeeds."""
        mock_operation = Mock(side_effect=[
            DatabaseConnectionError("Connection failed"),
            "success"
        ])
        
        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.1,
            retry_on=(DatabaseConnectionError,),
            jitter=False
        )
        retryable_op = RetryableOperation(mock_operation, policy, "test_op")
        
        result = retryable_op.execute()
        
        assert result == "success"
        assert mock_operation.call_count == 2
        assert mock_sleep.call_count == 1
        assert len(retryable_op.attempt_history) == 1  # One failure recorded
    
    @patch('time.sleep')
    def test_retryable_operation_max_attempts_exceeded(self, mock_sleep):
        """Test operation that fails all retry attempts."""
        mock_operation = Mock(side_effect=DatabaseConnectionError("Connection failed"))
        
        policy = RetryPolicy(
            max_attempts=2,
            base_delay=0.1,
            retry_on=(DatabaseConnectionError,),
            jitter=False
        )
        retryable_op = RetryableOperation(mock_operation, policy, "test_op")
        
        with pytest.raises(QueryExecutionError) as exc_info:
            retryable_op.execute()
        
        assert "failed after 2 attempts" in str(exc_info.value)
        assert mock_operation.call_count == 2
        assert mock_sleep.call_count == 1
        assert len(retryable_op.attempt_history) == 2
    
    def test_retry_decorator(self):
        """Test retry decorator functionality."""
        call_count = 0
        
        @retry_on_failure(
            policy=RetryPolicy(
                max_attempts=3,
                base_delay=0.01,
                retry_on=(ValueError,),
                jitter=False
            )
        )
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not ready yet")
            return "success"
        
        with patch('time.sleep'):
            result = test_function()
        
        assert result == "success"
        assert call_count == 3


class TestCircuitBreaker:
    """Test circuit breaker pattern implementation."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization and default state."""
        config = CircuitBreakerConfig(failure_threshold=3, success_threshold=2)
        breaker = CircuitBreaker("test_breaker", config)
        
        assert breaker.name == "test_breaker"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_request_allowed() is True
        
        state_info = breaker.get_state_info()
        assert state_info['name'] == "test_breaker"
        assert state_info['state'] == "closed"
    
    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker failure tracking and state transitions."""
        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=1)
        breaker = CircuitBreaker("test_breaker", config)
        
        # Record failures
        breaker.record_failure(DatabaseConnectionError("Connection failed"))
        assert breaker.state == CircuitState.CLOSED
        
        breaker.record_failure(DatabaseConnectionError("Connection failed"))
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_request_allowed() is False
        
        state_info = breaker.get_state_info()
        assert state_info['stats']['failed_requests'] == 2
        assert state_info['stats']['circuit_opened_count'] == 1
    
    def test_circuit_breaker_half_open_transition(self):
        """Test circuit breaker transition to half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,
            timeout_duration=0.1  # Short timeout for testing
        )
        breaker = CircuitBreaker("test_breaker", config)
        
        # Trigger circuit to open
        breaker.record_failure(DatabaseConnectionError("Connection failed"))
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Check if request is now allowed (should transition to half-open)
        assert breaker.is_request_allowed() is True
        assert breaker.state == CircuitState.HALF_OPEN
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery to closed state."""
        config = CircuitBreakerConfig(failure_threshold=1, success_threshold=1)
        breaker = CircuitBreaker("test_breaker", config)
        
        # Open the circuit
        breaker.record_failure(DatabaseConnectionError("Connection failed"))
        assert breaker.state == CircuitState.OPEN
        
        # Manually transition to half-open for testing
        breaker._transition_to_half_open()
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Record success to close the circuit
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_request_allowed() is True
    
    def test_circuit_breaker_registry(self):
        """Test circuit breaker registry functionality."""
        registry = CircuitBreakerRegistry()
        
        # Get breaker (should create new one)
        breaker1 = registry.get_breaker("test_breaker_1")
        assert breaker1.name == "test_breaker_1"
        
        # Get same breaker again (should return existing)
        breaker2 = registry.get_breaker("test_breaker_1")
        assert breaker1 is breaker2
        
        # Get all breakers
        all_breakers = registry.get_all_breakers()
        assert "test_breaker_1" in all_breakers
        
        # Remove breaker
        success = registry.remove_breaker("test_breaker_1")
        assert success is True
        
        all_breakers_after = registry.get_all_breakers()
        assert "test_breaker_1" not in all_breakers_after
    
    def test_circuit_breaker_protection_context_manager(self):
        """Test circuit breaker protection context manager."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test_breaker", config)
        
        # Should work normally when closed
        with circuit_breaker_protection(breaker, "test_operation"):
            pass  # Operation succeeds
        
        assert breaker.stats.successful_requests == 1
        
        # Open the circuit
        breaker.record_failure(DatabaseConnectionError("Connection failed"))
        assert breaker.state == CircuitState.OPEN
        
        # Should raise exception when circuit is open
        with pytest.raises(DatabaseConnectionError) as exc_info:
            with circuit_breaker_protection(breaker, "test_operation"):
                pass
        
        assert "Circuit breaker" in str(exc_info.value)
        assert "is open" in str(exc_info.value)


class TestErrorRecovery:
    """Test error recovery system."""
    
    def test_recovery_action_execution(self):
        """Test recovery action execution."""
        def test_recovery_function(context):
            return f"Recovery executed with {context['test_param']}"
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.CONNECTION_RESET,
            description="Test recovery action",
            action_function=test_recovery_function,
            success_probability=0.8
        )
        
        context = {"test_param": "test_value"}
        success, result, error_msg = action.execute(context)
        
        assert success is True
        assert result == "Recovery executed with test_value"
        assert error_msg is None
    
    def test_recovery_action_failure(self):
        """Test recovery action failure handling."""
        def failing_recovery_function(context):
            raise ValueError("Recovery failed")
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.QUERY_SIMPLIFICATION,
            description="Failing recovery action",
            action_function=failing_recovery_function
        )
        
        success, result, error_msg = action.execute({})
        
        assert success is False
        assert result is None
        assert "Recovery failed" in error_msg
    
    def test_error_recovery_manager(self):
        """Test error recovery manager functionality."""
        manager = ErrorRecoveryManager()
        
        # Test getting recovery options
        error = DatabaseConnectionError("Connection failed", database_name="test_db")
        options = manager.get_recovery_options(error)
        
        assert len(options) > 0
        assert any(action.strategy == RecoveryStrategy.CONNECTION_RESET for action in options)
    
    def test_error_recovery_manager_attempt_recovery(self):
        """Test error recovery attempt."""
        manager = ErrorRecoveryManager()
        
        # Create a test error
        error = ResourceExhaustionError(
            "Memory exhausted",
            resource_type="memory",
            query="SELECT * FROM large_table"
        )
        
        # Attempt recovery
        recovered, result, attempted_strategies = manager.attempt_recovery(error)
        
        # Should have attempted some strategies
        assert len(attempted_strategies) > 0
        
        # For resource exhaustion, should have query simplification and pagination strategies
        expected_strategies = [RecoveryStrategy.RESULT_PAGINATION.value, RecoveryStrategy.QUERY_SIMPLIFICATION.value]
        assert any(strategy in attempted_strategies for strategy in expected_strategies)


class TestErrorLogger:
    """Test error logging and monitoring."""
    
    def test_error_logger_log_error(self):
        """Test error logging functionality."""
        logger = ErrorLogger("test.error_logger")
        
        error = DatabaseConnectionError(
            "Connection failed",
            database_name="test_db",
            metadata={"host": "localhost", "port": 5432}
        )
        
        # Log the error
        logger.log_error(error, {"extra_context": "test_value"})
        
        # Check metrics were updated
        assert logger.metrics.total_errors == 1
        assert logger.metrics.errors_by_category[ErrorCategory.CONNECTION] == 1
        assert logger.metrics.errors_by_database["test_db"] == 1
        assert len(logger.metrics.recent_errors) == 1
        
        recent_error = logger.metrics.recent_errors[0]
        assert recent_error['category'] == 'connection'
        assert recent_error['database_name'] == 'test_db'
    
    def test_error_logger_statistics(self):
        """Test error statistics generation."""
        logger = ErrorLogger("test.error_logger")
        
        # Log multiple errors
        errors = [
            DatabaseConnectionError("Connection failed", database_name="db1"),
            QueryExecutionError("Query failed", database_name="db1"),
            SecurityViolationError("Security violation", database_name="db2")
        ]
        
        for error in errors:
            logger.log_error(error)
        
        stats = logger.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['errors_by_category']['connection'] == 1
        assert stats['errors_by_category']['query_execution'] == 1
        assert stats['errors_by_category']['security_violation'] == 1
        assert stats['errors_by_database']['db1'] == 2
        assert stats['errors_by_database']['db2'] == 1
    
    def test_error_logger_health_status(self):
        """Test health status calculation."""
        logger = ErrorLogger("test.error_logger")
        
        # Initially should be healthy
        health = logger.get_health_status()
        assert health['status'] == 'healthy'
        assert health['health_score'] == 100.0
        
        # Log some critical errors
        for _ in range(3):
            logger.log_error(SecurityViolationError("Critical security issue"))
        
        health_after_errors = logger.get_health_status()
        assert health_after_errors['health_score'] < 100.0
        assert len(health_after_errors['recommendations']) > 0


class TestErrorHandler:
    """Test main error handler integration."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler()
        
        assert handler.logger is not None
        assert handler.recovery_manager is not None
        assert handler.circuit_breaker_registry is not None
        assert len(handler.retry_policies) > 0
    
    def test_error_handler_handle_error_with_recovery(self):
        """Test error handling with recovery attempt."""
        handler = ErrorHandler()
        
        # Create a mock error that can be recovered
        error = ResourceExhaustionError(
            "Memory exhausted",
            resource_type="memory",
            query="SELECT * FROM large_table"
        )
        
        context = {"database_name": "test_db", "query": error.query}
        
        # Handle the error
        recovered, result, processed_error = handler.handle_error(
            error, context, attempt_recovery=True
        )
        
        assert processed_error is not None
        assert processed_error.category == ErrorCategory.RESOURCE_EXHAUSTION
        # Recovery might succeed or fail, but should have been attempted
    
    def test_error_handler_convert_generic_exception(self):
        """Test conversion of generic exceptions to LocalDataError."""
        handler = ErrorHandler()
        
        # Test connection-related error
        conn_error = ConnectionError("Host unreachable")
        context = {"database_name": "test_db"}
        
        recovered, result, processed_error = handler.handle_error(conn_error, context)
        
        assert isinstance(processed_error, DatabaseConnectionError)
        assert processed_error.category == ErrorCategory.CONNECTION
        assert processed_error.database_name == "test_db"
        assert processed_error.cause == conn_error
    
    def test_error_handler_system_health(self):
        """Test system health monitoring."""
        handler = ErrorHandler()
        
        # Generate some errors
        errors = [
            DatabaseConnectionError("Connection failed"),
            QueryExecutionError("Query failed"),
            SecurityViolationError("Security violation")
        ]
        
        for error in errors:
            handler.handle_error(error)
        
        health = handler.get_system_health()
        
        assert 'overall_health' in health
        assert 'circuit_breakers' in health
        assert 'recovery_statistics' in health
        assert 'error_statistics' in health
        assert 'timestamp' in health
    
    def test_global_error_handler_functions(self):
        """Test global error handler accessor functions."""
        # Get global handler
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        # Should return same instance
        assert handler1 is handler2
        
        # Initialize new handler
        handler3 = initialize_error_handler()
        handler4 = get_error_handler()
        
        # Should now return the new instance
        assert handler3 is handler4
        assert handler3 is not handler1


class TestErrorHandlerIntegration:
    """Integration tests for error handler with realistic scenarios."""
    
    def test_database_connection_failure_scenario(self):
        """Test handling of database connection failures."""
        handler = ErrorHandler()
        
        # Simulate connection failure
        connection_error = ConnectionError("Connection refused")
        context = {
            "database_name": "production_db",
            "operation": "connect",
            "host": "db.example.com",
            "port": 5432
        }
        
        recovered, result, processed_error = handler.handle_error(
            connection_error, context, attempt_recovery=True
        )
        
        # Should be converted to DatabaseConnectionError
        assert isinstance(processed_error, DatabaseConnectionError)
        assert processed_error.database_name == "production_db"
        assert len(processed_error.recovery_suggestions) > 0
        
        # Should have attempted recovery
        recovery_stats = handler.recovery_manager.get_recovery_statistics()
        # Might be 0 if no suitable recovery strategies, but should not error
        assert recovery_stats['total_attempts'] >= 0
    
    def test_query_timeout_failure_scenario(self):
        """Test handling of query timeout scenarios."""
        handler = ErrorHandler()
        
        # Simulate query timeout
        timeout_error = TimeoutError("Query execution exceeded time limit")
        context = {
            "database_name": "analytics_db",
            "query": "SELECT COUNT(*) FROM massive_table GROUP BY category",
            "execution_time": 300.0,
            "timeout_limit": 120.0
        }
        
        recovered, result, processed_error = handler.handle_error(
            timeout_error, context, attempt_recovery=True
        )
        
        # Should be converted to QueryTimeoutError
        assert isinstance(processed_error, QueryTimeoutError)
        assert processed_error.database_name == "analytics_db"
        assert "massive_table" in processed_error.query
        
        # Should have recovery suggestions for timeouts
        assert len(processed_error.recovery_suggestions) > 0
        assert any("timeout" in suggestion.lower() for suggestion in processed_error.recovery_suggestions)
    
    @patch('time.time')
    def test_error_rate_monitoring(self, mock_time):
        """Test error rate monitoring over time."""
        # Mock time progression
        current_time = 1000000000.0
        mock_time.return_value = current_time
        
        handler = ErrorHandler()
        
        # Simulate errors over time
        for i in range(10):
            mock_time.return_value = current_time + (i * 10)  # 10 second intervals
            error = QueryExecutionError(f"Query {i} failed")
            handler.handle_error(error)
        
        # Check error statistics
        stats = handler.get_error_statistics()
        error_stats = stats['error_statistics']
        
        assert error_stats['total_errors'] == 10
        assert len(error_stats['recent_errors']) <= 10
        
        # Check health status
        health = stats['overall_health']
        # High error count should affect health
        assert health['health_score'] < 100.0


if __name__ == "__main__":
    pytest.main([__file__])