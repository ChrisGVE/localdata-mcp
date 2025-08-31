"""Advanced Error Handling System for LocalData MCP v1.3.1.

This module provides comprehensive error handling with custom exception hierarchy,
retry mechanisms with exponential backoff, circuit breaker pattern for database
connections, and integration with existing security, timeout, and connection systems.

Key Features:
- Custom exception hierarchy for different error types
- Retry policies with exponential backoff and jitter
- Circuit breaker pattern for database connection failures
- Error recovery strategies and graceful degradation
- Comprehensive error logging and metadata preservation
- Integration with SecurityManager, ConnectionManager, and TimeoutManager
"""

import asyncio
import logging
import random
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for classification and handling."""
    CONNECTION = "connection"
    QUERY_EXECUTION = "query_execution"
    SECURITY_VIOLATION = "security_violation"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    DATA_VALIDATION = "data_validation"
    SYSTEM = "system"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED = "fixed"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI = "fibonacci"
    NO_RETRY = "no_retry"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, rejecting requests
    HALF_OPEN = "half_open" # Testing if service has recovered


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================

class LocalDataError(Exception):
    """Base exception for all LocalData MCP errors.
    
    Provides structured error information with metadata, context, and recovery suggestions.
    """
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 error_code: Optional[str] = None,
                 database_name: Optional[str] = None,
                 query: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None,
                 recovery_suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or f"{category.value}_{int(time.time())}"
        self.database_name = database_name
        self.query = query
        self.metadata = metadata or {}
        self.cause = cause
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and serialization."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'database_name': self.database_name,
            'query': self.query[:200] if self.query else None,
            'metadata': self.metadata,
            'cause': str(self.cause) if self.cause else None,
            'recovery_suggestions': self.recovery_suggestions,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        return f"[{self.category.value.upper()}] {self.message}"


class DatabaseConnectionError(LocalDataError):
    """Database connection related errors."""
    
    def __init__(self, message: str, database_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONNECTION,
            database_name=database_name,
            recovery_suggestions=[
                "Check database connection parameters",
                "Verify network connectivity",
                "Ensure database server is running",
                "Check firewall settings"
            ],
            **kwargs
        )


class QueryExecutionError(LocalDataError):
    """Query execution related errors."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.QUERY_EXECUTION,
            query=query,
            recovery_suggestions=[
                "Check SQL syntax",
                "Verify table and column names exist",
                "Ensure sufficient permissions",
                "Consider query optimization"
            ],
            **kwargs
        )


class SecurityViolationError(LocalDataError):
    """Security violation errors."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.SECURITY_VIOLATION,
            severity=ErrorSeverity.HIGH,
            query=query,
            recovery_suggestions=[
                "Review query for security issues",
                "Check SQL injection patterns",
                "Verify query permissions",
                "Contact administrator if needed"
            ],
            **kwargs
        )


class QueryTimeoutError(LocalDataError):
    """Query timeout related errors."""
    
    def __init__(self, message: str, execution_time: float = 0.0, timeout_limit: float = 0.0, **kwargs):
        metadata = kwargs.get('metadata', {})
        metadata.update({
            'execution_time': execution_time,
            'timeout_limit': timeout_limit
        })
        
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            metadata=metadata,
            recovery_suggestions=[
                "Increase query timeout limit",
                "Optimize query performance",
                "Add appropriate indexes",
                "Consider data partitioning"
            ],
            **kwargs
        )


class ResourceExhaustionError(LocalDataError):
    """Resource exhaustion errors (memory, CPU, connections)."""
    
    def __init__(self, message: str, resource_type: str = "unknown", **kwargs):
        metadata = kwargs.get('metadata', {})
        metadata['resource_type'] = resource_type
        
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE_EXHAUSTION,
            severity=ErrorSeverity.HIGH,
            metadata=metadata,
            recovery_suggestions=[
                "Reduce query complexity",
                "Implement data pagination",
                "Close unused connections",
                "Increase system resources"
            ],
            **kwargs
        )


class ConfigurationError(LocalDataError):
    """Configuration related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        metadata = kwargs.get('metadata', {})
        if config_key:
            metadata['config_key'] = config_key
        
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            metadata=metadata,
            recovery_suggestions=[
                "Check configuration file syntax",
                "Verify all required parameters",
                "Review default values",
                "Consult documentation"
            ],
            **kwargs
        )


# ============================================================================
# Retry Mechanism
# ============================================================================

@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on: Tuple[Type[Exception], ...] = field(default_factory=lambda: (Exception,))
    stop_on: Tuple[Type[Exception], ...] = field(default_factory=tuple)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a specific retry attempt."""
        if self.strategy == RetryStrategy.NO_RETRY:
            return 0.0
        elif self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        elif self.strategy == RetryStrategy.FIBONACCI:
            if attempt <= 2:
                delay = self.base_delay
            else:
                # Calculate Fibonacci number for delay
                fib_delay = self.base_delay
                prev_delay = self.base_delay
                for _ in range(attempt - 2):
                    fib_delay, prev_delay = fib_delay + prev_delay, fib_delay
                delay = fib_delay
        else:
            delay = self.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.jitter:
            jitter_factor = random.uniform(0.1, 0.1)  # 10% jitter
            delay += delay * jitter_factor
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        if attempt >= self.max_attempts:
            return False
        
        # Check if exception is in stop list
        if any(isinstance(exception, stop_type) for stop_type in self.stop_on):
            return False
        
        # Check if exception is in retry list
        return any(isinstance(exception, retry_type) for retry_type in self.retry_on)


class RetryableOperation:
    """Wrapper for operations that can be retried."""
    
    def __init__(self, operation: Callable, policy: RetryPolicy, 
                 operation_name: str = "unknown"):
        self.operation = operation
        self.policy = policy
        self.operation_name = operation_name
        self.attempt_history: List[Dict[str, Any]] = []
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the operation with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.policy.max_attempts + 1):
            attempt_start = time.time()
            
            try:
                result = self.operation(*args, **kwargs)
                
                # Log successful execution after retries
                if attempt > 1:
                    logger.info(f"Operation '{self.operation_name}' succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                attempt_duration = time.time() - attempt_start
                
                # Record attempt
                self.attempt_history.append({
                    'attempt': attempt,
                    'exception': str(e),
                    'exception_type': type(e).__name__,
                    'duration': attempt_duration,
                    'timestamp': attempt_start
                })
                
                # Check if we should retry
                if not self.policy.should_retry(e, attempt):
                    logger.error(f"Operation '{self.operation_name}' failed permanently after {attempt} attempts: {e}")
                    break
                
                # Calculate delay before next attempt
                if attempt < self.policy.max_attempts:
                    delay = self.policy.calculate_delay(attempt)
                    logger.warning(f"Operation '{self.operation_name}' failed on attempt {attempt}, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
        
        # All retries exhausted, raise the last exception with context
        if isinstance(last_exception, LocalDataError):
            # Add retry context to existing LocalData error
            last_exception.metadata['retry_attempts'] = len(self.attempt_history)
            last_exception.metadata['retry_history'] = self.attempt_history[-3:]  # Last 3 attempts
            raise last_exception
        else:
            # Wrap other exceptions in a LocalData error with retry context
            raise QueryExecutionError(
                message=f"Operation '{self.operation_name}' failed after {len(self.attempt_history)} attempts: {str(last_exception)}",
                cause=last_exception,
                metadata={
                    'retry_attempts': len(self.attempt_history),
                    'retry_history': self.attempt_history[-3:],
                    'operation_name': self.operation_name
                }
            )


def retry_on_failure(policy: RetryPolicy, operation_name: str = None):
    """Decorator for adding retry logic to functions."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            retryable_op = RetryableOperation(func, policy, name)
            return retryable_op.execute(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening circuit
    success_threshold: int = 3          # Successes before closing circuit
    timeout_duration: float = 60.0     # Seconds to wait before half-open
    monitor_window: float = 300.0       # Window to track failures (5 minutes)
    
    # Failure conditions
    failure_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (DatabaseConnectionError, ResourceExhaustionError)
    )


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_opened_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    failure_window: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    @property 
    def recent_failure_count(self) -> int:
        """Count recent failures within monitor window."""
        current_time = time.time()
        recent_failures = [
            failure_time for failure_time in self.failure_window
            if current_time - failure_time <= 300.0  # 5 minute window
        ]
        return len(recent_failures)


class CircuitBreaker:
    """Circuit breaker implementation for database connections."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.lock = threading.RLock()
        self.half_open_attempts = 0
        self.state_change_time = time.time()
    
    def is_request_allowed(self) -> bool:
        """Check if a request is allowed through the circuit breaker."""
        with self.lock:
            current_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                # Check if timeout has elapsed to move to half-open
                if current_time - self.state_change_time >= self.config.timeout_duration:
                    self._transition_to_half_open()
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                return self.half_open_attempts < self.config.success_threshold
        
        return False
    
    def record_success(self):
        """Record a successful operation."""
        with self.lock:
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.last_success_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_attempts += 1
                if self.half_open_attempts >= self.config.success_threshold:
                    self._transition_to_closed()
    
    def record_failure(self, exception: Exception):
        """Record a failed operation."""
        with self.lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            current_time = time.time()
            self.stats.last_failure_time = current_time
            self.stats.failure_window.append(current_time)
            
            # Check if this is a circuit-breaking failure
            is_circuit_failure = any(
                isinstance(exception, failure_type) 
                for failure_type in self.config.failure_exceptions
            )
            
            if is_circuit_failure:
                if self.state == CircuitState.HALF_OPEN:
                    # Failure during half-open immediately opens circuit
                    self._transition_to_open()
                elif self.state == CircuitState.CLOSED:
                    # Check if we should open the circuit
                    if self.stats.recent_failure_count >= self.config.failure_threshold:
                        self._transition_to_open()
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'stats': {
                    'total_requests': self.stats.total_requests,
                    'successful_requests': self.stats.successful_requests,
                    'failed_requests': self.stats.failed_requests,
                    'failure_rate': self.stats.failure_rate,
                    'recent_failure_count': self.stats.recent_failure_count,
                    'circuit_opened_count': self.stats.circuit_opened_count,
                    'last_failure_time': self.stats.last_failure_time,
                    'last_success_time': self.stats.last_success_time
                },
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'success_threshold': self.config.success_threshold,
                    'timeout_duration': self.config.timeout_duration
                },
                'state_change_time': self.state_change_time,
                'half_open_attempts': self.half_open_attempts
            }
    
    def reset(self):
        """Reset the circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.half_open_attempts = 0
            self.state_change_time = time.time()
            # Keep stats for monitoring, but could optionally reset them
            logger.info(f"Circuit breaker '{self.name}' has been reset")
    
    def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        self.state = CircuitState.OPEN
        self.state_change_time = time.time()
        self.stats.circuit_opened_count += 1
        self.half_open_attempts = 0
        logger.warning(f"Circuit breaker '{self.name}' opened due to failures")
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = time.time()
        self.half_open_attempts = 0
        logger.info(f"Circuit breaker '{self.name}' moved to half-open state")
    
    def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.state_change_time = time.time()
        self.half_open_attempts = 0
        logger.info(f"Circuit breaker '{self.name}' closed after successful recovery")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker for a given name."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def remove_breaker(self, name: str) -> bool:
        """Remove a circuit breaker."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False
    
    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        with self._lock:
            return self._breakers.copy()
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary of all circuit breakers."""
        with self._lock:
            summary = {
                'total_breakers': len(self._breakers),
                'states': defaultdict(int),
                'breakers': {}
            }
            
            for name, breaker in self._breakers.items():
                state_info = breaker.get_state_info()
                summary['states'][state_info['state']] += 1
                summary['breakers'][name] = state_info
            
            return summary


@contextmanager
def circuit_breaker_protection(breaker: CircuitBreaker, operation_name: str = "unknown"):
    """Context manager for circuit breaker protection."""
    if not breaker.is_request_allowed():
        raise DatabaseConnectionError(
            message=f"Circuit breaker '{breaker.name}' is open, rejecting request",
            metadata={
                'circuit_breaker_state': breaker.state.value,
                'operation_name': operation_name
            },
            recovery_suggestions=[
                "Wait for circuit breaker to recover",
                "Check database connectivity",
                "Review recent error patterns"
            ]
        )
    
    try:
        yield
        breaker.record_success()
    except Exception as e:
        breaker.record_failure(e)
        raise