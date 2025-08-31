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

from .logging_manager import get_logging_manager, get_logger

# Get structured logger
logger = get_logger(__name__)


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
                    logging_manager = get_logging_manager()
                    with logging_manager.context(
                        operation="retry_success",
                        component="error_handler",
                        operation_name=self.operation_name
                    ):
                        logger.info("Operation succeeded after retries",
                                  attempt=attempt,
                                  total_attempts=self.policy.max_attempts,
                                  retry_policy=self.policy.__class__.__name__)
                
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
                    logging_manager = get_logging_manager()
                    logging_manager.log_error(e, "error_handler", 
                                           operation_name=self.operation_name,
                                           final_attempt=attempt,
                                           retry_policy=self.policy.__class__.__name__)
                    break
                
                # Calculate delay before next attempt
                if attempt < self.policy.max_attempts:
                    delay = self.policy.calculate_delay(attempt)
                    logging_manager = get_logging_manager()
                    with logging_manager.context(
                        operation="retry_attempt",
                        component="error_handler",
                        operation_name=self.operation_name
                    ):
                        logger.warning("Operation failed, will retry",
                                     attempt=attempt,
                                     max_attempts=self.policy.max_attempts,
                                     retry_delay=delay,
                                     error_type=type(e).__name__,
                                     error_message=str(e))
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
            logging_manager = get_logging_manager()
            with logging_manager.context(
                operation="circuit_breaker_reset",
                component="error_handler",
                circuit_breaker_name=self.name
            ):
                logger.info("Circuit breaker reset",
                          failure_rate=self.stats.failure_rate,
                          total_requests=self.stats.total_requests,
                          failed_requests=self.stats.failed_requests)
    
    def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        self.state = CircuitState.OPEN
        self.state_change_time = time.time()
        self.stats.circuit_opened_count += 1
        self.half_open_attempts = 0
        logging_manager = get_logging_manager()
        logging_manager.log_security_event(
            "circuit_breaker_opened",
            "medium",
            f"Circuit breaker '{self.name}' opened due to excessive failures",
            circuit_breaker_name=self.name,
            failure_rate=self.stats.failure_rate,
            failed_requests=self.stats.failed_requests,
            threshold=self.config.failure_threshold
        )
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = time.time()
        self.half_open_attempts = 0
        logging_manager = get_logging_manager()
        with logging_manager.context(
            operation="circuit_breaker_half_open",
            component="error_handler",
            circuit_breaker_name=self.name
        ):
            logger.info("Circuit breaker moved to half-open state",
                      timeout_duration=self.config.timeout_duration,
                      open_duration=time.time() - self.state_change_time)
    
    def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.state_change_time = time.time()
        self.half_open_attempts = 0
        logging_manager = get_logging_manager()
        with logging_manager.context(
            operation="circuit_breaker_closed",
            component="error_handler",
            circuit_breaker_name=self.name
        ):
            logger.info("Circuit breaker closed after recovery",
                      half_open_attempts=self.half_open_attempts,
                      success_threshold=self.config.success_threshold,
                      total_downtime=time.time() - self.state_change_time)


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


# ============================================================================
# Error Recovery Strategies
# ============================================================================

class RecoveryStrategy(Enum):
    """Types of recovery strategies."""
    CONNECTION_RESET = "connection_reset"
    QUERY_SIMPLIFICATION = "query_simplification"
    RESULT_PAGINATION = "result_pagination"
    CACHE_FALLBACK = "cache_fallback"
    READ_REPLICA = "read_replica"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    PARTIAL_RESULTS = "partial_results"


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken."""
    strategy: RecoveryStrategy
    description: str
    action_function: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_probability: float = 0.5
    estimated_time: float = 0.0
    
    def execute(self, context: Dict[str, Any]) -> Tuple[bool, Any, Optional[str]]:
        """Execute the recovery action.
        
        Returns:
            Tuple[bool, Any, Optional[str]]: (success, result, error_message)
        """
        try:
            result = self.action_function(context)
            return True, result, None
        except Exception as e:
            return False, None, str(e)


class ErrorRecoveryManager:
    """Manages error recovery strategies and execution."""
    
    def __init__(self):
        self._recovery_strategies: Dict[ErrorCategory, List[RecoveryAction]] = defaultdict(list)
        self._recovery_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._register_default_strategies()
    
    def register_strategy(self, category: ErrorCategory, action: RecoveryAction):
        """Register a recovery strategy for an error category."""
        with self._lock:
            self._recovery_strategies[category].append(action)
    
    def get_recovery_options(self, error: LocalDataError) -> List[RecoveryAction]:
        """Get applicable recovery options for an error."""
        with self._lock:
            return self._recovery_strategies.get(error.category, []).copy()
    
    def attempt_recovery(self, error: LocalDataError, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any, List[str]]:
        """Attempt recovery from an error using available strategies.
        
        Returns:
            Tuple[bool, Any, List[str]]: (recovered, result, attempted_strategies)
        """
        context = context or {}
        context.update({
            'error': error,
            'database_name': error.database_name,
            'query': error.query,
            'metadata': error.metadata
        })
        
        recovery_options = self.get_recovery_options(error)
        attempted_strategies = []
        
        # Sort by success probability (highest first)
        recovery_options.sort(key=lambda x: x.success_probability, reverse=True)
        
        for action in recovery_options:
            attempted_strategies.append(action.strategy.value)
            
            try:
                success, result, error_msg = action.execute(context)
                
                # Record recovery attempt
                recovery_record = {
                    'timestamp': time.time(),
                    'error_code': error.error_code,
                    'error_category': error.category.value,
                    'strategy': action.strategy.value,
                    'success': success,
                    'error_message': error_msg,
                    'database_name': error.database_name
                }
                
                with self._lock:
                    self._recovery_history.append(recovery_record)
                
                if success:
                    logging_manager = get_logging_manager()
                    with logging_manager.context(
                        operation="error_recovery_success",
                        component="error_handler",
                        error_code=error.error_code
                    ):
                        logger.info("Error recovery successful",
                                  strategy=action.strategy.value,
                                  attempt_duration=recovery_record['duration'],
                                  attempts_before_success=len(attempted_strategies))
                    return True, result, attempted_strategies
                else:
                    logging_manager = get_logging_manager()
                    with logging_manager.context(
                        operation="error_recovery_attempt",
                        component="error_handler",
                        error_code=error.error_code
                    ):
                        logger.warning("Error recovery attempt failed",
                                     strategy=action.strategy.value,
                                     error_message=error_msg,
                                     attempt_duration=recovery_record['duration'])
                    
            except Exception as recovery_error:
                logging_manager = get_logging_manager()
                logging_manager.log_error(recovery_error, "error_handler",
                                        strategy=action.strategy.value,
                                        original_error=error.error_code)
        
        logging_manager = get_logging_manager()
        logging_manager.log_error(
            Exception(f"All recovery strategies exhausted for {error.error_code}"),
            "error_handler",
            error_code=error.error_code,
            attempted_strategies=attempted_strategies,
            strategy_count=len(attempted_strategies)
        )
        return False, None, attempted_strategies
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts."""
        with self._lock:
            if not self._recovery_history:
                return {
                    'total_attempts': 0,
                    'success_rate': 0.0,
                    'strategy_stats': {},
                    'category_stats': {}
                }
            
            total_attempts = len(self._recovery_history)
            successful_attempts = sum(1 for record in self._recovery_history if record['success'])
            success_rate = (successful_attempts / total_attempts) * 100
            
            # Strategy statistics
            strategy_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
            category_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
            
            for record in self._recovery_history:
                strategy = record['strategy']
                category = record['error_category']
                
                strategy_stats[strategy]['attempts'] += 1
                category_stats[category]['attempts'] += 1
                
                if record['success']:
                    strategy_stats[strategy]['successes'] += 1
                    category_stats[category]['successes'] += 1
            
            # Calculate success rates
            for stats in strategy_stats.values():
                if stats['attempts'] > 0:
                    stats['success_rate'] = (stats['successes'] / stats['attempts']) * 100
                else:
                    stats['success_rate'] = 0.0
            
            for stats in category_stats.values():
                if stats['attempts'] > 0:
                    stats['success_rate'] = (stats['successes'] / stats['attempts']) * 100
                else:
                    stats['success_rate'] = 0.0
            
            return {
                'total_attempts': total_attempts,
                'successful_attempts': successful_attempts,
                'success_rate': success_rate,
                'strategy_stats': dict(strategy_stats),
                'category_stats': dict(category_stats)
            }
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        
        # Connection recovery strategies
        def reset_connection(context: Dict[str, Any]) -> str:
            """Reset database connection."""
            database_name = context.get('database_name')
            if not database_name:
                raise ValueError("Database name required for connection reset")
            
            # This would integrate with ConnectionManager
            # For now, return a placeholder message
            return f"Connection reset initiated for database: {database_name}"
        
        self.register_strategy(
            ErrorCategory.CONNECTION,
            RecoveryAction(
                strategy=RecoveryStrategy.CONNECTION_RESET,
                description="Reset database connection and retry",
                action_function=reset_connection,
                success_probability=0.7,
                estimated_time=5.0
            )
        )
        
        # Query simplification for resource exhaustion
        def simplify_query(context: Dict[str, Any]) -> str:
            """Suggest query simplification."""
            query = context.get('query', '')
            suggestions = []
            
            if 'ORDER BY' in query.upper():
                suggestions.append("Remove or simplify ORDER BY clause")
            if 'GROUP BY' in query.upper():
                suggestions.append("Consider pre-aggregated tables")
            if 'JOIN' in query.upper():
                suggestions.append("Reduce number of JOINs or use EXISTS instead")
            if 'DISTINCT' in query.upper():
                suggestions.append("Remove DISTINCT if possible")
            
            if not suggestions:
                suggestions = ["Add LIMIT clause to reduce result set size"]
            
            return f"Query simplification suggestions: {'; '.join(suggestions)}"
        
        self.register_strategy(
            ErrorCategory.RESOURCE_EXHAUSTION,
            RecoveryAction(
                strategy=RecoveryStrategy.QUERY_SIMPLIFICATION,
                description="Suggest query optimizations to reduce resource usage",
                action_function=simplify_query,
                success_probability=0.6,
                estimated_time=0.1
            )
        )
        
        # Result pagination for large datasets
        def suggest_pagination(context: Dict[str, Any]) -> Dict[str, Any]:
            """Suggest result set pagination."""
            query = context.get('query', '')
            
            # Simple heuristic for pagination
            suggested_limit = 1000
            if 'LIMIT' not in query.upper():
                pagination_query = f"{query.rstrip(';')} LIMIT {suggested_limit}"
            else:
                pagination_query = query  # Already has LIMIT
            
            return {
                'strategy': 'pagination',
                'suggested_query': pagination_query,
                'suggested_limit': suggested_limit,
                'message': f"Consider paginating results with LIMIT {suggested_limit}"
            }
        
        self.register_strategy(
            ErrorCategory.RESOURCE_EXHAUSTION,
            RecoveryAction(
                strategy=RecoveryStrategy.RESULT_PAGINATION,
                description="Suggest result pagination to reduce memory usage",
                action_function=suggest_pagination,
                success_probability=0.8,
                estimated_time=0.1
            )
        )
        
        # Partial results for timeout errors
        def partial_results_strategy(context: Dict[str, Any]) -> Dict[str, Any]:
            """Suggest partial results approach."""
            return {
                'strategy': 'partial_results',
                'message': 'Consider using streaming execution or reducing query scope',
                'suggestions': [
                    'Use streaming query execution for large results',
                    'Add time-based filters to reduce data scope',
                    'Consider pre-computed aggregations',
                    'Use sampling for approximate results'
                ]
            }
        
        self.register_strategy(
            ErrorCategory.TIMEOUT,
            RecoveryAction(
                strategy=RecoveryStrategy.PARTIAL_RESULTS,
                description="Suggest partial results approaches for timeout issues",
                action_function=partial_results_strategy,
                success_probability=0.5,
                estimated_time=0.1
            )
        )


# ============================================================================
# Enhanced Error Logging and Monitoring
# ============================================================================

@dataclass
class ErrorMetrics:
    """Metrics for error monitoring and analysis."""
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_database: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_rate_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    first_error_time: Optional[float] = None
    last_error_time: Optional[float] = None


class ErrorLogger:
    """Enhanced error logging with structured metadata and monitoring."""
    
    def __init__(self, logger_name: str = "localdata.error_handler"):
        self.logger = logging.getLogger(logger_name)
        self.metrics = ErrorMetrics()
        self.lock = threading.RLock()
        
        # Setup structured logging format
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(error_code)s] %(message)s',
                defaults={'error_code': 'N/A'}
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_error(self, error: LocalDataError, extra_context: Optional[Dict[str, Any]] = None):
        """Log an error with full context and metadata."""
        current_time = time.time()
        
        with self.lock:
            # Update metrics
            self.metrics.total_errors += 1
            self.metrics.errors_by_category[error.category] += 1
            self.metrics.errors_by_severity[error.severity] += 1
            
            if error.database_name:
                self.metrics.errors_by_database[error.database_name] += 1
            
            # Track timing
            if self.metrics.first_error_time is None:
                self.metrics.first_error_time = current_time
            self.metrics.last_error_time = current_time
            
            # Add to recent errors
            error_record = {
                'timestamp': current_time,
                'error_code': error.error_code,
                'category': error.category.value,
                'severity': error.severity.value,
                'message': error.message,
                'database_name': error.database_name,
                'query_snippet': error.query[:100] if error.query else None
            }
            self.metrics.recent_errors.append(error_record)
            
            # Track error rate (errors per minute)
            minute_timestamp = int(current_time // 60)
            if not self.metrics.error_rate_per_minute or self.metrics.error_rate_per_minute[-1][0] != minute_timestamp:
                self.metrics.error_rate_per_minute.append([minute_timestamp, 1])
            else:
                self.metrics.error_rate_per_minute[-1][1] += 1
        
        # Create logging context
        log_context = {
            'error_code': error.error_code,
            'error_category': error.category.value,
            'error_severity': error.severity.value,
            'database_name': error.database_name,
            'error_metadata': error.metadata
        }
        
        if extra_context:
            log_context.update(extra_context)
        
        # Choose log level based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif error.severity == ErrorSeverity.HIGH:
            log_level = logging.ERROR
        elif error.severity == ErrorSeverity.MEDIUM:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        # Log the error
        self.logger.log(
            log_level,
            error.message,
            extra=log_context
        )
        
        # Log recovery suggestions if available
        if error.recovery_suggestions:
            self.logger.info(
                f"Recovery suggestions for {error.error_code}: {'; '.join(error.recovery_suggestions)}",
                extra=log_context
            )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self.lock:
            current_time = time.time()
            
            # Calculate error rates
            recent_error_rate = 0.0
            if self.metrics.error_rate_per_minute:
                # Average errors per minute over last 5 minutes
                recent_minutes = [
                    rate for timestamp, rate in self.metrics.error_rate_per_minute
                    if current_time - (timestamp * 60) <= 300  # 5 minutes
                ]
                if recent_minutes:
                    recent_error_rate = sum(recent_minutes) / len(recent_minutes)
            
            # Time-based statistics
            uptime = current_time - self.metrics.first_error_time if self.metrics.first_error_time else 0
            overall_error_rate = self.metrics.total_errors / (uptime / 60) if uptime > 0 else 0
            
            return {
                'total_errors': self.metrics.total_errors,
                'error_rate_per_minute': recent_error_rate,
                'overall_error_rate_per_minute': overall_error_rate,
                'uptime_minutes': uptime / 60,
                'errors_by_category': dict(self.metrics.errors_by_category),
                'errors_by_severity': dict(self.metrics.errors_by_severity),
                'errors_by_database': dict(self.metrics.errors_by_database),
                'first_error_time': self.metrics.first_error_time,
                'last_error_time': self.metrics.last_error_time,
                'recent_errors': list(self.metrics.recent_errors)[-10:]  # Last 10 errors
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status based on error patterns."""
        stats = self.get_error_statistics()
        
        # Health scoring based on error rates and severity
        health_score = 100.0
        
        # Deduct points for error rate
        if stats['error_rate_per_minute'] > 10:
            health_score -= 30
        elif stats['error_rate_per_minute'] > 5:
            health_score -= 15
        elif stats['error_rate_per_minute'] > 1:
            health_score -= 5
        
        # Deduct points for critical/high severity errors
        critical_errors = stats['errors_by_severity'].get(ErrorSeverity.CRITICAL.value, 0)
        high_errors = stats['errors_by_severity'].get(ErrorSeverity.HIGH.value, 0)
        
        health_score -= critical_errors * 10
        health_score -= high_errors * 5
        
        health_score = max(0, health_score)
        
        # Determine health status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "degraded"
        elif health_score >= 50:
            status = "unhealthy"
        else:
            status = "critical"
        
        return {
            'status': status,
            'health_score': health_score,
            'error_rate': stats['error_rate_per_minute'],
            'critical_errors': critical_errors,
            'high_errors': high_errors,
            'recommendations': self._get_health_recommendations(stats)
        }
    
    def _get_health_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on error patterns."""
        recommendations = []
        
        if stats['error_rate_per_minute'] > 5:
            recommendations.append("High error rate detected - investigate root causes")
        
        if stats['errors_by_severity'].get(ErrorSeverity.CRITICAL.value, 0) > 0:
            recommendations.append("Critical errors present - immediate attention required")
        
        # Database-specific recommendations
        db_errors = stats['errors_by_database']
        if db_errors:
            top_db = max(db_errors.items(), key=lambda x: x[1])
            if top_db[1] > stats['total_errors'] * 0.5:
                recommendations.append(f"Database '{top_db[0]}' has high error rate - check connectivity")
        
        # Category-specific recommendations
        category_errors = stats['errors_by_category']
        if category_errors.get(ErrorCategory.CONNECTION.value, 0) > stats['total_errors'] * 0.3:
            recommendations.append("Frequent connection errors - check network and database status")
        
        if category_errors.get(ErrorCategory.TIMEOUT.value, 0) > stats['total_errors'] * 0.3:
            recommendations.append("Frequent timeout errors - consider query optimization")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations


# ============================================================================
# Integration Utilities and Main Error Handler
# ============================================================================

class ErrorHandler:
    """Main error handler that coordinates all error handling components."""
    
    def __init__(self):
        self.logger = ErrorLogger()
        self.recovery_manager = ErrorRecoveryManager()
        self.circuit_breaker_registry = CircuitBreakerRegistry()
        self.lock = threading.RLock()
        
        # Default retry policies for different error types
        self.retry_policies = {
            ErrorCategory.CONNECTION: RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                base_delay=1.0,
                max_delay=30.0,
                retry_on=(DatabaseConnectionError,),
                stop_on=(SecurityViolationError, ConfigurationError)
            ),
            ErrorCategory.TIMEOUT: RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=2,
                base_delay=5.0,
                max_delay=60.0,
                retry_on=(QueryTimeoutError,)
            ),
            ErrorCategory.RESOURCE_EXHAUSTION: RetryPolicy(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                max_attempts=2,
                base_delay=10.0,
                max_delay=120.0,
                retry_on=(ResourceExhaustionError,)
            )
        }
    
    def handle_error(self, 
                    error: Exception, 
                    context: Optional[Dict[str, Any]] = None,
                    attempt_recovery: bool = True) -> Tuple[bool, Any, Optional[LocalDataError]]:
        """Main error handling entry point.
        
        Args:
            error: The exception that occurred
            context: Additional context for error handling
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Tuple[bool, Any, Optional[LocalDataError]]: (recovered, result, processed_error)
        """
        context = context or {}
        
        # Convert to LocalDataError if needed
        if isinstance(error, LocalDataError):
            processed_error = error
        else:
            processed_error = self._convert_to_localdata_error(error, context)
        
        # Log the error
        self.logger.log_error(processed_error, context)
        
        # Attempt recovery if requested
        recovery_result = None
        if attempt_recovery:
            recovered, recovery_result, attempted_strategies = self.recovery_manager.attempt_recovery(
                processed_error, context
            )
            
            if recovered:
                self.logger.logger.info(f"Error recovery successful for {processed_error.error_code} using strategies: {attempted_strategies}")
                return True, recovery_result, processed_error
            else:
                self.logger.logger.warning(f"Error recovery failed for {processed_error.error_code} after attempting: {attempted_strategies}")
        
        return False, None, processed_error
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get a circuit breaker for a specific resource."""
        return self.circuit_breaker_registry.get_breaker(name, config)
    
    def get_retry_policy(self, category: ErrorCategory) -> Optional[RetryPolicy]:
        """Get retry policy for an error category."""
        return self.retry_policies.get(category)
    
    def register_retry_policy(self, category: ErrorCategory, policy: RetryPolicy):
        """Register a retry policy for an error category."""
        with self.lock:
            self.retry_policies[category] = policy
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        error_health = self.logger.get_health_status()
        circuit_breaker_status = self.circuit_breaker_registry.get_status_summary()
        recovery_stats = self.recovery_manager.get_recovery_statistics()
        error_stats = self.logger.get_error_statistics()
        
        return {
            'overall_health': error_health,
            'circuit_breakers': circuit_breaker_status,
            'recovery_statistics': recovery_stats,
            'error_statistics': error_stats,
            'timestamp': time.time()
        }
    
    def _convert_to_localdata_error(self, error: Exception, context: Dict[str, Any]) -> LocalDataError:
        """Convert a generic exception to a LocalDataError."""
        error_message = str(error)
        error_type = type(error).__name__
        
        # Determine category based on error type and message
        if any(keyword in error_message.lower() for keyword in ['connection', 'network', 'host', 'port']):
            category = ErrorCategory.CONNECTION
            error_class = DatabaseConnectionError
        elif any(keyword in error_message.lower() for keyword in ['timeout', 'time', 'expired']):
            category = ErrorCategory.TIMEOUT
            error_class = QueryTimeoutError
        elif any(keyword in error_message.lower() for keyword in ['memory', 'resource', 'limit']):
            category = ErrorCategory.RESOURCE_EXHAUSTION
            error_class = ResourceExhaustionError
        elif any(keyword in error_message.lower() for keyword in ['security', 'permission', 'access']):
            category = ErrorCategory.SECURITY_VIOLATION
            error_class = SecurityViolationError
        elif any(keyword in error_message.lower() for keyword in ['config', 'setting', 'parameter']):
            category = ErrorCategory.CONFIGURATION
            error_class = ConfigurationError
        else:
            category = ErrorCategory.QUERY_EXECUTION
            error_class = QueryExecutionError
        
        # Determine severity
        severity = ErrorSeverity.MEDIUM
        if any(keyword in error_message.lower() for keyword in ['critical', 'fatal', 'severe']):
            severity = ErrorSeverity.CRITICAL
        elif any(keyword in error_message.lower() for keyword in ['error', 'fail', 'exception']):
            severity = ErrorSeverity.HIGH
        elif any(keyword in error_message.lower() for keyword in ['warning', 'warn']):
            severity = ErrorSeverity.LOW
        
        return error_class(
            message=f"{error_type}: {error_message}",
            database_name=context.get('database_name'),
            query=context.get('query'),
            metadata={
                'original_error_type': error_type,
                'context': context
            },
            cause=error,
            severity=severity
        )


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def initialize_error_handler() -> ErrorHandler:
    """Initialize a new global error handler instance."""
    global _error_handler
    _error_handler = ErrorHandler()
    return _error_handler


# Convenience functions for common operations
def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any, Optional[LocalDataError]]:
    """Handle an error using the global error handler."""
    return get_error_handler().handle_error(error, context)


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get a circuit breaker using the global error handler."""
    return get_error_handler().get_circuit_breaker(name, config)