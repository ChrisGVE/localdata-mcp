"""Advanced Query Timeout System for LocalData MCP.

Provides comprehensive timeout management with per-database configuration,
graceful cancellation, and proper resource cleanup. Integrates with the 
existing configuration system and streaming pipeline.
"""

import asyncio
import logging
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

from sqlalchemy.engine import Engine, Connection

# Import configuration system for timeout settings
from .config_manager import get_config_manager, DatabaseConfig, DatabaseType

logger = logging.getLogger(__name__)


class TimeoutReason(Enum):
    """Reasons for query timeout."""
    USER_TIMEOUT = "user_timeout"          # User-specified timeout exceeded
    DATABASE_TIMEOUT = "database_timeout"  # Database-specific timeout exceeded
    GLOBAL_TIMEOUT = "global_timeout"      # Global system timeout exceeded
    MEMORY_PRESSURE = "memory_pressure"    # Cancelled due to memory pressure
    MANUAL_CANCEL = "manual_cancel"        # Manually cancelled by user


@dataclass
class TimeoutConfig:
    """Timeout configuration for a specific operation."""
    query_timeout: int                      # Query execution timeout in seconds
    connection_timeout: int                 # Connection timeout in seconds
    database_name: str                      # Database identifier
    database_type: DatabaseType             # Type of database
    allow_cancellation: bool = True         # Whether cancellation is supported
    cleanup_on_timeout: bool = True         # Whether to cleanup resources on timeout
    timeout_buffer: int = 5                 # Grace period for cleanup (seconds)


@dataclass
class TimeoutResult:
    """Result of a timeout-managed operation."""
    timed_out: bool                         # Whether operation timed out
    timeout_reason: Optional[TimeoutReason] # Reason for timeout
    execution_time: float                   # Actual execution time in seconds
    cleanup_performed: bool                 # Whether cleanup was performed
    error_message: Optional[str] = None     # Error message if any


class QueryTimeoutError(Exception):
    """Exception raised when a query times out."""
    
    def __init__(self, message: str, timeout_reason: TimeoutReason, 
                 execution_time: float, database_name: str):
        super().__init__(message)
        self.timeout_reason = timeout_reason
        self.execution_time = execution_time
        self.database_name = database_name


class QueryTimeoutManager:
    """Advanced timeout management system for database queries."""
    
    def __init__(self):
        """Initialize the timeout manager."""
        self._config_manager = get_config_manager()
        self._active_operations: Dict[str, Dict[str, Any]] = {}
        self._operation_lock = threading.Lock()
        
    def get_timeout_config(self, database_name: str) -> TimeoutConfig:
        """Get timeout configuration for a specific database.
        
        Args:
            database_name: Name of the database
            
        Returns:
            TimeoutConfig: Database-specific timeout configuration
            
        Raises:
            ValueError: If database configuration is not found
        """
        db_config = self._config_manager.get_database_config(database_name)
        if not db_config:
            # Use performance config defaults if database config not found
            perf_config = self._config_manager.get_performance_config()
            logger.warning(f"Database '{database_name}' not found in config, using defaults")
            return TimeoutConfig(
                query_timeout=300,  # 5 minutes default
                connection_timeout=30,  # 30 seconds default
                database_name=database_name,
                database_type=DatabaseType.SQLITE,  # Default type
                allow_cancellation=True,
                cleanup_on_timeout=True
            )
            
        return TimeoutConfig(
            query_timeout=db_config.query_timeout,
            connection_timeout=db_config.connection_timeout,
            database_name=database_name,
            database_type=db_config.type,
            allow_cancellation=self._supports_cancellation(db_config.type),
            cleanup_on_timeout=True
        )
    
    def _supports_cancellation(self, db_type: DatabaseType) -> bool:
        """Check if database type supports query cancellation.
        
        Args:
            db_type: Database type to check
            
        Returns:
            bool: Whether cancellation is supported
        """
        # File-based databases generally don't support cancellation well
        file_based_types = {
            DatabaseType.CSV, DatabaseType.JSON, DatabaseType.YAML,
            DatabaseType.TOML, DatabaseType.EXCEL, DatabaseType.ODS,
            DatabaseType.NUMBERS, DatabaseType.XML, DatabaseType.INI,
            DatabaseType.TSV, DatabaseType.PARQUET, DatabaseType.FEATHER,
            DatabaseType.ARROW, DatabaseType.HDF5
        }
        
        if db_type in file_based_types:
            return False
            
        # Network-based databases generally support cancellation
        network_based_types = {
            DatabaseType.POSTGRESQL, DatabaseType.MYSQL,
            DatabaseType.REDIS, DatabaseType.ELASTICSEARCH,
            DatabaseType.MONGODB, DatabaseType.INFLUXDB,
            DatabaseType.NEO4J, DatabaseType.COUCHDB
        }
        
        if db_type in network_based_types:
            return True
            
        # SQLite and DuckDB have limited cancellation support
        if db_type in {DatabaseType.SQLITE, DatabaseType.DUCKDB}:
            return True  # Limited support, but we'll try
            
        return False
    
    @contextmanager
    def timeout_context(self, operation_id: str, timeout_config: TimeoutConfig, 
                       cleanup_func: Optional[Callable[[], None]] = None):
        """Context manager for timeout-protected operations.
        
        Args:
            operation_id: Unique identifier for this operation
            timeout_config: Timeout configuration
            cleanup_func: Optional cleanup function to call on timeout
            
        Yields:
            dict: Operation context with cancellation support
            
        Raises:
            QueryTimeoutError: If operation times out
        """
        start_time = time.time()
        operation_context = {
            'cancelled': False,
            'start_time': start_time,
            'timeout_config': timeout_config,
            'cleanup_func': cleanup_func
        }
        
        # Register active operation
        with self._operation_lock:
            self._active_operations[operation_id] = operation_context
            
        # Set up timeout monitoring
        timeout_thread = None
        if timeout_config.query_timeout > 0:
            timeout_thread = threading.Thread(
                target=self._monitor_timeout,
                args=(operation_id, timeout_config.query_timeout),
                daemon=True
            )
            timeout_thread.start()
            logger.info(f"Started timeout monitoring for operation '{operation_id}' "
                       f"({timeout_config.query_timeout}s timeout)")
        
        try:
            yield operation_context
            
        except QueryTimeoutError:
            raise  # Re-raise timeout errors
            
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Operation '{operation_id}' failed with error: {e}")
            raise
            
        finally:
            # Clean up operation tracking
            execution_time = time.time() - start_time
            
            with self._operation_lock:
                if operation_id in self._active_operations:
                    op_context = self._active_operations[operation_id]
                    if op_context.get('cancelled'):
                        logger.warning(f"Operation '{operation_id}' was cancelled after {execution_time:.2f}s")
                    else:
                        logger.info(f"Operation '{operation_id}' completed in {execution_time:.2f}s")
                    
                    del self._active_operations[operation_id]
            
            # Wait for timeout thread to finish
            if timeout_thread and timeout_thread.is_alive():
                timeout_thread.join(timeout=1.0)
    
    def _monitor_timeout(self, operation_id: str, timeout_seconds: int) -> None:
        """Monitor operation timeout in background thread.
        
        Args:
            operation_id: Operation to monitor
            timeout_seconds: Timeout in seconds
        """
        time.sleep(timeout_seconds)
        
        with self._operation_lock:
            if operation_id not in self._active_operations:
                return  # Operation already completed
                
            operation_context = self._active_operations[operation_id]
            if operation_context.get('cancelled'):
                return  # Already cancelled
                
            # Mark as cancelled
            operation_context['cancelled'] = True
            timeout_config = operation_context['timeout_config']
            
            logger.warning(f"Operation '{operation_id}' timed out after {timeout_seconds}s "
                          f"(database: {timeout_config.database_name})")
            
            # Perform cleanup if available
            cleanup_func = operation_context.get('cleanup_func')
            if cleanup_func and timeout_config.cleanup_on_timeout:
                try:
                    logger.info(f"Performing cleanup for timed out operation '{operation_id}'")
                    cleanup_func()
                    operation_context['cleanup_performed'] = True
                except Exception as e:
                    logger.error(f"Cleanup failed for operation '{operation_id}': {e}")
                    operation_context['cleanup_performed'] = False
    
    def is_cancelled(self, operation_id: str) -> bool:
        """Check if an operation has been cancelled.
        
        Args:
            operation_id: Operation to check
            
        Returns:
            bool: Whether operation is cancelled
        """
        with self._operation_lock:
            if operation_id not in self._active_operations:
                return False
            return self._active_operations[operation_id].get('cancelled', False)
    
    def cancel_operation(self, operation_id: str, reason: TimeoutReason = TimeoutReason.MANUAL_CANCEL) -> bool:
        """Manually cancel an operation.
        
        Args:
            operation_id: Operation to cancel
            reason: Reason for cancellation
            
        Returns:
            bool: Whether cancellation was successful
        """
        with self._operation_lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Cannot cancel operation '{operation_id}': not found")
                return False
                
            operation_context = self._active_operations[operation_id]
            if operation_context.get('cancelled'):
                logger.info(f"Operation '{operation_id}' already cancelled")
                return True
                
            # Mark as cancelled
            operation_context['cancelled'] = True
            operation_context['cancel_reason'] = reason
            
            logger.info(f"Manually cancelled operation '{operation_id}' (reason: {reason.value})")
            
            # Perform cleanup if available
            timeout_config = operation_context['timeout_config']
            cleanup_func = operation_context.get('cleanup_func')
            if cleanup_func and timeout_config.cleanup_on_timeout:
                try:
                    cleanup_func()
                    operation_context['cleanup_performed'] = True
                except Exception as e:
                    logger.error(f"Cleanup failed for cancelled operation '{operation_id}': {e}")
                    operation_context['cleanup_performed'] = False
            
            return True
    
    def get_active_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently active operations.
        
        Returns:
            dict: Active operations with their details
        """
        with self._operation_lock:
            current_time = time.time()
            result = {}
            
            for op_id, context in self._active_operations.items():
                start_time = context['start_time']
                timeout_config = context['timeout_config']
                
                result[op_id] = {
                    'database_name': timeout_config.database_name,
                    'database_type': timeout_config.database_type.value,
                    'running_time': current_time - start_time,
                    'timeout_limit': timeout_config.query_timeout,
                    'time_remaining': max(0, timeout_config.query_timeout - (current_time - start_time)),
                    'cancelled': context.get('cancelled', False),
                    'supports_cancellation': timeout_config.allow_cancellation
                }
            
            return result
    
    def create_timeout_error(self, operation_id: str, 
                           timeout_reason: TimeoutReason = TimeoutReason.USER_TIMEOUT) -> QueryTimeoutError:
        """Create a timeout error for an operation.
        
        Args:
            operation_id: Operation that timed out
            timeout_reason: Reason for timeout
            
        Returns:
            QueryTimeoutError: Configured timeout error
        """
        execution_time = 0.0
        database_name = "unknown"
        
        with self._operation_lock:
            if operation_id in self._active_operations:
                context = self._active_operations[operation_id]
                execution_time = time.time() - context['start_time']
                database_name = context['timeout_config'].database_name
        
        if timeout_reason == TimeoutReason.USER_TIMEOUT:
            message = f"Query timed out after {execution_time:.1f}s (database: {database_name})"
        elif timeout_reason == TimeoutReason.DATABASE_TIMEOUT:
            message = f"Database-specific timeout exceeded after {execution_time:.1f}s (database: {database_name})"
        elif timeout_reason == TimeoutReason.MEMORY_PRESSURE:
            message = f"Query cancelled due to memory pressure after {execution_time:.1f}s (database: {database_name})"
        else:
            message = f"Query cancelled ({timeout_reason.value}) after {execution_time:.1f}s (database: {database_name})"
        
        return QueryTimeoutError(
            message=message,
            timeout_reason=timeout_reason,
            execution_time=execution_time,
            database_name=database_name
        )


# Global timeout manager instance
_timeout_manager: Optional[QueryTimeoutManager] = None


def get_timeout_manager() -> QueryTimeoutManager:
    """Get or create global timeout manager instance.
    
    Returns:
        QueryTimeoutManager: Global timeout manager
    """
    global _timeout_manager
    if _timeout_manager is None:
        _timeout_manager = QueryTimeoutManager()
    return _timeout_manager


def with_timeout(database_name: str, operation_id: Optional[str] = None,
                cleanup_func: Optional[Callable[[], None]] = None):
    """Decorator for adding timeout management to database operations.
    
    Args:
        database_name: Name of the database
        operation_id: Optional operation ID (generated if not provided)
        cleanup_func: Optional cleanup function
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            timeout_manager = get_timeout_manager()
            timeout_config = timeout_manager.get_timeout_config(database_name)
            
            op_id = operation_id or f"{database_name}_{func.__name__}_{int(time.time())}"
            
            with timeout_manager.timeout_context(op_id, timeout_config, cleanup_func) as context:
                # Check for cancellation periodically during execution
                def check_cancellation():
                    if timeout_manager.is_cancelled(op_id):
                        raise timeout_manager.create_timeout_error(op_id, TimeoutReason.USER_TIMEOUT)
                
                # Add cancellation check to context
                context['check_cancellation'] = check_cancellation
                
                return func(*args, **kwargs)
        
        return wrapper
    return decorator