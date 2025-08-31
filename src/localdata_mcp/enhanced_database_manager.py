"""Enhanced Database Manager with Advanced Error Handling Integration.

This module provides enhanced database management that integrates the comprehensive
error handling system with existing components (SecurityManager, ConnectionManager,
TimeoutManager, StreamingExecutor) to provide robust, fault-tolerant database operations.

Key Features:
- Seamless integration with error handling system
- Circuit breaker protection for database operations
- Automatic retry with exponential backoff
- Error recovery strategies and graceful degradation
- Comprehensive monitoring and health checks
- Enhanced logging and error tracking
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

from .error_handler import (
    ErrorHandler, LocalDataError, DatabaseConnectionError, QueryExecutionError,
    QueryTimeoutError, ResourceExhaustionError, SecurityViolationError,
    CircuitBreaker, CircuitBreakerConfig, RetryPolicy, RetryStrategy,
    circuit_breaker_protection, retry_on_failure, get_error_handler
)

from .security_manager import get_security_manager, SecurityManager
from .connection_manager import get_enhanced_connection_manager, EnhancedConnectionManager
from .timeout_manager import get_timeout_manager, QueryTimeoutManager, QueryTimeoutError as BaseTimeoutError
from .streaming_executor import StreamingQueryExecutor

logger = logging.getLogger(__name__)


class EnhancedDatabaseManager:
    """Enhanced database manager with comprehensive error handling integration."""
    
    def __init__(self):
        """Initialize the enhanced database manager."""
        # Core components
        self.error_handler = get_error_handler()
        self.security_manager = get_security_manager()
        self.connection_manager = get_enhanced_connection_manager()
        self.timeout_manager = get_timeout_manager()
        self.streaming_executor = StreamingQueryExecutor()
        
        # Circuit breakers for different database operations
        self._setup_circuit_breakers()
        
        # Enhanced retry policies
        self._setup_retry_policies()
        
        logger.info("Enhanced Database Manager initialized with comprehensive error handling")
    
    def _setup_circuit_breakers(self):
        """Setup circuit breakers for different types of operations."""
        
        # General database connection circuit breaker
        self.connection_breaker = self.error_handler.get_circuit_breaker(
            "database_connections",
            CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout_duration=60.0,
                failure_exceptions=(DatabaseConnectionError, ResourceExhaustionError)
            )
        )
        
        # Query execution circuit breaker
        self.query_breaker = self.error_handler.get_circuit_breaker(
            "query_execution", 
            CircuitBreakerConfig(
                failure_threshold=10,
                success_threshold=5,
                timeout_duration=30.0,
                failure_exceptions=(QueryExecutionError, QueryTimeoutError)
            )
        )
        
        # Security validation circuit breaker
        self.security_breaker = self.error_handler.get_circuit_breaker(
            "security_validation",
            CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout_duration=120.0,
                failure_exceptions=(SecurityViolationError,)
            )
        )
    
    def _setup_retry_policies(self):
        """Setup enhanced retry policies for different operations."""
        
        # Connection retry policy
        connection_policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=3,
            base_delay=2.0,
            max_delay=30.0,
            backoff_multiplier=2.0,
            jitter=True,
            retry_on=(DatabaseConnectionError,),
            stop_on=(SecurityViolationError,)
        )
        self.error_handler.register_retry_policy('connection', connection_policy)
        
        # Query execution retry policy
        query_policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=2,
            base_delay=1.0,
            max_delay=10.0,
            retry_on=(QueryExecutionError,),
            stop_on=(SecurityViolationError, QueryTimeoutError)
        )
        self.error_handler.register_retry_policy('query_execution', query_policy)
    
    @contextmanager
    def enhanced_query_execution(self, database_name: str, query: str, connection_id: Optional[str] = None):
        """Context manager for enhanced query execution with comprehensive error handling."""
        context = {
            'database_name': database_name,
            'query': query,
            'connection_id': connection_id,
            'operation_type': 'query_execution'
        }
        
        try:
            # Security validation with circuit breaker protection
            with circuit_breaker_protection(self.security_breaker, "security_validation"):
                is_valid, error_msg, security_metadata = self.security_manager.validate_query_security(
                    query, database_name, connection_id
                )
                
                if not is_valid:
                    raise SecurityViolationError(
                        message=error_msg,
                        query=query,
                        database_name=database_name,
                        metadata=security_metadata
                    )
            
            # Connection management with circuit breaker protection  
            with circuit_breaker_protection(self.connection_breaker, "database_connection"):
                engine = self.connection_manager.get_engine(database_name)
                if engine is None:
                    raise DatabaseConnectionError(
                        message=f"Unable to get database engine for '{database_name}'",
                        database_name=database_name
                    )
            
            # Query execution context
            execution_context = {
                'engine': engine,
                'database_name': database_name,
                'query': query,
                'connection_id': connection_id,
                'security_metadata': security_metadata
            }
            
            yield execution_context
            
        except Exception as e:
            # Handle and potentially recover from errors
            recovered, result, processed_error = self.error_handler.handle_error(e, context)
            
            if not recovered:
                # Error could not be recovered, re-raise the processed error
                if processed_error:
                    raise processed_error
                else:
                    raise
            else:
                # Error was recovered, update context with recovery result
                execution_context['recovery_result'] = result
                yield execution_context
    
    def execute_query_with_error_handling(self, database_name: str, query: str, 
                                         connection_id: Optional[str] = None,
                                         enable_analysis: bool = True,
                                         chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """Execute a query with comprehensive error handling and recovery.
        
        Args:
            database_name: Name of the database to query
            query: SQL query to execute
            connection_id: Optional connection identifier
            enable_analysis: Whether to perform pre-query analysis
            chunk_size: Optional chunk size for result pagination
            
        Returns:
            Dict containing query results, metadata, and error handling information
        """
        
        @retry_on_failure(
            policy=RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                base_delay=1.0,
                retry_on=(DatabaseConnectionError, QueryExecutionError),
                stop_on=(SecurityViolationError,)
            ),
            operation_name=f"query_execution_{database_name}"
        )
        def _execute_with_retry():
            with self.enhanced_query_execution(database_name, query, connection_id) as context:
                
                # Query execution with circuit breaker protection
                with circuit_breaker_protection(self.query_breaker, "query_execution"):
                    
                    # Use timeout management
                    with self.timeout_manager.managed_timeout(
                        operation_id=f"query_{int(time.time())}",
                        database_name=database_name,
                        timeout_seconds=self.timeout_manager.get_query_timeout(database_name)
                    ):
                        
                        # Execute query using streaming executor
                        try:
                            # Create streaming source
                            from .streaming_executor import create_streaming_source
                            streaming_source = create_streaming_source(
                                engine=context['engine'],
                                query=query
                            )
                            
                            # Execute with streaming
                            query_id = f"{database_name}_{int(time.time() * 1000)}"
                            initial_chunk_size = chunk_size or 100
                            
                            first_chunk, streaming_metadata = self.streaming_executor.execute_streaming(
                                streaming_source,
                                query_id,
                                initial_chunk_size,
                                database_name=database_name
                            )
                            
                            # Build response with error handling metadata
                            response = {
                                'success': True,
                                'data': first_chunk.to_dict('records') if not first_chunk.empty else [],
                                'metadata': {
                                    'query_id': query_id,
                                    'database_name': database_name,
                                    'rows_returned': len(first_chunk),
                                    'streaming_metadata': streaming_metadata,
                                    'security_metadata': context.get('security_metadata', {}),
                                    'error_handling': {
                                        'circuit_breakers_status': {
                                            'connection': self.connection_breaker.get_state_info(),
                                            'query': self.query_breaker.get_state_info(),
                                            'security': self.security_breaker.get_state_info()
                                        }
                                    }
                                }
                            }
                            
                            # Add pagination info if more data available
                            if streaming_metadata.get('has_more_data', False):
                                response['pagination'] = {
                                    'query_id': query_id,
                                    'next_chunk_available': True,
                                    'use_next_chunk_tool': True
                                }
                            
                            return response
                            
                        except BaseTimeoutError as timeout_error:
                            # Convert timeout error to our custom timeout error
                            raise QueryTimeoutError(
                                message=f"Query timed out: {timeout_error}",
                                execution_time=getattr(timeout_error, 'execution_time', 0.0),
                                timeout_limit=self.timeout_manager.get_query_timeout(database_name),
                                query=query,
                                database_name=database_name
                            )
                        
                        except Exception as exec_error:
                            # Wrap execution errors
                            raise QueryExecutionError(
                                message=f"Query execution failed: {exec_error}",
                                query=query,
                                database_name=database_name,
                                cause=exec_error
                            )
        
        try:
            return _execute_with_retry()
            
        except Exception as e:
            # Final error handling - create error response
            recovered, recovery_result, processed_error = self.error_handler.handle_error(
                e, {
                    'database_name': database_name,
                    'query': query,
                    'connection_id': connection_id,
                    'operation': 'execute_query'
                }
            )
            
            error_response = {
                'success': False,
                'error': {
                    'type': type(processed_error).__name__ if processed_error else type(e).__name__,
                    'message': str(processed_error) if processed_error else str(e),
                    'category': processed_error.category.value if processed_error else 'unknown',
                    'severity': processed_error.severity.value if processed_error else 'medium',
                    'recovery_attempted': recovered,
                    'recovery_suggestions': processed_error.recovery_suggestions if processed_error else []
                },
                'metadata': {
                    'database_name': database_name,
                    'query_snippet': query[:200] if query else None,
                    'timestamp': time.time(),
                    'error_handling': {
                        'circuit_breakers_status': {
                            'connection': self.connection_breaker.get_state_info(),
                            'query': self.query_breaker.get_state_info(),
                            'security': self.security_breaker.get_state_info()
                        }
                    }
                }
            }
            
            if recovery_result:
                error_response['recovery_result'] = recovery_result
            
            return error_response
    
    def connect_database_with_error_handling(self, name: str, db_type: str, 
                                           conn_string: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """Connect to a database with comprehensive error handling.
        
        Args:
            name: Database connection name
            db_type: Type of database
            conn_string: Connection string or file path
            sheet_name: Optional sheet name for file formats
            
        Returns:
            Dict containing connection result and error handling metadata
        """
        
        @retry_on_failure(
            policy=RetryPolicy(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                base_delay=2.0,
                retry_on=(DatabaseConnectionError,),
                stop_on=(SecurityViolationError,)
            ),
            operation_name=f"database_connection_{name}"
        )
        def _connect_with_retry():
            with circuit_breaker_protection(self.connection_breaker, f"connect_{name}"):
                # Initialize database connection
                success = self.connection_manager.initialize_database(name)
                if not success:
                    raise DatabaseConnectionError(
                        message=f"Failed to initialize database connection '{name}'",
                        database_name=name
                    )
                
                # Get connection info
                connection_info = self.connection_manager.get_connection_info(name)
                if not connection_info:
                    raise DatabaseConnectionError(
                        message=f"Failed to retrieve connection info for '{name}'",
                        database_name=name
                    )
                
                return {
                    'success': True,
                    'connection_info': connection_info,
                    'metadata': {
                        'database_name': name,
                        'db_type': db_type,
                        'timestamp': time.time(),
                        'error_handling': {
                            'circuit_breaker_status': self.connection_breaker.get_state_info()
                        }
                    }
                }
        
        try:
            return _connect_with_retry()
            
        except Exception as e:
            # Handle connection errors
            recovered, recovery_result, processed_error = self.error_handler.handle_error(
                e, {
                    'database_name': name,
                    'db_type': db_type,
                    'conn_string': conn_string,
                    'operation': 'connect_database'
                }
            )
            
            return {
                'success': False,
                'error': {
                    'type': type(processed_error).__name__ if processed_error else type(e).__name__,
                    'message': str(processed_error) if processed_error else str(e),
                    'category': processed_error.category.value if processed_error else 'connection',
                    'severity': processed_error.severity.value if processed_error else 'high',
                    'recovery_attempted': recovered,
                    'recovery_suggestions': processed_error.recovery_suggestions if processed_error else []
                },
                'metadata': {
                    'database_name': name,
                    'db_type': db_type,
                    'timestamp': time.time(),
                    'error_handling': {
                        'circuit_breaker_status': self.connection_breaker.get_state_info()
                    }
                }
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the enhanced database manager."""
        try:
            # Get health from all components
            error_handler_health = self.error_handler.get_system_health()
            
            # Get circuit breaker status
            circuit_breaker_status = {
                'connection': self.connection_breaker.get_state_info(),
                'query': self.query_breaker.get_state_info(),
                'security': self.security_breaker.get_state_info()
            }
            
            # Get connection manager status
            connection_databases = self.connection_manager.list_databases()
            healthy_connections = sum(
                1 for db in connection_databases 
                if db.get('health', {}).get('is_healthy', False)
            )
            
            # Calculate overall health score
            total_databases = len(connection_databases)
            connection_health_rate = (healthy_connections / total_databases * 100) if total_databases > 0 else 100
            
            # Check circuit breaker states
            open_breakers = sum(
                1 for breaker_info in circuit_breaker_status.values()
                if breaker_info['state'] == 'open'
            )
            
            overall_health_score = error_handler_health['overall_health']['health_score']
            
            # Adjust score based on circuit breakers and connections
            if open_breakers > 0:
                overall_health_score -= open_breakers * 15
            
            overall_health_score = (overall_health_score + connection_health_rate) / 2
            overall_health_score = max(0, min(100, overall_health_score))
            
            # Determine overall status
            if overall_health_score >= 90:
                overall_status = "healthy"
            elif overall_health_score >= 70:
                overall_status = "degraded"
            elif overall_health_score >= 50:
                overall_status = "unhealthy"  
            else:
                overall_status = "critical"
            
            return {
                'overall_status': overall_status,
                'overall_health_score': overall_health_score,
                'components': {
                    'error_handler': error_handler_health,
                    'circuit_breakers': circuit_breaker_status,
                    'connection_manager': {
                        'total_databases': total_databases,
                        'healthy_connections': healthy_connections,
                        'health_rate': connection_health_rate,
                        'databases': connection_databases
                    }
                },
                'recommendations': self._get_health_recommendations(
                    overall_status, circuit_breaker_status, error_handler_health
                ),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                'overall_status': 'unknown',
                'overall_health_score': 0,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _get_health_recommendations(self, overall_status: str, circuit_breakers: Dict[str, Any], 
                                  error_health: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on system status."""
        recommendations = []
        
        # Overall status recommendations
        if overall_status == 'critical':
            recommendations.append("System in critical state - immediate attention required")
        elif overall_status == 'unhealthy':
            recommendations.append("System unhealthy - investigate error patterns and connectivity")
        elif overall_status == 'degraded':
            recommendations.append("System degraded - monitor for improvement or investigate issues")
        
        # Circuit breaker recommendations
        open_breakers = [name for name, info in circuit_breakers.items() if info['state'] == 'open']
        if open_breakers:
            recommendations.append(f"Circuit breakers open: {', '.join(open_breakers)} - check underlying services")
        
        # Error handler recommendations
        error_recommendations = error_health.get('overall_health', {}).get('recommendations', [])
        recommendations.extend(error_recommendations)
        
        if not recommendations:
            recommendations.append("System operating normally")
        
        return recommendations
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics from all components."""
        return self.error_handler.get_system_health()
    
    def reset_circuit_breakers(self) -> Dict[str, Any]:
        """Reset all circuit breakers - use with caution."""
        try:
            self.connection_breaker.reset()
            self.query_breaker.reset()  
            self.security_breaker.reset()
            
            return {
                'success': True,
                'message': 'All circuit breakers have been reset',
                'timestamp': time.time(),
                'circuit_breakers': {
                    'connection': self.connection_breaker.get_state_info(),
                    'query': self.query_breaker.get_state_info(),
                    'security': self.security_breaker.get_state_info()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }


# Global enhanced database manager instance
_enhanced_database_manager: Optional[EnhancedDatabaseManager] = None


def get_enhanced_database_manager() -> EnhancedDatabaseManager:
    """Get or create global enhanced database manager instance."""
    global _enhanced_database_manager
    if _enhanced_database_manager is None:
        _enhanced_database_manager = EnhancedDatabaseManager()
    return _enhanced_database_manager


def initialize_enhanced_database_manager() -> EnhancedDatabaseManager:
    """Initialize a new global enhanced database manager instance."""
    global _enhanced_database_manager
    _enhanced_database_manager = EnhancedDatabaseManager()
    return _enhanced_database_manager