"""Enhanced Database Tools using the EnhancedConnectionManager.

Provides improved database tools that integrate with the EnhancedConnectionManager
for better resource management, health monitoring, and configuration-driven operations.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .connection_manager import get_enhanced_connection_manager, EnhancedConnectionManager
from .config_manager import get_config_manager
from .query_parser import parse_and_validate_sql, SQLSecurityError
from .query_analyzer import analyze_query
from .streaming_executor import StreamingQueryExecutor, create_streaming_source
from .timeout_manager import get_timeout_manager, QueryTimeoutError
from .security_manager import get_security_manager, SecurityManager

logger = logging.getLogger(__name__)

# Create enhanced MCP tools
enhanced_mcp = FastMCP("localdata-enhanced")


class EnhancedDatabaseTools:
    """Enhanced database tools with improved connection management."""
    
    def __init__(self):
        """Initialize enhanced database tools."""
        self.connection_manager = get_enhanced_connection_manager()
        self.config_manager = get_config_manager()
        self.timeout_manager = get_timeout_manager()
        self.security_manager = get_security_manager()
        self.streaming_executor = StreamingQueryExecutor()
    
    def initialize_all_configured_databases(self) -> Dict[str, bool]:
        """Initialize all databases found in configuration.
        
        Returns:
            Dict[str, bool]: Database name to initialization success mapping
        """
        results = {}
        db_configs = self.config_manager.get_database_configs()
        
        for name, config in db_configs.items():
            if config.enabled:
                logger.info(f"Initializing database '{name}' (type: {config.type.value})")
                success = self.connection_manager.initialize_database(name, config)
                results[name] = success
                if success:
                    logger.info(f"Successfully initialized '{name}'")
                else:
                    logger.warning(f"Failed to initialize '{name}'")
            else:
                logger.info(f"Skipping disabled database '{name}'")
                results[name] = False
        
        return results
    
    def get_database_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive status for one or all databases.
        
        Args:
            name: Optional database name. If None, returns status for all databases.
            
        Returns:
            Dict[str, Any]: Database status information
        """
        if name:
            # Single database status
            conn_info = self.connection_manager.get_connection_info(name)
            if not conn_info:
                return {"error": f"Database '{name}' not found or not initialized"}
            
            # Add additional health information
            health_result = self.connection_manager.get_health_status(name)
            resource_status = self.connection_manager.get_resource_status(name)
            
            return {
                "database": conn_info,
                "detailed_health": {
                    "state": health_result.state.value if health_result else "unknown",
                    "is_healthy": health_result.is_healthy if health_result else False,
                    "response_time_ms": health_result.response_time_ms if health_result else 0,
                    "last_check": health_result.timestamp if health_result else 0,
                    "error_message": health_result.error_message if health_result else None
                },
                "resource_status": resource_status
            }
        else:
            # All databases status
            all_configs = self.config_manager.get_database_configs()
            status = {
                "total_configured": len(all_configs),
                "databases": {}
            }
            
            initialized_count = 0
            healthy_count = 0
            
            for db_name in all_configs.keys():
                db_status = self.get_database_status(db_name)
                if "error" not in db_status:
                    status["databases"][db_name] = db_status
                    initialized_count += 1
                    if db_status["detailed_health"]["is_healthy"]:
                        healthy_count += 1
                else:
                    status["databases"][db_name] = {"error": db_status["error"]}
            
            status.update({
                "initialized_count": initialized_count,
                "healthy_count": healthy_count,
                "unhealthy_count": initialized_count - healthy_count
            })
            
            return status
    
    def execute_enhanced_query(self, database_name: str, query: str, 
                             chunk_size: Optional[int] = None,
                             enable_analysis: bool = True,
                             connection_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute query using enhanced connection management with enterprise security.
        
        Args:
            database_name: Name of the database
            query: SQL query to execute
            chunk_size: Optional chunk size for pagination
            enable_analysis: Whether to perform pre-query analysis
            connection_id: Optional connection identifier for security tracking
            
        Returns:
            Dict[str, Any]: Query execution results with security metadata
        """
        try:
            # Basic SQL validation (Task 1 - QueryParser)
            try:
                validated_query = parse_and_validate_sql(query)
            except SQLSecurityError as e:
                return {"error": f"Basic Security Error: {e}"}
            
            # Advanced security validation (Task 10 - SecurityManager)
            connection_id = connection_id or f"enhanced_db_tools_{int(time.time() * 1000)}"
            is_secure, security_error, security_metadata = self.security_manager.validate_query_security(
                validated_query, database_name, connection_id
            )
            
            if not is_secure:
                return {
                    "error": f"Advanced Security Error: {security_error}",
                    "security_metadata": security_metadata
                }
            
            # Get engine using enhanced connection manager
            engine = self.connection_manager.get_engine(database_name)
            if not engine:
                return {"error": f"Database '{database_name}' not available"}
            
            # Get database configuration for timeout settings
            db_config = self.config_manager.get_database_config(database_name)
            if not db_config:
                return {"error": f"Database configuration not found for '{database_name}'"}
            
            # Pre-query analysis
            query_analysis = None
            if enable_analysis:
                try:
                    query_analysis = analyze_query(validated_query, engine, database_name)
                except Exception as e:
                    logger.warning(f"Query analysis failed: {e}")
            
            # Generate query ID for tracking
            query_id = f"{database_name}_{int(time.time() * 1000)}"
            
            # Execute with enterprise security and enhanced connection management
            with self.security_manager.secure_query_execution(validated_query, database_name, connection_id) as security_context:
                with self.connection_manager.managed_query_execution(database_name, query_id) as context:
                    # Create streaming source
                    streaming_source = create_streaming_source(
                        engine=engine,
                        query=validated_query,
                        query_analysis=query_analysis
                    )
                    
                    # Determine chunk size
                    initial_chunk_size = chunk_size or (
                        query_analysis.recommended_chunk_size if query_analysis and query_analysis.should_chunk 
                        else 100
                    )
                    
                    # Execute with timeout management
                    timeout_config = self.timeout_manager.get_timeout_config(database_name)
                    operation_id = f"query_{query_id}"
                    
                    try:
                        with self.timeout_manager.timeout_context(operation_id, timeout_config):
                            first_chunk, streaming_metadata = self.streaming_executor.execute_streaming(
                                streaming_source,
                                query_id,
                                initial_chunk_size,
                                database_name=database_name
                            )
                        
                        # Process results
                        if first_chunk.empty:
                            result = {"data": [], "metadata": {"total_rows": 0}}
                        else:
                            total_rows = streaming_metadata.get("total_rows_processed", len(first_chunk))
                            
                            result = {
                                "metadata": {
                                    "query_id": query_id,
                                    "database_name": database_name,
                                    "total_rows": total_rows,
                                    "showing_rows": f"1-{len(first_chunk)}",
                                    "chunked": total_rows > initial_chunk_size,
                                    "streaming": True,
                                    "execution_time_ms": (time.time() - context["start_time"]) * 1000
                                },
                                "data": json.loads(first_chunk.to_json(orient="records")),
                                "streaming_metadata": streaming_metadata
                            }
                            
                            # Add pagination if needed
                            if total_rows > len(first_chunk):
                                result["pagination"] = {
                                    "next_chunk": f"next_chunk(query_id='{query_id}', start_row={len(first_chunk) + 1}, chunk_size=100)",
                                    "get_all_remaining": f"next_chunk(query_id='{query_id}', start_row={len(first_chunk) + 1}, chunk_size='all')"
                                }
                        
                        # Add analysis results
                        if query_analysis:
                            result["analysis"] = {
                                "estimated_rows": query_analysis.estimated_rows,
                                "actual_rows": len(first_chunk),
                                "estimated_memory_mb": query_analysis.estimated_total_memory_mb,
                                "complexity_score": query_analysis.complexity_score,
                                "risk_levels": {
                                    "memory": query_analysis.memory_risk_level,
                                    "tokens": query_analysis.token_risk_level,
                                    "timeout": query_analysis.timeout_risk_level
                                }
                            }
                        
                        # Add security metadata to results
                        result["security"] = {
                            "fingerprint": security_metadata["fingerprint"],
                            "threat_level": security_metadata["threat_level"].value,
                            "checks_performed": security_metadata["checks_performed"],
                            "validation_time": security_metadata["validation_time"],
                            "complexity": security_metadata.get("complexity", {}),
                            "attack_patterns": security_metadata.get("attack_patterns", [])
                        }
                        
                        return result
                        
                    except QueryTimeoutError as e:
                        return {
                            "error": f"Query timeout: {e.message}",
                            "timeout_reason": e.timeout_reason.value,
                            "execution_time": e.execution_time,
                            "security": {
                                "fingerprint": security_metadata["fingerprint"],
                                "threat_level": security_metadata["threat_level"].value
                            }
                        }
                    
        except Exception as e:
            logger.error(f"Enhanced query execution failed: {e}")
            return {"error": f"Query execution failed: {str(e)}"}


# Create global instance
enhanced_tools = EnhancedDatabaseTools()


@enhanced_mcp.tool
def connect_enhanced_database(name: str) -> str:
    """Connect to a database using enhanced connection management.
    
    Automatically loads configuration from the config system and initializes
    the database with advanced features like connection pooling, health monitoring,
    and resource management.
    
    Args:
        name: Database name as defined in configuration
    """
    try:
        db_config = enhanced_tools.config_manager.get_database_config(name)
        if not db_config:
            return json.dumps({
                "error": f"Database '{name}' not found in configuration",
                "available_databases": list(enhanced_tools.config_manager.get_database_configs().keys())
            })
        
        success = enhanced_tools.connection_manager.initialize_database(name, db_config)
        
        if success:
            # Get connection info
            conn_info = enhanced_tools.connection_manager.get_connection_info(name)
            return json.dumps({
                "success": True,
                "message": f"Successfully connected to database '{name}'",
                "connection_info": conn_info
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": f"Failed to initialize database '{name}'"
            })
            
    except Exception as e:
        return json.dumps({"error": f"Connection failed: {str(e)}"})


@enhanced_mcp.tool
def execute_enhanced_query(name: str, query: str, chunk_size: Optional[int] = None, 
                         connection_id: Optional[str] = None) -> str:
    """Execute a query using enhanced connection management with enterprise-grade security.
    
    Provides automatic resource management, health monitoring, timeout handling,
    comprehensive query analysis, and advanced security validation.
    
    Features:
    - Basic SQL injection prevention (SELECT-only queries)
    - Advanced attack pattern detection (UNION, time-based, boolean-blind, etc.)
    - Rate limiting per connection with configurable thresholds
    - Resource exhaustion protection with memory and CPU monitoring
    - Query fingerprinting and comprehensive audit logging
    - Query complexity analysis and limits enforcement
    
    Args:
        name: Database name
        query: SQL query to execute (must be SELECT-only for security)
        chunk_size: Optional chunk size for large result sets
        connection_id: Optional connection identifier for security tracking
    """
    result = enhanced_tools.execute_enhanced_query(name, query, chunk_size, connection_id=connection_id)
    return json.dumps(result, indent=2, default=str)


@enhanced_mcp.tool
def get_database_status(name: Optional[str] = None) -> str:
    """Get comprehensive status information for databases.
    
    Provides detailed health, performance, and resource usage information
    for enhanced connection management.
    
    Args:
        name: Optional database name. If None, returns status for all databases.
    """
    status = enhanced_tools.get_database_status(name)
    return json.dumps(status, indent=2, default=str)


@enhanced_mcp.tool
def list_databases_by_tags(tags: str) -> str:
    """List databases filtered by tags.
    
    Args:
        tags: Comma-separated list of tags to filter by
    """
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    databases = enhanced_tools.connection_manager.list_databases(include_tags=tag_list)
    
    return json.dumps({
        "filters": tag_list,
        "matching_databases": databases,
        "count": len(databases)
    }, indent=2)


@enhanced_mcp.tool
def trigger_health_check(name: str) -> str:
    """Manually trigger a health check for a specific database.
    
    Args:
        name: Database name
    """
    try:
        result = enhanced_tools.connection_manager.trigger_health_check(name)
        if result:
            return json.dumps({
                "database": name,
                "health_check": {
                    "is_healthy": result.is_healthy,
                    "state": result.state.value,
                    "response_time_ms": result.response_time_ms,
                    "timestamp": result.timestamp,
                    "error_message": result.error_message
                }
            }, indent=2)
        else:
            return json.dumps({"error": f"Database '{name}' not found"})
            
    except Exception as e:
        return json.dumps({"error": f"Health check failed: {str(e)}"})


@enhanced_mcp.tool
def get_resource_usage(name: str) -> str:
    """Get detailed resource usage and limits for a database.
    
    Args:
        name: Database name
    """
    try:
        resource_status = enhanced_tools.connection_manager.get_resource_status(name)
        connection_info = enhanced_tools.connection_manager.get_connection_info(name)
        
        if not connection_info:
            return json.dumps({"error": f"Database '{name}' not found"})
        
        return json.dumps({
            "database": name,
            "resource_usage": resource_status,
            "current_metrics": connection_info["metrics"],
            "connection_config": connection_info["connection_config"]
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to get resource usage: {str(e)}"})


@enhanced_mcp.tool
def get_security_statistics() -> str:
    """Get comprehensive security statistics and event information.
    
    Provides insights into:
    - Security event counts and types
    - Rate limiting status for all connections
    - Resource monitoring metrics
    - Attack pattern detection results
    - Configuration status
    """
    try:
        stats = enhanced_tools.security_manager.get_security_statistics()
        return json.dumps(stats, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to get security statistics: {str(e)}"})


@enhanced_mcp.tool
def get_security_events(limit: Optional[int] = 50, 
                       event_type: Optional[str] = None,
                       threat_level: Optional[str] = None) -> str:
    """Get recent security events with optional filtering.
    
    Args:
        limit: Maximum number of events to return (default: 50)
        event_type: Filter by event type (query_blocked, injection_attempt, rate_limit_exceeded, etc.)
        threat_level: Filter by threat level (low, medium, high, critical)
    """
    try:
        from .security_manager import SecurityEventType, SecurityThreatLevel
        
        # Parse filter parameters
        event_types = None
        if event_type:
            try:
                event_types = [SecurityEventType(event_type.lower())]
            except ValueError:
                return json.dumps({"error": f"Invalid event_type: {event_type}. Valid types: {', '.join([e.value for e in SecurityEventType])}"})
        
        threat_levels = None
        if threat_level:
            try:
                threat_levels = [SecurityThreatLevel(threat_level.lower())]
            except ValueError:
                return json.dumps({"error": f"Invalid threat_level: {threat_level}. Valid levels: {', '.join([t.value for t in SecurityThreatLevel])}"})
        
        events = enhanced_tools.security_manager.get_security_events(
            limit=limit,
            event_types=event_types,
            threat_levels=threat_levels
        )
        
        # Convert events to dictionaries for JSON serialization
        events_data = [event.to_dict() for event in events]
        
        return json.dumps({
            "total_events": len(events_data),
            "events": events_data
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to get security events: {str(e)}"})


@enhanced_mcp.tool
def validate_query_security_standalone(query: str, database_name: str, 
                                     connection_id: Optional[str] = None) -> str:
    """Validate query security without executing it.
    
    Performs comprehensive security validation including:
    - Basic SQL injection prevention
    - Advanced attack pattern detection
    - Query complexity analysis
    - Rate limiting checks
    - Resource limit validation
    
    Args:
        query: SQL query to validate
        database_name: Target database name
        connection_id: Optional connection identifier for tracking
    """
    try:
        is_valid, error_msg, metadata = enhanced_tools.security_manager.validate_query_security(
            query, database_name, connection_id
        )
        
        return json.dumps({
            "is_valid": is_valid,
            "error_message": error_msg,
            "security_metadata": {
                "fingerprint": metadata["fingerprint"],
                "threat_level": metadata["threat_level"].value,
                "checks_performed": metadata["checks_performed"],
                "validation_time": metadata["validation_time"],
                "complexity": metadata.get("complexity", {}),
                "attack_patterns": metadata.get("attack_patterns", [])
            }
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": f"Security validation failed: {str(e)}"})


@enhanced_mcp.tool
def initialize_all_databases() -> str:
    """Initialize all databases defined in the configuration.
    
    Attempts to connect to all enabled databases found in the configuration system.
    """
    try:
        results = enhanced_tools.initialize_all_configured_databases()
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        return json.dumps({
            "total_databases": total_count,
            "successful_initializations": success_count,
            "failed_initializations": total_count - success_count,
            "results": results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Bulk initialization failed: {str(e)}"})


@enhanced_mcp.tool
def disconnect_enhanced_database(name: str) -> str:
    """Disconnect from a database and clean up resources.
    
    Args:
        name: Database name to disconnect
    """
    try:
        success = enhanced_tools.connection_manager.close_database(name)
        
        if success:
            return json.dumps({
                "success": True,
                "message": f"Successfully disconnected from database '{name}'"
            })
        else:
            return json.dumps({
                "success": False,
                "message": f"Failed to disconnect from database '{name}'"
            })
            
    except Exception as e:
        return json.dumps({"error": f"Disconnect failed: {str(e)}"})


def main():
    """Run enhanced database tools server."""
    enhanced_mcp.run(transport="stdio")


if __name__ == "__main__":
    main()