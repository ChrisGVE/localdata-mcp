#!/usr/bin/env python3
"""Usage examples for Enhanced Connection Management in LocalData MCP.

Demonstrates how to use the enhanced database connection management features
including configuration, health monitoring, resource management, and tag-based operations.
"""

import asyncio
import json
import time
from pathlib import Path

# Import the enhanced tools
from src.localdata_mcp.connection_manager import get_enhanced_connection_manager
from src.localdata_mcp.config_manager import initialize_config, DatabaseConfig, DatabaseType
from src.localdata_mcp.enhanced_database_tools import enhanced_tools


def example_1_basic_configuration():
    """Example 1: Basic configuration and connection setup."""
    print("=== Example 1: Basic Configuration Setup ===")
    
    # Initialize configuration with enhanced config file
    config_file = Path(__file__).parent / "enhanced-config.yaml"
    config_manager = initialize_config(config_file=str(config_file))
    
    # Get all configured databases
    db_configs = config_manager.get_database_configs()
    print(f"Found {len(db_configs)} configured databases:")
    
    for name, config in db_configs.items():
        print(f"  - {name}: {config.type.value} ({'enabled' if config.enabled else 'disabled'})")
        print(f"    Tags: {', '.join(config.tags)}")
        print(f"    Max connections: {config.max_connections}")
        print(f"    Query timeout: {config.query_timeout}s")
        print()


def example_2_initialize_databases():
    """Example 2: Initialize databases using enhanced connection management."""
    print("=== Example 2: Database Initialization ===")
    
    manager = get_enhanced_connection_manager()
    
    # Initialize all configured databases
    results = enhanced_tools.initialize_all_configured_databases()
    
    print("Database initialization results:")
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nSuccessfully initialized {sum(results.values())} out of {len(results)} databases")


def example_3_health_monitoring():
    """Example 3: Health monitoring and status checking."""
    print("=== Example 3: Health Monitoring ===")
    
    manager = get_enhanced_connection_manager()
    
    # Get status for all databases
    status = enhanced_tools.get_database_status()
    
    print("Database Health Summary:")
    print(f"  Total configured: {status['total_configured']}")
    print(f"  Initialized: {status['initialized_count']}")
    print(f"  Healthy: {status['healthy_count']}")
    print(f"  Unhealthy: {status['unhealthy_count']}")
    print()
    
    # Show detailed health for each database
    for name, db_info in status["databases"].items():
        if "error" in db_info:
            print(f"  {name}: ERROR - {db_info['error']}")
        else:
            health = db_info["detailed_health"]
            print(f"  {name}: {health['state'].upper()} "
                  f"({health['response_time_ms']:.1f}ms response)")


def example_4_tag_based_operations():
    """Example 4: Tag-based database operations."""
    print("=== Example 4: Tag-Based Operations ===")
    
    manager = get_enhanced_connection_manager()
    
    # Get databases by different tags
    tags_to_check = ["production", "development", "analytics", "cache"]
    
    for tag in tags_to_check:
        databases = manager.get_databases_by_tag(tag)
        print(f"Databases with tag '{tag}': {', '.join(databases) if databases else 'None'}")
    
    print()
    
    # List databases with multiple tags
    databases = manager.list_databases(include_tags=["production", "analytics"])
    print("Production OR Analytics databases:")
    for db_info in databases:
        print(f"  - {db_info['name']} ({db_info['type']}): {', '.join(db_info['tags'])}")


def example_5_resource_monitoring():
    """Example 5: Resource usage monitoring."""
    print("=== Example 5: Resource Monitoring ===")
    
    manager = get_enhanced_connection_manager()
    
    # Get all databases and check their resource usage
    all_configs = enhanced_tools.config_manager.get_database_configs()
    
    for name in all_configs.keys():
        conn_info = manager.get_connection_info(name)
        if not conn_info:
            continue
            
        print(f"Resource usage for '{name}':")
        metrics = conn_info["metrics"]
        limits = conn_info["resource_limits"]
        
        print(f"  Queries: {metrics['total_queries']} total, "
              f"{metrics['success_rate']:.1f}% success rate")
        print(f"  Average query time: {metrics['average_query_time']:.3f}s")
        print(f"  Memory usage: {metrics['memory_usage_mb']:.1f}MB")
        print(f"  Active connections: {metrics['active_connections']}")
        
        # Check resource limits
        memory_limit = limits.get("memory", {})
        if memory_limit:
            print(f"  Memory limit: {memory_limit['current_value']:.1f}MB / "
                  f"{memory_limit['max_value']:.1f}MB "
                  f"({'WARNING' if memory_limit['is_warning'] else 'OK'})")
        print()


def example_6_query_execution():
    """Example 6: Enhanced query execution with monitoring."""
    print("=== Example 6: Enhanced Query Execution ===")
    
    # This example requires an actual database connection
    # We'll demonstrate with a local SQLite database if available
    
    manager = get_enhanced_connection_manager()
    
    # Try to find a SQLite database in the configuration
    sqlite_databases = manager.get_databases_by_tag("sqlite")
    
    if not sqlite_databases:
        print("No SQLite databases configured for this example")
        return
    
    db_name = sqlite_databases[0]
    print(f"Using database: {db_name}")
    
    # Execute a simple query
    try:
        result = enhanced_tools.execute_enhanced_query(
            database_name=db_name,
            query="SELECT 1 as test_column, 'Hello World' as message",
            enable_analysis=True
        )
        
        if "error" in result:
            print(f"Query failed: {result['error']}")
        else:
            print("Query executed successfully!")
            print(f"  Execution time: {result['metadata']['execution_time_ms']:.1f}ms")
            print(f"  Rows returned: {result['metadata']['total_rows']}")
            print(f"  Data: {result['data']}")
            
            if "analysis" in result:
                analysis = result["analysis"]
                print(f"  Query complexity: {analysis['complexity_score']}/10")
                print(f"  Risk levels: Memory={analysis['risk_levels']['memory']}, "
                      f"Timeout={analysis['risk_levels']['timeout']}")
                      
    except Exception as e:
        print(f"Query execution failed: {e}")


def example_7_concurrent_operations():
    """Example 7: Concurrent database operations."""
    print("=== Example 7: Concurrent Operations ===")
    
    manager = get_enhanced_connection_manager()
    
    # Find databases that support concurrent operations
    sqlite_databases = manager.get_databases_by_tag("sqlite")
    
    if not sqlite_databases:
        print("No suitable databases found for concurrent testing")
        return
    
    db_name = sqlite_databases[0]
    print(f"Testing concurrent operations on: {db_name}")
    
    import threading
    import concurrent.futures
    
    def execute_test_query(query_num):
        """Execute a test query with unique identifier."""
        query_id = f"concurrent_test_{query_num}"
        
        try:
            with manager.managed_query_execution(db_name, query_id) as context:
                # Simulate some work
                time.sleep(0.1)
                return {
                    "query_id": context["query_id"],
                    "success": True,
                    "execution_time": time.time() - context["start_time"]
                }
        except Exception as e:
            return {
                "query_id": query_id,
                "success": False,
                "error": str(e)
            }
    
    # Execute multiple queries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(execute_test_query, i) for i in range(10)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    successful = sum(1 for r in results if r["success"])
    print(f"Executed {len(results)} concurrent operations: {successful} successful")
    
    # Check updated metrics
    conn_info = manager.get_connection_info(db_name)
    if conn_info:
        metrics = conn_info["metrics"]
        print(f"Updated metrics - Total queries: {metrics['total_queries']}, "
              f"Success rate: {metrics['success_rate']:.1f}%")


def example_8_configuration_hot_reload():
    """Example 8: Configuration hot reload demonstration."""
    print("=== Example 8: Configuration Hot Reload ===")
    print("Note: This example demonstrates the hot reload capability")
    print("In production, you would modify the config file and the system would auto-reload")
    
    config_manager = enhanced_tools.config_manager
    
    print(f"Current config file modification times:")
    print(f"Has config changed: {config_manager.has_config_changed()}")
    
    # Show current database count
    current_configs = config_manager.get_database_configs()
    print(f"Currently configured databases: {len(current_configs)}")
    
    # In a real scenario, you would:
    # 1. Modify the YAML configuration file
    # 2. The system would detect the change and reload automatically
    # 3. New databases would become available immediately
    
    print("To test hot reload:")
    print("1. Modify the enhanced-config.yaml file")
    print("2. Add or modify database configurations")
    print("3. The system will automatically detect and reload the changes")


def main():
    """Run all examples."""
    print("Enhanced Connection Management Examples")
    print("=" * 50)
    print()
    
    examples = [
        example_1_basic_configuration,
        example_2_initialize_databases,
        example_3_health_monitoring,
        example_4_tag_based_operations,
        example_5_resource_monitoring,
        example_6_query_execution,
        example_7_concurrent_operations,
        example_8_configuration_hot_reload,
    ]
    
    for example in examples:
        try:
            example()
            print("\n" + "-" * 50 + "\n")
        except Exception as e:
            print(f"Example failed: {e}")
            print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()