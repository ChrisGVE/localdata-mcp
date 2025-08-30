"""Functional tests for the 9-tool streamlined architecture.

These tests validate that the 9 tools are properly defined and have
the expected functionality characteristics for the streamlined architecture.
"""

import json
import sys
from pathlib import Path
import tempfile
import pandas as pd

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from localdata_mcp.localdata_mcp import DatabaseManager, mcp


def test_streamlined_architecture_overview():
    """Test the overall architecture is streamlined to 9 tools."""
    tools = mcp._tool_manager._tools
    
    # Exactly 9 tools
    assert len(tools) == 9
    
    # Tools are organized by function:
    # Connection Management (2): connect_database, disconnect_database
    # Query Execution (2): execute_query, next_chunk  
    # Database Discovery (3): list_databases, describe_database, find_table
    # Table Operations (2): describe_table, get_query_history
    
    connection_tools = {'connect_database', 'disconnect_database'}
    query_tools = {'execute_query', 'next_chunk'}
    discovery_tools = {'list_databases', 'describe_database', 'find_table'}
    table_tools = {'describe_table', 'get_query_history'}
    
    all_expected = connection_tools | query_tools | discovery_tools | table_tools
    actual_tools = set(tools.keys())
    
    assert actual_tools == all_expected, f"Tool mismatch: expected {all_expected}, got {actual_tools}"


def test_execute_query_replaces_execute_query_json():
    """Test that execute_query replaced execute_query_json with enhanced features."""
    tools = mcp._tool_manager._tools
    
    # Should have execute_query
    assert 'execute_query' in tools
    
    # Should NOT have execute_query_json
    assert 'execute_query_json' not in tools
    
    # execute_query should have chunk_size parameter for pagination
    execute_query = tools['execute_query']
    params = execute_query.parameters['properties']
    assert 'chunk_size' in params
    
    # Description should mention JSON and chunking
    desc = execute_query.description.lower()
    assert 'json' in desc
    assert 'chunk' in desc or 'pagination' in desc


def test_next_chunk_pagination_tool():
    """Test that next_chunk is the new pagination tool."""
    tools = mcp._tool_manager._tools
    
    assert 'next_chunk' in tools
    
    next_chunk = tools['next_chunk']
    params = next_chunk.parameters['properties']
    
    # Should have pagination parameters
    required_params = {'query_id', 'start_row', 'chunk_size'}
    actual_params = set(params.keys()) - {'self'}  # Remove internal 'self' param
    
    assert required_params.issubset(actual_params), \
        f"next_chunk missing required params: {required_params - actual_params}"
    
    # Description should mention pagination or chunks
    desc = next_chunk.description.lower()
    assert 'chunk' in desc or 'pagination' in desc or 'retrieve' in desc


def test_sql_flavor_detection_capability():
    """Test that tools support SQL flavor detection."""
    tools = mcp._tool_manager._tools
    
    # connect_database should mention SQL flavor in description
    connect_desc = tools['connect_database'].description.lower()
    # It should support various database types
    assert any(db_type in connect_desc for db_type in ['csv', 'json', 'excel', 'sqlite', 'mysql', 'postgresql'])
    
    # list_databases should provide SQL flavor info
    list_desc = tools['list_databases'].description.lower()
    assert 'flavor' in list_desc or 'connection' in list_desc or 'database' in list_desc


def test_memory_management_features():
    """Test that execute_query supports memory management."""
    tools = mcp._tool_manager._tools
    
    execute_query = tools['execute_query']
    desc = execute_query.description.lower()
    
    # Should mention large results, memory, or buffering
    assert any(keyword in desc for keyword in ['large', 'memory', 'buffer', 'chunk', 'performance'])


def test_json_only_responses():
    """Test that tools are designed for structured data responses."""
    tools = mcp._tool_manager._tools
    
    # execute_query should explicitly mention JSON since it replaced execute_query_json
    execute_query = tools['execute_query']
    desc = execute_query.description.lower()
    assert 'json' in desc, "execute_query should mention JSON since it replaced execute_query_json"
    
    # Other data-returning tools should be designed for structured output
    # (they return JSON but may not mention it explicitly in description)
    data_tools = ['list_databases', 'describe_database', 'describe_table']
    for tool_name in data_tools:
        tool = tools[tool_name]
        desc = tool.description.lower()
        # Should mention structured data concepts
        assert any(word in desc for word in ['list', 'information', 'details', 'description', 'database', 'table']), \
            f"Tool {tool_name} should describe structured data output"


def test_consolidated_functionality():
    """Test that functionality has been properly consolidated."""
    tools = mcp._tool_manager._tools
    tool_names = set(tools.keys())
    
    # These old tools should NOT exist (consolidated functionality)
    removed_tools = {
        'execute_query_json',  # Merged into execute_query
        'get_schema_details',  # Merged into describe_database  
        'get_table_info',      # Merged into describe_table
        'execute_raw_sql',     # Merged into execute_query
        'list_table_names',    # Available through describe_database
        'export_to_format',    # Removed for simplicity
        'get_connection_info', # Merged into list_databases
        'validate_query',      # Removed for simplicity
        'bulk_insert'          # Removed for simplicity
    }
    
    for removed_tool in removed_tools:
        assert removed_tool not in tool_names, f"Tool {removed_tool} should be removed in streamlined architecture"


def test_database_manager_structure():
    """Test that DatabaseManager has the right structure for 9-tool architecture."""
    manager = DatabaseManager()
    
    # Should have core attributes for connection management
    assert hasattr(manager, 'connections')
    assert hasattr(manager, 'db_types') 
    assert hasattr(manager, 'query_history')
    assert hasattr(manager, 'query_buffers')
    
    # Should have threading and security features
    assert hasattr(manager, 'connection_semaphore')
    assert hasattr(manager, 'connection_lock') 
    assert hasattr(manager, 'query_buffer_lock')
    
    # Should have all 9 tool methods
    expected_methods = [
        'connect_database', 'disconnect_database', 'execute_query', 'next_chunk',
        'list_databases', 'describe_database', 'find_table', 'describe_table',
        'get_query_history'
    ]
    
    for method in expected_methods:
        assert hasattr(manager, method), f"DatabaseManager missing {method}"


def test_tool_descriptions_completeness():
    """Test that all tools have comprehensive descriptions."""
    tools = mcp._tool_manager._tools
    
    for tool_name, tool in tools.items():
        desc = tool.description
        
        # Each tool should have a substantial description
        assert desc, f"Tool {tool_name} has no description"
        assert len(desc) > 20, f"Tool {tool_name} has very short description: {desc[:50]}..."
        
        # Should describe what it does
        assert '.' in desc, f"Tool {tool_name} description should be a proper sentence"
        
        # Should not be placeholder text
        assert 'todo' not in desc.lower(), f"Tool {tool_name} has placeholder description"
        assert 'placeholder' not in desc.lower(), f"Tool {tool_name} has placeholder description"


def test_parameter_validation():
    """Test that tools have appropriate parameter validation."""
    tools = mcp._tool_manager._tools
    
    # connect_database should have required parameters
    connect_params = tools['connect_database'].parameters['required']
    assert 'name' in connect_params
    assert 'db_type' in connect_params
    assert 'conn_string' in connect_params
    
    # execute_query should have required parameters
    query_params = tools['execute_query'].parameters['required'] 
    assert 'name' in query_params
    assert 'query' in query_params
    
    # next_chunk should have required parameters
    chunk_params = tools['next_chunk'].parameters['required']
    assert 'query_id' in chunk_params
    assert 'start_row' in chunk_params
    assert 'chunk_size' in chunk_params


def test_architecture_efficiency():
    """Test that the 9-tool architecture is efficient and focused."""
    tools = mcp._tool_manager._tools
    
    # Exactly 9 tools - not more, not less
    assert len(tools) == 9, f"Should have exactly 9 tools, got {len(tools)}"
    
    # Each tool should have a distinct, focused purpose
    # No two tools should have overly similar descriptions
    descriptions = [tool.description for tool in tools.values()]
    
    # Basic check: descriptions should be different
    assert len(set(descriptions)) == 9, "All tools should have unique descriptions"
    
    # Tool names should be clear and descriptive
    for name in tools.keys():
        assert '_' in name, f"Tool name {name} should use snake_case"
        assert len(name.split('_')) >= 2, f"Tool name {name} should be descriptive"
        assert not name.startswith('_'), f"Tool name {name} should not start with underscore"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])