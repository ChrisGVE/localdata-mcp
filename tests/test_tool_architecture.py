"""Test the 9-tool streamlined architecture."""

import sys
from pathlib import Path
import inspect

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from localdata_mcp.localdata_mcp import DatabaseManager, mcp


def test_nine_tools_exist():
    """Test that exactly 9 tools are defined."""
    # Get all tools from the MCP server tool manager
    tools = mcp._tool_manager._tools
    
    # Should have exactly 9 tools
    assert len(tools) == 9, f"Expected 9 tools, got {len(tools)}: {list(tools.keys())}"


def test_tool_names_match_architecture():
    """Test that the 9 expected tools exist with correct names."""
    tools = mcp._tool_manager._tools
    tool_names = list(tools.keys())
    
    expected_tools = {
        'connect_database',
        'disconnect_database', 
        'execute_query',
        'next_chunk',
        'get_query_history',
        'list_databases',
        'describe_database',
        'find_table',
        'describe_table'
    }
    
    actual_tools = set(tool_names)
    
    # Check we have all expected tools
    missing_tools = expected_tools - actual_tools
    assert not missing_tools, f"Missing tools: {missing_tools}"
    
    # Check we don't have unexpected tools
    extra_tools = actual_tools - expected_tools
    assert not extra_tools, f"Unexpected tools: {extra_tools}"


def test_execute_query_replaces_execute_query_json():
    """Test that execute_query exists instead of the old execute_query_json."""
    tools = mcp._tool_manager._tools
    tool_names = list(tools.keys())
    
    # Should have execute_query
    assert 'execute_query' in tool_names, "execute_query tool missing"
    
    # Should NOT have the old execute_query_json
    assert 'execute_query_json' not in tool_names, "Old execute_query_json tool still exists"


def test_next_chunk_is_new_tool():
    """Test that next_chunk is present as the new pagination tool."""
    tools = mcp._tool_manager._tools
    tool_names = list(tools.keys())
    
    assert 'next_chunk' in tool_names, "next_chunk tool missing"


def test_all_tools_have_descriptions():
    """Test that all tools have proper descriptions."""
    tools = mcp._tool_manager._tools
    
    for tool_name, tool in tools.items():
        assert tool.description, f"Tool '{tool_name}' missing description"
        assert len(tool.description) > 10, f"Tool '{tool_name}' has very short description"


def test_database_manager_has_tool_methods():
    """Test that DatabaseManager has the expected tool methods."""
    manager = DatabaseManager()
    
    expected_methods = [
        'connect_database',
        'disconnect_database',
        'execute_query', 
        'next_chunk',
        'get_query_history',
        'list_databases',
        'describe_database',
        'find_table',
        'describe_table'
    ]
    
    for method_name in expected_methods:
        assert hasattr(manager, method_name), f"DatabaseManager missing method: {method_name}"


def test_tool_parameters():
    """Test that key tools have expected parameters."""
    tools = mcp._tool_manager._tools
    
    # Test execute_query has chunk_size parameter
    execute_query = tools['execute_query']
    param_names = list(execute_query.parameters['properties'].keys())
    assert 'chunk_size' in param_names, "execute_query missing chunk_size parameter"
    
    # Test next_chunk has expected parameters
    next_chunk = tools['next_chunk']
    next_chunk_params = list(next_chunk.parameters['properties'].keys())
    expected_next_chunk_params = {'query_id', 'start_row', 'chunk_size'}
    actual_next_chunk_params = set(next_chunk_params)
    # Remove 'self' parameter which is internal
    actual_next_chunk_params.discard('self')
    assert expected_next_chunk_params.issubset(actual_next_chunk_params), \
        f"next_chunk missing parameters: {expected_next_chunk_params - actual_next_chunk_params}"


def test_no_removed_tools():
    """Test that removed tools from the previous 15-tool architecture are gone."""
    tools = mcp._tool_manager._tools
    tool_names = list(tools.keys())
    
    # These tools should be removed in the streamlined architecture
    removed_tools = [
        'execute_query_json',  # Renamed to execute_query
        'get_schema_details',  # Consolidated into describe_database
        'get_table_info',      # Consolidated into describe_table
        'execute_raw_sql',     # Consolidated into execute_query
        'list_table_names',    # Functionality in describe_database
        'export_to_format'     # Removed for simplicity
    ]
    
    for removed_tool in removed_tools:
        assert removed_tool not in tool_names, f"Removed tool '{removed_tool}' still exists"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])