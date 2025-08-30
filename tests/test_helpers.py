"""
Test helper utilities for accessing MCP tools directly for testing.
"""

from localdata_mcp.localdata_mcp import DatabaseManager
from unittest.mock import patch, MagicMock


class TestDatabaseManager:
    """
    Test wrapper for DatabaseManager that provides direct access to tool methods.
    This bypasses the MCP framework for testing purposes.
    """
    
    def __init__(self):
        # Create the actual manager
        self._manager = DatabaseManager()
        
        # Extract the underlying methods from the MCP tools
        # The @mcp.tool decorator stores the original function
        self.connect_database = self._extract_tool_function('connect_database')
        self.disconnect_database = self._extract_tool_function('disconnect_database')
        self.execute_query = self._extract_tool_function('execute_query')
        self.next_chunk = self._extract_tool_function('next_chunk')
        self.get_query_history = self._extract_tool_function('get_query_history')
        self.list_databases = self._extract_tool_function('list_databases')
        self.describe_database = self._extract_tool_function('describe_database')
        self.find_table = self._extract_tool_function('find_table')
        self.describe_table = self._extract_tool_function('describe_table')
        
        # Provide access to internal attributes and methods
        self.connections = self._manager.connections
        self.db_types = self._manager.db_types
        self.query_history = self._manager.query_history
        self.query_buffers = self._manager.query_buffers
        self.connection_count = self._manager.connection_count
    
    def _extract_tool_function(self, method_name):
        """Extract the underlying function from an MCP tool."""
        tool_obj = getattr(self._manager, method_name)
        
        if hasattr(tool_obj, 'fn'):
            # FastMCP stores the original function in fn attribute
            return tool_obj.fn.__get__(self._manager, DatabaseManager)
        elif hasattr(tool_obj, '_func'):
            # Alternative storage location
            return tool_obj._func.__get__(self._manager, DatabaseManager)
        elif hasattr(tool_obj, '__wrapped__'):
            # Some decorators store in __wrapped__
            return tool_obj.__wrapped__.__get__(self._manager, DatabaseManager)
        else:
            # Fallback: get the original method directly from the class
            original_method = getattr(DatabaseManager, method_name)
            return original_method.__get__(self._manager, DatabaseManager)
    
    def __getattr__(self, name):
        """Delegate any other attribute access to the underlying manager."""
        return getattr(self._manager, name)


def create_test_manager():
    """
    Create a TestDatabaseManager instance with all mocking applied.
    This is the main entry point for tests.
    """
    return TestDatabaseManager()