"""Basic functionality tests for localdata-mcp."""

from localdata_mcp import DatabaseManager


def test_imports():
    """Test that we can import the module."""
    from localdata_mcp import DatabaseManager, main

    assert DatabaseManager is not None
    assert main is not None


def test_database_manager_initialization():
    """Test DatabaseManager initialization."""
    manager = DatabaseManager()
    assert manager.connections == {}
    assert manager.query_history == {}
    assert manager.query_buffers == {}
    assert hasattr(manager, "connection_semaphore")
    assert hasattr(manager, "connection_lock")


