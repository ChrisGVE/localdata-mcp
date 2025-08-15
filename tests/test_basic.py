"""Basic functionality tests for localdata-mcp."""

import os
import tempfile
import sqlite3
import pytest
from pathlib import Path
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
    assert hasattr(manager, 'connection_semaphore')
    assert hasattr(manager, 'connection_lock')


def test_path_security():
    """Test path security functions."""
    manager = DatabaseManager()
    
    # Create a test file in current directory
    test_file = Path("test_security.txt")
    test_file.write_text("test content")
    
    try:
        # Test valid path (current directory)
        result = manager._sanitize_path(str(test_file))
        assert test_file.resolve() == Path(result)
        
        # Test invalid path (parent directory)
        with pytest.raises(ValueError, match="outside the allowed directory"):
            manager._sanitize_path("../etc/passwd")
        
        # Test non-existent file
        with pytest.raises(ValueError, match="File not found"):
            manager._sanitize_path("nonexistent_file.txt")
            
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def test_connection_management():
    """Test basic connection management."""
    manager = DatabaseManager()
    
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        db_path = temp_db.name
    
    try:
        # Test connection - use the underlying method, not the decorated tool
        result = manager.connect_database.fn(manager, "test_db", "sqlite", db_path)
        assert "Successfully connected" in result
        assert "test_db" in manager.connections
        
        # Test listing databases
        result = manager.list_databases.fn(manager)
        databases = eval(result)  # Safe since we control the output
        assert "test_db" in databases
        
        # Test disconnection
        result = manager.disconnect_database.fn(manager, "test_db")
        assert "Successfully disconnected" in result
        assert "test_db" not in manager.connections
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_file_size_detection():
    """Test file size detection."""
    manager = DatabaseManager()
    
    # Create a small test file
    test_file = Path("small_test.txt")
    test_file.write_text("small content")
    
    try:
        size = manager._get_file_size(str(test_file))
        assert size > 0
        
        is_large = manager._is_large_file(str(test_file), threshold_mb=1)  # 1MB threshold
        assert not is_large  # Small file should not be considered large
        
    finally:
        if test_file.exists():
            test_file.unlink()


def test_query_id_generation():
    """Test query ID generation."""
    manager = DatabaseManager()
    
    db_name = "testdb"  # Use simple name without underscores
    query = "SELECT * FROM test_table"
    
    query_id1 = manager._generate_query_id(db_name, query)
    query_id2 = manager._generate_query_id(db_name, query)
    
    # IDs should have the expected format: dbname_timestamp_hash
    assert db_name in query_id1
    assert "_" in query_id1
    
    # IDs generated at different times should be different (due to timestamp)
    # In practice, they might be the same if generated quickly, but the format should be correct
    parts1 = query_id1.split("_")
    parts2 = query_id2.split("_")
    
    assert len(parts1) == 3  # db_timestamp_hash
    assert len(parts2) == 3  # db_timestamp_hash
    assert parts1[0] == db_name
    assert parts2[0] == db_name


def test_safe_table_identifier():
    """Test safe table identifier validation."""
    manager = DatabaseManager()
    
    # Valid table names
    valid_names = ["users", "user_data", "_private", "table123", "CamelCase"]
    for name in valid_names:
        result = manager._safe_table_identifier(name)
        assert result is not None
        assert name in result  # The quoted name should contain the original
    
    # Invalid table names should raise ValueError
    invalid_names = ["user-data", "123table", "table with spaces", "table;drop", ""]
    for name in invalid_names:
        with pytest.raises(ValueError, match="Invalid table name"):
            manager._safe_table_identifier(name)


if __name__ == "__main__":
    pytest.main([__file__])