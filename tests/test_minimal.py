"""Minimal tests to ensure CI passes."""

import sys
from pathlib import Path

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_basic_import():
    """Test that we can import the module."""
    try:
        from localdata_mcp import DatabaseManager
        assert DatabaseManager is not None
    except ImportError as e:
        assert False, f"Failed to import DatabaseManager: {e}"

def test_main_import():
    """Test that we can import main function."""
    try:
        from localdata_mcp import main
        assert main is not None
    except ImportError as e:
        assert False, f"Failed to import main: {e}"

def test_database_manager_creation():
    """Test that we can create a DatabaseManager instance."""
    try:
        from localdata_mcp import DatabaseManager
        manager = DatabaseManager()
        assert manager is not None
        assert hasattr(manager, 'connections')
    except Exception as e:
        assert False, f"Failed to create DatabaseManager: {e}"