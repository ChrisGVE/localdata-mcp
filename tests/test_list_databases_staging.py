"""Tests for list_databases with include_staging parameter."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from localdata_mcp import DatabaseManager


def _make_staging_entry(name, parent, query="SELECT 1", row_count=10):
    """Create a staging dict matching StagingDatabase.to_dict() output."""
    return {
        "name": name,
        "type": "staging",
        "parent_connection": parent,
        "source_query": query,
        "created_at": datetime.now().isoformat(),
        "size_mb": 0.01,
        "row_count": row_count,
        "expires_at": None,
    }


def test_list_databases_default_no_staging():
    """Default call should not include staging_databases key."""
    manager = DatabaseManager()
    result = json.loads(manager.list_databases())
    assert "staging_databases" not in result
    assert result["databases"] == []


def test_list_databases_include_staging():
    """When include_staging=True, staging entries should appear."""
    entries = [_make_staging_entry("staging_db1", "parent_db")]
    manager = DatabaseManager()
    with patch("localdata_mcp.localdata_mcp.get_staging_manager") as mock_get:
        mock_mgr = MagicMock()
        mock_mgr.list_staging.return_value = entries
        mock_get.return_value = mock_mgr
        result = json.loads(manager.list_databases(include_staging=True))

    assert "staging_databases" in result
    assert len(result["staging_databases"]) == 1
    assert result["staging_databases"][0]["name"] == "staging_db1"


def test_staging_type_marker():
    """Each staging entry must have type='staging'."""
    entries = [
        _make_staging_entry("s1", "p1"),
        _make_staging_entry("s2", "p2"),
    ]
    manager = DatabaseManager()
    with patch("localdata_mcp.localdata_mcp.get_staging_manager") as mock_get:
        mock_mgr = MagicMock()
        mock_mgr.list_staging.return_value = entries
        mock_get.return_value = mock_mgr
        result = json.loads(manager.list_databases(include_staging=True))

    for entry in result["staging_databases"]:
        assert entry["type"] == "staging"


def test_staging_metadata_fields():
    """Staging entries must contain all expected metadata fields."""
    expected_fields = {
        "name",
        "type",
        "parent_connection",
        "source_query",
        "created_at",
        "size_mb",
        "row_count",
        "expires_at",
    }
    entries = [_make_staging_entry("s1", "p1", row_count=42)]
    manager = DatabaseManager()
    with patch("localdata_mcp.localdata_mcp.get_staging_manager") as mock_get:
        mock_mgr = MagicMock()
        mock_mgr.list_staging.return_value = entries
        mock_get.return_value = mock_mgr
        result = json.loads(manager.list_databases(include_staging=True))

    entry = result["staging_databases"][0]
    assert set(entry.keys()) == expected_fields
    assert entry["row_count"] == 42


def test_empty_staging():
    """include_staging=True with no staging dbs returns empty list."""
    manager = DatabaseManager()
    with patch("localdata_mcp.localdata_mcp.get_staging_manager") as mock_get:
        mock_mgr = MagicMock()
        mock_mgr.list_staging.return_value = []
        mock_get.return_value = mock_mgr
        result = json.loads(manager.list_databases(include_staging=True))

    assert result["staging_databases"] == []
