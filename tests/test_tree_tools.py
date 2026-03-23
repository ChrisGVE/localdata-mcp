"""Tests for tree_tools and tree_export modules."""

import json
import tempfile
import os

import pytest
import toml
import yaml
from sqlalchemy import create_engine

from localdata_mcp.tree_storage import (
    TreeStorageManager,
    ValueType,
    build_path,
    create_tree_schema,
)
from localdata_mcp.tree_parsers import (
    parse_dict_to_tree,
    parse_toml_to_tree,
)
from localdata_mcp.tree_tools import (
    tool_delete_key,
    tool_delete_node,
    tool_get_children,
    tool_get_node,
    tool_get_value,
    tool_list_keys,
    tool_set_node,
    tool_set_value,
)
from localdata_mcp.tree_export import (
    export_json,
    export_toml,
    export_yaml,
    reconstruct_tree,
    tool_export_structured,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """TreeStorageManager backed by in-memory SQLite."""
    engine = create_engine("sqlite:///:memory:")
    create_tree_schema(engine)
    return TreeStorageManager(engine)


@pytest.fixture
def populated(manager):
    """Manager with a small tree pre-loaded."""
    data = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "enabled": True,
        },
        "logging": {
            "level": "info",
            "file": "/var/log/app.log",
        },
    }
    parse_dict_to_tree(data, manager)
    return manager


NAME = "test_conn"


# ---------------------------------------------------------------------------
# tool_get_node
# ---------------------------------------------------------------------------


class TestGetNode:
    def test_root_summary_empty(self, manager):
        result = tool_get_node(manager, NAME)
        assert result["total_nodes"] == 0
        assert result["root_nodes"] == []

    def test_root_summary_populated(self, populated):
        result = tool_get_node(populated, NAME)
        assert result["total_nodes"] > 0
        assert "database" in result["root_nodes"]
        assert "logging" in result["root_nodes"]

    def test_specific_node(self, populated):
        result = tool_get_node(populated, NAME, path="database")
        assert result["path"] == "database"
        assert result["name"] == "database"
        assert result["children_count"] == 0  # host/port are props, not children
        assert result["property_count"] == 3
        assert "properties" in result

    def test_node_not_found(self, populated):
        result = tool_get_node(populated, NAME, path="nonexistent")
        assert "error" in result

    def test_parent_path(self, manager):
        parse_dict_to_tree({"a": {"b": {"c": 1}}}, manager)
        result = tool_get_node(manager, NAME, path="a.b")
        assert result["parent_path"] == "a"

    def test_root_node_parent_is_none(self, populated):
        result = tool_get_node(populated, NAME, path="database")
        assert result["parent_path"] is None


# ---------------------------------------------------------------------------
# tool_get_children
# ---------------------------------------------------------------------------


class TestGetChildren:
    def test_root_children(self, populated):
        result = tool_get_children(populated, NAME)
        assert result["parent_path"] is None
        names = [c["name"] for c in result["children"]]
        assert "database" in names
        assert "logging" in names

    def test_pagination(self, manager):
        for i in range(5):
            manager.create_node(f"item_{i}")
        result = tool_get_children(manager, NAME, offset=0, limit=2)
        assert len(result["children"]) == 2
        assert result["has_more"] is True
        assert result["total"] == 5

    def test_no_children(self, populated):
        # database has properties but no child nodes
        result = tool_get_children(populated, NAME, path="database")
        assert result["children"] == []
        assert result["has_more"] is False

    def test_children_of_nested(self, manager):
        parse_dict_to_tree({"a": {"b": {"x": 1}, "c": {"y": 2}}}, manager)
        result = tool_get_children(manager, NAME, path="a")
        names = [c["name"] for c in result["children"]]
        assert "b" in names
        assert "c" in names


# ---------------------------------------------------------------------------
# tool_set_node
# ---------------------------------------------------------------------------


class TestSetNode:
    def test_create_simple(self, manager):
        result = tool_set_node(manager, NAME, "alpha")
        assert result["created"] is True
        assert result["path"] == "alpha"

    def test_create_with_ancestors(self, manager):
        result = tool_set_node(manager, NAME, "a.b.c")
        assert result["created"] is True
        assert "a" in result["ancestors_created"]
        assert "a.b" in result["ancestors_created"]
        assert manager.node_exists("a")
        assert manager.node_exists("a.b")
        assert manager.node_exists("a.b.c")

    def test_idempotent(self, manager):
        tool_set_node(manager, NAME, "x")
        result = tool_set_node(manager, NAME, "x")
        assert result["created"] is False

    def test_empty_path(self, manager):
        result = tool_set_node(manager, NAME, "")
        assert "error" in result


# ---------------------------------------------------------------------------
# tool_delete_node
# ---------------------------------------------------------------------------


class TestDeleteNode:
    def test_delete_leaf(self, populated):
        # database is a leaf node (props only)
        result = tool_delete_node(populated, NAME, "database")
        assert result["nodes_deleted"] == 1
        assert result["properties_deleted"] == 3

    def test_delete_subtree(self, manager):
        parse_dict_to_tree({"a": {"b": {"c": 1, "d": 2}, "e": 3}}, manager)
        result = tool_delete_node(manager, NAME, "a")
        assert result["nodes_deleted"] >= 2
        assert not manager.node_exists("a")
        assert not manager.node_exists("a.b")

    def test_delete_not_found(self, manager):
        result = tool_delete_node(manager, NAME, "ghost")
        assert "error" in result


# ---------------------------------------------------------------------------
# tool_list_keys
# ---------------------------------------------------------------------------


class TestListKeys:
    def test_basic(self, populated):
        result = tool_list_keys(populated, NAME, "database")
        keys = [k["key"] for k in result["keys"]]
        assert "host" in keys
        assert "port" in keys
        assert "enabled" in keys

    def test_pagination(self, manager):
        manager.create_node("n")
        for i in range(5):
            manager.set_property("n", f"k{i}", i)
        result = tool_list_keys(manager, NAME, "n", offset=0, limit=2)
        assert len(result["keys"]) == 2
        assert result["has_more"] is True

    def test_node_not_found(self, manager):
        result = tool_list_keys(manager, NAME, "missing")
        assert "error" in result


# ---------------------------------------------------------------------------
# tool_get_value
# ---------------------------------------------------------------------------


class TestGetValue:
    def test_existing_key(self, populated):
        result = tool_get_value(populated, NAME, "database", "host")
        assert result["value"] == "localhost"
        assert result["value_type"] == "string"

    def test_integer_value(self, populated):
        result = tool_get_value(populated, NAME, "database", "port")
        assert result["value"] == 5432
        assert result["value_type"] == "integer"

    def test_missing_key(self, populated):
        result = tool_get_value(populated, NAME, "database", "nope")
        assert "error" in result


# ---------------------------------------------------------------------------
# tool_set_value
# ---------------------------------------------------------------------------


class TestSetValue:
    def test_set_string(self, manager):
        manager.create_node("cfg")
        result = tool_set_value(manager, NAME, "cfg", "name", "hello")
        assert result["value"] == "hello"
        assert result["value_type"] == "string"

    def test_set_with_inference(self, manager):
        manager.create_node("cfg")
        result = tool_set_value(manager, NAME, "cfg", "count", "42")
        assert result["value"] == 42
        assert result["value_type"] == "integer"

    def test_set_explicit_type(self, manager):
        manager.create_node("cfg")
        result = tool_set_value(
            manager, NAME, "cfg", "flag", "true", value_type="string"
        )
        assert result["value"] == "true"
        assert result["value_type"] == "string"

    def test_auto_creates_node(self, manager):
        result = tool_set_value(manager, NAME, "new_node", "k", "v")
        assert result["value"] == "v"
        assert manager.node_exists("new_node")

    def test_overwrite(self, manager):
        manager.create_node("n")
        tool_set_value(manager, NAME, "n", "x", "1")
        result = tool_set_value(manager, NAME, "n", "x", "2")
        assert result["value"] == 2


# ---------------------------------------------------------------------------
# tool_delete_key
# ---------------------------------------------------------------------------


class TestDeleteKey:
    def test_delete_existing(self, populated):
        result = tool_delete_key(populated, NAME, "database", "host")
        assert result["deleted"] is True
        # Confirm gone
        check = tool_get_value(populated, NAME, "database", "host")
        assert "error" in check

    def test_delete_missing(self, populated):
        result = tool_delete_key(populated, NAME, "database", "nope")
        assert "error" in result


# ---------------------------------------------------------------------------
# reconstruct_tree
# ---------------------------------------------------------------------------


class TestReconstructTree:
    def test_basic_roundtrip(self, populated):
        tree = reconstruct_tree(populated)
        assert tree["database"]["host"] == "localhost"
        assert tree["database"]["port"] == 5432
        assert tree["database"]["enabled"] is True
        assert tree["logging"]["level"] == "info"

    def test_subtree(self, populated):
        tree = reconstruct_tree(populated, path="database")
        assert "database" in tree
        assert tree["database"]["host"] == "localhost"
        assert "logging" not in tree

    def test_array_of_tables(self, manager):
        data = {
            "servers": [
                {"name": "alpha", "port": 80},
                {"name": "beta", "port": 443},
            ]
        }
        parse_dict_to_tree(data, manager)
        tree = reconstruct_tree(manager)
        assert isinstance(tree["servers"], list)
        assert len(tree["servers"]) == 2
        assert tree["servers"][0]["name"] == "alpha"
        assert tree["servers"][1]["port"] == 443

    def test_nonexistent_path(self, manager):
        with pytest.raises(ValueError, match="not found"):
            reconstruct_tree(manager, path="nope")


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


class TestExportFormats:
    def test_export_json(self, populated):
        output = export_json(populated)
        parsed = json.loads(output)
        assert parsed["database"]["host"] == "localhost"

    def test_export_yaml(self, populated):
        output = export_yaml(populated)
        parsed = yaml.safe_load(output)
        assert parsed["logging"]["level"] == "info"

    def test_export_toml(self, populated):
        output = export_toml(populated)
        parsed = toml.loads(output)
        assert parsed["database"]["port"] == 5432


# ---------------------------------------------------------------------------
# tool_export_structured
# ---------------------------------------------------------------------------


class TestToolExportStructured:
    def test_json_format(self, populated):
        result = tool_export_structured(populated, NAME, "json")
        assert result["format"] == "json"
        content = json.loads(result["content"])
        assert content["database"]["host"] == "localhost"

    def test_yaml_format(self, populated):
        result = tool_export_structured(populated, NAME, "yaml")
        assert result["format"] == "yaml"
        content = yaml.safe_load(result["content"])
        assert content["logging"]["file"] == "/var/log/app.log"

    def test_toml_format(self, populated):
        result = tool_export_structured(populated, NAME, "toml")
        assert result["format"] == "toml"
        content = toml.loads(result["content"])
        assert content["database"]["enabled"] is True

    def test_unsupported_format(self, populated):
        result = tool_export_structured(populated, NAME, "xml")
        assert "error" in result

    def test_subtree_export(self, populated):
        result = tool_export_structured(populated, NAME, "json", path="database")
        content = json.loads(result["content"])
        assert "database" in content
        assert "logging" not in content

    def test_nonexistent_path(self, populated):
        result = tool_export_structured(populated, NAME, "json", path="nope")
        assert "error" in result


# ---------------------------------------------------------------------------
# Round-trip: parse TOML -> modify -> export TOML -> verify
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_toml_roundtrip_with_modification(self):
        """Parse a TOML file, modify a value, export, and verify."""
        original = {
            "title": "Test Config",
            "owner": {
                "name": "Alice",
                "active": True,
            },
            "database": {
                "host": "localhost",
                "port": 5432,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            toml.dump(original, f)
            tmp_path = f.name

        try:
            engine = create_engine("sqlite:///:memory:")
            create_tree_schema(engine)
            mgr = TreeStorageManager(engine)

            parse_toml_to_tree(tmp_path, mgr)

            # Modify: change host, add a new key
            tool_set_value(mgr, NAME, "database", "host", "db.example.com")
            tool_set_value(mgr, NAME, "database", "ssl", "true")

            # Export back to TOML
            result = tool_export_structured(mgr, NAME, "toml")
            exported = toml.loads(result["content"])

            # Original values preserved
            assert exported["owner"]["name"] == "Alice"
            assert exported["owner"]["active"] is True
            assert exported["database"]["port"] == 5432

            # Modifications applied
            assert exported["database"]["host"] == "db.example.com"
            assert exported["database"]["ssl"] is True  # inferred as boolean
        finally:
            os.unlink(tmp_path)

    def test_json_roundtrip_array_of_tables(self):
        """Parse JSON with array-of-tables, export, verify structure."""
        data = {
            "items": [
                {"id": 1, "label": "first"},
                {"id": 2, "label": "second"},
            ]
        }
        engine = create_engine("sqlite:///:memory:")
        create_tree_schema(engine)
        mgr = TreeStorageManager(engine)
        parse_dict_to_tree(data, mgr)

        output = export_json(mgr)
        parsed = json.loads(output)
        assert isinstance(parsed["items"], list)
        assert len(parsed["items"]) == 2
        assert parsed["items"][0]["id"] == 1
        assert parsed["items"][1]["label"] == "second"
