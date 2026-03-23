"""Tests for tree_parsers module: TOML, JSON, and YAML parsing into tree storage."""

import json
import os
import tempfile

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
    parse_json_to_tree,
    parse_toml_to_tree,
    parse_yaml_to_tree,
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
def tmp_dir():
    """Temporary directory cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# TOML tests
# ---------------------------------------------------------------------------


class TestTomlParsing:
    """Tests for TOML file parsing."""

    def test_nested_config(self, manager, tmp_dir):
        """Nested TOML sections become child nodes with properties."""
        data = {
            "server": {"host": "localhost", "port": 8080},
            "database": {"url": "sqlite:///db.sqlite", "pool_size": 5},
            "logging": {"level": "INFO", "file": "/var/log/app.log"},
        }
        path = os.path.join(tmp_dir, "config.toml")
        with open(path, "w") as fh:
            toml.dump(data, fh)

        parse_toml_to_tree(path, manager)

        # Verify nodes exist
        assert manager.node_exists("server")
        assert manager.node_exists("database")
        assert manager.node_exists("logging")

        # Verify properties
        prop = manager.get_property("server", "host")
        assert prop is not None
        assert prop.to_python_value() == "localhost"

        prop = manager.get_property("server", "port")
        assert prop is not None
        assert prop.to_python_value() == 8080
        assert prop.value_type == ValueType.INTEGER

        prop = manager.get_property("database", "pool_size")
        assert prop is not None
        assert prop.to_python_value() == 5

    def test_array_of_tables(self, manager, tmp_dir):
        """TOML array-of-tables creates numbered child nodes."""
        content = """\
[[servers]]
name = "alpha"
ip = "10.0.0.1"

[[servers]]
name = "beta"
ip = "10.0.0.2"
"""
        path = os.path.join(tmp_dir, "servers.toml")
        with open(path, "w") as fh:
            fh.write(content)

        parse_toml_to_tree(path, manager)

        assert manager.node_exists("servers")
        assert manager.node_exists(build_path(["servers", "0"]))
        assert manager.node_exists(build_path(["servers", "1"]))

        node_0 = manager.get_node(build_path(["servers", "0"]))
        assert node_0 is not None
        assert node_0.is_array_item is True

        prop = manager.get_property(build_path(["servers", "0"]), "name")
        assert prop is not None
        assert prop.to_python_value() == "alpha"

        prop = manager.get_property(build_path(["servers", "1"]), "ip")
        assert prop is not None
        assert prop.to_python_value() == "10.0.0.2"


# ---------------------------------------------------------------------------
# JSON tests
# ---------------------------------------------------------------------------


class TestJsonParsing:
    """Tests for JSON file parsing."""

    def test_nested_objects(self, manager, tmp_dir):
        """Nested JSON objects become child nodes."""
        data = {
            "app": {
                "name": "myapp",
                "version": "1.0.0",
                "settings": {"debug": True, "timeout": 30.5},
            }
        }
        path = os.path.join(tmp_dir, "config.json")
        with open(path, "w") as fh:
            json.dump(data, fh)

        parse_json_to_tree(path, manager)

        assert manager.node_exists("app")
        assert manager.node_exists(build_path(["app", "settings"]))

        prop = manager.get_property("app", "name")
        assert prop is not None
        assert prop.to_python_value() == "myapp"

        prop = manager.get_property(build_path(["app", "settings"]), "debug")
        assert prop is not None
        assert prop.to_python_value() is True
        assert prop.value_type == ValueType.BOOLEAN

        prop = manager.get_property(build_path(["app", "settings"]), "timeout")
        assert prop is not None
        assert prop.to_python_value() == 30.5
        assert prop.value_type == ValueType.FLOAT

    def test_root_array_of_objects(self, manager, tmp_dir):
        """JSON root array of objects creates numbered root children."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        path = os.path.join(tmp_dir, "users.json")
        with open(path, "w") as fh:
            json.dump(data, fh)

        parse_json_to_tree(path, manager)

        assert manager.node_exists("root")
        assert manager.node_exists(build_path(["root", "0"]))
        assert manager.node_exists(build_path(["root", "1"]))

        node_0 = manager.get_node(build_path(["root", "0"]))
        assert node_0 is not None
        assert node_0.is_array_item is True

        prop = manager.get_property(build_path(["root", "0"]), "name")
        assert prop is not None
        assert prop.to_python_value() == "Alice"

        prop = manager.get_property(build_path(["root", "1"]), "id")
        assert prop is not None
        assert prop.to_python_value() == 2


# ---------------------------------------------------------------------------
# YAML tests
# ---------------------------------------------------------------------------


class TestYamlParsing:
    """Tests for YAML file parsing."""

    def test_yaml_config(self, manager, tmp_dir):
        """Single-document YAML is parsed like a dict."""
        data = {
            "database": {"host": "db.example.com", "port": 5432},
            "cache": {"enabled": True, "ttl": 300},
        }
        path = os.path.join(tmp_dir, "config.yaml")
        with open(path, "w") as fh:
            yaml.dump(data, fh)

        parse_yaml_to_tree(path, manager)

        assert manager.node_exists("database")
        assert manager.node_exists("cache")

        prop = manager.get_property("database", "host")
        assert prop is not None
        assert prop.to_python_value() == "db.example.com"

        prop = manager.get_property("cache", "enabled")
        assert prop is not None
        assert prop.to_python_value() is True

    def test_multi_document_yaml(self, manager, tmp_dir):
        """Multi-document YAML creates doc_0, doc_1, ... root nodes."""
        content = """\
name: first
value: 1
---
name: second
value: 2
---
name: third
value: 3
"""
        path = os.path.join(tmp_dir, "multi.yaml")
        with open(path, "w") as fh:
            fh.write(content)

        parse_yaml_to_tree(path, manager)

        assert manager.node_exists("doc_0")
        assert manager.node_exists("doc_1")
        assert manager.node_exists("doc_2")

        prop = manager.get_property("doc_0", "name")
        assert prop is not None
        assert prop.to_python_value() == "first"

        prop = manager.get_property("doc_1", "value")
        assert prop is not None
        assert prop.to_python_value() == 2

        prop = manager.get_property("doc_2", "name")
        assert prop is not None
        assert prop.to_python_value() == "third"


# ---------------------------------------------------------------------------
# Value type preservation tests
# ---------------------------------------------------------------------------


class TestValueTypePreservation:
    """Verify all value types survive a round-trip through the tree."""

    def test_all_scalar_types(self, manager, tmp_dir):
        """String, integer, float, boolean, null are stored with correct types."""
        data = {
            "str_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "null_val": None,
        }
        path = os.path.join(tmp_dir, "types.json")
        with open(path, "w") as fh:
            json.dump(data, fh)

        parse_json_to_tree(path, manager)

        # Root-level scalars go on a "root" node
        prop = manager.get_property("root", "str_val")
        assert prop is not None
        assert prop.value_type == ValueType.STRING
        assert prop.to_python_value() == "hello"

        prop = manager.get_property("root", "int_val")
        assert prop is not None
        assert prop.value_type == ValueType.INTEGER
        assert prop.to_python_value() == 42

        prop = manager.get_property("root", "float_val")
        assert prop is not None
        assert prop.value_type == ValueType.FLOAT
        assert prop.to_python_value() == 3.14

        prop = manager.get_property("root", "bool_val")
        assert prop is not None
        assert prop.value_type == ValueType.BOOLEAN
        assert prop.to_python_value() is True

        prop = manager.get_property("root", "null_val")
        assert prop is not None
        assert prop.value_type == ValueType.NULL
        assert prop.to_python_value() is None

    def test_scalar_list_stored_as_array(self, manager, tmp_dir):
        """A list of scalars is stored as an ARRAY property."""
        data = {"tags": ["alpha", "beta", "gamma"]}
        path = os.path.join(tmp_dir, "tags.json")
        with open(path, "w") as fh:
            json.dump(data, fh)

        parse_json_to_tree(path, manager)

        prop = manager.get_property("root", "tags")
        assert prop is not None
        assert prop.value_type == ValueType.ARRAY
        assert prop.to_python_value() == ["alpha", "beta", "gamma"]

    def test_nested_dict_with_mixed_types(self, manager, tmp_dir):
        """Nested dict with mixed scalar types preserves each correctly."""
        data = {
            "config": {
                "name": "test",
                "count": 10,
                "ratio": 0.75,
                "enabled": False,
            }
        }
        path = os.path.join(tmp_dir, "mixed.toml")
        with open(path, "w") as fh:
            toml.dump(data, fh)

        parse_toml_to_tree(path, manager)

        assert manager.node_exists("config")

        prop = manager.get_property("config", "name")
        assert prop.value_type == ValueType.STRING

        prop = manager.get_property("config", "count")
        assert prop.value_type == ValueType.INTEGER

        prop = manager.get_property("config", "ratio")
        assert prop.value_type == ValueType.FLOAT

        prop = manager.get_property("config", "enabled")
        assert prop.value_type == ValueType.BOOLEAN
        assert prop.to_python_value() is False
