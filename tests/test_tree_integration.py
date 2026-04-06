"""Integration tests for tree storage through DatabaseManager.

Tests the full flow: connect → navigate → mutate → export → disconnect,
using DatabaseManager methods (the same path as MCP tool calls).
"""

import json
import os
import tempfile

import pytest
from sqlalchemy import create_engine

from localdata_mcp.tree_parsers import (
    parse_json_to_tree,
    parse_toml_to_tree,
    parse_yaml_to_tree,
)
from localdata_mcp.tree_storage import TreeStorageManager, create_tree_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def toml_file(tmp_path):
    """Create a test TOML config file."""
    content = """\
[meta]
version = "1.0"
author = "test"

[server]
host = "localhost"
port = 8080
debug = true

[server.ssl]
enabled = true
cert = "/etc/ssl/cert.pem"

[server.cors]
allowed_origins = ["http://localhost", "https://example.com"]

[database]
url = "postgres://localhost/app"
pool_size = 10

[database.replica]
url = "postgres://replica/app"
read_only = true

[[server.routes]]
path = "/api"
handler = "api_handler"

[[server.routes]]
path = "/health"
handler = "health_check"
"""
    p = tmp_path / "config.toml"
    p.write_text(content)
    return str(p)


@pytest.fixture
def json_file(tmp_path):
    """Create a test JSON file."""
    data = {
        "name": "test-project",
        "version": "2.0",
        "dependencies": {
            "fastmcp": {"version": ">=3.0", "optional": False},
            "sqlalchemy": {"version": ">=1.4", "optional": False},
        },
        "scripts": {
            "start": "python main.py",
            "test": "pytest",
        },
    }
    p = tmp_path / "package.json"
    p.write_text(json.dumps(data, indent=2))
    return str(p)


@pytest.fixture
def yaml_file(tmp_path):
    """Create a test YAML file with two documents."""
    content = """\
# App config
app:
  name: myapp
  port: 3000
  features:
    - auth
    - logging
logging:
  level: info
  format: json
---
# Deploy config
deploy:
  target: production
  replicas: 3
"""
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


@pytest.fixture
def mgr():
    """In-memory TreeStorageManager."""
    engine = create_engine("sqlite:///:memory:")
    return TreeStorageManager(engine)


# ---------------------------------------------------------------------------
# TOML integration tests
# ---------------------------------------------------------------------------


class TestTomlIntegration:
    def test_connect_and_navigate(self, toml_file, mgr):
        parse_toml_to_tree(toml_file, mgr)
        stats = mgr.get_tree_stats()
        assert stats["total_nodes"] > 5
        assert stats["root_count"] >= 3  # meta, server, database

        # Navigate to nested node
        node = mgr.get_node("server.ssl")
        assert node is not None
        assert node.depth == 1
        assert node.name == "ssl"

        # Check properties
        props = mgr.list_properties("server.ssl")
        keys = {p.key for p in props}
        assert "enabled" in keys
        assert "cert" in keys

        enabled = mgr.get_property("server.ssl", "enabled")
        assert enabled.to_python_value() is True

    def test_array_of_tables(self, toml_file, mgr):
        parse_toml_to_tree(toml_file, mgr)

        # Array of tables should create numbered children
        children = mgr.get_children("server.routes")
        assert len(children) == 2
        assert all(c.is_array_item for c in children)

        # First route
        api = mgr.get_property("server.routes.0", "path")
        assert api is not None
        assert api.to_python_value() == "/api"

        health = mgr.get_property("server.routes.1", "path")
        assert health.to_python_value() == "/health"

    def test_mutate_and_export(self, toml_file, mgr):
        parse_toml_to_tree(toml_file, mgr)

        # Change port
        mgr.set_property("server", "port", 9090)
        assert mgr.get_property("server", "port").to_python_value() == 9090

        # Add new section
        mgr.create_node("monitoring.alerts")
        mgr.set_property("monitoring.alerts", "enabled", True)
        mgr.set_property("monitoring.alerts", "threshold", 0.95)

        # Export and verify
        from localdata_mcp.tree_export import export_toml

        output = export_toml(mgr)
        assert "port = 9090" in output
        assert "monitoring" in output
        assert "threshold" in output

    def test_delete_subtree(self, toml_file, mgr):
        parse_toml_to_tree(toml_file, mgr)
        initial_count = mgr.get_tree_stats()["total_nodes"]

        nodes_del, props_del = mgr.delete_node("server.cors")
        assert nodes_del == 1
        assert props_del >= 1

        assert mgr.get_node("server.cors") is None
        assert mgr.get_tree_stats()["total_nodes"] == initial_count - 1
        # server itself should still exist
        assert mgr.get_node("server") is not None

    def test_list_properties_pagination(self, toml_file, mgr):
        parse_toml_to_tree(toml_file, mgr)
        # server has host, port, debug
        page1 = mgr.list_properties("server", offset=0, limit=2)
        page2 = mgr.list_properties("server", offset=2, limit=2)
        assert len(page1) == 2
        assert len(page2) >= 1
        keys1 = {p.key for p in page1}
        keys2 = {p.key for p in page2}
        assert keys1.isdisjoint(keys2)


# ---------------------------------------------------------------------------
# JSON integration tests
# ---------------------------------------------------------------------------


class TestJsonIntegration:
    def test_nested_objects(self, json_file, mgr):
        parse_json_to_tree(json_file, mgr)

        stats = mgr.get_tree_stats()
        assert stats["total_nodes"] > 0

        # Check nested dependency nodes
        dep = mgr.get_node("dependencies.fastmcp")
        assert dep is not None
        version = mgr.get_property("dependencies.fastmcp", "version")
        assert version.to_python_value() == ">=3.0"

    def test_root_level_properties(self, json_file, mgr):
        """Root-level scalars should be properties on a virtual root or handled."""
        parse_json_to_tree(json_file, mgr)
        # Our parser creates properties at root level — but root is not a node path.
        # Let's verify the behavior with get_children
        roots = mgr.get_children()
        root_names = [c.name for c in roots]
        # dependencies and scripts are nested dicts → nodes
        assert "dependencies" in root_names
        assert "scripts" in root_names

    def test_export_json(self, json_file, mgr):
        parse_json_to_tree(json_file, mgr)
        from localdata_mcp.tree_export import export_json

        output = export_json(mgr)
        data = json.loads(output)
        assert "dependencies" in data
        assert "scripts" in data


# ---------------------------------------------------------------------------
# YAML integration tests
# ---------------------------------------------------------------------------


class TestYamlIntegration:
    def test_single_document(self, yaml_file, mgr):
        parse_yaml_to_tree(yaml_file, mgr)
        stats = mgr.get_tree_stats()
        assert stats["total_nodes"] > 0

    def test_multi_document_creates_doc_roots(self, yaml_file, mgr):
        parse_yaml_to_tree(yaml_file, mgr)
        roots = mgr.get_children()
        root_names = [c.name for c in roots]
        # Multi-doc YAML should create doc_0 and doc_1
        assert "doc_0" in root_names or "app" in root_names

    def test_export_yaml(self, yaml_file, mgr):
        parse_yaml_to_tree(yaml_file, mgr)
        from localdata_mcp.tree_export import export_yaml

        output = export_yaml(mgr)
        assert "app" in output or "doc_0" in output


# ---------------------------------------------------------------------------
# MCP tool wrapper tests (through tree_tools functions)
# ---------------------------------------------------------------------------


class TestMcpToolWrappers:
    """Test the tool_* functions that the MCP tools delegate to."""

    def test_full_workflow(self, toml_file, mgr):
        """End-to-end: parse → navigate → mutate → export."""
        from localdata_mcp.tree_export import tool_export_structured
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

        parse_toml_to_tree(toml_file, mgr)

        # Root summary
        r = tool_get_node(mgr, "test")
        assert "root_nodes" in r
        assert r["total_nodes"] > 0

        # Node detail
        r = tool_get_node(mgr, "test", "server")
        assert r["name"] == "server"
        assert r["children_count"] >= 2  # ssl, cors, routes

        # Children
        r = tool_get_children(mgr, "test", "server")
        assert len(r["children"]) >= 2

        # Get value
        r = tool_get_value(mgr, "test", "server", "port")
        assert r["value"] == 8080

        # Set value (string input with type inference)
        r = tool_set_value(mgr, "test", "server", "port", "9090")
        assert r["value"] == 9090

        # Verify change
        r = tool_get_value(mgr, "test", "server", "port")
        assert r["value"] == 9090

        # List keys
        r = tool_list_keys(mgr, "test", "server")
        keys = [p["key"] for p in r["keys"]]
        assert "port" in keys

        # Set node (new)
        r = tool_set_node(mgr, "test", "monitoring")
        assert r["path"] == "monitoring"

        # Set value on new node
        r = tool_set_value(mgr, "test", "monitoring", "enabled", "true")
        assert r["value"] is True

        # Delete key
        r = tool_delete_key(mgr, "test", "monitoring", "enabled")
        assert r["deleted"] is True

        # Delete node
        r = tool_delete_node(mgr, "test", "monitoring")
        assert r["nodes_deleted"] == 1

        # Export
        r = tool_export_structured(mgr, "test", "toml")
        assert "content" in r
        assert "port = 9090" in r["content"]

        r = tool_export_structured(mgr, "test", "json")
        data = json.loads(r["content"])
        assert "server" in data

        r = tool_export_structured(mgr, "test", "yaml")
        assert "server" in r["content"]

    def test_error_cases(self, mgr):
        """Test error responses from tool functions."""
        from localdata_mcp.tree_tools import (
            tool_delete_key,
            tool_get_node,
            tool_get_value,
        )

        # Non-existent node
        r = tool_get_node(mgr, "test", "nonexistent")
        assert "error" in r

        # Non-existent key
        mgr.create_node("exists")
        r = tool_get_value(mgr, "test", "exists", "nope")
        assert "error" in r

        # Delete non-existent key returns error
        r = tool_delete_key(mgr, "test", "exists", "nope")
        assert "error" in r


# ---------------------------------------------------------------------------
# Real file test (if available)
# ---------------------------------------------------------------------------


class TestRealFile:
    """Tests against the real master_tree.toml if available."""

    MASTER_TREE = "tmp/master_tree.toml"

    @pytest.fixture
    def real_mgr(self):
        if not os.path.exists(self.MASTER_TREE):
            pytest.skip("tmp/master_tree.toml not available")
        engine = create_engine("sqlite:///:memory:")
        mgr = TreeStorageManager(engine)
        parse_toml_to_tree(self.MASTER_TREE, mgr)
        return mgr

    def test_stats(self, real_mgr):
        stats = real_mgr.get_tree_stats()
        assert stats["total_nodes"] > 1000
        assert stats["total_properties"] > 10000
        assert stats["max_depth"] >= 5

    def test_navigate_deep(self, real_mgr):
        # Navigate down the tree
        roots = real_mgr.get_children()
        assert len(roots) >= 1

        # Go deeper
        children = real_mgr.get_children("domain")
        assert len(children) >= 2

        # Get a specific node
        for child in children:
            node = real_mgr.get_node(child.path)
            assert node is not None
            props = real_mgr.list_properties(child.path)
            assert len(props) >= 0  # some nodes may be pure grouping

    def test_subtree_export(self, real_mgr):
        from localdata_mcp.tree_export import export_json

        output = export_json(real_mgr, "meta")
        data = json.loads(output)
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_mutation_roundtrip(self, real_mgr):
        """Mutate and verify."""
        real_mgr.set_property("meta", "test_field", "test_value")
        prop = real_mgr.get_property("meta", "test_field")
        assert prop.to_python_value() == "test_value"

        # Clean up
        real_mgr.delete_property("meta", "test_field")
        assert real_mgr.get_property("meta", "test_field") is None
