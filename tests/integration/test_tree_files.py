"""Integration tests for tree storage (YAML, TOML, JSON)."""

import json

import pytest

from .mcp_test_client import call_tool

pytestmark = [pytest.mark.integration]


class TestYAMLTree:
    def test_connect_and_navigate(self, tmp_path):
        path = str(tmp_path / "config.yaml")
        with open(path, "w") as f:
            f.write(
                "database:\n"
                "  host: localhost\n"
                "  port: 5432\n"
                "  credentials:\n"
                "    user: admin\n"
                "    password: secret\n"
                "logging:\n"
                "  level: info\n"
            )
        call_tool(
            "connect_database",
            {"name": "yaml_tree", "db_type": "yaml", "conn_string": path},
        )
        try:
            # Get root children (no path = root level)
            result = call_tool("get_children", {"name": "yaml_tree"})
            assert "database" in str(result)

            # Navigate to nested value (paths have no leading slash)
            result = call_tool(
                "get_value",
                {"name": "yaml_tree", "path": "database", "key": "host"},
            )
            assert "localhost" in str(result)

            # Deep nesting (dot-separated path for nested nodes)
            result = call_tool(
                "get_value",
                {
                    "name": "yaml_tree",
                    "path": "database.credentials",
                    "key": "user",
                },
            )
            assert "admin" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "yaml_tree"})

    def test_list_keys(self, tmp_path):
        path = str(tmp_path / "keys.yaml")
        with open(path, "w") as f:
            f.write("server:\n  host: 0.0.0.0\n  port: 8080\n  debug: true\n")
        call_tool(
            "connect_database",
            {"name": "yaml_keys", "db_type": "yaml", "conn_string": path},
        )
        try:
            result = call_tool("list_keys", {"name": "yaml_keys", "path": "server"})
            keys = [k["key"] for k in result.get("keys", [])]
            assert "host" in keys
            assert "port" in keys
        finally:
            call_tool("disconnect_database", {"name": "yaml_keys"})

    def test_export_as_json(self, tmp_path):
        path = str(tmp_path / "exp.yaml")
        with open(path, "w") as f:
            f.write("a:\n  b: 1\n  c: 2\n")
        call_tool(
            "connect_database",
            {"name": "yaml_exp", "db_type": "yaml", "conn_string": path},
        )
        try:
            result = call_tool(
                "export_structured", {"name": "yaml_exp", "format": "json"}
            )
            assert "content" in result
            # Verify the exported JSON is parseable
            parsed = json.loads(result["content"])
            assert "a" in parsed
        finally:
            call_tool("disconnect_database", {"name": "yaml_exp"})

    def test_export_as_markdown(self, tmp_path):
        path = str(tmp_path / "md.yaml")
        with open(path, "w") as f:
            f.write("root:\n  child1: val1\n  child2: val2\n")
        call_tool(
            "connect_database",
            {"name": "yaml_md", "db_type": "yaml", "conn_string": path},
        )
        try:
            result = call_tool(
                "export_structured", {"name": "yaml_md", "format": "markdown"}
            )
            assert "content" in result
            assert result["format"] == "markdown"
        finally:
            call_tool("disconnect_database", {"name": "yaml_md"})

    def test_get_node_root_summary(self, tmp_path):
        path = str(tmp_path / "summary.yaml")
        with open(path, "w") as f:
            f.write("x:\n  y: 1\nz: 2\n")
        call_tool(
            "connect_database",
            {"name": "yaml_summary", "db_type": "yaml", "conn_string": path},
        )
        try:
            result = call_tool("get_node", {"name": "yaml_summary"})
            assert "total_nodes" in result
            assert result["total_nodes"] > 0
        finally:
            call_tool("disconnect_database", {"name": "yaml_summary"})


class TestTOMLTree:
    def test_connect_and_navigate(self, tmp_path):
        path = str(tmp_path / "config.toml")
        with open(path, "w") as f:
            f.write(
                '[server]\nhost = "0.0.0.0"\nport = 8080\n\n'
                '[database]\nurl = "sqlite:///app.db"\n'
            )
        call_tool(
            "connect_database",
            {"name": "toml_tree", "db_type": "toml", "conn_string": path},
        )
        try:
            result = call_tool("get_children", {"name": "toml_tree"})
            children_names = [c["name"] for c in result.get("children", [])]
            assert "server" in children_names
            assert "database" in children_names
        finally:
            call_tool("disconnect_database", {"name": "toml_tree"})

    def test_nested_value(self, tmp_path):
        path = str(tmp_path / "nested.toml")
        with open(path, "w") as f:
            f.write('[app]\nname = "myapp"\nversion = "1.0"\n')
        call_tool(
            "connect_database",
            {"name": "toml_nested", "db_type": "toml", "conn_string": path},
        )
        try:
            result = call_tool(
                "get_value",
                {"name": "toml_nested", "path": "app", "key": "name"},
            )
            assert "myapp" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "toml_nested"})

    def test_export_as_toml(self, tmp_path):
        path = str(tmp_path / "round.toml")
        with open(path, "w") as f:
            f.write('[section]\nkey = "value"\n')
        call_tool(
            "connect_database",
            {"name": "toml_export", "db_type": "toml", "conn_string": path},
        )
        try:
            result = call_tool(
                "export_structured", {"name": "toml_export", "format": "toml"}
            )
            assert "content" in result
        finally:
            call_tool("disconnect_database", {"name": "toml_export"})


class TestJSONTree:
    def test_nested_json_tree(self, tmp_path):
        path = str(tmp_path / "tree.json")
        data = {
            "config": {
                "db": {"host": "localhost", "port": 5432},
                "cache": {"ttl": 3600},
            }
        }
        with open(path, "w") as f:
            json.dump(data, f)
        call_tool(
            "connect_database",
            {"name": "json_tree", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool("get_children", {"name": "json_tree"})
            assert "config" in str(result)

            result = call_tool(
                "get_value",
                {"name": "json_tree", "path": "config.db", "key": "host"},
            )
            assert "localhost" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "json_tree"})

    def test_deep_nesting(self, tmp_path):
        """10+ levels of nesting."""
        path = str(tmp_path / "deep.json")
        nested = {"level": 0}
        current = nested
        for i in range(1, 12):
            current["child"] = {"level": i}
            current = current["child"]
        with open(path, "w") as f:
            json.dump(nested, f)
        call_tool(
            "connect_database",
            {"name": "json_deep", "db_type": "json", "conn_string": path},
        )
        try:
            # Verify root node summary reflects the depth
            result = call_tool("get_node", {"name": "json_deep"})
            assert result["max_depth"] >= 10

            # Drill into a nested child (dot-separated path, no leading slash)
            deep_path = ".".join(["child"] * 5)
            result = call_tool("get_node", {"name": "json_deep", "path": deep_path})
            assert "error" not in result
        finally:
            call_tool("disconnect_database", {"name": "json_deep"})

    def test_export_round_trip(self, tmp_path):
        path = str(tmp_path / "rt.json")
        data = {"a": {"b": 1, "c": "hello"}}
        with open(path, "w") as f:
            json.dump(data, f)
        call_tool(
            "connect_database",
            {"name": "json_rt", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool(
                "export_structured", {"name": "json_rt", "format": "json"}
            )
            assert "content" in result
            parsed = json.loads(result["content"])
            assert parsed["a"]["c"] == "hello"
        finally:
            call_tool("disconnect_database", {"name": "json_rt"})

    def test_set_and_get_value(self, tmp_path):
        """Mutate a tree value and verify it persists."""
        path = str(tmp_path / "mut.json")
        data = {"settings": {"theme": "dark"}}
        with open(path, "w") as f:
            json.dump(data, f)
        call_tool(
            "connect_database",
            {"name": "json_mut", "db_type": "json", "conn_string": path},
        )
        try:
            call_tool(
                "set_value",
                {
                    "name": "json_mut",
                    "path": "/settings",
                    "key": "theme",
                    "value": "light",
                },
            )
            result = call_tool(
                "get_value",
                {"name": "json_mut", "path": "/settings", "key": "theme"},
            )
            assert result["value"] == "light"
        finally:
            call_tool("disconnect_database", {"name": "json_mut"})
