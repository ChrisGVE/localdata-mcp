"""Tests for tree_storage module: schema, types, paths, and TreeStorageManager."""

import math
from datetime import datetime

import pytest
from sqlalchemy import create_engine, text

from localdata_mcp.tree_storage import (
    NodeProperty,
    TreeNode,
    TreeStorageManager,
    ValueType,
    build_path,
    create_tree_schema,
    deserialize_value,
    escape_path_segment,
    infer_value_type,
    infer_value_type_from_string,
    parse_path,
    serialize_value,
    unescape_path_segment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with tree schema."""
    eng = create_engine("sqlite:///:memory:")
    create_tree_schema(eng)
    return eng


@pytest.fixture
def mgr(engine):
    """TreeStorageManager backed by in-memory SQLite."""
    return TreeStorageManager(engine)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchema:
    def test_create_schema(self, engine):
        with engine.connect() as conn:
            tables = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
            names = {r[0] for r in tables}
            assert "nodes" in names
            assert "properties" in names

    def test_create_schema_idempotent(self, engine):
        # Second call should not raise
        create_tree_schema(engine)

    def test_unique_node_path(self, engine):
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            conn.execute(
                text(
                    "INSERT INTO nodes (parent_id, name, path, depth, created_at, updated_at) "
                    "VALUES (NULL, 'a', 'a', 0, 0, 0)"
                )
            )
            with pytest.raises(Exception):
                conn.execute(
                    text(
                        "INSERT INTO nodes (parent_id, name, path, depth, created_at, updated_at) "
                        "VALUES (NULL, 'a', 'a', 0, 0, 0)"
                    )
                )

    def test_unique_property_key_per_node(self, engine):
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            conn.execute(
                text(
                    "INSERT INTO nodes (parent_id, name, path, depth, created_at, updated_at) "
                    "VALUES (NULL, 'a', 'a', 0, 0, 0)"
                )
            )
            node_id = conn.execute(text("SELECT id FROM nodes")).fetchone()[0]
            conn.execute(
                text(
                    "INSERT INTO properties (node_id, key, value, value_type, created_at, updated_at) "
                    "VALUES (:nid, 'k', 'v', 'string', 0, 0)"
                ),
                {"nid": node_id},
            )
            with pytest.raises(Exception):
                conn.execute(
                    text(
                        "INSERT INTO properties (node_id, key, value, value_type, created_at, updated_at) "
                        "VALUES (:nid, 'k', 'v2', 'string', 0, 0)"
                    ),
                    {"nid": node_id},
                )

    def test_cascade_delete(self, engine):
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            conn.execute(
                text(
                    "INSERT INTO nodes (parent_id, name, path, depth, created_at, updated_at) "
                    "VALUES (NULL, 'a', 'a', 0, 0, 0)"
                )
            )
            node_id = conn.execute(text("SELECT id FROM nodes")).fetchone()[0]
            conn.execute(
                text(
                    "INSERT INTO properties (node_id, key, value, value_type, created_at, updated_at) "
                    "VALUES (:nid, 'k', 'v', 'string', 0, 0)"
                ),
                {"nid": node_id},
            )
            conn.execute(text("DELETE FROM nodes WHERE id = :nid"), {"nid": node_id})
            conn.commit()
            count = conn.execute(text("SELECT COUNT(*) FROM properties")).fetchone()[0]
            assert count == 0


# ---------------------------------------------------------------------------
# ValueType tests
# ---------------------------------------------------------------------------


class TestValueType:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, ValueType.NULL),
            (True, ValueType.BOOLEAN),
            (False, ValueType.BOOLEAN),
            (42, ValueType.INTEGER),
            (3.14, ValueType.FLOAT),
            ("hello", ValueType.STRING),
            ([1, 2], ValueType.ARRAY),
        ],
    )
    def test_infer_value_type(self, value, expected):
        assert infer_value_type(value) == expected

    def test_infer_bool_before_int(self):
        # bool is subclass of int — must detect boolean first
        assert infer_value_type(True) == ValueType.BOOLEAN

    def test_infer_datetime(self):
        assert infer_value_type(datetime(2024, 1, 1)) == ValueType.DATETIME

    def test_infer_unsupported_raises(self):
        with pytest.raises(TypeError):
            infer_value_type({"a": 1})

    @pytest.mark.parametrize(
        "value, vtype",
        [
            (None, ValueType.NULL),
            (True, ValueType.BOOLEAN),
            (False, ValueType.BOOLEAN),
            (42, ValueType.INTEGER),
            (-7, ValueType.INTEGER),
            (3.14, ValueType.FLOAT),
            (float("inf"), ValueType.FLOAT),
            (float("-inf"), ValueType.FLOAT),
            ("hello world", ValueType.STRING),
            ("", ValueType.STRING),
            ([1, "a", True], ValueType.ARRAY),
            ([], ValueType.ARRAY),
        ],
    )
    def test_serialize_deserialize_roundtrip(self, value, vtype):
        serialized, orig = serialize_value(value, vtype)
        result = deserialize_value(serialized, vtype, orig)
        if vtype == ValueType.FLOAT and math.isnan(value):
            assert math.isnan(result)
        else:
            assert result == value

    def test_nan_roundtrip(self):
        serialized, orig = serialize_value(float("nan"), ValueType.FLOAT)
        result = deserialize_value(serialized, ValueType.FLOAT, orig)
        assert math.isnan(result)

    def test_datetime_roundtrip(self):
        dt = datetime(2024, 6, 15, 10, 30, 0)
        serialized, orig = serialize_value(dt, ValueType.DATETIME)
        result = deserialize_value(serialized, ValueType.DATETIME, orig)
        assert result == dt


class TestInferFromString:
    def test_boolean(self):
        assert infer_value_type_from_string("true") == (ValueType.BOOLEAN, True)
        assert infer_value_type_from_string("false") == (ValueType.BOOLEAN, False)

    def test_integer(self):
        assert infer_value_type_from_string("42") == (ValueType.INTEGER, 42)

    def test_float(self):
        assert infer_value_type_from_string("3.14") == (ValueType.FLOAT, 3.14)

    def test_array(self):
        assert infer_value_type_from_string("[1, 2, 3]") == (ValueType.ARRAY, [1, 2, 3])

    def test_null(self):
        assert infer_value_type_from_string("null") == (ValueType.NULL, None)

    def test_string_fallback(self):
        assert infer_value_type_from_string("hello") == (ValueType.STRING, "hello")


# ---------------------------------------------------------------------------
# Path escaping tests
# ---------------------------------------------------------------------------


class TestPathEscaping:
    def test_no_special_chars(self):
        assert escape_path_segment("abc") == "abc"

    def test_escape_dot(self):
        assert escape_path_segment("a.b") == "a\\.b"

    def test_escape_backslash(self):
        assert escape_path_segment("a\\b") == "a\\\\b"

    def test_escape_combined(self):
        assert escape_path_segment("a\\.b") == "a\\\\\\.b"

    def test_unescape_dot(self):
        assert unescape_path_segment("a\\.b") == "a.b"

    def test_unescape_backslash(self):
        assert unescape_path_segment("a\\\\b") == "a\\b"

    def test_unescape_trailing_backslash_raises(self):
        with pytest.raises(ValueError, match="Trailing backslash"):
            unescape_path_segment("abc\\")

    def test_unescape_invalid_escape_raises(self):
        with pytest.raises(ValueError, match="Invalid escape"):
            unescape_path_segment("a\\x")

    def test_build_path_empty(self):
        assert build_path([]) == ""

    def test_build_path_simple(self):
        assert build_path(["a", "b", "c"]) == "a.b.c"

    def test_build_path_with_dots_in_keys(self):
        assert build_path(["root", "a.b", "key"]) == "root.a\\.b.key"

    def test_parse_path_empty(self):
        assert parse_path("") == []

    def test_parse_path_simple(self):
        assert parse_path("a.b.c") == ["a", "b", "c"]

    def test_parse_path_escaped_dot(self):
        assert parse_path("root.a\\.b.key") == ["root", "a.b", "key"]

    def test_roundtrip(self):
        segments = ["root", "a.b", "c\\d", "leaf"]
        assert parse_path(build_path(segments)) == segments

    def test_unicode(self):
        segments = ["日本語", "données", "键"]
        assert parse_path(build_path(segments)) == segments

    def test_single_segment(self):
        assert parse_path("root") == ["root"]
        assert build_path(["root"]) == "root"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_tree_node_defaults(self):
        node = TreeNode(id=1, parent_id=None, name="root", path="root", depth=0)
        assert node.is_array_item is False
        assert node.created_at == 0.0

    def test_node_property_value(self):
        prop = NodeProperty(
            id=1,
            node_id=1,
            key="port",
            value="8080",
            value_type=ValueType.INTEGER,
        )
        assert prop.to_python_value() == 8080

    def test_tree_node_from_row(self, engine):
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            conn.execute(
                text(
                    "INSERT INTO nodes (parent_id, name, path, depth, is_array_item, created_at, updated_at) "
                    "VALUES (NULL, 'test', 'test', 0, 0, 1.0, 2.0)"
                )
            )
            conn.commit()
            row = conn.execute(text("SELECT * FROM nodes")).fetchone()
            node = TreeNode.from_row(row)
            assert node.name == "test"
            assert node.parent_id is None
            assert node.depth == 0


# ---------------------------------------------------------------------------
# TreeStorageManager tests
# ---------------------------------------------------------------------------


class TestTreeStorageManager:
    def test_create_node(self, mgr):
        node = mgr.create_node("server")
        assert node.name == "server"
        assert node.path == "server"
        assert node.depth == 0
        assert node.parent_id is None

    def test_create_nested_node_auto_ancestors(self, mgr):
        node = mgr.create_node("server.ssl.certs")
        assert node.name == "certs"
        assert node.depth == 2
        # Ancestors should exist
        parent = mgr.get_node("server.ssl")
        assert parent is not None
        root = mgr.get_node("server")
        assert root is not None

    def test_create_node_idempotent(self, mgr):
        n1 = mgr.create_node("server")
        n2 = mgr.create_node("server")
        assert n1.id == n2.id

    def test_get_node_nonexistent(self, mgr):
        assert mgr.get_node("nonexistent") is None

    def test_node_exists(self, mgr):
        assert not mgr.node_exists("server")
        mgr.create_node("server")
        assert mgr.node_exists("server")

    def test_get_children_root(self, mgr):
        mgr.create_node("server")
        mgr.create_node("database")
        mgr.create_node("logging")
        children = mgr.get_children()
        names = [c.name for c in children]
        assert sorted(names) == ["database", "logging", "server"]

    def test_get_children_nested(self, mgr):
        mgr.create_node("server.ssl")
        mgr.create_node("server.cors")
        children = mgr.get_children("server")
        names = [c.name for c in children]
        assert sorted(names) == ["cors", "ssl"]

    def test_get_children_pagination(self, mgr):
        for i in range(10):
            mgr.create_node(f"item_{i:02d}")
        page1 = mgr.get_children(offset=0, limit=3)
        page2 = mgr.get_children(offset=3, limit=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].name != page2[0].name

    def test_get_children_count(self, mgr):
        mgr.create_node("a")
        mgr.create_node("b")
        mgr.create_node("c")
        assert mgr.get_children_count() == 3

    def test_delete_node(self, mgr):
        mgr.create_node("server.ssl")
        mgr.set_property("server", "port", 8080)
        mgr.set_property("server.ssl", "enabled", True)
        nodes_del, props_del = mgr.delete_node("server")
        assert nodes_del == 2  # server + server.ssl
        assert props_del == 2  # port + enabled
        assert mgr.get_node("server") is None
        assert mgr.get_node("server.ssl") is None

    def test_delete_nonexistent(self, mgr):
        nodes_del, props_del = mgr.delete_node("nonexistent")
        assert nodes_del == 0
        assert props_del == 0

    def test_set_and_get_property(self, mgr):
        mgr.create_node("server")
        mgr.set_property("server", "port", 8080)
        prop = mgr.get_property("server", "port")
        assert prop is not None
        assert prop.to_python_value() == 8080
        assert prop.value_type == ValueType.INTEGER

    def test_set_property_upsert(self, mgr):
        mgr.create_node("server")
        mgr.set_property("server", "port", 8080)
        mgr.set_property("server", "port", 9090)
        prop = mgr.get_property("server", "port")
        assert prop.to_python_value() == 9090

    def test_set_property_auto_creates_node(self, mgr):
        mgr.set_property("newnode", "key", "value")
        assert mgr.node_exists("newnode")
        prop = mgr.get_property("newnode", "key")
        assert prop.to_python_value() == "value"

    def test_list_properties(self, mgr):
        mgr.create_node("server")
        mgr.set_property("server", "host", "localhost")
        mgr.set_property("server", "port", 8080)
        mgr.set_property("server", "debug", True)
        props = mgr.list_properties("server")
        keys = [p.key for p in props]
        assert sorted(keys) == ["debug", "host", "port"]

    def test_list_properties_pagination(self, mgr):
        mgr.create_node("node")
        for i in range(10):
            mgr.set_property("node", f"key_{i:02d}", i)
        page1 = mgr.list_properties("node", offset=0, limit=3)
        page2 = mgr.list_properties("node", offset=3, limit=3)
        assert len(page1) == 3
        assert len(page2) == 3

    def test_get_property_count(self, mgr):
        mgr.create_node("node")
        mgr.set_property("node", "a", 1)
        mgr.set_property("node", "b", 2)
        assert mgr.get_property_count("node") == 2

    def test_delete_property(self, mgr):
        mgr.create_node("node")
        mgr.set_property("node", "key", "val")
        assert mgr.delete_property("node", "key") is True
        assert mgr.get_property("node", "key") is None

    def test_delete_property_nonexistent(self, mgr):
        mgr.create_node("node")
        assert mgr.delete_property("node", "nope") is False

    def test_get_tree_stats(self, mgr):
        mgr.create_node("server")
        mgr.create_node("server.ssl")
        mgr.create_node("database")
        mgr.set_property("server", "port", 8080)
        mgr.set_property("server.ssl", "enabled", True)
        stats = mgr.get_tree_stats()
        assert stats["total_nodes"] == 3
        assert stats["total_properties"] == 2
        assert stats["max_depth"] == 1
        assert stats["root_count"] == 2
        assert sorted(stats["root_nodes"]) == ["database", "server"]

    def test_all_value_types_stored(self, mgr):
        """Ensure every value type round-trips through the database."""
        mgr.create_node("types")
        mgr.set_property("types", "str", "hello")
        mgr.set_property("types", "int", 42)
        mgr.set_property("types", "float", 3.14)
        mgr.set_property("types", "bool", True)
        mgr.set_property("types", "arr", [1, 2, 3])
        mgr.set_property("types", "null", None, ValueType.NULL)

        assert mgr.get_property("types", "str").to_python_value() == "hello"
        assert mgr.get_property("types", "int").to_python_value() == 42
        assert mgr.get_property("types", "float").to_python_value() == 3.14
        assert mgr.get_property("types", "bool").to_python_value() is True
        assert mgr.get_property("types", "arr").to_python_value() == [1, 2, 3]
        assert mgr.get_property("types", "null").to_python_value() is None
