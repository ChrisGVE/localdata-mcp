"""Tests for graph_storage module: schema, dataclasses, and ValueType reuse."""

import time

import pytest
from sqlalchemy import create_engine, text

from localdata_mcp.graph_storage import (
    GraphEdge,
    GraphNode,
    GraphProperty,
    create_graph_schema,
)
from localdata_mcp.tree_storage import ValueType, deserialize_value, serialize_value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_INS_NODE = (
    "INSERT INTO graph_nodes (node_id, label, created_at, updated_at) "
    "VALUES (:nid, :label, :ca, :ua)"
)
_INS_EDGE = (
    "INSERT INTO graph_edges "
    "(source_id, target_id, label, weight, created_at, updated_at) "
    "VALUES (:s, :t, :l, :w, :ca, :ua)"
)
_INS_PROP = (
    "INSERT INTO graph_properties "
    "(owner_type, owner_id, key, value, value_type, created_at, updated_at) "
    "VALUES (:ot, :oid, :k, :v, :vt, :ca, :ua)"
)


@pytest.fixture
def engine():
    """In-memory SQLite engine with graph schema."""
    eng = create_engine("sqlite:///:memory:")
    create_graph_schema(eng)
    return eng


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestGraphSchema:
    def test_create_schema_tables(self, engine):
        """All three graph tables are created."""
        with engine.connect() as conn:
            tables = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
            names = {r[0] for r in tables}
            assert {"graph_nodes", "graph_edges", "graph_properties"} <= names

    def test_create_schema_idempotent(self, engine):
        """Calling create_graph_schema twice does not raise."""
        create_graph_schema(engine)

    def test_indexes_exist(self, engine):
        """All declared indexes are created."""
        with engine.connect() as conn:
            indexes = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='index'")
            ).fetchall()
            names = {r[0] for r in indexes}
            for idx in (
                "idx_graph_nodes_node_id",
                "idx_graph_edges_source",
                "idx_graph_edges_target",
                "idx_graph_properties_owner",
            ):
                assert idx in names

    def test_graph_nodes_columns(self, engine):
        with engine.connect() as conn:
            info = conn.execute(text("PRAGMA table_info(graph_nodes)")).fetchall()
            assert {r[1] for r in info} == {
                "id",
                "node_id",
                "label",
                "created_at",
                "updated_at",
            }

    def test_graph_edges_columns(self, engine):
        with engine.connect() as conn:
            info = conn.execute(text("PRAGMA table_info(graph_edges)")).fetchall()
            assert {r[1] for r in info} == {
                "id",
                "source_id",
                "target_id",
                "label",
                "weight",
                "created_at",
                "updated_at",
            }

    def test_graph_properties_columns(self, engine):
        with engine.connect() as conn:
            info = conn.execute(text("PRAGMA table_info(graph_properties)")).fetchall()
            assert {r[1] for r in info} == {
                "id",
                "owner_type",
                "owner_id",
                "key",
                "value",
                "value_type",
                "created_at",
                "updated_at",
            }

    def test_node_id_unique_constraint(self, engine):
        """Inserting duplicate node_id raises IntegrityError."""
        now = time.time()
        with engine.connect() as conn:
            conn.execute(
                text(_INS_NODE), {"nid": "A", "label": "A", "ca": now, "ua": now}
            )
            with pytest.raises(Exception):
                conn.execute(
                    text(_INS_NODE), {"nid": "A", "label": "Dup", "ca": now, "ua": now}
                )

    def test_edge_unique_constraint(self, engine):
        """Inserting duplicate (source_id, target_id, label) raises."""
        now = time.time()
        params = {"s": "A", "t": "B", "l": "knows", "w": 1.0, "ca": now, "ua": now}
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = OFF"))
            conn.execute(text(_INS_EDGE), params)
            with pytest.raises(Exception):
                conn.execute(text(_INS_EDGE), {**params, "w": 2.0})

    def test_properties_owner_type_check(self, engine):
        """owner_type CHECK constraint rejects invalid values."""
        now = time.time()
        with engine.connect() as conn:
            with pytest.raises(Exception):
                conn.execute(
                    text(_INS_PROP),
                    {
                        "ot": "invalid",
                        "oid": "1",
                        "k": "color",
                        "v": "red",
                        "vt": "string",
                        "ca": now,
                        "ua": now,
                    },
                )


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestGraphNode:
    def test_construction(self):
        now = time.time()
        node = GraphNode(
            id=1, node_id="A", label="Node A", created_at=now, updated_at=now
        )
        assert node.id == 1
        assert node.node_id == "A"
        assert node.label == "Node A"

    def test_construction_no_label(self):
        node = GraphNode(id=2, node_id="B", label=None)
        assert node.label is None

    def test_from_row_mapping(self):
        now = time.time()
        mapping = {
            "id": 1,
            "node_id": "X",
            "label": "Test",
            "created_at": now,
            "updated_at": now,
        }
        node = GraphNode.from_row(mapping)
        assert node.node_id == "X"
        assert node.label == "Test"

    def test_from_row_sqlalchemy(self, engine):
        now = time.time()
        with engine.connect() as conn:
            conn.execute(
                text(_INS_NODE), {"nid": "Z", "label": "DB", "ca": now, "ua": now}
            )
            conn.commit()
            row = conn.execute(
                text("SELECT * FROM graph_nodes WHERE node_id = 'Z'")
            ).fetchone()
            node = GraphNode.from_row(row)
            assert node.node_id == "Z"
            assert node.id >= 1


class TestGraphEdge:
    def test_construction(self):
        now = time.time()
        edge = GraphEdge(
            id=1,
            source_id="A",
            target_id="B",
            label="knows",
            weight=0.5,
            created_at=now,
            updated_at=now,
        )
        assert edge.source_id == "A"
        assert edge.target_id == "B"
        assert edge.label == "knows"
        assert edge.weight == 0.5

    def test_construction_defaults(self):
        edge = GraphEdge(id=2, source_id="C", target_id="D")
        assert edge.label is None
        assert edge.weight is None
        assert edge.created_at == 0.0

    def test_from_row_mapping(self):
        now = time.time()
        mapping = {
            "id": 5,
            "source_id": "X",
            "target_id": "Y",
            "label": "rel",
            "weight": 3.14,
            "created_at": now,
            "updated_at": now,
        }
        edge = GraphEdge.from_row(mapping)
        assert edge.id == 5
        assert edge.source_id == "X"
        assert edge.weight == 3.14

    def test_from_row_sqlalchemy(self, engine):
        now = time.time()
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = OFF"))
            conn.execute(
                text(_INS_EDGE),
                {"s": "P", "t": "Q", "l": "links", "w": 2.0, "ca": now, "ua": now},
            )
            conn.commit()
            row = conn.execute(
                text("SELECT * FROM graph_edges WHERE source_id = 'P'")
            ).fetchone()
            edge = GraphEdge.from_row(row)
            assert edge.source_id == "P"
            assert edge.target_id == "Q"
            assert edge.weight == 2.0


class TestGraphProperty:
    def test_construction(self):
        now = time.time()
        prop = GraphProperty(
            id=1,
            owner_type="node",
            owner_id="A",
            key="color",
            value="red",
            value_type=ValueType.STRING,
            created_at=now,
            updated_at=now,
        )
        assert prop.owner_type == "node"
        assert prop.key == "color"
        assert prop.value_type == ValueType.STRING

    def test_construction_edge_owner(self):
        prop = GraphProperty(
            id=2,
            owner_type="edge",
            owner_id="7",
            key="weight",
            value="42",
            value_type=ValueType.INTEGER,
        )
        assert prop.owner_type == "edge"
        assert prop.owner_id == "7"

    def test_from_row_mapping(self):
        now = time.time()
        mapping = {
            "id": 3,
            "owner_type": "node",
            "owner_id": "B",
            "key": "active",
            "value": "true",
            "value_type": "boolean",
            "created_at": now,
            "updated_at": now,
        }
        prop = GraphProperty.from_row(mapping)
        assert prop.value_type == ValueType.BOOLEAN
        assert prop.value == "true"

    def test_from_row_sqlalchemy(self, engine):
        now = time.time()
        with engine.connect() as conn:
            conn.execute(
                text(_INS_PROP),
                {
                    "ot": "node",
                    "oid": "N1",
                    "k": "size",
                    "v": "42",
                    "vt": "integer",
                    "ca": now,
                    "ua": now,
                },
            )
            conn.commit()
            row = conn.execute(
                text("SELECT * FROM graph_properties WHERE owner_id = 'N1'")
            ).fetchone()
            prop = GraphProperty.from_row(row)
            assert prop.key == "size"
            assert prop.value_type == ValueType.INTEGER


# ---------------------------------------------------------------------------
# ValueType reuse verification
# ---------------------------------------------------------------------------


class TestValueTypeReuse:
    def test_value_type_is_same_enum(self):
        """GraphProperty uses the exact same ValueType from tree_storage."""
        from localdata_mcp.graph_storage import ValueType as GraphVT
        from localdata_mcp.tree_storage import ValueType as TreeVT

        assert GraphVT is TreeVT

    def test_all_value_types_valid_for_properties(self):
        """Every ValueType member can be used in a GraphProperty."""
        now = time.time()
        for vt in ValueType:
            prop = GraphProperty(
                id=0,
                owner_type="node",
                owner_id="test",
                key=f"key_{vt.value}",
                value=None,
                value_type=vt,
                created_at=now,
                updated_at=now,
            )
            assert prop.value_type == vt

    def test_serialize_deserialize_roundtrip(self):
        """tree_storage serialize/deserialize work for graph property values."""
        test_cases = [
            (ValueType.STRING, "hello"),
            (ValueType.INTEGER, 42),
            (ValueType.FLOAT, 3.14),
            (ValueType.BOOLEAN, True),
            (ValueType.NULL, None),
            (ValueType.ARRAY, [1, 2, 3]),
        ]
        for vt, value in test_cases:
            serialized, orig = serialize_value(value, vt)
            restored = deserialize_value(serialized, vt, orig)
            assert restored == value, f"Roundtrip failed for {vt}: {value}"
