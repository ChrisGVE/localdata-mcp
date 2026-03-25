"""Tests for GraphStorageManager CRUD operations."""

import pytest
from sqlalchemy import create_engine, event, text

from localdata_mcp.graph_manager import GraphStorageManager
from localdata_mcp.graph_storage import (
    GraphEdge,
    GraphNode,
    GraphProperty,
)
from localdata_mcp.tree_storage import ValueType


@pytest.fixture()
def engine():
    """Create an in-memory SQLite engine with StaticPool for thread safety."""
    from sqlalchemy.pool import StaticPool

    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return eng


@pytest.fixture()
def mgr(engine):
    """Create a GraphStorageManager with a fresh in-memory database."""
    return GraphStorageManager(engine)


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------


class TestNodeOperations:
    def test_create_node(self, mgr):
        node = mgr.create_node("A", label="alpha")
        assert isinstance(node, GraphNode)
        assert node.node_id == "A"
        assert node.label == "alpha"
        assert node.created_at > 0

    def test_create_node_idempotent(self, mgr):
        n1 = mgr.create_node("A", label="first")
        n2 = mgr.create_node("A", label="second")
        assert n1.id == n2.id
        assert n2.label == "second"

    def test_create_node_no_label(self, mgr):
        node = mgr.create_node("X")
        assert node.label is None

    def test_get_node(self, mgr):
        mgr.create_node("A")
        node = mgr.get_node("A")
        assert node is not None
        assert node.node_id == "A"

    def test_get_node_missing(self, mgr):
        assert mgr.get_node("missing") is None

    def test_node_exists(self, mgr):
        assert mgr.node_exists("A") is False
        mgr.create_node("A")
        assert mgr.node_exists("A") is True

    def test_delete_node_simple(self, mgr):
        mgr.create_node("A")
        nodes, edges, props = mgr.delete_node("A")
        assert nodes == 1
        assert edges == 0
        assert props == 0
        assert mgr.get_node("A") is None

    def test_delete_node_missing(self, mgr):
        nodes, edges, props = mgr.delete_node("nonexistent")
        assert nodes == 0

    def test_delete_node_cascades_edges(self, mgr):
        mgr.add_edge("A", "B")
        mgr.add_edge("C", "A")
        nodes, edges, props = mgr.delete_node("A")
        assert nodes == 1
        assert edges == 2
        assert mgr.get_edge_count() == 0

    def test_delete_node_cascades_properties(self, mgr):
        mgr.create_node("A")
        mgr.set_property("node", "A", "color", "red")
        mgr.set_property("node", "A", "size", 10)
        nodes, edges, props = mgr.delete_node("A")
        assert props == 2

    def test_list_nodes(self, mgr):
        for name in ["C", "A", "B"]:
            mgr.create_node(name)
        nodes = mgr.list_nodes()
        assert [n.node_id for n in nodes] == ["A", "B", "C"]

    def test_list_nodes_pagination(self, mgr):
        for i in range(5):
            mgr.create_node(f"n{i}")
        page = mgr.list_nodes(offset=2, limit=2)
        assert len(page) == 2

    def test_get_node_count(self, mgr):
        assert mgr.get_node_count() == 0
        mgr.create_node("A")
        mgr.create_node("B")
        assert mgr.get_node_count() == 2

    def test_create_node_empty_string_raises(self, mgr):
        with pytest.raises(ValueError, match="node_id must be a non-empty string"):
            mgr.create_node("")

    def test_create_node_whitespace_only_raises(self, mgr):
        with pytest.raises(ValueError, match="node_id must be a non-empty string"):
            mgr.create_node("   ")


# ---------------------------------------------------------------------------
# Edge CRUD
# ---------------------------------------------------------------------------


class TestEdgeOperations:
    def test_add_edge(self, mgr):
        edge = mgr.add_edge("A", "B")
        assert isinstance(edge, GraphEdge)
        assert edge.source_id == "A"
        assert edge.target_id == "B"
        assert edge.label is None

    def test_add_edge_auto_creates_nodes(self, mgr):
        mgr.add_edge("X", "Y")
        assert mgr.node_exists("X")
        assert mgr.node_exists("Y")

    def test_add_edge_with_label_and_weight(self, mgr):
        edge = mgr.add_edge("A", "B", label="knows", weight=0.9)
        assert edge.label == "knows"
        assert edge.weight == 0.9

    def test_multigraph_different_labels(self, mgr):
        e1 = mgr.add_edge("A", "B", label="friend")
        e2 = mgr.add_edge("A", "B", label="colleague")
        assert e1.id != e2.id
        assert mgr.get_edge_count() == 2

    def test_multigraph_null_and_labeled(self, mgr):
        e1 = mgr.add_edge("A", "B")
        e2 = mgr.add_edge("A", "B", label="typed")
        assert e1.id != e2.id
        assert mgr.get_edge_count() == 2

    def test_add_edge_upsert(self, mgr):
        e1 = mgr.add_edge("A", "B", weight=1.0)
        e2 = mgr.add_edge("A", "B", weight=2.0)
        assert e1.id == e2.id
        assert e2.weight == 2.0

    def test_remove_edge(self, mgr):
        mgr.add_edge("A", "B")
        assert mgr.remove_edge("A", "B") is True
        assert mgr.get_edge("A", "B") is None

    def test_remove_edge_missing(self, mgr):
        assert mgr.remove_edge("X", "Y") is False

    def test_remove_edge_with_label(self, mgr):
        mgr.add_edge("A", "B", label="x")
        mgr.add_edge("A", "B", label="y")
        assert mgr.remove_edge("A", "B", label="x") is True
        assert mgr.get_edge_count() == 1
        assert mgr.get_edge("A", "B", label="y") is not None

    def test_get_edge(self, mgr):
        mgr.add_edge("A", "B", label="rel")
        edge = mgr.get_edge("A", "B", label="rel")
        assert edge is not None
        assert edge.label == "rel"

    def test_get_edge_null_label(self, mgr):
        mgr.add_edge("A", "B")
        assert mgr.get_edge("A", "B") is not None
        assert mgr.get_edge("A", "B", label="nope") is None

    def test_list_edges_all(self, mgr):
        mgr.add_edge("A", "B")
        mgr.add_edge("C", "D")
        edges = mgr.list_edges()
        assert len(edges) == 2

    def test_list_edges_filtered(self, mgr):
        mgr.add_edge("A", "B")
        mgr.add_edge("A", "C")
        mgr.add_edge("D", "E")
        edges = mgr.list_edges(node_id="A")
        assert len(edges) == 2

    def test_list_edges_pagination(self, mgr):
        for i in range(5):
            mgr.add_edge(f"s{i}", f"t{i}")
        page = mgr.list_edges(offset=2, limit=2)
        assert len(page) == 2

    def test_get_edge_count(self, mgr):
        assert mgr.get_edge_count() == 0
        mgr.add_edge("A", "B")
        assert mgr.get_edge_count() == 1

    def test_add_edge_empty_source_raises(self, mgr):
        with pytest.raises(ValueError, match="source must be a non-empty string"):
            mgr.add_edge("", "B")

    def test_add_edge_empty_target_raises(self, mgr):
        with pytest.raises(ValueError, match="target must be a non-empty string"):
            mgr.add_edge("A", "")

    def test_add_edge_whitespace_source_raises(self, mgr):
        with pytest.raises(ValueError, match="source must be a non-empty string"):
            mgr.add_edge("  ", "B")


# ---------------------------------------------------------------------------
# Neighbor operations
# ---------------------------------------------------------------------------


class TestNeighborOperations:
    def test_successors(self, mgr):
        mgr.add_edge("A", "B")
        mgr.add_edge("A", "C")
        succ = mgr.get_successors("A")
        assert sorted(succ) == ["B", "C"]

    def test_predecessors(self, mgr):
        mgr.add_edge("B", "A")
        mgr.add_edge("C", "A")
        pred = mgr.get_predecessors("A")
        assert sorted(pred) == ["B", "C"]

    def test_neighbors_both(self, mgr):
        mgr.add_edge("A", "B")
        mgr.add_edge("C", "A")
        neighbors = mgr.get_neighbors("A", direction="both")
        assert sorted(neighbors) == ["B", "C"]

    def test_neighbors_out(self, mgr):
        mgr.add_edge("A", "B")
        mgr.add_edge("C", "A")
        assert mgr.get_neighbors("A", direction="out") == ["B"]

    def test_neighbors_in(self, mgr):
        mgr.add_edge("A", "B")
        mgr.add_edge("C", "A")
        assert mgr.get_neighbors("A", direction="in") == ["C"]

    def test_neighbors_empty(self, mgr):
        mgr.create_node("isolated")
        assert mgr.get_successors("isolated") == []
        assert mgr.get_predecessors("isolated") == []
        assert mgr.get_neighbors("isolated") == []

    def test_successors_pagination(self, mgr):
        for i in range(5):
            mgr.add_edge("hub", f"t{i}")
        page = mgr.get_successors("hub", offset=2, limit=2)
        assert len(page) == 2

    def test_successors_distinct(self, mgr):
        mgr.add_edge("A", "B", label="x")
        mgr.add_edge("A", "B", label="y")
        succ = mgr.get_successors("A")
        assert succ == ["B"]


# ---------------------------------------------------------------------------
# Property operations
# ---------------------------------------------------------------------------


class TestPropertyOperations:
    def test_set_and_get_node_property(self, mgr):
        mgr.create_node("A")
        prop = mgr.set_property("node", "A", "color", "red")
        assert isinstance(prop, GraphProperty)
        assert prop.owner_type == "node"
        assert prop.owner_id == "A"
        assert prop.key == "color"
        assert prop.value == "red"
        assert prop.value_type == ValueType.STRING

    def test_set_and_get_edge_property(self, mgr):
        mgr.add_edge("A", "B")
        prop = mgr.set_property("edge", "A->B", "weight_class", "heavy")
        assert prop.owner_type == "edge"
        fetched = mgr.get_property("edge", "A->B", "weight_class")
        assert fetched is not None
        assert fetched.value == "heavy"

    def test_property_upsert(self, mgr):
        mgr.create_node("A")
        mgr.set_property("node", "A", "k", "v1")
        p2 = mgr.set_property("node", "A", "k", "v2")
        assert p2.value == "v2"
        assert mgr.get_property_count("node", "A") == 1

    def test_get_property_missing(self, mgr):
        assert mgr.get_property("node", "nope", "k") is None

    def test_list_properties(self, mgr):
        mgr.create_node("A")
        mgr.set_property("node", "A", "b_key", "val")
        mgr.set_property("node", "A", "a_key", "val")
        props = mgr.list_properties("node", "A")
        assert [p.key for p in props] == ["a_key", "b_key"]

    def test_list_properties_pagination(self, mgr):
        mgr.create_node("A")
        for i in range(5):
            mgr.set_property("node", "A", f"k{i}", f"v{i}")
        page = mgr.list_properties("node", "A", offset=2, limit=2)
        assert len(page) == 2

    def test_delete_property(self, mgr):
        mgr.create_node("A")
        mgr.set_property("node", "A", "k", "v")
        assert mgr.delete_property("node", "A", "k") is True
        assert mgr.get_property("node", "A", "k") is None

    def test_delete_property_missing(self, mgr):
        assert mgr.delete_property("node", "nope", "k") is False

    def test_get_property_count(self, mgr):
        mgr.create_node("A")
        assert mgr.get_property_count("node", "A") == 0
        mgr.set_property("node", "A", "k1", "v")
        mgr.set_property("node", "A", "k2", "v")
        assert mgr.get_property_count("node", "A") == 2

    def test_value_type_integer(self, mgr):
        mgr.create_node("A")
        prop = mgr.set_property("node", "A", "count", 42)
        assert prop.value_type == ValueType.INTEGER
        assert prop.value == "42"

    def test_value_type_float(self, mgr):
        mgr.create_node("A")
        prop = mgr.set_property("node", "A", "score", 3.14)
        assert prop.value_type == ValueType.FLOAT

    def test_value_type_boolean(self, mgr):
        mgr.create_node("A")
        prop = mgr.set_property("node", "A", "active", True)
        assert prop.value_type == ValueType.BOOLEAN
        assert prop.value == "true"

    def test_value_type_null(self, mgr):
        mgr.create_node("A")
        prop = mgr.set_property("node", "A", "empty", None)
        assert prop.value_type == ValueType.NULL
        assert prop.value is None

    def test_value_type_array(self, mgr):
        mgr.create_node("A")
        prop = mgr.set_property("node", "A", "tags", [1, 2, 3])
        assert prop.value_type == ValueType.ARRAY

    def test_explicit_value_type(self, mgr):
        mgr.create_node("A")
        prop = mgr.set_property("node", "A", "num", "42", value_type=ValueType.STRING)
        assert prop.value_type == ValueType.STRING
        assert prop.value == "42"


# ---------------------------------------------------------------------------
# Edge property cleanup on delete
# ---------------------------------------------------------------------------


class TestEdgePropertyCleanup:
    def test_delete_node_cleans_edge_properties(self, mgr):
        """Deleting a node must also remove properties on its edges."""
        edge = mgr.add_edge("A", "B")
        edge_id = str(edge.id)
        mgr.set_property("edge", edge_id, "color", "red")
        assert mgr.get_property("edge", edge_id, "color") is not None

        nodes_del, edges_del, props_del = mgr.delete_node("A")
        assert nodes_del == 1
        assert edges_del == 1
        # The edge property should be counted and deleted
        assert props_del >= 1
        assert mgr.get_property("edge", edge_id, "color") is None

    def test_remove_edge_cleans_edge_properties(self, mgr):
        """Removing an edge must also delete its properties."""
        edge = mgr.add_edge("A", "B")
        edge_id = str(edge.id)
        mgr.set_property("edge", edge_id, "style", "dashed")
        assert mgr.get_property("edge", edge_id, "style") is not None

        removed = mgr.remove_edge("A", "B")
        assert removed is True
        assert mgr.get_property("edge", edge_id, "style") is None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestGraphStats:
    def test_empty_graph(self, mgr):
        stats = mgr.get_graph_stats()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["property_count"] == 0
        assert stats["is_directed"] is True
        assert stats["density"] == 0.0

    def test_populated_graph(self, mgr):
        mgr.add_edge("A", "B")
        mgr.add_edge("B", "C")
        mgr.create_node("D")
        mgr.set_property("node", "A", "k", "v")
        stats = mgr.get_graph_stats()
        assert stats["node_count"] == 4
        assert stats["edge_count"] == 2
        assert stats["property_count"] == 1
        assert stats["is_directed"] is True

    def test_density_calculation(self, mgr):
        # 3 nodes, 3 edges => density = 3 / (3 * 2) = 0.5
        mgr.add_edge("A", "B")
        mgr.add_edge("B", "C")
        mgr.add_edge("C", "A")
        stats = mgr.get_graph_stats()
        assert stats["density"] == pytest.approx(0.5)

    def test_density_single_node(self, mgr):
        mgr.create_node("alone")
        stats = mgr.get_graph_stats()
        assert stats["density"] == 0.0

    def test_complete_graph_density(self, mgr):
        # 3 nodes, all 6 directed edges => density = 1.0
        for s in ["A", "B", "C"]:
            for t in ["A", "B", "C"]:
                if s != t:
                    mgr.add_edge(s, t)
        stats = mgr.get_graph_stats()
        assert stats["density"] == pytest.approx(1.0)

    def test_density_clamped(self, mgr):
        """Density must be clamped to 1.0 even with parallel edges."""
        # 2 nodes with multiple parallel edges -> raw density > 1.0
        mgr.add_edge("A", "B", label="x")
        mgr.add_edge("A", "B", label="y")
        mgr.add_edge("B", "A", label="x")
        mgr.add_edge("B", "A", label="y")
        stats = mgr.get_graph_stats()
        # 4 edges / (2 * 1) = 2.0 raw, should be clamped to 1.0
        assert stats["density"] <= 1.0
