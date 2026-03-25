"""Tests for graph_tools module."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from localdata_mcp.graph_manager import GraphStorageManager
from localdata_mcp.graph_storage import create_graph_schema
from localdata_mcp.graph_tools import (
    tool_add_edge,
    tool_delete_key_graph,
    tool_delete_node_graph,
    tool_export_graph,
    tool_find_path,
    tool_get_edges,
    tool_get_graph_stats,
    tool_get_neighbors,
    tool_get_node_graph,
    tool_get_value_graph,
    tool_list_keys_graph,
    tool_remove_edge,
    tool_set_node_graph,
    tool_set_value_graph,
    MAX_EXPORT_BYTES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """GraphStorageManager backed by in-memory SQLite."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_graph_schema(engine)
    return GraphStorageManager(engine)


@pytest.fixture
def populated(manager):
    """Manager with a small graph pre-loaded.

    Graph structure:
        A --[knows]--> B --[knows]--> C
        A --[likes]--> C
        D (isolated node)
    """
    manager.create_node("A", label="Alice")
    manager.create_node("B", label="Bob")
    manager.create_node("C", label="Charlie")
    manager.create_node("D", label="Dave")
    manager.add_edge("A", "B", label="knows", weight=1.0)
    manager.add_edge("B", "C", label="knows", weight=2.0)
    manager.add_edge("A", "C", label="likes", weight=0.5)
    # Set a property on node A
    manager.set_property("node", "A", "age", 30)
    manager.set_property("node", "A", "city", "NYC")
    return manager


# ---------------------------------------------------------------------------
# Navigation: get_node
# ---------------------------------------------------------------------------


class TestGetNodeGraph:
    """Tests for tool_get_node_graph."""

    def test_graph_summary(self, populated):
        result = tool_get_node_graph(populated, "test", node_id=None)
        assert result["node_count"] == 4
        assert result["edge_count"] == 3
        assert result["is_directed"] is True
        assert "density" in result
        assert len(result["first_node_ids"]) == 4

    def test_node_detail(self, populated):
        result = tool_get_node_graph(populated, "test", node_id="A")
        assert result["node_id"] == "A"
        assert result["label"] == "Alice"
        assert result["out_degree"] == 2
        assert result["in_degree"] == 0
        assert result["property_count"] == 2
        assert len(result["properties"]) == 2

    def test_node_not_found(self, populated):
        result = tool_get_node_graph(populated, "test", node_id="NOPE")
        assert "error" in result

    def test_empty_graph_summary(self, manager):
        result = tool_get_node_graph(manager, "test", node_id=None)
        assert result["node_count"] == 0
        assert result["edge_count"] == 0

    def test_get_node_property_truncation(self, manager):
        """When a node has >10 properties, only first 10 are returned."""
        manager.create_node("X")
        for i in range(15):
            manager.set_property("node", "X", f"key_{i:02d}", f"val_{i}")
        result = tool_get_node_graph(manager, "test", node_id="X")
        assert result["property_count"] == 15
        assert len(result["properties"]) == 10
        assert result["properties_truncated"] is True
        assert "hint" in result


# ---------------------------------------------------------------------------
# Navigation: get_neighbors
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    """Tests for tool_get_neighbors."""

    def test_out_neighbors(self, populated):
        result = tool_get_neighbors(populated, "test", "A", direction="out")
        assert result["total"] == 2
        ids = {n["neighbor_id"] for n in result["neighbors"]}
        assert ids == {"B", "C"}
        for n in result["neighbors"]:
            assert n["direction"] == "out"

    def test_in_neighbors(self, populated):
        result = tool_get_neighbors(populated, "test", "C", direction="in")
        assert result["total"] == 2
        ids = {n["neighbor_id"] for n in result["neighbors"]}
        assert ids == {"A", "B"}

    def test_both_neighbors(self, populated):
        result = tool_get_neighbors(populated, "test", "B", direction="both")
        # B has in-edge from A and out-edge to C
        assert result["total"] == 2
        ids = {n["neighbor_id"] for n in result["neighbors"]}
        assert ids == {"A", "C"}

    def test_no_neighbors(self, populated):
        result = tool_get_neighbors(populated, "test", "D", direction="both")
        assert result["total"] == 0
        assert result["neighbors"] == []

    def test_node_not_found(self, populated):
        result = tool_get_neighbors(populated, "test", "NOPE")
        assert "error" in result

    def test_invalid_direction(self, populated):
        result = tool_get_neighbors(populated, "test", "A", direction="sideways")
        assert "error" in result

    def test_pagination(self, populated):
        result = tool_get_neighbors(
            populated, "test", "A", direction="out", offset=0, limit=1
        )
        assert len(result["neighbors"]) == 1
        assert result["total"] == 2
        assert result["has_more"] is True

    def test_self_loop_direction(self, manager):
        """Self-loops must be classified with direction 'self'."""
        manager.create_node("X")
        manager.add_edge("X", "X", label="self_ref")
        result = tool_get_neighbors(manager, "test", "X", direction="both")
        assert result["total"] == 1
        assert result["neighbors"][0]["direction"] == "self"
        assert result["neighbors"][0]["neighbor_id"] == "X"


# ---------------------------------------------------------------------------
# Navigation: get_edges
# ---------------------------------------------------------------------------


class TestGetEdges:
    """Tests for tool_get_edges."""

    def test_all_edges(self, populated):
        result = tool_get_edges(populated, "test")
        assert result["total"] == 3
        assert len(result["edges"]) == 3

    def test_edges_filtered_by_node(self, populated):
        result = tool_get_edges(populated, "test", node_id="B")
        # B is source of B->C and target of A->B
        assert result["total"] == 2
        assert len(result["edges"]) == 2

    def test_edges_pagination(self, populated):
        result = tool_get_edges(populated, "test", offset=0, limit=1)
        assert len(result["edges"]) == 1
        assert result["total"] == 3
        assert result["has_more"] is True

    def test_edge_fields(self, populated):
        result = tool_get_edges(populated, "test")
        edge = result["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "label" in edge
        assert "weight" in edge


# ---------------------------------------------------------------------------
# Mutation: set_node, delete_node
# ---------------------------------------------------------------------------


class TestSetNodeGraph:
    """Tests for tool_set_node_graph."""

    def test_create_node(self, manager):
        result = tool_set_node_graph(manager, "test", "X", label="Xavier")
        assert result["created"] is True
        assert result["node_id"] == "X"
        assert result["label"] == "Xavier"

    def test_update_node(self, populated):
        result = tool_set_node_graph(populated, "test", "A", label="Alicia")
        assert result["created"] is False
        assert result["label"] == "Alicia"


class TestDeleteNodeGraph:
    """Tests for tool_delete_node_graph."""

    def test_delete_node_cascades(self, populated):
        result = tool_delete_node_graph(populated, "test", "A")
        assert result["nodes_deleted"] == 1
        assert result["edges_deleted"] == 2  # A->B, A->C
        assert result["properties_deleted"] == 2  # age, city

    def test_delete_nonexistent(self, populated):
        result = tool_delete_node_graph(populated, "test", "NOPE")
        assert "error" in result


# ---------------------------------------------------------------------------
# Mutation: add_edge, remove_edge
# ---------------------------------------------------------------------------


class TestAddEdge:
    """Tests for tool_add_edge."""

    def test_add_edge_existing_nodes(self, populated):
        result = tool_add_edge(populated, "test", "C", "D", label="calls")
        assert result["source"] == "C"
        assert result["target"] == "D"
        assert result["label"] == "calls"
        assert result["nodes_created"] == []

    def test_add_edge_creates_nodes(self, manager):
        result = tool_add_edge(manager, "test", "X", "Y", weight=3.14)
        assert "X" in result["nodes_created"]
        assert "Y" in result["nodes_created"]
        assert result["weight"] == 3.14


class TestRemoveEdge:
    """Tests for tool_remove_edge."""

    def test_remove_existing(self, populated):
        result = tool_remove_edge(populated, "test", "A", "B", label="knows")
        assert result["removed"] is True

    def test_remove_nonexistent(self, populated):
        result = tool_remove_edge(populated, "test", "C", "A")
        assert result["removed"] is False


# ---------------------------------------------------------------------------
# Algorithm: find_path
# ---------------------------------------------------------------------------


class TestFindPath:
    """Tests for tool_find_path."""

    def test_shortest_path(self, populated):
        result = tool_find_path(populated, "test", "A", "C")
        assert result["algorithm"] == "shortest"
        # Direct edge A->C exists
        assert result["path"] == ["A", "C"]
        assert result["path_length"] == 1

    def test_shortest_path_indirect(self, populated):
        # Remove direct A->C edge, path goes through B
        populated.remove_edge("A", "C", label="likes")
        result = tool_find_path(populated, "test", "A", "C")
        assert result["path"] == ["A", "B", "C"]
        assert result["path_length"] == 2

    def test_no_path(self, populated):
        result = tool_find_path(populated, "test", "D", "A")
        assert result["path"] is None
        assert "No path" in result["message"]

    def test_all_paths(self, populated):
        result = tool_find_path(populated, "test", "A", "C", algorithm="all")
        assert result["algorithm"] == "all"
        assert result["paths_count"] >= 2  # A->C direct and A->B->C

    def test_invalid_algorithm(self, populated):
        result = tool_find_path(populated, "test", "A", "C", algorithm="dijkstra")
        assert "error" in result

    def test_source_not_found(self, populated):
        result = tool_find_path(populated, "test", "NOPE", "A")
        assert "error" in result

    def test_target_not_found(self, populated):
        result = tool_find_path(populated, "test", "A", "NOPE")
        assert "error" in result

    def test_find_path_with_parallel_edges(self, manager):
        """find_path must work when parallel edges exist (MultiDiGraph)."""
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B", label="route1")
        manager.add_edge("A", "B", label="route2")
        result = tool_find_path(manager, "test", "A", "B")
        assert result["path"] == ["A", "B"]
        assert result["path_length"] == 1


# ---------------------------------------------------------------------------
# Algorithm: get_graph_stats
# ---------------------------------------------------------------------------


class TestGetGraphStats:
    """Tests for tool_get_graph_stats."""

    def test_basic_stats(self, populated):
        result = tool_get_graph_stats(populated, "test")
        assert result["node_count"] == 4
        assert result["edge_count"] == 3
        assert "density" in result
        assert "is_dag" in result
        assert result["is_dag"] is True  # no cycles in our test graph
        assert "connected_components" in result
        assert "average_degree" in result

    def test_max_degree_nodes(self, populated):
        result = tool_get_graph_stats(populated, "test")
        assert result["max_out_degree"]["node_id"] == "A"
        assert result["max_out_degree"]["out_degree"] == 2
        assert result["max_in_degree"]["node_id"] == "C"
        assert result["max_in_degree"]["in_degree"] == 2

    def test_empty_graph(self, manager):
        result = tool_get_graph_stats(manager, "test")
        assert result["node_count"] == 0
        assert result["edge_count"] == 0
        assert result["average_degree"] == 0.0


# ---------------------------------------------------------------------------
# Property tools
# ---------------------------------------------------------------------------


class TestPropertyTools:
    """Tests for graph property tools."""

    def test_get_value(self, populated):
        result = tool_get_value_graph(populated, "test", "A", "age")
        assert result["key"] == "age"
        assert result["value"] == 30

    def test_get_value_not_found(self, populated):
        result = tool_get_value_graph(populated, "test", "A", "nope")
        assert "error" in result

    def test_set_value(self, populated):
        result = tool_set_value_graph(populated, "test", "B", "role", "engineer")
        assert result["key"] == "role"
        assert result["value"] == "engineer"

    def test_set_value_creates_node(self, manager):
        result = tool_set_value_graph(manager, "test", "NEW", "key1", "val1")
        assert result["node_id"] == "NEW"
        assert manager.node_exists("NEW")

    def test_set_value_with_type(self, populated):
        result = tool_set_value_graph(
            populated, "test", "A", "score", "99", value_type="integer"
        )
        assert result["value"] == 99
        assert result["value_type"] == "integer"

    def test_delete_key(self, populated):
        result = tool_delete_key_graph(populated, "test", "A", "age")
        assert result["deleted"] is True

    def test_delete_key_not_found(self, populated):
        result = tool_delete_key_graph(populated, "test", "A", "nope")
        assert "error" in result

    def test_list_keys(self, populated):
        result = tool_list_keys_graph(populated, "test", "A")
        assert result["total"] == 2
        keys = {k["key"] for k in result["keys"]}
        assert keys == {"age", "city"}

    def test_list_keys_pagination(self, populated):
        result = tool_list_keys_graph(populated, "test", "A", offset=0, limit=1)
        assert len(result["keys"]) == 1
        assert result["has_more"] is True

    def test_list_keys_node_not_found(self, populated):
        result = tool_list_keys_graph(populated, "test", "NOPE")
        assert "error" in result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExportGraph:
    """Tests for tool_export_graph."""

    def test_export_dot(self, populated):
        result = tool_export_graph(populated, "test", "dot")
        assert result["format"] == "dot"
        assert "content" in result
        # DOT output should contain node references
        assert "A" in result["content"]

    def test_export_gml(self, populated):
        result = tool_export_graph(populated, "test", "gml")
        assert result["format"] == "gml"
        assert "content" in result
        assert "graph" in result["content"].lower()

    def test_export_graphml(self, populated):
        result = tool_export_graph(populated, "test", "graphml")
        assert result["format"] == "graphml"
        assert "content" in result
        assert "graphml" in result["content"].lower()

    def test_export_unsupported_format(self, populated):
        result = tool_export_graph(populated, "test", "csv")
        assert "error" in result

    def test_export_with_node_filter(self, populated):
        result = tool_export_graph(populated, "test", "dot", node_id="B")
        assert result["format"] == "dot"
        # Ego graph of B should include A, B, C
        assert "B" in result["content"]

    def test_export_node_not_found(self, populated):
        result = tool_export_graph(populated, "test", "dot", node_id="NOPE")
        assert "error" in result

    def test_export_graph_truncation(self, manager):
        """Large graph export must be truncated when exceeding MAX_EXPORT_BYTES."""
        # Create enough nodes with long labels to exceed 100KB in GML output
        for i in range(2000):
            manager.create_node(f"node_with_long_id_{'x' * 50}_{i:04d}")
        for i in range(1999):
            manager.add_edge(
                f"node_with_long_id_{'x' * 50}_{i:04d}",
                f"node_with_long_id_{'x' * 50}_{i + 1:04d}",
            )

        result = tool_export_graph(manager, "test", "gml")
        assert result["truncated"] is True
        assert "notice" in result
        # Content should be present but shorter than the full output
        assert len(result["content"].encode("utf-8")) <= MAX_EXPORT_BYTES
