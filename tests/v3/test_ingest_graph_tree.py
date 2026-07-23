"""tests/v3/test_ingest_graph_tree.py — E8.3: the graph/tree tool family.

FR-103's node/edge CRUD and tree-read L3 leg over the real stack:
tree structure (ancestor creation, subtree move, cascade delete with
FK-backed property cascade), graph structure (edge upsert with the
harvested integrity warnings, manual cascade delete), path finding
over guard-read topology, and the kind dispatch refusals — a tree
operation against a graph endpoint (and vice versa) is a structured
kind mismatch, mutations below read-write posture are refused by the
guard (NFR-113).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.ingest.connectors.graph_tree.graph_tools import (
    add_edge,
    find_path,
    get_edges,
    get_graph_stats,
    get_neighbors,
    remove_edge,
)
from localdata_mcp.ingest.connectors.graph_tree.tools import (
    delete_node,
    get_children,
    get_node,
    move_node,
    set_node,
)
from localdata_mcp.ingest.connectors.kv.tools import list_keys, set_value
from localdata_mcp.nexus.chokepoint.guard import (
    Chokepoint,
    GuardedExecutionError,
    GuardRefusedError,
)
from localdata_mcp.nexus.config.endpoints import EndpointDeclaration
from localdata_mcp.nexus.config.models import ConfigModel


def _config(tmp_path: Path) -> ConfigModel:
    return ConfigModel(
        endpoints={
            "config_tree": EndpointDeclaration(
                name="config_tree",
                dsn=f"tree+sqlite:///{tmp_path / 'tree.db'}",
                posture="read_write",
            ),
            "social": EndpointDeclaration(
                name="social",
                dsn=f"graph+sqlite:///{tmp_path / 'graph.db'}",
                posture="read_write",
            ),
            "social_ro": EndpointDeclaration(
                name="social_ro",
                dsn=f"graph+sqlite:///{tmp_path / 'graph.db'}",
                posture="read_only",
            ),
        }
    )


@pytest.fixture()
def booted(tmp_path: Path) -> Iterator[Chokepoint]:
    guard = Chokepoint.boot(_config(tmp_path), environ={})
    runtime.configure_ingest(guard)
    yield guard
    runtime._CHOKEPOINT = None
    guard.shutdown()


class TestTreeStructure:
    def test_set_node_creates_ancestors_and_reports_them(
        self, booted: Chokepoint
    ) -> None:
        outcome = set_node("config_tree", "a.b.c")
        assert outcome["created"] is True
        assert outcome["ancestors_created"] == ["a", "a.b"]
        again = set_node("config_tree", "a.b.c")
        assert again["created"] is False
        assert again["ancestors_created"] == []

    def test_get_node_detail_counts_not_property_dump(self, booted: Chokepoint) -> None:
        set_node("config_tree", "svc.db")
        set_value("config_tree", "svc.db", "host", "localhost")
        detail = get_node("config_tree", "svc.db")
        assert detail["name"] == "db"
        assert detail["parent_path"] == "svc"
        assert detail["property_count"] == 1
        assert "properties" not in detail
        assert "list_keys" in detail["hint"]

    def test_get_node_without_path_is_the_store_summary(
        self, booted: Chokepoint
    ) -> None:
        set_node("config_tree", "a.b")
        summary = get_node("config_tree")
        assert summary["total_nodes"] == 2
        assert summary["root_count"] == 1
        assert summary["max_depth"] == 1

    def test_get_children_lists_roots_and_paginates(self, booted: Chokepoint) -> None:
        for root in ("zeta", "alpha", "mid.leaf"):
            set_node("config_tree", root)
        roots = get_children("config_tree")
        assert [row[0] for row in roots.rows] == ["alpha", "mid", "zeta"]
        page = get_children("config_tree", offset=1, limit=1)
        assert [row[0] for row in page.rows] == ["mid"]
        leaves = get_children("config_tree", path="mid")
        assert [row[1] for row in leaves.rows] == ["mid.leaf"]

    def test_move_node_rewrites_subtree_paths(self, booted: Chokepoint) -> None:
        set_node("config_tree", "src.x.y")
        set_node("config_tree", "dst")
        outcome = move_node("config_tree", "src.x", new_parent="dst")
        assert outcome["new_path"] == "dst.x"
        assert outcome["nodes_moved"] == 2
        moved = get_node("config_tree", "dst.x.y")
        assert moved["depth"] == 2

    def test_move_into_own_subtree_refused(self, booted: Chokepoint) -> None:
        set_node("config_tree", "a.b")
        with pytest.raises(GuardedExecutionError, match="own subtree"):
            move_node("config_tree", "a", new_parent="a.b")

    def test_delete_node_cascades_subtree_and_properties(
        self, booted: Chokepoint
    ) -> None:
        set_node("config_tree", "gone.child")
        set_value("config_tree", "gone.child", "k", "v")
        outcome = delete_node("config_tree", "gone")
        assert outcome["nodes_deleted"] == 2
        assert outcome["properties_deleted"] == 1
        with pytest.raises(GuardedExecutionError, match="not found"):
            get_node("config_tree", "gone.child")
        # FK cascade proof through the tool surface: no orphan property
        # remains reachable anywhere in the store.
        summary = get_node("config_tree")
        assert summary["total_properties"] == 0

    def test_label_on_tree_node_refused_toward_set_value(
        self, booted: Chokepoint
    ) -> None:
        with pytest.raises(GuardedExecutionError, match="no label"):
            set_node("config_tree", "n", label="x")


class TestGraphStructure:
    def test_set_node_upserts_with_label_and_casing_warning(
        self, booted: Chokepoint
    ) -> None:
        first = set_node("social", "Alice", label="person")
        assert first == {"created": True, "node_id": "Alice", "label": "person"}
        variant = set_node("social", "alice")
        assert variant["created"] is True
        assert variant["warnings"][0]["code"] == "duplicate_casing"

    def test_add_edge_auto_creates_and_warns(self, booted: Chokepoint) -> None:
        outcome = add_edge("social", "a", "b", label="knows", weight=1.0)
        assert outcome["nodes_created"] == ["a", "b"]
        assert outcome["label"] == "knows"
        loop = add_edge("social", "a", "a")
        codes = {warning["code"] for warning in loop["warnings"]}
        assert {"self_loop", "missing_edge_labels"} <= codes

    def test_contradictory_reverse_edge_warns(self, booted: Chokepoint) -> None:
        add_edge("social", "x", "y", label="follows")
        outcome = add_edge("social", "y", "x", label="follows")
        codes = {warning["code"] for warning in outcome["warnings"]}
        assert "contradictory_edges" in codes

    def test_remove_edge_reports_orphans(self, booted: Chokepoint) -> None:
        add_edge("social", "p", "q", label="l")
        outcome = remove_edge("social", "p", "q", label="l")
        assert outcome["removed"] is True
        codes = {warning["code"] for warning in outcome.get("warnings", ())}
        assert "orphan_nodes" in codes
        again = remove_edge("social", "p", "q", label="l")
        assert again["removed"] is False

    def test_get_neighbors_classifies_directions(self, booted: Chokepoint) -> None:
        add_edge("social", "hub", "out1")
        add_edge("social", "in1", "hub")
        both = get_neighbors("social", "hub")
        assert set(both.columns) == {
            "neighbor_id",
            "edge_label",
            "edge_weight",
            "direction",
        }
        by_direction = {row[3]: row[0] for row in both.rows}
        assert by_direction == {"in": "in1", "out": "out1"}
        outgoing = get_neighbors("social", "hub", direction="out")
        assert [row[0] for row in outgoing.rows] == ["out1"]

    def test_get_edges_filters_by_node(self, booted: Chokepoint) -> None:
        add_edge("social", "a", "b")
        add_edge("social", "c", "d")
        assert len(get_edges("social").rows) == 2
        touching = get_edges("social", node_id="a")
        assert [(row[0], row[1]) for row in touching.rows] == [("a", "b")]

    def test_delete_node_cascades_edges_and_properties(
        self, booted: Chokepoint
    ) -> None:
        add_edge("social", "hub", "spoke")
        set_value("social", "hub", "k", "v")
        outcome = delete_node("social", "hub")
        assert outcome["nodes_deleted"] == 1
        assert outcome["edges_deleted"] == 1
        assert outcome["properties_deleted"] == 1
        assert get_graph_stats("social")["edge_count"] == 0

    def test_graph_summary_via_get_node(self, booted: Chokepoint) -> None:
        add_edge("social", "a", "b")
        summary = get_node("social")
        assert summary["node_count"] == 2
        assert summary["edge_count"] == 1
        assert summary["is_directed"] is True


class TestFindPath:
    def test_shortest_path_found(self, booted: Chokepoint) -> None:
        add_edge("social", "a", "b")
        add_edge("social", "b", "c")
        outcome = find_path("social", "a", "c")
        assert outcome["path"] == ["a", "b", "c"]
        assert outcome["path_length"] == 2

    def test_no_path_is_a_message_not_an_error(self, booted: Chokepoint) -> None:
        set_node("social", "island")
        set_node("social", "mainland")
        outcome = find_path("social", "island", "mainland")
        assert outcome["path"] is None
        assert "No path" in outcome["message"]

    def test_all_paths_enumerates_alternatives(self, booted: Chokepoint) -> None:
        add_edge("social", "s", "m1")
        add_edge("social", "s", "m2")
        add_edge("social", "m1", "t")
        add_edge("social", "m2", "t")
        outcome = find_path("social", "s", "t", algorithm="all")
        assert outcome["paths_count"] == 2
        assert outcome["truncated"] is False

    def test_missing_node_refused_with_guidance(self, booted: Chokepoint) -> None:
        set_node("social", "real")
        with pytest.raises(GuardedExecutionError) as refusal:
            find_path("social", "real", "ghost")
        assert "get_graph_stats" in refusal.value.structured.suggestion


class TestKindAndPostureRefusals:
    def test_tree_operation_on_graph_endpoint_refused(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            get_children("social")
        assert "kv / tree" in refusal.value.structured.message

    def test_graph_operation_on_tree_endpoint_refused(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardedExecutionError) as refusal:
            get_neighbors("config_tree", "n")
        assert "graph" in refusal.value.structured.message

    def test_mutation_refused_on_read_only_posture(self, booted: Chokepoint) -> None:
        with pytest.raises(GuardRefusedError, match="read_only"):
            add_edge("social_ro", "a", "b")

    def test_reads_permitted_on_read_only_posture(self, booted: Chokepoint) -> None:
        add_edge("social", "a", "b")
        assert get_graph_stats("social_ro")["edge_count"] == 1

    def test_kv_surface_reaches_graph_properties(self, booted: Chokepoint) -> None:
        """The harvested dual, end to end: kv tools address graph nodes."""
        set_node("social", "alice")
        set_value("social", "alice", "age", "30")
        listed = list_keys("social", "alice")
        assert listed.rows == (("age", 30, "integer"),)
