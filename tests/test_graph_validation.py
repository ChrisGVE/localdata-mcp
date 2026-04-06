"""Tests for graph validation checks."""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from localdata_mcp.graph_algorithms import _storage_to_networkx
from localdata_mcp.graph_manager import GraphStorageManager
from localdata_mcp.graph_storage import create_graph_schema
from localdata_mcp.graph_validation import (
    check_conflicting_parallel_labels,
    check_contradictory_edges,
    check_duplicate_casing,
    check_duplicate_edges,
    check_missing_common_properties,
    check_missing_edge_labels,
    check_near_duplicate_names,
    check_orphan_nodes,
    check_self_loops,
    validate_graph,
)
from localdata_mcp.graph_validation_nx import (
    check_cycles,
    check_diamond_ambiguity,
    check_disconnected_components,
    check_redundant_transitive,
)


@pytest.fixture
def manager():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_graph_schema(engine)
    return GraphStorageManager(engine)


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


class TestSelfLoops:
    def test_detects_self_loop(self, manager):
        manager.create_node("A")
        manager.add_edge("A", "A", label="self")
        warnings = check_self_loops(manager)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "self_loop"

    def test_no_self_loop(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B")
        assert check_self_loops(manager) == []


class TestDuplicateEdges:
    def test_detects_duplicates(self, manager):
        """Simulate duplicate edges via direct SQL (bypasses upsert)."""
        manager.create_node("A")
        manager.create_node("B")
        with manager.engine.connect() as conn:
            for _ in range(2):
                conn.execute(
                    text(
                        "INSERT INTO graph_edges "
                        "(source_id, target_id, label, weight, "
                        "created_at, updated_at) "
                        "VALUES ('A', 'B', NULL, NULL, 0, 0)"
                    )
                )
            conn.commit()
        warnings = check_duplicate_edges(manager)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "duplicate_edges"

    def test_different_labels_not_duplicate(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B", label="knows")
        manager.add_edge("A", "B", label="likes")
        assert check_duplicate_edges(manager) == []


class TestOrphanNodes:
    def test_detects_orphan(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.create_node("lonely")
        manager.add_edge("A", "B")
        warnings = check_orphan_nodes(manager)
        assert len(warnings) == 1
        assert "lonely" in warnings[0]["details"]["nodes"]

    def test_no_orphans(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B")
        assert check_orphan_nodes(manager) == []


class TestMissingEdgeLabels:
    def test_detects_missing(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B")
        warnings = check_missing_edge_labels(manager)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "missing_edge_labels"

    def test_all_labeled(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B", label="knows")
        assert check_missing_edge_labels(manager) == []


# ---------------------------------------------------------------------------
# Semantic checks
# ---------------------------------------------------------------------------


class TestContradictoryEdges:
    def test_detects_contradictory(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B", label="broader")
        manager.add_edge("B", "A", label="broader")
        warnings = check_contradictory_edges(manager)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "contradictory_edges"

    def test_different_labels_ok(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B", label="broader")
        manager.add_edge("B", "A", label="narrower")
        assert check_contradictory_edges(manager) == []


class TestConflictingParallelLabels:
    def test_detects_conflicting(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B", label="broader")
        manager.add_edge("A", "B", label="related")
        warnings = check_conflicting_parallel_labels(manager)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "conflicting_parallel_labels"

    def test_single_label_ok(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B", label="knows")
        assert check_conflicting_parallel_labels(manager) == []


# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------


class TestDuplicateCasing:
    def test_detects_casing_variants(self, manager):
        manager.create_node("Data science")
        manager.create_node("Data Science")
        warnings = check_duplicate_casing(manager)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "duplicate_casing"

    def test_different_ids_ok(self, manager):
        manager.create_node("Alpha")
        manager.create_node("Beta")
        assert check_duplicate_casing(manager) == []


class TestMissingCommonProperties:
    def test_detects_missing(self, manager):
        for nid in ["A", "B", "C", "D"]:
            manager.create_node(nid)
        for nid in ["A", "B", "C"]:
            manager.set_property("node", nid, "category", "test")
        # D is missing "category" which 75% of nodes have
        warnings = check_missing_common_properties(manager)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "missing_common_property"
        assert "D" in warnings[0]["details"]["examples"]

    def test_no_common_properties(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        assert check_missing_common_properties(manager) == []


class TestNearDuplicateNames:
    def test_detects_similar_labels(self, manager):
        manager.create_node("A", label="Microbiology and Virology")
        manager.create_node("B", label="Microbiology and Immunology")
        warnings = check_near_duplicate_names(manager)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "near_duplicate_names"

    def test_different_labels_ok(self, manager):
        manager.create_node("A", label="Mathematics")
        manager.create_node("B", label="Literature")
        assert check_near_duplicate_names(manager) == []


# ---------------------------------------------------------------------------
# NetworkX-based checks
# ---------------------------------------------------------------------------


class TestCycles:
    def test_detects_cycle(self, manager):
        for nid in ["A", "B", "C"]:
            manager.create_node(nid)
        manager.add_edge("A", "B", label="x")
        manager.add_edge("B", "C", label="x")
        manager.add_edge("C", "A", label="x")
        G = _storage_to_networkx(manager)
        warnings = check_cycles(G)
        assert any(w["code"] == "cycle" for w in warnings)

    def test_dag_no_cycles(self, manager):
        for nid in ["A", "B", "C"]:
            manager.create_node(nid)
        manager.add_edge("A", "B", label="x")
        manager.add_edge("B", "C", label="x")
        G = _storage_to_networkx(manager)
        assert check_cycles(G) == []


class TestDisconnectedComponents:
    def test_detects_disconnected(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.create_node("X")
        manager.create_node("Y")
        manager.add_edge("A", "B", label="x")
        manager.add_edge("X", "Y", label="x")
        G = _storage_to_networkx(manager)
        warnings = check_disconnected_components(G)
        assert len(warnings) >= 1
        assert warnings[0]["code"] == "disconnected_component"

    def test_single_component_ok(self, manager):
        for nid in ["A", "B", "C"]:
            manager.create_node(nid)
        manager.add_edge("A", "B", label="x")
        manager.add_edge("B", "C", label="x")
        G = _storage_to_networkx(manager)
        assert check_disconnected_components(G) == []


class TestRedundantTransitive:
    def test_detects_redundant(self, manager):
        for nid in ["A", "B", "C"]:
            manager.create_node(nid)
        manager.add_edge("A", "B", label="x")
        manager.add_edge("B", "C", label="x")
        manager.add_edge("A", "C", label="x")  # redundant
        G = _storage_to_networkx(manager)
        warnings = check_redundant_transitive(G)
        assert any(w["code"] == "redundant_transitive" for w in warnings)

    def test_no_redundant_in_simple_dag(self, manager):
        for nid in ["A", "B", "C"]:
            manager.create_node(nid)
        manager.add_edge("A", "B", label="x")
        manager.add_edge("B", "C", label="x")
        G = _storage_to_networkx(manager)
        assert check_redundant_transitive(G) == []

    def test_skipped_on_cyclic_graph(self, manager):
        for nid in ["A", "B"]:
            manager.create_node(nid)
        manager.add_edge("A", "B", label="x")
        manager.add_edge("B", "A", label="x")
        G = _storage_to_networkx(manager)
        assert check_redundant_transitive(G) == []


class TestDiamondAmbiguity:
    def test_detects_multiple_parents(self, manager):
        for nid in ["A", "B", "C"]:
            manager.create_node(nid)
        manager.add_edge("A", "C", label="x")
        manager.add_edge("B", "C", label="x")
        G = _storage_to_networkx(manager)
        warnings = check_diamond_ambiguity(G)
        assert any(
            w["code"] == "diamond_ambiguity" and w["details"]["node"] == "C"
            for w in warnings
        )

    def test_single_parent_ok(self, manager):
        for nid in ["A", "B"]:
            manager.create_node(nid)
        manager.add_edge("A", "B", label="x")
        G = _storage_to_networkx(manager)
        assert check_diamond_ambiguity(G) == []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestValidateGraph:
    def test_clean_graph(self, manager):
        manager.create_node("A", label="Alice")
        manager.create_node("B", label="Bob")
        manager.add_edge("A", "B", label="knows")
        manager.set_property("node", "A", "age", 30)
        manager.set_property("node", "B", "age", 25)
        warnings = validate_graph(manager)
        assert warnings == []

    def test_multiple_issues(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.create_node("orphan")
        manager.add_edge("A", "B")  # no label
        warnings = validate_graph(manager)
        codes = {w["code"] for w in warnings}
        assert "orphan_nodes" in codes
        assert "missing_edge_labels" in codes

    def test_expensive_checks_skippable(self, manager):
        manager.create_node("A")
        manager.create_node("B")
        manager.add_edge("A", "B", label="x")
        w_full = validate_graph(manager, include_expensive=True)
        w_cheap = validate_graph(manager, include_expensive=False)
        # Both should work without errors
        assert isinstance(w_full, list)
        assert isinstance(w_cheap, list)
