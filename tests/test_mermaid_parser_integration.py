"""Integration tests for Mermaid parser with fixture files and storage."""

import os
import tempfile

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

from localdata_mcp.graph_manager import GraphStorageManager
from localdata_mcp.mermaid_parser import MermaidFlowchartParser, parse_mermaid_to_graph

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_MMD = os.path.join(FIXTURES_DIR, "sample.mmd")
EDGE_CASES_MMD = os.path.join(FIXTURES_DIR, "mermaid_edge_cases.mmd")


def make_manager() -> GraphStorageManager:
    """Create an in-memory GraphStorageManager for testing."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return GraphStorageManager(engine)


# -- Integration: sample.mmd -------------------------------------------------


class TestSampleMmd:
    def test_parse_sample_file(self):
        mgr = make_manager()
        stats = parse_mermaid_to_graph(SAMPLE_MMD, mgr)
        assert stats["node_count"] == 7
        assert stats["edge_count"] == 7

    def test_sample_node_labels(self):
        G = MermaidFlowchartParser().parse(open(SAMPLE_MMD, encoding="utf-8").read())
        assert G.nodes["A"]["label"] == "API Gateway"
        assert G.nodes["B"]["label"] == "Auth Service"
        assert G.nodes["C"]["label"] == "Cache Layer"
        assert G.nodes["D"]["label"] == "Database"
        assert G.nodes["E"]["label"] == "Decision"
        assert G.nodes["F"]["label"] == "Process"
        assert G.nodes["G"]["label"] == "Reject"

    def test_sample_node_shapes(self):
        G = MermaidFlowchartParser().parse(open(SAMPLE_MMD, encoding="utf-8").read())
        assert G.nodes["A"]["shape"] == "rectangle"
        assert G.nodes["C"]["shape"] == "rounded"
        assert G.nodes["D"]["shape"] == "database"
        assert G.nodes["E"]["shape"] == "diamond"

    def test_sample_edge_labels(self):
        G = MermaidFlowchartParser().parse(open(SAMPLE_MMD, encoding="utf-8").read())
        assert G.edges["E", "F"]["label"] == "yes"
        assert G.edges["E", "G"]["label"] == "no"

    def test_sample_subgraph(self):
        G = MermaidFlowchartParser().parse(open(SAMPLE_MMD, encoding="utf-8").read())
        for nid in ["B", "D", "E", "F", "G"]:
            assert G.nodes[nid]["subgraph"] == "Backend"
        assert "subgraph" not in G.nodes.get("A", {})
        assert "subgraph" not in G.nodes.get("C", {})

    def test_sample_direction(self):
        G = MermaidFlowchartParser().parse(open(SAMPLE_MMD, encoding="utf-8").read())
        assert G.graph["direction"] == "TD"

    def test_storage_integration(self):
        mgr = make_manager()
        parse_mermaid_to_graph(SAMPLE_MMD, mgr)
        nodes = mgr.list_nodes(limit=100)
        node_ids = {n.node_id for n in nodes}
        assert node_ids == {"A", "B", "C", "D", "E", "F", "G"}
        edges = mgr.list_edges(limit=100)
        edge_pairs = {(e.source_id, e.target_id) for e in edges}
        expected = {
            ("A", "B"),
            ("A", "C"),
            ("B", "D"),
            ("C", "D"),
            ("B", "E"),
            ("E", "F"),
            ("E", "G"),
        }
        assert edge_pairs == expected


# -- Integration: mermaid_edge_cases.mmd --------------------------------------


class TestEdgeCasesMmd:
    def test_parse_edge_cases_file(self):
        mgr = make_manager()
        stats = parse_mermaid_to_graph(EDGE_CASES_MMD, mgr)
        # X, Y, Z, P, Q, R, S, T, U, V, W = 11 nodes
        assert stats["node_count"] == 11

    def test_chained_edges(self):
        G = MermaidFlowchartParser().parse(
            open(EDGE_CASES_MMD, encoding="utf-8").read()
        )
        assert G.has_edge("X", "Y")
        assert G.has_edge("Y", "Z")

    def test_undirected_edge(self):
        G = MermaidFlowchartParser().parse(
            open(EDGE_CASES_MMD, encoding="utf-8").read()
        )
        assert G.has_edge("P", "Q")
        assert G.has_edge("Q", "P")

    def test_dotted_edge(self):
        G = MermaidFlowchartParser().parse(
            open(EDGE_CASES_MMD, encoding="utf-8").read()
        )
        assert G.has_edge("R", "S")
        assert G.edges["R", "S"]["style"] == "dotted"

    def test_node_shapes_in_edge_cases(self):
        G = MermaidFlowchartParser().parse(
            open(EDGE_CASES_MMD, encoding="utf-8").read()
        )
        assert G.nodes["P"]["shape"] == "circle"
        assert G.nodes["Q"]["shape"] == "asymmetric"
        assert G.nodes["R"]["shape"] == "subroutine"
        assert G.nodes["S"]["shape"] == "stadium"

    def test_pre_label_edge(self):
        G = MermaidFlowchartParser().parse(
            open(EDGE_CASES_MMD, encoding="utf-8").read()
        )
        assert G.has_edge("T", "U")
        assert G.edges["T", "U"]["label"] == "label text"

    def test_pipe_label_edge(self):
        G = MermaidFlowchartParser().parse(
            open(EDGE_CASES_MMD, encoding="utf-8").read()
        )
        assert G.has_edge("V", "W")
        assert G.edges["V", "W"]["label"] == "pipe label"

    def test_direction_lr(self):
        G = MermaidFlowchartParser().parse(
            open(EDGE_CASES_MMD, encoding="utf-8").read()
        )
        assert G.graph["direction"] == "LR"


# -- Error handling -----------------------------------------------------------


class TestErrorHandling:
    def test_file_not_found(self):
        mgr = make_manager()
        with pytest.raises(ValueError, match="File not found"):
            parse_mermaid_to_graph("/nonexistent/file.mmd", mgr)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mmd", delete=False) as f:
            f.write("")
            f.flush()
            try:
                mgr = make_manager()
                with pytest.raises(
                    ValueError, match="No valid Mermaid flowchart header"
                ):
                    parse_mermaid_to_graph(f.name, mgr)
            finally:
                os.unlink(f.name)

    def test_non_flowchart_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mmd", delete=False) as f:
            f.write("sequenceDiagram\n    Alice->>Bob: Hello\n")
            f.flush()
            try:
                mgr = make_manager()
                with pytest.raises(
                    ValueError, match="No valid Mermaid flowchart header"
                ):
                    parse_mermaid_to_graph(f.name, mgr)
            finally:
                os.unlink(f.name)
