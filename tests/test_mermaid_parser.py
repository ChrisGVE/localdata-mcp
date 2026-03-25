"""Unit tests for MermaidFlowchartParser (headers, nodes, edges, subgraphs)."""

import pytest

from localdata_mcp.mermaid_parser import MermaidFlowchartParser


# -- Header detection ---------------------------------------------------------


class TestHeaderDetection:
    def test_graph_td(self):
        G = MermaidFlowchartParser().parse("graph TD\n    A --> B")
        assert G.graph["direction"] == "TD"

    def test_flowchart_lr(self):
        G = MermaidFlowchartParser().parse("flowchart LR\n    A --> B")
        assert G.graph["direction"] == "LR"

    def test_graph_no_direction_defaults_td(self):
        G = MermaidFlowchartParser().parse("graph\n    A --> B")
        assert G.graph["direction"] == "TD"

    def test_flowchart_bt(self):
        G = MermaidFlowchartParser().parse("flowchart BT\n    A --> B")
        assert G.graph["direction"] == "BT"

    def test_flowchart_rl(self):
        G = MermaidFlowchartParser().parse("flowchart RL\n    A --> B")
        assert G.graph["direction"] == "RL"

    def test_flowchart_tb(self):
        G = MermaidFlowchartParser().parse("flowchart TB\n    A --> B")
        assert G.graph["direction"] == "TB"

    def test_case_insensitive_header(self):
        G = MermaidFlowchartParser().parse("GRAPH td\n    A --> B")
        assert G.graph["direction"] == "TD"

    def test_invalid_header_raises(self):
        with pytest.raises(ValueError, match="No valid Mermaid flowchart header"):
            MermaidFlowchartParser().parse("digraph {\n  A -> B\n}")

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="No valid Mermaid flowchart header"):
            MermaidFlowchartParser().parse("")


# -- Node shapes --------------------------------------------------------------


class TestNodeShapes:
    def _parse_single(self, declaration: str):
        text = f"graph TD\n    {declaration}"
        return MermaidFlowchartParser().parse(text)

    def test_bare_node(self):
        G = self._parse_single("A")
        assert "A" in G.nodes
        assert G.nodes["A"]["shape"] == "rectangle"

    def test_rectangle(self):
        G = self._parse_single("A[My Label]")
        assert G.nodes["A"]["label"] == "My Label"
        assert G.nodes["A"]["shape"] == "rectangle"

    def test_rounded(self):
        G = self._parse_single("A(Rounded)")
        assert G.nodes["A"]["label"] == "Rounded"
        assert G.nodes["A"]["shape"] == "rounded"

    def test_diamond(self):
        G = self._parse_single("A{Diamond}")
        assert G.nodes["A"]["label"] == "Diamond"
        assert G.nodes["A"]["shape"] == "diamond"

    def test_circle(self):
        G = self._parse_single("A((Circle))")
        assert G.nodes["A"]["label"] == "Circle"
        assert G.nodes["A"]["shape"] == "circle"

    def test_stadium(self):
        G = self._parse_single("A([Stadium])")
        assert G.nodes["A"]["label"] == "Stadium"
        assert G.nodes["A"]["shape"] == "stadium"

    def test_subroutine(self):
        G = self._parse_single("A[[Subroutine]]")
        assert G.nodes["A"]["label"] == "Subroutine"
        assert G.nodes["A"]["shape"] == "subroutine"

    def test_database(self):
        G = self._parse_single("A[(Database)]")
        assert G.nodes["A"]["label"] == "Database"
        assert G.nodes["A"]["shape"] == "database"

    def test_asymmetric(self):
        G = self._parse_single("A>Asymmetric]")
        assert G.nodes["A"]["label"] == "Asymmetric"
        assert G.nodes["A"]["shape"] == "asymmetric"

    def test_label_with_spaces(self):
        G = self._parse_single("A[My Long Label]")
        assert G.nodes["A"]["label"] == "My Long Label"


# -- Edge styles and labels ---------------------------------------------------


class TestEdges:
    def _parse(self, body: str):
        return MermaidFlowchartParser().parse(f"graph TD\n    {body}")

    def test_solid_directed(self):
        G = self._parse("A --> B")
        assert G.has_edge("A", "B")
        assert G.edges["A", "B"]["style"] == "solid"

    def test_thick_directed(self):
        G = self._parse("A ==> B")
        assert G.has_edge("A", "B")
        assert G.edges["A", "B"]["style"] == "thick"

    def test_dotted_directed(self):
        G = self._parse("A -.-> B")
        assert G.has_edge("A", "B")
        assert G.edges["A", "B"]["style"] == "dotted"

    def test_solid_undirected_creates_bidirectional(self):
        G = self._parse("A --- B")
        assert G.has_edge("A", "B")
        assert G.has_edge("B", "A")
        assert G.edges["A", "B"]["style"] == "solid"
        assert G.edges["B", "A"]["style"] == "solid"

    def test_pipe_label(self):
        G = self._parse("A -->|yes| B")
        assert G.has_edge("A", "B")
        assert G.edges["A", "B"]["label"] == "yes"

    def test_pre_label(self):
        G = self._parse("A -- label text --> B")
        assert G.has_edge("A", "B")
        assert G.edges["A", "B"]["label"] == "label text"

    def test_chained_edges(self):
        G = self._parse("A --> B --> C")
        assert G.has_edge("A", "B")
        assert G.has_edge("B", "C")
        assert len(G.edges) == 2


# -- Subgraphs ---------------------------------------------------------------


class TestSubgraphs:
    def test_basic_subgraph(self):
        text = "graph TD\n    subgraph MyGroup\n        A\n        B\n    end\n    C"
        G = MermaidFlowchartParser().parse(text)
        assert G.nodes["A"]["subgraph"] == "MyGroup"
        assert G.nodes["B"]["subgraph"] == "MyGroup"
        assert "subgraph" not in G.nodes["C"]

    def test_nodes_in_edge_inherit_subgraph(self):
        text = "graph TD\n    subgraph Backend\n        A --> B\n    end\n"
        G = MermaidFlowchartParser().parse(text)
        assert G.nodes["A"]["subgraph"] == "Backend"
        assert G.nodes["B"]["subgraph"] == "Backend"


# -- Comments and semicolons --------------------------------------------------


class TestCommentsAndSemicolons:
    def test_comment_lines_ignored(self):
        text = "graph TD\n    %% this is a comment\n    A --> B"
        G = MermaidFlowchartParser().parse(text)
        assert len(G.nodes) == 2

    def test_inline_comment_stripped(self):
        text = "graph TD\n    A --> B %% inline comment"
        G = MermaidFlowchartParser().parse(text)
        assert G.has_edge("A", "B")

    def test_semicolons_stripped(self):
        text = "graph TD\n    A --> B;"
        G = MermaidFlowchartParser().parse(text)
        assert G.has_edge("A", "B")


# -- Re-export from graph_parsers ---------------------------------------------


class TestReExport:
    def test_import_from_graph_parsers(self):
        from localdata_mcp.graph_parsers import (
            MermaidFlowchartParser as ReExported,
            parse_mermaid_to_graph as re_exported_fn,
        )
        from localdata_mcp.mermaid_parser import (
            parse_mermaid_to_graph,
        )

        assert ReExported is MermaidFlowchartParser
        assert re_exported_fn is parse_mermaid_to_graph
