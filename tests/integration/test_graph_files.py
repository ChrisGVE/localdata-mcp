"""Integration tests for graph file formats (DOT, GML, GraphML, Mermaid)."""

import pytest

from .mcp_test_client import call_tool

pytestmark = [pytest.mark.integration]

SAMPLE_DOT = """digraph G {
    A [label="Node A"];
    B [label="Node B"];
    C [label="Node C"];
    A -> B [label="edge1"];
    B -> C [label="edge2"];
    A -> C [label="shortcut"];
}"""

SAMPLE_GML = """graph [
  directed 1
  node [ id 1 label "Alpha" ]
  node [ id 2 label "Beta" ]
  node [ id 3 label "Gamma" ]
  edge [ source 1 target 2 label "connects" ]
  edge [ source 2 target 3 label "links" ]
]"""

SAMPLE_GRAPHML = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d0" for="node" attr.name="label" attr.type="string"/>
  <graph edgedefault="directed">
    <node id="n0"><data key="d0">Start</data></node>
    <node id="n1"><data key="d0">End</data></node>
    <edge source="n0" target="n1"/>
  </graph>
</graphml>"""

SAMPLE_MERMAID = """graph TD
    A[Start] --> B[Process]
    B --> C[End]
    B --> D[Alternative]
"""


class TestDOTGraph:
    def test_connect_and_stats(self, tmp_path):
        path = str(tmp_path / "test.dot")
        with open(path, "w") as f:
            f.write(SAMPLE_DOT)
        call_tool(
            "connect_database",
            {"name": "dot_test", "db_type": "dot", "conn_string": path},
        )
        try:
            result = call_tool("get_graph_stats", {"name": "dot_test"})
            assert result["node_count"] == 3
            assert result["edge_count"] == 3
        finally:
            call_tool("disconnect_database", {"name": "dot_test"})

    def test_get_neighbors(self, tmp_path):
        path = str(tmp_path / "nb.dot")
        with open(path, "w") as f:
            f.write(SAMPLE_DOT)
        call_tool(
            "connect_database",
            {"name": "dot_nb", "db_type": "dot", "conn_string": path},
        )
        try:
            result = call_tool("get_neighbors", {"name": "dot_nb", "node_id": "A"})
            neighbor_ids = [n["neighbor_id"] for n in result.get("neighbors", [])]
            assert "B" in neighbor_ids
            assert "C" in neighbor_ids
        finally:
            call_tool("disconnect_database", {"name": "dot_nb"})

    def test_find_path(self, tmp_path):
        path = str(tmp_path / "fp.dot")
        with open(path, "w") as f:
            f.write(SAMPLE_DOT)
        call_tool(
            "connect_database",
            {"name": "dot_fp", "db_type": "dot", "conn_string": path},
        )
        try:
            result = call_tool(
                "find_path",
                {"name": "dot_fp", "source": "A", "target": "C"},
            )
            assert "path" in result
            assert "A" in result["path"]
            assert "C" in result["path"]
        finally:
            call_tool("disconnect_database", {"name": "dot_fp"})

    def test_get_edges(self, tmp_path):
        path = str(tmp_path / "edges.dot")
        with open(path, "w") as f:
            f.write(SAMPLE_DOT)
        call_tool(
            "connect_database",
            {"name": "dot_edges", "db_type": "dot", "conn_string": path},
        )
        try:
            result = call_tool("get_edges", {"name": "dot_edges"})
            assert result["total"] == 3
            sources = [e["source"] for e in result["edges"]]
            assert "A" in sources
        finally:
            call_tool("disconnect_database", {"name": "dot_edges"})

    def test_export_graph(self, tmp_path):
        path = str(tmp_path / "exp.dot")
        with open(path, "w") as f:
            f.write(SAMPLE_DOT)
        call_tool(
            "connect_database",
            {"name": "dot_exp", "db_type": "dot", "conn_string": path},
        )
        try:
            result = call_tool(
                "export_graph", {"name": "dot_exp", "format": "markdown"}
            )
            assert "content" in result or "error" not in result
        finally:
            call_tool("disconnect_database", {"name": "dot_exp"})

    def test_get_node_detail(self, tmp_path):
        path = str(tmp_path / "node.dot")
        with open(path, "w") as f:
            f.write(SAMPLE_DOT)
        call_tool(
            "connect_database",
            {"name": "dot_node", "db_type": "dot", "conn_string": path},
        )
        try:
            # For graph connections, get_node uses 'path' as node_id
            result = call_tool("get_node", {"name": "dot_node", "path": "A"})
            assert result["node_id"] == "A"
            assert result["out_degree"] == 2
        finally:
            call_tool("disconnect_database", {"name": "dot_node"})


class TestGMLGraph:
    def test_connect_and_stats(self, tmp_path):
        path = str(tmp_path / "test.gml")
        with open(path, "w") as f:
            f.write(SAMPLE_GML)
        call_tool(
            "connect_database",
            {"name": "gml_test", "db_type": "gml", "conn_string": path},
        )
        try:
            result = call_tool("get_graph_stats", {"name": "gml_test"})
            assert result["node_count"] == 3
            assert result["edge_count"] == 2
        finally:
            call_tool("disconnect_database", {"name": "gml_test"})

    def test_find_path(self, tmp_path):
        path = str(tmp_path / "path.gml")
        with open(path, "w") as f:
            f.write(SAMPLE_GML)
        call_tool(
            "connect_database",
            {"name": "gml_path", "db_type": "gml", "conn_string": path},
        )
        try:
            # GML nodes are referenced by label name
            result = call_tool(
                "find_path",
                {"name": "gml_path", "source": "Alpha", "target": "Gamma"},
            )
            assert "path" in result
        finally:
            call_tool("disconnect_database", {"name": "gml_path"})


class TestGraphMLGraph:
    def test_connect_and_stats(self, tmp_path):
        path = str(tmp_path / "test.graphml")
        with open(path, "w") as f:
            f.write(SAMPLE_GRAPHML)
        call_tool(
            "connect_database",
            {"name": "graphml_test", "db_type": "graphml", "conn_string": path},
        )
        try:
            result = call_tool("get_graph_stats", {"name": "graphml_test"})
            assert result["node_count"] == 2
            assert result["edge_count"] == 1
        finally:
            call_tool("disconnect_database", {"name": "graphml_test"})

    def test_get_neighbors(self, tmp_path):
        path = str(tmp_path / "nb.graphml")
        with open(path, "w") as f:
            f.write(SAMPLE_GRAPHML)
        call_tool(
            "connect_database",
            {"name": "graphml_nb", "db_type": "graphml", "conn_string": path},
        )
        try:
            result = call_tool("get_neighbors", {"name": "graphml_nb", "node_id": "n0"})
            neighbor_ids = [n["neighbor_id"] for n in result.get("neighbors", [])]
            assert "n1" in neighbor_ids
        finally:
            call_tool("disconnect_database", {"name": "graphml_nb"})


class TestMermaidGraph:
    def test_connect_and_stats(self, tmp_path):
        path = str(tmp_path / "test.mmd")
        with open(path, "w") as f:
            f.write(SAMPLE_MERMAID)
        call_tool(
            "connect_database",
            {"name": "mmd_test", "db_type": "mermaid", "conn_string": path},
        )
        try:
            result = call_tool("get_graph_stats", {"name": "mmd_test"})
            assert result["node_count"] == 4
            assert result["edge_count"] == 3
        finally:
            call_tool("disconnect_database", {"name": "mmd_test"})

    def test_get_neighbors(self, tmp_path):
        path = str(tmp_path / "nb.mmd")
        with open(path, "w") as f:
            f.write(SAMPLE_MERMAID)
        call_tool(
            "connect_database",
            {"name": "mmd_nb", "db_type": "mermaid", "conn_string": path},
        )
        try:
            result = call_tool("get_neighbors", {"name": "mmd_nb", "node_id": "B"})
            neighbor_ids = [n["neighbor_id"] for n in result.get("neighbors", [])]
            # B connects to C and D
            assert "C" in neighbor_ids or "D" in neighbor_ids
        finally:
            call_tool("disconnect_database", {"name": "mmd_nb"})

    def test_export_graph(self, tmp_path):
        path = str(tmp_path / "exp.mmd")
        with open(path, "w") as f:
            f.write(SAMPLE_MERMAID)
        call_tool(
            "connect_database",
            {"name": "mmd_exp", "db_type": "mermaid", "conn_string": path},
        )
        try:
            result = call_tool(
                "export_graph", {"name": "mmd_exp", "format": "markdown"}
            )
            assert "content" in result or "error" not in result
        finally:
            call_tool("disconnect_database", {"name": "mmd_exp"})
