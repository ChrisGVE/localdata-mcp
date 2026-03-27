"""Tests for graph markdown export functionality."""

from localdata_mcp.graph_markdown_export import (
    _generate_edge_table_markdown,
    _generate_graph_summary_markdown,
    _generate_mermaid_block,
    _generate_node_table_markdown,
    export_graph_markdown,
)
from localdata_mcp.markdown_export import MAX_EXPORT_BYTES


class TestGraphSummary:
    """Tests for _generate_graph_summary_markdown."""

    def test_basic_summary(self):
        stats = {"node_count": 5, "edge_count": 8}
        result = _generate_graph_summary_markdown(stats)
        assert "**Nodes**: 5" in result
        assert "**Edges**: 8" in result

    def test_summary_with_density(self):
        stats = {"node_count": 3, "edge_count": 2, "density": 0.3333}
        result = _generate_graph_summary_markdown(stats)
        assert "**Density**: 0.3333" in result

    def test_summary_with_dag_and_components(self):
        stats = {
            "node_count": 4,
            "edge_count": 3,
            "is_dag": True,
            "connected_components": 2,
        }
        result = _generate_graph_summary_markdown(stats)
        assert "**DAG**: Yes" in result
        assert "**Connected components**: 2" in result

    def test_summary_dag_false(self):
        stats = {"node_count": 1, "edge_count": 0, "is_dag": False}
        result = _generate_graph_summary_markdown(stats)
        assert "**DAG**: No" in result


class TestNodeTable:
    """Tests for _generate_node_table_markdown."""

    def test_node_table(self):
        nodes = [
            {"id": "A", "label": "Alice", "properties": {"role": "admin"}},
            {"id": "B", "name": "Bob", "properties": {}},
        ]
        result = _generate_node_table_markdown(nodes)
        assert "## Nodes" in result
        assert "| ID | Label | Properties |" in result
        assert "Alice" in result
        assert "role=admin" in result

    def test_empty_nodes(self):
        result = _generate_node_table_markdown([])
        assert result == "*No nodes*"

    def test_node_uses_name_fallback(self):
        nodes = [{"id": "X", "name": "Xavier"}]
        result = _generate_node_table_markdown(nodes)
        assert "Xavier" in result


class TestEdgeTable:
    """Tests for _generate_edge_table_markdown."""

    def test_edge_table(self):
        edges = [
            {"source": "A", "target": "B", "label": "knows", "weight": 1.5},
        ]
        result = _generate_edge_table_markdown(edges)
        assert "## Edges" in result
        assert "| Source | Target | Label | Weight |" in result
        assert "knows" in result
        assert "1.5" in result

    def test_empty_edges(self):
        result = _generate_edge_table_markdown([])
        assert result == "*No edges*"


class TestMermaidBlock:
    """Tests for _generate_mermaid_block."""

    def test_mermaid_with_labels(self):
        nodes = [
            {"id": "A", "label": "Alice"},
            {"id": "B", "label": "Bob"},
        ]
        edges = [{"source": "A", "target": "B", "label": "knows"}]
        result = _generate_mermaid_block(nodes, edges)
        assert result is not None
        assert "```mermaid" in result
        assert "graph TD" in result
        assert "A[Alice]" in result
        assert "A -->|knows| B" in result

    def test_mermaid_too_large(self):
        nodes = [{"id": str(i), "label": f"N{i}"} for i in range(51)]
        edges = []
        result = _generate_mermaid_block(nodes, edges, max_nodes=50)
        assert result is None

    def test_mermaid_edge_without_label(self):
        nodes = [{"id": "X"}, {"id": "Y"}]
        edges = [{"source": "X", "target": "Y"}]
        result = _generate_mermaid_block(nodes, edges)
        assert "X --> Y" in result

    def test_mermaid_spaces_replaced(self):
        nodes = [{"id": "node one", "label": "Node One"}]
        edges = []
        result = _generate_mermaid_block(nodes, edges)
        assert "node_one[Node One]" in result


class TestExportGraphMarkdown:
    """Tests for export_graph_markdown."""

    def _sample_nodes(self):
        return [
            {"id": "A", "label": "Alice", "properties": {"role": "admin"}},
            {"id": "B", "label": "Bob", "properties": {}},
        ]

    def _sample_edges(self):
        return [{"source": "A", "target": "B", "label": "knows", "weight": 1.0}]

    def test_full_graph_export(self):
        stats = {"node_count": 2, "edge_count": 1, "density": 0.5}
        result = export_graph_markdown(
            self._sample_nodes(), self._sample_edges(), stats=stats
        )
        assert result["format"] == "markdown"
        assert "## Graph Summary" in result["content"]
        assert "## Nodes" in result["content"]
        assert "## Edges" in result["content"]
        assert "```mermaid" in result["content"]

    def test_graph_with_title(self):
        result = export_graph_markdown(
            self._sample_nodes(), self._sample_edges(), title="My Graph"
        )
        assert result["content"].startswith("# My Graph")

    def test_graph_without_mermaid(self):
        result = export_graph_markdown(
            self._sample_nodes(), self._sample_edges(), include_mermaid=False
        )
        assert "```mermaid" not in result["content"]
        assert "## Nodes" in result["content"]

    def test_graph_truncation(self):
        big_nodes = [
            {"id": str(i), "label": "x" * 500, "properties": {"data": "y" * 500}}
            for i in range(300)
        ]
        big_edges = [
            {"source": str(i), "target": str(i + 1), "label": "z" * 200}
            for i in range(299)
        ]
        result = export_graph_markdown(
            big_nodes, big_edges, max_rows=300, include_mermaid=False
        )
        assert result["truncated"] is True
        assert "truncated due to size limits" in result["content"]
        assert len(result["content"].encode("utf-8")) <= MAX_EXPORT_BYTES + 200

    def test_graph_no_stats(self):
        result = export_graph_markdown(self._sample_nodes(), self._sample_edges())
        assert "## Graph Summary" not in result["content"]
        assert "## Nodes" in result["content"]
