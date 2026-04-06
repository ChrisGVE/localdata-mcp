"""Integration tests for markdown export across tree and graph pipelines.

Verifies the full flow: create storage -> populate -> export with
format='markdown' (and 'md' alias) -> validate markdown content.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from localdata_mcp.graph_manager import GraphStorageManager
from localdata_mcp.graph_storage import create_graph_schema
from localdata_mcp.graph_tools import tool_export_graph
from localdata_mcp.markdown_export import format_query_results_as_markdown
from localdata_mcp.tree_export import tool_export_structured
from localdata_mcp.tree_parsers import parse_json_to_tree
from localdata_mcp.tree_storage import TreeStorageManager, create_tree_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tree_mgr(tmp_path):
    """TreeStorageManager with a small JSON tree loaded."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_tree_schema(engine)
    mgr = TreeStorageManager(engine)

    json_file = tmp_path / "data.json"
    json_file.write_text(
        '{"server": {"host": "localhost", "port": 8080}, "debug": true}'
    )
    parse_json_to_tree(str(json_file), mgr)
    return mgr


@pytest.fixture
def graph_mgr():
    """GraphStorageManager with a small graph pre-loaded."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_graph_schema(engine)
    mgr = GraphStorageManager(engine)
    mgr.create_node("A", label="Alice")
    mgr.create_node("B", label="Bob")
    mgr.create_node("C", label="Charlie")
    mgr.add_edge("A", "B", label="knows", weight=1.0)
    mgr.add_edge("B", "C", label="likes", weight=0.5)
    return mgr


# ---------------------------------------------------------------------------
# Tree markdown integration
# ---------------------------------------------------------------------------


class TestTreeMarkdownIntegration:
    """End-to-end tree export with format='markdown'."""

    def test_export_structured_markdown_format(self, tree_mgr):
        result = tool_export_structured(tree_mgr, "test", "markdown")
        assert "content" in result
        assert result["format"] == "markdown"
        assert "error" not in result

    def test_md_alias_resolves_to_markdown(self, tree_mgr):
        """The 'md' alias should resolve to format='markdown'."""
        md_result = tool_export_structured(tree_mgr, "test", "md")
        assert md_result["format"] == "markdown"
        assert "error" not in md_result

    def test_md_and_markdown_produce_same_content(self, tree_mgr):
        md_result = tool_export_structured(tree_mgr, "test", "md")
        markdown_result = tool_export_structured(tree_mgr, "test", "markdown")
        assert md_result["content"] == markdown_result["content"]


# ---------------------------------------------------------------------------
# Graph markdown integration
# ---------------------------------------------------------------------------


class TestGraphMarkdownIntegration:
    """End-to-end graph export with format='markdown'."""

    def test_export_graph_markdown_format(self, graph_mgr):
        result = tool_export_graph(graph_mgr, "test-graph", "markdown")
        assert result["format"] == "markdown"
        assert "## Nodes" in result["content"]
        assert "## Edges" in result["content"]
        assert "Alice" in result["content"]
        assert "knows" in result["content"]

    def test_md_alias_resolves_to_markdown(self, graph_mgr):
        """The 'md' alias should resolve to format='markdown'."""
        md_result = tool_export_graph(graph_mgr, "test-graph", "md")
        assert md_result["format"] == "markdown"
        assert "error" not in md_result

    def test_md_and_markdown_produce_same_content(self, graph_mgr):
        md_result = tool_export_graph(graph_mgr, "test-graph", "md")
        markdown_result = tool_export_graph(graph_mgr, "test-graph", "markdown")
        assert md_result["content"] == markdown_result["content"]

    def test_graph_markdown_includes_mermaid(self, graph_mgr):
        result = tool_export_graph(graph_mgr, "test-graph", "markdown")
        assert "```mermaid" in result["content"]
        assert "graph TD" in result["content"]


# ---------------------------------------------------------------------------
# Query results markdown bridge
# ---------------------------------------------------------------------------


class TestQueryResultsMarkdownBridge:
    """Test format_query_results_as_markdown with realistic data shapes."""

    def test_roundtrip_from_dict_records(self):
        """Simulate execute_query JSON 'data' field -> markdown."""
        data = [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.0},
            {"id": 3, "name": "Charlie", "score": 72.3},
        ]
        result = format_query_results_as_markdown(
            data, query="SELECT * FROM students", total_rows=3
        )
        assert result["format"] == "markdown"
        assert "| id | name | score |" in result["content"]
        assert "Alice" in result["content"]
        assert "**Query**: `SELECT * FROM students`" in result["content"]
        assert result["total_rows"] == 3
        assert result["truncated"] is False

    def test_numeric_alignment_from_dict_records(self):
        """Numeric columns should be right-aligned."""
        data = [{"label": "x", "value": 42}]
        result = format_query_results_as_markdown(data)
        lines = result["content"].split("\n")
        sep = [l for l in lines if "---" in l][0]
        parts = sep.split("|")
        # label = left, value = right
        assert parts[1].strip() == "---"
        assert parts[2].strip() == "---:"
