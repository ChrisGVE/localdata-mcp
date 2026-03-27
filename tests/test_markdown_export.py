"""Tests for markdown export functionality."""

from localdata_mcp.markdown_export import (
    MAX_EXPORT_BYTES,
    _render_properties_markdown,
    _tree_node_to_markdown,
    escape_markdown_cell,
    escape_markdown_text,
    export_query_results_markdown,
    export_tree_markdown,
    format_query_results_as_markdown,
    generate_markdown_table,
)


class TestMarkdownEscaping:
    """Tests for escape_markdown_cell and escape_markdown_text."""

    def test_escape_pipe_characters(self):
        assert escape_markdown_cell("a|b") == "a\\|b"

    def test_escape_newlines(self):
        assert escape_markdown_cell("a\nb") == "a b"

    def test_escape_none(self):
        assert escape_markdown_cell(None) == ""

    def test_escape_numbers(self):
        assert escape_markdown_cell(42) == "42"

    def test_escape_empty_string(self):
        assert escape_markdown_cell("") == ""


class TestEscapeMarkdownText:
    """Tests for escape_markdown_text."""

    def test_escape_special_chars(self):
        result = escape_markdown_text("a*b_c")
        assert result == "a\\*b\\_c"

    def test_escape_none(self):
        assert escape_markdown_text(None) == ""

    def test_escape_backticks(self):
        assert escape_markdown_text("`code`") == "\\`code\\`"


class TestMarkdownTable:
    """Tests for generate_markdown_table."""

    def test_basic_table(self):
        result, truncated, total = generate_markdown_table(
            columns=["Name", "Age"],
            rows=[["Alice", 30], ["Bob", 25]],
        )
        assert not truncated
        assert total == 2
        lines = result.split("\n")
        assert lines[0] == "| Name | Age |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| Alice | 30 |"
        assert lines[3] == "| Bob | 25 |"

    def test_column_alignment_right(self):
        result, _, _ = generate_markdown_table(
            columns=["Name", "Score"],
            rows=[["Alice", 100]],
            alignments=["left", "right"],
        )
        lines = result.split("\n")
        assert "---:" in lines[1]
        assert lines[1] == "| --- | ---: |"

    def test_center_alignment(self):
        result, _, _ = generate_markdown_table(
            columns=["Name"],
            rows=[["Alice"]],
            alignments=["center"],
        )
        lines = result.split("\n")
        assert ":---:" in lines[1]

    def test_truncation_at_row_limit(self):
        rows = [[f"val_{i}"] for i in range(100)]
        result, truncated, total = generate_markdown_table(
            columns=["Col"],
            rows=rows,
            max_rows=10,
        )
        assert truncated is True
        assert total == 100
        assert "90 more rows not shown" in result
        assert "(total: 100)" in result
        # Count data rows (exclude header, separator, truncation notice)
        data_lines = [
            line
            for line in result.split("\n")
            if line.startswith("|")
            and "---" not in line
            and line != result.split("\n")[0]
        ]
        # Header is first pipe line, so data lines = pipe lines - 1 (header) - 1 (separator)
        pipe_lines = [line for line in result.split("\n") if line.startswith("|")]
        assert len(pipe_lines) == 12  # 1 header + 1 separator + 10 data

    def test_no_truncation_under_limit(self):
        rows = [[f"val_{i}"] for i in range(5)]
        result, truncated, total = generate_markdown_table(
            columns=["Col"],
            rows=rows,
            max_rows=50,
        )
        assert truncated is False
        assert total == 5
        assert "more rows not shown" not in result

    def test_special_chars_in_data(self):
        result, _, _ = generate_markdown_table(
            columns=["Data"],
            rows=[["has|pipe"], ["has\nnewline"]],
        )
        assert "\\|" in result
        assert "\n" not in result.split("\n")[2].replace("\n", "X")
        # Verify pipe is escaped in data row
        lines = result.split("\n")
        assert "has\\|pipe" in lines[2]

    def test_empty_rows(self):
        result, truncated, total = generate_markdown_table(
            columns=["A", "B"],
            rows=[],
        )
        assert truncated is False
        assert total == 0
        lines = result.split("\n")
        assert len(lines) == 2  # header + separator only

    def test_empty_columns(self):
        result, truncated, total = generate_markdown_table(
            columns=[],
            rows=[["a", "b"]],
        )
        assert result == ""
        assert truncated is False
        assert total == 0

    def test_single_column(self):
        result, _, total = generate_markdown_table(
            columns=["Only"],
            rows=[["one"], ["two"]],
        )
        assert total == 2
        lines = result.split("\n")
        assert lines[0] == "| Only |"
        assert lines[2] == "| one |"

    def test_single_row(self):
        result, truncated, total = generate_markdown_table(
            columns=["A", "B"],
            rows=[["x", "y"]],
        )
        assert truncated is False
        assert total == 1
        lines = result.split("\n")
        assert len(lines) == 3  # header + separator + 1 data row

    def test_returns_total_rows(self):
        _, _, total = generate_markdown_table(
            columns=["C"],
            rows=[[1], [2], [3], [4], [5]],
        )
        assert total == 5

    def test_byte_limit_truncation(self):
        """Generate a table exceeding MAX_EXPORT_BYTES and verify truncation."""
        # Each row ~200 bytes, need ~600 rows to exceed 100KB
        long_value = "x" * 180
        rows = [[long_value] for _ in range(1000)]
        result, truncated, total = generate_markdown_table(
            columns=["BigCol"],
            rows=rows,
            max_rows=1000,  # Allow all rows initially
        )
        assert truncated is True
        assert total == 1000
        assert len(result.encode("utf-8")) <= MAX_EXPORT_BYTES
        assert "more rows not shown" in result

    def test_row_shorter_than_columns(self):
        """Rows with fewer elements than columns get empty cells."""
        result, _, _ = generate_markdown_table(
            columns=["A", "B", "C"],
            rows=[["only_one"]],
        )
        lines = result.split("\n")
        assert lines[2] == "| only_one |  |  |"


class TestExportQueryResultsMarkdown:
    """Tests for export_query_results_markdown."""

    def test_basic_export(self):
        result = export_query_results_markdown(
            columns=["Name", "Age"],
            rows=[["Alice", 30], ["Bob", 25]],
        )
        assert result["format"] == "markdown"
        assert "| Name | Age |" in result["content"]
        assert "| Alice | 30 |" in result["content"]

    def test_with_query_header(self):
        result = export_query_results_markdown(
            columns=["Id"],
            rows=[[1]],
            query="SELECT * FROM users",
        )
        assert "**Query**: `SELECT * FROM users`" in result["content"]

    def test_without_query_header(self):
        result = export_query_results_markdown(
            columns=["Id"],
            rows=[[1]],
            query=None,
        )
        assert "Query" not in result["content"]

    def test_numeric_right_alignment(self):
        result = export_query_results_markdown(
            columns=["Name", "Score"],
            rows=[["Alice", 95], ["Bob", 80]],
        )
        lines = result["content"].split("\n")
        separator = [l for l in lines if "---" in l][0]
        # First column (str) left, second column (int) right
        assert "---:" in separator
        parts = separator.split("|")
        # parts: ['', ' --- ', ' ---: ', '']
        assert parts[1].strip() == "---"
        assert parts[2].strip() == "---:"

    def test_truncation_flag(self):
        rows = [[i] for i in range(10)]
        result = export_query_results_markdown(
            columns=["Val"],
            rows=rows,
            max_rows=5,
        )
        assert result["truncated"] is True
        assert result["total_rows"] == 10

    def test_total_rows_from_param(self):
        rows = [[i] for i in range(10)]
        result = export_query_results_markdown(
            columns=["Val"],
            rows=rows,
            total_rows=1000,
        )
        assert result["total_rows"] == 1000
        assert result["truncated"] is True
        assert "1000" in result["content"]

    def test_empty_results(self):
        result = export_query_results_markdown(
            columns=["A", "B"],
            rows=[],
        )
        assert result["format"] == "markdown"
        assert result["truncated"] is False
        assert result["total_rows"] == 0
        assert "| A | B |" in result["content"]

    def test_returns_dict_structure(self):
        result = export_query_results_markdown(
            columns=["X"],
            rows=[[1]],
        )
        assert set(result.keys()) == {"format", "content", "truncated", "total_rows"}


class TestTreeNodeToMarkdown:
    """Tests for _tree_node_to_markdown."""

    def test_simple_node(self):
        node = {"name": "Root"}
        lines = _tree_node_to_markdown(node, depth=1)
        assert lines[0] == "## Root"

    def test_nested_nodes(self):
        node = {
            "name": "Parent",
            "children": [{"name": "Child"}],
        }
        lines = _tree_node_to_markdown(node, depth=1)
        assert "## Parent" in lines
        assert "### Child" in lines

    def test_deep_nesting_beyond_headings(self):
        """Depth > 6 should use bullet points instead of headings."""
        node = {"name": "Deep"}
        lines = _tree_node_to_markdown(node, depth=7)
        assert any("- **Deep**" in line for line in lines)
        assert not any(line.startswith("#") for line in lines)

    def test_leaf_with_value(self):
        node = {"name": "Leaf", "value": 42}
        lines = _tree_node_to_markdown(node, depth=1)
        assert "## Leaf" in lines
        assert any("42" in line for line in lines)

    def test_node_with_properties(self):
        """A node with a properties dict should render bullet points."""
        node = {"name": "Config", "properties": {"env": "prod", "debug": False}}
        lines = _tree_node_to_markdown(node, depth=1)
        assert "## Config" in lines
        assert any("env" in line and "prod" in line for line in lines)
        assert any("debug" in line and "False" in line for line in lines)

    def test_node_with_key_fallback(self):
        """A node with 'key' instead of 'name' should still render."""
        node = {"key": "FallbackName"}
        lines = _tree_node_to_markdown(node, depth=1)
        assert any("FallbackName" in line for line in lines)


class TestRenderProperties:
    """Tests for _render_properties_markdown."""

    def test_simple_properties(self):
        props = {"color": "red", "size": 10}
        lines = _render_properties_markdown(props)
        assert len(lines) == 2
        assert any("color" in line and "red" in line for line in lines)
        assert any("size" in line and "10" in line for line in lines)

    def test_list_property(self):
        props = {"tags": ["a", "b", "c"]}
        lines = _render_properties_markdown(props)
        assert any("tags" in line for line in lines)
        assert any("1. a" in line for line in lines)
        assert any("2. b" in line for line in lines)
        assert any("3. c" in line for line in lines)

    def test_dict_property(self):
        props = {"nested": {"x": 1}}
        lines = _render_properties_markdown(props)
        assert len(lines) == 1
        assert "`" in lines[0]  # dict shown inline with backticks


class TestExportTreeMarkdown:
    """Tests for export_tree_markdown."""

    def test_basic_tree_export(self):
        tree = {
            "name": "Root",
            "children": [{"name": "A"}, {"name": "B"}],
        }
        result = export_tree_markdown(tree)
        assert result["format"] == "markdown"
        assert "## Root" in result["content"]
        assert "### A" in result["content"]
        assert "### B" in result["content"]

    def test_with_title(self):
        tree = {"name": "Node"}
        result = export_tree_markdown(tree, title="My Tree")
        assert result["content"].startswith("# My Tree")

    def test_list_of_nodes(self):
        nodes = [{"name": "First"}, {"name": "Second"}]
        result = export_tree_markdown(nodes)
        assert "## First" in result["content"]
        assert "## Second" in result["content"]

    def test_truncation_large_tree(self):
        """A tree that exceeds MAX_EXPORT_BYTES should be truncated."""
        big_value = "x" * 1000
        children = [{"name": f"child_{i}", "value": big_value} for i in range(200)]
        tree = {"name": "Big", "children": children}
        result = export_tree_markdown(tree)
        assert result["truncated"] is True
        assert "truncated due to size limits" in result["content"]
        assert len(result["content"].encode("utf-8")) <= (
            MAX_EXPORT_BYTES + 200  # small overhead from truncation message
        )

    def test_empty_tree(self):
        result = export_tree_markdown({})
        assert result["format"] == "markdown"
        assert result["truncated"] is False


class TestFormatQueryResultsAsMarkdown:
    """Tests for the execute_query bridge helper."""

    def test_basic_conversion(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = format_query_results_as_markdown(data)
        assert result["format"] == "markdown"
        assert "| name | age |" in result["content"]
        assert "Alice" in result["content"]
        assert result["total_rows"] == 2

    def test_empty_data(self):
        result = format_query_results_as_markdown([])
        assert result["format"] == "markdown"
        assert result["total_rows"] == 0

    def test_with_query_string(self):
        data = [{"id": 1}]
        result = format_query_results_as_markdown(data, query="SELECT id FROM t")
        assert "**Query**: `SELECT id FROM t`" in result["content"]

    def test_with_total_rows(self):
        data = [{"x": i} for i in range(5)]
        result = format_query_results_as_markdown(data, total_rows=500)
        assert result["total_rows"] == 500
        assert result["truncated"] is True

    def test_max_rows_respected(self):
        data = [{"v": i} for i in range(20)]
        result = format_query_results_as_markdown(data, max_rows=5)
        assert result["truncated"] is True
        assert "15 more rows" in result["content"]

    def test_preserves_column_order(self):
        data = [{"z_col": 1, "a_col": 2}]
        result = format_query_results_as_markdown(data)
        lines = result["content"].split("\n")
        header = lines[0]
        z_pos = header.index("z_col")
        a_pos = header.index("a_col")
        assert z_pos < a_pos
