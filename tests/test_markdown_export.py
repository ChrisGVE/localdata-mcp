"""Tests for markdown export functionality."""

from localdata_mcp.markdown_export import (
    MAX_EXPORT_BYTES,
    escape_markdown_cell,
    escape_markdown_text,
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
