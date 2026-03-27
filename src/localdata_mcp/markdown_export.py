"""Markdown export for query results, trees, and graphs.

Provides LLM-friendly markdown output for all LocalData data types.
Tables use GitHub-Flavored Markdown pipe syntax.
"""

from typing import Any, Dict, List, Optional, Tuple

MAX_EXPORT_BYTES = 100_000  # 100KB limit


def escape_markdown_cell(value: Any) -> str:
    """Escape a value for use in a markdown table cell."""
    if value is None:
        return ""
    text = str(value)
    # Escape pipe characters
    text = text.replace("|", "\\|")
    # Replace newlines with spaces
    text = text.replace("\n", " ").replace("\r", "")
    return text


def escape_markdown_text(value: Any) -> str:
    """Escape special markdown characters in general text."""
    if value is None:
        return ""
    text = str(value)
    # Escape characters that have markdown meaning
    for ch in [
        "\\",
        "`",
        "*",
        "_",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        "#",
        "+",
        "-",
        ".",
        "!",
    ]:
        text = text.replace(ch, f"\\{ch}")
    return text


def generate_markdown_table(
    columns: List[str],
    rows: List[List[Any]],
    max_rows: int = 50,
    alignments: Optional[List[str]] = None,
) -> Tuple[str, bool, int]:
    """Generate a GitHub-Flavored Markdown table.

    Args:
        columns: Column header names.
        rows: List of row data (each row is a list of values).
        max_rows: Maximum rows before truncation.
        alignments: Per-column alignment ("left", "right", "center").
            Default: left.

    Returns:
        Tuple of (markdown_string, was_truncated, total_rows).
    """
    if not columns:
        return "", False, 0

    total_rows = len(rows)
    truncated = total_rows > max_rows
    display_rows = rows[:max_rows]

    # Header
    header = "| " + " | ".join(escape_markdown_cell(c) for c in columns) + " |"

    # Separator with alignment
    if alignments is None:
        alignments = ["left"] * len(columns)
    sep_parts: list[str] = []
    for align in alignments:
        if align == "right":
            sep_parts.append("---:")
        elif align == "center":
            sep_parts.append(":---:")
        else:
            sep_parts.append("---")
    separator = "| " + " | ".join(sep_parts) + " |"

    # Data rows
    data_lines: list[str] = []
    for row in display_rows:
        cells = [
            escape_markdown_cell(row[i] if i < len(row) else "")
            for i in range(len(columns))
        ]
        data_lines.append("| " + " | ".join(cells) + " |")

    parts = [header, separator] + data_lines

    if truncated:
        parts.append(
            f"\n*... {total_rows - max_rows} more rows not shown (total: {total_rows})*"
        )

    result = "\n".join(parts)

    # Check byte limit
    if len(result.encode("utf-8")) > MAX_EXPORT_BYTES:
        # Re-truncate to fit
        while len(result.encode("utf-8")) > MAX_EXPORT_BYTES and data_lines:
            data_lines.pop()
            shown = len(data_lines)
            parts = [header, separator] + data_lines
            parts.append(
                f"\n*... {total_rows - shown} more rows "
                f"not shown (total: {total_rows})*"
            )
            result = "\n".join(parts)
            truncated = True

    return result, truncated, total_rows


def _render_properties_markdown(properties: Dict[str, Any]) -> List[str]:
    """Render node properties as markdown bullet points."""
    lines: List[str] = []
    for key, value in properties.items():
        if isinstance(value, dict):
            # Nested object — inline
            lines.append(f"- **{escape_markdown_text(str(key))}**: `{value}`")
        elif isinstance(value, list):
            lines.append(f"- **{escape_markdown_text(str(key))}**:")
            for i, item in enumerate(value):
                lines.append(f"  {i + 1}. {escape_markdown_cell(item)}")
        else:
            lines.append(
                f"- **{escape_markdown_text(str(key))}**: {escape_markdown_cell(value)}"
            )
    return lines


def _tree_node_to_markdown(
    node: Dict[str, Any],
    depth: int = 1,
    max_heading_depth: int = 6,
) -> List[str]:
    """Convert a tree node dict to markdown lines using heading hierarchy."""
    lines: List[str] = []
    name = node.get("name", node.get("key", ""))

    if depth <= max_heading_depth:
        lines.append(f"{'#' * (depth + 1)} {name}")
    else:
        indent = "  " * (depth - max_heading_depth)
        lines.append(f"{indent}- **{name}**")

    # Render value if leaf node
    value = node.get("value")
    if value is not None and not isinstance(value, (dict, list)):
        lines.append(f"\n{value}\n")

    # Render properties
    properties = node.get("properties", {})
    if properties:
        lines.extend(_render_properties_markdown(properties))

    # Recurse into children
    children = node.get("children", [])
    for child in children:
        lines.extend(_tree_node_to_markdown(child, depth + 1, max_heading_depth))

    return lines


def export_tree_markdown(
    tree_data: Dict[str, Any],
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Export structured tree data as markdown.

    Args:
        tree_data: Nested dict representing the tree
            (with "name", "children", "properties" keys).
        title: Optional title for the document.

    Returns:
        Dict with 'format', 'content', 'truncated'.
    """
    parts: List[str] = []
    if title:
        parts.append(f"# {title}\n")

    if isinstance(tree_data, dict):
        parts.extend(_tree_node_to_markdown(tree_data, depth=1))
    elif isinstance(tree_data, list):
        for item in tree_data:
            parts.extend(_tree_node_to_markdown(item, depth=1))

    content = "\n".join(parts)
    truncated = False

    if len(content.encode("utf-8")) > MAX_EXPORT_BYTES:
        encoded = content.encode("utf-8")[:MAX_EXPORT_BYTES]
        content = encoded.decode("utf-8", errors="ignore").rsplit("\n", 1)[0]
        content += "\n\n*... output truncated due to size limits*"
        truncated = True

    return {
        "format": "markdown",
        "content": content,
        "truncated": truncated,
    }


def export_query_results_markdown(
    columns: List[str],
    rows: List[List[Any]],
    total_rows: Optional[int] = None,
    query: Optional[str] = None,
    max_rows: int = 50,
) -> Dict[str, Any]:
    """Export query results as markdown table for LLM consumption.

    Args:
        columns: Column header names.
        rows: Row data.
        total_rows: Total rows in full result (may exceed len(rows)).
        query: Original query string (shown in header if provided).
        max_rows: Maximum rows to display.

    Returns:
        Dict with 'format', 'content', 'truncated', 'total_rows'.
    """
    alignments: List[str] = []
    if rows:
        for val in rows[0]:
            if isinstance(val, (int, float)):
                alignments.append("right")
            else:
                alignments.append("left")

    table_md, truncated, shown_total = generate_markdown_table(
        columns,
        rows,
        max_rows=max_rows,
        alignments=alignments or None,
    )

    actual_total = total_rows if total_rows is not None else shown_total

    parts: List[str] = []
    if query:
        parts.append(f"**Query**: `{query}`\n")
    parts.append(table_md)
    if actual_total > shown_total and not truncated:
        parts.append(f"\n*Showing {shown_total} of {actual_total} total rows.*")
        truncated = True

    return {
        "format": "markdown",
        "content": "\n".join(parts),
        "truncated": truncated,
        "total_rows": actual_total,
    }
