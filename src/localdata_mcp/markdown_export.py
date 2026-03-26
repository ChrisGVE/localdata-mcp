"""Markdown export for query results, trees, and graphs.

Provides LLM-friendly markdown output for all LocalData data types.
Tables use GitHub-Flavored Markdown pipe syntax.
"""

from typing import Any, List, Optional, Tuple

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
