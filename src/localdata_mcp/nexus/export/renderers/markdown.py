"""renderers/markdown.py — tabular payload as a Markdown table document (FR-902).

The harvested `markdown_export.generate_markdown_table` design: GFM
table with cell escaping so no value breaks the structure. Markdown
tables are routinely pasted into spreadsheets, so the CWE-1236
neutralizer applies here too (E7.3's declared renderer set). NX-7's
inline region renders its own budget-bounded table (envelope.py) —
this renderer is the FILE artifact, a different concern with a
different owner.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..interface import as_dataframe, neutralize_formulas

FORMAT = "markdown"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render(payload: Any) -> bytes:
    frame = neutralize_formulas(as_dataframe(payload))
    return (_table_text(frame) + "\n").encode("utf-8")


def _table_text(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(_cell(c) for c in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join(
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    )
    return f"{header}\n{divider}\n{body}" if len(frame) else f"{header}\n{divider}"
