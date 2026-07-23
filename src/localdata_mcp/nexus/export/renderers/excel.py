"""renderers/excel.py — tabular payload as XLSX bytes (FR-902).

The primary CWE-1236 target: Excel executes formula-leading cells, so
neutralization is mandatory here (E7.3).
"""

from __future__ import annotations

import io
from typing import Any

from ..interface import as_dataframe, neutralize_formulas

FORMAT = "excel"


def render(payload: Any) -> bytes:
    frame = neutralize_formulas(as_dataframe(payload))
    buffer = io.BytesIO()
    frame.to_excel(buffer, engine="openpyxl", index=False)
    return buffer.getvalue()
