"""renderers/csv.py — tabular payload as CSV bytes (FR-902).

Formula-injection defense applied (CWE-1236, E7.3): spreadsheet
software executes `=`/`+`/`-`/`@`-leading cells, so string cells are
neutralized before serialization.
"""

from __future__ import annotations

from typing import Any

from ..interface import as_dataframe, neutralize_formulas

FORMAT = "csv"


def render(payload: Any) -> bytes:
    frame = neutralize_formulas(as_dataframe(payload))
    return frame.to_csv(index=False).encode("utf-8")
