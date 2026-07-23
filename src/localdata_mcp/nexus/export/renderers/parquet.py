"""renderers/parquet.py — tabular payload as Parquet bytes (FR-902).

A typed binary format no spreadsheet interprets — the CWE-1236
neutralizer deliberately does NOT apply (it would corrupt data in a
format that carries no formula semantics).
"""

from __future__ import annotations

import io
from typing import Any

from ..interface import as_dataframe

FORMAT = "parquet"


def render(payload: Any) -> bytes:
    buffer = io.BytesIO()
    as_dataframe(payload).to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()
