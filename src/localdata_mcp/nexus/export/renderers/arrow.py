"""renderers/arrow.py — tabular payload as Arrow IPC file bytes (FR-902).

Typed binary interchange (same no-neutralization rationale as
parquet.py).
"""

from __future__ import annotations

import io
from typing import Any

import pyarrow
import pyarrow.ipc

from ..interface import as_dataframe

FORMAT = "arrow"


def render(payload: Any) -> bytes:
    table = pyarrow.Table.from_pandas(as_dataframe(payload), preserve_index=False)
    buffer = io.BytesIO()
    with pyarrow.ipc.new_file(buffer, table.schema) as writer:
        writer.write_table(table)
    return buffer.getvalue()
