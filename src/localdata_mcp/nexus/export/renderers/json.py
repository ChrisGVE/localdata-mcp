"""renderers/json.py — payload as JSON bytes (FR-902).

Tabular payloads serialize as a records array; a mapping payload
(schema-like or key-value results) serializes as the object it is.
JSON carries no formula semantics — no CWE-1236 neutralization.
"""

from __future__ import annotations

import json as _json
from typing import Any, Mapping

from ..interface import as_dataframe

FORMAT = "json"


def render(payload: Any) -> bytes:
    if isinstance(payload, Mapping):
        return _json.dumps(payload, indent=2, default=str).encode("utf-8")
    frame = as_dataframe(payload)
    records = frame.to_dict(orient="records")
    return _json.dumps(records, indent=2, default=str).encode("utf-8")
