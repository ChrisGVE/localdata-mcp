"""localdata_mcp/ingest/connectors/file/tools.py — the file family (E8.2).

I-2's two tools, thin over the guard: `read_file` resolves the format
(declared or by suffix), crosses NX-6 containment BEFORE any reader
touches the path, and dispatches to the hardened 14-format table
(readers.py); `query_file` runs SQL over a local SQLite/DuckDB file
through the guard's ephemeral seam — contained, allow-list screened,
read-only by default (NFR-114 i-b), load-then-serve by declaration
(the connection never outlives the call; the result is read whole
under the admission gate). An over-budget result becomes the
structured refusal whose suggestion names the SQL-narrowing recovery
(refusals.py). Neighbors: readers.py owns format hardening;
runtime.py supplies the guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from localdata_mcp.nexus.chokepoint.guard import (
    EphemeralEngineKind,
    QueryRequest,
    ResourceRefusedError,
    Result,
)
from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ...refusals import over_budget_refusal
from ...runtime import chokepoint
from .readers import FileIngestError, read_path, resolve_format

# Local file-ENGINE suffixes (I-2's query_file surface) — distinct from
# the document/table formats readers.py owns.
_ENGINE_SUFFIXES: dict[str, EphemeralEngineKind] = {
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    ".duckdb": "duckdb",
    ".ddb": "duckdb",
}


@tool_spec(
    name="read_file",
    summary=(
        "Read a local data file (14 core formats: CSV, TSV, JSON, YAML, "
        "TOML, INI, XML, Excel, ODS, Numbers, Parquet, Feather, Arrow, "
        "HDF5) inside the operator's allowed paths."
    ),
    params=(
        Param("path", str, "The file path (must lie inside allowed_paths)."),
        Param(
            "format",
            str,
            'The format name, or "auto" to infer from the suffix.',
            required=False,
        ),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
    streaming_capable=True,
    domain="ingest",
)
def read_file(path: str, format: str = "auto") -> Any:
    real = chokepoint().contain_path(path, mode="read")
    format_name = resolve_format(real, format)
    loaded = read_path(real, format_name)
    if isinstance(loaded, pd.DataFrame):
        result = Result(
            columns=tuple(str(column) for column in loaded.columns),
            rows=tuple(tuple(row) for row in loaded.itertuples(index=False)),
            category="local_file_read",
        )
        # I-4: beyond the inline budget the guard registers the loaded
        # frame as a load-then-serve stream (I-2's classification).
        return chokepoint().serve_result(result, str(real))
    return loaded


@tool_spec(
    name="query_file",
    summary=(
        "Run a read-only SQL statement over a local SQLite or DuckDB "
        "file (ad-hoc, contained, read-only unless the operator grants "
        "otherwise; results are read whole under the memory budget)."
    ),
    params=(
        Param("path", str, "The database file path (inside allowed_paths)."),
        Param("sql", str, "One SQL statement."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
    streaming_capable=True,
    domain="ingest",
)
def query_file(path: str, sql: str) -> Any:
    engine_kind = _ENGINE_SUFFIXES.get(Path(path).suffix.lower())
    if engine_kind is None:
        raise FileIngestError(
            f"suffix {Path(path).suffix!r} is not a local file-engine "
            f"(supported: {sorted(_ENGINE_SUFFIXES)}) — use read_file for "
            "document and table formats"
        )
    try:
        admitted = chokepoint().guarded_file_query(
            path, QueryRequest(text=sql), engine_kind
        )
    except ResourceRefusedError as refusal:
        raise over_budget_refusal(str(refusal)) from refusal
    # Load-then-serve by declaration (I-2): the whole result was read
    # under the admission gate; past the inline budget it is served
    # chunk by chunk from that admitted buffer.
    return chokepoint().serve_result(admitted, path)
