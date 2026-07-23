"""localdata_mcp/explore/addressing.py — the exactly-one-source contract (E9).

X-2/X-3/X-4's shared disambiguation, one home: two separate optional
parameters — `endpoint=` (operator-declared name, NX-5 resolution)
and `path=` (local file, NX-6 containment) — EXACTLY one supplied;
zero or both is a structured invalid-arguments refusal naming both.
The second slot follows the same rule for endpoint sources: `table=`
(catalog membership verified, preparer-quoted by the guard's
read_table seam) or `query=` (SQL through `guarded_query`), exactly
one. Path sources take an optional `query=` for local SQLite/DuckDB
files (the I-2 ephemeral seam) and none for document/table formats —
`table=` with `path=` is refused (a file carries no catalog). No
string-sniffing overload anywhere (the I-1/I-2 precedent). The
resolved source lands as a pandas DataFrame for the profiling tools.
Neighbors: quality.py, search.py, categorical.py resolve through
here; ingest/refusals.py owns the wording.
"""

from __future__ import annotations

import contextlib
import contextvars
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from localdata_mcp.nexus.chokepoint.guard import (
    QueryRequest,
    Result,
    UnknownEndpointError,
)

from ..ingest.connectors.file.readers import read_path, resolve_format
from ..ingest.refusals import (
    invalid_source_refusal,
    missing_entity_refusal,
    unknown_endpoint_refusal,
)
from ..ingest.runtime import chokepoint

# Local file-ENGINE suffixes (the I-2 query_file surface) — a path
# whose suffix is an engine file takes query=; formats take none.
_ENGINE_SUFFIXES = {
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    ".duckdb": "duckdb",
    ".ddb": "duckdb",
}


# The E11 stage-injection channel: while a composition pipeline runs a
# DEPENDENT stage, the engine parks the upstream stage's output here
# and calls the stage tool with NO source parameters; resolve_frame
# picks it up below. Context-local by construction — a standalone call
# (empty context) still gets the X-2 exactly-one-source refusal, and
# the seam stays inside the ONE addressing home (E9.2).
_PIPELINE_INPUT: "contextvars.ContextVar[tuple[pd.DataFrame, str] | None]" = (
    contextvars.ContextVar("pipeline_input", default=None)
)


@contextlib.contextmanager
def pipeline_input(frame: pd.DataFrame, label: str) -> Iterator[None]:
    """Inject `frame` as the addressed data for tool calls made inside
    this context — the composition engine's stage-handoff seam."""
    token = _PIPELINE_INPUT.set((frame, label))
    try:
        yield
    finally:
        _PIPELINE_INPUT.reset(token)


def resolve_frame(
    endpoint: str | None,
    path: str | None,
    table: str | None,
    query: str | None,
) -> tuple[pd.DataFrame, str]:
    """The addressed data as a DataFrame plus a source label."""
    if endpoint is None and path is None:
        injected = _PIPELINE_INPUT.get()
        if injected is not None:
            frame, label = injected
            # A copy: a stage must not mutate its upstream sibling's
            # input (fan-out hands the same frame to several stages).
            return frame.copy(), label
    if (endpoint is None) == (path is None):
        raise invalid_source_refusal(
            "Exactly one of endpoint= and path= must be supplied "
            f"(got endpoint={endpoint!r}, path={path!r})."
        )
    if endpoint is not None:
        return _endpoint_frame(endpoint, table, query)
    assert path is not None
    return _path_frame(path, table, query)


def _endpoint_frame(
    endpoint: str, table: str | None, query: str | None
) -> tuple[pd.DataFrame, str]:
    if (table is None) == (query is None):
        raise invalid_source_refusal(
            "For an endpoint source, exactly one of table= and query= "
            f"must be supplied (got table={table!r}, query={query!r})."
        )
    try:
        if table is not None:
            result = chokepoint().read_table(endpoint, table)
            if result is None:
                raise missing_entity_refusal(
                    f"Table {table!r} does not exist on endpoint {endpoint!r}.",
                    "Call describe_database(endpoint) to list the tables.",
                )
            return _frame_of(result), f"{endpoint}:{table}"
        assert query is not None
        result = chokepoint().guarded_query(endpoint, QueryRequest(text=query))
        return _frame_of(result), f"{endpoint}:query"
    except UnknownEndpointError:
        raise unknown_endpoint_refusal(endpoint) from None


def _path_frame(
    path: str, table: str | None, query: str | None
) -> tuple[pd.DataFrame, str]:
    if table is not None:
        raise invalid_source_refusal(
            "table= applies to endpoint sources only — a file carries no "
            "table catalog; use query= for a local database file, or the "
            "bare path for a document/table format."
        )
    engine_kind = _ENGINE_SUFFIXES.get(Path(path).suffix.lower())
    if engine_kind is not None:
        if query is None:
            raise invalid_source_refusal(
                f"Path {path!r} is a local {engine_kind} database file — "
                "supply query= (one read-only SQL statement) to address "
                "its data."
            )
        result = chokepoint().guarded_file_query(
            path,
            QueryRequest(text=query),
            engine_kind,  # type: ignore[arg-type]
        )
        return _frame_of(result), f"{path}:query"
    if query is not None:
        raise invalid_source_refusal(
            "query= applies to endpoint sources and local database files "
            f"only — {path!r} is a document/table format; supply the bare "
            "path."
        )
    real = chokepoint().contain_path(path, mode="read")
    loaded = read_path(real, resolve_format(real, "auto"))
    if isinstance(loaded, pd.DataFrame):
        return loaded, str(path)
    raise missing_entity_refusal(
        f"File {path!r} loads as a document shape, not a table — the "
        "profiling tools address tabular data.",
        "Use read_file(path) to retrieve the document shape directly.",
    )


def _frame_of(result: Result) -> pd.DataFrame:
    return pd.DataFrame(list(result.rows), columns=list(result.columns))


def source_params() -> tuple[Any, ...]:
    """The four shared addressing Params, declared once (NFR-402)."""
    from localdata_mcp.nexus.contract.spec import Param

    return (
        Param(
            "endpoint",
            str,
            "The operator-declared endpoint name (exactly one of endpoint/path).",
            required=False,
        ),
        Param(
            "path",
            str,
            "A local file inside allowed_paths (exactly one of endpoint/path).",
            required=False,
        ),
        Param(
            "table",
            str,
            "A table on the endpoint (exactly one of table/query for "
            "endpoint sources).",
            required=False,
        ),
        Param(
            "query",
            str,
            "One read-only SQL statement (endpoint sources and local database files).",
            required=False,
        ),
    )
