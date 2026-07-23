"""localdata_mcp/ingest/connectors/sql/tools.py — the SQL family (E8.1).

I-1's two statements tools, THIN over NX-6 by design: `query` crosses
the guard's I-4 cutover seam (`query_or_stream` — an inline-budget
result comes back whole, a larger one as a `StreamOpened` reference
served by fetch_chunk, any posture), `write_query` crosses
`guarded_mutation` (read-write posture only, NFR-113) — every screen (E6.2 allow-list,
containment, admission, posture) lives in the guard, none here (§3's
connector boundary: no cross-connector knowledge, no security logic).
An undeclared endpoint name becomes the one NFR-114 refusal whose
suggestion names list_endpoints() (refusals.py). There is no
connect/disconnect surface — endpoint lifecycle is NX-5's (§4e).
Neighbors: runtime.py supplies the guard; endpoints.py is the
discovery surface the refusals point to.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.chokepoint.guard import (
    QueryRequest,
    StreamAdmissionRefusedError,
    UnknownEndpointError,
)
from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ...refusals import stream_refusal, unknown_endpoint_refusal
from ...runtime import chokepoint


@tool_spec(
    name="query",
    summary=(
        "Run a read-only SQL statement against a declared endpoint and "
        "return the rows (guarded: allow-list validated, any posture)."
    ),
    params=(
        Param("endpoint", str, "The operator-declared endpoint name."),
        Param("sql", str, "One read-only SQL statement."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
    streaming_capable=True,
    domain="ingest",
)
def query(endpoint: str, sql: str) -> Any:
    try:
        return chokepoint().query_or_stream(endpoint, QueryRequest(text=sql))
    except UnknownEndpointError:
        raise unknown_endpoint_refusal(endpoint) from None
    except StreamAdmissionRefusedError as refused:
        raise stream_refusal(refused) from refused


@tool_spec(
    name="write_query",
    summary=(
        "Run a mutating SQL statement (INSERT/UPDATE/DELETE or a "
        "write-side local-file construct) against a declared read-write "
        "endpoint (guarded: posture enforced, allow-list validated)."
    ),
    params=(
        Param("endpoint", str, "The operator-declared endpoint name."),
        Param("sql", str, "One mutating SQL statement."),
    ),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
    domain="ingest",
)
def write_query(endpoint: str, sql: str) -> Any:
    try:
        return chokepoint().guarded_mutation(endpoint, QueryRequest(text=sql))
    except UnknownEndpointError:
        raise unknown_endpoint_refusal(endpoint) from None
