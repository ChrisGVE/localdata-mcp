"""localdata_mcp/ingest/endpoints.py — list_endpoints, the discovery tool (E8.1).

I-1's backend-kind-AGNOSTIC enumerator, homed at top level by contract
— every declared endpoint (SQL, kv, graph/tree alike) with its
`backend_kind`, `posture`, and last-known health summary, reached
through the NX-6 seam like every data touch and DSN-redacted by
construction (the guard's summaries carry NX-5's already-redacted
health text, never a DSN). There is no caller-facing connect or
disconnect: endpoint lifecycle is NX-5's, driven by NX-2 declarations
(§4e). A legitimately empty enumeration renders the explicit "zero
endpoints declared" statement naming the operator configuration step —
never an empty list (I-1/O-1). Neighbors: runtime.py supplies the
guard; every NFR-114 refusal (refusals.py) points here.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.chokepoint.guard import Result
from localdata_mcp.nexus.contract.spec import TypeShape, tool_spec

from .runtime import chokepoint


@tool_spec(
    name="list_endpoints",
    summary=(
        "Enumerate every operator-declared endpoint (SQL, key-value, and "
        "graph/tree alike) with its backend kind, posture, and health."
    ),
    params=(),
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
    domain="ingest",
)
def list_endpoints() -> Any:
    summaries = chokepoint().endpoint_summaries()
    if not summaries:
        return (
            "Zero endpoints declared — the operator has not configured any "
            "endpoint yet. Declare endpoints in the operator configuration "
            "(the endpoints section of the LocalData config file) and "
            "restart the server. This is a legitimate first-contact state, "
            "not a failure."
        )
    return Result(
        columns=("name", "backend_kind", "posture", "healthy", "health_detail"),
        rows=tuple(
            (
                summary.name,
                summary.backend_kind,
                summary.posture,
                summary.healthy,
                summary.health_detail,
            )
            for summary in summaries
        ),
        category="query",
    )
