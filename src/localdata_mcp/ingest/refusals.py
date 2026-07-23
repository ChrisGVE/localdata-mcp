"""localdata_mcp/ingest/refusals.py — NFR-114 refusal guidance, one home (E8.1).

Every unknown-endpoint refusal across the whole tool surface carries
the SAME `suggestion` content (I-1's refusal-guidance contract): it
names `list_endpoints()` as the discovery path and states that
endpoints are operator-declared configuration, not caller-supplied —
the four-branch battery asserts the content, not merely the refusal.
One builder here so the wording cannot drift per connector family.
Neighbors: sql/tools.py and every later connector family route their
UnknownEndpointError through this.
"""

from __future__ import annotations

from localdata_mcp.nexus.chokepoint.guard import GuardedExecutionError
from localdata_mcp.nexus.error.model import ErrorType, StructuredError

_NFR114_SUGGESTION = (
    "Call list_endpoints() to discover the declared endpoints. Endpoints "
    "are operator-declared configuration (the endpoints section of the "
    "LocalData config), never caller-supplied — a DSN or new endpoint "
    "cannot be introduced from a tool call."
)


def unknown_endpoint_refusal(endpoint_name: str) -> GuardedExecutionError:
    """The structured NFR-114 refusal for an undeclared endpoint name."""
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.CONFIGURATION,
            message=f"no declared endpoint named {endpoint_name!r} (NFR-114)",
            suggestion=_NFR114_SUGGESTION,
            retryable=False,
        )
    )


def over_budget_refusal(detail: str) -> GuardedExecutionError:
    """I-2's over-budget admission refusal: the suggestion names the
    CALLER-side recovery (narrow the SQL) and states that the ceiling
    is operator configuration — asserted on content by the NFR-202
    matrix row."""
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.RESOURCE_ERROR,
            message=f"result refused by the memory-budget gate: {detail}",
            suggestion=(
                "Narrow the SQL predicate so the admitted result fits the "
                "memory budget — add a WHERE clause or LIMIT, or aggregate "
                "in SQL instead of retrieving raw rows. The memory ceiling "
                "itself is operator configuration "
                "(resources.memory_ceiling_bytes), not caller-adjustable."
            ),
            retryable=False,
        )
    )
