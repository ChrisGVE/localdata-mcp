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


def store_kind_refusal(
    endpoint_name: str, backend_kind: str, expected_kinds: tuple[str, ...]
) -> GuardedExecutionError:
    """E8.3's kind-mismatch refusal: a store-family tool was pointed at
    an endpoint whose declared kind it does not serve (the successor of
    `main`'s "'X' is not a tree-structured connection" errors, shaped
    through NX-3 with the NFR-114 discovery guidance)."""
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.CONFIGURATION,
            message=(
                f"endpoint {endpoint_name!r} is declared "
                f"backend_kind={backend_kind!r}; this tool serves "
                f"{' / '.join(expected_kinds)} endpoints"
            ),
            suggestion=(
                "Call list_endpoints() to see each declared endpoint's "
                "backend_kind. SQL endpoints are served by query/"
                "write_query, kv/tree/graph stores by the kv and "
                "graph/tree tool families, and rdf stores by query/"
                "write_query with SPARQL text."
            ),
            retryable=False,
        )
    )


def missing_entity_refusal(detail: str, discovery_hint: str) -> GuardedExecutionError:
    """A named node/property/edge does not exist in the store — the
    structured successor of `main`'s `{"error": "... not found"}` dict
    returns, with the discovery tool named for recovery."""
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.DATA_VALIDATION,
            message=detail,
            suggestion=discovery_hint,
            retryable=False,
        )
    )


def stream_refusal(failure: Exception) -> GuardedExecutionError:
    """I-4's structured stream refusals: the registry's own message
    (already caller-actionable) with the recovery path as the
    suggestion — expiry and served-chunk misses recover by re-issuing
    the originating query, admission by closing a finished stream."""
    text = str(failure)
    if "close_stream" in text or "admission" in text:
        suggestion = (
            "Call close_stream(stream_id) on a finished stream to free a "
            "slot, then re-issue the request. The per-endpoint stream cap "
            "is operator configuration "
            "(query.max_concurrent_streams_per_endpoint)."
        )
    elif "look-ahead" in text:
        suggestion = (
            "Retrieve the currently-servable chunks with fetch_chunk "
            "first — the look-ahead buffer tops up as chunks are consumed."
        )
    else:
        suggestion = (
            "Re-issue the originating query (or read_file/query_file "
            "call) to open a fresh stream — served chunks and expired "
            "streams are not resumable (cursor semantics)."
        )
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.RESOURCE_ERROR,
            message=text,
            suggestion=suggestion,
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
