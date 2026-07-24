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


def invalid_source_refusal(detail: str) -> GuardedExecutionError:
    """X-2's exactly-one-source contract (E9): zero or both of the two
    addressing parameters — or of a tool's second slot — is a
    structured invalid-arguments refusal NAMING the parameters, never
    a string-sniffing overload."""
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.DATA_VALIDATION,
            message=detail,
            suggestion=(
                "Supply exactly one source: endpoint= (an operator-declared "
                "endpoint name — discover with list_endpoints()) OR path= (a "
                "local file inside allowed_paths). For endpoint sources, "
                "supply exactly one of table= or query=."
            ),
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


def export_source_refusal(detail: str) -> GuardedExecutionError:
    """E13's exactly-one-source contract for `export_result`: the tool
    exports inline `source=`, a `stream_id=`, or a composition leaf
    (the upstream stage's output, injected when it runs as a pipeline
    terminal). Zero or more than one of the explicit slots — or neither
    slot with no upstream stage — is a structured refusal NAMING the
    slots, the addressing precedent (E9) applied to the write surface."""
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.DATA_VALIDATION,
            message=detail,
            suggestion=(
                "Supply exactly one export source: source= (inline data — a "
                "records list, a mapping, or a rendered artifact envelope) OR "
                "stream_id= (a buffered result to drain to the file). As a "
                "composition terminal, supply neither — the upstream stage's "
                "output is injected automatically."
            ),
            retryable=False,
        )
    )


def unknown_format_refusal(detail: str) -> GuardedExecutionError:
    """E13's format-name guard for `export_result`: an unregistered
    format is a caller-fixable argument error, not an internal lookup
    failure — the structured refusal names the requested format and the
    supported set (the NX-3 shape, the render_chart format precedent)."""
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.DATA_VALIDATION,
            message=detail,
            suggestion=(
                "Pass one of the supported export formats. Tabular data "
                "renders as csv, parquet, arrow, json, excel, or markdown; a "
                "schema mapping as schema; a graph or tree structure as graph "
                "or tree; a rendered chart as svg or png."
            ),
            retryable=False,
        )
    )


def export_shape_refusal(detail: str) -> GuardedExecutionError:
    """E13's payload-shape guard: a renderer refused the export source
    because it is the wrong shape for the requested format (a chart
    handed to a tabular format, a DataFrame handed to svg). The
    renderer's own message already names the mismatch; the suggestion
    points at the format-to-shape mapping."""
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.DATA_VALIDATION,
            message=detail,
            suggestion=(
                "Match the format to the source shape: tabular data (records "
                "or a result) to csv/parquet/arrow/json/excel/markdown, a "
                "schema mapping to schema, a graph/tree mapping to graph/tree, "
                "and a rendered chart artifact to svg/png."
            ),
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


def file_over_budget_refusal(detail: str) -> GuardedExecutionError:
    """The over-budget refusal for a whole-file read (`read_file`,
    `profile_data`): there is no SQL predicate to narrow, so the
    suggestion names the file-appropriate recovery (CR-035 — the
    Intention-Driven-Interface guidance must be actionable for the tool
    the agent actually called)."""
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.RESOURCE_ERROR,
            message=f"result refused by the memory-budget gate: {detail}",
            suggestion=(
                "The file's estimated in-memory size exceeds the memory "
                "budget. Supply a smaller or less-compressed file, select "
                "fewer columns upstream, or split it; for a delimited file "
                "(CSV/TSV) the reader streams, so a large row count is fine "
                "but very wide rows are not. The memory ceiling itself is "
                "operator configuration (resources.memory_ceiling_bytes), "
                "not caller-adjustable."
            ),
            retryable=False,
        )
    )
