"""localdata_mcp/ingest/connectors/store_dispatch.py — kind dispatch (E8.3).

The one place a store-family tool learns which table shape a named
endpoint speaks: resolve the declared backend kind through the guard's
`endpoint_summary` seam (never NX-5 — FR-802), refuse a kind the
family does not serve (store_kind_refusal), and hand back the kind for
the tree-vs-graph branch. Both families dispatch through here so an
undeclared name and a kind mismatch produce IDENTICAL structured
guidance everywhere (the NFR-114 one-wording discipline refusals.py
establishes). Also the shared row/dict helpers over the guard's
capability-narrow `Result`. Neighbors: kv/tools.py and
graph_tree/tools.py call `resolve_store_kind`; refusals.py owns the
wording.
"""

from __future__ import annotations

from typing import Any, Mapping

from localdata_mcp.nexus.chokepoint.guard import Result, UnknownEndpointError

from ..refusals import store_kind_refusal, unknown_endpoint_refusal
from ..runtime import chokepoint

# The kinds each store family serves: the kv family reads and writes
# node properties on tree-schema stores (kv/tree) AND graph stores;
# structure tools split per shape.
TREE_KINDS: tuple[str, ...] = ("kv", "tree")
GRAPH_KINDS: tuple[str, ...] = ("graph",)
PROPERTY_KINDS: tuple[str, ...] = TREE_KINDS + GRAPH_KINDS


def resolve_store_kind(endpoint: str, expected_kinds: tuple[str, ...]) -> str:
    """The endpoint's declared kind, refused unless in `expected_kinds`."""
    try:
        summary = chokepoint().endpoint_summary(endpoint)
    except UnknownEndpointError:
        raise unknown_endpoint_refusal(endpoint) from None
    if summary.backend_kind not in expected_kinds:
        raise store_kind_refusal(endpoint, summary.backend_kind, expected_kinds)
    return summary.backend_kind


def one_row(result: Result) -> tuple[Any, ...] | None:
    """The first row of a guard result, or None when empty."""
    return result.rows[0] if result.rows else None


def one_value(result: Result) -> Any:
    """The single scalar a COUNT/EXISTS-style statement returns."""
    return result.rows[0][0]


def rows_as_mappings(result: Result) -> list[Mapping[str, Any]]:
    """Result rows as column-keyed mappings."""
    return [dict(zip(result.columns, row)) for row in result.rows]
