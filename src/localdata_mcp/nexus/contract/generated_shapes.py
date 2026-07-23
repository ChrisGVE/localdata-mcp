"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.typeshape_registry — DO NOT EDIT.

Per-tool type-shape declarations (ARCHITECTURE.md 6.1 artifact 5).
Regenerate via `python -m localdata_mcp.nexus.contract.generate`;
hand edits fail CI through nexus/contract/check_drift.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from localdata_mcp.nexus.contract.spec import TypeShape


@dataclass(frozen=True)
class ShapeEntry:
    """One tool's declared composition facts."""

    input_shape: TypeShape
    output_shape: TypeShape
    streaming_capable: bool
    domain: "str | None"


TOOL_SHAPES: Final[Mapping[str, ShapeEntry]] = {
    "ping": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain=None,
    ),
    "probe_table": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain=None,
    ),
    "probe_vector": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.VECTOR,
        streaming_capable=False,
        domain=None,
    ),
    "probe_matrix": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.MATRIX,
        streaming_capable=False,
        domain=None,
    ),
    "probe_model": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.FITTED_MODEL,
        streaming_capable=False,
        domain=None,
    ),
    "probe_graph": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.GRAPH,
        streaming_capable=False,
        domain=None,
    ),
    "probe_geo": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.GEO,
        streaming_capable=False,
        domain=None,
    ),
    "probe_chart": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.CHART_SPEC,
        streaming_capable=False,
        domain=None,
    ),
    "probe_sink": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.NONE,
        streaming_capable=False,
        domain=None,
    ),
    "list_endpoints": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "fetch_chunk": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=True,
        domain='ingest',
    ),
    "close_stream": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "query": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=True,
        domain='ingest',
    ),
    "write_query": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "read_file": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=True,
        domain='ingest',
    ),
    "query_file": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=True,
        domain='ingest',
    ),
    "get_value": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "set_value": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "delete_key": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "list_keys": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "get_node": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "set_node": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "delete_node": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "get_children": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "move_node": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "get_neighbors": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "get_edges": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "add_edge": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "remove_edge": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "find_path": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "get_graph_stats": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='ingest',
    ),
    "describe_database": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "describe_table": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "find_table": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "profile_data": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "search_data": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "map_categories": ShapeEntry(
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='explore',
    ),
    "analyze_hypothesis_test": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='process',
    ),
    "analyze_anova": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='process',
    ),
    "analyze_effect_sizes": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='process',
    ),
    "analyze_ab_test": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='process',
    ),
    "analyze_regression": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.FITTED_MODEL,
        streaming_capable=False,
        domain='process',
    ),
    "evaluate_model_performance": ShapeEntry(
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain='process',
    ),
}
