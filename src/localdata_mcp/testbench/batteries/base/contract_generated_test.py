"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.test_stub — DO NOT EDIT.

Parametrized L3 contract stubs (ARCHITECTURE.md 6.1 artifact 4):
every registered tool answers a well-formed fastmcp.Client call, and
the coverage check fails if any registered ToolSpec lacks an entry
here. Regenerate via `python -m localdata_mcp.nexus.contract.generate`;
hand edits fail CI through nexus/contract/check_drift.py.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest
from fastmcp import Client

from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.server.mcp_app import app

GENERATED_TOOL_CALLS: "tuple[tuple[str, dict[str, Any]], ...]" = (
    ("ping", {}),
    ("probe_table", {"rows": 3}),
    ("probe_vector", {"length": 3}),
    ("probe_matrix", {"size": 3}),
    ("probe_model", {"points": 3}),
    ("probe_graph", {"nodes": 3}),
    ("probe_geo", {"points": 3}),
    ("probe_chart", {"points": 3}),
    ("probe_sink", {"text": "probe"}),
    ("list_endpoints", {}),
    ("fetch_chunk", {"stream_id": "probe"}),
    ("close_stream", {"stream_id": "probe"}),
    ("query", {"endpoint": "probe", "sql": "probe"}),
    ("write_query", {"endpoint": "probe", "sql": "probe"}),
    ("read_file", {"path": "probe"}),
    ("query_file", {"path": "probe", "sql": "probe"}),
    ("get_value", {"endpoint": "probe", "path": "probe", "key": "probe"}),
    ("set_value", {"endpoint": "probe", "path": "probe", "key": "probe", "value": "probe"}),
    ("delete_key", {"endpoint": "probe", "path": "probe", "key": "probe"}),
    ("list_keys", {"endpoint": "probe", "path": "probe"}),
    ("get_node", {"endpoint": "probe"}),
    ("set_node", {"endpoint": "probe", "path": "probe"}),
    ("delete_node", {"endpoint": "probe", "path": "probe"}),
    ("get_children", {"endpoint": "probe"}),
    ("move_node", {"endpoint": "probe", "path": "probe"}),
    ("get_neighbors", {"endpoint": "probe", "node_id": "probe"}),
    ("get_edges", {"endpoint": "probe"}),
    ("add_edge", {"endpoint": "probe", "source": "probe", "target": "probe"}),
    ("remove_edge", {"endpoint": "probe", "source": "probe", "target": "probe"}),
    ("find_path", {"endpoint": "probe", "source": "probe", "target": "probe"}),
    ("get_graph_stats", {"endpoint": "probe"}),
    ("describe_database", {"endpoint": "probe"}),
    ("describe_table", {"endpoint": "probe", "table": "probe"}),
    ("find_table", {"endpoint": "probe", "name_pattern": "probe"}),
)

_ENVELOPE_REGIONS = {"inline", "data", "composition_metadata", "error"}


def _envelope_of(result: Any) -> "dict[str, Any]":
    """The wire envelope from a client result: structured content when
    the transport carries it, else the JSON text block."""
    if isinstance(result.structured_content, dict) and (
        set(result.structured_content) >= _ENVELOPE_REGIONS
    ):
        return result.structured_content
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    ("name", "arguments"),
    GENERATED_TOOL_CALLS,
    ids=[name for name, _ in GENERATED_TOOL_CALLS],
)
def test_tool_answers_well_formed(name: str, arguments: "dict[str, Any]") -> None:
    async def session() -> None:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            envelope = _envelope_of(result)
            # FR-403: the four-region schema on every tool, and error
            # exclusive with the other regions.
            assert set(envelope) >= _ENVELOPE_REGIONS
            if envelope["error"] is None:
                assert envelope["inline"] is not None
            else:
                assert envelope["inline"] is None
                assert envelope["data"] is None
                assert envelope["composition_metadata"] is None

    anyio.run(session)


def test_every_registered_spec_has_a_generated_entry() -> None:
    load_spec_modules()
    registered = {spec.name for spec in default_registry()}
    generated = {name for name, _ in GENERATED_TOOL_CALLS}
    missing = registered - generated
    assert not missing, (
        f"registered ToolSpecs lacking generated L3 entries: {sorted(missing)}"
    )
