"""testbench/batteries/security/dynamic_bypass_test.py — NFR-103 at the L3 seam.

The single-chokepoint invariant proven at runtime — the object-side
complement to the static import-graph gates (test_persistence_import_graph
and test_nexus_import_graph, which enforce the module side). Every
data-touching tool reaches its backend only through the one process
`runtime.chokepoint()` handle (NX-6, E8.1); there is no second path to the
read/write substrate.

The battery boots the wire (so response shaping and the envelope contract
are live), then yanks *only* the installed chokepoint — the exact
fail-closed condition runtime.py documents ("before that installation
every data-touching tool answers with a structured configuration
refusal"). With the guard removed, every tool across the SQL, file,
key-value, tree, and graph families is driven through the wire and each
answers with the structured not-booted refusal — none slips through to
data. A positive control proves the same surface serves data the moment
the guard is back, so the refusals are the missing chokepoint, not a
blanket-dead wire.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import localdata_mcp.ingest.runtime as runtime

from . import _seam

# The NX-6-crossing tool families, each with surface-valid arguments so the
# call reaches `chokepoint()` (the first thing every one of them does)
# rather than tripping an earlier argument check. Mirrors the ingest /
# persistence / export tool surface the import-graph gates cover statically.
_DATA_TOUCHING: tuple[tuple[str, dict], ...] = (
    ("query", {"endpoint": "x", "sql": "SELECT 1"}),
    ("write_query", {"endpoint": "x", "sql": "INSERT INTO t VALUES (2)"}),
    ("read_file", {"path": "in.csv"}),
    ("query_file", {"path": "f.db", "sql": "SELECT 1"}),
    ("export_result", {"format": "csv", "path": "o.csv", "source": [{"x": 1}]}),
    ("set_value", {"endpoint": "x", "path": "a.b", "key": "k", "value": "1"}),
    ("get_value", {"endpoint": "x", "path": "a.b", "key": "k"}),
    ("delete_key", {"endpoint": "x", "path": "a.b", "key": "k"}),
    ("set_node", {"endpoint": "x", "path": "a.b"}),
    ("get_node", {"endpoint": "x", "path": "a.b"}),
    ("move_node", {"endpoint": "x", "path": "a.b", "new_parent": "a"}),
    ("delete_node", {"endpoint": "x", "path": "a.b"}),
    ("add_edge", {"endpoint": "x", "source": "a", "target": "b", "label": "e"}),
    ("remove_edge", {"endpoint": "x", "source": "a", "target": "b", "label": "e"}),
    ("get_graph_stats", {"endpoint": "x"}),
    ("list_keys", {"endpoint": "x", "path": "a.b"}),
)


@pytest.fixture()
def guard_yanked(tmp_path: Path):
    """Boot the wire, then uninstall the process chokepoint — leaving the
    response-shaping envelope contract intact so the fault surfaces as a
    structured refusal, not a transport crash."""
    with _seam.booted(allowed_paths=(str(tmp_path),)):
        runtime._CHOKEPOINT = None
        yield tmp_path


@pytest.mark.parametrize(
    ("tool", "arguments"), _DATA_TOUCHING, ids=[row[0] for row in _DATA_TOUCHING]
)
def test_data_tool_refuses_without_the_chokepoint(
    guard_yanked, tool: str, arguments: dict
) -> None:
    error = _seam.expect_refused(_seam.call_envelope(tool, arguments))
    # The refusal is the fail-closed not-booted signature — the tool tried
    # to cross the seam and found it absent, rather than reaching data.
    assert "startup" in error["message"], (tool, error["message"])


def test_positive_control_surface_serves_data_with_the_chokepoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "in.csv"
    source.write_text("a,b\nalpha,beta\n", encoding="utf-8")
    with _seam.booted(allowed_paths=(str(tmp_path),)):
        data = _seam.expect_ok(_seam.call_envelope("read_file", {"path": str(source)}))
    assert data["rows"][0][0] == "alpha"
