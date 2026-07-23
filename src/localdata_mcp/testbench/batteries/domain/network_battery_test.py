"""testbench/batteries/domain/network_battery_test.py — E10.i slice.

The NFR-502c domain battery's network rows: FR-301 L3 coverage and
the FR-304/NFR-505 dual-assertion oracle. Published fixture:
Zachary's karate club (via networkx's bundled copy — the canonical
published social network): 34 nodes, 78 edges, and node 0 (the
instructor) / node 33 (the president) as the highest-degree actors —
facts from the 1977 study itself. Reference legs recompute every
reported measure with networkx directly on the same edge list.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import anyio
import networkx as nx
import pandas as pd
import pytest
from fastmcp import Client

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Path]:
    config = ConfigModel(
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
    )
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    yield tmp_path
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def session() -> dict[str, Any]:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            if isinstance(result.structured_content, dict) and (
                "inline" in result.structured_content
            ):
                return result.structured_content
            payload = json.loads(result.content[0].text)
            assert isinstance(payload, dict)
            return payload

    return anyio.run(session)


def _data(envelope: dict[str, Any]) -> Any:
    assert envelope["error"] is None, envelope["error"]
    return envelope["data"]


def _karate_csv(tmp_path: Path) -> tuple[str, "nx.Graph[Any]"]:
    graph = nx.karate_club_graph()
    target = tmp_path / "karate.csv"
    pd.DataFrame([{"a": str(u), "b": str(v)} for u, v in graph.edges()]).to_csv(
        target, index=False
    )
    return str(target), graph


def test_karate_club_matches_reference_and_published(bench: Path) -> None:
    """analyze_network — both oracle legs on the canonical network."""
    path, graph = _karate_csv(bench)
    data = _data(
        _call(
            "analyze_network",
            {"path": path, "source_column": "a", "target_column": "b"},
        )
    )
    # Published: 34 members, 78 ties, one connected component.
    assert data["n_nodes"] == 34
    assert data["n_edges"] == 78
    assert data["is_connected"] is True
    assert data["n_components"] == 1
    config = ConfigModel()
    rtol = config.testbench.tol_closed_form_rtol
    # Reference legs: networkx recomputation on the same edges.
    assert data["density"] == pytest.approx(nx.density(graph), rel=rtol)
    assert data["average_clustering"] == pytest.approx(
        nx.average_clustering(nx.Graph((str(u), str(v)) for u, v in graph.edges())),
        rel=rtol,
    )
    # Published: the instructor (0) and the president (33) dominate.
    top_degree_nodes = {entry["node"] for entry in data["centrality"]["degree"][:2]}
    assert top_degree_nodes == {"0", "33"}


def test_directed_components_are_reported(bench: Path) -> None:
    """analyze_network directed — two weak components, no centrality."""
    target = bench / "two_islands.csv"
    pd.DataFrame({"src": ["a", "b", "x"], "dst": ["b", "c", "y"]}).to_csv(
        target, index=False
    )
    data = _data(
        _call(
            "analyze_network",
            {
                "path": str(target),
                "source_column": "src",
                "target_column": "dst",
                "directed": True,
                "include_centrality": False,
            },
        )
    )
    assert data["directed"] is True
    assert data["is_connected"] is False
    assert data["n_components"] == 2
    assert "centrality" not in data
    assert "average_clustering" not in data
