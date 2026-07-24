"""testbench/batteries/domain/network_build_test.py — CR-027.

Proves the vectorized graph builders (nx.from_pandas_edgelist for the
edge-list analyzer, add_nodes_from for the geospatial node table) produce
graphs identical to the row-wise construction they replace — same nodes,
edges, weights, and node attributes. These exercise the private builders
directly so the geospatial row runs without the geospatial extra
(_network never calls require_geostack; only the public tools do).
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from localdata_mcp.process.domains.geospatial_analysis.network_ops import _network
from localdata_mcp.process.domains.network_graph.network import _build_graph


class TestEdgeListBuilder:
    def test_unweighted_undirected_matches_the_edge_list(self) -> None:
        frame = pd.DataFrame({"s": ["a", "b", "c"], "t": ["b", "c", "a"]})
        graph = _build_graph(frame, "s", "t", None, directed=False)
        assert not graph.is_directed()
        assert set(graph.nodes()) == {"a", "b", "c"}
        assert set(map(frozenset, graph.edges())) == {
            frozenset({"a", "b"}),
            frozenset({"b", "c"}),
            frozenset({"c", "a"}),
        }

    def test_directed_preserves_direction(self) -> None:
        frame = pd.DataFrame({"s": ["a", "b"], "t": ["b", "c"]})
        graph = _build_graph(frame, "s", "t", None, directed=True)
        assert graph.is_directed()
        assert set(graph.edges()) == {("a", "b"), ("b", "c")}

    def test_weight_attribute_is_carried_under_the_weight_key(self) -> None:
        frame = pd.DataFrame({"s": ["a", "b"], "t": ["b", "c"], "w": [2.5, 4.0]})
        graph = _build_graph(frame, "s", "t", "w", directed=False)
        assert graph["a"]["b"]["weight"] == 2.5
        assert graph["b"]["c"]["weight"] == 4.0

    def test_missing_endpoints_are_dropped(self) -> None:
        frame = pd.DataFrame({"s": ["a", None, "c"], "t": ["b", "c", None]})
        graph = _build_graph(frame, "s", "t", None, directed=False)
        assert set(map(frozenset, graph.edges())) == {frozenset({"a", "b"})}

    def test_nan_weight_row_stays_an_unweighted_edge(self) -> None:
        frame = pd.DataFrame({"s": ["a", "b"], "t": ["b", "c"], "w": [2.5, None]})
        graph = _build_graph(frame, "s", "t", "w", directed=False)
        assert graph["a"]["b"]["weight"] == 2.5
        assert "weight" not in graph["b"]["c"]


class TestNodeTableBuilder:
    def test_nodes_carry_their_coordinates(self) -> None:
        frame = pd.DataFrame({"id": ["n1", "n2"], "x": [1.0, 3.0], "y": [2.0, 4.0]})
        graph = _network(frame, [["n1", "n2", 1.5]], "id", "x", "y")
        assert graph.nodes["n1"] == {"x": 1.0, "y": 2.0}
        assert graph.nodes["n2"] == {"x": 3.0, "y": 4.0}
        assert graph["n1"]["n2"]["weight"] == 1.5
