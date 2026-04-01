"""Tests for graph_parsers module (DOT, GML, GraphML → GraphStorageManager)."""

import os

import networkx as nx
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

from localdata_mcp.graph_parsers import (
    _networkx_to_storage,
    parse_dot_to_graph,
    parse_gml_to_graph,
    parse_graphml_to_graph,
)
from localdata_mcp.graph_manager import GraphStorageManager

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
DOT_PATH = os.path.join(FIXTURES_DIR, "sample.dot")
GML_PATH = os.path.join(FIXTURES_DIR, "sample.gml")
GRAPHML_PATH = os.path.join(FIXTURES_DIR, "sample.graphml")


def make_manager() -> GraphStorageManager:
    """Create an in-memory GraphStorageManager for testing."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return GraphStorageManager(engine)


# -- Expected topology --------------------------------------------------------

EXPECTED_NODE_IDS = {"api", "auth", "cache", "config", "database", "logger"}
EXPECTED_EDGES = {
    ("api", "auth"),
    ("api", "cache"),
    ("api", "logger"),
    ("auth", "database"),
    ("auth", "logger"),
    ("cache", "database"),
    ("config", "api"),
}


def _get_stored_node_ids(manager: GraphStorageManager) -> set:
    nodes = manager.list_nodes(limit=100)
    return {n.node_id for n in nodes}


def _get_stored_edges(manager: GraphStorageManager) -> set:
    edges = manager.list_edges(limit=100)
    return {(e.source_id, e.target_id) for e in edges}


# -- DOT tests ---------------------------------------------------------------


class TestParseDot:
    def test_node_count(self):
        mgr = make_manager()
        stats = parse_dot_to_graph(DOT_PATH, mgr)
        assert stats["node_count"] == 6

    def test_edge_count(self):
        mgr = make_manager()
        stats = parse_dot_to_graph(DOT_PATH, mgr)
        assert stats["edge_count"] == 7

    def test_topology(self):
        mgr = make_manager()
        parse_dot_to_graph(DOT_PATH, mgr)
        assert _get_stored_node_ids(mgr) == EXPECTED_NODE_IDS
        assert _get_stored_edges(mgr) == EXPECTED_EDGES

    def test_file_not_found(self):
        mgr = make_manager()
        with pytest.raises(ValueError, match="File not found"):
            parse_dot_to_graph("/nonexistent/file.dot", mgr)


# -- GML tests ---------------------------------------------------------------


class TestParseGml:
    def test_node_count(self):
        mgr = make_manager()
        stats = parse_gml_to_graph(GML_PATH, mgr)
        assert stats["node_count"] == 6

    def test_edge_count(self):
        mgr = make_manager()
        stats = parse_gml_to_graph(GML_PATH, mgr)
        assert stats["edge_count"] == 7

    def test_topology(self):
        mgr = make_manager()
        parse_gml_to_graph(GML_PATH, mgr)
        assert _get_stored_node_ids(mgr) == EXPECTED_NODE_IDS
        assert _get_stored_edges(mgr) == EXPECTED_EDGES

    def test_file_not_found(self):
        mgr = make_manager()
        with pytest.raises(ValueError, match="File not found"):
            parse_gml_to_graph("/nonexistent/file.gml", mgr)


# -- GraphML tests ------------------------------------------------------------


class TestParseGraphml:
    def test_node_count(self):
        mgr = make_manager()
        stats = parse_graphml_to_graph(GRAPHML_PATH, mgr)
        assert stats["node_count"] == 6

    def test_edge_count(self):
        mgr = make_manager()
        stats = parse_graphml_to_graph(GRAPHML_PATH, mgr)
        assert stats["edge_count"] == 7

    def test_topology(self):
        mgr = make_manager()
        parse_graphml_to_graph(GRAPHML_PATH, mgr)
        assert _get_stored_node_ids(mgr) == EXPECTED_NODE_IDS
        assert _get_stored_edges(mgr) == EXPECTED_EDGES

    def test_file_not_found(self):
        mgr = make_manager()
        with pytest.raises(ValueError, match="File not found"):
            parse_graphml_to_graph("/nonexistent/file.graphml", mgr)


# -- Cross-format topology consistency ---------------------------------------


class TestTopologyConsistency:
    """All three fixtures represent the same graph; verify identical topology."""

    def test_same_node_ids(self):
        ids = []
        for parser, path in [
            (parse_dot_to_graph, DOT_PATH),
            (parse_gml_to_graph, GML_PATH),
            (parse_graphml_to_graph, GRAPHML_PATH),
        ]:
            mgr = make_manager()
            parser(path, mgr)
            ids.append(_get_stored_node_ids(mgr))
        assert ids[0] == ids[1] == ids[2]

    def test_same_edges(self):
        edge_sets = []
        for parser, path in [
            (parse_dot_to_graph, DOT_PATH),
            (parse_gml_to_graph, GML_PATH),
            (parse_graphml_to_graph, GRAPHML_PATH),
        ]:
            mgr = make_manager()
            parser(path, mgr)
            edge_sets.append(_get_stored_edges(mgr))
        assert edge_sets[0] == edge_sets[1] == edge_sets[2]


# -- Property preservation ---------------------------------------------------


class TestNodeProperties:
    """Verify node attributes are stored as properties."""

    @pytest.fixture(params=["dot", "gml", "graphml"])
    def loaded_manager(self, request):
        mgr = make_manager()
        parsers = {
            "dot": (parse_dot_to_graph, DOT_PATH),
            "gml": (parse_gml_to_graph, GML_PATH),
            "graphml": (parse_graphml_to_graph, GRAPHML_PATH),
        }
        parser, path = parsers[request.param]
        parser(path, mgr)
        return mgr

    def test_node_label_stored(self, loaded_manager):
        node = loaded_manager.get_node("api")
        assert node is not None
        assert node.label == "API Gateway"

    def test_node_color_property(self, loaded_manager):
        prop = loaded_manager.get_property("node", "api", "color")
        assert prop is not None
        assert prop.value == "blue"

    def test_node_type_property(self, loaded_manager):
        prop = loaded_manager.get_property("node", "database", "type")
        assert prop is not None
        assert prop.value == "infrastructure"


class TestEdgeProperties:
    """Verify edge weight is stored as the weight field."""

    @pytest.fixture(params=["dot", "gml", "graphml"])
    def loaded_manager(self, request):
        mgr = make_manager()
        parsers = {
            "dot": (parse_dot_to_graph, DOT_PATH),
            "gml": (parse_gml_to_graph, GML_PATH),
            "graphml": (parse_graphml_to_graph, GRAPHML_PATH),
        }
        parser, path = parsers[request.param]
        parser(path, mgr)
        return mgr

    def test_edge_weight(self, loaded_manager):
        edge = loaded_manager.get_edge("api", "auth", label="authenticates")
        assert edge is not None
        assert edge.weight == pytest.approx(1.0)

    def test_edge_label(self, loaded_manager):
        edge = loaded_manager.get_edge("cache", "database", label="syncs")
        assert edge is not None
        assert edge.label == "syncs"

    def test_edge_without_label(self, loaded_manager):
        edge = loaded_manager.get_edge("api", "logger", label=None)
        assert edge is not None
        assert edge.weight == pytest.approx(0.5)


# -- _networkx_to_storage with programmatic graph ----------------------------


class TestNetworkxToStorage:
    def test_programmatic_digraph(self):
        G = nx.DiGraph()
        G.add_node("a", label="Node A", color="red")
        G.add_node("b", label="Node B")
        G.add_edge("a", "b", label="connects", weight=2.5, style="dashed")

        mgr = make_manager()
        stats = _networkx_to_storage(G, mgr)

        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1

        node_a = mgr.get_node("a")
        assert node_a is not None
        assert node_a.label == "Node A"

        color_prop = mgr.get_property("node", "a", "color")
        assert color_prop is not None
        assert color_prop.value == "red"

        edge = mgr.get_edge("a", "b", label="connects")
        assert edge is not None
        assert edge.weight == pytest.approx(2.5)

        # Extra edge attribute stored as property
        style_prop = mgr.get_property("edge", str(edge.id), "style")
        assert style_prop is not None
        assert style_prop.value == "dashed"

    def test_empty_graph(self):
        G = nx.DiGraph()
        mgr = make_manager()
        stats = _networkx_to_storage(G, mgr)
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0


# -- Round-trip stats ---------------------------------------------------------


class TestRoundTripStats:
    @pytest.mark.parametrize(
        "parser,path",
        [
            (parse_dot_to_graph, DOT_PATH),
            (parse_gml_to_graph, GML_PATH),
            (parse_graphml_to_graph, GRAPHML_PATH),
        ],
    )
    def test_stats_match_after_parse(self, parser, path):
        mgr = make_manager()
        stats = parser(path, mgr)
        assert stats["node_count"] == 6
        assert stats["edge_count"] == 7
        assert stats["is_directed"] is True
        assert stats["density"] > 0
        # Re-query must match (excluding validation warnings which are
        # added by _store_and_validate but not returned by get_graph_stats)
        base_stats = {k: v for k, v in stats.items() if k != "warnings"}
        assert mgr.get_graph_stats() == base_stats
