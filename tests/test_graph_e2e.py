"""End-to-end integration tests for graph, RDF, and SPARQL workflows.

Exercises the full stack through DatabaseManager methods (the same path
as MCP tool calls): connect, query, mutate, export, disconnect.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from localdata_mcp import DatabaseManager

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _fp(filename: str) -> str:
    return os.path.join(FIXTURES_DIR, filename)


def _json(raw: str) -> dict:
    return json.loads(raw)


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager()


# ---------------------------------------------------------------------------
# 1. Full graph lifecycle (DOT, GML, GraphML, Mermaid)
# ---------------------------------------------------------------------------

GRAPH_FORMATS = [
    ("dot", "sample.dot"),
    ("gml", "sample.gml"),
    ("graphml", "sample.graphml"),
    ("mermaid", "sample.mmd"),
]


class TestGraphWorkflow:
    @pytest.mark.parametrize("fmt,filename", GRAPH_FORMATS)
    def test_full_lifecycle(self, db: DatabaseManager, fmt: str, filename: str) -> None:
        result = _json(db.connect_database("g", fmt, _fp(filename)))
        assert result["success"] is True
        gs = result["graph_summary"]
        assert gs["node_count"] > 0 and gs["edge_count"] > 0

        summary = db.get_node("g")
        assert summary["node_count"] > 0
        nodes = summary.get("sample_nodes") or summary.get("first_node_ids", [])
        node_id = nodes[0] if nodes else "api"

        detail = db.get_node("g", node_id)
        assert "node_id" in detail or "error" not in detail

        neighbors = db.get_neighbors("g", node_id)
        assert "neighbors" in neighbors or "successors" in neighbors

        edges = db.get_edges("g")
        assert edges["total"] > 0
        assert "edges" in db.get_edges("g", node_id=node_id)

        add_res = db.add_edge("g", node_id, node_id, label="self_loop", weight=0.5)
        assert add_res.get("source") == node_id or "edge" in str(add_res).lower()

        db.set_value("g", node_id, "test_prop", "hello")
        assert db.get_value("g", node_id, "test_prop")["value"] == "hello"
        key_names = [k["key"] for k in db.list_keys("g", node_id)["keys"]]
        assert "test_prop" in key_names

        if edges["edges"]:
            src, tgt = edges["edges"][0]["source"], edges["edges"][0]["target"]
            path = db.find_path("g", src, tgt)
            assert "path" in path or "paths" in path

        stats = db.get_graph_stats("g")
        assert stats["node_count"] > 0 and "density" in stats

        for exp_fmt in ("dot", "gml", "graphml"):
            exported = db.export_graph("g", exp_fmt)
            assert exported.get("content")

        count_before = db.get_graph_stats("g")["node_count"]
        assert db.delete_node("g", node_id)["nodes_deleted"] >= 1
        assert db.get_graph_stats("g")["node_count"] < count_before

        assert "Successfully disconnected" in db.disconnect_database("g")


# ---------------------------------------------------------------------------
# 1b. Mermaid-specific features
# ---------------------------------------------------------------------------


class TestMermaidSpecific:
    """Tests for Mermaid-specific features preserved through the parser."""

    def test_node_shapes_stored_as_properties(self, db: DatabaseManager) -> None:
        """Node shapes ([], (), {}, etc.) should be stored as node properties."""
        result = _json(db.connect_database("m", "mermaid", _fp("sample.mmd")))
        assert result["success"] is True

        # The sample.mmd has nodes with different shapes:
        # A[API Gateway] -> rectangle, B[Auth Service] -> rectangle,
        # C(Cache Layer) -> rounded, D[(Database)] -> cylindrical,
        # E{Decision} -> rhombus
        stats = db.get_graph_stats("m")
        assert stats["node_count"] >= 5

        # Check that shape properties exist on at least one node
        keys_a = db.list_keys("m", "A")
        key_names = [k["key"] for k in keys_a.get("keys", [])]
        assert (
            "shape" in key_names
        ), "Node A should have a 'shape' property from Mermaid bracket syntax"

        db.disconnect_database("m")

    def test_edge_styles_stored_as_properties(self, db: DatabaseManager) -> None:
        """Edge styles (-->, -.-, ==> etc.) should be stored as edge properties."""
        result = _json(db.connect_database("m", "mermaid", _fp("sample.mmd")))
        assert result["success"] is True

        edges = db.get_edges("m")
        assert edges["total"] > 0

        # Check that at least one edge has style-related properties or labels
        # The sample has labeled edges like E -->|yes| F
        found_label = False
        for edge in edges["edges"]:
            if edge.get("label"):
                found_label = True
                break
        # At least the |yes| and |no| edges should have labels
        assert (
            found_label
        ), "Expected at least one edge with a label from Mermaid syntax"

        db.disconnect_database("m")

    def test_subgraph_membership_stored_as_property(self, db: DatabaseManager) -> None:
        """Nodes inside a subgraph should have a subgraph property."""
        result = _json(db.connect_database("m", "mermaid", _fp("sample.mmd")))
        assert result["success"] is True

        # B is inside 'subgraph Backend'
        keys_b = db.list_keys("m", "B")
        key_names = [k["key"] for k in keys_b.get("keys", [])]
        assert (
            "subgraph" in key_names
        ), "Node B should have a 'subgraph' property indicating membership"

        val = db.get_value("m", "B", "subgraph")
        assert val["value"] is not None

        db.disconnect_database("m")

    def test_chained_edges_create_multiple_edges(self, db: DatabaseManager) -> None:
        """Chained edges like A-->B-->C should produce two separate edges."""
        # The sample.mmd has: E -->|yes| F and E -->|no| G
        # which are separate edges, but also A --> B and A --> C
        result = _json(db.connect_database("m", "mermaid", _fp("sample.mmd")))
        assert result["success"] is True

        edges = db.get_edges("m")
        # Count edges originating from A (should have at least 2: A->B, A->C)
        a_edges = [e for e in edges["edges"] if e["source"] == "A"]
        assert (
            len(a_edges) >= 2
        ), f"Node A should have at least 2 outgoing edges, got {len(a_edges)}"

        db.disconnect_database("m")


# ---------------------------------------------------------------------------
# 2. Full RDF lifecycle (Turtle / N-Triples)
# ---------------------------------------------------------------------------


class TestRdfWorkflow:
    def test_turtle_lifecycle(self, db: DatabaseManager) -> None:
        result = _json(db.connect_database("r", "turtle", _fp("sample.ttl")))
        assert result["success"] is True
        rs = result["rdf_summary"]
        assert rs["triple_count"] > 0 and isinstance(rs["namespaces"], list)

        raw = db.execute_query(
            "r",
            "SELECT ?s ?label WHERE "
            "{ ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label }",
        )
        sel = _json(raw)
        assert sel["success"] is True and len(sel["results"]) > 0
        assert "label" in sel["results"][0]

        ask = _json(
            db.execute_query("r", "ASK { <http://example.org/software#api> ?p ?o }")
        )
        assert ask["success"] is True
        assert ask["results"] == [{"result": True}]

        assert db.get_node("r")["triple_count"] > 0

        detail = db.get_node("r", "http://example.org/software#api")
        assert len(detail["predicates"]) > 0

        turtle_out = db.export_structured("r", "turtle")
        assert "example.org" in turtle_out["content"]
        nt_out = db.export_structured("r", "ntriples")
        assert len(nt_out["content"]) > 0

        desc = _json(db.describe_database("r"))
        assert desc["storage"] == "rdf" and desc["triple_count"] > 0

        assert "Successfully disconnected" in db.disconnect_database("r")

    def test_ntriples_connect_and_query(self, db: DatabaseManager) -> None:
        result = _json(db.connect_database("nt", "ntriples", _fp("sample.nt")))
        assert result["success"] is True and result["rdf_summary"]["triple_count"] > 0

        sel = _json(db.execute_query("nt", "SELECT ?s WHERE { ?s ?p ?o } LIMIT 5"))
        assert sel["success"] is True and len(sel["results"]) <= 5
        db.disconnect_database("nt")


# ---------------------------------------------------------------------------
# 3. SPARQL endpoint workflow (mocked)
# ---------------------------------------------------------------------------

_SELECT_BINDINGS = [
    {"s": {"type": "uri", "value": "http://example.org/A"}},
    {"s": {"type": "uri", "value": "http://example.org/B"}},
]
_COUNT_BINDINGS = [
    {
        "count": {
            "type": "literal",
            "value": "42",
            "datatype": "http://www.w3.org/2001/XMLSchema#integer",
        }
    }
]


class TestSPARQLEndpointWorkflow:
    @pytest.fixture
    def mock_sparql(self):
        """Patch SPARQLWrapper so no network calls are made."""
        with patch("localdata_mcp.sparql_endpoint.SPARQLWrapper") as MockSW:
            inst = MagicMock()
            MockSW.return_value = inst

            def fake_query():
                res = MagicMock()
                q = inst.setQuery.call_args[0][0].strip().upper()
                if q.startswith("ASK"):
                    res.convert.return_value = {"boolean": True}
                elif "COUNT" in q:
                    res.convert.return_value = {
                        "results": {"bindings": _COUNT_BINDINGS}
                    }
                else:
                    res.convert.return_value = {
                        "results": {"bindings": _SELECT_BINDINGS}
                    }
                return res

            inst.query = fake_query
            yield MockSW

    def test_endpoint_lifecycle(self, db: DatabaseManager, mock_sparql) -> None:
        result = _json(db.connect_database("ep", "sparql", "http://example.org/sparql"))
        assert result["success"] is True
        assert result["connection_info"]["storage"] == "sparql_endpoint"

        sel = _json(db.execute_query("ep", "SELECT ?s WHERE { ?s ?p ?o } LIMIT 2"))
        assert sel["count"] == 2 and sel["results"][0]["s"] == "http://example.org/A"

        ask = _json(db.execute_query("ep", "ASK { <http://example.org/A> ?p ?o }"))
        assert ask["results"] == [{"result": True}]

        desc = _json(db.describe_database("ep"))
        assert desc["storage"] == "sparql_endpoint"
        assert desc["endpoint_url"] == "http://example.org/sparql"

        names = [d["name"] for d in _json(db.list_databases())["databases"]]
        assert "ep" in names

        assert "Successfully disconnected" in db.disconnect_database("ep")
        names = [d["name"] for d in _json(db.list_databases())["databases"]]
        assert "ep" not in names


# ---------------------------------------------------------------------------
# 4. Cross-format workflow
# ---------------------------------------------------------------------------


class TestCrossFormatWorkflow:
    def test_graph_and_rdf_simultaneously(self, db: DatabaseManager) -> None:
        g_res = _json(db.connect_database("graph", "dot", _fp("sample.dot")))
        r_res = _json(db.connect_database("rdf", "turtle", _fp("sample.ttl")))
        assert g_res["success"] is True and r_res["success"] is True

        listing = _json(db.list_databases())
        storages = {d["name"]: d["storage"] for d in listing["databases"]}
        assert storages["graph"] == "graph" and storages["rdf"] == "rdf"

        assert db.get_graph_stats("graph")["node_count"] > 0
        sel = _json(db.execute_query("rdf", "SELECT ?s WHERE { ?s ?p ?o } LIMIT 3"))
        assert sel["success"] is True

        db.disconnect_database("graph")
        db.disconnect_database("rdf")
        assert _json(db.list_databases())["databases"] == []


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_connect_nonexistent_file(self, db: DatabaseManager) -> None:
        result = db.connect_database("bad", "dot", "/nonexistent/file.dot")
        assert "failed" in result.lower() or "error" in result.lower()

    def test_query_disconnected_name(self, db: DatabaseManager) -> None:
        with pytest.raises((ValueError, KeyError)):
            db.get_node("nonexistent")

    def test_invalid_sparql_syntax(self, db: DatabaseManager) -> None:
        db.connect_database("r", "turtle", _fp("sample.ttl"))
        result = _json(db.execute_query("r", "SELECTX ?s WHERE { ?s ?p ?o }"))
        assert result["success"] is False and "error" in result

    def test_graph_tools_on_rdf_connection(self, db: DatabaseManager) -> None:
        db.connect_database("r", "turtle", _fp("sample.ttl"))
        with pytest.raises((ValueError, KeyError)):
            db.get_neighbors("r", "http://example.org/software#api")

    def test_export_structured_invalid_rdf_format(self, db: DatabaseManager) -> None:
        db.connect_database("r", "turtle", _fp("sample.ttl"))
        assert "error" in db.export_structured("r", "graphml")
