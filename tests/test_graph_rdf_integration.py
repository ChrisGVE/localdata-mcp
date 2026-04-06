"""Integration tests for graph and RDF storage through DatabaseManager.

Tests the full connect / query / disconnect flow for DOT, GML, GraphML,
Turtle, and N-Triples file types via DatabaseManager methods.
"""

import json
import os

import pytest

from localdata_mcp import DatabaseManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture_path(filename: str) -> str:
    return os.path.join(FIXTURES_DIR, filename)


@pytest.fixture
def db():
    """Fresh DatabaseManager for each test."""
    return DatabaseManager()


# ---------------------------------------------------------------------------
# Graph connection tests (DOT, GML, GraphML)
# ---------------------------------------------------------------------------


class TestGraphConnections:
    def test_connect_dot(self, db):
        result = json.loads(
            db.connect_database("g", "dot", _fixture_path("sample.dot"))
        )
        assert result["success"] is True
        gs = result["graph_summary"]
        assert gs["node_count"] > 0
        assert gs["edge_count"] > 0
        assert isinstance(gs["density"], float)
        assert gs["is_directed"] is True

    def test_connect_gml(self, db):
        result = json.loads(
            db.connect_database("g", "gml", _fixture_path("sample.gml"))
        )
        assert result["success"] is True
        gs = result["graph_summary"]
        assert gs["node_count"] > 0
        assert gs["edge_count"] > 0

    def test_connect_graphml(self, db):
        result = json.loads(
            db.connect_database("g", "graphml", _fixture_path("sample.graphml"))
        )
        assert result["success"] is True
        gs = result["graph_summary"]
        assert gs["node_count"] > 0
        assert gs["edge_count"] > 0

    def test_disconnect_graph(self, db):
        db.connect_database("g", "dot", _fixture_path("sample.dot"))
        assert "g" in db._graph_managers
        result = db.disconnect_database("g")
        assert "Successfully disconnected" in result
        assert "g" not in db._graph_managers
        assert "g" not in db.connections

    def test_list_databases_shows_graph(self, db):
        db.connect_database("g", "dot", _fixture_path("sample.dot"))
        listing = json.loads(db.list_databases())
        entry = listing["databases"][0]
        assert entry["storage"] == "graph"
        assert entry["db_type"] == "dot"


# ---------------------------------------------------------------------------
# RDF connection tests (Turtle, N-Triples)
# ---------------------------------------------------------------------------


class TestRdfConnections:
    def test_connect_turtle(self, db):
        result = json.loads(
            db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        )
        assert result["success"] is True
        rs = result["rdf_summary"]
        assert rs["triple_count"] > 0
        assert rs["subject_count"] > 0
        assert isinstance(rs["namespaces"], list)

    def test_connect_ntriples(self, db):
        result = json.loads(
            db.connect_database("r", "ntriples", _fixture_path("sample.nt"))
        )
        assert result["success"] is True
        rs = result["rdf_summary"]
        assert rs["triple_count"] > 0

    def test_disconnect_rdf(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        assert "r" in db._rdf_managers
        result = db.disconnect_database("r")
        assert "Successfully disconnected" in result
        assert "r" not in db._rdf_managers
        assert "r" not in db.connections

    def test_list_databases_shows_rdf(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        listing = json.loads(db.list_databases())
        entry = listing["databases"][0]
        assert entry["storage"] == "rdf"
        assert entry["db_type"] == "turtle"


# ---------------------------------------------------------------------------
# SPARQL execution tests
# ---------------------------------------------------------------------------


class TestSparqlExecution:
    def test_sparql_select(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        raw = db.execute_query(
            "r",
            "SELECT ?s WHERE { ?s a <http://www.w3.org/2000/01/rdf-schema#Class> }",
        )
        result = json.loads(raw)
        assert result["success"] is True
        assert len(result["results"]) > 0
        assert "s" in result["results"][0]

    def test_sparql_ask(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        raw = db.execute_query(
            "r",
            "ASK { <http://example.org/software#api> ?p ?o }",
        )
        result = json.loads(raw)
        assert result["success"] is True
        assert result["results"] == [{"result": True}]

    def test_sparql_syntax_error(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        raw = db.execute_query("r", "SELECTX ?s WHERE { ?s ?p ?o }")
        result = json.loads(raw)
        assert result["success"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# Duplicate name and connection limit tests
# ---------------------------------------------------------------------------


class TestConnectionConstraints:
    def test_duplicate_name_rejected(self, db):
        db.connect_database("dup", "dot", _fixture_path("sample.dot"))
        result = db.connect_database("dup", "turtle", _fixture_path("sample.ttl"))
        assert "already connected" in result

    def test_connection_limit_enforced(self, db):
        # Fill up 10 connection slots
        for i in range(10):
            db.connect_database(f"c{i}", "dot", _fixture_path("sample.dot"))
        # The 11th should fail
        result = db.connect_database("c10", "dot", _fixture_path("sample.dot"))
        assert "Maximum number" in result
