"""Tests for graph/RDF polish features (tasks 107-112).

Covers RDF export, describe_database for graph/RDF, check_compatibility
library detection, get_node dispatch to RDF, and memory bounds reporting
for graph/RDF connections.
"""

import json
import os

import pytest

from localdata_mcp import DatabaseManager

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture_path(filename: str) -> str:
    return os.path.join(FIXTURES_DIR, filename)


@pytest.fixture
def db():
    """Fresh DatabaseManager for each test."""
    return DatabaseManager()


# ---------------------------------------------------------------------------
# Task 107: RDF export to Turtle and N-Triples
# ---------------------------------------------------------------------------


class TestRdfExport:
    def test_export_turtle(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        result = db.export_structured("r", "turtle")
        assert "error" not in result
        assert result["format"] == "turtle"
        assert "sw:api" in result["content"] or "example.org" in result["content"]

    def test_export_turtle_via_ttl_alias(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        result = db.export_structured("r", "ttl")
        assert "error" not in result
        assert result["format"] == "turtle"

    def test_export_ntriples(self, db):
        db.connect_database("r", "ntriples", _fixture_path("sample.nt"))
        result = db.export_structured("r", "ntriples")
        assert "error" not in result
        assert result["format"] == "nt"
        assert "example.org" in result["content"]

    def test_export_ntriples_via_nt_alias(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        result = db.export_structured("r", "nt")
        assert "error" not in result
        assert result["format"] == "nt"

    def test_export_rdf_unsupported_format(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        result = db.export_structured("r", "csv")
        assert "error" in result
        assert "Unsupported RDF format" in result["error"]


# ---------------------------------------------------------------------------
# Task 108: RDF navigation bridge via get_node
# ---------------------------------------------------------------------------


class TestRdfGetNode:
    def test_get_node_rdf_summary(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        result = db.get_node("r")
        assert "triple_count" in result
        assert result["triple_count"] > 0
        assert "namespaces" in result
        assert "subject_count" in result

    def test_get_node_rdf_subject_detail(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        result = db.get_node("r", path="http://example.org/software#api")
        assert "subject" in result
        assert result["subject"] == "http://example.org/software#api"
        assert "predicates" in result
        # The api subject should have predicates like rdf:type, rdfs:label, etc.
        assert len(result["predicates"]) > 0

    def test_get_node_rdf_subject_no_data(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        result = db.get_node("r", path="http://example.org/nonexistent")
        assert result["subject"] == "http://example.org/nonexistent"
        assert result["predicates"] == {}


# ---------------------------------------------------------------------------
# Task 109: describe_database for graph and RDF
# ---------------------------------------------------------------------------


class TestDescribeDatabase:
    def test_describe_graph(self, db):
        db.connect_database("g", "dot", _fixture_path("sample.dot"))
        raw = db.describe_database("g")
        info = json.loads(raw)
        assert info["storage"] == "graph"
        assert info["node_count"] > 0
        assert info["edge_count"] > 0
        assert "density" in info
        assert "is_directed" in info
        assert "first_node_ids" in info

    def test_describe_rdf(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        raw = db.describe_database("r")
        info = json.loads(raw)
        assert info["storage"] == "rdf"
        assert info["triple_count"] > 0
        assert "namespaces" in info
        assert info["subject_count"] > 0
        assert info["predicate_count"] > 0
        assert info["object_count"] > 0


# ---------------------------------------------------------------------------
# Task 110: check_compatibility includes library checks
# ---------------------------------------------------------------------------


class TestCheckCompatibility:
    def test_libraries_in_report(self, db):
        raw = db.check_compatibility()
        report = json.loads(raw)
        assert "libraries" in report
        libs = report["libraries"]
        for lib_name in ("networkx", "rdflib"):
            assert lib_name in libs
            assert libs[lib_name]["available"] is True
            assert "version" in libs[lib_name]
        # SPARQLWrapper and pydot may or may not be installed
        assert "SPARQLWrapper" in libs
        assert "pydot" in libs


# ---------------------------------------------------------------------------
# Task 112: Memory bounds for graph and RDF
# ---------------------------------------------------------------------------


class TestMemoryBounds:
    def test_memory_bounds_includes_graph(self, db):
        db.connect_database("g", "dot", _fixture_path("sample.dot"))
        raw = db.manage_memory_bounds()
        result = json.loads(raw)
        info = result["database_manager_info"]
        assert "graph_connections" in info
        assert "g" in info["graph_connections"]
        gc = info["graph_connections"]["g"]
        assert "node_count" in gc
        assert "edge_count" in gc
        assert "estimated_bytes" in gc
        assert gc["estimated_bytes"] > 0

    def test_memory_bounds_includes_rdf(self, db):
        db.connect_database("r", "turtle", _fixture_path("sample.ttl"))
        raw = db.manage_memory_bounds()
        result = json.loads(raw)
        info = result["database_manager_info"]
        assert "rdf_connections" in info
        assert "r" in info["rdf_connections"]
        rc = info["rdf_connections"]["r"]
        assert "triple_count" in rc
        assert "estimated_bytes" in rc
        assert rc["estimated_bytes"] > 0

    def test_memory_bounds_no_graph_when_none(self, db):
        raw = db.manage_memory_bounds()
        result = json.loads(raw)
        info = result["database_manager_info"]
        assert "graph_connections" not in info
        assert "rdf_connections" not in info
