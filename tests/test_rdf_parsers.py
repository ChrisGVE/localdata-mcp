"""Tests for RDF parsers (Turtle and N-Triples)."""

import os

import pytest

from localdata_mcp.rdf_storage import RDFStorageManager
from localdata_mcp.rdf_parsers import parse_ntriples_to_rdf, parse_turtle_to_rdf

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def manager():
    return RDFStorageManager()


class TestParseTurtleToRdf:
    def test_triple_count(self, manager):
        result = parse_turtle_to_rdf(os.path.join(FIXTURES, "sample.ttl"), manager)
        assert result["triple_count"] == 43

    def test_namespaces_present(self, manager):
        result = parse_turtle_to_rdf(os.path.join(FIXTURES, "sample.ttl"), manager)
        assert "namespaces" in result
        ns_uris = [ns["uri"] for ns in result["namespaces"]]
        assert any("software" in uri for uri in ns_uris)

    def test_classes_extracted(self, manager):
        result = parse_turtle_to_rdf(os.path.join(FIXTURES, "sample.ttl"), manager)
        assert "classes" in result
        assert len(result["classes"]) > 0

    def test_properties_extracted(self, manager):
        result = parse_turtle_to_rdf(os.path.join(FIXTURES, "sample.ttl"), manager)
        assert "properties" in result
        assert len(result["properties"]) > 0

    def test_stats_keys(self, manager):
        result = parse_turtle_to_rdf(os.path.join(FIXTURES, "sample.ttl"), manager)
        for key in (
            "triple_count",
            "namespaces",
            "subject_count",
            "predicate_count",
            "object_count",
        ):
            assert key in result

    def test_file_not_found(self, manager):
        with pytest.raises(Exception):
            parse_turtle_to_rdf("/nonexistent/file.ttl", manager)


class TestParseNtriplesToRdf:
    def test_triple_count(self, manager):
        result = parse_ntriples_to_rdf(os.path.join(FIXTURES, "sample.nt"), manager)
        assert result["triple_count"] == 43

    def test_stats_keys(self, manager):
        result = parse_ntriples_to_rdf(os.path.join(FIXTURES, "sample.nt"), manager)
        for key in ("triple_count", "subject_count", "predicate_count", "object_count"):
            assert key in result

    def test_no_classes_key(self, manager):
        """N-Triples parser returns stats only, no ontology metadata."""
        result = parse_ntriples_to_rdf(os.path.join(FIXTURES, "sample.nt"), manager)
        assert "classes" not in result

    def test_file_not_found(self, manager):
        with pytest.raises(Exception):
            parse_ntriples_to_rdf("/nonexistent/file.nt", manager)


class TestRoundTrip:
    def test_turtle_and_ntriples_same_data(self):
        """Both formats should produce the same number of triples."""
        m1 = RDFStorageManager()
        m2 = RDFStorageManager()
        r1 = parse_turtle_to_rdf(os.path.join(FIXTURES, "sample.ttl"), m1)
        r2 = parse_ntriples_to_rdf(os.path.join(FIXTURES, "sample.nt"), m2)
        assert r1["triple_count"] == r2["triple_count"]
