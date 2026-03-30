"""Integration tests for RDF file formats (Turtle, N-Triples) with SPARQL queries."""

import pytest

from .mcp_test_client import call_tool

pytestmark = [pytest.mark.integration]

SAMPLE_TURTLE = """\
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

ex:Alice rdf:type ex:Person ;
    ex:name "Alice" ;
    ex:age "30" ;
    ex:knows ex:Bob .

ex:Bob rdf:type ex:Person ;
    ex:name "Bob" ;
    ex:age "25" .
"""

SAMPLE_NTRIPLES = """\
<http://example.org/Alice> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> .
<http://example.org/Alice> <http://example.org/name> "Alice" .
<http://example.org/Bob> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> .
<http://example.org/Bob> <http://example.org/name> "Bob" .
"""


class TestTurtleParsing:
    def test_connect_turtle(self, tmp_path):
        path = str(tmp_path / "test.ttl")
        with open(path, "w") as f:
            f.write(SAMPLE_TURTLE)
        result = call_tool(
            "connect_database",
            {"name": "ttl_connect", "db_type": "turtle", "conn_string": path},
        )
        try:
            assert result["success"] is True
            rdf_summary = result["rdf_summary"]
            # Turtle file has 7 triples (Alice: type, name, age, knows; Bob: type, name, age)
            assert rdf_summary["triple_count"] == 7
            assert rdf_summary["subject_count"] == 2
        finally:
            call_tool("disconnect_database", {"name": "ttl_connect"})


class TestNTriplesParsing:
    def test_connect_ntriples(self, tmp_path):
        path = str(tmp_path / "test.nt")
        with open(path, "w") as f:
            f.write(SAMPLE_NTRIPLES)
        result = call_tool(
            "connect_database",
            {"name": "nt_connect", "db_type": "ntriples", "conn_string": path},
        )
        try:
            assert result["success"] is True
            rdf_summary = result["rdf_summary"]
            assert rdf_summary["triple_count"] == 4
            assert rdf_summary["subject_count"] == 2
        finally:
            call_tool("disconnect_database", {"name": "nt_connect"})


class TestSPARQLSelect:
    def test_select_names(self, tmp_path):
        path = str(tmp_path / "select.ttl")
        with open(path, "w") as f:
            f.write(SAMPLE_TURTLE)
        call_tool(
            "connect_database",
            {"name": "ttl_select", "db_type": "turtle", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "ttl_select",
                    "query": (
                        "PREFIX ex: <http://example.org/> "
                        "SELECT ?name WHERE { ?s ex:name ?name }"
                    ),
                },
            )
            assert result["success"] is True
            names = [row["name"] for row in result["results"]]
            assert "Alice" in names
            assert "Bob" in names
        finally:
            call_tool("disconnect_database", {"name": "ttl_select"})

    def test_select_with_filter(self, tmp_path):
        path = str(tmp_path / "filter.ttl")
        with open(path, "w") as f:
            f.write(SAMPLE_TURTLE)
        call_tool(
            "connect_database",
            {"name": "ttl_filter", "db_type": "turtle", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "ttl_filter",
                    "query": (
                        "PREFIX ex: <http://example.org/> "
                        "SELECT ?s ?name WHERE { "
                        "  ?s ex:name ?name . "
                        "  ?s ex:knows ?other "
                        "}"
                    ),
                },
            )
            assert result["success"] is True
            names = [row["name"] for row in result["results"]]
            assert "Alice" in names
            assert "Bob" not in names
        finally:
            call_tool("disconnect_database", {"name": "ttl_filter"})


class TestSPARQLAsk:
    def test_ask_existing_triple(self, tmp_path):
        path = str(tmp_path / "ask.ttl")
        with open(path, "w") as f:
            f.write(SAMPLE_TURTLE)
        call_tool(
            "connect_database",
            {"name": "ttl_ask", "db_type": "turtle", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "ttl_ask",
                    "query": (
                        "PREFIX ex: <http://example.org/> "
                        "ASK { ex:Alice ex:knows ex:Bob }"
                    ),
                },
            )
            assert result["success"] is True
            assert result["results"][0]["result"] is True
        finally:
            call_tool("disconnect_database", {"name": "ttl_ask"})

    def test_ask_nonexistent_triple(self, tmp_path):
        path = str(tmp_path / "ask_no.ttl")
        with open(path, "w") as f:
            f.write(SAMPLE_TURTLE)
        call_tool(
            "connect_database",
            {"name": "ttl_ask_no", "db_type": "turtle", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "ttl_ask_no",
                    "query": (
                        "PREFIX ex: <http://example.org/> "
                        "ASK { ex:Bob ex:knows ex:Alice }"
                    ),
                },
            )
            assert result["success"] is True
            assert result["results"][0]["result"] is False
        finally:
            call_tool("disconnect_database", {"name": "ttl_ask_no"})


class TestDescribeDatabase:
    def test_describe_rdf(self, tmp_path):
        path = str(tmp_path / "desc.ttl")
        with open(path, "w") as f:
            f.write(SAMPLE_TURTLE)
        call_tool(
            "connect_database",
            {"name": "ttl_desc", "db_type": "turtle", "conn_string": path},
        )
        try:
            result = call_tool("describe_database", {"name": "ttl_desc"})
            assert result["name"] == "ttl_desc"
            assert result["storage"] == "rdf"
            assert result["triple_count"] == 7
            assert result["subject_count"] == 2
            assert "namespaces" in result
        finally:
            call_tool("disconnect_database", {"name": "ttl_desc"})


class TestDisconnect:
    def test_disconnect_rdf(self, tmp_path):
        path = str(tmp_path / "disc.ttl")
        with open(path, "w") as f:
            f.write(SAMPLE_TURTLE)
        call_tool(
            "connect_database",
            {"name": "ttl_disc", "db_type": "turtle", "conn_string": path},
        )
        result = call_tool("disconnect_database", {"name": "ttl_disc"})
        assert "Successfully disconnected" in result["message"]

    def test_disconnect_ntriples(self, tmp_path):
        path = str(tmp_path / "disc.nt")
        with open(path, "w") as f:
            f.write(SAMPLE_NTRIPLES)
        call_tool(
            "connect_database",
            {"name": "nt_disc", "db_type": "ntriples", "conn_string": path},
        )
        result = call_tool("disconnect_database", {"name": "nt_disc"})
        assert "Successfully disconnected" in result["message"]
