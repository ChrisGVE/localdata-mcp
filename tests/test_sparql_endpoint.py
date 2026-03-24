"""Tests for SPARQL endpoint connection support.

All tests use mocked SPARQLWrapper to avoid real HTTP calls.
"""

import json
import threading
from unittest.mock import MagicMock, patch, call

import pytest

from localdata_mcp.sparql_endpoint import SPARQLEndpointConnection


# ---------------------------------------------------------------------------
# SPARQLEndpointConnection unit tests
# ---------------------------------------------------------------------------


class TestSPARQLEndpointInit:
    """Test SPARQLEndpointConnection initialization."""

    def test_default_timeout(self) -> None:
        conn = SPARQLEndpointConnection("http://example.org/sparql")
        assert conn.endpoint_url == "http://example.org/sparql"
        assert conn.timeout == 60

    def test_custom_timeout(self) -> None:
        conn = SPARQLEndpointConnection("http://example.org/sparql", timeout=120)
        assert conn.timeout == 120


class TestBindingToPython:
    """Test _binding_to_python for all binding types."""

    def _make_conn(self) -> SPARQLEndpointConnection:
        return SPARQLEndpointConnection("http://example.org/sparql")

    def test_uri(self) -> None:
        conn = self._make_conn()
        result = conn._binding_to_python(
            {"type": "uri", "value": "http://example.org/resource"}
        )
        assert result == "http://example.org/resource"

    def test_literal_string(self) -> None:
        conn = self._make_conn()
        result = conn._binding_to_python({"type": "literal", "value": "hello world"})
        assert result == "hello world"

    def test_literal_integer(self) -> None:
        conn = self._make_conn()
        result = conn._binding_to_python(
            {
                "type": "literal",
                "value": "42",
                "datatype": "http://www.w3.org/2001/XMLSchema#integer",
            }
        )
        assert result == 42
        assert isinstance(result, int)

    def test_literal_float(self) -> None:
        conn = self._make_conn()
        result = conn._binding_to_python(
            {
                "type": "literal",
                "value": "3.14",
                "datatype": "http://www.w3.org/2001/XMLSchema#double",
            }
        )
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_literal_decimal(self) -> None:
        conn = self._make_conn()
        result = conn._binding_to_python(
            {
                "type": "literal",
                "value": "2.5",
                "datatype": "http://www.w3.org/2001/XMLSchema#decimal",
            }
        )
        assert result == pytest.approx(2.5)

    def test_literal_boolean_true(self) -> None:
        conn = self._make_conn()
        result = conn._binding_to_python(
            {
                "type": "literal",
                "value": "true",
                "datatype": "http://www.w3.org/2001/XMLSchema#boolean",
            }
        )
        assert result is True

    def test_literal_boolean_false(self) -> None:
        conn = self._make_conn()
        result = conn._binding_to_python(
            {
                "type": "literal",
                "value": "false",
                "datatype": "http://www.w3.org/2001/XMLSchema#boolean",
            }
        )
        assert result is False

    def test_bnode(self) -> None:
        conn = self._make_conn()
        result = conn._binding_to_python({"type": "bnode", "value": "b0"})
        assert result == "_:b0"

    def test_unknown_type(self) -> None:
        conn = self._make_conn()
        result = conn._binding_to_python({"type": "unknown", "value": "foo"})
        assert result == "foo"


class TestExecuteQuery:
    """Test execute_query with mocked SPARQLWrapper."""

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_select_query(self, mock_wrapper_cls: MagicMock) -> None:
        mock_instance = mock_wrapper_cls.return_value
        mock_result = MagicMock()
        mock_result.convert.return_value = {
            "results": {
                "bindings": [
                    {
                        "s": {
                            "type": "uri",
                            "value": "http://example.org/Alice",
                        },
                        "name": {
                            "type": "literal",
                            "value": "Alice",
                        },
                    },
                    {
                        "s": {
                            "type": "uri",
                            "value": "http://example.org/Bob",
                        },
                        "name": {
                            "type": "literal",
                            "value": "Bob",
                        },
                    },
                ]
            }
        }
        mock_instance.query.return_value = mock_result

        conn = SPARQLEndpointConnection("http://example.org/sparql")
        results = conn.execute_query("SELECT ?s ?name WHERE { ?s ?p ?name }")
        assert len(results) == 2
        assert results[0]["s"] == "http://example.org/Alice"
        assert results[0]["name"] == "Alice"
        assert results[1]["s"] == "http://example.org/Bob"

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_ask_query(self, mock_wrapper_cls: MagicMock) -> None:
        mock_instance = mock_wrapper_cls.return_value
        mock_result = MagicMock()
        mock_result.convert.return_value = {"boolean": True}
        mock_instance.query.return_value = mock_result

        conn = SPARQLEndpointConnection("http://example.org/sparql")
        results = conn.execute_query("ASK { ?s ?p ?o }")
        assert results == [{"result": True}]

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_ask_query_false(self, mock_wrapper_cls: MagicMock) -> None:
        mock_instance = mock_wrapper_cls.return_value
        mock_result = MagicMock()
        mock_result.convert.return_value = {"boolean": False}
        mock_instance.query.return_value = mock_result

        conn = SPARQLEndpointConnection("http://example.org/sparql")
        results = conn.execute_query("ASK { <http://none> ?p ?o }")
        assert results == [{"result": False}]

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_error_raises_valueerror(self, mock_wrapper_cls: MagicMock) -> None:
        mock_instance = mock_wrapper_cls.return_value
        mock_instance.query.side_effect = Exception("Connection refused")

        conn = SPARQLEndpointConnection("http://example.org/sparql")
        with pytest.raises(ValueError, match="SPARQL query failed"):
            conn.execute_query("SELECT * WHERE { ?s ?p ?o }")

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_empty_bindings(self, mock_wrapper_cls: MagicMock) -> None:
        mock_instance = mock_wrapper_cls.return_value
        mock_result = MagicMock()
        mock_result.convert.return_value = {"results": {"bindings": []}}
        mock_instance.query.return_value = mock_result

        conn = SPARQLEndpointConnection("http://example.org/sparql")
        results = conn.execute_query("SELECT ?s WHERE { ?s ?p ?o }")
        assert results == []

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_non_dict_result(self, mock_wrapper_cls: MagicMock) -> None:
        mock_instance = mock_wrapper_cls.return_value
        mock_result = MagicMock()
        mock_result.convert.return_value = "some raw string"
        mock_instance.query.return_value = mock_result

        conn = SPARQLEndpointConnection("http://example.org/sparql")
        results = conn.execute_query("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
        assert results == []


class TestConcurrentQueries:
    """Test that concurrent queries do not interfere with each other."""

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_concurrent_queries_dont_interfere(
        self, mock_wrapper_cls: MagicMock
    ) -> None:
        """Two threads sending queries concurrently must each get correct results."""
        # Each call to SPARQLWrapper() returns a new mock instance
        instances = []

        def make_instance(url):
            inst = MagicMock()
            instances.append(inst)
            return inst

        mock_wrapper_cls.side_effect = make_instance

        conn = SPARQLEndpointConnection("http://example.org/sparql")

        result_a = {"results": {"bindings": [{"x": {"type": "literal", "value": "A"}}]}}
        result_b = {"results": {"bindings": [{"x": {"type": "literal", "value": "B"}}]}}

        results = {}
        barrier = threading.Barrier(2)

        def run_query(name, expected_result):
            # Configure mock for this thread's instance
            barrier.wait()
            r = conn.execute_query(f"SELECT ?x WHERE {{ ?x a '{name}' }}")
            results[name] = r

        # Pre-configure the instances that will be created
        mock_a_result = MagicMock()
        mock_a_result.convert.return_value = result_a
        mock_b_result = MagicMock()
        mock_b_result.convert.return_value = result_b

        # Each execute_query creates a new SPARQLWrapper instance
        # So we set up side_effect to return different results per call
        call_count = [0]
        original_side_effect = mock_wrapper_cls.side_effect

        def make_instance_with_results(url):
            inst = MagicMock()
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                mock_result = MagicMock()
                mock_result.convert.return_value = result_a
                inst.query.return_value = mock_result
            else:
                mock_result = MagicMock()
                mock_result.convert.return_value = result_b
                inst.query.return_value = mock_result
            return inst

        mock_wrapper_cls.side_effect = make_instance_with_results

        t1 = threading.Thread(target=run_query, args=("thread1", result_a))
        t2 = threading.Thread(target=run_query, args=("thread2", result_b))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Both threads should have completed and gotten results
        assert len(results) == 2
        # Each thread created its own SPARQLWrapper instance (no shared state)
        assert mock_wrapper_cls.call_count >= 2


class TestGetStats:
    """Test get_stats method."""

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_stats_with_count(self, mock_wrapper_cls: MagicMock) -> None:
        mock_instance = mock_wrapper_cls.return_value
        mock_result = MagicMock()
        mock_result.convert.return_value = {
            "results": {
                "bindings": [
                    {
                        "count": {
                            "type": "literal",
                            "value": "1000",
                            "datatype": "http://www.w3.org/2001/XMLSchema#integer",
                        }
                    }
                ]
            }
        }
        mock_instance.query.return_value = mock_result

        conn = SPARQLEndpointConnection("http://example.org/sparql")
        stats = conn.get_stats()
        assert stats["endpoint_url"] == "http://example.org/sparql"
        assert stats["type"] == "sparql_endpoint"
        assert stats["timeout"] == 60
        assert stats["approximate_triples"] == 1000

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_stats_when_count_fails(self, mock_wrapper_cls: MagicMock) -> None:
        mock_instance = mock_wrapper_cls.return_value
        mock_instance.query.side_effect = Exception("Timeout")

        conn = SPARQLEndpointConnection("http://example.org/sparql")
        stats = conn.get_stats()
        assert stats["approximate_triples"] == "unavailable"
        assert stats["endpoint_url"] == "http://example.org/sparql"


# ---------------------------------------------------------------------------
# Integration with DatabaseManager
# ---------------------------------------------------------------------------


class TestDatabaseManagerSPARQLIntegration:
    """Test SPARQL endpoint integration with DatabaseManager."""

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_connect_sparql_endpoint(self, mock_wrapper_cls: MagicMock) -> None:
        """Test connecting to a SPARQL endpoint via connect_database."""
        from localdata_mcp.localdata_mcp import DatabaseManager

        # Mock the stats query for connection summary
        mock_instance = mock_wrapper_cls.return_value
        mock_query_result = MagicMock()
        mock_query_result.convert.return_value = {"results": {"bindings": []}}
        mock_instance.query.return_value = mock_query_result

        mgr = DatabaseManager()
        result = mgr.connect_database(
            "wikidata", "sparql", "https://query.wikidata.org/sparql"
        )
        parsed = json.loads(result)

        assert parsed["success"] is True
        assert "wikidata" in mgr._sparql_connections
        assert mgr._sparql_connections["wikidata"].endpoint_url == (
            "https://query.wikidata.org/sparql"
        )

        # Clean up
        mgr.disconnect_database("wikidata")

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_execute_query_on_sparql_endpoint(
        self, mock_wrapper_cls: MagicMock
    ) -> None:
        """Test execute_query dispatches to SPARQL endpoint."""
        from localdata_mcp.localdata_mcp import DatabaseManager

        mock_instance = mock_wrapper_cls.return_value

        # First call: stats during connect (returns empty)
        # Second call: actual query
        stats_result = MagicMock()
        stats_result.convert.return_value = {"results": {"bindings": []}}

        query_result = MagicMock()
        query_result.convert.return_value = {
            "results": {
                "bindings": [
                    {
                        "item": {
                            "type": "uri",
                            "value": "http://www.wikidata.org/entity/Q42",
                        }
                    }
                ]
            }
        }
        mock_instance.query.side_effect = [stats_result, query_result]

        mgr = DatabaseManager()
        mgr.connect_database("wikidata", "sparql", "https://query.wikidata.org/sparql")

        result = mgr.execute_query(
            "wikidata", "SELECT ?item WHERE { ?item ?p ?o } LIMIT 1"
        )
        parsed = json.loads(result)

        assert parsed["query_type"] == "SPARQL"
        assert parsed["count"] == 1
        assert parsed["results"][0]["item"] == ("http://www.wikidata.org/entity/Q42")

        mgr.disconnect_database("wikidata")

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_disconnect_cleans_up(self, mock_wrapper_cls: MagicMock) -> None:
        """Test disconnecting removes the SPARQL connection."""
        from localdata_mcp.localdata_mcp import DatabaseManager

        mock_instance = mock_wrapper_cls.return_value
        mock_query_result = MagicMock()
        mock_query_result.convert.return_value = {"results": {"bindings": []}}
        mock_instance.query.return_value = mock_query_result

        mgr = DatabaseManager()
        mgr.connect_database("dbpedia", "sparql", "http://dbpedia.org/sparql")
        assert "dbpedia" in mgr._sparql_connections

        mgr.disconnect_database("dbpedia")
        assert "dbpedia" not in mgr._sparql_connections

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_list_databases_includes_sparql(self, mock_wrapper_cls: MagicMock) -> None:
        """Test list_databases shows SPARQL endpoint connections."""
        from localdata_mcp.localdata_mcp import DatabaseManager

        mock_instance = mock_wrapper_cls.return_value
        mock_query_result = MagicMock()
        mock_query_result.convert.return_value = {"results": {"bindings": []}}
        mock_instance.query.return_value = mock_query_result

        mgr = DatabaseManager()
        mgr.connect_database("wikidata", "sparql", "https://query.wikidata.org/sparql")

        result = json.loads(mgr.list_databases())
        db_entry = next(d for d in result["databases"] if d["name"] == "wikidata")
        assert db_entry["storage"] == "sparql_endpoint"
        assert db_entry["sql_flavor"] == "SPARQL Endpoint"

        mgr.disconnect_database("wikidata")

    @patch("localdata_mcp.sparql_endpoint.SPARQLWrapper")
    def test_execute_query_error_returns_json(
        self, mock_wrapper_cls: MagicMock
    ) -> None:
        """Test that query errors are returned as JSON, not raised."""
        from localdata_mcp.localdata_mcp import DatabaseManager

        mock_instance = mock_wrapper_cls.return_value

        stats_result = MagicMock()
        stats_result.convert.return_value = {"results": {"bindings": []}}

        mock_instance.query.side_effect = [
            stats_result,
            Exception("Endpoint unavailable"),
        ]

        mgr = DatabaseManager()
        mgr.connect_database("broken", "sparql", "http://broken.example.org/sparql")

        result = mgr.execute_query("broken", "SELECT ?s WHERE { ?s ?p ?o }")
        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["query_type"] == "SPARQL"

        mgr.disconnect_database("broken")
