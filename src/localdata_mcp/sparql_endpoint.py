"""SPARQL endpoint connection for remote triple stores.

Provides a thin wrapper around SPARQLWrapper for querying remote
SPARQL endpoints such as Wikidata, DBpedia, UniProt, etc.
"""

import logging
from typing import Any, Dict, List
from urllib.parse import urlparse

from SPARQLWrapper import JSON, SPARQLWrapper

logger = logging.getLogger(__name__)


class SPARQLEndpointConnection:
    """Connection to a remote SPARQL endpoint."""

    def __init__(self, endpoint_url: str, timeout: int = 60) -> None:
        parsed = urlparse(endpoint_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"SPARQL endpoint URL must use http or https; got '{parsed.scheme}'"
            )
        self.endpoint_url = endpoint_url
        self.timeout = timeout

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SPARQL query and return results as list of dicts.

        SELECT queries return a list of ``{var: value}`` dicts.
        ASK queries return ``[{"result": True/False}]``.

        A new SPARQLWrapper instance is created per query to ensure
        thread safety.

        Raises:
            ValueError: When the SPARQL query fails or the response
                format is unexpected.
        """
        sparql = SPARQLWrapper(self.endpoint_url)
        sparql.setTimeout(self.timeout)
        sparql.setReturnFormat(JSON)
        sparql.setQuery(query)

        try:
            results = sparql.query().convert()
        except Exception as e:
            raise ValueError(f"SPARQL query failed: {e}") from e

        if isinstance(results, dict):
            if "boolean" in results:
                return [{"result": results["boolean"]}]
            if "results" in results and "bindings" in results["results"]:
                bindings = results["results"]["bindings"]
                return [
                    {var: self._binding_to_python(binding[var]) for var in binding}
                    for binding in bindings
                ]
        raise ValueError(f"Unexpected SPARQL response format: {type(results).__name__}")

    def _binding_to_python(self, binding: Dict[str, str]) -> Any:
        """Convert a single SPARQL JSON binding value to a Python value."""
        btype = binding.get("type", "")
        value = binding.get("value", "")

        if btype == "uri":
            return value
        if btype == "literal":
            datatype = binding.get("datatype", "")
            if "integer" in datatype:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return value
            if "decimal" in datatype or "float" in datatype or "double" in datatype:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return value
            if "boolean" in datatype:
                return value.lower() == "true"
            return value
        if btype == "bnode":
            return f"_:{value}"
        return value

    def get_stats(self) -> Dict[str, Any]:
        """Return endpoint metadata.

        Attempts a triple-count query but does not fail if unsupported.
        """
        stats: Dict[str, Any] = {
            "endpoint_url": self.endpoint_url,
            "timeout": self.timeout,
            "type": "sparql_endpoint",
        }
        try:
            count_result = self.execute_query(
                "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o } LIMIT 1"
            )
            if count_result:
                stats["approximate_triples"] = count_result[0].get("count", "unknown")
        except Exception as e:
            logger.warning(
                "Failed to query triple count from %s: %s", self.endpoint_url, e
            )
            stats["approximate_triples"] = "unavailable"
        return stats
