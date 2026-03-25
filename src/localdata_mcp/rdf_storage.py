"""RDF graph storage backed by rdflib.

Provides in-memory RDF triple storage with Turtle and N-Triples format
support, SPARQL query execution, and graph navigation methods that
bridge to the tool interface.
"""

import logging
import os
from typing import Any, Dict, List

import rdflib
from rdflib import BNode, Graph, Literal, URIRef

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

_VALID_FORMATS = {"turtle", "nt"}

MAX_RDF_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


def _validate_format(fmt: str) -> None:
    """Raise ``ValueError`` if *fmt* is not a supported RDF format."""
    if fmt not in _VALID_FORMATS:
        raise ValueError(
            f"Unsupported format '{fmt}'. Must be one of: {sorted(_VALID_FORMATS)}"
        )


# ---------------------------------------------------------------------------
# RDF Storage Manager
# ---------------------------------------------------------------------------


class RDFStorageManager:
    """In-memory RDF graph storage backed by rdflib.

    Supports Turtle and N-Triples formats, SPARQL queries,
    and graph navigation bridging to the tool interface.
    """

    def __init__(self) -> None:
        self.graph: Graph = Graph()

    # -- Parsing ------------------------------------------------------------

    def parse_file(self, file_path: str, format: str) -> Dict[str, Any]:
        """Parse an RDF file into the graph.

        Args:
            file_path: Path to the RDF file.
            format: Serialization format (``'turtle'`` or ``'nt'``).

        Returns:
            A summary dict with ``triple_count``, ``subject_count``,
            ``predicate_count``, and ``object_count``.
        """
        _validate_format(format)
        file_size = os.path.getsize(file_path)
        if file_size > MAX_RDF_FILE_BYTES:
            logger.warning(
                "RDF file %s is %d MB; loading entirely into memory",
                file_path,
                file_size // (1024 * 1024),
            )
        self.graph.parse(file_path, format=format)
        return {
            "triple_count": len(self.graph),
            "subject_count": len(set(self.graph.subjects())),
            "predicate_count": len(set(self.graph.predicates())),
            "object_count": len(set(self.graph.objects())),
        }

    # -- SPARQL execution ---------------------------------------------------

    def execute_sparql(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SPARQL query against the graph.

        - **SELECT** queries return a list of dicts keyed by variable name.
        - **ASK** queries return ``[{"result": bool}]``.
        - **CONSTRUCT** queries return ``[{"triples": str}]`` with the
          constructed graph serialized as Turtle.

        Raises:
            ValueError: On invalid SPARQL syntax or execution errors.
        """
        try:
            result = self.graph.query(query)
        except Exception as exc:
            raise ValueError(f"SPARQL error: {exc}") from exc
        return self._format_result(result)

    def _format_result(self, result: rdflib.query.Result) -> List[Dict[str, Any]]:
        """Convert an rdflib query result to a list of dicts."""
        if result.type == "ASK":
            return [{"result": bool(result.askAnswer)}]

        if result.type == "CONSTRUCT":
            constructed = Graph()
            for triple in result:
                constructed.add(triple)
            return [{"triples": constructed.serialize(format="turtle")}]

        # SELECT
        rows: List[Dict[str, Any]] = []
        for row in result:
            record: Dict[str, Any] = {}
            for var in result.vars:
                record[str(var)] = self._term_to_python(row[var])
            rows.append(record)
        return rows

    def _term_to_python(self, term: Any) -> Any:
        """Convert an RDF term to a native Python value.

        - ``URIRef`` -> ``str``
        - ``Literal`` -> ``toPython()`` (falls back to ``str()`` on error)
        - ``BNode`` -> ``'_:xxx'``
        - ``None`` -> ``None``
        """
        if term is None:
            return None
        if isinstance(term, URIRef):
            return str(term)
        if isinstance(term, Literal):
            try:
                return term.toPython()
            except Exception as e:
                logger.warning("Failed to convert RDF literal %r: %s", term, e)
                return str(term)
        if isinstance(term, BNode):
            return f"_:{term}"
        return str(term)

    def _resolve_term(self, value: str):
        """Convert a string back to the appropriate rdflib term."""
        if value.startswith("_:"):
            return BNode(value[2:])
        return URIRef(value)

    # -- Statistics ---------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the current graph.

        Keys: ``triple_count``, ``namespaces`` (list of ``{prefix, uri}``
        dicts), ``subject_count``, ``predicate_count``, ``object_count``.
        """
        namespaces = [
            {"prefix": str(prefix), "uri": str(uri)}
            for prefix, uri in self.graph.namespaces()
        ]
        return {
            "triple_count": len(self.graph),
            "namespaces": namespaces,
            "subject_count": len(set(self.graph.subjects())),
            "predicate_count": len(set(self.graph.predicates())),
            "object_count": len(set(self.graph.objects())),
        }

    # -- Graph navigation ---------------------------------------------------

    def get_subjects(self, offset: int = 0, limit: int = 50) -> List[str]:
        """List unique subjects with pagination.

        Returns URIs/blank-node identifiers as strings, sorted for
        deterministic output.
        """
        subjects = sorted(self._term_to_python(s) for s in set(self.graph.subjects()))
        return subjects[offset : offset + limit]

    def get_predicates_for_subject(self, subject: str) -> List[str]:
        """List unique predicates for a given subject URI or blank node."""
        term = self._resolve_term(subject)
        return sorted(set(str(p) for p in self.graph.predicates(subject=term)))

    def get_objects(self, subject: str, predicate: str) -> List[Any]:
        """Get all objects for a subject-predicate pair.

        Objects are converted to Python values via ``_term_to_python``.
        """
        s_term = self._resolve_term(subject)
        p_term = URIRef(predicate)
        return [
            self._term_to_python(o)
            for o in self.graph.objects(subject=s_term, predicate=p_term)
        ]

    # -- Export -------------------------------------------------------------

    def serialize(self, format: str = "turtle") -> str:
        """Serialize the graph to a string.

        Args:
            format: Output format (``'turtle'`` or ``'nt'``).

        Returns:
            The serialized RDF string.
        """
        _validate_format(format)
        return self.graph.serialize(format=format)
