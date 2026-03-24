"""Parsers for loading Turtle and N-Triples RDF files into RDF storage.

Each parser populates an :class:`RDFStorageManager` via rdflib, using
the manager's :meth:`parse_file` method as the shared engine.
"""

from typing import Any, Dict, List

from rdflib import OWL, RDF, RDFS

from localdata_mcp.rdf_storage import RDFStorageManager


def parse_turtle_to_rdf(
    file_path: str,
    manager: RDFStorageManager,
) -> Dict[str, Any]:
    """Parse a Turtle file and populate the RDF storage.

    Returns a summary with triple count, namespaces, and ontology
    metadata (classes and properties found in the graph).

    Args:
        file_path: Path to the ``.ttl`` file.
        manager: The RDF storage manager to populate.
    """
    manager.parse_file(file_path, format="turtle")
    stats = manager.get_stats()

    # Extract ontology metadata
    classes: List[str] = sorted(
        {str(s) for s in manager.graph.subjects(RDF.type, RDFS.Class)}
        | {str(s) for s in manager.graph.subjects(RDF.type, OWL.Class)}
    )
    properties: List[str] = sorted(
        {str(s) for s in manager.graph.subjects(RDF.type, RDF.Property)}
        | {str(s) for s in manager.graph.subjects(RDF.type, OWL.ObjectProperty)}
        | {str(s) for s in manager.graph.subjects(RDF.type, OWL.DatatypeProperty)}
    )

    return {
        **stats,
        "classes": classes[:20],
        "properties": properties[:20],
    }


def parse_ntriples_to_rdf(
    file_path: str,
    manager: RDFStorageManager,
) -> Dict[str, Any]:
    """Parse an N-Triples file and populate the RDF storage.

    Returns a summary with triple count and subject/predicate/object counts.

    Args:
        file_path: Path to the ``.nt`` file.
        manager: The RDF storage manager to populate.
    """
    return manager.parse_file(file_path, format="nt")
