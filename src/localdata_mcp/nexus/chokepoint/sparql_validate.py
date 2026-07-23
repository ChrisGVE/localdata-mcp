"""localdata_mcp/nexus/chokepoint/sparql_validate.py — SPARQL screen (E6.2b).

NFR-104's construct-validation discipline extended to the second query
language (I-3), as declared data beside the SQL policy: a
deny-by-default construct screen every SPARQL string crosses before it
reaches an rdflib store.

- The READ path (`guarded_query`): SELECT/ASK/CONSTRUCT/DESCRIBE only —
  rdflib's query parser refuses update forms structurally, so a read
  string that parses at all is already one of the four. On top of that,
  `SERVICE` (federated query — an engine-issued outbound HTTP request,
  the SSRF shape the SQL deny-set closes for DuckDB `httpfs`) is refused
  at ANY depth via an algebra walk. A read string that does not parse is
  refused (never a vacuous pass).
- The WRITE path (`guarded_mutation`, read-write posture only): the
  update forms (`INSERT`/`DELETE`/`LOAD`/`CLEAR`/`DROP`/`CREATE`/`ADD`/
  `MOVE`/`COPY`). `LOAD` and the dataset-management forms are the
  federated/side-effecting class; they are permitted ONLY here, never on
  the read path, so a read query can neither exfiltrate via `SERVICE`
  nor mutate via an update form.

The denied read-construct set is the ONE named constant
(`DENIED_READ_CONSTRUCTS`) — documentation references it, never
restates. Neighbors: guard.py calls `screen_read`/`screen_update`; the
security battery drives a `SERVICE`-clause payload row through here.
"""

from __future__ import annotations

from typing import Any

from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery, parseUpdate
from rdflib.plugins.sparql.parserutils import CompValue

# The algebra node names refused on the read path, at any depth (§I-3).
# ServiceGraphPattern is SPARQL `SERVICE`; the update/dataset-management
# forms cannot appear in a parsed read query, but naming them documents
# the full deny-by-default surface in one place.
DENIED_READ_CONSTRUCTS: frozenset[str] = frozenset({"ServiceGraphPattern"})


class SparqlRefusedError(ValueError):
    """A SPARQL string failed the construct screen — structured refusal
    (never a partial verdict); guard.py shapes it through NX-3."""


def screen_read(sparql: str) -> None:
    """Refuse anything but a SERVICE-free SELECT/ASK/CONSTRUCT/DESCRIBE.

    Raises SparqlRefusedError on a parse failure, on an update form
    (which the query parser rejects), or on a denied construct at any
    nesting depth.
    """
    try:
        query = translateQuery(parseQuery(sparql))
    except Exception as failure:
        raise SparqlRefusedError(
            f"SPARQL read does not parse as a query: {failure}"
        ) from failure
    found = _construct_names(query.algebra)
    denied = found & DENIED_READ_CONSTRUCTS
    if denied:
        raise SparqlRefusedError(
            f"SPARQL read uses denied construct(s) {sorted(denied)} — refused"
        )


def screen_update(sparql: str) -> None:
    """Confirm the string parses as a SPARQL UPDATE (the guarded_mutation
    path). A read-only query string is refused here — it belongs on the
    query path — and unparseable input is refused."""
    try:
        parsed = parseUpdate(sparql)
    except Exception as failure:
        raise SparqlRefusedError(
            f"SPARQL update does not parse: {failure}"
        ) from failure
    if not parsed:
        raise SparqlRefusedError("SPARQL update is empty — refused")


def _construct_names(algebra: Any) -> frozenset[str]:
    """Every CompValue node name in the algebra tree — the one walk both
    the SERVICE screen and any future read-construct rule read from."""
    names: set[str] = set()
    _collect(algebra, names)
    return frozenset(names)


def _collect(node: Any, names: set[str]) -> None:
    if isinstance(node, CompValue):
        names.add(node.name)
        for value in node.values():
            _collect(value, names)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _collect(value, names)
