"""localdata_mcp/nexus/persistence/rdf.py — the RDF store handle (E8.3).

I-3's "RDF/SPARQL access is core" made concrete as an NX-5 engine
handle: an `rdf+turtle://` / `rdf+ntriples://` endpoint declaration
becomes an `rdflib.Graph` parsed from the declared file, served behind
the same `EngineHandle` protocol every backend answers to (§5 allows a
non-SQL client handle in `pool`). The connection object is
CURSOR-SHAPED — `execute(text, params)` returning an object with
`description`/`fetchmany`/`rowcount` — so the chokepoint's structural
execution adapter (execution.py's non-SQLAlchemy branch) runs SPARQL
through the guard with zero special-casing; the guard has already
screened the text through the E6.2b construct policy by backend kind.
Updates apply and serialize back to the declared file immediately
(autocommit, matching the native-connection contract execution.py
assumes) and are DOUBLY gated: NX-6 refuses updates below read_write
posture, and a read-only handle refuses them here too (defense in
depth, GP3). Parameters bind through `initBindings` — never spliced
into the query text. Neighbors: engines.py builds this handle;
health.py probes it with `ASK {}`; chokepoint/sparql_validate.py
screens every string before it arrives.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, cast

from rdflib import Graph, Literal, URIRef

# DSN sub-scheme → rdflib serialization format name.
RDF_FORMATS: Mapping[str, str] = {
    "turtle": "turtle",
    "ttl": "turtle",
    "ntriples": "nt",
    "nt": "nt",
}


class UnsupportedRdfFormatError(ValueError):
    """The declared rdf DSN names a serialization v3 has no format for."""


def rdf_format_of(sub_scheme: str) -> str:
    """The rdflib format name for a DSN sub-scheme (`rdf+turtle` → turtle)."""
    try:
        return RDF_FORMATS[sub_scheme.lower()]
    except KeyError:
        raise UnsupportedRdfFormatError(
            f"rdf DSN sub-scheme {sub_scheme!r} is not supported "
            f"(supported: {sorted(set(RDF_FORMATS))})"
        ) from None


def _to_binding(value: Any) -> Any:
    """A bound parameter as an rdflib term: terms pass through, an
    http(s) IRI string binds as a reference, everything else binds as a
    typed literal — values never splice into the query text."""
    if isinstance(value, (Literal, URIRef)):
        return value
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return URIRef(value)
    return Literal(value)


class _RdfCursor:
    """The cursor shape execution.py reads: `description` names the
    columns (None for updates), `fetchmany` serves plain tuples."""

    def __init__(
        self,
        columns: tuple[str, ...] | None,
        rows: list[tuple[Any, ...]],
        rowcount: int,
    ) -> None:
        self.description = None if columns is None else [(name,) for name in columns]
        self.rowcount = rowcount
        self._rows = rows
        self._served = 0

    def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        batch = self._rows[self._served : self._served + size]
        self._served += len(batch)
        return batch


class RdfReadOnlyError(PermissionError):
    """An update reached a read-only rdf handle — the engine-level half
    of the posture gate (NX-6 refuses first; this cannot be reached
    through the guard, GP3's defense in depth)."""


# The SPARQL update keywords rdflib's query parser refuses — used only
# to ROUTE a string to `update()` vs `query()`; the actual screen is
# the chokepoint's (sparql_validate.py), applied before execution.
_UPDATE_PREFIXES = (
    "insert",
    "delete",
    "load",
    "clear",
    "drop",
    "create",
    "add",
    "move",
    "copy",
    "with",
)


def _is_update(sparql: str) -> bool:
    head = sparql.lstrip().split(None, 1)
    first = head[0].lower() if head else ""
    # PREFIX/BASE headers precede both forms; strip them for routing.
    if first in ("prefix", "base"):
        remainder = sparql.lstrip()[len(head[0]) :].split(">", 1)
        return _is_update(remainder[1]) if len(remainder) > 1 else False
    return first in _UPDATE_PREFIXES


class RdfConnection:
    """One live view over the handle's graph, scoped to a `with` block."""

    def __init__(self, handle: "RdfHandle") -> None:
        self._handle = handle

    def execute(
        self, sparql: str, parameters: Mapping[str, Any] | None = None
    ) -> _RdfCursor:
        bindings = {
            name: _to_binding(value) for name, value in (parameters or {}).items()
        }
        if _is_update(sparql):
            return self._handle._apply_update(sparql)
        return self._handle._run_query(sparql, bindings)


@dataclass
class RdfHandle:
    """An rdflib store behind the `EngineHandle` protocol.

    The graph is parsed once at creation (engines.py); every
    `connect()` serves the same in-memory graph — `dispose()` drops it,
    and a reissue re-parses the file (matching dispose-and-reissue §5).
    """

    path: str
    format: str
    read_only: bool

    def __post_init__(self) -> None:
        self._graph: Graph | None = Graph()
        if Path(self.path).exists():
            self._graph.parse(self.path, format=self.format)

    @contextmanager
    def _open(self) -> Iterator[RdfConnection]:
        yield RdfConnection(self)

    def connect(self) -> Any:
        return self._open()

    def dispose(self) -> None:
        self._graph = None

    # -- execution halves (called through RdfConnection) --------------

    def _live_graph(self) -> Graph:
        if self._graph is None:
            raise RuntimeError("rdf handle is disposed")
        return self._graph

    def _run_query(self, sparql: str, bindings: Mapping[str, Any]) -> _RdfCursor:
        result = self._live_graph().query(sparql, initBindings=dict(bindings))
        if result.type == "ASK":
            return _RdfCursor(("ask",), [(bool(result.askAnswer),)], -1)
        if result.type in ("CONSTRUCT", "DESCRIBE"):
            triples = [(str(s), str(p), str(o)) for s, p, o in result.graph or ()]
            return _RdfCursor(("subject", "predicate", "object"), triples, -1)
        columns = tuple(str(var) for var in (result.vars or ()))
        rows: list[tuple[Any, ...]] = []
        for row in result.bindings:
            rows.append(
                tuple(
                    None
                    if (term := row.get(var)) is None
                    else cast(Any, term).toPython()
                    for var in (result.vars or ())
                )
            )
        return _RdfCursor(columns, rows, -1)

    def _apply_update(self, sparql: str) -> _RdfCursor:
        if self.read_only:
            raise RdfReadOnlyError(
                f"rdf store {self.path!r} is read-only; updates need read_write posture"
            )
        graph = self._live_graph()
        graph.update(sparql)
        graph.serialize(destination=self.path, format=self.format)
        return _RdfCursor(None, [], -1)
