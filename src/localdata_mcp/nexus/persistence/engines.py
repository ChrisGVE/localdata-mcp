"""localdata_mcp/nexus/persistence/engines.py — engine creation, one home.

The harvested successor of `connection_manager/engine_factory.py`:
every backend engine v3 opens is built here, from an NX-2 endpoint
declaration, with posture applied AT CREATION (E5.4, §8.1 `cef73b00`)
— SQLite gets `PRAGMA query_only = ON` on every pooled connect, DuckDB
opens `read_only=True` — so a read-only file engine cannot write even
if a later layer misbehaves. Networked backends (PostgreSQL/MySQL/
MSSQL/Oracle) have no equivalent creation-time switch; their posture is
enforced at the NX-6 chokepoint (NFR-113), which refuses mutation
constructs before they reach the engine. Credentials are injected at
connection-issue time from the declaration's `credentials_ref`
environment variable (NFR-110) — never present in any config file.

DuckDB is served by the native `duckdb` package behind the same
`EngineHandle` protocol (§5 allows a non-SQL client handle in `pool`;
§7.1's locked manifest declares `duckdb`, not `duckdb-engine`).
Neighbors: record.py stores the handle; manager.py disposes and
reissues it through the protocol; health.py probes through it.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable

import duckdb
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import QueuePool, StaticPool

from localdata_mcp.nexus.config.endpoints import (
    EndpointDeclaration,
    Posture,
    resolve_credential,
)
from localdata_mcp.nexus.persistence.limits import ResourceLimits
from localdata_mcp.nexus.persistence.rdf import RdfHandle, rdf_format_of
from localdata_mcp.nexus.persistence.store_schemas import ensure_store_schema

# Backends SQLAlchemy pools with a QueuePool (networked servers).
_NETWORKED_KINDS = frozenset({"postgresql", "mysql", "mssql", "oracle"})

# The declared store families (E8.3): SQLite files carrying the
# store_schemas.py table shapes, DSN-declared as `<kind>+sqlite://…`.
_STORE_KINDS = frozenset({"kv", "tree", "graph"})


class UnsupportedBackendError(ValueError):
    """The declared DSN names a backend v3 has no engine for."""


@runtime_checkable
class EngineHandle(Protocol):
    """What NX-5 assumes about a pool — nothing more.

    `connect()` yields a live backend connection scoped to the `with`
    block; `dispose()` discards every retained physical connection
    (§5's dispose half — disposal cannot fail the way a rollback can).
    """

    def connect(self) -> AbstractContextManager[Any]: ...

    def dispose(self) -> None: ...


@dataclass(frozen=True)
class SqlAlchemyHandle:
    """A SQLAlchemy `Engine` behind the handle protocol."""

    engine: Engine

    def connect(self) -> AbstractContextManager[Any]:
        return self.engine.connect()

    def dispose(self) -> None:
        self.engine.dispose()


@dataclass(frozen=True)
class DuckDbHandle:
    """Native DuckDB behind the handle protocol.

    Each `connect()` opens a fresh `duckdb.connect(path, read_only=…)`
    scoped to the block — nothing is retained between calls, so
    `dispose()` has nothing to discard and a reissue is simply the next
    open. `read_only=True` is DuckDB's `access_mode=READ_ONLY`
    (§8.1 `cef73b00`).
    """

    path: str
    read_only: bool

    @contextmanager
    def _open(self) -> Iterator[duckdb.DuckDBPyConnection]:
        connection = duckdb.connect(self.path, read_only=self.read_only)
        try:
            yield connection
        finally:
            connection.close()

    def connect(self) -> AbstractContextManager[Any]:
        return self._open()

    def dispose(self) -> None:
        """Nothing pooled between calls — nothing to discard."""


def backend_kind_of(dsn: str) -> str:
    """The backend family of a DSN — the scheme with any SQLAlchemy
    driver suffix dropped (`postgresql+psycopg2` → `postgresql`)."""
    scheme = dsn.split("://", 1)[0]
    return scheme.split("+", 1)[0].lower()


def create_handle(
    declaration: EndpointDeclaration,
    limits: ResourceLimits,
    environ: Any,
) -> EngineHandle:
    """One endpoint's engine, posture and limits applied at creation.

    `environ` is the mapping `credentials_ref` resolves against at
    connection-issue time (NFR-110) — passed in, never read globally,
    so tests and the process entrypoint share one code path.
    """
    kind = backend_kind_of(declaration.dsn)
    if kind == "duckdb":
        return DuckDbHandle(
            path=make_url(declaration.dsn).database or "",
            read_only=declaration.posture == "read_only",
        )
    if kind == "sqlite":
        return _sqlite_handle(declaration.dsn, declaration.posture)
    if kind in _STORE_KINDS:
        return _store_handle(declaration.dsn, declaration.posture, kind)
    if kind == "rdf":
        return _rdf_handle(declaration.dsn, declaration.posture)
    if kind in _NETWORKED_KINDS:
        return _networked_handle(declaration, limits, environ)
    raise UnsupportedBackendError(
        f"endpoint {declaration.name!r} declares unsupported backend {kind!r}"
    )


def _sqlite_handle(dsn: str, posture: Posture) -> SqlAlchemyHandle:
    """SQLite: StaticPool (one embedded connection, harvested pattern)
    with `PRAGMA query_only = ON` on connect when read-only (§8.1)."""
    engine = create_engine(
        dsn,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    if posture == "read_only":

        @event.listens_for(engine, "connect")
        def _apply_query_only(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA query_only = ON")
            finally:
                cursor.close()

    return SqlAlchemyHandle(engine=engine)


def _store_handle(dsn: str, posture: Posture, store_kind: str) -> SqlAlchemyHandle:
    """A kv/tree/graph store (E8.3): the SQLite engine behind the
    family-prefixed DSN (`kv+sqlite:///f` → `sqlite:///f`), with
    `PRAGMA foreign_keys = ON` on every connect (the tree schema's
    property cascade relies on it — SQLite defaults it off) and the
    store_schemas.py tables ensured at creation on read-write posture
    (a read-only store expects a pre-seeded file, like every read-only
    file engine)."""
    sqlite_dsn = "sqlite://" + dsn.split("://", 1)[1]
    handle = _sqlite_handle(sqlite_dsn, posture)

    @event.listens_for(handle.engine, "connect")
    def _apply_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
        finally:
            cursor.close()

    if posture == "read_write":
        ensure_store_schema(handle.engine, store_kind)
    return handle


def _rdf_handle(dsn: str, posture: Posture) -> RdfHandle:
    """An rdf store (E8.3): the rdflib graph behind `rdf+<format>://…`,
    parsed from the declared file at creation; read-only posture makes
    the handle itself refuse updates (defense in depth under NX-6)."""
    sub_scheme = dsn.split("://", 1)[0].split("+", 1)[1]
    return RdfHandle(
        path=make_url("sqlite://" + dsn.split("://", 1)[1]).database or "",
        format=rdf_format_of(sub_scheme),
        read_only=posture == "read_only",
    )


def _networked_handle(
    declaration: EndpointDeclaration,
    limits: ResourceLimits,
    environ: Any,
) -> SqlAlchemyHandle:
    """A networked server engine: hard-capped QueuePool from the NX-2
    limits (max_overflow=0 — the S8 row-3 ceiling is a ceiling, GP3),
    pre-ping validation instead of any time-based recycle literal, and
    the credential injected into the URL from the environment."""
    url = make_url(declaration.dsn)
    secret = resolve_credential(declaration, environ)
    if secret is not None:
        url = url.set(password=secret)
    engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=limits.max_connections,
        max_overflow=0,
        pool_timeout=limits.statement_timeout_seconds,
        pool_pre_ping=True,
    )
    return SqlAlchemyHandle(engine=engine)
