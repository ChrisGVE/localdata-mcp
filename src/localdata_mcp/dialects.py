"""What each database backend has to supply, and what it gets for free.

A tag is a database, and a database is reached through a SQLAlchemy engine. Most
of what this server does above that line is ordinary Core — a table is created,
rows are inserted, a result is streamed, a schema is inspected — and none of it
needs to know which backend answered.

A few things genuinely cannot be said portably, and this module is where they are
said. The test for belonging here is not "is this SQL awkward" but **"does this
mean something different, or nothing at all, on another backend"**:

* **Read-only posture.** Every backend can refuse writes; none of them spell it
  the same way. SQLite sets ``query_only`` on the connection, PostgreSQL starts a
  read-only transaction. The *guarantee* is portable, the mechanism is not.
* **Residency.** How much a database is holding in *this process* is a question
  that only means anything for an in-process, memory-backed database. For a file
  or a server it is either unobservable or somebody else's memory, and the honest
  answer is ``None`` — not zero, which would read as "holding nothing".
* **Snapshotting.** Writing a consistent copy of a database to a file is
  ``VACUUM INTO`` on SQLite and something else everywhere else.
* **Storage classes.** SQLite lets a column hold values of different types and
  ``typeof()`` counts them. On a backend with real column types the question does
  not arise, and an empty histogram is the truthful answer rather than a gap.

Everything else — creating tables, inserting, introspection via ``inspect()``,
streaming reads — is Core and lives in :mod:`loader`. When something new turns out
to need a per-backend answer, it earns an entry here; wrapping a statement in
``text()`` to change its transport is not that, and does not belong.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.pool import StaticPool

__all__ = [
    "BACKENDS",
    "Backend",
    "Engines",
    "Refusal",
    "SQLiteBackend",
    "UnsupportedOperation",
    "backend_for",
]


class UnsupportedOperation(RuntimeError):
    """Something this kind of datasource cannot be asked to do."""


# ---------------------------------------------------------------------------
# What a refusal knows about itself
# ---------------------------------------------------------------------------


@dataclass
class Refusal:
    """What a read-only connection last refused, in words.

    Enforcement does not depend on this — the connection's posture is what
    refuses. This exists so the error can name *what was attempted*, which is the
    difference between an agent that can fix its statement and one that only
    knows it was denied. A backend that cannot name the action leaves it ``None``
    and the message degrades to its general form; nothing else changes.
    """

    what: str | None = None

    def take(self) -> str | None:
        """Read the last refusal and forget it, so it cannot leak into the next."""
        seen, self.what = self.what, None
        return seen


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Engines:
    """The two postures a tag is reached through.

    Separate engines rather than one engine reconfigured per call. A connection's
    posture is set once, at connect time, and never toggled around an operation —
    ``docs/CONSTRAINTS.md`` §3 is explicit that flipping it per call is the shape
    that loses data. A read that cannot write is a *different connection*, not the
    same connection asked nicely.
    """

    write: Engine
    read: Engine
    refusal: Refusal

    def dispose(self) -> None:
        self.read.dispose()
        self.write.dispose()


class Backend:
    """The per-dialect answers — and the generic ones are the whole answer.

    **This class is not an interface every database must implement in order to
    be reachable.** Reaching a database is ``create_engine``'s job, and it
    already works for everything SQLAlchemy speaks; :meth:`open` below is that
    call and nothing more. A dialect subclasses this only when it can say
    something the generic answer cannot — how much memory it is holding, how to
    write itself to a file — and a dialect nobody has subclassed still opens,
    still queries, still composes.

    That is the direction the default implementations lean: an unknown backend
    answers *honestly and usefully*, never "unsupported".
    """

    #: SQLAlchemy's own backend name for the dialect. Descriptive, so an error
    #: can say which database declined; never used to decide reachability.
    name = "generic"

    def open(self, url: str | URL, *, writable: bool) -> Engines:
        """Two engines onto one datasource — one reading, one writing.

        Generic because ``create_engine`` is generic: SQLAlchemy resolves the
        driver, the dialect and the connection arguments from the URL, and
        nothing here needs to know which database answered. ``writable`` is not
        consulted in the generic case because there is nothing portable to do
        with it — the read engine refuses writes by never committing them (see
        :meth:`read_posture`), and what the *write* engine may do is the
        datasource's own business, enforced by its own grants.
        """
        refusal = Refusal()
        engines = Engines(
            write=create_engine(url), read=create_engine(url), refusal=refusal
        )
        self.read_posture(engines.read, refusal)
        return engines

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        """Make this engine's connections refuse to *persist* a write.

        The generic guarantee is transactional, and it needs no per-dialect
        code: :meth:`loader.Workspace.query` opens a connection, never commits,
        and closes it — so anything a statement changed is rolled back and no
        trace of it survives. That is the floor. A dialect may raise it, and
        SQLite does, refusing at statement preparation instead so that the
        statement never runs at all and the refusal can name what was attempted.
        """
        return None

    def resident_bytes(self, engine: Engine) -> int | None:
        """Bytes this database is holding in *our* process, or ``None``.

        ``None`` means the question does not apply — the data is on disk or on a
        server. It is deliberately not ``0``, which a caller would read as an
        empty database and spill nothing for.
        """
        return None

    def snapshot(self, engine: Engine, target: Path) -> None:
        """Write a consistent copy of this database to a local file.

        Refused generically, and the refusal is the truthful answer rather than
        a gap: a database this server merely *reaches* is not one it holds, and
        there is no local file to write out. Copying its rows into a slot of
        our own — ``add_table`` — is the route, and that slot saves.
        """
        raise UnsupportedOperation(
            f"A {self.name} datasource is reached over its own connection, not "
            f"held here, so there is no local database to write out. Copy the "
            f"rows you want into a slot of your own with add_table, and save "
            f"that."
        )

    def storage_classes(
        self, conn: Connection, table: str, column: str
    ) -> dict[str, int]:
        """How many values of each storage class the column actually holds.

        Empty for a backend whose columns have real types — there is one class
        per column by construction, so counting them says nothing.
        """
        return {}


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


#: What a read query is allowed to do, as SQLite authorizer action codes.
#:
#: A **whitelist**, deliberately. SQLite has some thirty action codes and gains
#: more between versions; a denylist would silently admit whatever arrives next,
#: which is the wrong direction to fail in. Anything not named here is refused —
#: including ``ATTACH``, which would otherwise be a way to reach a second
#: database from inside a read and quietly rebuild cross-tag joins.
_READ_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)

#: Refused actions in words, so the error names what was attempted rather than
#: quoting a number. Written out rather than derived from ``vars(sqlite3)``,
#: because the result codes share integer values with action codes
#: (``SQLITE_DENY`` is 1, and so is ``SQLITE_CREATE_INDEX``) and a derived map
#: would mislabel them.
_ACTION_NAMES = {
    sqlite3.SQLITE_INSERT: "INSERT",
    sqlite3.SQLITE_UPDATE: "UPDATE",
    sqlite3.SQLITE_DELETE: "DELETE",
    sqlite3.SQLITE_CREATE_TABLE: "CREATE TABLE",
    sqlite3.SQLITE_CREATE_VIEW: "CREATE VIEW",
    sqlite3.SQLITE_CREATE_INDEX: "CREATE INDEX",
    sqlite3.SQLITE_CREATE_TRIGGER: "CREATE TRIGGER",
    sqlite3.SQLITE_CREATE_TEMP_TABLE: "CREATE TEMP TABLE",
    sqlite3.SQLITE_CREATE_TEMP_VIEW: "CREATE TEMP VIEW",
    sqlite3.SQLITE_CREATE_TEMP_INDEX: "CREATE TEMP INDEX",
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER: "CREATE TEMP TRIGGER",
    sqlite3.SQLITE_DROP_TABLE: "DROP TABLE",
    sqlite3.SQLITE_DROP_VIEW: "DROP VIEW",
    sqlite3.SQLITE_DROP_INDEX: "DROP INDEX",
    sqlite3.SQLITE_DROP_TRIGGER: "DROP TRIGGER",
    sqlite3.SQLITE_DROP_TEMP_TABLE: "DROP TEMP TABLE",
    sqlite3.SQLITE_DROP_TEMP_VIEW: "DROP TEMP VIEW",
    sqlite3.SQLITE_ALTER_TABLE: "ALTER TABLE",
    sqlite3.SQLITE_REINDEX: "REINDEX",
    sqlite3.SQLITE_ANALYZE: "ANALYZE",
    sqlite3.SQLITE_ATTACH: "ATTACH a database",
    sqlite3.SQLITE_DETACH: "DETACH a database",
    sqlite3.SQLITE_PRAGMA: "set a PRAGMA",
    sqlite3.SQLITE_TRANSACTION: "control a transaction",
}


def _describe(action: int, target: str | None) -> str:
    """Name a refused action in words an agent can act on.

    One wrinkle worth stating, because the naive version is actively misleading:
    SQLite authorizes a DDL statement's write to ``sqlite_master`` *before* it
    authorizes the statement's own action code. The first refusal therefore
    arrives as ``INSERT``, and reporting that verbatim tells an agent that wrote
    ``CREATE VIEW`` it attempted an INSERT. Naming the schema write for what it
    is keeps the message true without guessing at the statement.
    """
    if action in (
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
    ) and (target or "").startswith("sqlite_"):
        return "change the database schema"
    return _ACTION_NAMES.get(action, f"perform action {action}")


@dataclass(frozen=True)
class SQLiteBackend(Backend):
    """SQLite, reached through pysqlite.

    **Memory-backed tags use a named shared-cache database, not ``:memory:``.**
    An anonymous in-memory database belongs to the single connection that made
    it, so a read-only connection could not see it at all. Naming it and asking
    for a shared cache is what lets one database answer through two connections
    with different postures — which is the whole read-only design. The name
    carries a per-session token so two workspaces in one process never land on
    each other's database.
    """

    name: str = "sqlite"
    #: Passed to every connection. URI mode is a connection-level flag, and the
    #: tool bodies are dispatched to worker OS threads (``CONSTRAINTS`` §3.4), so
    #: the connection is legitimately reached from more than one of them.
    connect_args: dict[str, Any] = field(
        default_factory=lambda: {"uri": True, "check_same_thread": False}
    )

    # -- opening -----------------------------------------------------------

    def open(self, url: str | URL, *, writable: bool) -> Engines:
        """A ``sqlite:`` URL names a local file, so open it as one.

        The generic :meth:`Backend.open` would work, but it would lose what
        SQLite can do better: ``mode=ro`` in the URI, which the database itself
        enforces rather than this code remembering to.
        """
        database = make_url(url).database or ""
        if not database or database == ":memory:":
            raise UnsupportedOperation(
                "An anonymous SQLite memory database cannot be shared between "
                "connections. Use open_memory, which names one."
            )
        return self.open_file(Path(database), writable=writable)

    def open_memory(self, tag: str, token: str) -> Engines:
        url = f"sqlite:///file:{tag}_{token}?mode=memory&cache=shared&uri=true"
        return self._pair(url, url)

    def open_file(self, path: Path, *, writable: bool) -> Engines:
        """Open a database file. Read-only is carried by the URI, not by care.

        A guarantee implemented as an interception point can be walked around by
        reaching the intercepted object; a URI opened ``mode=ro`` has nothing to
        reach around, because SQLite itself is what refuses.
        """
        base = path.resolve().as_uri()
        write_url = f"sqlite:///{base}?uri=true"
        if not writable:
            write_url = f"sqlite:///{base}?mode=ro&uri=true"
        return self._pair(write_url, f"sqlite:///{base}?mode=ro&uri=true")

    def _pair(self, write_url: str, read_url: str) -> Engines:
        refusal = Refusal()
        write = self._engine(write_url)
        read = self._engine(read_url)
        self.read_posture(read, refusal)
        return Engines(write=write, read=read, refusal=refusal)

    def _engine(self, url: str) -> Engine:
        # StaticPool, obligatorily: a memory-backed database exists only while a
        # connection to it does, so a pool that opens and closes connections
        # would find an empty database on the next checkout.
        return create_engine(
            url, poolclass=StaticPool, connect_args=dict(self.connect_args)
        )

    # -- read-only posture -------------------------------------------------

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        """Refuse every write on this engine's connections, from birth.

        Stronger than the generic rollback floor, and that is why it overrides:
        a refused statement never runs, and the refusal can name what it was.

        Two mechanisms, doing two different jobs. ``query_only`` is the posture —
        it is what SQLAlchemy would set for any backend asked for a read-only
        connection, and it is set once at connect time rather than toggled around
        a call. The authorizer is narrower and older: it refuses at *statement
        preparation*, so a refused statement never runs, it covers what
        ``query_only`` does not (``ATTACH`` is not a write, but it is a way to
        reach a second database from inside a read), and it is the only thing
        here that can say *which* action offended.
        """

        @event.listens_for(engine, "connect")
        def _posture(dbapi_conn, _record):  # noqa: ANN001 - SQLAlchemy's signature
            dbapi_conn.execute("PRAGMA query_only = ON")

            def authorize(action, arg1, _arg2, _database, _trigger):
                if action in _READ_ACTIONS:
                    return sqlite3.SQLITE_OK
                refusal.what = _describe(action, arg1)
                return sqlite3.SQLITE_DENY

            dbapi_conn.set_authorizer(authorize)

    # -- residency ---------------------------------------------------------

    def resident_bytes(self, engine: Engine) -> int | None:
        """How much this database is actually holding, right now.

        **Freelist-corrected, and that is the whole point.** ``page_count`` does
        not shrink when a table is dropped — 219 pages before and after, with 218
        of them free — so a database that was loaded and emptied would otherwise
        keep measuring at its high-water mark and be spilled for data it no
        longer holds. See ``docs/CONSTRAINTS.md`` §6.

        Measured rather than estimated. Deciding from file size or metadata what
        a load *will* cost is the fail-open pattern this project has already been
        bitten by; this asks the database what it *has*.
        """
        if not self.is_memory_resident(engine):
            return None
        with engine.connect() as conn:
            pages = conn.execute(text("PRAGMA page_count")).scalar_one()
            free = conn.execute(text("PRAGMA freelist_count")).scalar_one()
            size = conn.execute(text("PRAGMA page_size")).scalar_one()
        return max(pages - free, 0) * size

    @staticmethod
    def is_memory_resident(engine: Engine) -> bool:
        """Whether this database lives in our process rather than on disk.

        Read from ``url.query`` and not from ``url.database``: SQLAlchemy parses
        ``sqlite:///file:tag?mode=memory&cache=shared`` into a database of
        ``file:tag`` with the mode lifted into the query mapping, so looking for
        ``mode=memory`` in the database string finds nothing and reports a memory
        database as disk-backed. That answer fails in the dangerous direction —
        residency reads as "not applicable", the budget never counts the database,
        and it is never spilled however large it grows.
        """
        url = engine.url
        if url.query.get("mode") == "memory":
            return True
        return (url.database or "") in ("", ":memory:")

    # -- snapshotting ------------------------------------------------------

    def snapshot(self, engine: Engine, target: Path) -> None:
        """Write a consistent, compacted copy of this database to a file.

        ``VACUUM`` cannot run inside a transaction (``docs/CONSTRAINTS.md``
        §4.3), and SQLAlchemy begins one on first execute unless told otherwise —
        hence ``AUTOCOMMIT``, which is the execution option that means "this
        statement manages itself" rather than a way of avoiding transactions.

        The target must not exist; SQLite refuses rather than overwriting, and
        that refusal is worth keeping rather than working around.
        """
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql("VACUUM INTO ?", (str(target),))

    # -- storage classes ---------------------------------------------------

    def storage_classes(
        self, conn: Connection, table: str, column: str
    ) -> dict[str, int]:
        """Count actual storage classes present. Measured, not inferred.

        SQLite is why :attr:`loader.ColumnInfo.is_mixed` exists at all: a column
        may hold an integer in one row and text in the next, and an aggregate
        over it silently coerces the text to 0 while keeping it in the
        denominator.
        """
        rows = conn.execute(
            text(
                f'SELECT typeof("{column}") AS storage_class, count(*) AS n '
                f'FROM "{table}" GROUP BY 1'
            )
        ).all()
        return {storage_class: count for storage_class, count in rows}


_SQLITE = SQLiteBackend()
_GENERIC = Backend()

#: Dialect name to the backend that has something *extra* to say about it. An
#: absence here is not a gap: it means SQLAlchemy's own answers are the whole
#: answer for that database, which is the ordinary case rather than the
#: exceptional one.
BACKENDS: dict[str, Backend] = {"sqlite": _SQLITE}


def backend_for(dialect: str) -> Backend:
    """The backend that answers for a dialect. Never a refusal.

    An unknown dialect gets :data:`_GENERIC`, and that is the point: this server
    reaches whatever SQLAlchemy reaches, and no database has to be enumerated
    here to be usable. What an unregistered dialect loses is only the extras —
    residency reads ``None`` (not zero: unknown, not empty), ``save`` says
    plainly that there is no local database to write, and the storage-class
    histogram is empty because a real type system makes it meaningless.

    That is the honest kind of degradation. The fail-open shape this project has
    been bitten by is a *guess* presented as a measurement; ``None`` and an
    explicit refusal are neither.
    """
    return BACKENDS.get(dialect, _GENERIC)
