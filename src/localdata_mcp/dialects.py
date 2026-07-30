"""What each database backend has to supply, and what it gets for free.

A tag is a database, and a database is reached through a SQLAlchemy engine. Most
of what this server does above that line is ordinary Core — a table is created,
rows are inserted, a result is streamed, a schema is inspected — and none of it
needs to know which backend answered.

A few things genuinely cannot be said portably, and this module is where they are
said. The test for belonging here is not "is this SQL awkward" but **"does this
mean something different, or nothing at all, on another backend"**:

* **Read-only posture.** Every backend can refuse writes; none of them spell it
  the same way. SQLite sets ``query_only`` on the connection, MySQL opens a
  read-only session. The *guarantee* is portable, the mechanism is not — and on
  MySQL the generic guarantee is not even available, because DDL there commits
  implicitly and there is nothing left to roll back.
* **Residency.** How much a database is holding in *this process* is a question
  that only means anything for an in-process, memory-backed database. For a file
  or a server it is either unobservable or somebody else's memory, and the honest
  answer is ``None`` — not zero, which would read as "holding nothing".
* **Snapshotting.** Writing a consistent copy of a database to a file is
  ``VACUUM INTO`` on SQLite and something else everywhere else.
* **Storage classes.** SQLite lets a column hold values of different types and
  ``typeof()`` counts them. On a backend with real column types the question does
  not arise, and an empty histogram is the truthful answer rather than a gap.
* **What a ``CREATE TABLE`` must carry.** Most backends have a default table
  storage and a bare ``CREATE TABLE`` is complete. ClickHouse has none, and the
  statement does not compile at all without an engine clause.
* **Whether an index is a thing this verb can make.** Naming columns is the whole
  of an index on every backend whose indexes can afterwards be found by
  reflection. ClickHouse's cannot, and its secondary indexes answer a different
  question, so the truthful answer there is that the verb does not apply.

Everything else — creating tables, inserting, introspection via ``inspect()``,
streaming reads — is Core and lives in :mod:`loader`. When something new turns out
to need a per-backend answer, it earns an entry here; wrapping a statement in
``text()`` to change its transport is not that, and does not belong.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from sqlalchemy import (
    DOUBLE_PRECISION,
    INTEGER,
    REAL,
    TEXT,
    Connection,
    Double,
    Engine,
    Index,
    Integer,
    LargeBinary,
    String,
    Table,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.types import TypeEngine
from sqlalchemy.pool import StaticPool

__all__ = [
    "BACKENDS",
    "Backend",
    "ClickHouseBackend",
    "CrateDBBackend",
    "DuckDBBackend",
    "Engines",
    "FirebirdBackend",
    "MySQLBackend",
    "PostgreSQLBackend",
    "Refusal",
    "SQLiteBackend",
    "TrinoBackend",
    "UnsupportedOperation",
    "backend_for",
    "backend_for_url",
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


#: The declared type of a loaded column, to the portable Core type that renders
#: it. Keyed by the three names :mod:`loader` decides between; what each becomes
#: in SQL is SQLAlchemy's business and differs per dialect, which is exactly what
#: makes this the generic answer.
_PORTABLE_TYPES: dict[str, Any] = {"INTEGER": Integer, "REAL": Double, "TEXT": Text}


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


@dataclass(frozen=True)
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
    #:
    #: Carried as a field rather than fixed per class so that an *unregistered*
    #: dialect still knows its own name (:func:`backend_for` builds one). A
    #: refusal that says "a generic datasource" names our fallback rather than
    #: the database the caller opened, which is the same defect MariaDB being
    #: registered in its own right exists to avoid.
    name: str = "generic"

    #: How this dialect is told, **in the URL**, to open a file read-only —
    #: as query parameters, not as a string to append. Empty means it has no way
    #: of being told, and the generic transactional floor is the whole guarantee
    #: (see :meth:`read_posture`). A ``ClassVar`` because it is a property of the
    #: dialect, not of an instance.
    read_only_query: ClassVar[Mapping[str, str]] = {}

    #: Engines that answer on this dialect but are not the one it is named for,
    #: as a fragment of the version banner they carry mapped to the name each
    #: should be known by. Empty for a dialect only its own engine speaks.
    #:
    #: **A dialect name identifies a wire protocol and a driver, never an
    #: engine.** TiDB and OceanBase answer on MySQL's; YugabyteDB, Greenplum and
    #: OpenGauss on PostgreSQL's. Keyed by dialect alone, every one of them is
    #: handed the answers written for the engine whose dialect it borrowed —
    #: which is at best the wrong name in a refusal and at worst, where that
    #: engine has a subclass, behaviour it never asked for. Issue #45.
    #:
    #: **Every fragment here is read from a live server, never taken from
    #: documentation.** A guessed fragment fails in both directions: too loose
    #: and the real engine matches its own impostor, too tight and nothing does.
    #: An entry arrives when its container does.
    impostors: ClassVar[Mapping[str, str]] = {}

    #: What to ask for the version banner, where reading one can tell two
    #: engines apart. Empty alongside an empty :attr:`impostors`, and read only
    #: when that is not — so a dialect nobody shares is never asked anything,
    #: and no claim is made here about how the rest spell it.
    banner_query: ClassVar[str] = ""

    def named_by(self, banner: str | None) -> Backend:
        """This backend, or the one the server's own banner says is answering.

        Pure, and separate from reading the banner on purpose: which engine a
        version string names is the part worth pinning in a test, and it needs
        no database to pin. :func:`backend_for_url` does the connecting.

        ``None`` — the probe failed, or was never made — resolves to ``self``,
        which is precisely the behaviour before any of this existed. A
        datasource that will not answer its banner query is therefore no worse
        off than it was; refusing to open it because an *identity* probe failed
        would be very much worse than naming its dialect and carrying on.

        An impostor with an entry in :data:`BACKENDS` gets that entry, so an
        engine that has earned its own answers actually receives them. One
        without gets a plain :class:`Backend` carrying its own name, which is
        the whole of what an engine with nothing extra to say needs.
        """
        if not banner:
            return self
        folded = banner.lower()
        for fragment, name in self.impostors.items():
            if fragment.lower() in folded:
                return BACKENDS.get(name) or Backend(name=name)
        return self

    def open(self, url: str | URL, *, writable: bool) -> Engines:
        """Two engines onto one datasource — one reading, one writing.

        Generic because ``create_engine`` is generic: SQLAlchemy resolves the
        driver, the dialect and the connection arguments from the URL, and
        nothing here needs to know which database answered. ``writable`` is not
        consulted here because there is nothing portable to do with it — what
        the *write* engine may do is the datasource's own business, enforced by
        its own grants.

        The **read** engine carries :attr:`read_only_query` where the dialect
        has one, exactly as :meth:`open_file` does. A dialect that can be told
        in the URL to open read-only should be told so however it was reached:
        a server URL is not a weaker claim on the posture than a file path is,
        and on a backend with no transactions the URL is the *only* place the
        posture can be stated — there is no rollback floor underneath it.
        Where the mapping is empty nothing is added and nothing changes.

        Anything the *driver* has to be told that a URL cannot carry comes from
        :meth:`connect_args`, and is given to both engines: a read and a write
        onto one datasource must not disagree about how values are spelled.
        """
        refusal = Refusal()
        connect_args = dict(self.connect_args())
        engines = Engines(
            write=create_engine(url, connect_args=connect_args),
            read=create_engine(self._read_only(url), connect_args=connect_args),
            refusal=refusal,
        )
        self.read_posture(engines.read, refusal)
        return engines

    def connect_args(self) -> Mapping[str, Any]:
        """Driver arguments this dialect needs that a URL cannot express. None here.

        Distinct from :attr:`read_only_query`, which is part of the URL and says
        what the *connection may do*. This says what the driver must be handed —
        a Python object, not a string — and applies to reading and writing
        alike.

        Empty for every dialect whose driver already returns the types its
        columns declare, which is all of them but CrateDB.
        """
        return {}

    def _read_only(self, url: str | URL) -> URL:
        """``url`` with this dialect's read-only query parameters merged in.

        Through ``update_query_dict`` rather than by appending to a string,
        for the reason :meth:`open_file` gives: a URL is parsed, not
        concatenated, and a datasource whose URL already carries query
        parameters would otherwise gain a second ``?``.
        """
        parsed = make_url(url)
        if not self.read_only_query:
            return parsed
        return parsed.update_query_dict(dict(self.read_only_query))

    def open_file(self, path: Path, *, writable: bool) -> Engines:
        """Open a database that lives in a file.

        Generic because a file is reached by a URL like anything else:
        ``dialect:///absolute/path``, which is all SQLAlchemy needs. This exists
        as a method rather than as a branch in :mod:`loader` because the two
        things a file open can differ in — how the path becomes a URL, and how
        this dialect is told to open read-only — are both per-dialect facts, and
        a per-dialect fact stated in shared code is a dispatch on dialect name
        however it is spelled.

        ``writable=False`` carries :attr:`read_only_query` where the dialect has
        a way of being told. Where it has none, nothing is added and nothing is
        lost: the read engine still never commits, which is the floor
        :meth:`read_posture` documents.

        **Built through :meth:`URL.create` rather than by formatting a string.**
        A path is not a URL and interpolating one into a URL re-reads it as
        syntax: ``why? not.db`` splits at the ``?`` and the database becomes
        ``why``, which then fails to open with a message about the wrong file.
        ``URL.create`` takes the path as a *value*, so nothing in it is parsed.
        """
        url = URL.create(
            self.name,
            database=str(path.resolve()),
            query={} if writable else dict(self.read_only_query),
        )
        return self.open(url, writable=writable)

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

    def denies_write(self, exc: Exception) -> bool:
        """Whether this error is this backend's own refusal of a write.

        ``False`` generically, and truthfully so: the generic posture does not
        refuse a write, it declines to keep one, so there is no refusal to
        recognise. A dialect that *does* refuse answers here — by the driver's
        own error code, never by its prose, because prose carries a locale and a
        version and an error code carries neither.

        What this is for: the refusal reaches the caller as whatever the driver
        said, and a driver says "read only transaction" without saying which verb
        to use instead. :meth:`loader.Workspace._explain` asks this so it can
        answer with the one that does.
        """
        return False

    def ddl_survives_refusal(self) -> bool:
        """Whether DDL sent to a read connection takes effect despite refusal.

        ``False`` here, and true of every backend that either refuses DDL
        outright or keeps it inside the transaction that is never committed.
        Oracle is the exception: it commits DDL as it runs it, and the implicit
        commit ends the read-only transaction before the statement is
        considered, so there is nothing left to refuse *with*.

        Reported rather than papered over. A refusal that says a statement did
        not happen, when it did, is the same lie in the other direction as
        reporting a rolled-back write as a success.
        """
        return False

    def dml_survives_refusal(self) -> bool:
        """Whether *data* sent to a read connection takes effect despite refusal.

        The same admission as :meth:`ddl_survives_refusal`, asked about rows
        rather than about schema, and ``False`` everywhere the transactional
        floor exists — an ``INSERT`` on a connection that never commits is
        rolled back, so the refusal and the outcome agree.

        They come apart on a database with **no transactions at all**. ClickHouse
        has none either, but supplies a posture of its own (``readonly=1``) that
        refuses the statement before it reaches the data, so its answer is still
        ``False``. CrateDB has neither: no transaction to withhold and no
        read-only session to ask for, so a write reaches the data and the
        refusal that follows is a true statement about what this server *permits*
        and a false one about what happened.

        Split from ``ddl_survives_refusal`` rather than folded into it because
        the two are genuinely independent — Oracle's DDL survives while its DML
        does not — and a single axis would have to lie about one of them.
        """
        return False

    def settle(self, conn: Connection, table: str) -> None:
        """Make rows just written visible to the next read. Nothing, generically.

        Every transactional backend here has already done this by the time a
        write commits: the commit *is* the point rows become visible, so there is
        nothing left to ask for and this stays a no-op.

        It exists for the search-engine lineage, where a write is durable long
        before it is visible. CrateDB writes into a Lucene index refreshed on a
        timer, so a table counted immediately after an insert answers **0** and
        answers correctly a second later. Left alone, that turns every write into
        a race: ``insert_frame`` reports the row count it read, and the number it
        would report is whatever the timer happened to have done — which is not a
        flaky test so much as a payload that is wrong for a reason the caller
        cannot see.

        Waiting it out was the alternative and is worse: it trades a wrong answer
        for a slow one, and it picks a timeout by guessing at a setting the
        server is free to change.
        """
        return None

    def sees_new_tables_in_transaction(self) -> bool:
        """Whether a table created here can be written to before the DDL commits.

        ``True`` for every backend but one, and true for two different reasons
        that happen to agree. Oracle, MySQL, MariaDB and SQL Server commit DDL as
        they run it, so by the time the ``INSERT`` is prepared the table has been
        committed whether anyone asked for that or not; the rest keep the DDL
        inside the transaction *and* let the transaction see what it did.

        Firebird does neither. Its DDL is genuinely transactional — a
        ``CREATE TABLE`` on a connection that never commits leaves nothing behind,
        which is the floor working exactly as intended — but statements are
        prepared against *committed* metadata, so the table is invisible to the
        very transaction that made it. Measured: ``INSERT`` straight after
        ``CREATE`` on one connection fails with ``-204 Table unknown``, and the
        identical pair succeeds when the DDL commits in between.

        :meth:`loader.Workspace.insert_frame` asks this and splits its one write
        block into two where the answer is ``False``. **The split has a cost, and
        naming it here is the point of the docstring:** with the DDL committed
        first, a failure part-way through the rows leaves an empty table behind
        where the single transaction would have left no table at all. That is
        strictly worse and is accepted only because the alternative on this
        backend is that the write cannot happen at all.

        Asked as a question about *this* fact rather than folded into
        ``ddl_survives_refusal``: the two look adjacent and are opposites here.
        Firebird's DDL does not survive a refusal (``False``) and still cannot be
        used by its own transaction (``False`` here), so one axis carrying both
        would have to lie about one of them — the same reason ``dml_`` and
        ``ddl_survives_refusal`` are separate.
        """
        return True

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
        our own — ``create`` — is the route, and that slot saves.
        """
        raise UnsupportedOperation(
            f"A {self.name} datasource is reached over its own connection, not "
            f"held here, so there is no local database to write out. Copy the "
            f"rows you want into a slot of your own with create, and save that."
        )

    def renames_tables(self) -> bool:
        """Whether ``update(type='table', name=…, to=…)`` means anything here.

        ``True`` for every backend with any way at all to give an existing table
        a new name — by ``ALTER TABLE … RENAME TO``, or by whatever else it calls
        that, which is what an override of :meth:`rename_table` is for.

        ``False`` says the verb does not apply, and it exists for the same reason
        :meth:`builds_indexes` does: Firebird has **no rename-table statement of
        any kind**, in any version. Measured, not read — ``ALTER TABLE x RENAME TO
        y`` is rejected at ``-104 Token unknown … RENAME``.

        Faking it was the alternative and is refused. Copying the rows into a new
        table and dropping the old one reads like a rename and is not one: this
        method promises rows, types *and indexes*, and a copy keeps only the
        first. Handing back a name that is a table missing its indexes would be a
        lie the caller then builds on — the same judgement
        :meth:`snapshot` makes about a database this server does not hold.

        Nothing dispatches on this in shared code. It is here so that a *test*
        can tell "this backend cannot rename" apart from "renaming is broken",
        and because a dialect fact stated in a test fixture is the same defect as
        one stated in shared code.
        """
        return True

    def rename_table(self, conn: Connection, table: str, to: str) -> None:
        """Rename a table in place, keeping its rows, types and indexes.

        ``ALTER TABLE … RENAME TO …`` is the generic answer because it is the
        one every dialect this server targets accepts — SQLite, DuckDB,
        PostgreSQL, MySQL 8 and Oracle all take it verbatim. SQL Server is the
        known exception (``sp_rename``) and is exactly what an override of this
        method is for.

        Core has no rename construct — renaming is schema migration, Alembic's
        remit rather than Core's — so this is one of the few places that must
        state SQL. Both identifiers go through the dialect's own preparer rather
        than into an f-string, so a name needing quoting is quoted the way
        *this* database quotes it, and a name arriving from outside cannot
        become syntax.
        """
        prepare = conn.dialect.identifier_preparer.quote
        conn.execute(text(f"ALTER TABLE {prepare(table)} RENAME TO {prepare(to)}"))

    def folds_identifiers(self) -> bool:
        """Whether this backend renames a table to a case it chose itself.

        ``False`` for every backend that keeps a **quoted** identifier verbatim,
        which is all of them bar one: the whole reason
        :meth:`rename_table` goes through the dialect's own preparer is that an
        *unquoted* ``Mixed`` folds on half these dialects, and quoting is what
        stops it. Trino is the exception — it lower-cases every identifier at
        the connector, quoted or not, so a table asked for as ``Mixed`` is
        stored as ``mixed`` and there is nothing a preparer can do about it.

        Nothing dispatches on this. :meth:`loader.Workspace.landed_as` asks the
        database what the table ended up called and needs no dialect fact at
        all, which is the right way round — a name is observable, so observe it.
        This exists so a *test* can tell the two outcomes apart: asserting only
        that the reported name is findable would let a genuine folding
        regression through on PostgreSQL, and asserting case-insensitively
        would let all of them through. A dialect fact stated in a test fixture
        is the same defect as one stated in shared code, so it is stated here.
        """
        return False

    def column_type(self, declared: str, *, longest: int | None = None) -> TypeEngine:
        """The Core type a loaded column is created as, for this backend.

        ``longest`` is how many characters the widest value in the column
        actually has, measured rather than assumed, and ``None`` when the column
        holds no text. It is ignored generically — an unbounded text type needs
        no size — and used by the one dialect that has no unbounded text type
        worth having.

        Generic through SQLAlchemy's *portable* types, which is the whole point
        of them: ``Text`` is ``TEXT`` on PostgreSQL and ``CLOB`` on Oracle, and
        neither name has to appear here. The uppercase forms this used to name
        directly meant "emit this exact token", which is why a table could not be
        created on Oracle at all — it has no ``TEXT`` — and why a float64 column
        landed in PostgreSQL's ``REAL``, four bytes wide, losing precision with
        nothing said.

        ``Double`` rather than ``Float`` deliberately: the data is float64, and
        ``FLOAT`` is single precision on MySQL. A backend whose own spelling
        matters more than the portable one overrides this — SQLite does, because
        its three declared types decide column affinity.
        """
        return _PORTABLE_TYPES[declared]()

    def table_options(self) -> Mapping[str, Any]:
        """Dialect keyword arguments every ``CREATE TABLE`` here must carry.

        Empty generically, because on every backend with a default table
        storage a bare ``CREATE TABLE`` is a complete statement. ClickHouse is
        the exception this exists for: it has no default engine, so a table
        created without one is not a table it will make — the DDL does not fail
        at the database, it fails at compile time with nothing created.

        A mapping rather than a flag, because what has to be said is the
        dialect's own vocabulary and only its own compiler reads it. Shared code
        passes it through to ``Table(...)`` without knowing what is in it, which
        is what keeps this from being a dispatch on dialect name.
        """
        return {}

    def driver_errors(self) -> tuple[type[BaseException], ...]:
        """Failures from this driver that SQLAlchemy will not have wrapped.

        Empty generically, and empty for every well-behaved driver: SQLAlchemy
        catches whatever is a subclass of the DBAPI module's own ``Error`` and
        re-raises it as a ``SQLAlchemyError`` carrying the original on ``.orig``.
        Shared code catches that one type and every backend is covered.

        A driver whose exceptions are *not* subclasses of the ``Error`` it
        exports breaks that contract without saying so: nothing is wrapped, the
        failure arrives as the driver's own class with no ``.orig``, and every
        ``except SQLAlchemyError`` it passes through does not see it. Naming the
        types here is what puts such a driver back under the same handling —
        deliberately a named set rather than widening a guard to ``Exception``,
        which would pull unrelated failures into an explainer written for
        driver errors.
        """
        return ()

    def unstorable_column_types(self) -> frozenset[str]:
        """Core column types a value cannot make the round trip through here.

        Empty generically. This is not about the three types a *loaded file*
        produces — :meth:`column_type` answers for those, and every backend can
        hold all three. It is about the much wider value space of a table
        somebody else made and this server merely reaches, which is the only
        kind an endpoint ever is: Oracle has no time-of-day type at all, and
        ClickHouse's driver can bind neither ``bytes`` nor a ``time``.

        "Cannot hold" covers the way in as well as the column itself: a type
        whose column is created and then refuses every value belongs here too,
        because what a caller can do with it is the same as if it did not exist.

        Named as strings rather than as classes so that stating one costs no
        import, and so a backend can name a type this module never mentions.
        """
        return frozenset()

    def builds_indexes(self) -> bool:
        """Whether ``create(type='index')`` means anything on this backend.

        ``True`` for every backend whose indexes are created by naming columns
        and can afterwards be found by reflection — which is what ``info``
        listing one and ``drop`` removing one both depend on.

        ``False`` says the whole verb does not apply here, and it is reported
        rather than faked for the same reason :meth:`snapshot` refuses: an index
        that cannot be listed or dropped, created under a name the caller is
        handed, would be a lie the caller then builds on. See
        :class:`ClickHouseBackend`.
        """
        return True

    def build_index(
        self, name: str, table: Table, columns: Sequence[str]
    ) -> tuple[Index, tuple[str, ...]]:
        """The index to create, and anything the caller should be told about it.

        Generic because Core's ``Index`` is generic: naming the columns is the
        whole of it on every dialect that can index a column outright. A dialect
        that cannot returns an index built the way it *can* be, together with the
        words for what that cost — an index that covers less than it was asked to
        is a fact about the answer, not an implementation detail.
        """
        return Index(name, *[table.c[column] for column in columns]), ()

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


#: SQLite's three storage classes, named exactly. ``INTEGER``/``REAL``/``TEXT``
#: are what set a column's affinity, so these are the spellings that have to
#: reach the DDL rather than whatever a portable type would render.
_SQLITE_TYPES: dict[str, Any] = {"INTEGER": INTEGER, "REAL": REAL, "TEXT": TEXT}


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

    # -- column types ------------------------------------------------------

    def column_type(self, declared: str, *, longest: int | None = None) -> TypeEngine:
        """The exact token, because on SQLite the token *is* the behaviour.

        A column's declared type decides its affinity, and affinity decides
        whether ``'42'`` and ``42`` compare equal — so ``REAL`` and ``DOUBLE``
        are not two spellings of one thing here, and neither is what ``info``
        reports the column to be. The portable types are right everywhere the
        declaration is only a declaration; this is the one place it is not.
        """
        return _SQLITE_TYPES[declared]()

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

        ``VACUUM INTO`` takes an *expression*, so the destination binds like any
        other value — through ``text()`` and Core's own parameter style, rather
        than through the driver's ``?``. Reaching for the driver's paramstyle
        where Core has one is the kind of small bypass this module's docstring
        warns about; it also ties this statement to pysqlite specifically for no
        gain.
        """
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("VACUUM INTO :target"), {"target": str(target)})

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


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuckDBBackend(Backend):
    """DuckDB, which can be *told* to open a file read-only.

    The one thing it adds, and the reason it is registered at all: a read-only
    posture carried by the URL. ``access_mode=read_only`` is refused by DuckDB
    itself at open time, so a read connection cannot write however it is reached
    — the same shape as SQLite's ``mode=ro``, and stronger than the generic
    transactional floor, which allows the write and then discards it.

    Everything else here is still the generic answer. This class deliberately
    holds one fact; it used to live in :mod:`loader` as a dictionary keyed by
    dialect name, which is a dispatch on dialect name wearing a different hat.
    """

    name: str = "duckdb"
    read_only_query: ClassVar[Mapping[str, str]] = {"access_mode": "read_only"}


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostgreSQLBackend(Backend):
    """PostgreSQL, which adds nothing — and is registered to say who borrows it.

    Every answer here is the generic one, and that is the finding rather than an
    omission: five sessions of endpoint work have not turned up a single thing
    PostgreSQL needs said for it. This class exists for the *other* reason a
    dialect earns an entry — several engines answer on it and are not it.

    ``-YB-`` is read from a live YugabyteDB, not from its documentation:

    * YugabyteDB — ``PostgreSQL 15.12-YB-2.25.2.0-b0 on x86_64-pc-linux-gnu…``
    * PostgreSQL — ``PostgreSQL 16.14 on x86_64-pc-linux-musl…``

    The fragment is deliberately tighter than the obvious one. YugabyteDB's
    banner says ``yugabyte`` twice, once in the compiler's source URL, so
    matching that would work — but it is the *version* that identifies the
    engine, and a fragment that leans on a build detail is a fragment waiting to
    stop matching. ``-YB-`` is the part PostgreSQL itself can never carry.

    Greenplum and OpenGauss belong here too and are absent on purpose: neither
    has a container yet, so neither has a measured banner, and an entry guessed
    from documentation is the failure mode the whole table is written to avoid.
    """

    name: str = "postgresql"
    banner_query: ClassVar[str] = "SELECT version()"
    impostors: ClassVar[Mapping[str, str]] = {"-YB-": "yugabytedb"}


# ---------------------------------------------------------------------------
# MySQL and MariaDB
# ---------------------------------------------------------------------------


#: MySQL's own code for "this statement cannot run in a read-only transaction",
#: shared by MariaDB. Matched on the code rather than the sentence: the sentence
#: has a locale and a version, the code has neither.
_MYSQL_READ_ONLY = 1792

#: InnoDB's limit on the total size of an index key, in bytes. The published
#: figure for the default page size, not an estimate.
_INNODB_KEY_BYTES = 3072

#: How much of an unbounded text column to index when the whole of it cannot be.
#: The conventional MySQL prefix, and comfortably inside the key limit for a
#: single column; several columns divide the limit between them instead.
_TEXT_PREFIX_CHARS = 255


@dataclass(frozen=True)
class MySQLBackend(Backend):
    """MySQL and MariaDB, where the transactional floor is not a floor.

    **DDL on MySQL commits implicitly.** A ``CREATE TABLE`` sent through a read
    connection that never commits is therefore *permanent* — measured against
    both containers, and the table was still there on the next connection. The
    generic guarantee assumes a write can be left uncommitted and thereby undone;
    that assumption simply does not hold here, so this raises the floor from "not
    kept" to "not run": a read-only session refuses DML and DDL alike, before
    either reaches the data.

    Set once, at connect time, for the same reason SQLite's ``query_only`` is:
    a posture toggled around a call has a window in which it is something else.
    """

    name: str = "mysql"

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        @event.listens_for(engine, "connect")
        def _posture(dbapi_conn, _record):  # noqa: ANN001 - SQLAlchemy's signature
            with dbapi_conn.cursor() as cursor:
                cursor.execute("SET SESSION TRANSACTION READ ONLY")

    def denies_write(self, exc: Exception) -> bool:
        origin = getattr(exc, "orig", exc)
        arguments = getattr(origin, "args", ())
        return bool(arguments) and arguments[0] == _MYSQL_READ_ONLY

    def build_index(
        self, name: str, table: Table, columns: Sequence[str]
    ) -> tuple[Index, tuple[str, ...]]:
        """Index a prefix of any column MySQL will not index whole.

        ``TEXT`` and ``BLOB`` have no length in the row, so MySQL refuses to key
        on them at all without being told how much to key on — error 1170. Every
        text column a loaded file produces is ``TEXT``, so without this,
        ``create(type='index')`` simply does not work on this dialect.

        The prefix is derived, not chosen: InnoDB's key limit is 3072 bytes and
        ``utf8mb4`` costs four per character, so the columns needing a prefix
        share that budget, capped at the 255 characters that are conventional
        and ample. A prefix index still answers a query over the whole value —
        MySQL narrows on the prefix and rechecks the rest — so what it costs is
        selectivity, and that is what the note is for.
        """
        unbounded = [
            column
            for column in columns
            if isinstance(table.c[column].type, (Text, LargeBinary))
        ]
        if not unbounded:
            return super().build_index(name, table, columns)

        prefix = max(
            1, min(_TEXT_PREFIX_CHARS, _INNODB_KEY_BYTES // (4 * len(unbounded)))
        )
        index = Index(
            name,
            *[table.c[column] for column in columns],
            **{f"{self.name}_length": dict.fromkeys(unbounded, prefix)},
        )
        named = ", ".join(unbounded)
        return index, (
            f"{self.name} cannot key on a whole TEXT or BLOB column, so {named} "
            f"{'are' if len(unbounded) > 1 else 'is'} indexed on the first "
            f"{prefix} characters. A lookup on the whole value still uses the "
            f"index; two rows agreeing on that prefix simply do not narrow it "
            f"any further.",
        )


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


#: The widest ``VARCHAR2`` a database without extended string sizes will take.
#: Past this there is only ``CLOB``, which Oracle will not group, sort or index.
_VARCHAR2_MAX = 4000


@dataclass(frozen=True)
class OracleBackend(Backend):
    """Oracle, where the portable text type is the wrong one.

    ``Text`` renders as ``CLOB`` here, and a ``CLOB`` cannot be used as a
    comparison key — ``GROUP BY``, ``ORDER BY``, ``DISTINCT`` and every index on
    it are refused with ORA-22848. A loaded file's text columns are the ones an
    agent groups and joins on, so the portable answer, correct everywhere else,
    makes the table nearly unusable here. ``VARCHAR2`` sized from what the column
    actually holds is what Oracle wants, and it is measured rather than guessed
    because the frame is right there.
    """

    name: str = "oracle"

    def column_type(self, declared: str, *, longest: int | None = None) -> TypeEngine:
        if declared != "TEXT":
            return super().column_type(declared, longest=longest)
        width = max(1, longest or 1)
        if width > _VARCHAR2_MAX:
            # Nothing else will hold it. The column loses grouping and
            # indexing, and Oracle says so plainly the first time it is used
            # that way, which is better than truncating the values to fit.
            return Text()
        return String(width)

    def unstorable_column_types(self) -> frozenset[str]:
        """Oracle has no time-of-day type.

        A bare time is a ``DATE`` with the date part ignored, or an
        ``INTERVAL`` — neither of which is the column that was asked for, so
        there is nothing a value could be put into and come back out of.
        """
        return frozenset({"Time"})

    def ddl_survives_refusal(self) -> bool:
        """Oracle commits DDL as it runs it, and nothing here can get in first.

        The generic transactional floor is kept rather than raised, which is
        the opposite of the decision MySQL got, and it is the measurement that
        decides it. Oracle *does* roll DML back, so the floor holds for
        everything it can hold for. The obvious way to raise it —
        ``SET TRANSACTION READ ONLY`` — takes a read-consistent snapshot, and a
        read-only transaction then refuses to read a table whose definition
        changed within the same second (ORA-01466). That is exactly the compose
        arc: ``create`` a table, then query it. It would have broken the
        ordinary path to tighten a guarantee that already holds by rollback.

        The DDL gap it would not have closed anyway: the implicit commit runs
        *before* the statement is considered and ends the read-only transaction,
        so there is nothing to refuse with. Hence this, and the caveat the
        refusal carries.
        """
        return True


# ---------------------------------------------------------------------------
# SQL Server
# ---------------------------------------------------------------------------


#: The widest ``VARCHAR`` SQL Server will take before ``VARCHAR(max)``, which is
#: a large object and cannot be part of an index key.
_VARCHAR_MAX = 8000


@dataclass(frozen=True)
class MSSQLBackend(Backend):
    """SQL Server, which renames tables by procedure and has no usable ``TEXT``.

    Two things it cannot be asked in the portable way, and both are the kind
    this module exists for — a statement that means something else here, rather
    than one that is merely awkward.
    """

    name: str = "mssql"

    def column_type(self, declared: str, *, longest: int | None = None) -> TypeEngine:
        """``VARCHAR`` sized from the data, because ``TEXT`` is a dead end here.

        SQLAlchemy's portable ``Text`` renders ``TEXT``, which SQL Server has
        deprecated and will not let into an index key at all — so a loaded file's
        text columns could be neither indexed nor, on the way out, relied on. A
        sized ``VARCHAR`` is the modern spelling and is indexable while it stays
        inside the 900-byte key limit, which is the caller's to stay inside.
        """
        if declared != "TEXT":
            return super().column_type(declared, longest=longest)
        width = max(1, longest or 1)
        # Past 8000 the only spelling is VARCHAR(max), which is a large object
        # again — the same trade Oracle makes at 4000, and for the same reason.
        return String(width) if width <= _VARCHAR_MAX else String()

    def rename_table(self, conn: Connection, table: str, to: str) -> None:
        """``sp_rename``, because SQL Server has no ``ALTER TABLE … RENAME``.

        The generic ``ALTER TABLE … RENAME TO`` is a syntax error here, and the
        transaction it fails in is then doomed — so the caller saw not "no such
        syntax" but a rollback complaining about a transaction that no longer
        existed. Both names are bound as parameters rather than interpolated:
        ``sp_rename`` takes them as strings, so there is nothing to quote and
        nothing that could become syntax.
        """
        conn.execute(
            text("EXEC sp_rename :existing, :wanted"), {"existing": table, "wanted": to}
        )


# ---------------------------------------------------------------------------
# ClickHouse
# ---------------------------------------------------------------------------


#: ClickHouse's own code for "this query cannot run in readonly mode", raised
#: for DML and DDL alike. Matched on the code for the reason MySQL's is: a
#: sentence carries a locale and a version and a code carries neither.
_CLICKHOUSE_READ_ONLY = 164

#: What a table loaded from a file is ordered by. ``tuple()`` is ClickHouse's
#: spelling for *no ordering key*, and it is the honest one here: a file has no
#: natural key, and inventing one out of the first column would silently decide
#: the physical layout — and the primary index — on the caller's behalf.
_NO_ORDERING_KEY = "tuple()"


@dataclass(frozen=True)
class ClickHouseBackend(Backend):
    """ClickHouse, which has no transactions and therefore no floor to stand on.

    Every other backend here either refuses a write or declines to keep it. This
    one does neither by default: there is no transaction to leave uncommitted, so
    an ``INSERT`` sent through a read connection is simply *applied* — measured,
    and the row was still there on the next connection. The generic guarantee is
    not weakened here, it is absent, so the posture has to come from the database
    itself. ``readonly=1`` in the URL is what supplies it, and it refuses DML and
    DDL alike before either reaches the data.

    That makes ClickHouse the first backend where :attr:`read_only_query` is
    load-bearing for a *server* rather than a file, which is why
    :meth:`Backend.open` now carries it.

    Reached over HTTP through the dialect that ships inside ``clickhouse-connect``
    as ``clickhousedb``. The third-party ``clickhouse-sqlalchemy`` is a different
    project and not the live one; ``docs/CONSTRAINTS.md`` §11.1 records how that
    was established.
    """

    name: str = "clickhousedb"
    read_only_query: ClassVar[Mapping[str, str]] = {"readonly": "1"}

    def denies_write(self, exc: Exception) -> bool:
        """Recognise ClickHouse's own refusal, by code rather than by prose.

        The driver puts the server's error code on the exception as ``code``, so
        there is nothing to parse out of the message — which is what the base
        class asks for and the reason it asks: this same refusal renders with a
        version string and a URL in it, both of which move.
        """
        origin = getattr(exc, "orig", exc)
        return getattr(origin, "code", None) == _CLICKHOUSE_READ_ONLY

    def driver_errors(self) -> tuple[type[BaseException], ...]:
        """This driver's failures reach us unwrapped, so name them.

        ``clickhouse_connect`` exports an ``Error`` from its DBAPI module as PEP
        249 requires, but the exceptions it actually raises do not inherit from
        it — ``driver.exceptions.ClickHouseError`` and
        ``dbapi.Error`` are unrelated classes. SQLAlchemy therefore never
        recognises a ClickHouse failure as a DBAPI error and re-raises it
        untouched, so without this every refusal reached the caller as the
        driver's raw sentence instead of the words naming the verb to use
        instead. See ``docs/CONSTRAINTS.md`` §11.3.
        """
        from clickhouse_connect.driver.exceptions import ClickHouseError

        return (ClickHouseError,)

    def table_options(self) -> Mapping[str, Any]:
        """Every table needs an engine, and ``MergeTree`` is the one to give it.

        ClickHouse has no default table engine, and the dialect's DDL compiler
        raises rather than guessing — so without this, ``create`` does not
        produce a failed statement, it produces no statement at all.

        ``MergeTree`` because it is the ordinary storage engine and the only one
        that supports what the rest of this server then does with a table.
        Imported here rather than at module scope so that installing the server
        without the ``clickhouse`` extra still imports this module — which is the
        same reason every other driver stays out of the import list.
        """
        from clickhouse_connect.cc_sqlalchemy.engines import MergeTree

        return {"clickhousedb_engine": MergeTree(order_by=_NO_ORDERING_KEY)}

    def column_type(self, declared: str, *, longest: int | None = None) -> TypeEngine:
        """``Nullable`` on every column, because a loaded file has gaps.

        A ClickHouse column is ``NOT NULL`` unless it says otherwise, and the
        portable types render as aliases that inherit that — so a CSV with an
        empty cell cannot be stored in one. **The failure is worse than a
        refusal in one direction:** a single-row insert of ``None`` into a
        non-nullable ``String`` stores the empty string and reports success, so
        a missing value silently becomes a present one; the same ``None`` inside
        a multi-row batch raises instead. Measured both ways — see
        ``docs/CONSTRAINTS.md`` §11.2.

        Wrapping in ``Nullable`` is what makes the two agree, and it makes them
        agree on the truthful answer: a missing value comes back missing, and an
        aggregate skips it rather than counting an empty string as a value.
        """
        from clickhouse_connect.cc_sqlalchemy.types import (
            Float64,
            Int64,
            Nullable,
            String,
        )

        return Nullable({"INTEGER": Int64, "REAL": Float64, "TEXT": String}[declared])

    def unstorable_column_types(self) -> frozenset[str]:
        """Two the driver cannot bind — and both are the driver's gap, not the
        database's.

        ``LargeBinary``: ClickHouse stores binary perfectly well, ``String``
        holds arbitrary bytes. What is missing is in
        ``clickhouse_connect.dbapi``, which does not define the ``Binary``
        constructor PEP 249 requires; SQLAlchemy's bind processor calls it and
        gets an ``AttributeError`` before any statement is sent.

        ``Time``: the column is created — ``TIME`` is a real ClickHouse type,
        stored as an integer number of seconds — but a Python ``time`` reaches
        the server as the bare literal ``14:30:00`` where an ``Int64`` was
        expected, and the insert fails to parse. So the column exists and cannot
        be written to, which is worse than not having it.

        Stated as the driver's gaps rather than the database's so that nobody
        later "fixes" ClickHouse for them. See ``docs/CONSTRAINTS.md`` §11.4.
        """
        return frozenset({"LargeBinary", "Time"})

    def rename_table(self, conn: Connection, table: str, to: str) -> None:
        """``RENAME TABLE``, which is a statement of its own here.

        ``ALTER TABLE … RENAME TO`` is not merely unsupported — ClickHouse parses
        ``ALTER TABLE … RENAME`` as the start of ``RENAME COLUMN`` and fails at
        the ``TO``, so the generic spelling produces a syntax error naming a
        clause the caller never wrote. Both identifiers go through the dialect's
        own preparer, as the generic implementation's do.
        """
        prepare = conn.dialect.identifier_preparer.quote
        conn.execute(text(f"RENAME TABLE {prepare(table)} TO {prepare(to)}"))

    def builds_indexes(self) -> bool:
        return False

    def build_index(
        self, name: str, table: Table, columns: Sequence[str]
    ) -> tuple[Index, tuple[str, ...]]:
        """Refused, because ClickHouse's index is not this kind of index.

        Three separate things fail here and only the first is about syntax.
        ``CREATE INDEX`` without a ``TYPE`` is refused outright (code 80). What
        ClickHouse does have — ``ALTER TABLE … ADD INDEX … TYPE minmax`` — is a
        *data-skipping* index: it prunes granules that cannot match, and it
        offers neither the point lookup nor the uniqueness a caller asking for an
        index is asking for. And the dialect reflects no indexes at all, so one
        created that way could not afterwards be listed by ``info`` nor found by
        ``drop`` — measured, ``system.data_skipping_indices`` shows it while
        reflection returns nothing.

        Creating one anyway, under a name handed back to the caller, would
        produce exactly the shape this server refuses elsewhere: an answer that
        reads as done and cannot be acted on. So this says what is true, and
        names the thing that actually orders a ClickHouse table.
        """
        raise UnsupportedOperation(
            f"A {self.name} table is indexed by the ordering key it was created "
            f"with, not by adding an index afterwards. Its secondary indexes are "
            f"data-skipping indexes, which cannot be listed or dropped through "
            f"this interface and do not answer a lookup the way an index does. "
            f"To make {', '.join(columns)} fast to filter on, create the table "
            f"ordered by those columns in {self.name} itself, and attach it here."
        )


# ---------------------------------------------------------------------------
# Trino
# ---------------------------------------------------------------------------


#: Trino's own name for "this catalog will not write except in autocommit",
#: raised for DML and DDL alike once a connection is out of autocommit. Matched
#: on the name the server sends rather than on the sentence, for the reason
#: MySQL's and ClickHouse's numeric codes are: a sentence carries a locale and a
#: version, an error name carries neither.
_TRINO_AUTOCOMMIT_WRITE = "AUTOCOMMIT_WRITE_CONFLICT"

#: What the read engine is set to, and the reason is availability rather than
#: strictness — see :meth:`TrinoBackend.read_posture`. Anything other than
#: autocommit would do; this is the only one that can be reached.
_TRINO_ISOLATION = "SERIALIZABLE"


@dataclass(frozen=True)
class TrinoBackend(Backend):
    """Trino, which is a query engine rather than a database.

    It owns no storage. Every table it can see belongs to a *catalog* — a
    configured connector onto some other system — so several of the questions
    this seam asks have answers that are about Trino's position in the stack
    rather than about a feature it lacks:

    * **It has no indexes at all**, and not as an omission. Trino's speed comes
      from pushing predicates down into the catalog, so what makes a column fast
      to filter on is how the *underlying* system stores it. There is no
      ``CREATE INDEX`` to fail; the verb does not apply.
    * **It folds every identifier to lower case**, quoted or not, at the
      connector rather than in the parser. Both spellings still resolve, so
      nothing breaks — but a table asked for as ``Mixed`` is called ``mixed``,
      and no preparer can prevent it.
    * **Its driver defaults to autocommit**, which is what makes
      :meth:`read_posture` load-bearing here rather than a refinement.
    """

    name: str = "trino"

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        """Take the read engine out of autocommit, so there is a floor at all.

        The generic guarantee is that a read connection never commits and
        whatever it changed is therefore rolled back. **This driver connects in
        ``AUTOCOMMIT`` by default**, which does not weaken that guarantee so
        much as delete it: every statement commits itself as it runs, so an
        ``INSERT`` sent through the read connection was simply *applied* —
        measured, and the row was still there on the next connection. The same
        shape ClickHouse has, arrived at from the opposite direction: there the
        database has no transactions, here it has them and the driver declines
        to use them.

        Naming any real isolation level restores the floor, and what happens
        next depends on the catalog rather than on this code. A catalog that
        writes transactionally accepts the statement and has it rolled back —
        the generic floor, working. A catalog that writes only in autocommit —
        ``memory``, which the test harness uses — refuses it outright with
        :data:`_TRINO_AUTOCOMMIT_WRITE`, which :meth:`denies_write` then
        recognises. Both are safe; the default is the one that is not.

        ``SERIALIZABLE`` is not a strictness decision. It is the **only** level
        SQLAlchemy can hand this dialect: SQLAlchemy normalises an isolation
        level to spaces (``READ UNCOMMITTED``) and the dialect looks it up in an
        enum keyed with underscores (``READ_UNCOMMITTED``), so every level whose
        name has two words raises ``KeyError`` at connect time. The one-word
        name is the one that survives the round trip. If that is ever fixed
        upstream the weakest level becomes reachable and is the better choice,
        since nothing here wants a stricter snapshot — only a transaction.
        ``docs/CONSTRAINTS.md`` §16.2 records the measurement.

        Set on the engine rather than per call, for the reason SQLite's
        ``query_only`` and MySQL's read-only session are: a posture toggled
        around a statement has a window in which it is something else.
        """
        engine.update_execution_options(isolation_level=_TRINO_ISOLATION)

    def denies_write(self, exc: Exception) -> bool:
        """Recognise the catalog's refusal, by the name the server sent.

        Only catalogs that cannot write inside a transaction raise this; one
        that can will have accepted the write and had it rolled back, where the
        generic answer of ``False`` is the correct one and there is no refusal
        to recognise.
        """
        origin = getattr(exc, "orig", exc)
        return getattr(origin, "error_name", None) == _TRINO_AUTOCOMMIT_WRITE

    def folds_identifiers(self) -> bool:
        return True

    def unstorable_column_types(self) -> frozenset[str]:
        """One, and it is the driver's gap rather than Trino's.

        Trino has ``VARBINARY`` and stores binary perfectly well. What is
        missing is in ``trino.dbapi``, whose literal formatter calls ``.encode``
        on the value it was given — which is what you do to a ``str``, so a
        ``bytes`` raises ``AttributeError`` before any statement is sent. The
        column is created and cannot be written to, which is worse than not
        having it.

        Stated as the driver's gap so that nobody later "fixes" Trino for it.
        """
        return frozenset({"LargeBinary"})

    def builds_indexes(self) -> bool:
        return False

    def build_index(
        self, name: str, table: Table, columns: Sequence[str]
    ) -> tuple[Index, tuple[str, ...]]:
        """Refused, because Trino has no indexes to build — by design, not by gap.

        Unlike ClickHouse, which has an index of a different kind, Trino has
        none whatsoever: it holds no data, so there is nothing of its own to
        index. A filter is made fast by being pushed down into the catalog,
        where the underlying system's own layout — partitioning, sorting, its
        real indexes — decides what it costs. Creating something here under a
        name handed back to the caller would be an answer that reads as done and
        cannot be acted on, which is what this refuses.
        """
        raise UnsupportedOperation(
            f"A {self.name} datasource holds no data of its own, so it has no "
            f"indexes to create — a filter is made fast by the catalog it reads "
            f"through, not here. To make {', '.join(columns)} fast to filter on, "
            f"index or partition {table.name} in the system behind the catalog, "
            f"and query it through {self.name} as before."
        )


# ---------------------------------------------------------------------------
# CrateDB
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrateDBBackend(Backend):
    """CrateDB, where a write is durable before it is visible and cannot be undone.

    A distributed SQL layer over Lucene, and the first backend here from the
    search-engine lineage rather than the database one. Two of its properties
    follow from that ancestry and neither has a precedent in this seam.

    **There are no transactions, and unlike ClickHouse there is no posture to put
    in their place.** ClickHouse has no transactions either and answers
    ``readonly=1``, which refuses a statement before it reaches the data.
    CrateDB offers no session-level equivalent — its read-only setting is a
    cluster-wide block that would stop the *write* engine too — so a write sent
    through a read connection is simply applied. Measured: an ``INSERT`` on a
    connection that never commits was still there afterwards, and so was a
    ``CREATE TABLE``. Both survivals are declared rather than papered over.

    **A committed write is not immediately readable.** Rows land in a Lucene
    index refreshed on a timer, so a count taken straight after an insert
    answers ``0`` and answers ``1`` about a second later. :meth:`settle` asks for
    the refresh explicitly rather than waiting for it.

    Reached over the HTTP endpoint on 4200 through Crate.io's own dialect. The
    same server also speaks the PostgreSQL wire on 5432; addressing it that way
    would load PostgreSQL's dialect and answer PostgreSQL's questions, which is
    the mistake ``docs/CONSTRAINTS.md`` §12 records for CockroachDB.
    """

    name: str = "crate"

    def connect_args(self) -> Mapping[str, Any]:
        """The driver's own type converter, without which a date arrives as a number.

        CrateDB's HTTP protocol carries values untyped and the column types
        beside them, so the driver only spells a value as a Python object if it
        is asked to. Unasked, a ``TIMESTAMP`` reaches the caller as
        ``1709251200000`` — epoch milliseconds, an integer JSON is perfectly
        happy to carry and an agent will read as a quantity. That is the exact
        failure ``test_every_value_reaches_the_wire_as_something_json_can_hold``
        exists to catch, and the reason dates are canonical ISO 8601 text here.

        Not reachable through SQLAlchemy's own typing, which is why it is set on
        the driver. A Core ``select()`` over a reflected table converts correctly
        because SQLAlchemy knows the column types; ``query`` runs the caller's
        own text, where it knows nothing and the DBAPI cursor description
        supplies no type codes at all — every field is ``None``. The converter is
        the driver's public answer to that, and it works off the ``col_types``
        the server sends.
        """
        from crate.client.converter import DefaultTypeConverter

        return {"converter": DefaultTypeConverter()}

    def dml_survives_refusal(self) -> bool:
        """``True``, and it is the honest answer rather than a resigned one.

        No transaction to withhold and no read-only session to ask for, so the
        row is in the index by the time this server has anything to say about
        it. The refusal that follows is true about what is permitted here and
        false about what happened, and a caller told otherwise would go looking
        for a row that exists.
        """
        return True

    def ddl_survives_refusal(self) -> bool:
        """``True``, for the same reason and measured the same way.

        Oracle reaches this state by committing DDL implicitly; CrateDB reaches
        it by never having had a transaction. The consequence is identical, which
        is why the axis is asked rather than the dialect named.
        """
        return True

    def settle(self, conn: Connection, table: str) -> None:
        """``REFRESH TABLE``, so a row just written can be read back.

        The default refresh interval is a second, which is *fast* and entirely
        beside the point: the question is not how long the wrong answer lasts
        but whether this server ever gives one. ``insert_frame`` counts the rows
        it just wrote and puts that number in the payload, so without this the
        number reported is whatever the timer had done by then — 0 on a fast
        machine, correct on a slow one, and no way for the caller to tell which
        they were handed.

        Run on the same connection as the insert, inside the same block, so a
        write and the visibility of that write cannot be separated by a failure.
        """
        conn.execute(
            text(f"REFRESH TABLE {conn.dialect.identifier_preparer.quote(table)}")
        )

    def unstorable_column_types(self) -> frozenset[str]:
        """Three, and the third is the dangerous one because it does not fail.

        ``LargeBinary`` and ``Time`` are the database's own absences, unlike
        ClickHouse's two of the same name which are its *driver's*: here
        ``CREATE TABLE`` is refused at parse time — ``Cannot find data type:
        blob``, ``Cannot find data type: time`` — so the column is never made and
        there is nothing to write to. CrateDB stores binary as a base64
        ``STRING`` and time-of-day inside a ``TIMESTAMP``; neither is what the
        portable type means, and substituting one would put a value in a column
        whose type says something else.

        ``Numeric`` is here for the opposite reason: **nothing fails.** The
        dialect renders it as ``BIGINT``, so ``Decimal("12345.6789")`` is stored
        as ``12345`` and read back as ``Decimal("12345.0000")`` — the scale is
        re-applied on the way out by SQLAlchemy's own type, which is what makes
        the loss invisible. Measured against the alternative: a column declared
        ``NUMERIC(10,2)`` in raw SQL holds ``1.25`` exactly, so **the database
        supports the type and the dialect does not use it**. Stated as the
        dialect's defect so nobody later fixes CrateDB for it — issue #52.

        A column that silently corrupts is worse than one that cannot be created,
        which is the same judgement ClickHouse's ``Time`` entry records. Nothing
        this server writes reaches it — ``_declared_type`` emits only
        ``INTEGER``, ``REAL`` and ``TEXT`` — so the exposure is a caller's own
        table, and naming it here is what keeps the harness from asserting a
        value it would silently be handed wrong.
        """
        return frozenset({"LargeBinary", "Time", "Numeric"})

    def builds_indexes(self) -> bool:
        return False

    def build_index(
        self, name: str, table: Table, columns: Sequence[str]
    ) -> tuple[Index, tuple[str, ...]]:
        """Refused, because CrateDB has already done it.

        ``CREATE INDEX`` is not merely unsupported, it is unparseable — ``no
        viable alternative at input 'CREATE INDEX'`` — and that is the design
        rather than a gap: every column is indexed on write unless the table
        says ``INDEX OFF``. There is no index to add because there is no column
        without one.

        This is the third user of :meth:`builds_indexes`, after ClickHouse and
        Trino, and the three refuse for three different reasons — an index of
        another kind, no data to index, and an index already there. The axis
        carries the fact; only the sentence differs.
        """
        raise UnsupportedOperation(
            f"A {self.name} datasource indexes every column as it is written, so "
            f"there is no index to add — {', '.join(columns)} on {table.name} is "
            f"already fast to filter on. Creating one here would report work that "
            f"was never done."
        )


#: The widest ``VARCHAR`` this backend asks Firebird for, in **characters**.
#:
#: Firebird's limit is 32,765 *bytes*, and the two only coincide on a single-byte
#: character set. A database created with UTF8 spends up to four bytes a
#: character, so the same declaration that fits one database is refused by
#: another — and which one the caller has is not something this server chooses or
#: can see cheaply. 8,191 is the widest width that fits the byte limit under
#: *every* character set (8191 × 4 = 32,764), so the declaration cannot fail for a
#: reason that depends on how somebody else created their database.
#:
#: Deliberately not the measured 32,765: that number is true of the test
#: container, which is charset ``NONE``, and using it would be a measurement from
#: one configuration presented as a property of the engine.
_FIREBIRD_VARCHAR_MAX = 8191


@dataclass(frozen=True)
class FirebirdBackend(Backend):
    """Firebird, whose transactions are stricter than anything else here.

    The oldest engine in this catalogue and the only one registered under a
    dialect named after a **driver**. ``sqlalchemy-firebirdsql`` registers itself
    as ``firebirdsql``, which is the Python package speaking the wire; the engine
    answering is Firebird. So this is :data:`BACKENDS`' first entry whose key and
    :attr:`name` deliberately differ, and the reason ``name`` is a field rather
    than a property of the class — a refusal here has to say *Firebird*, not the
    name of a library the caller has never heard of. Issue #45 in its quieter
    form: the earlier cases were two engines sharing one dialect, this is one
    engine whose dialect is named after neither.

    Three facts had to be measured, and each is an override below.

    **DDL cannot be followed by DML in the same transaction** (#53). Firebird's
    DDL is properly transactional — a ``CREATE TABLE`` that never commits leaves
    nothing behind — but statements are prepared against committed metadata, so
    the new table is invisible to the transaction that created it. See
    :meth:`sees_new_tables_in_transaction`.

    **There is no way to rename a table.** Not a missing convenience — no
    statement exists, in any Firebird version. See :meth:`renames_tables`.

    **Neither portable type a loaded file needs is usable as rendered** (#55, #56).
    Core's ``Double`` becomes bare ``DOUBLE``, which Firebird's parser rejects
    outright; Core's ``Text`` becomes ``BLOB``, which is accepted and then groups
    by identity rather than by value. See :meth:`column_type` for both.

    Everything else is generic and was measured to be: the transactional floor
    holds for rows and for schema alike, a quoted mixed-case name survives,
    ``Numeric`` round-trips exactly where CrateDB truncated it, ``Time``,
    ``Date``, ``TIMESTAMP`` and ``LargeBinary`` all store and read back, indexes
    build and reflect and drop, and the driver's errors arrive properly wrapped as
    ``SQLAlchemyError`` so :meth:`driver_errors` stays empty.
    """

    name: str = "firebird"

    def sees_new_tables_in_transaction(self) -> bool:
        """``False`` — and the transactional floor is *why*, not a casualty of it.

        Worth stating in that order, because the tempting reading is that
        Firebird is somehow lax here. The opposite: every other backend that
        would fail this test avoids it by committing DDL behind the caller's
        back. Firebird refuses to do that, keeps the ``CREATE TABLE`` inside the
        transaction where it belongs, and consequently cannot let the same
        transaction address a table that is not yet committed.

        Measured both ways round, on one server, one variable changed:

        ==========================================  ========================
        Sequence                                    Result
        ==========================================  ========================
        ``CREATE`` then ``INSERT``, one transaction  ``-204 Table unknown``
        ``CREATE``, commit, then ``INSERT``          the rows land
        ``DROP … checkfirst`` then ``CREATE``        accepted together
        ==========================================  ========================

        The third row is why :meth:`loader.Workspace.insert_frame` splits at the
        DDL→DML boundary and not before it: schema statements are free to share a
        transaction with each other, so the drop-and-create pair stays atomic.
        """
        return False

    def renames_tables(self) -> bool:
        """``False``. Firebird has no rename-table statement, and never has.

        ``ALTER TABLE … RENAME TO …`` is not unsupported so much as unparseable —
        ``-104 Token unknown - line 1, column 24 RENAME`` — and there is no
        vendor-specific alternative in the way SQL Server has ``sp_rename``. A
        column can be renamed here; a table cannot.
        """
        return False

    def rename_table(self, conn: Connection, table: str, to: str) -> None:
        """Refused, with the route that does work.

        Reached only if something calls this without asking
        :meth:`renames_tables` first, which is why it refuses rather than
        assuming the guard held.
        """
        raise UnsupportedOperation(
            f"A {self.name} datasource has no statement that renames a table, so "
            f"{table} cannot become {to}. Copy the rows into a table of the new "
            f"name with create, then drop the old one — that is two tables and "
            f"loses the indexes on the first, which is why it is not done for you."
        )

    def column_type(self, declared: str, *, longest: int | None = None) -> TypeEngine:
        """Two of the three portable types have to be respelled here.

        ``REAL`` → **``DOUBLE PRECISION``** (#55). Core's ``Double`` is the
        portable 64-bit float and every other backend renders it usably. Both
        Firebird dialects render it ``DOUBLE``, a keyword Firebird does not have —
        the parser waits for ``PRECISION`` and fails on whatever follows, so the
        ``CREATE TABLE`` dies with ``-104 Token unknown`` and no table is made.
        ``DOUBLE_PRECISION`` holds a float64 exactly: ``0.1`` reads back as
        ``0.1``. ``Float(53)`` lands in the same column and was rejected in favour
        of this — both work, and only one says what it means.

        ``TEXT`` → **``VARCHAR`` sized from the data** (#56), and this is the
        dangerous one, because nothing fails. Core's ``Text`` renders ``BLOB``,
        which Firebird accepts, stores and reads back perfectly — and then treats
        as an *opaque handle* for every set operation. ``GROUP BY`` puts each row
        in its own group, ``DISTINCT`` counts five values where there are four,
        and ``ORDER BY`` does not sort. No error, no warning: a five-row table
        answers a ``GROUP BY department`` with five rows and correct-looking sums
        that are each one row's salary. The first backend here whose wrong answer
        arrives dressed as a right one.

        This is Oracle's problem with a different mechanism and the same remedy —
        there ``CLOB`` cannot be grouped and is refused outright, here ``BLOB``
        can be and lies. ``longest`` is the widest value the column actually
        holds, measured from the frame, so the ``VARCHAR`` is sized to the data
        rather than guessed at. Beyond :data:`_FIREBIRD_VARCHAR_MAX` there is
        nothing else to use and it falls back to ``Text``: the grouping is lost,
        which is bad, and the values are kept whole, which matters more than
        truncating them to fit.

        Integers need no help. Both respellings are *client-library* defects
        rather than database limits — Firebird holds a float and a comparable
        string quite happily once asked in its own words.
        """
        if declared == "REAL":
            return DOUBLE_PRECISION()
        if declared == "TEXT":
            width = max(1, longest or 1)
            return String(width) if width <= _FIREBIRD_VARCHAR_MAX else Text()
        return super().column_type(declared, longest=longest)


_SQLITE = SQLiteBackend()

#: Dialect name to the backend that has something *extra* to say about it. An
#: absence here is not a gap: it means SQLAlchemy's own answers are the whole
#: answer for that database, which is the ordinary case rather than the
#: exceptional one.
#:
#: MariaDB is registered in its own right rather than aliased to MySQL, so a
#: refusal names the database the caller actually opened.
BACKENDS: dict[str, Backend] = {
    "sqlite": _SQLITE,
    "duckdb": DuckDBBackend(),
    # Nothing extra to say, and registered anyway: it is the dialect three other
    # engines answer on, and `impostors` is where that is written down.
    "postgresql": PostgreSQLBackend(),
    "mysql": MySQLBackend(),
    "mariadb": MySQLBackend(name="mariadb"),
    "oracle": OracleBackend(),
    "mssql": MSSQLBackend(),
    "clickhousedb": ClickHouseBackend(),
    "trino": TrinoBackend(),
    "crate": CrateDBBackend(),
    # The one key here that is not the name of an engine. `sqlalchemy-firebirdsql`
    # registers the dialect under its *driver's* name, so this is what a URL
    # resolves to, while the backend it maps to calls itself `firebird` — which is
    # what a refusal has to say.
    "firebirdsql": FirebirdBackend(),
}


def backend_for(dialect: str) -> Backend:
    """The backend that answers for a dialect. Never a refusal.

    An unknown dialect gets a plain :class:`Backend` **carrying its own name**,
    and that is the point: this server reaches whatever SQLAlchemy reaches, and
    no database has to be enumerated here to be usable. What an unregistered
    dialect loses is only the extras — residency reads ``None`` (not zero:
    unknown, not empty), ``save`` says plainly that there is no local database to
    write, and the storage-class histogram is empty because a real type system
    makes it meaningless.

    That is the honest kind of degradation. The fail-open shape this project has
    been bitten by is a *guess* presented as a measurement; ``None`` and an
    explicit refusal are neither.

    **The name is carried rather than defaulted to "generic"** because these
    refusals are read by the caller: telling someone who opened PostgreSQL that
    "a generic datasource" cannot be saved names our fallback instead of their
    database. Same reason MariaDB is registered separately.
    """
    known = BACKENDS.get(dialect)
    return known if known is not None else Backend(name=dialect)


def backend_for_url(url: str | URL) -> Backend:
    """The backend for the engine actually answering, not merely for its dialect.

    :func:`backend_for` resolves from the URL alone, which is right for every
    dialect one engine speaks and wrong for the several that more than one does.
    This asks the server which it is, and asks **only** where the question can
    have a second answer: a dialect with no :attr:`Backend.impostors` returns
    immediately and no connection is made.

    Where a probe does happen it costs one short-lived connection, on a path
    that is about to open two engines and reflect a table list — so it is not a
    round trip the caller would otherwise have avoided.

    **Every failure resolves to the dialect's own backend**, deliberately and
    with the guard as wide as it is. A refused ``SELECT version()``, a permission
    the credentials lack, a driver that raises something unrelated — none of
    them is a reason to refuse a datasource that would otherwise open, and all
    of them land on exactly the behaviour that preceded this function. This is
    the one place a bare ``except Exception`` is the correct width: the question
    is optional, so *nothing* it can raise may propagate.
    """
    known = backend_for(make_url(url).get_backend_name())
    if not known.impostors or not known.banner_query:
        return known
    return known.named_by(_server_banner(url, known.banner_query))


def _server_banner(url: str | URL, query: str) -> str | None:
    """What the server calls itself, or ``None`` if it would not say.

    Read through a throwaway engine rather than the datasource's own, because
    the datasource's do not exist yet — which engine to build is what this
    answers. Disposed immediately: it exists for one row.
    """
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return str(conn.execute(text(query)).scalar())
    except Exception:  # noqa: BLE001 - see backend_for_url; the question is optional
        return None
    finally:
        engine.dispose()
