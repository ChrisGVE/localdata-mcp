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
  read-only session, CrateDB, which has neither a transaction nor a session to
  ask, reads each statement's first keyword, which its grammar makes decisive.
  The *guarantee* is portable, the mechanism is not — and on MySQL the generic
  guarantee is not even available, because DDL there commits implicitly and
  there is nothing left to roll back.
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
  question, so the truthful answer there is that the verb does not apply. Nor can
  DuckDB's, whose adapter reflects no index at all, and the answer is the same.
* **Whether a table may exist without a key.** Every backend here makes a heap
  table from a bare column list; YDB refuses one outright. A file has no key to
  offer, so what the loader does about it has to be decided once — and the axis
  carries only the *fact*, so the decision stays in one place.

Everything else — creating tables, inserting, introspection via ``inspect()``,
streaming reads — is Core and lives in :mod:`loader`. When something new turns out
to need a per-backend answer, it earns an entry here.

Two things that look like they belong and do not. **Changing a statement's
transport is not an axis** — wrapping SQL in ``text()`` to move it somewhere else
does not become a per-backend fact by being awkward. And **choosing the right Core
construct is not an axis either, it is just correctness**: a schema statement
issued as ``DDL`` rather than as ``text`` travels the DDL execution path on every
dialect, which is where a backend gets to route it. Thirteen dialects could not
tell those two apart and YDB could, and the fix was to stop stating a schema
statement as an opaque one rather than to give YDB an override for it (#59).
"""

from __future__ import annotations

import ctypes
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, partial
from inspect import signature
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, ClassVar
from xml.etree import ElementTree

from sqlalchemy import (
    DDL,
    DOUBLE_PRECISION,
    INTEGER,
    REAL,
    TEXT,
    BigInteger,
    Connection,
    Double,
    Engine,
    Index,
    LargeBinary,
    String,
    Table,
    Text,
    Unicode,
    cast,
    create_engine,
    event,
    func,
    literal_column,
    null,
    text,
)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.sql.elements import Label
from sqlalchemy.sql.visitors import replacement_traverse
from sqlalchemy.types import TypeEngine
from sqlalchemy.pool import StaticPool

__all__ = [
    "BACKENDS",
    "Backend",
    "ClickHouseBackend",
    "CockroachDBBackend",
    "CrateDBBackend",
    "DuckDBBackend",
    "Engines",
    "ExasolBackend",
    "FirebirdBackend",
    "MonetDBBackend",
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

    ``shape`` says the refusal is a rule about how the statement is *written*
    — a ``;``, a ``U&"…"`` name, a refused word inside a quoted string — which
    a read meets as surely as a write. Such a refusal is not explained as a
    refused write, and its ``what`` carries the rephrase that gets the read
    through (#135).

    ``lock`` says the gate refused a row-locking clause in a statement that
    otherwise reads, and ``what`` is then why, worded to follow "its <clause>
    clause". Such a refusal is explained as #144 explains a locking read the
    session refused, with the clause named and the same read without it; only
    the reason differs. Firebird's reader sets it, because its driver fetches no
    rows for a select carrying ``FOR UPDATE`` (#145).
    """

    what: str | None = None
    shape: bool = False
    lock: bool = False

    def take(self) -> str | None:
        """Read the last refusal and forget it, so it cannot leak into the next."""
        seen, self.what, self.shape, self.lock = self.what, None, False, False
        return seen


#: The declared type of a loaded column, to the portable Core type that renders
#: it. Keyed by the three names :mod:`loader` decides between; what each becomes
#: in SQL is SQLAlchemy's business and differs per dialect, which is exactly what
#: makes this the generic answer.
#:
#: ``BigInteger`` rather than ``Integer`` because a loaded integer is an int64:
#: ``Integer`` renders ``INTEGER``, which is 32-bit on PostgreSQL, MySQL and SQL
#: Server, so one value past 2,147,483,647 refused the whole load (#110).
#: ``BigInteger`` is ``BIGINT`` there and ``NUMBER(19)`` on Oracle.
_PORTABLE_TYPES: dict[str, Any] = {"INTEGER": BigInteger, "REAL": Double, "TEXT": Text}


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


#: Backend names spelled with a consonant but *spoken* with a vowel, because
#: they are read out letter by letter: "an em-ess-cue-ell datasource". English
#: takes its article from the sound, and these refusals are read by a person.
#: Every other name this project ships agrees with its spelling — mysql and
#: mariadb are read "my-" and "maria-", so they take "a" and are not listed.
_SPOKEN_AS_INITIALS = frozenset({"mssql"})


def article_for(name: str) -> str:
    """``"a"`` or ``"an"`` for a backend name, by sound rather than spelling.

    Its own function rather than inline because every generic refusal below
    opens with this word, and an unregistered dialect brings a name nobody here
    chose — so the rule has to hold for a string this file has never seen.
    """
    if name.lower() in _SPOKEN_AS_INITIALS:
        return "an"
    # `first and` rather than the slice alone: `"" in "aeiou"` is True, so an
    # empty name would take the vowel branch — the one string the rule above
    # promises to hold for and the one it would not have.
    first = name[:1].lower()
    return "an" if first and first in "aeiou" else "a"


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
    #: engine.** TiDB and OceanBase answer on MySQL's; YugabyteDB and Greenplum
    #: on PostgreSQL's. Keyed by dialect alone, every one of them is
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
        and on ClickHouse, which has no transactions, the URL is where the
        posture is stated — there is no rollback floor underneath it.
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

    def without_row_locks(self, sql: str) -> tuple[str, str] | None:
        """``sql``'s row-locking clause and the same read without it, or ``None``.

        Asked by :meth:`loader.Workspace._explain` only where
        :meth:`denies_write` has recognised the backend's own refusal, because
        a read-only session can refuse a locking read with the code it refuses
        a write with: on PostgreSQL ``SELECT * FROM t FOR UPDATE`` is answered
        25006, as an ``INSERT`` is. Such a statement does read, so explaining
        it as a write is false (#144). A backend whose session refuses a
        locking clause answers here with the clause as written and the
        statement with it removed, which reads the same rows; generically no
        session refuses one, and the answer is ``None``.
        """
        return None

    def syntax_error(self, engine: Engine, sql: str, message: str) -> str | None:
        """The server's own syntax error for ``sql``, found without running it.

        Asked by :meth:`loader.Workspace._explain` only where it is about to
        report a statement as a kind the reader refuses, because the server
        objected to its first word inside a cursor declaration (#143). A
        misspelled first word looks exactly like that: ``SELEC 1`` arrives as
        ``syntax error at or near "SELEC"`` beside ``CURSOR FOR SELEC``, just
        as ``INSERT`` does. ``None`` means the statement parses, so the
        refused-kind reading stands; generically there is no one to ask, and
        the answer is ``None``.

        ``message`` is the driver's own words for the failure, for a backend
        that can tell a typo apart only from them.

        **Whatever asks must not run the statement or open a write path**: it
        is asked of the reader, about a statement nothing has checked yet.
        """
        return None

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

        There is no sibling for rows, because no backend here keeps a refused
        row: each one either refuses the write before it runs or rolls it
        back. CrateDB kept one until its statement gate (#118), and the axis
        it needed went with it.
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
        a race: ``insert_source`` reports the row count it read, and the number it
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

        :meth:`loader.Workspace.insert_source` asks this and splits its one write
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
        there is no local file to write out. The route round it is three calls:
        ``query(…, path=…)`` writes the rows to a file, ``attach`` on that file
        makes a database of our own — file-derived, so writable and saveable —
        and ``save`` writes it out. The middle verb is ``attach`` and not
        ``create``: ``create`` lands a file *inside a slot that is already
        open*, and the caller reading this refusal holds only the slot that
        just refused them.
        """
        article = article_for(self.name).capitalize()
        raise UnsupportedOperation(
            f"{article} {self.name} datasource is reached over its own connection, not "
            f"held here, so there is no local database to write out. Write the "
            f"rows you want to a file with query(nickname, sql, path=…), attach "
            f"that file as a datasource of your own, and save that; further "
            f"tables go into it with create before you save."
        )

    def saves(self) -> bool:
        """Whether ``save`` means anything here — whether :meth:`snapshot` writes one.

        ``False`` generically, for the reason :meth:`snapshot` refuses: a
        database this server reaches is not one it holds. ``True`` only on a
        backend that overrides :meth:`snapshot` with a copy of its own, and the
        two are pinned together by a test across every registered backend.

        It exists so a caller can know *before* asking. ``attach`` and
        ``directory`` report it as ``saveable``, because a SQLite file and a
        DuckDB file both arrive as ``kind: "database"`` and only one of them can
        be saved — and finding out by calling ``save`` is what a capability
        answer is for avoiding (issue #81).
        """
        return False

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

        **Through ``DDL`` rather than through ``text``, and the difference is not
        cosmetic** (#59). Both render the same string; only one of them tells
        SQLAlchemy that the string is *schema*. ``DDL`` is an
        ``ExecutableDDLElement``, so it travels the DDL execution path — which is
        where a dialect gets to say that a schema statement does not belong
        inside the surrounding transaction. ``text`` is an opaque statement and
        stays wherever the connection already is.

        Eleven dialects cannot tell the two apart, and that agreement is the
        reason this stood as ``text`` for so long. YDB can: every schema
        operation there is refused outright with ``Scheme operations cannot be
        executed inside transaction`` when it arrives as ``text``, and accepted
        when it arrives as ``DDL`` — measured, one variable changed, on the same
        connection. Every other schema statement this server issues was already a
        Core construct (``Table.create``, ``Table.drop``, ``Index.create``), so
        this was the one place the signal was being dropped.
        """
        prepare = conn.dialect.identifier_preparer.quote
        conn.execute(DDL(f"ALTER TABLE {prepare(table)} RENAME TO {prepare(to)}"))

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

        ``longest`` is how wide the widest value in the column actually is,
        measured rather than assumed, in the unit :meth:`text_width` chose for
        this backend — characters everywhere but SQL Server — and ``None`` when
        the column holds no text. It is ignored generically — an unbounded text type needs
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

    def text_width(
        self,
        *,
        characters: int | None,
        utf16_units: int | None,
        utf8_bytes: int | None = None,
    ) -> int | None:
        """The width a loaded text column is sized to here, in this engine's unit.

        The loader measures the widest value three ways, because the engines
        that size text do not agree on what a width counts. Characters,
        generically, which is what Firebird's ``CHARACTER SET UTF8`` column
        counts (#56). SQL Server's ``NVARCHAR`` counts UTF-16 code units
        instead, and a character outside the Basic Multilingual Plane — every
        emoji — is two of them (:meth:`MSSQLBackend.text_width`). Oracle sizes in
        characters and caps in UTF-8 bytes (:meth:`OracleBackend.text_width`).

        Whatever this answers is what :meth:`column_type` and
        :meth:`column_note` receive as ``longest``, asked once per column, so
        the declaration and the note are always about the same width.
        """
        return characters

    def column_note(
        self, name: str, declared: str, *, longest: int | None = None
    ) -> str | None:
        """What the caller must be told about the type a column landed as.

        ``None`` generically, because on every backend but one the type
        :meth:`column_type` chose answers queries correctly or refuses them out
        loud. The exception is a type that is accepted, stores every value, and
        then answers some queries *wrongly* with no error — Firebird's ``BLOB``,
        which groups by storage handle. A load that produced one has to say so
        for that column, or the table looks perfectly ordinary.

        Takes the same ``declared`` and ``longest`` as :meth:`column_type`, so
        the two cannot disagree about which columns the fallback reached.
        """
        return None

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

    def requires_primary_key(self) -> bool:
        """Whether a table here is refused unless it declares a primary key.

        ``False`` everywhere but YDB, which rejects a keyless ``CREATE TABLE``
        outright — ``Primary key is required for ydb tables.`` — rather than
        making a heap the way every other backend does.

        This is a question about **loading a file**, which is the only place this
        server creates a table it did not receive a definition for. A file's rows
        have an order and no key: nothing in a CSV is guaranteed unique, so there
        is no column that could be nominated without the loader inventing a
        constraint the data does not have. Where this answers ``True``,
        :meth:`loader.Workspace.insert_source` adds a surrogate key holding each
        row's ordinal, and **says so in the table's notes** — the column is real,
        it will show up in ``directory`` and in ``SELECT *``, and a caller told
        nothing would rightly call that a lie.

        The alternative shapes were measured and rejected. Nominating *every*
        column as the key makes duplicate rows collide, and YDB resolves a
        primary-key collision by merging rather than by failing — so two
        identical rows in a CSV would land as one, silently. Refusing the load
        entirely was the other option, and it fails this project's rule that an
        unknown backend answers honestly and usefully rather than
        "unsupported": the rows can be stored, exactly, and only the shape of
        the table has to give.
        """
        return False

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
        and can afterwards be found by reflection — which is what ``directory``
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

    def optional_aggregates(self) -> Mapping[str, Any]:
        """Statistics beyond the universal floor that this engine can compute.

        A mapping from the name ``stats`` reports to something that turns a
        column expression into the aggregate over it. **Empty generically**, and
        that is the whole of the cost rule this verb is governed by: *free we
        take, expensive we leave.* A function the engine lacks is simply not
        reported — never emulated, never approximated, and never computed in a
        second pass in Python.

        The floor needs nothing from here. ``count``, ``min``, ``max`` and
        ``avg`` are in every dialect this server reaches, so they are issued
        generically and are the same four everywhere; a column's null count is
        ``count(*) - count(col)``, which is stock SQL and is what the design
        rests on.

        Empty rather than full is the safe direction to default in. A backend
        nobody has measured under-reports — the caller sees no median and knows
        it, which is a true statement about what this server offers. Defaulting
        the other way would have an unsubclassed dialect fail a statement mid
        profile, turning "we do not report this" into "the verb does not work
        here". So an entry appears only where it has been driven against the
        engine and answered correctly. Every registered backend has been
        (#95); ``tests/test_stats_aggregates.py`` holds the record and drives
        each one, and a backend added later starts here, at nothing.

        A mapping rather than a set of flags because each name is also *how* the
        aggregate is spelled, and the two are not separable: a median is
        ``median(col)`` on one engine and ``percentile_cont(0.5) WITHIN GROUP
        (ORDER BY col)`` on the next. Keeping the spelling beside the capability
        is what stops a second dispatch on dialect name growing somewhere else.
        """
        return {}


def _percentile_cont_median(column: Any) -> Any:
    """The median as the SQL standard spells it, for an engine with no ``median()``.

    ``percentile_cont(0.5) WITHIN GROUP (ORDER BY col)`` is the *continuous*
    percentile: with an even count it interpolates between the two middle
    values, which is what a median is. ``percentile_disc`` would return one of
    them instead, and is not used anywhere here.

    The fraction is written into the statement as a literal rather than bound
    as a parameter: it is a constant of the statistic, not a value from the
    caller, and some engines want the fraction of a ``WITHIN GROUP`` aggregate
    to be one.
    """
    return func.percentile_cont(literal_column("0.5")).within_group(column)


def _percentile_cont_median_of_doubles(column: Any) -> Any:
    """:func:`_percentile_cont_median` over the column read as ``DOUBLE PRECISION``.

    For an engine whose ``percentile_cont`` is defined over floating point
    alone and refuses an integer or a decimal column outright, rather than
    converting it as PostgreSQL does. The cast is the conversion PostgreSQL
    makes implicitly, written out, so the answer is the same one: exact for
    every integer up to 2**53 and every decimal a double can hold, and rounded
    to a double beyond that, as PostgreSQL's own median is.
    """
    return _percentile_cont_median(cast(column, Double))


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
        # A ``URL`` object, not a string to parse. The ``file:`` URI is itself
        # percent-encoded, and SQLAlchemy 2.1 unquotes a URL string's database on
        # parse (#70), so a string would turn ``%3F`` back into the ``?`` that
        # SQLite then reads as the start of its query: ``why? not.db`` opened as
        # ``why``. An object is handed to the engine exactly as built.
        base = path.resolve().as_uri()
        write_query = {"uri": "true"}
        if not writable:
            write_query = {"mode": "ro", "uri": "true"}
        write_url = URL.create("sqlite", database=base, query=write_query)
        read_url = URL.create(
            "sqlite", database=base, query={"mode": "ro", "uri": "true"}
        )
        return self._pair(write_url, read_url)

    def _pair(self, write_url: str | URL, read_url: str | URL) -> Engines:
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
        are not two spellings of one thing here, and neither is what ``directory``
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

    def saves(self) -> bool:
        """SQLite is the database this server holds, so ``save`` writes it out."""
        return True

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


def _adapt_duckdb_column_reflection(dialect: Any) -> None:
    """Let ``duckdb-engine`` 0.17 reflect columns under SQLAlchemy 2.1 (#70).

    SQLAlchemy 2.1 changed two PostgreSQL internals that ``duckdb-engine``
    0.17.0 inherits for its ``get_multi_columns``, and this adapts both on the
    one dialect instance, leaving SQLAlchemy's own queries and types in charge:

    * **The column query.** It joins ``pg_catalog.pg_collation`` for each
      column's ``collation``. DuckDB has no such table and binds every table a
      statement names, so the query fails before any row is read. The
      ``collation`` column becomes ``NULL``, and the rest of the query, every
      FROM and WHERE, is left as SQLAlchemy built it. DuckDB has no non-default
      collations for this server to report.
    * **The column-info call.** ``duckdb-engine`` passes eagerly built
      ``domains`` and ``enums`` mappings as separate arguments. SQLAlchemy 2.1
      takes one lazy ``named_type_loader`` with those two attributes. The
      mappings are handed over under that shape, so the type reflection they
      drive is SQLAlchemy's own.

    ``duckdb-engine`` marks the collation path as a TODO for SQLAlchemy 2. The
    adaptation applies only where the PostgreSQL method has the 2.1 signature,
    and the upstream fix is to remove it once a ``duckdb-engine`` release
    supports SQLAlchemy 2.1 itself.
    """
    dialect_class = type(dialect)

    def _null_collation(
        element,
    ):
        if isinstance(element, Label) and element.name == "collation":
            return null().label("collation")
        return None

    original_query = dialect_class._columns_query

    def _columns_query(self, schema, has_filter_names, scope, kind):
        query = original_query(self, schema, has_filter_names, scope, kind)
        return replacement_traverse(query, {}, _null_collation)

    dialect._columns_query = MethodType(_columns_query, dialect)

    original_info = dialect_class._get_columns_info
    if "named_type_loader" not in signature(original_info).parameters:
        return

    def _get_columns_info(self, rows, domains, enums, schema):
        loader = SimpleNamespace(domains=domains, enums=enums)
        return original_info(self, rows, loader, schema)

    dialect._get_columns_info = MethodType(_get_columns_info, dialect)


def _seal_duckdb(dbapi_conn, _record) -> None:  # noqa: ANN001 - SQLAlchemy's signature
    """Switch off DuckDB's own file access on a new connection, then lock it.

    ``enable_external_access = false`` stops everything a statement can open
    by itself: ``read_*``, ``glob``, ``COPY`` in either direction, ``ATTACH``
    of another file, ``EXPORT DATABASE``, ``INSTALL``/``LOAD`` and URLs. It
    does **not** stop DuckDB using its own files — the database, its WAL, its
    spill directory — which is all a datasource needs. Measured on DuckDB
    1.5.5: reads, ``CREATE``/``DROP``/``CHECKPOINT`` on a writable file, and a
    sort spilling past ``memory_limit`` all still work. The exemption also lets
    a ``COPY`` write the WAL, which is why the reader refuses every statement
    that is not a query before it runs (:func:`_refuse_duckdb_non_read`,
    #137). ``lock_configuration
    = true`` then pins every other setting (``temp_directory``,
    ``memory_limit``, extension autoloading), so no statement can move it.

    **No ``allowed_directories``, deliberately.** Opening DuckDB onto the
    allowed paths would let ``read_csv`` reach a file there, and ``COPY ...
    TO`` write one there too — and ``query`` only reads. A second file reaches
    a datasource through ``create(type='table')``, which the writable grant
    governs.

    **Set after the connection opens, not passed as startup config.** Two
    measured reasons. ``duckdb_engine`` merges a URL's query parameters
    *over* ``connect_args``, so ``duckdb:///x.db?enable_external_access=true``
    would win; a ``SET`` after the open is the last word. And DuckDB names the
    WAL of ``why? not.duckdb`` as ``why.wal? not.duckdb`` — it splits at the
    ``?`` — while its startup exemption for its own files expects the other
    spelling, so with startup config such a file does not open at all. The
    cost of the ``SET`` route, also measured: a *writable* file with a ``?`` in
    its name opens and reads but cannot commit, for the same WAL reason.

    Settings are per database instance and DuckDB shares one instance between
    connections to the same file, so a second connection finds the instance
    already locked. That is the seal holding, not a failure — unless the lock
    arrived with file access still on (a URL can ask for exactly that), in
    which case the connection is refused rather than handed out open.
    """
    cursor = dbapi_conn.cursor()
    try:
        cursor.execute(
            "SELECT current_setting('lock_configuration'),"
            " current_setting('enable_external_access')"
        )
        locked, external = cursor.fetchone()
        if not locked:
            cursor.execute("SET enable_external_access = false")
            cursor.execute("SET lock_configuration = true")
        elif external:
            raise PermissionError(
                "This DuckDB database has its configuration locked with file "
                "access switched on, so it cannot be kept to its own file. "
                "Attach it without lock_configuration in the URL."
            )
    finally:
        cursor.close()


#: How a DuckDB read connection opens every transaction. Measured on DuckDB
#: 1.5.5: inside it, ``INSERT``, ``CREATE`` and the rest are refused with
#: "transaction is launched in read-only mode", and nothing reaches the file.
_DUCKDB_READ_ONLY_BEGIN = "BEGIN TRANSACTION READ ONLY"

#: What a statement refused for ending its transaction was asking to do,
#: worded to follow "This statement asks to" in ``query``'s refusal.
_DUCKDB_TRANSACTION_CONTROL = "begin, commit or roll back a transaction of its own"


def _begin_read_only(dbapi_conn, _record) -> None:  # noqa: ANN001 - SQLAlchemy's signature
    """Make every transaction on this connection a read-only one.

    ``duckdb_engine`` begins a transaction by calling the DBAPI connection's
    ``begin()``, which starts an ordinary read-write one, and DuckDB has no
    session setting that changes that — there is no
    ``default_transaction_read_only`` and no ``SET TRANSACTION`` (both
    measured on 1.5.5). So the connection's own ``begin`` is replaced, once,
    when the connection is made: from then on SQLAlchemy's autobegin opens
    ``BEGIN TRANSACTION READ ONLY`` instead. The wrapper ``duckdb_engine``
    hands SQLAlchemy is a plain Python object, so the replacement is an
    instance attribute and touches nothing outside this connection.

    Pinned by the server tests that send ``INSERT … RETURNING`` to ``query``
    on a writable slot: if a later ``duckdb_engine`` begins some other way,
    those writes stop being refused and the tests fail.
    """
    dbapi_conn.begin = partial(dbapi_conn.execute, _DUCKDB_READ_ONLY_BEGIN)


#: The statement kinds the DuckDB reader runs as they stand. Measured on
#: 1.5.5 and 2.0.0, DuckDB's parser calls every spelling of a query a
#: ``SELECT``: ``WITH``, ``FROM t``, ``TABLE t``, ``VALUES``, ``SUMMARIZE``,
#: ``DESCRIBE``, ``SHOW``, ``PIVOT … IN (…)`` and ``UNPIVOT``. A ``PRAGMA``
#: that only reads (``table_info``, ``show_tables``, ``database_size``,
#: ``version``) was a ``SELECT`` on 1.5.5 and is a ``PRAGMA`` on 2.0.0, so it
#: is refused there, with its query spellings named (:data:`_PRAGMA_AS_QUERY`).
#: ``CALL f()`` is its spelling of ``SELECT * FROM f()``, so it reaches no
#: function a ``SELECT`` cannot; ``CHECKPOINT`` is one, and does what closing
#: the datasource does (``docs/CONSTRAINTS.md`` §29.4). ``EXPLAIN`` is not
#: here, because ``EXPLAIN ANALYZE`` runs what it explains: see
#: :func:`_duckdb_explained`.
_DUCKDB_RUNS = frozenset({"SELECT", "CALL"})

#: How a refused ``PRAGMA`` points at the same reads asked as a query. DuckDB
#: 2.0.0 parses every ``PRAGMA`` as kind ``PRAGMA``, even one that only reads,
#: and the reader keeps to the parser's word rather than a list of PRAGMA
#: names. Every spelling named here was run through ``query`` on 2.0.0 and
#: parses as a ``SELECT`` on 1.5.5 too (``tests/test_duckdb_pragma_rephrase.py``).
_PRAGMA_AS_QUERY = (
    "a PRAGMA that only reads is answered when asked as a query: SHOW TABLES "
    "or SHOW ALL TABLES for the tables, DESCRIBE <table> for a table's "
    "columns, SELECT * FROM pragma_table_info('<table>') for them as "
    "table_info gives them, and SELECT * FROM pragma_version() or "
    "pragma_database_size()"
)

#: How a refusal names each DuckDB statement kind whose name alone does not say
#: what it asks for, worded to follow "This statement asks to". Every other
#: kind is named by itself, ``MERGE_INTO`` as ``MERGE INTO``.
_DUCKDB_KIND_WORDS = {
    "CREATE_FUNC": "CREATE FUNCTION",
    "COPY": "COPY (a result is written to a file by passing path= to query)",
    "COPY_DATABASE": "COPY FROM DATABASE",
    "EXPORT": "EXPORT DATABASE",
    "LOAD": "INSTALL or LOAD an extension",
    "PRAGMA": f"run a PRAGMA that is not a query ({_PRAGMA_AS_QUERY})",
    "SET": "change a setting (SET, RESET, USE or SET VARIABLE)",
    "VACUUM": "VACUUM or ANALYZE",
}


def _duckdb_kind_words(kind: str) -> str:
    """A DuckDB statement kind in the words a refusal names it by."""
    return _DUCKDB_KIND_WORDS.get(kind, kind.replace("_", " "))


#: How a refused PIVOT without an ``IN`` list is named, worded to follow
#: "This statement asks to". DuckDB cannot know such a PIVOT's columns until it
#: has read the pivoted column's values, so its parser expands the statement:
#: measured, into a ``CREATE`` with no text of its own and a ``SELECT`` on
#: 1.5.5 and 1.5.6, and into one statement of kind ``MULTI`` on 2.0.0, which a
#: refusal naming the kind alone left unexplained. With its values named the
#: same PIVOT parses as one ``SELECT`` on both versions and answers
#: (``tests/test_duckdb_pivot_rephrase.py``).
_DUCKDB_UNLISTED_PIVOT = (
    "PIVOT without naming its values in an IN list, which DuckDB does not "
    "parse as one query (name them, as in PIVOT t ON k IN ('a', 'b') USING "
    "sum(v); SELECT DISTINCT k FROM t lists them)"
)

#: The keywords that begin DuckDB's PIVOT statement.
_DUCKDB_PIVOT_KEYWORDS = frozenset({"PIVOT", "PIVOT_WIDER"})


def _duckdb_unlisted_pivot(statement: Any, sql: str) -> bool:
    """Whether a refused ``statement`` is DuckDB's expansion of a PIVOT with no ``IN``.

    Both halves come from DuckDB. The statement is one the parser made rather
    than one the caller wrote: kind ``MULTI`` (2.0.0), or a statement with no
    source text (the ``CREATE`` 1.5.x adds). And DuckDB's tokenizer finds a
    ``PIVOT`` keyword in its text, so ``'PIVOT'`` inside a string or a comment
    does not count. Measured on both versions, a PIVOT inside a subquery or
    after a ``WITH`` expands the same way, which is why the keyword is looked
    for anywhere rather than as the first word. Wrong either way, this changes
    only the words of a refusal, never whether the statement is refused.
    """
    import duckdb

    if statement.type.name != "MULTI" and statement.query.strip():
        return False
    text = statement.query or sql
    for offset, kind in duckdb.tokenize(text):
        found = re.match(r"\w+", text[offset:])
        if (
            kind == duckdb.token_type.keyword
            and found is not None
            and found.group(0).upper() in _DUCKDB_PIVOT_KEYWORDS
        ):
            return True
    return False


def _duckdb_explained(connection: Any, explain: str) -> str | None:
    """The kind of the statement an ``EXPLAIN`` explains, or ``None`` if unclear.

    Asked because ``EXPLAIN ANALYZE`` *runs* the statement: measured on 1.5.5,
    ``EXPLAIN ANALYZE COPY (SELECT 'g') TO '<file>.wal' (USE_TMP_FILE false)``
    wrote the WAL, and DuckDB's parser calls it ``EXPLAIN`` whatever it
    explains. The Python API does not hand back the explained statement, so
    this finds where it begins with DuckDB's own tokenizer, skipping
    whitespace and comments, and gives the rest back to DuckDB's parser.
    Comments are skipped here rather than by the tokenizer: measured, 1.5.5
    gave a comment no token, while 2.0.0 gives it one of its own
    (``token_type.comment``), and read as the word ``/`` that comment made
    ``EXPLAIN /* b */ ANALYZE SELECT 1`` an ``EXPLAIN`` of ``ANALYZE …``,
    refusing a read. The
    grammar has three prefixes, all measured: ``EXPLAIN``, ``EXPLAIN
    ANALYZE``, and ``EXPLAIN (options)`` such as ``(FORMAT json, ANALYZE)``.
    A parenthesised query, ``EXPLAIN (SELECT 1)``, looks like the third, so
    the text after the group is tried first and the text including it second;
    the first that parses as one statement decides.

    **Fail-closed is the invariant.** A prefix read wrongly leaves text that
    does not parse as one statement, or parses as something that is not a
    query, and either answer refuses. No statement DuckDB explains begins with
    ``(`` unless it is a query, so the second try cannot turn a write into a
    read.
    """
    import duckdb

    comment = duckdb.token_type.comment
    offsets = [offset for offset, kind in duckdb.tokenize(explain) if kind != comment]

    def word(index: int) -> str:
        found = re.match(r"\w+|\S", explain[offsets[index] :])
        return found.group(0).upper() if found else ""

    if not offsets or word(0) != "EXPLAIN":
        return None
    candidates = [1]
    if len(offsets) > 1 and word(1) == "ANALYZE":
        candidates = [2]
    elif len(offsets) > 1 and word(1) == "(":
        depth, index = 0, 1
        while index < len(offsets):
            depth += {"(": 1, ")": -1}.get(word(index), 0)
            index += 1
            if depth == 0:
                break
        candidates = [index, 1]
    for index in candidates:
        if index >= len(offsets):
            continue
        try:
            inner = connection.extract_statements(explain[offsets[index] :])
        except duckdb.ParserException:
            continue
        if len(inner) == 1:
            return inner[0].type.name
    return None


#: What the DuckDB reader says when a statement calls ``force_checkpoint`` (#138).
#: DuckDB refuses it inside the reader's read-only transaction in its own words,
#: "Cannot FORCE CHECKPOINT: the current transaction has been started"; both
#: statements named here ran through ``query`` on 2.0.0, measured.
_DUCKDB_FORCE_CHECKPOINT = (
    "call force_checkpoint(), which DuckDB runs only outside a transaction, and "
    "query reads inside a read-only one (CHECKPOINT and FORCE CHECKPOINT run "
    "here)"
)


def _duckdb_calls_force_checkpoint(sql: str) -> bool:
    """Whether ``sql`` calls ``force_checkpoint``, read with DuckDB's own tokenizer.

    The call is an identifier ``force_checkpoint``, quoted or not and in any
    case, whose next token past any comment is ``(``. A string constant is a
    token of its own kind and a column or an alias has no ``(`` after it, so
    neither is a call. Only the words depend on this: DuckDB refuses the call
    itself, measured, so a spelling missed here is still refused, in DuckDB's
    words.
    """
    import duckdb

    comment = duckdb.token_type.comment
    identifier = duckdb.token_type.identifier
    tokens = [(start, kind) for start, kind in duckdb.tokenize(sql) if kind != comment]
    for (start, kind), (after, _) in zip(tokens, tokens[1:]):
        if kind != identifier or not sql.startswith("(", after):
            continue
        name = re.match(r'"?(\w+)"?', sql[start:])
        if name is not None and name.group(1).lower() == "force_checkpoint":
            return True
    return False


def _duckdb_read_refusal(connection: Any, sql: str) -> str | None:
    """What running ``sql`` on the reader would ask to do, or ``None`` for a read.

    Worded to follow "This statement asks to" in ``query``'s refusal. Every
    answer comes from DuckDB's own parser, ``extract_statements``, which runs
    nothing (pinned by ``tests/test_duckdb_reader_gate.py``). One statement is
    expanded while it is parsed: ``PRAGMA import_database`` reads the
    directory's ``schema.sql``, a read the seal refuses on this connection, so
    it raises DuckDB's permission error here. A statement the parser cannot
    parse raises its own parser error, which is the error running it would
    have raised. In order:

    1. **A statement that ends the transaction**, anywhere in the string:
       ``BEGIN``, ``COMMIT``, ``ROLLBACK``, ``ABORT``, ``END``, all of kind
       ``TRANSACTION``. ``BEGIN TRANSACTION READ ONLY`` holds only until
       something ends it, and DuckDB runs every statement of a string:
       measured on 1.5.5, ``COMMIT; INSERT … RETURNING …`` committed the empty
       transaction, then ran the ``INSERT`` in autocommit, and the row was
       kept (#79).
    2. **The first statement that is not a query** (#137). The read-only
       transaction and ``access_mode=read_only`` refuse a row change and DDL,
       and the seal (:func:`_seal_duckdb`) refuses another file, but none of
       them refuses a ``COPY`` onto the database's own ``.wal``. Measured on
       1.5.5, and on 2.0.0 in #137: with a committed row still in the WAL, ``COPY (SELECT
       'g') TO '<file>.wal' (USE_TMP_FILE false)`` replaced it, answered
       ``ok``, and the next open of the file had lost the row. ``LOAD``, ``SET
       VARIABLE``, ``PREPARE``, ``EXECUTE``, ``VACUUM`` and a ``PRAGMA`` that
       is not a query answered ``ok`` as well, and none of them reads. So only
       :data:`_DUCKDB_RUNS` runs; a string of several statements names them
       all, because one statement can be several to DuckDB: on 1.5.x a
       ``PIVOT`` without ``IN`` is a ``CREATE`` and a ``SELECT``, and on 2.0.0
       it is one statement of kind ``MULTI``. Either way its refusal says so
       in plain words, with the one-query form (:func:`_duckdb_unlisted_pivot`).
    3. **A call to ``force_checkpoint``** (:func:`_duckdb_calls_force_checkpoint`),
       which DuckDB refuses inside the transaction in words about a
       transaction the caller never opened (#138).
    4. **More than one statement**, even all of them queries: DuckDB answers
       with the last one's rows alone, so the others ran for nothing the
       caller can see.
    5. **An ``EXPLAIN`` of anything but a query** (:func:`_duckdb_explained`).

    A string that parses to no statement at all, such as a lone comment, is
    left to DuckDB, which runs nothing and so answers no rows.
    """
    statements = connection.extract_statements(sql)
    kinds = [statement.type.name for statement in statements]
    if "TRANSACTION" in kinds:
        return _DUCKDB_TRANSACTION_CONTROL
    several = f"DuckDB reads this as {len(kinds)} statements: {', '.join(kinds)}"
    for statement, kind in zip(statements, kinds):
        if kind not in _DUCKDB_RUNS and kind != "EXPLAIN":
            named = (
                _DUCKDB_UNLISTED_PIVOT
                if _duckdb_unlisted_pivot(statement, sql)
                else _duckdb_kind_words(kind)
            )
            return f"{named} ({several})" if len(kinds) > 1 else named
    if _duckdb_calls_force_checkpoint(sql):
        return _DUCKDB_FORCE_CHECKPOINT
    if len(kinds) > 1:
        return f"run {len(kinds)} statements ({', '.join(kinds)}) in one call"
    if kinds == ["EXPLAIN"]:
        explained = _duckdb_explained(connection, statements[0].query)
        if explained != "SELECT":
            what = (
                f"{article_for(explained)} {explained.replace('_', ' ')} statement"
                if explained
                else "a statement that is not a query"
            )
            return (
                f"EXPLAIN {what} (EXPLAIN is answered over a query only, "
                f"because EXPLAIN ANALYZE runs the statement it explains)"
            )
    return None


def _refuse_duckdb_non_read(refusal: Refusal, connection: Any, sql: str) -> None:
    """Refuse ``sql`` before it runs unless it is exactly one query.

    What counts, and why, is :func:`_duckdb_read_refusal`'s to say. Raises
    DuckDB's own ``PermissionException``, a DBAPI ``Error``, so SQLAlchemy
    wraps it and ``query`` explains it as a refused write.
    """
    import duckdb

    refused = _duckdb_read_refusal(connection, sql)
    if refused is not None:
        refusal.what = refused
        raise duckdb.PermissionException(f"query may not {refused}.")


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

    It now holds a second: the two statistics DuckDB computes that SQLite does
    not. See :meth:`optional_aggregates`.

    And a third: DuckDB reaches files **by itself**, which no other dialect
    here does from inside a statement. See :meth:`open`.

    And a fourth: DuckDB holds one configuration per file per process, so the
    two engines of a slot cannot differ in posture the way the generic
    :meth:`Backend.open` makes them. The URL posture above is therefore the
    *slot's*, on both engines or on neither, and a writable slot's reader is
    held to reading by a read-only transaction instead. See :meth:`open` and
    :meth:`read_posture` (issue #79).

    And a fifth: its indexes cannot be found again by reflection, so the index
    verbs do not apply here. See :meth:`build_index` (issue #112).
    """

    name: str = "duckdb"
    read_only_query: ClassVar[Mapping[str, str]] = {"access_mode": "read_only"}

    def open(self, url: str | URL, *, writable: bool) -> Engines:
        """Two engines onto one DuckDB configuration, each sealed off from other files.

        **One configuration for both engines (issue #79).** DuckDB keeps one
        database instance per file per process, and the first connection fixes
        its configuration: a second connection asking for a different one is
        refused with "Can't open a connection to same database file with a
        different configuration than existing connections". The generic
        :meth:`Backend.open` puts :attr:`read_only_query` on the reader only,
        so on DuckDB whichever engine connected second failed — every verb
        but ``query`` on a writable file, and ``directory`` and ``stats`` on a
        read-only URL. So the posture follows the *slot* here, not the engine:

        * **not writable** — both engines carry ``access_mode=read_only``, so
          DuckDB itself refuses a write whichever engine is asked, which is
          what :meth:`loader.Workspace.describe` says of a read-only
          datasource;
        * **writable** — neither carries it, and the reader is held to reading
          by :meth:`read_posture` instead.

        **Sealed, both of them.** ``access_mode=read_only`` protects the
        *database file* and nothing else. DuckDB's own table functions and
        ``COPY`` open files directly — ``read_text('/etc/hosts')``, ``read_csv``
        anywhere on disk, ``COPY (...) TO '/tmp/x.csv'`` — and none of that
        passes through :mod:`paths`, so on a file attached read-only ``query``
        both read outside the allowed paths and wrote a file (issue #105). Both
        engines are sealed, not only the reader ``query`` uses: a statement the
        writer runs for ``create`` must not reach a file either.
        :meth:`Backend.open_file` comes through here, so a file and a URL are
        configured and sealed by the same lines. What the seal is, and what it
        costs, is :func:`_seal_duckdb`'s to say.
        """
        shared = make_url(url) if writable else self._read_only(url)
        connect_args = dict(self.connect_args())
        refusal = Refusal()
        engines = Engines(
            write=create_engine(shared, connect_args=connect_args),
            read=create_engine(shared, connect_args=connect_args),
            refusal=refusal,
        )
        self.read_posture(engines.read, refusal)
        for engine in (engines.read, engines.write):
            event.listen(engine, "connect", _seal_duckdb)
            _adapt_duckdb_column_reflection(engine.dialect)
        return engines

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        """Run every read in a ``READ ONLY`` transaction it cannot leave.

        Stronger than the generic floor, and needed: on a writable slot the
        reader no longer opens the file ``access_mode=read_only`` (see
        :meth:`open`), and under the floor alone ``INSERT … RETURNING`` sent to
        ``query`` answers with the row it inserted while the closing rollback
        throws the row away. A read-only transaction refuses the write instead,
        so the answer and the outcome agree.

        Two hooks, for the two halves: :func:`_begin_read_only` makes every
        transaction read-only, and :func:`_refuse_duckdb_non_read` refuses,
        before it runs, every statement DuckDB's own parser does not call a
        query — the ones that would end the transaction early (#79), and the
        ones neither the transaction nor the seal refuses, such as a ``COPY``
        onto the database's own ``.wal`` (#137). Installed on a read-only
        slot's reader as well, where ``access_mode=read_only`` already refuses
        a row change — one posture for the reader however the slot was granted.

        The refusal is worded by this server, so ``query`` answers in its own
        words rather than DuckDB's, which speak of a transaction to an agent
        that opened none (#116). A query that writes all the same, such as
        ``SELECT nextval('s')``, is still refused by the read-only transaction,
        in DuckDB's words.
        """
        event.listen(engine, "connect", _begin_read_only)

        @event.listens_for(engine, "do_execute")
        def _with_parameters(cursor, statement, _parameters, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            connection = context.root_connection.connection.dbapi_connection
            _refuse_duckdb_non_read(refusal, connection, statement)
            return False

        @event.listens_for(engine, "do_execute_no_params")
        def _without_parameters(cursor, statement, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            connection = context.root_connection.connection.dbapi_connection
            _refuse_duckdb_non_read(refusal, connection, statement)
            return False

    def optional_aggregates(self) -> Mapping[str, Any]:
        """Both of them, driven against the engine rather than read about.

        DuckDB spells the median ``median(col)`` directly, so no
        ``WITHIN GROUP`` clause is needed here even though it also accepts
        ``percentile_cont``. ``stddev_samp`` is the sample standard deviation,
        which is the one worth reporting over a loaded file: a table is a sample
        of the thing it describes far more often than it is the whole population.
        """
        return {"median": func.median, "stddev": func.stddev_samp}

    def builds_indexes(self) -> bool:
        return False

    def build_index(
        self, name: str, table: Table, columns: Sequence[str]
    ) -> tuple[Index, tuple[str, ...]]:
        """Refused, because nothing here could find the index again.

        Unlike the four backends that refuse before it, DuckDB *does* build the
        index — ``CREATE INDEX`` succeeds and ``duckdb_indexes()`` lists it. The
        gap is in the adapter: duckdb-engine 0.17.0 answers every index
        reflection with ``[]`` and a warning that it does not support one, so
        ``directory`` would never list the index and ``drop`` would never find
        it. Reading ``duckdb_indexes()`` directly is no way round it, because
        that view gives an index's columns only as one string of SQL —
        measured on DuckDB 1.5.5, an index on ``("Dept x", n)`` reads
        ``['"Dept x"', n]``, one on a bare ``name`` reads ``['"name"']``,
        quoted, and one on ``lower(name)`` reads ``['(lower("name"))']`` — and
        reflecting columns out of that would mean parsing SQL, which fails
        open on the first spelling it was not written for (issue #112).

        The refusal says what DuckDB does instead, in its documentation's
        words rather than this server's: a min-max index is created
        automatically for the columns of all general-purpose types, and an
        explicit (ART) index *"does not affect the performance of join,
        aggregation, and sorting queries"*
        (https://duckdb.org/docs/current/sql/indexes.html and
        https://duckdb.org/docs/current/guides/performance/indexing.html).
        """
        article = article_for(self.name).capitalize()
        raise UnsupportedOperation(
            f"{article} {self.name} datasource can build an index on "
            f"{', '.join(columns)}, but this adapter cannot list or find an index "
            f"afterwards, so directory could not show it and drop could not "
            f"remove it — creating one would hand back a name nothing else here "
            f"can use. {self.name} already keeps a min-max index on the columns "
            f"of {table.name} by itself, and its documentation says an explicit "
            f"index does not speed up a join; run the join as it stands."
        )


# ---------------------------------------------------------------------------
# One statement, for every reader gate that decides it without a lexer
# ---------------------------------------------------------------------------


def _semicolon_refusal(why: str, rephrase: str) -> str:
    """What a reader gate says when :func:`_one_statement` refuses.

    It must be true in every case it fires, and it fires on ``SELECT 'a;b'`` —
    one statement — as surely as on ``SELECT 1; SELECT 2``. So it names the
    ``;``, not a second statement, says why a quoted one is refused as well,
    and offers a way to put a ``;`` in the data.

    ``why`` is what a second statement would do on that server; ``rephrase``
    is an expression answering ``;``, with an example. A rephrase is offered
    only by a gate whose every engine ran it through ``query``, measured.
    """
    return (
        "put a ';' before its end (refused even inside a quoted string or a "
        "comment: this server does not parse SQL to tell that ';' from one "
        f"ending a statement, {why}; for a ';' in the data write {rephrase})"
    )


def _quoted_halves(
    text: str, join: Callable[[str, str], str], passes: Callable[[str], bool]
) -> str | None:
    """``text`` as two quoted halves joined back together, for a rephrase.

    A refused word in a quoted string is refused because the gate matches
    words, not because the string does anything; split in two, no whole word
    is left to match, and the engine joins the halves back into ``text``.
    ``join`` is the family's concatenation, ``passes`` the gate's own check of
    ``SELECT <rephrase>``: the split nearest the middle that the gate admits is
    offered, so a rephrase is never one the gate itself would refuse.
    ``None`` where no split passes, a one-character text among them, and then
    no rephrase is offered at all.
    """
    middle = len(text) // 2
    for offset in range(len(text)):
        for cut in (middle - offset, middle + offset):
            if 0 < cut < len(text):
                rephrase = join(f"'{text[:cut]}'", f"'{text[cut:]}'")
                if passes(f"SELECT {rephrase}"):
                    return rephrase
    return None


def _quoted_text_refusal(text: str, why: str, rephrase: str) -> str:
    """What a gate says when the text it refuses is only inside a quoted string.

    Worded to follow "This statement asks to", like every refusal. It names
    the text, says why it is refused even there, and gives the rephrase
    :func:`_quoted_halves` measured against the gate (#135).
    """
    return (
        f"hold {text} in a quoted string (refused wherever it appears, a quoted "
        f"string included, because {why}; for that text in the data write it in "
        f"two halves, as in SELECT {rephrase})"
    )


def _masked(sql: str, opaque: re.Pattern[str]) -> str:
    """``sql`` with what ``opaque`` matches masked, keeping every length.

    A comment becomes spaces and a string or a quoted name a run of ``x``, so a
    word inside one is no longer a word of the statement, and a quoted name
    still reads as a name (:func:`_without_row_locks`).
    """

    def mask(found: re.Match[str]) -> str:
        quoted = found.group(0)
        filler = " " if quoted.startswith(("--", "/*", "#")) else "x"
        return filler * len(quoted)

    return opaque.sub(mask, sql)


def _without_row_locks(
    sql: str, opaque: re.Pattern[str], clause: re.Pattern[str]
) -> tuple[str, str] | None:
    """``sql``'s first row-locking clause and ``sql`` with every one removed (#144).

    ``None`` where it holds none. Used only to word a read-only session's
    refusal of a locking read, never to decide one: the session refuses, and
    this says what it refused and gives the same read without the locks.
    ``opaque`` is what the family's grammar writes that a clause can sit beside
    without being one, its strings, quoted names and comments; ``clause`` is a
    locking clause in that grammar, with the whitespace before it.

    The opaque parts are masked before the clause is looked for, so
    ``'FOR UPDATE'`` in the data is not one, and they keep their length, so a
    clause found in the mask is cut out of the statement as written. A comment
    is masked with spaces and the rest with ``x``, so a quoted name in an
    ``OF`` list still reads as a name. The clause is named with its whitespace
    collapsed, as the statement would read on one line.
    """
    clauses = list(clause.finditer(_masked(sql, opaque)))
    if not clauses:
        return None
    first = clauses[0]
    named = " ".join(sql[first.start() : first.end()].split())
    kept, start = [], 0
    for found in clauses:
        kept.append(sql[start : found.start()])
        start = found.end()
    kept.append(sql[start:])
    return named, "".join(kept)


def _one_statement(sql: str) -> bool:
    """Whether ``sql`` holds no ``;`` but an optional trailing one.

    Decided without lexing, and failing closed: a ``;`` inside a literal or a
    comment costs the caller that read, and never lets a second statement
    through. Each gate using it says why a second statement matters on its
    server; :func:`_monetdb_single_statement` was the first.
    """
    return ";" not in sql.rstrip().removesuffix(";")


# ---------------------------------------------------------------------------
# PostgreSQL, and the engines that answer on its wire with its catalog
# ---------------------------------------------------------------------------


#: PostgreSQL's code for "cannot execute … in a read-only transaction",
#: ``read_only_sql_transaction``. CockroachDB answers a write with it too.
_PG_READ_ONLY = "25006"

#: PostgreSQL's code for a statement it cannot parse, ``syntax_error``.
_PG_SYNTAX_ERROR = "42601"

#: libpq's ``PGRES_FATAL_ERROR``, the status of a result the server refused.
_PGRES_FATAL_ERROR = 7


@cache
def _psycopg2_libpq() -> ctypes.CDLL:
    """libpq exactly as psycopg 2 loaded it, with the calls a Parse needs (#143).

    Resolved through psycopg 2's own extension module rather than by a library
    name: looking a symbol up in a loaded library also searches the libraries it
    depends on, so ``PQprepare`` here is the libpq that made the ``PGconn`` behind
    ``connection.pgconn_ptr``, never another copy whose structures could differ
    (``psycopg2-binary`` bundles its own, and psycopg 3 may carry a second).
    """
    from psycopg2 import _psycopg

    libpq = ctypes.CDLL(_psycopg.__file__)
    libpq.PQprepare.restype = ctypes.c_void_p
    libpq.PQprepare.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    libpq.PQresultStatus.argtypes = [ctypes.c_void_p]
    libpq.PQresultErrorField.restype = ctypes.c_char_p
    libpq.PQresultErrorField.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libpq.PQresultErrorMessage.restype = ctypes.c_char_p
    libpq.PQresultErrorMessage.argtypes = [ctypes.c_void_p]
    libpq.PQclear.argtypes = [ctypes.c_void_p]
    return libpq


def _parsed_by_psycopg3(driver: Any, sql: str) -> str | None:
    """The server's syntax error for ``sql`` from a Parse on psycopg 3's connection."""
    from psycopg import errors, pq

    encoding = driver.info.encoding
    result = driver.pgconn.prepare(b"", sql.encode(encoding))
    if result.status != pq.ExecStatus.FATAL_ERROR:
        return None
    code = result.error_field(pq.DiagnosticField.SQLSTATE)
    if code != _PG_SYNTAX_ERROR.encode():
        return None
    return str(errors.error_from_result(result, encoding=encoding))


def _parsed_by_psycopg2(driver: Any, sql: str) -> str | None:
    """The same Parse on psycopg 2's connection, through the libpq it loaded.

    The server's message comes back as libpq words it, ``ERROR:  …``; the
    severity is taken off, as psycopg does, so the answer reads as it does on
    the psycopg 3 engines.
    """
    from psycopg2.extensions import encodings

    encoding = encodings.get(driver.encoding, "utf-8")
    libpq = _psycopg2_libpq()
    result = libpq.PQprepare(driver.pgconn_ptr, b"", sql.encode(encoding), 0, None)
    try:
        if libpq.PQresultStatus(result) != _PGRES_FATAL_ERROR:
            return None
        if libpq.PQresultErrorField(result, ord("C")) != _PG_SYNTAX_ERROR.encode():
            return None
        message = (libpq.PQresultErrorMessage(result) or b"").decode(
            encoding, errors="replace"
        )
        severity = (libpq.PQresultErrorField(result, ord("S")) or b"").decode(
            encoding, errors="replace"
        )
    finally:
        libpq.PQclear(result)
    return message.removeprefix(f"{severity}:  ").rstrip("\n")


#: The schemas the server itself provides. A function in one of these is the
#: server's own, and is refused only if it can act past the transaction.
_PG_SYSTEM_SCHEMAS = frozenset({"pg_catalog", "information_schema"})

#: Return types that mark a function nothing can call from SQL: trigger
#: functions, and the handlers behind access methods, languages, foreign data
#: wrappers and ``TABLESAMPLE``. Left out so their names cost no read —
#: ``system`` and ``bernoulli`` are both volatile, and both ordinary words.
_PG_UNCALLABLE_RETURNS = (
    "trigger",
    "event_trigger",
    "internal",
    "table_am_handler",
    "index_am_handler",
    "language_handler",
    "tsm_handler",
    "fdw_handler",
)

#: Every callable function, with what decides whether a read may call it.
#: ``pg_depend`` marks a function an extension installed (``deptype 'e'``);
#: ``current_schemas(false)`` is the search path without the implicit
#: ``pg_catalog``, which is where an unqualified name can resolve. Runs on
#: PostgreSQL 16, YugabyteDB 2.25 and CockroachDB 26.3, measured;
#: CockroachDB has no ``internal`` type, hence the names rather than casts.
_PG_FUNCTIONS = (
    "SELECT n.nspname, p.proname, p.provolatile = 'v', "
    "EXISTS (SELECT 1 FROM pg_catalog.pg_depend d "
    "WHERE d.classid = 'pg_catalog.pg_proc'::regclass "
    "AND d.objid = p.oid AND d.deptype = 'e'), "
    "n.nspname = ANY (current_schemas(false)) "
    "FROM pg_catalog.pg_proc p "
    "JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace "
    "WHERE p.prorettype NOT IN (SELECT oid FROM pg_catalog.pg_type "
    f"WHERE typname IN ({', '.join(repr(t) for t in _PG_UNCALLABLE_RETURNS)}))"
)

#: Each operator whose function is not a plain read of the server's own,
#: with that function described as :data:`_PG_FUNCTIONS` describes one. An
#: operator is another spelling of a function call, one a name cannot catch.
_PG_OPERATORS = (
    "SELECT o.oprname, n.nspname, p.proname, p.provolatile = 'v', "
    "EXISTS (SELECT 1 FROM pg_catalog.pg_depend d "
    "WHERE d.classid = 'pg_catalog.pg_proc'::regclass "
    "AND d.objid = p.oid AND d.deptype = 'e'), "
    "n.nspname = ANY (current_schemas(false)) "
    "FROM pg_catalog.pg_operator o "
    "JOIN pg_catalog.pg_proc p ON p.oid = o.oprcode "
    "JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace "
    "WHERE n.nspname NOT IN ('pg_catalog', 'information_schema') "
    "OR p.provolatile = 'v'"
)

#: Each cast whose function is not a plain read of the server's own, once
#: for each type it converts from or to that is not the server's own: that
#: type's oid, then the function as :data:`_PG_FUNCTIONS` describes one. A
#: cast is a function call no word names, ``1::money_c`` or, implicitly,
#: ``c + 1`` over a column of the type (#125).
_PG_CASTS = (
    "SELECT t.oid, n.nspname, p.proname, p.provolatile = 'v', "
    "EXISTS (SELECT 1 FROM pg_catalog.pg_depend d "
    "WHERE d.classid = 'pg_catalog.pg_proc'::regclass "
    "AND d.objid = p.oid AND d.deptype = 'e'), "
    "n.nspname = ANY (current_schemas(false)) "
    "FROM pg_catalog.pg_cast c "
    "JOIN pg_catalog.pg_proc p ON p.oid = c.castfunc "
    "JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace "
    "JOIN pg_catalog.pg_type t ON t.oid IN (c.castsource, c.casttarget) "
    "JOIN pg_catalog.pg_namespace tn ON tn.oid = t.typnamespace "
    "WHERE tn.nspname NOT IN ('pg_catalog', 'information_schema') "
    "AND (n.nspname NOT IN ('pg_catalog', 'information_schema') "
    "OR p.provolatile = 'v')"
)

#: Each cast between two of the server's own types whose function is not a
#: plain read of the server's own: the source and target types as the server
#: prints them, then the function as :data:`_PG_FUNCTIONS` describes one. The
#: rows :data:`_PG_CASTS` leaves out, because no word can refuse them: a read
#: naming ``bytea`` or ``integer`` is every read of them, and an implicit cast
#: runs with neither named (#129, :func:`_pg_server_cast_refusal`).
_PG_SERVER_CASTS = (
    "SELECT pg_catalog.format_type(c.castsource, NULL), "
    "pg_catalog.format_type(c.casttarget, NULL), "
    "n.nspname, p.proname, p.provolatile = 'v', "
    "EXISTS (SELECT 1 FROM pg_catalog.pg_depend d "
    "WHERE d.classid = 'pg_catalog.pg_proc'::regclass "
    "AND d.objid = p.oid AND d.deptype = 'e'), "
    "n.nspname = ANY (current_schemas(false)) "
    "FROM pg_catalog.pg_cast c "
    "JOIN pg_catalog.pg_proc p ON p.oid = c.castfunc "
    "JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace "
    "JOIN pg_catalog.pg_type s ON s.oid = c.castsource "
    "JOIN pg_catalog.pg_namespace sn ON sn.oid = s.typnamespace "
    "JOIN pg_catalog.pg_type g ON g.oid = c.casttarget "
    "JOIN pg_catalog.pg_namespace gn ON gn.oid = g.typnamespace "
    "WHERE sn.nspname IN ('pg_catalog', 'information_schema') "
    "AND gn.nspname IN ('pg_catalog', 'information_schema') "
    "AND (n.nspname NOT IN ('pg_catalog', 'information_schema') "
    "OR p.provolatile = 'v')"
)

#: Where type ``t`` (in namespace ``tn``) is, and whether the server has a
#: type of the same name: what :class:`_PGType` needs to choose its word.
_PG_TYPE_PLACE = (
    "tn.nspname = ANY (current_schemas(false)), "
    "EXISTS (SELECT 1 FROM pg_catalog.pg_type s "
    "JOIN pg_catalog.pg_namespace sn ON sn.oid = s.typnamespace "
    "WHERE s.typname = t.typname "
    "AND sn.nspname IN ('pg_catalog', 'information_schema'))"
)

#: Every type that is not the server's own, as :class:`_PGType` describes
#: one, keyed by oid. A table's or view's row is a type of the same name.
_PG_TYPES = (
    "SELECT t.oid, tn.nspname, t.typname, "
    f"{_PG_TYPE_PLACE}, "
    "coalesce(r.relkind IN ('r', 'p', 'v', 'm', 'f'), false) "
    "FROM pg_catalog.pg_type t "
    "JOIN pg_catalog.pg_namespace tn ON tn.oid = t.typnamespace "
    "LEFT JOIN pg_catalog.pg_class r ON r.oid = t.typrelid "
    "WHERE tn.nspname NOT IN ('pg_catalog', 'information_schema')"
)

#: Each ``CHECK`` on a domain that is not the server's own: the domain's oid,
#: the domain as :class:`_PGType` describes one (less ``is_relation``, which a
#: domain never is), then the check as the server prints it. A value becoming
#: one of the domain runs its checks, and no word in the statement need name
#: what they call: ``1::d``, or ``ROW(1)::t`` for a table with a column of it
#: (#126). Answers on PostgreSQL 16, YugabyteDB 2.25 and openGauss 7.0.0-RC3,
#: measured; CockroachDB 26.3 answers it too, and refuses a function or a
#: sequence in a domain's check, so nothing it returns there reaches one.
#: Asked from the domains outwards: only a domain has a ``contypid``, and
#: CockroachDB, whose catalog is built per question, answers in 7.8 ms
#: instead of 14.8 when it starts from ``pg_constraint``.
_PG_DOMAIN_CHECKS = (
    "SELECT t.oid, tn.nspname, t.typname, "
    f"{_PG_TYPE_PLACE}, "
    "coalesce(pg_catalog.pg_get_constraintdef(c.oid), '') "
    "FROM pg_catalog.pg_type t "
    "JOIN pg_catalog.pg_namespace tn ON tn.oid = t.typnamespace "
    "JOIN pg_catalog.pg_constraint c ON c.contypid = t.oid "
    "WHERE t.typtype = 'd' AND c.contype = 'c' "
    "AND tn.nspname NOT IN ('pg_catalog', 'information_schema')"
)

#: (type, part) for each type built from another: an array from its element,
#: a domain from its base, a composite (a table's row among them) from each
#: of its columns. A value of the whole carries a value of the part, and
#: with it the part's casts.
_PG_TYPE_PARTS = (
    "SELECT oid, typelem FROM pg_catalog.pg_type WHERE typelem <> 0 "
    "UNION ALL "
    "SELECT oid, typbasetype FROM pg_catalog.pg_type WHERE typbasetype <> 0 "
    "UNION ALL "
    "SELECT t.oid, a.atttypid FROM pg_catalog.pg_type t "
    "JOIN pg_catalog.pg_attribute a ON a.attrelid = t.typrelid "
    "WHERE t.typrelid <> 0 AND a.attnum > 0 AND NOT a.attisdropped"
)

#: SQL that runs whenever a relation is read, by the relation's name: a
#: view's definition, and a row-security policy's ``USING`` expression for
#: reads. A view whose definition is withheld comes back empty, and an empty
#: definition counts as reaching (:func:`_reachable_names`).
_PG_VIEWS = "SELECT viewname, coalesce(definition, '') FROM pg_catalog.pg_views"
_PG_RELATIONS = (
    f"{_PG_VIEWS} UNION ALL "
    "SELECT tablename, qual FROM pg_catalog.pg_policies "
    "WHERE cmd IN ('SELECT', 'ALL') AND qual IS NOT NULL"
)

#: :data:`_PG_RELATIONS` for openGauss, which has no ``pg_policies``: its
#: policies are in ``pg_rlspolicies``, under other column names, with the
#: same ``SELECT`` / ``ALL`` / ``UPDATE`` … commands. Measured on 7.0.0-RC3.
_OPENGAUSS_RELATIONS = (
    f"{_PG_VIEWS} UNION ALL "
    "SELECT tablename, policyqual FROM pg_catalog.pg_rlspolicies "
    "WHERE policycmd IN ('SELECT', 'ALL') AND policyqual IS NOT NULL"
)

#: Volatile catalog functions a read may still call, because they are
#: volatile only in that their answer varies: randomness, the clock, and how
#: large something is. ``version`` is volatile on CockroachDB, and its
#: SQLAlchemy dialect asks for it on every new connection.
_PG_ANSWER_VARIES = frozenset(
    {
        "random",
        "random_normal",
        "array_sample",
        "array_shuffle",
        "gen_random_uuid",
        "clock_timestamp",
        "timeofday",
        "version",
        "pg_database_size",
        "pg_relation_size",
        "pg_table_size",
        "pg_total_relation_size",
        "pg_indexes_size",
        "pg_tablespace_size",
    }
)

#: Catalog functions that read whatever relation their argument names. Not
#: volatile, but a computed name reaches a view no word in the statement
#: names: ``table_to_xml``, ``schema_to_xml`` and ``database_to_xml`` and
#: their ``xmlschema`` forms. (``query_to_xml`` and ``cursor_to_xml`` run SQL
#: text and are volatile already.)
_PG_RUNS_WHAT_IT_IS_GIVEN = re.compile(r"_to_xml", re.IGNORECASE)

#: A quoted identifier spelled with Unicode escapes, ``U&"lo\005fexport"``,
#: which names ``lo_export`` without containing the word. Measured: it ran.
_PG_UNICODE_IDENTIFIER = re.compile(r'u&"', re.IGNORECASE)

#: A string constant as PostgreSQL writes one: ``'…'`` with ``''`` for a quote,
#: ``E'…'`` with backslash escapes, and ``$tag$…$tag$``. Used only to word a
#: refusal (#135), never to decide one: what the gate refuses is decided on the
#: whole statement, a quoted string included.
_PG_LITERAL = re.compile(
    r"(?<![\w$])[Ee]'(?:[^'\\]|\\.|'')*'"
    r"|'(?:[^']|'')*'"
    r"|(?<![\w$])\$([A-Za-z_]\w*|)\$[\s\S]*?\$\1\$"
)

#: What a locking clause can sit beside without being one: a string constant
#: (:data:`_PG_LITERAL`, first, because its dollar quote refers to its own
#: group 1), a quoted identifier, and a comment.
_PG_OPAQUE = re.compile(
    _PG_LITERAL.pattern + r'|"(?:[^"]|"")*"' + r"|--[^\n]*|/\*[\s\S]*?\*/"
)

#: A row-locking clause as PostgreSQL's grammar writes one, with the
#: whitespace before it: ``FOR`` and a lock strength, an optional ``OF`` and
#: the tables it names, and an optional ``NOWAIT`` or ``SKIP LOCKED``. Matched
#: on a statement whose strings, quoted names and comments are masked
#: (:func:`_pg_without_row_locks`), so a quoted name in the ``OF`` list is a
#: run of word characters there.
_PG_LOCKING_CLAUSE = re.compile(
    r"\s*\bFOR\s+(?:UPDATE|NO\s+KEY\s+UPDATE|SHARE|KEY\s+SHARE)\b"
    r"(?:\s+OF\s+[\w$.]+(?:\s*,\s*[\w$.]+)*)?"
    r"(?:\s+(?:NOWAIT|SKIP\s+LOCKED)\b)?",
    re.IGNORECASE,
)


def _pg_without_row_locks(sql: str) -> tuple[str, str] | None:
    """:func:`_without_row_locks` in PostgreSQL's grammar (#144)."""
    return _without_row_locks(sql, _PG_OPAQUE, _PG_LOCKING_CLAUSE)


@dataclass(frozen=True)
class _PGFunction:
    """One row of :data:`_PG_FUNCTIONS`."""

    schema: str
    name: str
    volatile: bool
    from_extension: bool
    on_search_path: bool


def _pg_refused_word(function: _PGFunction, system_names: set[str]) -> str | None:
    """The word that refuses a statement for reaching ``function``, if any.

    * **The server's own** (:data:`_PG_SYSTEM_SCHEMAS`): its name, when it is
      volatile and not in :data:`_PG_ANSWER_VARIES`, or when it reads a
      relation its argument names. Volatile rather than a list of known
      effects so that a function a later release adds is refused until
      someone has looked at it: ``lo_export``, ``pg_reload_conf``,
      ``pg_terminate_backend``, ``pg_advisory_lock``, ``set_config``,
      ``setval`` and ``query_to_xml`` all fall out of it.
    * **An extension's**: its name, when it is volatile. An extension is
      installed by an administrator from the server's own files, so its
      volatility is taken at its word, which is what keeps PostGIS readable
      while ``dblink_exec`` is not.
    * **Anyone else's**: always. Its body may do anything, whatever it
      declares. Its name, if its schema is on the search path; otherwise it
      can only be called qualified, so its name — unless that name is also
      one of the server's own, when the word is its schema's instead, so that
      ``crdb_internal.log`` does not cost every read of ``log(x)``.
    """
    if function.schema in _PG_SYSTEM_SCHEMAS:
        if _PG_RUNS_WHAT_IT_IS_GIVEN.search(function.name):
            return function.name
        if function.volatile and function.name not in _PG_ANSWER_VARIES:
            return function.name
        return None
    if function.from_extension:
        return function.name if function.volatile else None
    if function.on_search_path or function.name not in system_names:
        return function.name
    return function.schema


@dataclass(frozen=True)
class _PGType:
    """One row of :data:`_PG_TYPES`."""

    schema: str
    name: str
    on_search_path: bool
    named_like_a_system_type: bool
    is_relation: bool


def _pg_type_word(found: _PGType) -> str:
    """The word that refuses a statement for naming the type ``found``.

    Its name, or its schema's when the name is also one of the server's own
    types and the schema is off the search path, as :func:`_pg_refused_word`
    does for functions.
    """
    hidden = found.named_like_a_system_type and not found.on_search_path
    return found.schema if hidden else found.name


def _pg_outwards(seeds: Iterable[int], wholes: Mapping[int, set[int]]) -> set[int]:
    """``seeds``, and every type built from one of them, and so on outwards."""
    reached = set(seeds)
    frontier = list(reached)
    while frontier:
        for whole in wholes.get(frontier.pop(), ()):
            if whole not in reached:
                reached.add(whole)
                frontier.append(whole)
    return reached


def _pg_refused_types(
    converted: set[int],
    types: Mapping[int, _PGType],
    parts: Iterable[tuple[int, int]],
    checked: Iterable[int] = (),
) -> dict[str, str]:
    """The words, and why, for every type a read may not hold a value of.

    ``converted`` are the types a refused cast converts (:data:`_PG_CASTS`);
    ``checked`` are the domains whose ``CHECK`` reaches a refused name
    (:data:`_PG_DOMAIN_CHECKS`). Every type built from one of them — its
    arrays, its domains, the composites and tables with a column of it, and
    so on outwards — is refused with it, because a value of the whole
    reaches the cast or the check through the part: measured,
    ``SELECT c + 1 FROM t`` ran a cast's function with neither the type nor
    the function named (#125), and ``ROW(1)::t`` ran the check of a domain
    one of ``t``'s columns has (#126). A table is refused by its own name,
    which is its row type's; each type by :func:`_pg_type_word`. A type
    reached both ways is explained by its cast.
    """
    wholes: dict[int, set[int]] = {}
    for whole, part in parts:
        wholes.setdefault(part, set()).add(whole)
    checked = set(checked)
    by_cast = _pg_outwards(converted, wholes)
    by_check = _pg_outwards(checked, wholes) - by_cast
    words: dict[str, str] = {}
    for oid in sorted(by_cast | by_check):
        found = types.get(oid)
        if found is None:
            continue
        if oid in by_cast:
            why = (
                "a table or view with a column whose type converts through such "
                "a function"
                if found.is_relation
                else "a type whose conversion calls such a function"
            )
        elif oid in checked:
            why = "a domain whose CHECK calls such a function"
        elif found.is_relation:
            why = "a table or view with a column of a domain whose CHECK calls such a function"
        else:
            why = "a type built from a domain whose CHECK calls such a function"
        words.setdefault(_pg_type_word(found), why)
    return words


def _pg_converted_types(cursor: Any, system_names: set[str]) -> set[int]:
    """The types a cast calling a refused function converts, asked on ``cursor``."""
    cursor.execute(_PG_CASTS)
    return {
        oid
        for oid, *described in cursor.fetchall()
        if _pg_refused_word(_PGFunction(*described), system_names) is not None
    }


def _pg_type_catalog(
    cursor: Any,
) -> tuple[dict[int, _PGType], list[tuple[int, int]]]:
    """Every type that is not the server's own, and what each is built from."""
    cursor.execute(_PG_TYPES)
    types = {oid: _PGType(*described) for oid, *described in cursor.fetchall()}
    cursor.execute(_PG_TYPE_PARTS)
    return types, cursor.fetchall()


def _pg_typed_and_reaching(
    cursor: Any,
    named: set[str],
    converted: set[int],
    checks: list[tuple[int, str, str]],
    definitions: Mapping[str, str],
) -> tuple[dict[str, str], set[str]]:
    """The types a read may not hold a value of, and the names that reach a refused one.

    ``named`` are the refused functions and schemas; ``converted`` the types
    a refused cast converts; ``checks`` each domain's oid, word and ``CHECK``;
    ``definitions`` the SQL a read of each relation runs, the checks among
    them under their domains' words. Returns :func:`_pg_refused_types`'s
    words, and every view, policy or domain reaching one of these names or
    those types (:func:`_reachable_names`).

    Followed to a fixed point, because each half feeds the other: a domain
    whose check reaches a refused name is refused, and with it every type
    built from it, whose names a view or another domain's check may mention
    in turn. A domain sharing its word with a view that reaches is taken as
    reaching too, which costs reads and admits none. The types are read only
    when some cast or check needs them, so a server with neither pays for
    nothing beyond its casts and checks.
    """
    catalog = _pg_type_catalog(cursor) if converted else None
    typed = _pg_refused_types(converted, *catalog) if catalog else {}
    checked: set[int] = set()
    while True:
        refused = named | set(typed)
        reaching = _reachable_names(refused, definitions) - refused
        newly = {oid for oid, word, _ in checks if word in reaching} - checked
        if not newly:
            return typed, reaching
        checked |= newly
        catalog = catalog or _pg_type_catalog(cursor)
        typed = _pg_refused_types(converted, *catalog, checked)
        if set(typed) <= refused | reaching:
            # Nothing the pass above has not already followed: another would
            # reach the same names, and find no domain it has not checked.
            return typed, reaching - set(typed)


#: What the PostgreSQL-family reader says when :func:`_one_statement` refuses
#: (#127), on every engine :class:`PostgreSQLBackend` serves.
#:
#: The rephrase was measured through ``query`` on each of them — PostgreSQL
#: 16.15, CockroachDB 26.3.4, YugabyteDB 2.25.2.0 and openGauss 7.0.0-RC3 —
#: and ``tests/test_semicolon_refusal.py`` runs the example on all four.
#: ``chr`` is the server's own function, and the catalog half of the gate let
#: it through on all four.
_PG_SEMICOLON = _semicolon_refusal(
    "and the checks here are made for a single statement",
    "chr(59), as in SELECT 'a' || chr(59) || 'b'",
)


def _pg_shape_refusal(sql: str) -> str | None:
    """What ``sql``'s shape alone refuses it for, or ``None``.

    One statement, because psycopg 2 sends a named cursor's ``DECLARE`` as a
    simple query, where ``SELECT 1; COMMIT; …`` is three; psycopg 3 and
    CockroachDB refuse a second statement themselves. And no ``U&"…"``
    identifier, whose escapes spell a name no word match can see.
    """
    if not _one_statement(sql):
        return _PG_SEMICOLON
    if _PG_UNICODE_IDENTIFIER.search(sql):
        return (
            'spell a name with Unicode escapes (U&"…"), which hides what it '
            'names; write the name itself, quoted if it needs quotes, as in "data" '
            'for U&"d\\0061ta"'
        )
    return None


@dataclass(frozen=True)
class _PGUnsafe:
    """What a read on this server may not mention, and why each is refused."""

    words: dict[str, str]
    operators: dict[str, str]


def _pg_unsafe(cursor: Any, relations_query: str) -> _PGUnsafe:
    """Ask the catalog, on ``cursor``, for every word and operator a read may not use.

    ``relations_query`` is the backend's own :data:`_PG_RELATIONS`, the one
    question whose catalog differs between the engines here. A domain's
    checks are followed as a view's definition is, under the domain's word:
    a view's SQL runs when the view is read, a domain's checks when a value
    becomes one of it (:func:`_pg_typed_and_reaching`).
    """
    cursor.execute(_PG_FUNCTIONS)
    functions = [_PGFunction(*row) for row in cursor.fetchall()]
    system_names = {f.name for f in functions if f.schema in _PG_SYSTEM_SCHEMAS}
    refused = {
        word
        for function in functions
        if (word := _pg_refused_word(function, system_names)) is not None
    }
    converted = _pg_converted_types(cursor, system_names)
    cursor.execute(_PG_DOMAIN_CHECKS)
    checks = [
        (oid, _pg_type_word(_PGType(*described, is_relation=False)), check)
        for oid, *described, check in cursor.fetchall()
    ]
    cursor.execute(relations_query)
    sources = [*cursor.fetchall(), *((word, check) for _, word, check in checks)]
    definitions: dict[str, str] = {}
    for name, definition in sources:
        definitions[name] = f"{definitions.get(name, '')}\n{definition}".strip()
    typed, reaching = _pg_typed_and_reaching(
        cursor, refused, converted, checks, definitions
    )
    words = dict.fromkeys(
        refused,
        "a function that can change what a rollback does not undo, or read "
        "a relation named at run time",
    )
    words.update(
        dict.fromkeys(
            reaching,
            "a view or a row-security policy that calls such a function",
        )
    )
    for word in refused - {f.name for f in functions}:
        words[word] = "a schema holding a function that can do that"
    for word, why in typed.items():
        words.setdefault(word, why)
    cursor.execute(_PG_OPERATORS)
    operators = {}
    for symbol, *described in cursor.fetchall():
        function = _PGFunction(*described)
        if _pg_refused_word(function, system_names) is not None:
            operators[symbol] = function.name
    return _PGUnsafe(words=words, operators=operators)


def _pg_join(left: str, right: str) -> str:
    """Two quoted strings joined on the PostgreSQL family."""
    return f"{left} || {right}"


def _pg_quoted_refusal(sql: str, unsafe: _PGUnsafe) -> str | None:
    """:func:`_pg_refusal` worded as a shape rule, where only a quoted string trips it.

    ``None`` when the statement mentions a refused name or operator outside
    a string constant (:data:`_PG_LITERAL`), which may be a call: that keeps
    the refused-write wording. Otherwise the refusal says the text is refused
    even inside a string and gives the rephrase (#135).
    """
    if _pg_refusal(_PG_LITERAL.sub("''", sql), unsafe) is not None:
        return None
    word = _named_as_word(sql, set(unsafe.words))
    if word is not None:
        text = word
        why = f"the word names {unsafe.words[word]}"
    else:
        symbols = [symbol for symbol in sorted(unsafe.operators) if symbol in sql]
        if not symbols:
            return None
        text = symbols[0]
        why = f"the symbol is an operator calling {unsafe.operators[text]}"

    def passes(statement: str) -> bool:
        return _pg_shape_refusal(statement) is None and (
            _pg_refusal(statement, unsafe) is None
        )

    rephrase = _quoted_halves(text, _pg_join, passes)
    return None if rephrase is None else _quoted_text_refusal(text, why, rephrase)


def _pg_refusal(sql: str, unsafe: _PGUnsafe) -> str | None:
    """What ``sql`` is refused for by what it mentions, or ``None``."""
    word = _named_as_word(sql, set(unsafe.words))
    if word is not None:
        return (
            f"use {word}, {unsafe.words[word]} (refused wherever the name "
            f"appears as a word)"
        )
    for symbol, function in sorted(unsafe.operators.items()):
        if symbol in sql:
            return (
                f"use the operator {symbol}, which calls {function} (refused "
                f"wherever the symbol appears)"
            )
    return None


@dataclass(frozen=True)
class _PGServerCast:
    """One row of :data:`_PG_SERVER_CASTS`: a cast no word in a statement names."""

    source: str
    target: str
    function: _PGFunction


def _pg_refuses(function: _PGFunction) -> bool:
    """Whether the gate refuses a read reaching ``function``, by whatever word.

    :func:`_pg_refused_word` needs the server's own function names only to
    choose *which* word refuses an off-path function; whether there is one does
    not depend on them, so none are asked for.
    """
    return _pg_refused_word(function, system_names=set()) is not None


def _pg_server_casts(cursor: Any) -> list[_PGServerCast]:
    """Each cast between two of the server's own types whose function the gate refuses."""
    cursor.execute(_PG_SERVER_CASTS)
    casts = [
        _PGServerCast(source, target, _PGFunction(*described))
        for source, target, *described in cursor.fetchall()
    ]
    return [cast for cast in casts if _pg_refuses(cast.function)]


def _pg_server_cast_refusal(casts: Sequence[_PGServerCast]) -> str | None:
    """Why every statement is refused on a server holding ``casts``, or ``None``.

    **Fail-closed, on purpose** (#129). A cast between two of the server's own
    types runs its function whenever a value is converted: ``SELECT 1::bytea``
    names only ``bytea``, and one created ``AS IMPLICIT`` runs with no type
    named at all. Refusing the types by name would refuse every read of them
    and still miss the implicit form, so no per-statement word check can be
    sound while such a cast exists. Measured on openGauss 7.0.0-RC3, whose
    read-only session runs ``nextval``: ``SELECT 1::bytea`` through ``query``
    moved a sequence. The gate therefore refuses everything on that server,
    and says which casts an administrator must drop — the invariant CrateDB's
    reader gate already states (:func:`_cratedb_first_keyword`): being stricter
    than the server can cost a read, never a write.

    Only an administrator can create such a cast; the ``query`` caller cannot.
    """
    if not casts:
        return None
    ordered = sorted(casts, key=lambda cast: (cast.source, cast.target))
    named = "; ".join(
        f"from {cast.source} to {cast.target} through "
        f"{cast.function.schema}.{cast.function.name}"
        for cast in ordered
    )
    drops = ", ".join(f"DROP CAST ({cast.source} AS {cast.target})" for cast in ordered)
    one = len(ordered) == 1
    return (
        f"run on this server while it holds {'a cast' if one else 'casts'} between "
        f"two of its own types ({named}), through "
        f"{'a function' if one else 'functions'} that can change what a rollback "
        f"does not undo. Every statement on this datasource is refused while "
        f"{'it exists' if one else 'they exist'}, because such a cast can run "
        f"implicitly, with no word in a statement to catch it. An administrator "
        f"of the server must drop {'it' if one else 'them'} ({drops})"
    )


def _refuse_postgresql_write(
    refusal: Refusal,
    cursor: Any,
    statement: str,
    parameters: Any,
    context: Any,
    relations_query: str,
) -> None:
    """Refuse ``statement`` unless nothing in it can act past the read (#124).

    The read-only session refuses a row change, DDL, ``lo_import`` and
    ``nextval``. Measured on PostgreSQL 16 as a superuser, it still ran
    ``lo_export`` (a file written on the server), ``pg_reload_conf``,
    ``pg_terminate_backend``, ``pg_switch_wal`` and ``pg_advisory_lock``
    (held on the pooled connection after the rollback); YugabyteDB ran
    ``pg_reload_conf``, and CockroachDB ran ``setval``, whose sequence stayed
    moved. So every statement on the reader passes its shape
    (:func:`_pg_shape_refusal`), then the server itself
    (:func:`_pg_server_cast_refusal`: a cast between two of its own types
    through a refused function refuses everything, #129), and then what it
    mentions (:func:`_pg_unsafe`). Both are asked of the catalog on a cursor of
    its own, since ``cursor`` may be the named cursor about to be declared.

    **A statement with bound parameters is not checked, and is this server's
    own**, for the reason :func:`_refuse_mssql_write` gives.

    Raises the driver's own ``ProgrammingError``, a DBAPI ``Error``, so
    SQLAlchemy wraps it and ``query`` explains it as a refused write. Taken
    from the dialect rather than imported, because psycopg 2 and psycopg 3 can
    both reach this backend.
    """
    if parameters:
        return
    refused = _pg_shape_refusal(statement)
    shape = refused is not None
    if refused is None:
        catalog = cursor.connection.cursor()
        try:
            # Fail-closed on the whole datasource first: a cast between two of
            # the server's own types through a refused function can run with no
            # word to catch it (#129, docs/CONSTRAINTS.md §34, 2026-11-03).
            refused = _pg_server_cast_refusal(_pg_server_casts(catalog))
            if refused is None:
                unsafe = _pg_unsafe(catalog, relations_query)
                refused = _pg_refusal(statement, unsafe)
                if refused is not None:
                    quoted = _pg_quoted_refusal(statement, unsafe)
                    if quoted is not None:
                        refused, shape = quoted, True
        finally:
            catalog.close()
        if refused is None:
            return
    refusal.what = refused
    refusal.shape = shape
    raise context.dialect.loaded_dbapi.ProgrammingError(
        f"query may not {refusal.what}."
    )


@dataclass(frozen=True)
class PostgreSQLBackend(Backend):
    """PostgreSQL, and the engines that answer with its catalog.

    **A transaction that never commits is not the whole floor here** (#124).
    It undoes what a transaction can undo, and a function can act where a
    rollback does not reach: a file, the server's configuration, another
    session, a sequence, a session lock. So the reader is opened read-only
    and passes every statement through :func:`_refuse_postgresql_write`. Set
    once, at connect time, for the reason :class:`MySQLBackend` gives.

    Registered for CockroachDB's own dialect and for YugabyteDB as well,
    because both have the hole and both answer the catalog questions the gate
    asks, measured. CockroachDB does not report ``transaction_read_only`` as
    ``on`` inside a cursor, yet refuses a write there with 25006, measured.

    It is also registered to say who borrows the dialect. ``-YB-`` is read
    from a live YugabyteDB, not from its documentation:

    * YugabyteDB — ``PostgreSQL 15.12-YB-2.25.2.0-b0 on x86_64-pc-linux-gnu…``
    * PostgreSQL — ``PostgreSQL 16.14 on x86_64-pc-linux-musl…``

    The fragment is deliberately tighter than the obvious one. YugabyteDB's
    banner says ``yugabyte`` twice, once in the compiler's source URL, so
    matching that would work — but it is the *version* that identifies the
    engine, and a fragment that leans on a build detail is a fragment waiting to
    stop matching. ``-YB-`` is the part PostgreSQL itself can never carry.

    Greenplum belongs here too and is absent on purpose: it has no container
    yet, so no measured banner, and an entry guessed from documentation is the
    failure mode the whole table is written to avoid. openGauss is not an
    impostor at all: it has a dialect of its own (:class:`OpenGaussBackend`).
    """

    name: str = "postgresql"
    banner_query: ClassVar[str] = "SELECT version()"
    impostors: ClassVar[Mapping[str, str]] = {"-YB-": "yugabytedb"}

    #: What the gate asks for the SQL a read of each relation runs; the one
    #: catalog question that differs between the engines this class serves.
    relations_query: ClassVar[str] = _PG_RELATIONS

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        @event.listens_for(engine, "connect")
        def _posture(dbapi_conn, _record):  # noqa: ANN001 - SQLAlchemy's signature
            cursor = dbapi_conn.cursor()
            try:
                cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            finally:
                cursor.close()
            # psycopg opens a transaction around the SET; ending it here keeps
            # the posture and leaves the connection idle for the pool.
            dbapi_conn.commit()

        @event.listens_for(engine, "do_execute")
        def _with_parameters(cursor, statement, parameters, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_postgresql_write(
                refusal, cursor, statement, parameters, context, self.relations_query
            )
            return False

        @event.listens_for(engine, "do_execute_no_params")
        def _without_parameters(cursor, statement, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_postgresql_write(
                refusal, cursor, statement, None, context, self.relations_query
            )
            return False

    def optional_aggregates(self) -> Mapping[str, Any]:
        """The median and the sample standard deviation, both measured (#95).

        PostgreSQL has no ``median()``; ``percentile_cont`` is how it spells
        one. Measured on 16 through ``stats``: 5.5 over 1, 2, 4, 7, 10, 12 —
        the two middle values averaged — and exactly 100000000 as the median
        of the 20001 squares 0..20000², so it is exact rather than estimated.
        ``stddev_samp`` answered sqrt(98 / 5) over the same six values, the
        sample deviation; ``stddev_pop`` answered sqrt(98 / 6).

        Over an integer column ``stddev_samp`` returns ``numeric`` at a scale
        PostgreSQL chooses for itself: over the squares it answered 119266276
        where the exact value is 119266275.712 — nine significant figures, a
        relative difference of 2e-9. That is the engine's own rounding of its
        own answer, not an estimate, so it is reported as it stands.
        """
        return {"median": _percentile_cont_median, "stddev": func.stddev_samp}

    def denies_write(self, exc: Exception) -> bool:
        """``read_only_sql_transaction``, under either psycopg's name for the code."""
        origin = getattr(exc, "orig", exc)
        return any(
            getattr(origin, attribute, None) == _PG_READ_ONLY
            for attribute in ("sqlstate", "pgcode")
        )

    def without_row_locks(self, sql: str) -> tuple[str, str] | None:
        """The locking clause the read-only session refused with 25006 (#144).

        Measured on PostgreSQL 16: ``FOR UPDATE``, ``FOR NO KEY UPDATE``,
        ``FOR SHARE`` and ``FOR KEY SHARE``, with ``NOWAIT``, ``SKIP LOCKED`` or
        an ``OF`` list, each passed the gate and was refused by the session,
        and the same SELECT without the clause answered its five rows
        (docs/CONSTRAINTS.md §34). See :func:`_pg_without_row_locks`.
        """
        return _pg_without_row_locks(sql)

    def syntax_error(self, engine: Engine, sql: str, message: str) -> str | None:
        """The server's syntax error for ``sql``, from a parse that runs nothing (#143).

        The question is the extended protocol's *Parse* message, libpq's
        ``PQprepare`` on psycopg 3's connection, with the unnamed statement:
        the server parses and analyses ``sql`` and answers whether it could,
        and nothing is planned, bound or executed, so nothing it says can
        happen. It takes any kind of statement, where SQL's ``PREPARE`` takes
        a query or DML only and so could not tell ``CREATE`` from a typo.
        Only a syntax error (42601) is an answer; anything else, a missing
        table included, leaves the refused-kind reading as it was.

        psycopg 2, openGauss's driver, sends no Parse of its own, but it hands
        out its connection's libpq handle as ``pgconn_ptr``, and the same
        ``PQprepare`` is asked through the libpq psycopg 2 loaded. Measured on
        openGauss 7.0.0-RC3, that Parse answers ``FROM t`` with 42601 and lets
        ``CREATE``, ``DROP``, ``INSERT`` and ``SHOW`` parse, running none of
        them, where SQL ``PREPARE`` refused them all alike. A driver offering
        neither is not asked.
        """
        with engine.connect() as conn:
            driver = conn.connection.driver_connection
            if getattr(driver, "pgconn", None) is not None:
                return _parsed_by_psycopg3(driver, sql)
            if getattr(driver, "pgconn_ptr", None) is not None:
                return _parsed_by_psycopg2(driver, sql)
            return None


@dataclass(frozen=True)
class CockroachDBBackend(PostgreSQLBackend):
    """CockroachDB: PostgreSQL's gate and catalog answers, and its own median.

    Everything is inherited except the one answer CockroachDB gives
    differently, measured on 2026-11-03 (#95). Its ``percentile_cont`` is
    defined over floating point only: over a ``FLOAT8`` column it answered
    5.5, but over an ``INT`` or a ``DECIMAL`` column it refused the statement —
    ``unknown signature: percentile_cont_impl(decimal, int)`` — and with it the
    whole one-``SELECT`` profile. PostgreSQL converts such a column for itself;
    CockroachDB has to be asked to, which is
    :func:`_percentile_cont_median_of_doubles`.
    """

    name: str = "cockroachdb"

    def optional_aggregates(self) -> Mapping[str, Any]:
        """The median over the column cast to a double, and ``stddev_samp``.

        ``stddev_samp`` needs no cast: it answered sqrt(98 / 5) over the
        float, the integer and the ``DECIMAL(12, 4)`` column alike.
        """
        return {
            "median": _percentile_cont_median_of_doubles,
            "stddev": func.stddev_samp,
        }


@dataclass(frozen=True)
class OpenGaussBackend(PostgreSQLBackend):
    """openGauss, through its own dialect, with PostgreSQL's reader gate (#124).

    It had the hole, measured through ``query`` on a read-only attach:
    ``setval`` persisted, ``pg_advisory_lock`` stayed held, and
    ``pg_reload_conf`` and ``set_config`` ran. The gate fits it unchanged —
    the function and operator questions answer, the read-only session
    refuses a write with 25006 — except for row security, which lives in
    :data:`_OPENGAUSS_RELATIONS`'s catalog.

    No banner and no impostors: nobody else answers on this dialect, and a
    banner that does not begin ``PostgreSQL`` is why PostgreSQL's own dialect
    cannot reach openGauss at all (``tests/endpoints.py``).
    """

    name: str = "opengauss"
    banner_query: ClassVar[str] = ""
    impostors: ClassVar[Mapping[str, str]] = {}
    relations_query: ClassVar[str] = _OPENGAUSS_RELATIONS


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

#: The first keywords a read begins with on MySQL and MariaDB. ``(`` opens a
#: parenthesised ``SELECT``. ``EXPLAIN ANALYZE`` runs its statement, but only a
#: ``SELECT`` or ``TABLE`` (a write is refused by the read-only transaction),
#: and the names that statement uses are checked like any other. Everything
#: else is refused, ``SET``, ``DO``, ``CALL`` and ``HANDLER`` included.
_MYSQL_READS = frozenset(
    {"SELECT", "WITH", "TABLE", "VALUES", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "("}
)

#: What may stand before a statement's first keyword: whitespace and the three
#: comment forms. ``--`` opens a comment only when whitespace or a control
#: character follows it, so ``--x`` is not skipped. A ``/*!`` or ``/*M!``
#: comment is not a comment, it is SQL the server runs, and is never skipped.
_MYSQL_LEADING = re.compile(
    r"(?:\s+|--[\s\x00-\x1f][^\n]*(?:\n|$)|#[^\n]*(?:\n|$)|/\*(?!M?!)[\s\S]*?\*/)*",
    re.IGNORECASE,
)

#: An executable comment, MySQL's ``/*! … */`` and MariaDB's ``/*M! … */``.
_MYSQL_EXECUTABLE_COMMENT = re.compile(r"/\*M?!", re.IGNORECASE)

#: The two words that make a ``SELECT`` or a ``TABLE`` write a file.
_MYSQL_FILE_WRITE = re.compile(r"outfile|dumpfile", re.IGNORECASE)

#: The built-in that reads a file on the database server (#36). It is a read, so
#: the read-only transaction lets it run, and no catalog lists it.
_MYSQL_FILE_READ = "load_file"

#: ``LOAD_FILE`` as a call: the name as a word, then, past any whitespace, the
#: opening parenthesis or the start of a comment that could stand before one.
#: Backquoted too, so a quoted spelling is refused whether or not the server
#: resolves it to the built-in. A comment after the name is refused whatever
#: follows it, which costs a column called ``load_file`` with a comment after it
#: and never admits a call spelled ``LOAD_FILE/**/(…)``. A column, an alias or a
#: literal that merely holds the word, with no parenthesis after it, is a read.
_MYSQL_FILE_READ_CALL = re.compile(
    r"(?<![\w$])`?load_file`?\s*(?:\(|/\*|#|--)", re.IGNORECASE
)

#: Every stored function a read could call, outside ``sys`` (MySQL's own
#: helpers, which only an administrator can add to), and every view with its
#: definition. Both readable by an unprivileged user, measured on MySQL 8.0 and
#: MariaDB 11; a view's definition comes back empty without ``SHOW VIEW``.
_MYSQL_FUNCTIONS = (
    "SELECT ROUTINE_NAME FROM information_schema.ROUTINES "
    "WHERE ROUTINE_TYPE = 'FUNCTION' AND ROUTINE_SCHEMA <> 'sys'"
)
_MYSQL_VIEWS = (
    "SELECT TABLE_NAME, VIEW_DEFINITION FROM information_schema.VIEWS "
    "WHERE TABLE_SCHEMA <> 'sys'"
)

#: Where each server lists its loadable functions (UDFs, native code). Neither
#: is readable without privileges, measured: an unprivileged user is refused
#: both (1142), and MariaDB has no ``user_defined_functions`` (1146). What is
#: readable is used; what is not leaves the residual CONSTRAINTS §34 records.
_MYSQL_LOADABLE_FUNCTIONS = (
    "SELECT UDF_NAME FROM performance_schema.user_defined_functions",
    "SELECT name FROM mysql.func",
)


#: What the MySQL and MariaDB reader says when :func:`_one_statement` refuses
#: (#127).
#:
#: The rephrase was measured through ``query`` on MySQL 8.0.46 and MariaDB
#: 11.8.9, and ``tests/test_semicolon_refusal.py`` runs the example on both.
#: ``USING utf8mb4`` is not optional: ``CHAR(59)`` alone has the charset
#: ``binary`` on both servers, and ``query`` answers it with the text
#: ``0x3b``, not ``;`` (``CONCAT('a', CHAR(59), 'b')`` with ``0x613b62``).
#: ``CONCAT`` rather than ``||``, which both read as ``OR``: ``'a' || … || 'b'``
#: answered ``0``.
_MYSQL_SEMICOLON = _semicolon_refusal(
    "and the checks here are made for a single statement",
    "CHAR(59 USING utf8mb4), as in SELECT CONCAT('a', CHAR(59 USING utf8mb4), 'b')",
)


#: A string constant as MySQL writes one: ``'…'`` with ``''`` or a backslash
#: escape. A ``"…"`` string is left out, because under ``ANSI_QUOTES`` it is a
#: name. Used only to word a refusal (#135), never to decide one.
_MYSQL_LITERAL = re.compile(r"'(?:[^'\\]|\\.|'')*'")


#: What a locking clause can sit beside without being one on MySQL and
#: MariaDB: a string in single or double quotes (double quotes make a string
#: unless ``ANSI_QUOTES`` is set, and either way they are not a clause), a
#: backquoted name, and the three comment forms.
_MYSQL_OPAQUE = re.compile(
    r"'(?:[^'\\]|\\.|'')*'"
    r'|"(?:[^"\\]|\\.|"")*"'
    r"|`(?:[^`]|``)*`"
    r"|--[\s\x00-\x1f][^\n]*|#[^\n]*|/\*[\s\S]*?\*/"
)

#: A locking clause as MySQL 8.0 and MariaDB 11 write one, with the whitespace
#: before it: ``FOR UPDATE`` or ``FOR SHARE`` (MySQL only) with an optional
#: ``OF`` list, or ``LOCK IN SHARE MODE``, then an optional ``NOWAIT``,
#: ``SKIP LOCKED`` or ``WAIT n`` (MariaDB). Matched on a masked statement
#: (:func:`_without_row_locks`).
_MYSQL_LOCKING_CLAUSE = re.compile(
    r"\s*(?:\bFOR\s+(?:UPDATE|SHARE)\b(?:\s+OF\s+[\w$.]+(?:\s*,\s*[\w$.]+)*)?"
    r"|\bLOCK\s+IN\s+SHARE\s+MODE\b)"
    r"(?:\s+(?:NOWAIT|SKIP\s+LOCKED|WAIT\s+\d+)\b)?",
    re.IGNORECASE,
)


def _mysql_without_row_locks(sql: str) -> tuple[str, str] | None:
    """:func:`_without_row_locks` in MySQL's and MariaDB's grammar (#144)."""
    return _without_row_locks(sql, _MYSQL_OPAQUE, _MYSQL_LOCKING_CLAUSE)


def _mysql_join(left: str, right: str) -> str:
    """Two quoted strings joined on MySQL and MariaDB, where ``||`` is ``OR``."""
    return f"CONCAT({left}, {right})"


#: What the reader says when a statement calls ``LOAD_FILE`` (#36).
_MYSQL_FILE_READ_REFUSAL = (
    "read a file on the database server with LOAD_FILE(), which a read-only "
    "transaction does not stop (refused wherever LOAD_FILE is followed by a "
    "parenthesis or a comment, even inside a quoted string; to read a file of "
    "your own, attach it)"
)


def _mysql_shape_refusal(sql: str) -> str | None:
    """What ``sql``'s shape alone refuses it for, or ``None`` if it may be a read.

    The words of :func:`_mysql_shape`, without whether they are a shape rule.
    """
    shaped = _mysql_shape(sql)
    return None if shaped is None else shaped[0]


def _mysql_quoted_only(sql: str, rule: re.Pattern[str]) -> str | None:
    """The text ``rule`` matches, where it matches only inside quoted strings."""
    if rule.search(_MYSQL_LITERAL.sub("''", sql)) is not None:
        return None
    found = rule.search(sql)
    return None if found is None else found.group(0)


def _mysql_shape(sql: str) -> tuple[str, bool] | None:
    """What ``sql``'s shape alone refuses it for, and whether that is a shape rule.

    The second half is ``True`` where a read meets the rule as surely as a
    write: a ``;``, and an executable comment, ``OUTFILE``/``DUMPFILE`` or a
    ``LOAD_FILE(`` that appears only inside quoted strings, whose refusal
    gives the rephrase (#135). Anywhere else they may be what they look like,
    and keep the refused-write wording.

    Decided without lexing, each rule failing closed, for the reason
    :func:`_monetdb_single_statement` gives: a server that parses another
    server's grammar will one day parse it differently from that server.

    * **One statement.** PyMySQL sends one unless the URL's ``client_flag``
      asks for more, so any ``;`` but a trailing one is refused.
    * **No executable comment.** ``/*!80000 SET GLOBAL … */`` is a ``SET``.
    * **Neither OUTFILE nor DUMPFILE, anywhere.** Not even inside a quoted
      string, which costs that read and says so, so the caller can rephrase.
    * **No call to LOAD_FILE** (:data:`_MYSQL_FILE_READ_CALL`), which reads a
      file on the database server inside a read-only transaction (#36).
    * **A first keyword in** :data:`_MYSQL_READS`.
    """
    if not _one_statement(sql):
        return _MYSQL_SEMICOLON, True
    rules = (
        (
            _MYSQL_EXECUTABLE_COMMENT,
            "run an executable comment (/*! … */), which the server runs as SQL",
            "the server runs an executable comment as SQL",
        ),
        (
            _MYSQL_FILE_WRITE,
            "write a file with INTO OUTFILE or INTO DUMPFILE (the two words are "
            "refused anywhere in the statement, even inside a quoted string)",
            "OUTFILE and DUMPFILE make a statement write a file",
        ),
        (
            _MYSQL_FILE_READ_CALL,
            _MYSQL_FILE_READ_REFUSAL,
            "LOAD_FILE( reads a file on the database server",
        ),
    )
    for rule, refused, why in rules:
        if rule.search(sql) is None:
            continue
        quoted = _mysql_quoted_only(sql, rule)
        if quoted is None:
            return refused, False
        rephrase = _quoted_halves(
            quoted, _mysql_join, lambda statement: _mysql_shape(statement) is None
        )
        if rephrase is None:
            return refused, False
        return _quoted_text_refusal(quoted, why, rephrase), True
    rest = sql[_MYSQL_LEADING.match(sql).end() :]
    first = re.match(r"\(|[A-Za-z_]+|\S+", rest)
    if first is None:
        return "run a statement that does not begin with a keyword", False
    keyword = first.group(0).upper()
    if keyword in _MYSQL_READS:
        return None
    return f"run a {keyword} statement", False


def _named_as_word(sql: str, names: set[str]) -> str | None:
    """The first of ``names`` that ``sql`` mentions as a whole word, if any.

    A word, because ``fset`` is not in ``fset_total``; ``$`` counts as part of
    a word because MySQL and PostgreSQL identifiers may both contain it.
    Case-insensitive, because function names are, and a view's may be.
    Anywhere in the statement, a quoted string included: that costs a read
    naming one in a literal, and never admits a call spelled in a way this did
    not anticipate. Shared by the MySQL (#123) and PostgreSQL (#124) gates.
    Of several mentioned, the first in case-insensitive order is named.
    """
    return _word_finder(names)(sql)


#: A run of word characters as :func:`_named_as_word` counts them.
_WORD = re.compile(r"[\w$]+")


def _word_finder(names: Iterable[str]) -> Callable[[str], str | None]:
    """:func:`_named_as_word` for one set of ``names``, built once.

    Built once because the PostgreSQL gate asks it of every catalog view on
    every statement: a search per name per view cost 1.8 s a statement on
    PostgreSQL 16, measured (about 200 names, 140 views). A name made only of
    word characters matches exactly where a maximal run of them equals it, so
    each run of the statement is tried against one pattern holding every such
    name, in order, and the first alternative that matches is the first name.
    Any other name — a quoted view called ``my view`` — keeps its own search.
    Where the run and every such name are ASCII, the pattern's
    case-insensitive match is plain lower-case equality, so a lookup answers
    instead; Unicode case-folding (``K``, the Kelvin sign, matches ``k``)
    keeps the pattern for everything else.
    """
    ordered = sorted(names, key=str.lower)
    plain = [i for i, name in enumerate(ordered) if _WORD.fullmatch(name)]
    others = [i for i, name in enumerate(ordered) if not _WORD.fullmatch(name)]
    either = re.compile(
        "|".join(f"(?P<n{i}>{re.escape(ordered[i])})" for i in plain),
        re.IGNORECASE,
    )
    all_ascii = all(ordered[i].isascii() for i in plain)
    lowered: dict[str, int] = {}
    for i in plain:  # ascending, so each entry keeps its first name
        lowered.setdefault(ordered[i].lower(), i)

    def first_of_run(run: str) -> int | None:
        if all_ascii and run.isascii():
            return lowered.get(run.lower())
        match = either.fullmatch(run)
        return None if match is None else int(match.lastgroup.removeprefix("n"))

    def first(sql: str) -> str | None:
        found = len(ordered)
        if plain:
            for run in dict.fromkeys(_WORD.findall(sql)):  # in order, once each
                if (i := first_of_run(run)) is not None:
                    found = min(found, i)
        for i in others:
            if i >= found:
                break
            pattern = rf"(?<![\w$]){re.escape(ordered[i])}(?![\w$])"
            if re.search(pattern, sql, re.IGNORECASE):
                found = i
                break
        return ordered[found] if found < len(ordered) else None

    return first


def _reachable_names(functions: set[str], views: Mapping[str, str]) -> set[str]:
    """Every name through which a read could reach one of ``functions``.

    The functions, and each view whose definition mentions one of them or
    another such view, followed to a fixed point: a pass that adds a view
    is followed by another, which looks for that view too. A view whose
    definition is hidden (empty) counts as reaching one whenever there is
    any to reach. Shared by the MySQL (#123) and PostgreSQL (#124) gates.
    """
    unsafe = set(functions)
    changed = bool(unsafe)
    while changed:
        changed = False
        mentions = _word_finder(unsafe)
        for view, definition in views.items():
            if view in unsafe:
                continue
            if not definition or mentions(definition):
                unsafe.add(view)
                changed = True
    return unsafe


# The names these two had while only MySQL used them, which the frozen
# tests/test_mysql_read_gate.py imports.
_mysql_named = _named_as_word
_mysql_reachable_names = _reachable_names


def _mysql_unsafe_names(cursor: Any) -> set[str]:
    """Ask the catalog, on ``cursor``, for every name a read may not use."""
    import pymysql

    cursor.execute(_MYSQL_FUNCTIONS)
    functions = {name for (name,) in cursor.fetchall()}
    for listing in _MYSQL_LOADABLE_FUNCTIONS:
        try:
            cursor.execute(listing)
        except pymysql.err.MySQLError:
            continue
        functions |= {name for (name,) in cursor.fetchall()}
    cursor.execute(_MYSQL_VIEWS)
    views = {name: definition or "" for name, definition in cursor.fetchall()}
    # LOAD_FILE is followed into views like a stored function, so a view
    # reading a server file is refused by its name. The word itself is left
    # out of what a statement may not mention: the shape half refuses the
    # call, and a column or a literal holding the word is a read (#36).
    reaching = _reachable_names(functions | {_MYSQL_FILE_READ}, views)
    return reaching - ({_MYSQL_FILE_READ} - functions)


def _refuse_mysql_write(
    refusal: Refusal, cursor: Any, statement: str, parameters: Any
) -> None:
    """Refuse ``statement`` unless nothing in it can reach past a read (#123).

    The read-only transaction refuses a row change and DDL. What it does not
    refuse is checked here, before the statement runs: its shape
    (:func:`_mysql_shape_refusal`), then every name in it against the
    catalog's functions and the views that reach them
    (:func:`_mysql_unsafe_names`). Measured on MySQL 8.0 and MariaDB 11, a
    stored function called from a ``SELECT`` ran ``SET GLOBAL`` and wrote a
    file inside a read-only transaction; its ``INSERT`` was refused.

    **A statement with bound parameters is not checked, and is this server's
    own**, for the reason :func:`_refuse_mssql_write` gives: ``query`` sends
    the caller's SQL with no values (``tests/test_query_sends_no_parameters.py``).

    Raises PyMySQL's own ``ProgrammingError``, a DBAPI ``Error``, so
    SQLAlchemy wraps it and ``query`` explains it as a refused write.
    """
    if parameters:
        return
    shaped = _mysql_shape(statement)
    if shaped is None:
        unsafe = _mysql_unsafe_names(cursor)
        name = _named_as_word(statement, unsafe)
        if name is None:
            return
        shaped = _mysql_named_refusal(statement, name, unsafe)
    import pymysql

    refusal.what, refusal.shape = shaped
    raise pymysql.err.ProgrammingError(f"query may not {refusal.what}.")


def _mysql_named_refusal(sql: str, name: str, unsafe: set[str]) -> tuple[str, bool]:
    """Why ``sql`` is refused for naming ``name``, and whether that is a shape rule.

    A shape rule where every unsafe name ``sql`` mentions is inside a quoted
    string, which a read meets as surely as a call, with the rephrase (#135).
    """
    why = (
        "a stored or loadable function, or a view calling one, which can change "
        "the server even inside a read-only transaction"
    )
    named = f"use {name}, {why} (refused wherever the name appears as a word)"
    if _named_as_word(_MYSQL_LITERAL.sub("''", sql), unsafe) is not None:
        return named, False

    def passes(statement: str) -> bool:
        return _mysql_shape(statement) is None and (
            _named_as_word(statement, unsafe) is None
        )

    rephrase = _quoted_halves(name, _mysql_join, passes)
    if rephrase is None:
        return named, False
    return _quoted_text_refusal(name, f"{name} is {why}", rephrase), True


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

    **A read-only transaction is not the whole posture** (#123): it leaves
    files, server-wide settings and stored functions within reach, so every
    statement on the reader also passes :func:`_refuse_mysql_write` first.
    """

    name: str = "mysql"

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        @event.listens_for(engine, "connect")
        def _posture(dbapi_conn, _record):  # noqa: ANN001 - SQLAlchemy's signature
            with dbapi_conn.cursor() as cursor:
                cursor.execute("SET SESSION TRANSACTION READ ONLY")

        @event.listens_for(engine, "do_execute")
        def _with_parameters(cursor, statement, parameters, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_mysql_write(refusal, cursor, statement, parameters)
            return False

        @event.listens_for(engine, "do_execute_no_params")
        def _without_parameters(cursor, statement, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_mysql_write(refusal, cursor, statement, None)
            return False

    def denies_write(self, exc: Exception) -> bool:
        origin = getattr(exc, "orig", exc)
        arguments = getattr(origin, "args", ())
        return bool(arguments) and arguments[0] == _MYSQL_READ_ONLY

    def without_row_locks(self, sql: str) -> tuple[str, str] | None:
        """The locking clause the read-only session refused with 1792 (#144).

        Measured through ``query`` on MySQL 8.0 and MariaDB 11: ``FOR UPDATE``,
        with ``NOWAIT``, ``SKIP LOCKED`` or MariaDB's ``WAIT 1``, was refused,
        and the same SELECT without it answered its five rows. ``FOR SHARE``
        (MySQL) and ``LOCK IN SHARE MODE`` (both) are not refused: they ran
        and answered the five rows (docs/CONSTRAINTS.md §34). They are found
        here all the same, so a statement holding one beside a refused
        ``FOR UPDATE`` is given back with every clause removed.
        """
        return _mysql_without_row_locks(sql)

    def optional_aggregates(self) -> Mapping[str, Any]:
        """The sample standard deviation, and no median, on both engines (#95).

        ``stddev_samp`` answered sqrt(98 / 5) over 1, 2, 4, 7, 10, 12 in a float
        column on MySQL 8.0 and MariaDB 11. **Not** ``stddev``: on both it is
        the *population* deviation, sqrt(98 / 6), so the name that reads as the
        obvious one is the wrong statistic here.

        **The column is read as a double first, because MariaDB rounds.** Over
        an integer column MariaDB's ``stddev_samp`` answered **1.8708** where
        the deviation is 1.87082869…, and 119266275.712 over the 20001 squares:
        an exact-value argument gets an exact-value answer at MariaDB's own
        scale, four decimal places over the integer column, so a small
        deviation keeps few of its digits. Over a double the answer is a
        double. MySQL gave the full value
        either way. SQLAlchemy writes the cast as ``CAST(… AS DOUBLE)`` only
        for a server that has one (MySQL 8.0.17, MariaDB 10.4.5) and leaves it
        out, with a warning, for an older one — which is the rounded answer
        again, not a failure.

        Neither has a median that can sit in one aggregate ``SELECT``. MySQL
        has no ``median`` and no ``percentile_cont`` at all (``FUNCTION
        testdb.median does not exist``, and a syntax error at ``WITHIN
        GROUP``). On MariaDB both ``median(col)`` and ``percentile_cont(0.5)
        WITHIN GROUP (ORDER BY col)`` were syntax errors in an aggregate
        ``SELECT``.
        """
        return {"stddev": lambda column: func.stddev_samp(cast(column, Double))}

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

        **The budget is the whole key, not the part of it that needs a prefix.**
        A bounded ``VARCHAR`` in the same index spends the same 3072 bytes, so
        what it spends is subtracted before the rest is divided. Dividing the
        whole limit among the unbounded columns alone put a ``VARCHAR(600)``
        beside a ``TEXT`` at 3420 bytes and InnoDB refused the statement with
        error 1071 — measured, localdata#93. Only columns that declare a length
        can be charged for here; a fixed-width scalar is left at zero, which
        keeps this an estimate that can still be a little low, and the estimate
        is the one MySQL's own limit is written in.
        """
        unbounded = [
            column
            for column in columns
            if isinstance(table.c[column].type, (Text, LargeBinary))
        ]
        if not unbounded:
            return super().build_index(name, table, columns)

        bounded = {
            column: length
            for column in columns
            if column not in unbounded
            and (length := getattr(table.c[column].type, "length", None))
        }
        budget = _INNODB_KEY_BYTES - 4 * sum(bounded.values())
        if budget < 4 * len(unbounded):
            widest = max(bounded, key=bounded.__getitem__)
            raise UnsupportedOperation(
                f"Cannot index {', '.join(columns)} together on {self.name}: "
                f"{widest} alone is declared {bounded[widest]} characters wide, "
                f"which leaves nothing of the {_INNODB_KEY_BYTES}-byte key for "
                f"{', '.join(unbounded)}. Drop {widest} from the index, or index "
                f"it on its own."
            )

        prefix = max(1, min(_TEXT_PREFIX_CHARS, budget // (4 * len(unbounded))))
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

    def text_width(
        self,
        *,
        characters: int | None,
        utf16_units: int | None,
        utf8_bytes: int | None = None,
    ) -> int | None:
        """Characters, unless the widest value is more bytes than a ``VARCHAR2`` holds.

        SQLAlchemy writes ``String(n)`` as ``VARCHAR2(n CHAR)`` on every Oracle
        since 9, so the width counts characters and ``été`` fits the three it
        was measured at — measured on Oracle Free, where the session's own
        ``NLS_LENGTH_SEMANTICS`` is ``BYTE`` (#132). The 4,000 ceiling is
        another matter: on a database without extended string sizes it counts
        **bytes** whatever the semantics, and 1,400 CJK ideographs — 1,400
        characters, 4,200 bytes — were refused on insert into the
        ``VARCHAR2(1400 CHAR)`` they were given.

        So past the ceiling in bytes the byte count is the answer: it is wider
        than :data:`_VARCHAR2_MAX`, and :meth:`column_type` makes it the
        ``CLOB`` nothing narrower can be. ``utf8_bytes`` is ``None`` only for a
        caller that did not measure it, which then gets characters as before.
        """
        if utf8_bytes is not None and utf8_bytes > _VARCHAR2_MAX:
            return utf8_bytes
        return characters

    def unstorable_column_types(self) -> frozenset[str]:
        """Oracle has no time-of-day type.

        A bare time is a ``DATE`` with the date part ignored, or an
        ``INTERVAL`` — neither of which is the column that was asked for, so
        there is nothing a value could be put into and come back out of.
        """
        return frozenset({"Time"})

    def optional_aggregates(self) -> Mapping[str, Any]:
        """``MEDIAN`` and ``STDDEV_SAMP``, both measured (#95).

        On Oracle Database Free (``gvenzl/oracle-free:slim``), ``MEDIAN``
        answered 5.5 over 1, 2, 4, 7, 10, 12 — the two middle values averaged —
        3.5 and 6 over an integer and a ``NUMBER`` column, and exactly 100000000
        over the 20001 squares, so it is exact. ``STDDEV_SAMP`` answered
        sqrt(98 / 5) over the float, integer and ``NUMBER`` columns alike.
        Oracle's own ``STDDEV`` is the sample deviation too; ``STDDEV_SAMP``
        is used because its name says which one it is.
        """
        return {"median": func.median, "stddev": func.stddev_samp}

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


#: The widest ``NVARCHAR`` SQL Server will take before ``NVARCHAR(max)``, in
#: UTF-16 code units. Measured on SQL Server 2022: ``NVARCHAR(4000)`` is created
#: and ``NVARCHAR(4001)`` refused with "The size (4001) given to the parameter
#: 'v' exceeds the maximum allowed (4000)". ``NVARCHAR(max)`` past it is a large
#: object, which groups, sorts and compares correctly and cannot be an index key.
_NVARCHAR_MAX = 4000


#: The statement kinds SQL Server's own plan reports for a read, and the only
#: ones its reader will run. ``SELECT WITHOUT QUERY`` is a ``SELECT`` with no
#: ``FROM`` (``SELECT 1``); ``XML SELECT`` is one ending ``FOR XML``. Measured on
#: 2022 (16.x): ``SELECT … INTO`` reports ``SELECT INTO``, a ``DECLARE`` with a
#: value ``ASSIGN``, ``EXEC('…')`` ``EXECUTE STRING`` — none of them here.
_MSSQL_READS = frozenset({"SELECT", "SELECT WITHOUT QUERY", "XML SELECT"})

#: The namespace every ``SHOWPLAN_XML`` document is written in.
_SHOWPLAN = "{http://schemas.microsoft.com/sqlserver/2004/07/showplan}"

#: An operator that sends work to another server. A remote statement's own
#: kind is not in the local plan, so a read cannot be told from a write there;
#: refusing every remote operator is the fail-closed answer, and it costs a
#: linked-server read, never a local one.
_MSSQL_REMOTE_OPERATOR = re.compile(r'PhysicalOp="Remote')


def _mssql_batch_kinds(plan: str) -> list[str]:
    """The kinds of the statements the caller wrote, from one plan document.

    Only the batch's own statements, the children of each
    ``Batch/Statements``. A plan also carries the bodies of functions a
    statement calls: ``SELECT 1 FROM fn_listextendedproperty(…)`` plans one
    ``SELECT`` and, inside it, the function's dozens of ``INSERT`` into its
    own return table, which writes nothing that outlives the call. Counting
    those would refuse SQLAlchemy's own connect-time probe.
    """
    root = ElementTree.fromstring(plan)
    return [
        statement.get("StatementType", statement.tag.removeprefix(_SHOWPLAN))
        for statements in root.iterfind(
            f"{_SHOWPLAN}BatchSequence/{_SHOWPLAN}Batch/{_SHOWPLAN}Statements"
        )
        for statement in statements
    ]


def _mssql_planned_kinds(cursor: Any, statement: str) -> tuple[list[str], bool]:
    """Every statement kind SQL Server plans for ``statement``, without running it.

    ``SET SHOWPLAN_XML ON`` makes the server compile the batch and answer one
    plan per statement instead of executing anything, so the kinds come from
    SQL Server's own parser rather than from one of ours. Measured: with it on,
    ``COMMIT; INSERT …`` answers ``COMMIT TRANSACTION`` and ``INSERT`` and the
    row count does not move. Also answers whether any plan reaches another
    server. Switched off again whatever happens, on the same cursor, so the
    connection goes back to the pool executing statements.
    """
    cursor.execute("SET SHOWPLAN_XML ON")
    try:
        cursor.execute(statement)
        kinds: list[str] = []
        remote = False
        while True:
            for (plan,) in cursor.fetchall():
                kinds += _mssql_batch_kinds(plan)
                remote = remote or bool(_MSSQL_REMOTE_OPERATOR.search(plan))
            if not cursor.nextset():
                break
        return kinds, remote
    finally:
        cursor.execute("SET SHOWPLAN_XML OFF")


def _refuse_mssql_write(
    refusal: Refusal, cursor: Any, statement: str, parameters: Any
) -> None:
    """Refuse ``statement`` unless SQL Server plans it as exactly one read.

    One statement, because T-SQL runs a whole batch and needs no separator
    between statements: ``SELECT 1 SELECT 2`` is two, and ``COMMIT; INSERT …``
    ends the read transaction before writing outside it (#122). A batch of two
    reads is refused as well; it is not a write, but ``query`` answers one
    result, and the rule stays a count rather than a judgement.

    **A statement with bound parameters is not planned, and is this server's
    own.** ``SHOWPLAN_XML`` answers no plan at all for one (measured: pyodbc
    then reports "No results"), so asking would refuse SQLAlchemy's
    reflection. The caller's SQL cannot arrive with parameters:
    :meth:`loader.Workspace.query_stream` runs it as ``text(sql)`` with no
    values, so a ``:name`` in it is refused by SQLAlchemy before anything is
    sent, and a ``?`` reaches the driver with no values and fails there. Every
    parameterised statement on the reader is one this server composed.
    ``tests/test_query_sends_no_parameters.py`` pins that, so the day
    ``query`` takes values from its caller this gate is reconsidered. Raises
    pyodbc's own ``ProgrammingError``, a DBAPI ``Error``, so SQLAlchemy wraps
    it and ``query`` explains it as a refused write.
    """
    if parameters:
        return
    kinds, remote = _mssql_planned_kinds(cursor, statement)
    if len(kinds) == 1 and kinds[0] in _MSSQL_READS and not remote:
        return
    import pyodbc

    if remote:
        refusal.what = "reach another server"
    elif len(kinds) != 1:
        refusal.what = f"run {len(kinds)} statements in one batch"
    else:
        refusal.what = f"run {article_for(kinds[0])} {kinds[0]} statement"
    raise pyodbc.ProgrammingError(f"query may not {refusal.what}.")


@dataclass(frozen=True)
class MSSQLBackend(Backend):
    """SQL Server, which renames tables by procedure and has no usable ``TEXT``.

    Two things it cannot be asked in the portable way, and both are the kind
    this module exists for — a statement that means something else here, rather
    than one that is merely awkward.
    """

    name: str = "mssql"

    def column_type(self, declared: str, *, longest: int | None = None) -> TypeEngine:
        """``NVARCHAR`` sized from the data, because nothing narrower holds text.

        SQLAlchemy's portable ``Text`` renders ``TEXT``, which SQL Server has
        deprecated and will not let into an index key at all — so a loaded
        file's text columns could be neither indexed nor, on the way out, relied
        on. A sized column is the modern spelling.

        **Unicode, not ``VARCHAR``** (#128). A ``VARCHAR`` holds the column
        collation's code page and nothing else. Measured on SQL Server 2022
        under the default ``SQL_Latin1_General_CP1_CI_AS``: a CSV holding
        ``日本`` and ``😀`` loaded without an error and read back as ``??`` and
        ``??``, while ``été``, which code page 1252 has, survived. Under a UTF-8
        collation the same column refused ``été`` instead, because there a
        ``VARCHAR`` width counts bytes and the width measured here counted
        characters. ``NVARCHAR`` stores any string a Python ``str`` can hold,
        under every collation, and its width counts UTF-16 code units whatever
        the collation — which is what ``longest`` is in here
        (:meth:`text_width`).

        Comparison and ordering follow the database's collation, as they did for
        ``VARCHAR``, but under a ``SQL_`` collation the Unicode rules are not
        the code-page ones: measured, ``ORDER BY`` on ``NVARCHAR`` puts
        ``coop`` before ``co-op`` and ``'q`` beside ``q``, where ``VARCHAR`` put
        punctuation first.

        A nonclustered index key holds 1,700 bytes, which is 850 code units of
        ``NVARCHAR``. SQL Server refuses an index over a column holding a wider
        value, naming the limit, and creates one over a column only declared
        wider. SQL Server's warning for the latter is about later inserts, and
        no verb here writes into a table once it is made.
        """
        if declared != "TEXT":
            return super().column_type(declared, longest=longest)
        width = max(1, longest or 1)
        # Past 4000 the only spelling is NVARCHAR(max), which is a large object
        # again — the same trade Oracle makes at 4000, and for the same reason.
        # `Unicode()` rather than `UnicodeText()`: offline, the latter compiles
        # to the deprecated NTEXT.
        return Unicode(width) if width <= _NVARCHAR_MAX else Unicode()

    def text_width(
        self,
        *,
        characters: int | None,
        utf16_units: int | None,
        utf8_bytes: int | None = None,
    ) -> int | None:
        """UTF-16 code units, because that is what ``NVARCHAR(n)`` counts.

        An emoji is one character and two code units, and a width counted in
        characters is one short for each one a value holds: ``NVARCHAR(1)``
        refuses ``😀``, measured, and ``NVARCHAR(2)`` holds it.
        """
        return utf16_units

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

    def optional_aggregates(self) -> Mapping[str, Any]:
        """The sample standard deviation as SQL Server spells it, and no median (#95).

        ``STDEV`` is the sample deviation (``STDEVP`` the population one): it
        answered sqrt(98 / 5) over 1, 2, 4, 7, 10, 12 on SQL Server 2022, on
        the float, integer and ``NUMERIC(12, 4)`` columns alike, and exactly
        over the 20001 squares. ``stddev_samp`` and ``stddev`` are not
        functions here.

        ``PERCENTILE_CONT`` is exact — 5.5 over the six, exactly 100000000 over
        the squares — but only as a window function: without ``OVER`` it is
        refused (*"The function 'percentile_cont' must have an OVER clause"*),
        and with ``OVER ()`` it answers once per row, which cannot sit in the
        one-row ``SELECT`` a profile is. There is no ``MEDIAN``.
        """
        return {"stddev": func.stdev}

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        """Refuse, before it runs, anything SQL Server does not plan as one read.

        SQL Server has no read-only transaction or session to open the reader
        in, and the generic floor (never commit) does not hold here: the batch
        is the caller's, so ``COMMIT; INSERT …`` ends the transaction this
        server wrapped around it and the ``INSERT`` autocommits outside it —
        measured, the row stayed (#122). So every statement on the reader is
        first compiled under ``SHOWPLAN_XML`` and run only if the server's own
        plan says it is one ``SELECT`` (:func:`_refuse_mssql_write`). Nothing
        here parses SQL; the server classifies its own grammar.

        Installed on the reader only, through the same two hooks as
        :meth:`CrateDBBackend.read_posture`. Every statement this server runs
        on the reader itself, reflection, ``stats`` and the counts, is a single
        ``SELECT``. The cost is one extra compile per statement.
        """

        @event.listens_for(engine, "do_execute")
        def _with_parameters(cursor, statement, parameters, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_mssql_write(refusal, cursor, statement, parameters)
            return False

        @event.listens_for(engine, "do_execute_no_params")
        def _without_parameters(cursor, statement, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_mssql_write(refusal, cursor, statement, None)
            return False


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

    # The dialect registers itself as ``clickhousedb``; the engine is ClickHouse,
    # and a refusal has to name what the caller opened, not the driver package
    # that carries it — the same reason Firebird, YDB and Exasol carry a name.
    name: str = "clickhouse"
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

    def optional_aggregates(self) -> Mapping[str, Any]:
        """The exact median over the column as ``Float64``, and ``stddevSamp`` (#95).

        ClickHouse has several medians and only one is a median, measured on
        25.3 over 1, 2, 4, 7, 10, 12 and over the 20001 squares 0..20000²:

        * ``median`` is ``quantile``, a reservoir sample: 5.5 over six values,
          but 101072862.5 over the squares, where the median is exactly
          100000000 — an estimate, which this server does not report;
        * ``medianExact`` answered **7.0** over the six, the upper of the two
          middle values rather than their average;
        * ``quantileExactInclusive`` (level 0.5 by default) answered 5.5 and
          exactly 100000000, but refused a ``Decimal(12, 4)`` column outright
          (``Illegal type Decimal(12, 4) of argument``).

        So the column is converted first, and with ``toFloat64`` rather than
        ``CAST(… AS DOUBLE)``: the cast refused a nullable column holding a null
        (``Cannot convert NULL value to non-Nullable type``), where
        ``toFloat64`` keeps the null and the aggregate skips it. That way the
        median answered 5.5 and 6.0 over the two float columns, 3.5 over the
        integer one, 6.0 over the decimal one and exactly 100000000 over the
        squares. ``stddevSamp`` answered
        sqrt(98 / 5) over all of them with no conversion; ``stddev`` is not a
        function here.
        """
        return {
            "median": lambda column: func.quantileExactInclusive(
                func.toFloat64(column)
            ),
            "stddev": func.stddevSamp,
        }

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
        created that way could not afterwards be listed by ``directory`` nor found by
        ``drop`` — measured, ``system.data_skipping_indices`` shows it while
        reflection returns nothing.

        Creating one anyway, under a name handed back to the caller, would
        produce exactly the shape this server refuses elsewhere: an answer that
        reads as done and cannot be acted on. So this says what is true, and
        names the thing that actually orders a ClickHouse table.
        """
        article = article_for(self.name).capitalize()
        raise UnsupportedOperation(
            f"{article} {self.name} table is indexed by the ordering key it "
            f"was created with, not by adding an index afterwards. Its "
            f"secondary indexes are data-skipping indexes, which cannot be "
            f"listed or dropped through this interface and do not answer a "
            f"lookup the way an index does. To make {', '.join(columns)} fast "
            f"to filter on, create the table ordered by those columns in "
            f"{self.name} itself, and attach it here."
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

    def optional_aggregates(self) -> Mapping[str, Any]:
        """The sample standard deviation, and no median (#95).

        Measured on Trino 476 against the memory catalog: ``stddev_samp``
        answered sqrt(98 / 5) over 1, 2, 4, 7, 10, 12, on the float, integer
        and ``DECIMAL(12, 4)`` columns alike.

        Trino's only median is ``approx_percentile(col, 0.5)``, an estimate by
        its own name, and it answered 7.0 over those six values where the
        median is 5.5. ``median`` is not a function here and ``percentile_cont
        … WITHIN GROUP`` is a syntax error.
        """
        return {"stddev": func.stddev_samp}

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
        article = article_for(self.name).capitalize()
        raise UnsupportedOperation(
            f"{article} {self.name} datasource holds no data of its own, so it has no "
            f"indexes to create — a filter is made fast by the catalog it reads "
            f"through, not here. To make {', '.join(columns)} fast to filter on, "
            f"index or partition {table.name} in the system behind the catalog, "
            f"and query it through {self.name} as before."
        )


# ---------------------------------------------------------------------------
# MonetDB
# ---------------------------------------------------------------------------


#: The relational operators MonetDB's ``EXPLAIN`` prints for a read, and the
#: only ones its reader runs. Measured on 11.55.7 (Dec2025-SP3) over reads of
#: every shape ``query`` is for: projections, filters, ``LIMIT``/``OFFSET``,
#: grouping, every join, the set operations, ``EXISTS``/``IN`` subqueries,
#: window functions, ``SAMPLE``, recursive CTEs, catalog views and table
#: functions. ``REF`` names a shared subplan; the operator it points at is
#: listed in the same plan and is checked there. A write prints ``insert``,
#: ``update``, ``delete``, ``truncate …`` or ``ddl`` (``CREATE``, ``COPY INTO``,
#: ``CALL``, ``SET``), none of them here, and an operator nobody measured is
#: refused rather than guessed at.
_MONETDB_READ_OPERATORS = frozenset(
    {
        "REF",
        "antijoin",
        "crossproduct",
        "distinct except",
        "distinct intersect",
        "distinct munion",
        "full outer join",
        "group by",
        "join",
        "left outer join",
        "munion",
        "project",
        "recursive munion",
        "right outer join",
        "sample",
        "select",
        "semijoin",
        "table",
        "top N",
    }
)

#: One line of an ``EXPLAIN`` plan: the tree drawing (``|`` and ``&``), then
#: the operator's name, then its arguments, which open with ``(`` or ``[``, or
#: a number for ``REF``. A line that opens with ``)`` or ``[`` after the
#: drawing is the tail of an operator already read, not an operator.
_MONETDB_PLAN_LINE = re.compile(r"[ |&]*(?:([A-Za-z][A-Za-z ]*?) *(?:[(\[]|\d|$))?")


def _monetdb_single_statement(sql: str) -> bool:
    """Whether ``sql`` is one statement, decided without reading its SQL.

    MonetDB needs a ``;`` between two statements: ``SELECT 1 SELECT 2`` and a
    statement on the next line are both ``expecting SCOLON``, measured. So a
    string with no ``;`` but an optional trailing one is one statement. Any
    other ``;`` is refused, including one inside a literal or a comment that
    a lexer could have told apart: that costs the caller a read, never lets a
    second statement through, and keeps this server from parsing MonetDB's
    grammar, which it would one day parse differently from MonetDB.

    Why it matters: ``EXPLAIN`` covers the first statement only. Measured,
    ``EXPLAIN COMMIT; INSERT …`` printed the plan of the ``COMMIT`` and then
    ran the ``INSERT``.
    """
    return _one_statement(sql)


#: What the MonetDB reader says when :func:`_monetdb_single_statement` refuses,
#: in the words :func:`_semicolon_refusal` gives every such gate.
#:
#: The rephrase was measured on ``monetdb/monetdb:latest`` (11.55.7): MonetDB
#: has no ``chr``, and ``char(59)`` is a syntax error there because ``CHAR`` is
#: a type keyword, but ``code(59)`` answers ``;``. It is a system function
#: recorded free of side effects, so the plan check lets it through, and
#: ``tests/test_monetdb_read_gate.py`` runs the example through ``query``.
_MONETDB_SEMICOLON = _semicolon_refusal(
    "after which MonetDB would run the rest unchecked",
    "code(59), as in SELECT 'a' || code(59) || 'b'",
)


def _monetdb_plan_operators(plan: Iterable[str]) -> list[tuple[str | None, bool]]:
    """Each operator in an ``EXPLAIN`` plan, and whether it is a read's.

    An operator line that does not fit the shape :data:`_MONETDB_PLAN_LINE`
    describes comes back as ``(None, False)``: unreadable is refused, the same
    fail-closed rule as an operator nobody measured.
    """
    operators: list[tuple[str | None, bool]] = []
    for line in plan:
        start = line.lstrip(" |&")
        if not start or not start[0].isalpha():
            continue
        name = _MONETDB_PLAN_LINE.match(line).group(1)
        operators.append((name, name in _MONETDB_READ_OPERATORS))
    return operators


#: A quoted ``"a"."b"`` pair as an ``EXPLAIN`` plan prints it, with ``"`` and
#: ``\\`` inside a name escaped by a backslash (measured: ``"we\\"ird"``).
#: Functions, aggregates, tables, columns and aliases all print this way;
#: which pairs are functions is the catalog's to say.
_MONETDB_PLAN_NAME = re.compile(r'"((?:[^"\\]|\\.)*)"\."((?:[^"\\]|\\.)*)"')

#: The functions among ``names`` that a read may not call: any not both a
#: system function and recorded free of side effects. ``side_effect`` alone
#: is not enough, measured: a SQL function whose body is an ``INSERT`` is
#: recorded ``side_effect = false``. A user-defined function is refused
#: whatever it does, which costs a read that calls one and is the only way
#: to be sure. The system ones refused include ``hot_snapshot`` (writes a
#: file), ``next_value_for`` (advances a sequence) and ``vacuum``.
_MONETDB_UNSAFE_FUNCTIONS = (
    "SELECT s.name, f.name FROM sys.functions f "
    "JOIN sys.schemas s ON s.id = f.schema_id "
    "WHERE (NOT f.system OR f.side_effect) AND ({pairs})"
)


def _monetdb_plan_names(plan: Iterable[str]) -> set[tuple[str, str]]:
    """Every quoted ``"a"."b"`` pair in an ``EXPLAIN`` plan, unescaped."""

    def unescape(name: str) -> str:
        return re.sub(r"\\(.)", r"\1", name)

    return {
        (unescape(first), unescape(second))
        for line in plan
        for first, second in _MONETDB_PLAN_NAME.findall(line)
    }


def _monetdb_unsafe_function(cursor: Any, names: set[tuple[str, str]]) -> str | None:
    """The first function in ``names`` a read may not call, as ``schema.name``.

    Asked of the catalog with the pairs bound as parameters, so a name from
    the plan is never spliced into SQL.
    """
    if not names:
        return None
    ordered = sorted(names)
    pairs = " OR ".join("(s.name = %s AND f.name = %s)" for _ in ordered)
    cursor.execute(
        _MONETDB_UNSAFE_FUNCTIONS.format(pairs=pairs),
        [part for pair in ordered for part in pair],
    )
    found = sorted(cursor.fetchall())
    return f"{found[0][0]}.{found[0][1]}" if found else None


def _refuse_monetdb_write(
    refusal: Refusal, cursor: Any, statement: str, parameters: Any
) -> None:
    """Refuse ``statement`` unless it is one statement MonetDB plans as a read.

    The plan comes from ``EXPLAIN`` on the same cursor, which prints the
    relational plan and runs nothing: measured, ``EXPLAIN`` of an ``INSERT``, a
    ``DELETE``, a ``TRUNCATE``, a ``DROP TABLE`` and a ``CREATE TABLE`` left the
    data and the catalog as they were. An ``EXPLAIN`` that fails raises
    MonetDB's own error for the statement, the one running it would have given.

    **A statement with bound parameters is not planned, and is this server's
    own**, for the reason :func:`_refuse_mssql_write` gives: ``query`` sends
    the caller's SQL with no values (``tests/test_query_sends_no_parameters.py``).

    **A function call is checked against the catalog.** A stored function
    whose body writes, called as ``SELECT f()``, plans as a plain projection,
    so every function the plan names must be a system function recorded free
    of side effects (:data:`_MONETDB_UNSAFE_FUNCTIONS`).

    Raises pymonetdb's own ``ProgrammingError``, a DBAPI ``Error``, so
    SQLAlchemy wraps it and ``query`` explains it as a refused write.
    """
    if parameters:
        return
    shape = False
    if not _monetdb_single_statement(statement):
        refused, shape = _MONETDB_SEMICOLON, True
    else:
        cursor.execute(f"EXPLAIN {statement}")
        plan = [line for (line,) in cursor.fetchall()]
        writes = [name for name, read in _monetdb_plan_operators(plan) if not read]
        if writes:
            refused = (
                f"run a statement MonetDB plans as {writes[0]}"
                if writes[0]
                else "run a statement MonetDB does not plan as a read"
            )
        else:
            function = _monetdb_unsafe_function(cursor, _monetdb_plan_names(plan))
            if function is None:
                return
            refused = (
                f"call {function}, which is not a system function free of side effects"
            )
    from pymonetdb.exceptions import ProgrammingError

    refusal.what = refused
    refusal.shape = shape
    raise ProgrammingError(f"query may not {refusal.what}.")


@dataclass(frozen=True)
class MonetDBBackend(Backend):
    """MonetDB, which has no read-only transaction and runs several statements.

    Every generic answer holds here but one: the read posture. The generic
    floor never commits, and MonetDB keeps it, so nothing written through the
    reader survives. But ``query`` also promises that an ``ok`` reports what
    happened, and before #119 ``INSERT … RETURNING`` answered ``ok`` with a
    row the rollback then removed. MonetDB offers nothing to refuse it with:
    ``START TRANSACTION READ ONLY`` is ``Readonly transactions not
    supported``, and the cursor describes ``INSERT … RETURNING name`` and
    ``SELECT 'x' AS name`` identically. CrateDB's first-keyword gate is not
    exact either, because MonetDB nests a write under ``WITH`` and runs a
    string of statements.

    So :meth:`read_posture` asks MonetDB itself, through ``EXPLAIN``, as SQL
    Server's reader asks ``SHOWPLAN_XML`` (:func:`_refuse_monetdb_write`).
    """

    name: str = "monetdb"

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        """Refuse, before it runs, anything MonetDB does not plan as one read.

        The same two hooks as :meth:`MSSQLBackend.read_posture`, one for a
        statement with parameters and one without, installed on the reader
        only, so ``create``, ``update`` and ``drop`` keep writing through the
        writer. Every statement this server runs on the reader, reflection,
        ``stats`` and the count reads, is a ``SELECT`` and plans as one.
        """

        @event.listens_for(engine, "do_execute")
        def _with_parameters(cursor, statement, parameters, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_monetdb_write(refusal, cursor, statement, parameters)
            return False

        @event.listens_for(engine, "do_execute_no_params")
        def _without_parameters(cursor, statement, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_monetdb_write(refusal, cursor, statement, None)
            return False

    def optional_aggregates(self) -> Mapping[str, Any]:
        """``median_avg`` over the column read as a double, and ``stddev_samp`` (#95).

        MonetDB has two medians and neither is right as it stands, measured on
        2026-11-03 over 1, 2, 4, 7, 10, 12:

        * ``median`` answered **4.0** — the lower of the two middle values, not
          their average, so over an even count it is a different statistic;
        * ``median_avg`` averages them and answered 5.5 over a float column,
          4 over an integer one and exactly 100000000 over the 20001 squares
          (so it is exact, not estimated), but over a ``DECIMAL(12, 4)`` column
          holding 1.5 … 12.5 it answered **60000.0** where the median is 6.0 —
          the decimal's scale is lost, by a factor of 10**4.

        Read as a double, the decimal column answers 6.0, which is the cast
        PostgreSQL makes for its own median
        (:func:`_percentile_cont_median_of_doubles`). ``percentile_cont`` does
        not exist here (``no such operator``). ``stddev_samp`` answered
        sqrt(98 / 5) over the float, integer and decimal columns alike, and
        ``stddev`` does not exist.
        """
        return {
            "median": lambda column: func.median_avg(cast(column, Double)),
            "stddev": func.stddev_samp,
        }


# ---------------------------------------------------------------------------
# CrateDB
# ---------------------------------------------------------------------------


#: The statement kinds CrateDB runs without writing, as their first keyword.
#: ``EXPLAIN`` included because it returns a plan and runs nothing, measured on
#: 6.5.0 with ``EXPLAIN INSERT …``. ``EXPLAIN ANALYZE`` does run its statement,
#: and CrateDB refuses it for a write (``UnsupportedFeatureException``), so the
#: one form that executes can only execute a read.
_CRATEDB_READS = frozenset({"SELECT", "WITH", "VALUES", "SHOW", "EXPLAIN"})

#: What a statement with no keyword to read was asking to do, worded to follow
#: "This statement asks to" in ``query``'s refusal.
_CRATEDB_NO_KEYWORD = (
    "run a statement that does not begin with SELECT, WITH, VALUES, SHOW or EXPLAIN"
)

#: What CrateDB skips between tokens, ASCII only. Measured: a leading no-break
#: space or byte-order mark is a ``ParsingException``, so treating either as
#: text, and refusing, loses no read.
_CRATEDB_WHITESPACE = frozenset(" \t\n\r\x0b\x0c")


def _cratedb_first_keyword(sql: str) -> str | None:
    """The first keyword of ``sql`` as CrateDB lexes it, upper-cased, or ``None``.

    **Legitimate here and nowhere else.** This server does not parse SQL to
    decide what a statement does: ``query`` relies on the connection's posture,
    because a parser of its own would mis-parse somebody's grammar. CrateDB has
    no posture to rely on, and its grammar makes the first keyword decisive,
    measured on ``crate:latest`` (6.5.0) over HTTP:

    * one statement per request: ``SELECT 1; INSERT …`` is a
      ``ParsingException`` ("expecting <EOF>");
    * no write nests in a read: ``WITH q AS (SELECT 1) INSERT …``,
      ``WITH q AS (INSERT …) SELECT …`` and ``SELECT * FROM (INSERT … RETURNING
      …) x`` are all ``ParsingException``, and so is ``(SELECT 1)``.

    So this is CrateDB's posture, stated lexically because its grammar makes the
    lexeme the statement's kind. On a backend whose grammar nests a write in a
    read, MonetDB for one, the same reading would not be exact (#119).

    What is skipped before the keyword, exactly as CrateDB skips it, measured:

    * whitespace, ASCII only (:data:`_CRATEDB_WHITESPACE`);
    * a ``--`` comment, up to and including the first ``\\r`` or ``\\n``.
      ``--x\\rSELECT 1`` runs ``SELECT 1``; a vertical tab or a form feed does
      not end the comment. With neither, it runs to the end of the string;
    * a ``/* */`` comment, up to the first ``*/``, because CrateDB does not nest
      them: ``/* a /* b */ SELECT 3`` runs ``SELECT 3``. An unterminated one
      leaves nothing. ``/*+ hint */`` is a comment like any other.

    The keyword is the whole identifier that follows, so ``SELECTED`` is not
    ``SELECT``. Anything else in first place, a parenthesis or a quoted name or
    nothing at all, gives ``None``.

    **Fail-closed is the invariant.** Being stricter than CrateDB can cost a
    read, never a write: ``None`` and an unknown keyword are both refused.
    Disagreeing about where a comment ends could read ``SELECT`` where CrateDB
    reads ``INSERT``, which is why those rules are the measured ones and are
    pinned by test.
    """
    position, end = 0, len(sql)
    while position < end:
        if sql[position] in _CRATEDB_WHITESPACE:
            position += 1
        elif sql.startswith("--", position):
            breaks = [
                found
                for found in (sql.find("\n", position), sql.find("\r", position))
                if found >= 0
            ]
            position = min(breaks) + 1 if breaks else end
        elif sql.startswith("/*", position):
            closed = sql.find("*/", position + 2)
            position = closed + 2 if closed >= 0 else end
        else:
            break
    keyword = re.match(r"[A-Za-z_][A-Za-z0-9_]*", sql[position:])
    return keyword.group(0).upper() if keyword else None


def _refuse_cratedb_write(refusal: Refusal, sql: str) -> None:
    """Refuse ``sql`` unless its first keyword is a query's (:data:`_CRATEDB_READS`).

    Records what was attempted on :class:`Refusal`, worded to follow "This
    statement asks to" in ``query``'s refusal, and raises the ``crate``
    driver's own ``ProgrammingError``. That class is a subclass of the DBAPI
    ``Error`` the dialect declares, so SQLAlchemy wraps it and ``query``
    explains it as a refused write rather than passing on a driver error.
    """
    keyword = _cratedb_first_keyword(sql)
    if keyword in _CRATEDB_READS:
        return
    from crate.client.exceptions import ProgrammingError

    refusal.what = (
        f"run {article_for(keyword)} {keyword} statement"
        if keyword
        else _CRATEDB_NO_KEYWORD
    )
    raise ProgrammingError(f"query may not {refusal.what}.")


@dataclass(frozen=True)
class CrateDBBackend(Backend):
    """CrateDB, where a write is durable before it is visible and cannot be undone.

    A distributed SQL layer over Lucene, and the first backend here from the
    search-engine lineage rather than the database one. Two of its properties
    follow from that ancestry and neither has a precedent in this seam.

    **There are no transactions and no read-only session, so the posture is a
    statement gate.** ClickHouse has no transactions either and answers
    ``readonly=1``, which refuses a statement before it reaches the data.
    CrateDB offers no session-level equivalent: its read-only setting is a
    cluster-wide block that would stop the *write* engine too, and ``SET SESSION
    CHARACTERISTICS AS TRANSACTION READ ONLY`` and ``SET
    default_transaction_read_only = true`` are accepted and ignored (measured on
    6.5.0, an ``INSERT … RETURNING`` after either kept its row). Left at the
    generic floor, a write sent through a read connection was simply applied,
    ``INSERT``, ``CREATE TABLE`` and ``INSERT … RETURNING`` alike, the last
    answered ``ok`` with its row (#118).

    What CrateDB does have is a grammar in which a statement's first keyword
    *is* its kind: one statement per request, and no write can be nested inside
    a read. So :meth:`read_posture` reads that keyword before a statement is
    sent and refuses anything that is not a query (:func:`_refuse_cratedb_write`).
    Nothing written through the reader survives any more, which is why this
    class no longer overrides :meth:`Backend.ddl_survives_refusal`: CrateDB
    answers the generic ``False``.

    **A committed write is not immediately readable.** Rows land in a Lucene
    index refreshed on a timer, so a count taken straight after an insert
    answers ``0`` and answers ``1`` about a second later. :meth:`settle` asks for
    the refresh explicitly rather than waiting for it.

    Reached over the HTTP endpoint on 4200 through Crate.io's own dialect. The
    same server also speaks the PostgreSQL wire on 5432; addressing it that way
    would load PostgreSQL's dialect and answer PostgreSQL's questions, which is
    the mistake ``docs/CONSTRAINTS.md`` §12 records for CockroachDB.
    """

    # The dialect key is ``crate``; the database is CrateDB, which is what a
    # refusal says. See ClickHouse above for the same correction.
    name: str = "cratedb"

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

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        """Refuse, before it is sent, any statement that is not a query.

        The same two hooks as DuckDB's :meth:`DuckDBBackend.read_posture`,
        because SQLAlchemy calls one for a statement with parameters and the
        other for one without. Each runs :func:`_refuse_cratedb_write` and
        returns ``False``, which lets SQLAlchemy send a statement that passed.
        A refused one raises the driver's own DBAPI error, so it is wrapped and
        explained like any refusal the server sends.

        Installed on the reader only, so ``create``, ``update`` and ``drop``
        keep writing through the writer. Every statement this server itself
        runs on the reader, reflection, ``stats`` and the count reads, is a
        ``SELECT``. :meth:`settle`'s ``REFRESH TABLE`` runs on the writer.
        """

        @event.listens_for(engine, "do_execute")
        def _with_parameters(cursor, statement, _parameters, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_cratedb_write(refusal, statement)
            return False

        @event.listens_for(engine, "do_execute_no_params")
        def _without_parameters(cursor, statement, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_cratedb_write(refusal, statement)
            return False

    def settle(self, conn: Connection, table: str) -> None:
        """``REFRESH TABLE``, so a row just written can be read back.

        The default refresh interval is a second, which is *fast* and entirely
        beside the point: the question is not how long the wrong answer lasts
        but whether this server ever gives one. ``insert_source`` counts the rows
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
        """Two, both of them the database's own absences.

        Unlike ClickHouse's two of the same name, which are its *driver's*, these
        are gaps in CrateDB itself: it has no binary column and no storable
        time-of-day column. ``LargeBinary`` renders ``BLOB`` and ``CREATE TABLE``
        is refused at parse time — ``Cannot find data type: blob`` — so the column
        is never made. ``Time`` is refused the same way when it reaches the
        server as ``TIME``; from ``sqlalchemy-cratedb`` 0.44 the dialect renders
        it ``STRING`` instead, the server records the column as ``TEXT``, and
        ``time(14, 30)`` is stored as ``'14:30:00'`` — which is what raw SQL
        through ``query`` reads back, measured. That is exactly the substitution
        this axis exists to decline, a value in a column whose type says
        something else, so it stays declined.

        ``Numeric`` used to be the third entry, and it was the dangerous one
        because nothing failed (issue #52). Up to 0.43 the dialect rendered it
        ``BIGINT``, so ``Decimal("12345.6789")`` was stored as ``12345`` and read
        back as ``Decimal("12345.0000")``. From 0.44 it renders
        ``NUMERIC(p, s)`` and ``Decimal("1.25")`` round-trips exactly — measured
        against the container, ``SHOW CREATE TABLE`` reporting
        ``"c" NUMERIC(10, 2)``. The ``cratedb`` extra's floor is 0.44 for that
        reason, so the entry is gone rather than kept as a precaution. A bare
        ``Numeric()`` with no precision is now refused at compile time with a
        ``CompileError`` naming the problem, which is a loud failure and needs
        nothing here.
        """
        return frozenset({"LargeBinary", "Time"})

    def optional_aggregates(self) -> Mapping[str, Any]:
        """The sample standard deviation, and no median (#95).

        ``stddev_samp`` answered sqrt(98 / 5) over 1, 2, 4, 7, 10, 12 on the
        float, integer and numeric columns alike (``stddev`` is the same
        statistic here, measured, but ``stddev_samp`` says which one it is).

        CrateDB's one median is ``percentile(col, 0.5)``, and it fails all
        three tests, measured on 2026-11-03: it answered 7.0 over the six
        values where the median is 5.5, 99494117.3 over the 20001 squares where
        it is exactly 100000000 — an estimate, not a median — and over a
        ``NUMERIC`` column it refused the statement (``Cannot cast value
        `7.5000` to type `byte```). There is no ``median`` and no
        ``percentile_cont``.
        """
        return {"stddev": func.stddev_samp}

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
        article = article_for(self.name).capitalize()
        raise UnsupportedOperation(
            f"{article} {self.name} datasource indexes every column as it is "
            f"written, so there is no index to add — {', '.join(columns)} on "
            f"{table.name} is already fast to filter on. Creating one here "
            f"would report work that was never done."
        )


#: The widest ``VARCHAR`` this backend asks Firebird for, in **characters**.
#:
#: Firebird's limit is 32,765 *bytes*. A loaded text column is declared
#: ``CHARACTER SET UTF8`` (see :meth:`FirebirdBackend.column_type`), which
#: reserves four bytes a character, so the widest such column is 8,191
#: characters — measured on Firebird 5: ``VARCHAR(8191) CHARACTER SET UTF8`` is
#: created, ``VARCHAR(8192)`` is refused with ``Implementation limit exceeded``.
#: Because the column names its character set, this holds on every database
#: whatever its default, rather than only on one that happens to match.
#:
#: Deliberately not the 32,765 the test container accepts for a column with no
#: character set: that column inherits the database default, which there is
#: ``NONE``, and its width counts bytes.
_FIREBIRD_VARCHAR_MAX = 8191

#: What a locking clause can sit beside without being one on Firebird: a
#: string with ``''`` for a quote, a quoted name, and the two comment forms.
_FIREBIRD_OPAQUE = re.compile(
    r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*[\s\S]*?\*/"
)

#: Firebird's ``WITH LOCK``, optionally followed by ``SKIP LOCKED``. Not a
#: common table expression called ``lock``, which ``WITH LOCK AS (…)`` or
#: ``WITH LOCK (…)`` would be.
_FIREBIRD_WITH_LOCK = r"\bWITH\s+LOCK\b(?!\s*(?:AS\b|\())(?:\s+SKIP\s+LOCKED\b)?"

#: Firebird's locking clause, with the whitespace before it: ``FOR UPDATE`` and
#: an optional ``OF`` list, optionally followed by ``WITH LOCK``, or
#: ``WITH LOCK`` alone (:data:`_FIREBIRD_WITH_LOCK`). Matched on a masked
#: statement (:func:`_without_row_locks`).
_FIREBIRD_LOCKING_CLAUSE = re.compile(
    r"\s*(?:\bFOR\s+UPDATE\b(?:\s+OF\s+[\w$.]+(?:\s*,\s*[\w$.]+)*)?"
    rf"(?:\s+{_FIREBIRD_WITH_LOCK})?|{_FIREBIRD_WITH_LOCK})",
    re.IGNORECASE,
)


def _firebird_without_row_locks(sql: str) -> tuple[str, str] | None:
    """:func:`_without_row_locks` in Firebird's grammar (#144)."""
    return _without_row_locks(sql, _FIREBIRD_OPAQUE, _FIREBIRD_LOCKING_CLAUSE)


#: A statement Firebird runs as a select: its first word, after any comment,
#: is ``SELECT`` or ``WITH``. Matched on a masked statement (:func:`_masked`).
_FIREBIRD_SELECT = re.compile(r"\s*(?:SELECT|WITH)\b", re.IGNORECASE)


def _firebird_for_update(sql: str) -> tuple[str, str] | None:
    """``sql``'s locking clause and the read without it, where it holds ``FOR UPDATE``.

    Only a statement that is itself a select: Firebird reports one carrying
    ``FOR UPDATE`` as a select for update, which the driver fetches nothing
    for, and an ``EXECUTE BLOCK`` whose body selects ``FOR UPDATE`` as a plain
    select, which it fetches. Measured, the block answered its five rows
    (docs/CONSTRAINTS.md §34). ``None`` as well for a statement with no locking
    clause, and for ``WITH LOCK`` alone, which the read-only transaction
    refuses itself (#144). See :func:`_refuse_firebird_for_update` for why
    ``FOR UPDATE`` is refused here.
    """
    if not _FIREBIRD_SELECT.match(_masked(sql, _FIREBIRD_OPAQUE)):
        return None
    found = _firebird_without_row_locks(sql)
    if found is None or not re.match(r"FOR\s", found[0], re.IGNORECASE):
        return None
    return found


#: Why a select carrying ``FOR UPDATE`` is refused on Firebird, worded to
#: follow "its <clause> clause" (:class:`Refusal`, ``lock``).
_FIREBIRD_UNFETCHED = (
    "makes it a select that the Firebird driver this server uses fetches no "
    "rows for, read-only or not, so it would answer with no rows whatever the "
    "table holds, and that empty answer would be false"
)


def _refuse_firebird_for_update(refusal: Refusal, statement: str, context: Any) -> None:
    """Refuse a select carrying ``FOR UPDATE`` before it is sent (#145).

    ``firebirdsql`` 1.5.0 fetches rows only for a statement Firebird reports as
    a plain select. Firebird 5 reports one carrying ``FOR UPDATE`` as a select
    for update, so the driver never asks for its rows and the cursor holds
    none: ``query`` answered such a statement ``ok`` with no rows whatever the
    table held, on a read-only connection and a read-write one alike. Every
    variant measured did, with or without ``OF`` or ``WITH LOCK`` and behind a
    ``WHERE`` (docs/CONSTRAINTS.md §34). An empty answer that is false is the
    worst one to give an agent, so the reader refuses the clause, and the
    explainer gives the same select without it, which reads the rows.

    Raises the driver's own ``ProgrammingError``, a DBAPI ``Error``, so
    SQLAlchemy wraps it and ``query`` explains it.
    """
    found = _firebird_for_update(statement)
    if found is None:
        return
    refusal.what, refusal.lock = _FIREBIRD_UNFETCHED, True
    raise context.dialect.loaded_dbapi.ProgrammingError(
        f"query may not run a select carrying {found[0]}."
    )


#: ``isc_read_only_trans``: "attempted update during read-only transaction".
#: Firebird's own code for a write its read-only transaction refused, carried by
#: ``firebirdsql`` on the exception's ``gds_codes``. Measured on Firebird 5 with
#: ``firebirdsql`` 1.5.0: an ``INSERT … RETURNING`` is refused with exactly this
#: code, a ``CREATE TABLE`` with this code among two others.
_FIREBIRD_READ_ONLY = 335544361


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

    Four facts had to be measured, and each is an override below.

    **The read engine opens a read-only transaction** (#118). The transactional
    floor alone declines to keep a write without refusing it, and that is not
    enough for ``INSERT … RETURNING``: it hands rows back like a read, so
    ``query`` answered ``ok`` with a row the closing rollback then discarded.
    ``readonly=true`` in the read URL is how ``sqlalchemy-firebirdsql`` is told
    to pass ``readonly=True`` to ``firebirdsql.connect``, which opens every
    transaction ``ISOLATION_LEVEL_READ_COMMITED_RO``. Measured on Firebird 5
    with ``firebirdsql`` 1.5.0: ``INSERT … RETURNING``, an ``EXECUTE BLOCK``
    that inserts and ``CREATE TABLE`` are all refused before they change
    anything, and ``SELECT`` reads as before. See :attr:`read_only_query` and
    :meth:`denies_write`.

    **DDL cannot be followed by DML in the same transaction** (#53). Firebird's
    DDL is properly transactional — a ``CREATE TABLE`` that never commits leaves
    nothing behind — but statements are prepared against committed metadata, so
    the new table is invisible to the transaction that created it. See
    :meth:`sees_new_tables_in_transaction`.

    **There is no way to rename a table.** Not a missing convenience — no
    statement exists, in any Firebird version. See :meth:`renames_tables`.

    **Neither portable type a loaded file needs is usable as rendered** (#55, #56).
    Core's ``Double`` becomes bare ``DOUBLE`` before ``sqlalchemy-firebirdsql``
    0.3.0, which Firebird's parser rejects outright; Core's ``Text`` becomes
    ``BLOB`` in every version, which is accepted and then groups by identity
    rather than by value. See :meth:`column_type` for both.

    Everything else is generic and was measured to be: nothing a refused
    statement did survives it, for rows or for schema, a quoted mixed-case name
    survives, ``Numeric`` round-trips exactly where CrateDB truncated it,
    ``Time``, ``Date``, ``TIMESTAMP`` and ``LargeBinary`` all store and read
    back, indexes build and reflect and drop, and the driver's errors arrive
    properly wrapped as ``SQLAlchemyError`` so :meth:`driver_errors` stays empty.
    """

    name: str = "firebird"
    read_only_query: ClassVar[Mapping[str, str]] = {"readonly": "true"}

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        """Refuse a select carrying ``FOR UPDATE`` before it runs (#145).

        The read-only transaction is the posture, and it comes from the URL
        (:attr:`read_only_query`); this adds only the gate
        :func:`_refuse_firebird_for_update` explains, on every statement the
        reader sends.
        """

        @event.listens_for(engine, "do_execute")
        def _with_parameters(cursor, statement, parameters, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_firebird_for_update(refusal, statement, context)
            return False

        @event.listens_for(engine, "do_execute_no_params")
        def _without_parameters(cursor, statement, context):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
            _refuse_firebird_for_update(refusal, statement, context)
            return False

    def denies_write(self, exc: Exception) -> bool:
        """Recognise the read-only transaction's refusal, by its code.

        ``gds_codes`` is the set of Firebird status codes the server sent, so
        membership of :data:`_FIREBIRD_READ_ONLY` is the whole test and the
        message, which carries the statement and the line it failed on, is
        never read. A ``CREATE TABLE`` refusal carries the code beside the
        generic metadata-update failures, which is why this asks for
        membership rather than equality.
        """
        origin = getattr(exc, "orig", exc)
        return _FIREBIRD_READ_ONLY in (getattr(origin, "gds_codes", None) or ())

    def without_row_locks(self, sql: str) -> tuple[str, str] | None:
        """The locking clause the read-only transaction refused (#144).

        Measured through ``query`` on Firebird 5.0.4 with ``firebirdsql``
        1.5.0: ``SELECT * FROM t WITH LOCK`` was refused with
        :data:`_FIREBIRD_READ_ONLY`, and ``SELECT * FROM t`` answered its five
        rows. ``FOR UPDATE``, with or without ``WITH LOCK``, is not refused by
        the transaction: the driver answers it with no rows at all, read-only or
        not, because it fetches only for a plain select statement. It is found
        here as well, for the reader's own gate, which refuses it before it is
        sent (:meth:`read_posture`, #145, docs/CONSTRAINTS.md §34).
        """
        return _firebird_without_row_locks(sql)

    def optional_aggregates(self) -> Mapping[str, Any]:
        """The sample standard deviation, and no median (#95).

        Measured on Firebird 5: ``stddev_samp`` answered sqrt(98 / 5) over 1, 2,
        4, 7, 10, 12, on the floating-point column a loaded file makes, on an
        integer one and on a ``NUMERIC(12, 4)`` one alike. ``median``, ``percentile_cont … WITHIN
        GROUP`` and ``stddev`` were each refused as a ``Dynamic SQL Error``:
        Firebird has no median aggregate to report.
        """
        return {"stddev": func.stddev_samp}

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

        The third row is why :meth:`loader.Workspace.insert_source` splits at the
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
        article = article_for(self.name).capitalize()
        raise UnsupportedOperation(
            f"{article} {self.name} datasource has no statement that renames a "
            f"table, so {table} cannot become {to}. Copy the rows into a table "
            f"of the new name with create, then drop the old one — that is two "
            f"tables and loses the indexes on the first, which is why it is "
            f"not done for you."
        )

    def column_type(self, declared: str, *, longest: int | None = None) -> TypeEngine:
        """Two of the three portable types have to be respelled here.

        ``REAL`` → **``DOUBLE PRECISION``** (#55). Core's ``Double`` is the
        portable 64-bit float and every other backend renders it usably.
        ``sqlalchemy-firebirdsql`` 0.1.0 and ``sqlalchemy-firebird`` 2.2.0 both
        render it ``DOUBLE``, a keyword Firebird does not have — the parser waits
        for ``PRECISION`` and fails on whatever follows, so the ``CREATE TABLE``
        dies with ``-104 Token unknown`` and no table is made.
        ``DOUBLE_PRECISION`` holds a float64 exactly: ``0.1`` reads back as
        ``0.1``. ``Float(53)`` lands in the same column and was rejected in favour
        of this — both work, and only one says what it means.

        ``sqlalchemy-firebirdsql`` 0.3.0 fixed the spelling upstream, measured:
        ``Double`` now renders ``DOUBLE PRECISION`` and ``0.1`` round-trips. The
        respelling stays all the same, because 0.3.0 requires Python 3.11 and
        this project supports 3.10, where the resolver still installs 0.1.0. On
        0.3.0 it asks for the type the dialect would have chosen anyway, so it
        costs nothing there. Delete it when the Python floor reaches 3.11 and
        the ``firebird`` extra's floor can say ``>=0.3``.

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
        holds, measured over the whole source before the table is made, so the
        ``VARCHAR`` is sized to the data rather than guessed at.

        **The column states ``CHARACTER SET UTF8``**, because ``longest`` counts
        characters and a ``VARCHAR`` with no character set does not. It takes
        the database's default, and on a database created ``NONE`` — the
        Firebird image's default, and the test container's — the width counts
        *bytes*. Measured: ``été`` is three characters and five bytes, and a
        ``VARCHAR(3)`` with no character set refused it with ``string right
        truncation, expected length 3, actual 5``, so any file holding one
        accented letter failed to load. Under UTF8 the width is characters on
        every database — three emoji, twelve bytes, fit a ``VARCHAR(3)`` — and
        every value a Python string can hold is storable, where a single-byte
        default would refuse the characters it lacks. Ordering is unchanged:
        UTF8's default collation orders by code point, which is the order the
        same UTF-8 bytes had under ``NONE``.

        Stated through ``with_variant`` and the dialect's own ``FBVARCHAR``,
        which is how SQLAlchemy spells a type only one dialect has; the column
        is still a portable ``String`` of that width to anything that inspects
        it. The import is deferred to here for the reason ClickHouse's is: it is
        only reached with the ``firebird`` extra installed.

        Beyond :data:`_FIREBIRD_VARCHAR_MAX` there is nothing else to use and it
        falls back to ``Text``: the grouping is lost, which is bad, and the
        values are kept whole, which matters more than truncating them to fit.
        :meth:`column_note` is what says so, column by column.

        Integers need no help. Both respellings are *client-library* defects
        rather than database limits — Firebird holds a float and a comparable
        string quite happily once asked in its own words.
        """
        if declared == "REAL":
            return DOUBLE_PRECISION()
        if declared == "TEXT":
            width = max(1, longest or 1)
            if width > _FIREBIRD_VARCHAR_MAX:
                return Text()
            from sqlalchemy_firebirdsql.types import FBVARCHAR

            return String(width).with_variant(
                FBVARCHAR(width, charset="UTF8"), "firebirdsql"
            )
        return super().column_type(declared, longest=longest)

    def column_note(
        self, name: str, declared: str, *, longest: int | None = None
    ) -> str | None:
        """Name what a text column too wide for a ``VARCHAR`` will answer wrongly.

        Such a column stays a ``BLOB`` (:meth:`column_type`). Measured on
        Firebird 5 over twenty-one rows holding three values: ``GROUP BY``
        answered twenty-one groups, ``DISTINCT`` twenty-one rows, ``ORDER BY``
        left them in insertion order, and ``CREATE INDEX`` was refused — while
        ``=`` and ``LIKE`` compared the text correctly. Only the first three are
        silent, which is why the note leads with them; the index refusal at
        least arrives as an error.
        """
        if declared != "TEXT" or (longest or 0) <= _FIREBIRD_VARCHAR_MAX:
            return None
        return (
            f"Column {name} holds values up to {longest} characters, wider than "
            f"the {_FIREBIRD_VARCHAR_MAX} a Firebird VARCHAR can hold, so it is "
            f"stored as a BLOB: every value is kept whole, and = and LIKE compare "
            f"the text correctly, but GROUP BY, DISTINCT and ORDER BY on it "
            f"compare storage handles rather than text and answer wrongly with "
            f"no error — one group per row, unsorted. It cannot be given an "
            f"index either."
        )


# ---------------------------------------------------------------------------
# YDB
# ---------------------------------------------------------------------------


#: The isolation level that gives YDB's read connection a floor at all.
#:
#: Named rather than inlined for the reason :data:`_TRINO_ISOLATION` is: it is a
#: measurement, not a preference, and the measurement is recorded beside it.
#:
#: Unlike Trino, every level this dialect declares survives SQLAlchemy's
#: normalisation — the driver's enum *values* carry the spaces SQLAlchemy puts
#: in, so the two-word names round-trip where Trino's raised ``KeyError``. So
#: this is a free choice among four read-only levels, and a snapshot is the one
#: that means what a read connection should mean: the latest committed state,
#: consistent for as long as the connection holds it. See docs/CONSTRAINTS.md
#: §22.3.
_YDB_ISOLATION = "SNAPSHOT READONLY"

#: YDB's own code for "this operation cannot be performed in a read only
#: transaction", carried on the nested issue rather than on the status. Matched
#: on the code for the reason MySQL's and ClickHouse's are: a sentence carries a
#: locale and a version and a code carries neither.
_YDB_READ_ONLY = 2008

#: The status YDB answers a **schema** statement with when it arrives inside a
#: transaction — ``PRECONDITION_FAILED``, carrying
#: ``Scheme operations cannot be executed inside transaction``.
#:
#: Broader than :data:`_YDB_READ_ONLY` and knowingly so: the nested issue carries
#: ``issue_code: 0``, so the status is the only machine-readable part of this
#: refusal and it is a general "not in this state" code rather than a specific
#: one. What bounds it is where it is consulted — :meth:`Backend.denies_write` is
#: asked only about a statement that failed on the *read* connection, and the
#: only statement this server sends there that a transaction can be the wrong
#: place for is a schema statement. A false positive would replace a driver
#: message with an accurate sentence about which verb to use instead, which is
#: the cost worth taking.
_YDB_SCHEME_IN_TRANSACTION = 400120

#: The name YDB gives a column that a loaded file did not bring. Prefixed with an
#: underscore so it cannot collide with a sanitised CSV header — ``_sanitize``
#: never produces a leading underscore — and named for what it holds rather than
#: for why it exists, because the caller reading ``directory`` sees the column and not
#: this comment.
_YDB_SURROGATE_KEY = "_row"


@dataclass(frozen=True)
class YDBBackend(Backend):
    """YDB, whose one rule about transactions explains almost everything else.

    Registered under ``yql`` because that is what the dialect calls itself —
    after YDB's **query language**, not after the engine and not after the
    driver. The third distinct way a dialect name has failed to be an identity
    here: YugabyteDB borrows another engine's dialect, Firebird's is named after
    the Python package that speaks the wire, and this one is named after the SQL.
    A refusal that said ``yql`` would name a syntax to somebody who opened a
    database. Issue #45's family, third variant.

    **The rule: a schema operation may not be inside a transaction.** Everything
    below follows from it, and it is why the shared :meth:`Backend.rename_table`
    changed rather than being overridden (#59) — the rename was the one schema
    statement this server stated as ``text``, and ``text`` keeps a statement
    inside the transaction the connection is already in.

    **A table must declare a primary key** — see :meth:`requires_primary_key`,
    the one axis this backend adds.

    **Up to ``ydb-dbapi`` 0.1.22 there was no rollback to be had.** Not because
    YDB lacks transactions: it has them, the driver exposes them, and they work
    when driven directly. That client library bound a cursor's transaction
    context *at cursor construction*, and SQLAlchemy constructs the cursor before
    it opens the transaction — so every statement ran in implicit autocommit and
    the transaction opened a moment later was empty. ``rollback()`` then rolled
    that empty transaction back and **returned normally**, reporting success over
    a write that stands. 0.1.24 reads the transaction from the connection at
    execute time instead, and an uncommitted write is now rolled back, measured
    (#61, docs/CONSTRAINTS.md §22.2). Nothing in this project requires 0.1.24,
    though, so :meth:`read_posture` still refuses rather than relying on the
    transactional floor. Refusing is the stronger guarantee in any case.

    **It has no time-of-day type**, which is Oracle's gap arrived at
    independently — see :meth:`unstorable_column_types`.

    Everything else is generic and was measured to be: ``Numeric`` round-trips
    exactly, ``Double`` holds a float64, ``Text`` groups by value, ``Date``,
    ``DateTime``, ``LargeBinary`` and ``Boolean`` all store and read back,
    indexes build and reflect and drop, a quoted mixed-case name survives
    verbatim, and the driver's errors arrive properly wrapped as
    ``SQLAlchemyError`` so :meth:`driver_errors` stays empty.
    """

    name: str = "ydb"

    def read_posture(self, engine: Engine, refusal: Refusal) -> None:
        """Refuse the write outright, rather than trust a rollback underneath.

        The generic guarantee is that a read connection never commits, so
        anything it changed is rolled back. With ``ydb-dbapi`` 0.1.22 that
        guarantee was not weakened but **absent**: an ``INSERT`` sent through a
        read connection was applied, survived an explicit ``rollback()`` that
        raised nothing, and was still there on the next connection (#61). 0.1.24
        fixes the driver, measured. The posture stays because the ``ydb`` extra
        does not require 0.1.24, and because refusing a write is a stronger
        guarantee than undoing it.

        Trino's remedy was tried first and does not work here. There, naming a
        real isolation level takes the driver out of autocommit and the floor
        comes back; here ``SERIALIZABLE`` *is* honoured — the level reads back as
        set — and the write still lands, because the defect is the cursor's
        binding time rather than the level.

        So the posture is raised instead of restored, which is the shape SQLite
        and ClickHouse already have: a **read-only** isolation level makes the
        server refuse the statement before it reaches the data.
        ``Operation 'InsertAbort' can't be performed in read only transaction``,
        and the row count is unchanged. That is a stronger guarantee than the one
        it replaces, not a weaker one — the write never happens rather than
        happening and being undone.

        Set on the engine rather than per statement, for the reason SQLite's
        ``query_only`` and MySQL's read-only session are: a posture toggled
        around a statement has a window in which it is something else.
        """
        engine.update_execution_options(isolation_level=_YDB_ISOLATION)

    def denies_write(self, exc: Exception) -> bool:
        """Recognise both of the refusals a read connection here produces.

        **Rows** come back as ``issue_code: 2008`` on a *nested* issue rather
        than on the exception's status: ``status`` is ``GENERIC_ERROR`` (400080)
        for a missing primary key and an unknown column type as well, so the
        status identifies nothing and the nested code identifies exactly this.

        **Schema** comes back the other way round — ``PRECONDITION_FAILED``
        (400120) on the status, with ``issue_code: 0`` on the issue, because a
        schema statement is not refused for being a *write* but for being inside
        a transaction at all. Two refusals, two codes, one meaning as far as the
        caller is concerned: this statement will not happen here, and there is a
        verb that does it.

        Both are needed and the second was nearly missed, because without it the
        outcome was already right — the ``CREATE`` was refused and nothing was
        created — and only the *explanation* was wrong. A wrong explanation
        attached to a right outcome passes every test that asserts on the
        outcome, which is why the endpoint test asserts on the text.

        Read defensively — ``getattr`` and a guarded loop — because this walks a
        driver's internal error structure rather than a documented API, and a
        backend whose refusal-recogniser raised would turn a clean refusal into a
        crash.
        """
        origin = getattr(exc, "orig", exc)
        if getattr(getattr(origin, "status", None), "value", None) == (
            _YDB_SCHEME_IN_TRANSACTION
        ):
            return True
        for issue in getattr(origin, "issues", ()) or ():
            if getattr(issue, "issue_code", None) == _YDB_READ_ONLY:
                return True
        return False

    def requires_primary_key(self) -> bool:
        """``True``. YDB has no heap tables; a keyless ``CREATE TABLE`` is refused.

        Measured, one variable changed: the same statement with a primary key
        declared is accepted and the rows land, and without one it fails at
        parse time with ``"Primary key is required for ydb tables."``
        (``issue_code: 1020``). Nothing is created either way, so this is a loud
        failure rather than a silent one — which is the only reason it was cheap
        to find.
        """
        return True

    def optional_aggregates(self) -> Mapping[str, Any]:
        """``stddev_samp`` over the column read as a double, and no median (#95).

        Measured on local-ydb 26.1: ``stddev_samp`` answered sqrt(98 / 5) over
        1, 2, 4, 7, 10, 12 in a float column and exactly over the 20001 squares,
        but over a ``Decimal(12, 4)`` column it refused the whole statement
        (``Type annotation``, ``issue_code: 1030``). Read as a ``Double`` the
        same column answered sqrt(98 / 5), so the cast goes in, as it does for
        the median on CockroachDB (:func:`_percentile_cont_median_of_doubles`).

        YQL's ``median`` answered 5.5 over the six values, but 100007405 over
        the squares, where the median is exactly 100000000: an estimate, which
        is not reported. ``percentile(col, 0.5)`` is the same function and gave
        the same answers. ``percentile_cont … WITHIN GROUP`` does not parse.
        """
        return {"stddev": lambda column: func.stddev_samp(cast(column, Double))}

    def unstorable_column_types(self) -> frozenset[str]:
        """YDB has no time-of-day type.

        ``Unknown simple type 'TIME'``, raised while compiling the column, so
        there is no column for a value to go into. The same gap Oracle has,
        reached independently — and the reason this axis is a set of names rather
        than a flag, since the two backends share exactly one entry and nothing
        else.
        """
        return frozenset({"Time"})


# ---------------------------------------------------------------------------
# Exasol
# ---------------------------------------------------------------------------


def _exasol_mapped_connection() -> type:
    """The driver's connection class, with the type mapper its DBAPI omits.

    ``exasol.driver.websocket`` builds every connection through one options
    dictionary and hardcodes ``fetch_mapper`` to ``None`` in it, while
    ``connect()`` takes a ``connection_class`` — so the supported way to change
    an option the DBAPI does not expose is to subclass what it constructs. That
    is what this is, and it is why the seam can state the fact at all.

    Reported as `exasol/pyexasol#361
    <https://github.com/exasol/pyexasol/issues/361>`_, which asks for the option
    to be a connect argument. **Delete this and the ``connect_args`` override
    together on the day it is**: ``_options`` is private, and reaching into it is
    the whole reason this exists rather than a keyword.

    Imported here rather than at module scope because the extra is optional, as
    CrateDB's converter is.
    """
    import pyexasol
    from exasol.driver.websocket.dbapi2 import DefaultConnection

    class _MappedConnection(DefaultConnection):
        def connect(self):  # noqa: ANN201 - the vendor's signature, untyped
            self._options["fetch_mapper"] = pyexasol.exasol_mapper
            return super().connect()

    return _MappedConnection


@dataclass(frozen=True)
class ExasolBackend(Backend):
    """Exasol, whose values arrive as text unless the driver is asked otherwise.

    Registered under ``exa`` — the backend name in its URL — while calling
    itself ``exasol``, which is the **fourth** distinct way a dialect name has
    failed to be an identity in this project and the first where the gap is
    inside one package: ``sqlalchemy-exasol`` registers the entry point ``exa``,
    names the dialect ``exasol``, and reports a *module path* —
    ``exasol.driver.websocket.dbapi2`` — as its driver. So a URL says ``exa``,
    ``Dialect.name`` says ``exasol``, and neither of them is what a refusal
    should print, which is why the name is carried here.
    """

    name: str = "exasol"

    #: Autocommit off, for the **read** engine only, and it is what puts the
    #: transactional floor under this backend at all.
    #:
    #: The DBAPI's ``autocommit`` defaults to ``True`` (measured:
    #: ``_connection.py`` signature, and an ``INSERT`` followed by ``rollback()``
    #: leaves the row behind), so every statement a read connection ran would
    #: persist and :meth:`Backend.read_posture`'s "open, never commit, close"
    #: guarantee would be no guarantee at all. The dialect maps this query
    #: parameter onto the connect argument, and with it the same drill rolls the
    #: row back.
    #:
    #: Set here rather than through :meth:`connect_args` deliberately: the write
    #: engine must keep the driver's own default, and ``read_only_query`` is the
    #: one lever that reaches exactly one of the two engines.
    read_only_query: ClassVar[Mapping[str, str]] = {"AUTOCOMMIT": "n"}

    def connect_args(self) -> Mapping[str, Any]:
        """The type mapper, without which a widened number arrives as text.

        Exasol's WebSocket protocol carries a ``DECIMAL`` as a JSON **string**
        once its precision outgrows what a double holds exactly, and the DBAPI
        this dialect ships passes the payload through untouched. Measured on one
        table: a ``DECIMAL(18,0)`` column reads back as ``42``, and
        ``SUM`` of that same column — which Exasol widens to ``DECIMAL(29,0)`` —
        reads back as the string ``'42'``. A ``TIMESTAMP`` is the same shape of
        answer: ``'2020-01-02 03:04:05.000000'``, text.

        That is the defect ``_ON_THE_WIRE`` exists for, arriving in the one form
        it cannot fix: a ``Decimal`` is spelled as a number on the way out, but a
        ``str`` is indistinguishable from a column that really is text, so the
        agent gets ``"155000"`` and adds it as prose. CrateDB's converter is the
        same fact from the same cause — an untyped wire protocol — and the same
        remedy.

        Not reachable through SQLAlchemy's typing: a Core ``select()`` converts,
        because the dialect's ``colspecs`` know the column is numeric, and
        ``query`` runs the caller's own text where nothing does.
        """
        return {"connection_class": _exasol_mapped_connection()}

    def rename_table(self, conn: Connection, table: str, to: str) -> None:
        """``RENAME TABLE``, which is the only spelling this database has.

        ``ALTER TABLE … RENAME TO`` is refused — measured, and refused with an
        empty message, which is this driver's habit rather than the database's:
        ``_cursor.py`` raises ``Error()`` with the server's text dropped, for
        every failure alike (`exasol/pyexasol#360
        <https://github.com/exasol/pyexasol/issues/360>`_). The preparer quotes
        both identifiers, as the generic implementation does.

        Through ``DDL`` rather than ``text`` for the generic implementation's
        reason (#59): only one of the two carries the signal that this is schema,
        and an override that reaches for ``text`` drops it for the sake of
        changing the verb. This dialect happens not to care — measured both ways
        — which is exactly the agreement that let the generic one stand as
        ``text`` for eleven dialects before YDB disagreed.
        """
        prepare = conn.dialect.identifier_preparer.quote
        conn.execute(DDL(f"RENAME TABLE {prepare(table)} TO {prepare(to)}"))

    def builds_indexes(self) -> bool:
        return False

    def build_index(
        self, name: str, table: Table, columns: Sequence[str]
    ) -> tuple[Index, tuple[str, ...]]:
        """Refused, because this database indexes itself.

        Exasol creates and drops indexes on its own, from the joins and filters
        it actually sees, and offers no statement for one: the dialect refuses
        ``CREATE INDEX`` at compile time with *"Exasol manages indexes
        internally"*, and the database refuses the raw statement too. There is
        nothing to create under a name that could then be listed or dropped, so
        this says what is true rather than reporting an index nobody made.
        """
        raise UnsupportedOperation(
            f"{self.name} maintains its own indexes: it creates and drops them "
            f"from the queries it actually runs, and has no statement for making "
            f"one by hand. There is nothing to create for {', '.join(columns)} "
            f"and nothing this interface could afterwards list or drop, so this "
            f"verb refuses rather than report an index that does not exist."
        )

    def unstorable_column_types(self) -> frozenset[str]:
        """``LargeBinary``, and this one is refused before anything is sent.

        Exasol has no binary column type at all, and the dialect says so at
        **compile** time — ``BLOB is not supported by the Exasol dialect`` — so
        the table is not made and no value is bound. That is the honest end of
        the range this axis covers: ClickHouse's and Trino's binary columns can
        be made and not written to, and here the type does not exist and nothing
        pretends otherwise.

        Nothing else is missing. ``Time``, ``Numeric``, ``Boolean``, ``Date``,
        ``DateTime``, ``Float``, ``Integer`` and ``String`` were each created and
        dropped on a live server, one statement at a time.
        """
        return frozenset({"LargeBinary"})

    def optional_aggregates(self) -> Mapping[str, Any]:
        """``MEDIAN`` and ``STDDEV_SAMP``, both measured (#95).

        On Exasol 2026.1, ``MEDIAN`` answered 5.5 over 1, 2, 4, 7, 10, 12 —
        the two middle values averaged — 3.5 and 6.0 over an integer and a
        ``DECIMAL(12, 4)`` column, and exactly 100000000 over the 20001
        squares, so it is exact. ``STDDEV_SAMP`` answered sqrt(98 / 5) over
        the float, integer and decimal columns alike. ``STDDEV`` is the sample
        deviation here too; ``STDDEV_SAMP`` says so by its name.
        """
        return {"median": func.median, "stddev": func.stddev_samp}


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
    # PostgreSQL's reader gate (#124), and the dialect other engines answer on:
    # `impostors` is where that is written down.
    "postgresql": PostgreSQLBackend(),
    # Both had #124's hole and both answer the gate's catalog questions,
    # measured. CockroachDB has a dialect of its own, and a subclass of its own
    # for the one answer it gives differently, its median (#95); YugabyteDB
    # arrives through PostgreSQL's `-YB-` impostor, which resolves to this entry.
    "cockroachdb": CockroachDBBackend(),
    "yugabytedb": PostgreSQLBackend(name="yugabytedb"),
    # The same hole and the same gate, through a dialect of its own, with row
    # security asked of its own catalog.
    "opengauss": OpenGaussBackend(),
    "mysql": MySQLBackend(),
    "mariadb": MySQLBackend(name="mariadb"),
    "oracle": OracleBackend(),
    "mssql": MSSQLBackend(),
    "clickhousedb": ClickHouseBackend(),
    "trino": TrinoBackend(),
    "crate": CrateDBBackend(),
    "monetdb": MonetDBBackend(),
    # The one key here that is not the name of an engine. `sqlalchemy-firebirdsql`
    # registers the dialect under its *driver's* name, so this is what a URL
    # resolves to, while the backend it maps to calls itself `firebird` — which is
    # what a refusal has to say.
    "firebirdsql": FirebirdBackend(),
    # The second key here that is not the name of an engine, and for a third
    # distinct reason: `ydb-sqlalchemy` registers its dialect as `yql`, after
    # YDB's query language. Firebird's key is a driver's name, YugabyteDB's
    # problem was another engine's name, and this is neither.
    "yql": YDBBackend(),
    # The third key that is not the name of an engine, and the first where the
    # engine's own package is what spells it differently: `sqlalchemy-exasol`
    # registers `exa`, and the dialect it registers calls itself `exasol`.
    "exa": ExasolBackend(),
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
