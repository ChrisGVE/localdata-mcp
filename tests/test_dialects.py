"""What a backend has to supply, and what it gets for free.

The rule these defend is a negative one: **a database does not have to be
enumerated here to be usable.** Reaching a datasource is ``create_engine``'s job
and it already works for everything SQLAlchemy speaks, so :func:`backend_for`
answers for every dialect and refuses none. A dialect earns an entry in
``BACKENDS`` only when it can say something the generic answer cannot.

The failure this replaces is worth naming, because it is the one that made the
"anything SQLAlchemy speaks" claim false: ``backend_for`` used to raise for any
dialect nobody had subclassed, so a Postgres URL could not be opened at all —
not because SQLAlchemy could not reach it, but because this module had not been
told about it.
"""

from __future__ import annotations

from pathlib import Path

import foreign
import pytest
from sqlalchemy import DOUBLE_PRECISION, Integer, String, Text, text
from sqlalchemy.engine import make_url

from endpoints import Unavailable
from localdata_mcp.dialects import (
    BACKENDS,
    Backend,
    DatabendBackend,
    FirebirdBackend,
    MySQLBackend,
    ReadRefused,
    Refusal,
    SQLiteBackend,
    UnsupportedOperation,
    YDBBackend,
    backend_for,
    _DATABEND_NOT_A_QUERY,
    _FIREBIRD_VARCHAR_MAX,
    _YDB_ISOLATION,
    _YDB_READ_ONLY,
    _YDB_SCHEME_IN_TRANSACTION,
)


def build_database(path: Path, *, dialect: str = "sqlite") -> Path:
    return foreign.build_database(
        path,
        "products",
        [("sku", Text), ("name", Text)],
        [("a", "Widget")],
        dialect=dialect,
    )


# ---------------------------------------------------------------------------
# An unregistered dialect is usable, not refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", ["postgresql", "mysql", "duckdb", "wobble"])
def test_every_dialect_gets_a_backend(dialect):
    """No dialect is turned away. Being unlisted is the ordinary case."""
    assert isinstance(backend_for(dialect), Backend)


def test_a_registered_dialect_gets_its_own_answers():
    assert isinstance(backend_for("sqlite"), SQLiteBackend)
    assert BACKENDS["sqlite"] is backend_for("sqlite")


def test_the_generic_backend_actually_opens_a_datasource(tmp_path):
    """The generic path is real code, not a placeholder that raises.

    Exercised against SQLite because it is the only driver present without a
    server — but through :class:`Backend`, not :class:`SQLiteBackend`, so what
    runs here is the same ``create_engine`` call any other dialect would get.
    """
    database = build_database(tmp_path / "generic.db")
    engines = Backend().open(f"sqlite:///{database}", writable=False)
    try:
        with engines.read.connect() as connection:
            rows = connection.execute(text("SELECT sku FROM products")).fetchall()
        assert [row[0] for row in rows] == ["a"]
    finally:
        engines.dispose()


def test_the_generic_read_engine_does_not_persist_a_write(tmp_path):
    """The generic read floor: a write may execute, but none of it survives.

    Transactional rather than refusing-at-preparation, which is what a backend
    with no dialect-specific posture can honestly promise. SQLite raises that
    floor and refuses outright; this is what everything else still guarantees.
    """
    database = build_database(tmp_path / "generic.db")
    engines = Backend().open(f"sqlite:///{database}", writable=False)
    try:
        with engines.read.connect() as connection:
            connection.execute(text("INSERT INTO products VALUES ('b', 'Gadget')"))
            # No commit, and the connection closes here.
        with engines.read.connect() as connection:
            total = connection.execute(
                text("SELECT count(*) FROM products")
            ).scalar_one()
        assert total == 1
    finally:
        engines.dispose()


# ---------------------------------------------------------------------------
# What an unregistered dialect loses — honestly, and only the extras
# ---------------------------------------------------------------------------


def test_residency_is_unknown_rather_than_zero(tmp_path):
    """``None`` means the question does not apply; ``0`` would read as empty."""
    database = build_database(tmp_path / "generic.db")
    engines = Backend().open(f"sqlite:///{database}", writable=False)
    try:
        assert Backend().resident_bytes(engines.write) is None
    finally:
        engines.dispose()


def test_a_generic_datasource_says_why_it_cannot_be_saved(tmp_path):
    """Refused with a route out, rather than silently producing a wrong file."""
    database = build_database(tmp_path / "generic.db")
    engines = Backend().open(f"sqlite:///{database}", writable=False)
    try:
        with pytest.raises(UnsupportedOperation) as raised:
            Backend().snapshot(engines.write, tmp_path / "copy.db")
        assert "with create" in str(raised.value)
    finally:
        engines.dispose()


def test_storage_classes_are_empty_where_the_question_is_meaningless(tmp_path):
    """A backend with real column types has one class per column by construction."""
    database = build_database(tmp_path / "generic.db")
    engines = Backend().open(f"sqlite:///{database}", writable=False)
    try:
        with engines.read.connect() as connection:
            assert Backend().storage_classes(connection, "products", "sku") == {}
    finally:
        engines.dispose()


def test_the_refusal_names_the_database_the_caller_opened(tmp_path):
    """Not "a generic datasource" — that names our fallback, not their database.

    An unregistered dialect gets a plain :class:`Backend`, and it used to be one
    shared instance called "generic", so this sentence reached a PostgreSQL user
    naming a database that does not exist. Same defect MariaDB being registered
    in its own right already avoids.
    """
    with pytest.raises(UnsupportedOperation) as raised:
        backend_for("postgresql").snapshot(None, tmp_path / "copy.db")
    assert "postgresql" in str(raised.value)
    assert "generic" not in str(raised.value)


# ---------------------------------------------------------------------------
# Opening a file: the generic answer, and what a path may contain
# ---------------------------------------------------------------------------


def test_a_question_mark_in_a_filename_survives_the_generic_file_open(tmp_path):
    """A path is a value, not URL syntax — on every dialect, not just SQLite.

    ``loader`` used to build this URL by formatting a string, so ``why? not``
    split at the ``?`` and the database became ``why``. SQLite never showed it
    because its own override goes through ``as_uri()``; every *other* file-based
    dialect took the string path, and nothing exercised one. This is that gap:
    DuckDB, through the generic :meth:`Backend.open_file`.
    """
    awkward = build_database(
        tmp_path / "why? not.duckdb",
        dialect="duckdb",
    )
    engines = backend_for("duckdb").open_file(awkward, writable=False)
    try:
        with engines.read.connect() as connection:
            rows = connection.execute(text("SELECT sku FROM products")).fetchall()
        assert [row[0] for row in rows] == ["a"]
    finally:
        engines.dispose()


def test_the_generic_file_open_carries_the_read_only_posture(tmp_path):
    """DuckDB is *told* read-only in the URL, rather than merely not committed.

    The fact lives on the backend now. It used to be a dictionary in ``loader``
    keyed by dialect name, which is a dispatch on dialect name in shared code.
    """
    database = build_database(tmp_path / "warehouse.duckdb", dialect="duckdb")
    engines = backend_for("duckdb").open_file(database, writable=False)
    try:
        assert engines.read.url.query["access_mode"] == "read_only"
        with engines.read.connect() as connection:
            with pytest.raises(Exception):
                connection.execute(text("INSERT INTO products VALUES ('b', 'Gadget')"))
    finally:
        engines.dispose()

    writable = backend_for("duckdb").open_file(database, writable=True)
    try:
        # Writable is the absence of the posture, not a different code path.
        assert "access_mode" not in writable.write.url.query
    finally:
        writable.dispose()


def test_the_generic_url_open_carries_the_read_only_posture(tmp_path):
    """A server URL is not a weaker claim on the posture than a file path is.

    ``open_file`` has always carried :attr:`Backend.read_only_query`; ``open``
    did not, and left the read engine to the transactional floor. That was
    adequate only while every endpoint had a floor to stand on. ClickHouse has
    no transactions, so the URL is the *only* place its posture can be stated —
    and a dialect that can be told read-only in the URL should be told so
    however it was reached.

    Exercised on DuckDB rather than on ClickHouse because it needs no container:
    the behaviour under test is generic, and picking the dialect that is
    reachable from a file keeps it in the fast suite.
    """
    database = build_database(tmp_path / "warehouse.duckdb", dialect="duckdb")
    engines = backend_for("duckdb").open(f"duckdb:///{database}", writable=True)
    try:
        # The read engine is told, and the write engine is not — writable is the
        # absence of the posture on the writer, not a second code path.
        assert engines.read.url.query["access_mode"] == "read_only"
        assert "access_mode" not in engines.write.url.query
    finally:
        engines.dispose()


def test_a_backend_with_nothing_to_say_leaves_the_url_alone(tmp_path):
    """An empty ``read_only_query`` must add nothing, not an empty query string.

    The generic floor is the whole guarantee for most dialects, and a URL that
    gained a stray ``?`` on the way to the read engine would be a change to
    every one of them for the benefit of none.
    """
    database = build_database(tmp_path / "plain.db")
    engines = backend_for("postgresql").open(f"sqlite:///{database}", writable=True)
    try:
        assert engines.read.url == engines.write.url
    finally:
        engines.dispose()


#: The password the Postgres container really carries, and the one both tests
#: below build with. Every character in it moves a URL boundary — ``@`` starts
#: the host, ``/`` the path, ``?`` the query, ``#`` the fragment — so a builder
#: that interpolates rather than passing values cannot survive it.
HOSTILE_PASSWORD = "p@ss:w/rd?x#y"


def test_an_endpoint_url_survives_a_hostile_password():
    """The harness may not interpolate a credential into a URL. See issue #43.

    A URL is parsed, not concatenated. Every character below moves a boundary:
    ``@`` starts the host, ``/`` starts the path, ``?`` starts the query, ``#``
    starts the fragment. Built by formatting, this password reaches the driver
    as something else entirely and the harness connects somewhere nobody
    configured.

    The assertion is on the **parsed** value rather than on the rendered text,
    because that is what the driver will act on — and it is what fails on the
    formatted version, where ``make_url`` reads ``h@ck`` as a host.
    """
    import endpoints

    hostile = HOSTILE_PASSWORD
    built = endpoints._url(
        "postgresql+psycopg",
        username="us@r",
        password=hostile,
        port=15432,
        database="testdb",
    )

    parsed = make_url(built)
    assert parsed.password == hostile
    assert parsed.username == "us@r"
    assert parsed.host == endpoints.HOST
    assert parsed.port == 15432
    assert parsed.database == "testdb"


class _EveryValueHostile(dict):
    """An environment that answers *any* variable with the same hostile value.

    What this replaces was a list of the variable names the builders happened to
    read — which is a per-endpoint fact living in a fixture, and it decayed the
    way those do. A builder added for a database whose image names its password
    something new raised :class:`KeyError` out of the sweep rather than being
    covered by it, so the test that advertises "a builder added later is covered
    the day it appears" instead broke on the day it appeared. MonetDB, whose
    image asks for ``MDB_DB_ADMIN_PASS``, is the one that proved it.

    Answering every key removes the list rather than lengthening it. Usernames
    and database names come back hostile too, which costs nothing: the assertion
    is that a credential survives the round trip as a *value*, and a name full of
    URL delimiters is the same demand made of one more field.
    """

    def __missing__(self, key: str) -> str:
        return HOSTILE_PASSWORD


def test_every_endpoint_builder_round_trips_its_own_credentials():
    """The rule holds for all of them, so a new endpoint cannot quietly opt out.

    ``tests/endpoints.py`` gains one builder per database as the backend
    catalogue is worked through, and the obvious way to write the next one is to
    copy the last. This sweeps the table rather than naming builders, so a
    builder added later is covered the day it appears.
    """
    import endpoints

    hostile = HOSTILE_PASSWORD
    for endpoint in endpoints.ENDPOINTS:
        environment = _EveryValueHostile()

        try:
            built = endpoint.url(environment, 15432)
        except Unavailable:
            # Only MSSQL, and only where no ODBC driver is installed. The
            # builder cannot be exercised without one, and that is a skip
            # everywhere else in this harness too.
            continue

        parsed = make_url(built)
        # Parsing at all is half the assertion: the formatted form this replaced
        # does not survive make_url, it raises.
        assert parsed.host == endpoints.HOST, endpoint.dialect
        assert parsed.port == 15432, endpoint.dialect
        if parsed.password is None:
            # CockroachDB runs --insecure and takes no password, so there is no
            # credential to make hostile. Its builder is covered by the host and
            # port above — an interpolated URL would have lost both.
            continue
        assert parsed.password == hostile, endpoint.dialect


def test_every_endpoint_has_its_own_identity_even_when_it_shares_a_dialect():
    """Two endpoints may share a dialect; they may never share a name. Issue #44.

    A dialect is not an identity. TiDB and OceanBase speak MySQL's wire and have
    no dialect of their own; YugabyteDB, Greenplum and OpenGauss are addressed as
    PostgreSQL. Keyed on dialect, the second of any such pair reads the first's
    URL out of the probe cache and runs its whole suite against a container it
    never named — reporting green for a database that was never reached.

    Asserted over the table rather than at the two places that consume it, so an
    endpoint added later cannot reintroduce it without this failing.
    """
    import endpoints

    names = [endpoint.name for endpoint in endpoints.ENDPOINTS]
    assert len(names) == len(set(names)), f"duplicate endpoint names: {names}"

    services = [endpoint.service for endpoint in endpoints.ENDPOINTS]
    assert len(services) == len(set(services)), f"duplicate services: {services}"


# ---------------------------------------------------------------------------
# A dialect names a wire protocol, not the engine answering on it
# ---------------------------------------------------------------------------


#: Real banners, read from the live containers rather than composed here. A
#: fragment guessed from documentation is exactly the thing this table exists to
#: stop, so nothing goes in it that has not been measured.
POSTGRESQL_BANNER = (
    "PostgreSQL 16.14 on x86_64-pc-linux-musl, compiled by gcc "
    "(Alpine 15.2.0) 15.2.0, 64-bit"
)
YUGABYTEDB_BANNER = (
    "PostgreSQL 15.12-YB-2.25.2.0-b0 on x86_64-pc-linux-gnu, compiled by clang "
    "version 19.1.0 (https://github.com/yugabyte/llvm-project.git "
    "a2a6b655e14e7fa1fcf1011a6cb29cb8575249c0), 64-bit"
)


def test_a_banner_that_names_another_engine_resolves_to_that_engine():
    """The whole point: YugabyteDB answers on PostgreSQL's dialect and is not it.

    Reached as ``postgresql``, it used to get ``Backend(name="postgresql")`` —
    so a refusal named PostgreSQL to somebody who opened YugabyteDB, and there
    was nowhere to put an answer of its own that would not also change real
    PostgreSQL's. See issue #45.
    """
    resolved = backend_for("postgresql").named_by(YUGABYTEDB_BANNER)

    assert resolved.name == "yugabytedb"


def test_the_engine_whose_dialect_it_is_keeps_it():
    """The ordinary case, and the one a loose fragment would break.

    YugabyteDB's banner mentions ``yugabyte`` twice — once in the version and
    once in a compiler URL — so a fragment chosen carelessly is easy. What must
    never happen is the reverse: real PostgreSQL matching one of its impostors
    and being handed somebody else's answers.
    """
    postgresql = backend_for("postgresql")

    assert postgresql.named_by(POSTGRESQL_BANNER) is postgresql


@pytest.mark.parametrize("banner", [None, "", "something else entirely"])
def test_an_unrecognised_banner_changes_nothing(banner):
    """Including when the probe failed and there is no banner to read.

    Falling back to the dialect's own backend is exactly the behaviour before
    any of this existed, so a datasource that cannot answer ``SELECT version()``
    is no worse off than it was — and refusing to attach it because an identity
    probe failed would be very much worse.
    """
    postgresql = backend_for("postgresql")

    assert postgresql.named_by(banner) is postgresql


def test_a_dialect_nobody_shares_is_never_asked():
    """No impostors means no probe, so the ordinary datasource pays nothing."""
    sqlite = backend_for("sqlite")

    assert sqlite.impostors == {}
    assert sqlite.named_by(YUGABYTEDB_BANNER) is sqlite


def test_a_resolved_engine_keeps_whatever_answers_it_has_of_its_own():
    """Resolution hands back the *registered* backend where the engine has one.

    The two halves of #45 need different things. An engine with nothing to say
    of its own needs only its own name, so a refusal stops naming somebody
    else's database. An engine that has earned a subclass — Greenplum, whose
    every ``CREATE TABLE`` needs a ``DISTRIBUTED BY`` clause — needs that
    subclass to actually arrive, or resolution would have replaced one wrong
    answer with another.

    Built here rather than waiting for that entry, so the guarantee is pinned
    before something depends on it.
    """
    real = MySQLBackend(name="pretend")
    BACKENDS["pretend"] = real
    try:
        sharer = Backend(name="shared")
        object.__setattr__(sharer, "impostors", {"PRETEND": "pretend"})
        assert sharer.named_by("Server 1.0 PRETEND build") is real
    finally:
        del BACKENDS["pretend"]


def test_an_impostor_with_no_entry_of_its_own_still_gets_its_own_name():
    """The user-facing half of #45, and all YugabyteDB actually needs.

    ``Backend.name`` is printed at the caller: "A {name} datasource is reached
    over its own connection…". Reached as ``postgresql``, YugabyteDB put
    PostgreSQL's name in front of somebody who never opened PostgreSQL.
    """
    resolved = backend_for("postgresql").named_by(YUGABYTEDB_BANNER)

    assert resolved.name == "yugabytedb"
    assert "yugabytedb" not in BACKENDS, (
        "this asserts the no-entry path; give YugabyteDB a subclass and it "
        "belongs in the test above instead"
    )


# ---------------------------------------------------------------------------
# A dialect may be named after neither the engine nor another engine
# ---------------------------------------------------------------------------


def test_a_dialect_named_after_its_driver_still_names_its_engine():
    """Firebird's dialect is called ``firebirdsql``, which is a Python package.

    The quiet third form of #45. The loud ones are two engines sharing a dialect
    — TiDB on MySQL's, YugabyteDB on PostgreSQL's — and ``impostors`` resolves
    those by asking the server. This one needs no probe and no banner: only
    Firebird speaks this dialect, and the dialect is simply named after the
    driver that carries it, because ``sqlalchemy-firebirdsql`` registers itself
    under the name of ``firebirdsql``.

    Left alone, ``backend_for`` would hand back ``Backend(name="firebirdsql")``
    and every refusal would name a library the caller has never installed
    knowingly, let alone opened. So this is the one :data:`BACKENDS` key that is
    deliberately not the name of an engine, and this test is what says the
    difference is intended rather than a typo somebody should tidy up.
    """
    resolved = backend_for("firebirdsql")

    assert resolved.name == "firebird"
    assert isinstance(resolved, FirebirdBackend)
    # No banner is involved, and asserting that is the point: this identity is
    # known from the dialect alone, so it costs no connection.
    assert not resolved.impostors
    assert resolved.named_by(None) is resolved


def test_a_refusal_names_the_engine_rather_than_the_driver():
    """The user-facing half, the same way #45's is asserted for YugabyteDB."""
    with pytest.raises(UnsupportedOperation) as refused:
        backend_for("firebirdsql").snapshot(None, Path("/tmp/unused.db"))

    assert "firebird datasource" in str(refused.value)
    assert "firebirdsql" not in str(refused.value)


# ---------------------------------------------------------------------------
# The two axes Firebird added
# ---------------------------------------------------------------------------


def test_the_generic_backend_writes_rows_beside_the_schema_that_holds_them():
    """Both new axes answer the permissive way generically.

    Worth pinning rather than assuming: these are the defaults every dialect
    nobody has subclassed relies on, and a default that flipped would split a
    transaction on eleven backends that never needed it and would refuse a
    rename that works everywhere.
    """
    generic = Backend()

    assert generic.sees_new_tables_in_transaction() is True
    assert generic.renames_tables() is True


def test_a_backend_that_cannot_rename_refuses_rather_than_copying():
    """Firebird has no rename-table statement, and the refusal has to be usable.

    Two halves, and the second is the one that matters. That the axis says
    ``False`` is bookkeeping; that :meth:`rename_table` *also* refuses is the
    guarantee, because it is reached by anything that calls it without asking the
    axis first — and a method that silently did nothing would report a rename
    that never happened.

    The refusal names the engine and says what to do instead, including what
    that alternative costs. A caller told only "no" has been told nothing.
    """
    backend = backend_for("firebirdsql")

    assert backend.renames_tables() is False
    with pytest.raises(UnsupportedOperation) as refused:
        backend.rename_table(None, "orders", "sales")

    message = str(refused.value)
    assert "firebird" in message
    assert "orders" in message and "sales" in message
    # The route that does work, and the cost of taking it.
    assert "create" in message and "indexes" in message


def test_the_backend_that_cannot_see_its_own_new_tables_says_so():
    """Firebird's DDL is transactional *and* invisible to its own transaction.

    Asserted next to ``ddl_survives_refusal`` on purpose. The two look adjacent
    and are opposite here: the DDL does **not** survive a refusal, which is the
    floor working, and it still cannot be used by the transaction that ran it.
    A single axis carrying both would have to lie about one.
    """
    backend = backend_for("firebirdsql")

    assert backend.sees_new_tables_in_transaction() is False
    assert backend.ddl_survives_refusal() is False
    assert backend.dml_survives_refusal() is False


def test_firebird_respells_the_two_portable_types_its_dialect_renders_unusably():
    """``Double`` and ``Text`` both need replacing, for opposite reasons (#55, #56).

    ``Double`` renders ``DOUBLE``, which Firebird has no such keyword for, so the
    ``CREATE TABLE`` fails and nothing is made — loud, and therefore harmless.
    ``Text`` renders ``BLOB``, which is created, stores every byte, reads back
    exactly, and then groups by handle instead of by value: five rows become five
    groups with no error anywhere. The silent one is the dangerous one.
    """
    backend = backend_for("firebirdsql")

    assert isinstance(backend.column_type("REAL"), DOUBLE_PRECISION)
    # Sized from the data rather than guessed, so the column is comparable.
    sized = backend.column_type("TEXT", longest=11)
    assert isinstance(sized, String) and sized.length == 11
    # A column with no text in it still has to be a legal width.
    assert backend.column_type("TEXT", longest=None).length == 1
    # Integers need no help, and saying so keeps the override honest about its
    # own scope.
    assert isinstance(backend.column_type("INTEGER"), Integer)


def test_text_too_wide_for_a_varchar_keeps_the_values_and_loses_the_grouping():
    """Beyond the ceiling there is nothing else to use, and truncating is worse.

    The ceiling is in *characters* while Firebird's limit is in bytes, so it is
    set to the widest width that fits under any character set — a database
    created with UTF8 spends four bytes a character and would refuse a
    declaration a single-byte one accepts. That is why the constant is not the
    32,765 the test container happens to allow.
    """
    backend = backend_for("firebirdsql")

    at_ceiling = backend.column_type("TEXT", longest=_FIREBIRD_VARCHAR_MAX)
    assert type(at_ceiling) is String
    assert at_ceiling.length == _FIREBIRD_VARCHAR_MAX

    # ``type(...) is``, not ``isinstance``: Core's ``Text`` is a *subclass* of
    # ``String``, so an isinstance check cannot tell the two answers apart and
    # would pass whichever one came back. The distinction is the assertion.
    beyond = backend.column_type("TEXT", longest=_FIREBIRD_VARCHAR_MAX + 1)
    assert type(beyond) is Text


# ---------------------------------------------------------------------------
# YDB
# ---------------------------------------------------------------------------


def test_the_dialect_named_after_a_query_language_answers_as_the_database():
    """``yql`` is the dialect's name; ``ydb`` is the engine's, and that is what shows.

    The third distinct way a dialect name has failed to be an identity here, and
    each one breaks differently. YugabyteDB borrowed *another engine's* dialect,
    so it was handed that engine's answers. Firebird's dialect is named after the
    Python **driver**, so a refusal named a library. This one is named after the
    **query language**, so a refusal would name a syntax. All three end at the
    same assertion: whatever the key is, the backend knows which database it is.
    """
    backend = backend_for("yql")

    assert isinstance(backend, YDBBackend)
    assert backend.name == "ydb"
    # And the engine's own name is *not* a key, so nothing resolves by accident.
    assert "ydb" not in BACKENDS
    assert backend_for("ydb").name == "ydb"
    assert not isinstance(backend_for("ydb"), YDBBackend)


def test_the_backend_that_demands_a_primary_key_says_so_and_stands_alone():
    """One axis, one backend, and the default is what every other dialect uses.

    The axis is a bare fact — this engine refuses a keyless table — and the
    *response* to it lives in the loader, so a second engine that ever states it
    inherits the whole answer without writing any of it.
    """
    assert backend_for("yql").requires_primary_key() is True
    assert Backend().requires_primary_key() is False
    for dialect in BACKENDS:
        if dialect != "yql":
            assert backend_for(dialect).requires_primary_key() is False, dialect


def test_ydb_has_no_time_of_day_type_and_says_which_type_that_is():
    """The same gap Oracle has, reached independently — hence a set, not a flag.

    Asserted as a set membership rather than as equality with Oracle's so that
    the two can diverge: they share exactly one entry today and nothing says they
    must tomorrow.
    """
    unstorable = backend_for("yql").unstorable_column_types()

    assert "Time" in unstorable
    # Everything else a table somebody else made can hold, this holds — measured.
    assert not {"Numeric", "Date", "DateTime", "LargeBinary", "Boolean"} & unstorable


def test_the_read_posture_asks_for_a_read_only_level_rather_than_a_strict_one():
    """The floor here is a refusal, because there is no rollback to build one on.

    The distinction this pins is the whole finding: Trino's remedy names a
    *transactional* level to take the driver out of autocommit, and that does not
    work here — the level is honoured and the write still lands, because the
    client library binds a cursor's transaction before the transaction exists.
    So the level named must be a **read-only** one, which makes the server refuse
    the statement instead of the client failing to undo it.
    """
    assert "READONLY" in _YDB_ISOLATION.replace(" ", "")

    class Recorder:
        def __init__(self) -> None:
            self.options: dict = {}

        def update_execution_options(self, **options) -> None:
            self.options.update(options)

    engine = Recorder()
    backend_for("yql").read_posture(engine, None)
    assert engine.options == {"isolation_level": _YDB_ISOLATION}


def test_both_of_ydbs_refusals_are_recognised_and_nothing_else_is():
    """Rows and schema are refused by different codes, in different places.

    A row refusal carries its code on a *nested issue*; a schema refusal carries
    it on the *status* and leaves the issue code at zero. Recognising only the
    first leaves ``CREATE`` through ``query`` explained by the driver rather than
    by the verb that does it — a right outcome with a wrong explanation, which is
    exactly the class that passes every test asserting on the outcome.
    """
    backend = backend_for("yql")

    class Issue:
        def __init__(self, code):
            self.issue_code = code

    class Status:
        def __init__(self, value):
            self.value = value

    class Failure(Exception):
        def __init__(self, status=None, issues=()):
            self.status = Status(status) if status is not None else None
            self.issues = issues

    # Rows: the code is on the nested issue.
    assert backend.denies_write(Failure(status=400080, issues=[Issue(_YDB_READ_ONLY)]))
    # Schema: the code is on the status, and the issue carries zero.
    assert backend.denies_write(
        Failure(status=_YDB_SCHEME_IN_TRANSACTION, issues=[Issue(0)])
    )
    # Anything else is not a refusal, and must not be dressed up as one.
    assert not backend.denies_write(Failure(status=400080, issues=[Issue(1020)]))
    assert not backend.denies_write(Failure())
    # An exception carrying none of this structure must not raise on the way past.
    assert not backend.denies_write(ValueError("nothing structured here"))


def test_a_rename_is_issued_as_schema_rather_than_as_an_opaque_statement():
    """``DDL``, not ``text``, and the difference is which path the statement takes.

    Both render the same string. Only ``DDL`` is an ``ExecutableDDLElement``, and
    only that carries the "this is schema" signal a dialect can route on — which
    is the difference between accepted and ``Scheme operations cannot be executed
    inside transaction`` on YDB. Eleven dialects could not tell the two apart,
    which is why it stood as ``text``.

    Asserted on the *type* of what is executed rather than on the SQL, because
    the SQL was never wrong.
    """
    from sqlalchemy.schema import ExecutableDDLElement

    executed = []

    class Preparer:
        @staticmethod
        def quote(name):
            return f'"{name}"'

    class Dialect:
        identifier_preparer = Preparer()

    class Conn:
        dialect = Dialect()

        @staticmethod
        def execute(statement):
            executed.append(statement)

    Backend().rename_table(Conn(), "orders", "sales")

    assert len(executed) == 1
    assert isinstance(executed[0], ExecutableDDLElement)
    assert str(executed[0]) == 'ALTER TABLE "orders" RENAME TO "sales"'


# ---------------------------------------------------------------------------
# Databend
# ---------------------------------------------------------------------------


class _Cursor:
    """A DBAPI cursor that records what it was asked to explain.

    ``accepts`` decides which statements it will take, which is how a test says
    what the *server* thinks of a statement without needing a server. Its
    ``mogrify`` is the real one's contract — substitute the parameters, leave the
    text alone otherwise.
    """

    def __init__(self, accepts) -> None:
        self.accepts = accepts
        self.asked: list[str] = []

    @staticmethod
    def mogrify(statement, parameters):
        return statement % parameters if parameters else statement

    def execute(self, statement):
        self.asked.append(statement)
        if not self.accepts(statement):
            raise RuntimeError(f"the server would not take: {statement}")

    def close(self) -> None:
        return None


class _Connection:
    """The SQLAlchemy ``Connection`` the posture is handed, and one raw cursor."""

    def __init__(self, cursor) -> None:
        self.connection = self
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class _Engine:
    """Just enough engine to catch the listener the posture registers."""

    def __init__(self) -> None:
        self.listeners: dict[str, list] = {}


def _posture(accepts):
    """Install Databend's read posture on a fake engine and return what to drive it.

    Returns the hook itself, the cursor it will interrogate, and the
    :class:`Refusal` it fills in — which together are the whole of what the
    posture does, with no container and no driver.
    """
    import sqlalchemy

    engine = _Engine()
    hooks: list = []
    original = sqlalchemy.event.listens_for

    def capture(target, identifier, **kw):
        def decorate(fn):
            assert target is engine
            assert identifier == "before_cursor_execute"
            hooks.append(fn)
            return fn

        return decorate

    sqlalchemy.event.listens_for = capture
    try:
        refusal = Refusal()
        backend_for("databend").read_posture(engine, refusal)
    finally:
        sqlalchemy.event.listens_for = original

    assert len(hooks) == 1
    cursor = _Cursor(accepts)
    return hooks[0], cursor, refusal


def _offer(hook, cursor, sql, parameters=None, executemany=False):
    hook(_Connection(cursor), cursor, sql, parameters, None, executemany)


def test_the_databend_dialect_answers_as_itself():
    """Four consecutive entries needed the engine spelled out; this one does not.

    Worth an assertion rather than a shrug: the *reason* the previous three
    diverged — a borrowed dialect, a dialect named after its driver, a dialect
    named after a query language — is absent here, and what proves it is the
    backend agreeing with its own key.
    """
    backend = backend_for("databend")

    assert isinstance(backend, DatabendBackend)
    assert backend.name == "databend"


def test_a_statement_the_server_calls_a_query_runs_unmodified():
    """The wrap is explained, never executed, so the caller's SQL is untouched.

    This is the property that makes the posture safe to put in front of every
    read: what the database is asked about is a wrapped copy, and what runs is
    the statement as written. A posture that rewrote the statement could change
    the result — column names, ordering, duplicate labels — and nothing about a
    passing read would show it.
    """
    hook, cursor, refusal = _posture(lambda sql: True)

    _offer(hook, cursor, "SELECT a FROM orders")

    assert cursor.asked == ["EXPLAIN SELECT * FROM (\nSELECT a FROM orders\n)"]
    assert refusal.what is None


def test_a_write_is_refused_before_the_database_runs_it():
    """The whole finding: on this backend nothing *after* the statement can tell.

    An ``INSERT`` here answers with a one-column result set named ``number of
    rows inserted``, and a ``REPLACE INTO`` answers with the table's own columns —
    so the not-a-read floor, which asks whether rows with columns came back, is
    satisfied by both. The refusal therefore has to happen before the statement
    runs, and this asserts that order: the statement is never offered to the
    cursor for execution, only for explanation.
    """
    # The server takes the statement itself but not the statement as a subquery,
    # which is exactly what it does with every write.
    hook, cursor, refusal = _posture(lambda sql: not sql.startswith("EXPLAIN SELECT *"))

    with pytest.raises(ReadRefused):
        _offer(hook, cursor, "INSERT INTO orders (a) VALUES (1)")

    assert refusal.what == _DATABEND_NOT_A_QUERY
    # Two questions, both of them explanations. Nothing was run.
    assert [sql.split()[0] for sql in cursor.asked] == ["EXPLAIN", "EXPLAIN"]
    assert len(cursor.asked) == 2


def test_the_refusal_names_a_verb_that_does_the_job_and_a_verb_for_schema():
    """A caller told only "no" sends the same statement again.

    ``Refusal`` is what :meth:`Workspace._explain` puts after "This statement
    asks to", so this has to read as a phrase in that sentence *and* point
    somewhere. Both halves are asserted because a ``SHOW`` fails this posture too
    — it reads, but it is not a query — and ``info`` is where that caller has to
    be sent.
    """
    assert "write" in _DATABEND_NOT_A_QUERY
    assert "info" in _DATABEND_NOT_A_QUERY
    assert "SHOW" in _DATABEND_NOT_A_QUERY


def test_a_broken_statement_is_diagnosed_by_the_server_not_called_a_write():
    """The wrong-explanation failure this posture nearly shipped with.

    ``SELECT * FROM nowhere`` fails the wrap — but because the table does not
    exist, not because it writes. Its first draft reported it as a statement that
    was not a read, which sends an agent looking for a verb when what it has is a
    typo. So a statement the server cannot explain *either way* is broken, and its
    own diagnosis is what reaches the caller.
    """
    hook, cursor, refusal = _posture(lambda sql: False)

    with pytest.raises(RuntimeError) as failure:
        _offer(hook, cursor, "SELECT * FROM nowhere")

    assert not isinstance(failure.value, ReadRefused)
    assert "the server would not take" in str(failure.value)
    # And no refusal is claimed, so the explainer falls through to the message.
    assert refusal.what is None


def test_the_posture_proves_the_statement_it_is_given_parameters_and_all():
    """Reflection's statements carry placeholders, and they must still be provable.

    ``information_schema`` selects are how this dialect reflects, they arrive with
    pyformat parameters, and they are reads — so the proof has to be made of the
    statement as the driver will send it. A posture that asked about the unbound
    text would be asking about SQL the server never sees.
    """
    hook, cursor, refusal = _posture(lambda sql: "information_schema" in sql)

    _offer(
        hook,
        cursor,
        "select table_name from information_schema.tables where table_schema = %(s)s",
        {"s": "default"},
    )

    assert cursor.asked[0].endswith("table_schema = default\n)")
    assert refusal.what is None


def test_one_parameter_set_is_enough_to_prove_a_repeated_statement():
    """``executemany`` hands a sequence of sets; the statement is the same in each.

    Proving it with the first is proof about the statement, which is what is
    being asked. Reaching into the sequence as though it were one set would
    interpolate a tuple into the text and fail on a statement that is a perfectly
    good read.
    """
    hook, cursor, refusal = _posture(lambda sql: True)

    _offer(hook, cursor, "SELECT %(a)s", [{"a": "1"}, {"a": "2"}], executemany=True)

    assert cursor.asked == ["EXPLAIN SELECT * FROM (\nSELECT 1\n)"]
    assert refusal.what is None


def test_a_read_ending_in_a_comment_is_still_a_read():
    """The closing parenthesis goes on its own line, and this is why.

    A statement ending in a ``--`` comment would comment out a parenthesis put on
    the same line, so the wrap would not parse and a legitimate read would be
    refused — a refusal produced by the posture's own formatting rather than by
    anything the caller wrote.

    The fake server here has to model the comment to be worth anything: it
    discards what follows ``--`` on each line, exactly as a parser does, and then
    asks whether the subquery was ever closed. Asserting only that the wrap ends
    in ``)`` would pass with the parenthesis commented out, which is the bug.
    """

    def parses(sql):
        uncommented = "\n".join(line.split("--")[0] for line in sql.splitlines())
        return uncommented.strip().endswith(")")

    hook, cursor, refusal = _posture(parses)

    _offer(hook, cursor, "SELECT a FROM orders -- only the first column")

    assert refusal.what is None


def test_the_posture_refusal_is_caught_with_everything_else_a_statement_raises():
    """SQLAlchemy wraps what a statement raises, not what a hook raises.

    Without this the refusal would pass through every ``except SQLAlchemyError``
    written to explain a failed statement and reach the caller as a traceback,
    with the sentence :class:`Refusal` prepared for it never read. The driver's
    own error is named for the same reason — it is what a broken statement raises
    out of the same hook.
    """
    errors = backend_for("databend").driver_errors()

    assert ReadRefused in errors
    assert Backend().driver_errors() == ()


def test_an_index_this_backend_cannot_make_is_refused_rather_than_reported():
    """The dialect compiles ``CREATE INDEX`` to the empty string and succeeds.

    Measured: the statement reaching the driver is ``''``, the connector returns
    early for a falsy statement, and afterwards reflection and ``SHOW INDEXES``
    are both empty. So an index reported as created here would be a name the
    caller could neither find nor drop — the shape ClickHouse's entry refuses for
    a different reason, which is why one axis carries both.

    The refusal names the database, because a caller told "no" by "a generic
    datasource" has been told nothing, and it names the statement that does work
    in Databend itself.
    """
    from sqlalchemy import Column, Integer, MetaData, Table

    backend = backend_for("databend")
    assert backend.builds_indexes() is False

    table = Table("orders", MetaData(), Column("region", Integer))
    with pytest.raises(UnsupportedOperation) as refused:
        backend.build_index("ix_orders_region", table, ["region"])

    message = str(refused.value)
    assert "databend" in message
    assert "CLUSTER BY (region)" in message
    # And it says the failure mode, so nobody reads the refusal as pedantry.
    assert "creating nothing" in message


def test_databend_declines_two_column_types_for_two_different_reasons():
    """One absent from the database, one the driver cannot carry.

    ``Time`` does not exist in Databend at all — its parser lists every type it
    accepts and TIME is not among them — and the dialect renders it as
    ``DATETIME``, so the column is *made* and holds a timestamp on the epoch. A
    typed ``select()`` hides that by re-deriving the time; ``query`` reads the
    column as text and sees ``1970-01-01T14:30:00Z``.

    ``LargeBinary`` is the opposite way round and worse than a flat refusal: the
    database stores binary fine, and the driver renders parameters into the
    statement text, so whether a value binds depends on **whether those bytes
    happen to be valid UTF-8**. ``b'\\x00\\x01'`` lands; ``b'\\x00\\xff'`` raises.

    Asserted as membership rather than equality with anyone else's set: Oracle
    and YDB also lack a time type, Trino has the same bytes defect, and none of
    those three shares the other half.
    """
    unstorable = backend_for("databend").unstorable_column_types()

    assert unstorable == {"Time", "LargeBinary"}
    # What it does hold, measured rather than assumed by omission.
    assert not {"Numeric", "Date", "DateTime", "Boolean", "Text"} & unstorable


def test_the_transactionless_backends_do_not_all_answer_the_same_way():
    """Databend has no transactions and still refuses before the write happens.

    CrateDB, the other transactionless entry, declares both survivals because it
    has no posture to put in their place. Databend obtains one from the server, so
    the generic ``False`` is the truthful answer here — and asserting the two
    apart is what keeps "no transactions" from being read as an excuse.
    """
    databend = backend_for("databend")
    crate = backend_for("crate")

    assert databend.dml_survives_refusal() is False
    assert databend.ddl_survives_refusal() is False
    assert crate.dml_survives_refusal() is True
    assert crate.ddl_survives_refusal() is True
