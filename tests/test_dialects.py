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
from sqlalchemy import Text, text
from sqlalchemy.engine import make_url

from endpoints import Unavailable
from localdata_mcp.dialects import (
    BACKENDS,
    Backend,
    MySQLBackend,
    SQLiteBackend,
    UnsupportedOperation,
    backend_for,
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

    hostile = "p@ss:w/rd?x#y"
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


def test_every_endpoint_builder_round_trips_its_own_credentials():
    """The rule holds for all of them, so a new endpoint cannot quietly opt out.

    ``tests/endpoints.py`` gains one builder per database as the backend
    catalogue is worked through, and the obvious way to write the next one is to
    copy the last. This sweeps the table rather than naming builders, so a
    builder added later is covered the day it appears.
    """
    import endpoints

    hostile = "p@ss:w/rd?x#y"
    for endpoint in endpoints.ENDPOINTS:
        environment = {
            key: hostile
            for key in (
                "POSTGRES_PASSWORD",
                "MYSQL_PASSWORD",
                "MARIADB_PASSWORD",
                "MSSQL_SA_PASSWORD",
                "APP_USER_PASSWORD",
                "CLICKHOUSE_PASSWORD",
            )
        }
        environment.update(
            {
                key: "testuser"
                for key in ("POSTGRES_USER", "MYSQL_USER", "MARIADB_USER", "APP_USER")
            }
        )
        environment["CLICKHOUSE_USER"] = "testuser"
        environment.update(
            {
                key: "testdb"
                for key in ("POSTGRES_DB", "MYSQL_DATABASE", "MARIADB_DATABASE")
            }
        )
        environment["CLICKHOUSE_DB"] = "testdb"

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
