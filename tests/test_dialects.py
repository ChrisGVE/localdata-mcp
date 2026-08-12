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

import os
import tempfile
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
    article_for,
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

#: What every *non*-credential field is made in the sweeps below — a name, a
#: database, a path. Hostile in the same way and for the same reason, and short
#: of two characters, which is the finding rather than a concession.
#:
#: ``URL.create`` quotes the username, the password and every query value, and
#: renders the **database raw**; ``make_url`` unquotes the first two and leaves
#: the database alone as well. Measured, that makes two characters unusable in a
#: database name whatever a caller does: a ``?`` ends the database and starts the
#: query, always, and an ``@`` is read as the userinfo separator whenever no
#: password precedes it. ``#``, ``:``, ``/``, a space and a literal ``%`` all
#: survive intact.
#:
#: Using the hostile *password* in every field, which is what these sweeps used
#: to do, therefore produced URLs that could not be parsed back at all — and the
#: sweep passed anyway, because it asserted the host, the port and the password
#: and never the database. The database was silently arriving truncated. Two
#: values rather than one is what lets the database be asserted at all.
#:
#: **Both characters are fixed upstream in SQLAlchemy 2.1** (issue #11234), which
#: has not reached a stable release — this project pins 2.0.51. When it moves,
#: this constant can go back to being the hostile password; see
#: :func:`test_a_url_carries_a_database_name_that_two_characters_can_still_break`.
HOSTILE_NAME = "n#me:with/parts"


def _hostile(key: str) -> str:
    """The hostile value for a variable, by what kind of thing it names.

    A rule about kinds of secret rather than a list of variable names: the list
    is what decayed before (see :class:`_EveryValueHostile`), and every image in
    this harness spells its password with ``PASS`` in the variable's name —
    ``POSTGRES_PASSWORD``, ``MSSQL_SA_PASSWORD``, ``MDB_DB_ADMIN_PASS``,
    ``QUERY_DEFAULT_PASSWORD``. A new image that does not can only fail towards
    the safer of the two values.
    """
    return HOSTILE_PASSWORD if ("PASS" in key or "PWD" in key) else HOSTILE_NAME


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
    and database names come back hostile too: the assertion is that a value
    survives the round trip as a *value*, and a name full of URL delimiters is
    the same demand made of one more field.

    What it does **not** do any more is answer every key with the same string.
    That version said the uniformity "costs nothing", and it cost the whole
    database assertion — see :data:`HOSTILE_NAME`. A password may hold a ``?``;
    a database name that holds one cannot be expressed in a URL at all, so a
    sweep that put one there had to give up on parsing the URL back.
    """

    def __missing__(self, key: str) -> str:
        return _hostile(key)


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
        # The database survives too, which this could not assert while every
        # field carried the same `?`-bearing string. Several builders wrap the
        # value — Firebird prefixes a directory, Trino names a fixed catalog — so
        # what is asserted is that the hostile name is not *mangled*, not that it
        # is the whole component.
        if parsed.database is not None and HOSTILE_NAME[:4] in parsed.database:
            assert HOSTILE_NAME in parsed.database, endpoint.dialect
        if parsed.password is None:
            # CockroachDB runs --insecure and takes no password, so there is no
            # credential to make hostile. Its builder is covered by the host and
            # port above — an interpolated URL would have lost both.
            continue
        assert parsed.password == hostile, endpoint.dialect
        # Directly, and not only through the parse: a credential that reached the
        # URL as text would appear in it as text. Nothing else in the environment
        # carries this value any more, so its absence is about the password.
        assert hostile not in built, endpoint.dialect


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
# The authentication axis — the same databases, reached other ways
# ---------------------------------------------------------------------------


def test_every_target_has_its_own_identity_including_its_auth_mode():
    """The same defect as #44, one axis further out, and the same assertion.

    A target's name keys the probe cache. Two targets sharing one would mean the
    second reading the first's ``Reached`` out of it and running its whole suite
    against a mode it never used — reporting an authentication path as covered
    when the connection was made another way. That is exactly issue #44's shape
    with ``mode`` in place of ``dialect``, and it is worth asserting separately
    because the endpoint-level test above passes while it is broken.
    """
    import endpoints

    names = [target.name for target in endpoints.TARGETS]
    assert len(names) == len(set(names)), f"duplicate target names: {names}"

    # Every endpoint's own credentialed mode is a target, and it is the first of
    # that endpoint's. A mode list that replaced it rather than adding to it
    # would leave the ordinary path untested while looking like more coverage.
    for endpoint in endpoints.ENDPOINTS:
        mine = [t for t in endpoints.TARGETS if t.endpoint is endpoint]
        assert mine, endpoint.name
        assert mine[0].auth is None, endpoint.name
        assert len(mine) == 1 + len(endpoint.auth), endpoint.name


def test_every_target_names_a_service_the_compose_file_defines():
    """A mode may redirect to its own container, and that container must exist.

    A mode needs a service of its own whenever the authentication method is a
    property of the *server* — PostgreSQL's ``trust``, ClickHouse's user with an
    empty password. Naming one the compose file does not define fails at probe
    time with a clear message, but only on a machine where the rest of the
    container is up; here it fails everywhere, including a laptop with no Docker
    at all, which is where a typo is actually made.
    """
    import yaml

    import endpoints

    defined = set(yaml.safe_load(endpoints.COMPOSE.read_text())["services"])
    for target in endpoints.TARGETS:
        assert target.service in defined, (
            f"{target.name} names service {target.service!r}, which "
            f"{endpoints.COMPOSE.name} does not define"
        )


def test_every_auth_mode_builder_round_trips_its_own_credentials():
    """The sweep the endpoint builders get, applied to the modes as well.

    A mode builds a URL the same way an endpoint's does and can get it wrong the
    same way, so it is swept the same way rather than trusted for being newer.
    What differs is the assertion about the password: half these modes exist
    precisely because the credential is **not** in the URL, so "the password
    survives" becomes "if there is one, it survives" — and the modes with none
    are still held to the host and the port, which an interpolated URL loses
    first.
    """
    import endpoints

    swept = 0
    for endpoint in endpoints.ENDPOINTS:
        for mode in endpoint.auth:
            environment = _EveryValueHostile()
            scratch = Path(tempfile.mkdtemp(prefix="localdata-auth-sweep-"))
            try:
                reached = mode.reach(environment, 15432, scratch)
            except Unavailable:
                continue
            swept += 1

            parsed = make_url(reached.url)
            if parsed.port is None:
                # A URL with no port is naming a **data source** rather than an
                # address — the ODBC DSN shape — and the host position holds
                # that name. There is no address in it to check, so what is
                # asserted is that something is there to look up.
                assert parsed.host, mode.mode
            else:
                assert parsed.host == endpoints.HOST, mode.mode
                assert parsed.port == 15432, mode.mode
            if parsed.password is not None and parsed.password != "":
                assert parsed.password == HOSTILE_PASSWORD, mode.mode
            # The URL never carries the credential as text — including for the
            # modes whose whole point is that it carries it not at all, where
            # this is what says so.
            assert HOSTILE_PASSWORD not in reached.url, mode.mode
            # A mode may name its credential file in the URL rather than in the
            # environment — MySQL's option file is `read_default_file`, a client
            # key is `sslkey` — and the file is its responsibility either way.
            #
            # Certificates are the exception and not an oversight: a CA
            # certificate and a client certificate are *published* halves, and
            # holding them to 0600 would be asserting that a public key is a
            # secret. The parameter's name is what says which kind it is.
            for parameter, value in parsed.query.items():
                if "cert" in parameter or not isinstance(value, str):
                    continue
                if Path(value).is_file():
                    assert Path(value).stat().st_mode & 0o077 == 0, (
                        f"{mode.mode} names {value} as {parameter}, "
                        f"and others can read it"
                    )
            # An environment variable is a value the way a URL component is not:
            # nothing re-reads it as syntax, so a mode that carries a credential
            # there must hand it over verbatim rather than escaping it for a
            # format it is not going into.
            for name, value in reached.environ.items():
                if name.endswith("PASSWORD"):
                    assert value == HOSTILE_PASSWORD, f"{mode.mode}:{name}"
            # Everything a mode writes into its scratch directory is private to
            # this process: a password file, an option file, a credential cache,
            # a Kerberos configuration. Rather than deciding per file which of
            # them holds a secret — a judgement that gets it wrong once and then
            # leaks — the rule is that the whole directory is 0600. A file that
            # is genuinely public does not belong in it, and none is: the
            # certificates live where the CA service put them.
            for written in sorted(scratch.rglob("*")):
                if written.is_file():
                    assert (
                        written.stat().st_mode & 0o077 == 0
                    ), f"{mode.mode} wrote {written.name} readable by others"

    assert swept, "no auth modes were swept, so this asserted nothing"


def test_the_pgpass_mode_escapes_a_password_that_holds_the_field_separator():
    """``.pgpass`` is colon-separated and this endpoint's password holds a colon.

    The same defect as a credential formatted into a URL — a value re-read as
    syntax — arriving through a different file format, and the reason the mode
    exists in a harness whose password is deliberately hostile. Unescaped, libpq
    reads the password as everything up to the first colon and the connection
    fails naming the credential rather than the file that mangled it.

    The reverse parser below is written from libpq's rule rather than shared with
    the builder, so this compares two independent readings of the format instead
    of comparing the builder with itself. It is still only a model of libpq —
    what proves the escaping against the real one is the endpoint suite
    connecting through this mode to a live server.
    """
    import endpoints

    scratch = Path(tempfile.mkdtemp(prefix="localdata-pgpass-"))
    reached = endpoints._postgres_pgpassfile(_EveryValueHostile(), 15432, scratch)

    passfile = Path(reached.environ["PGPASSFILE"])
    assert passfile.stat().st_mode & 0o077 == 0, (
        "libpq ignores a pgpass file that is group- or world-readable, and then "
        "reports 'no password supplied' rather than saying so"
    )
    fields = _unescaped_fields(passfile.read_text().rstrip("\n"))
    assert len(fields) == 5, fields
    assert fields[4] == HOSTILE_PASSWORD


def _unescaped_fields(line: str) -> list[str]:
    """Split a ``.pgpass`` line on its *unescaped* colons, undoing the escapes.

    libpq's rule, stated in its own documentation: a colon or a backslash inside
    a field is written with a leading backslash, and nothing else is special.
    """
    fields: list[str] = [""]
    escaped = False
    for character in line:
        if escaped:
            fields[-1] += character
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":":
            fields.append("")
        else:
            fields[-1] += character
    return fields


def test_a_url_carries_a_database_name_that_two_characters_can_still_break():
    """The limit :data:`HOSTILE_NAME` exists for, pinned so it cannot widen quietly.

    ``URL.create`` quotes the username, the password and the query values and
    renders the database **raw**; ``make_url`` unquotes the first two and leaves
    the database alone in turn. The pair is self-consistent and it means two
    characters cannot appear in a database name in any SQLAlchemy URL string,
    however carefully it is built:

    * a ``?`` ends the database and begins the query — the name arrives
      **truncated**, with no error anywhere;
    * an ``@`` is read as the userinfo separator, so with no password before it
      the host becomes whatever followed the ``@``.

    Percent-encoding is not a way round it, because nothing decodes the database
    on the way back — an encoded name reaches the driver encoded.

    It bounds the URL strings a caller may hand to ``attach`` — the only place a
    URL is parsed rather than built — and **not** the local-file path:
    ``Backend.open_file`` passes a ``URL`` *object* to ``create_engine`` and never
    renders it, which is why
    :func:`test_a_question_mark_in_a_filename_survives_the_generic_file_open`
    passes on this same version. Where it reaches for real is Firebird, whose
    database component is a **filesystem path**.

    **It is a fact about the version, and the version is the one that ships.**
    Upstream reported it as sqlalchemy/sqlalchemy#11234 — whose reproducer is
    this exact case, on a SQLite *filename* — and fixed it in commit
    ``feb17832f``, milestone **2.1**: the database is quoted on the way out and
    unquoted on the way back, symmetrically. That has not reached a stable
    release. The newest stable is **2.0.51**, which is what this project pins and
    what these assertions describe; 2.1 exists as betas only, and **2.1.0b3 was
    measured in a throwaway virtualenv to round-trip every case below**,
    including the two that fail here.

    So this test is expected to **fail on the day this project moves to 2.1**,
    and that failure is the signal rather than a regression: delete the two
    assertions at the bottom, move their cases up into the surviving list, and
    drop the Firebird caveat from ``CONSTRAINTS.md`` §25 and from
    :data:`HOSTILE_NAME`.
    """
    from sqlalchemy.engine import URL

    def round_trip(database: str, password: str | None = "pw"):
        url = URL.create(
            "postgresql+psycopg",
            username="u",
            password=password,
            host="127.0.0.1",
            port=15432,
            database=database,
        )
        return make_url(url.render_as_string(hide_password=False))

    # Survive: a fragment marker, the field separator, a path separator, a space
    # and a literal percent. This is what makes HOSTILE_NAME usable.
    for benign in ("plain", "a#b", "a:b", "a/b", "a b", "a%2Fb", HOSTILE_NAME):
        assert round_trip(benign).database == benign, benign

    # Do not, and silently. Both are fixed in 2.1 and neither is fixed in any
    # release this project can pin — see the docstring before changing them.
    assert round_trip("a?b").database == "a"
    assert round_trip("a@b", password=None).database is None
    assert round_trip("a@b", password=None).host == "b"


def test_applying_a_mode_environment_puts_the_machine_back_as_it_was():
    """A credential left in the environment makes the *next* target pass wrongly.

    The quietest failure this axis can have: ``PGPASSWORD`` still set after the
    mode that needed it would let a passwordless URL connect for a reason that
    has nothing to do with the mode under test, and the suite would report
    coverage it does not have. Both directions matter — a variable the machine
    already had must come back, not be deleted.
    """
    import endpoints

    already = "LOCALDATA_TEST_ALREADY_SET"
    fresh = "LOCALDATA_TEST_NOT_SET_BEFORE"
    os.environ[already] = "the machine's own value"
    os.environ.pop(fresh, None)
    try:
        with endpoints.applied({already: "the mode's value", fresh: "borrowed"}):
            assert os.environ[already] == "the mode's value"
            assert os.environ[fresh] == "borrowed"
        assert os.environ[already] == "the machine's own value"
        assert fresh not in os.environ
    finally:
        os.environ.pop(already, None)
        os.environ.pop(fresh, None)


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


def test_the_refusal_names_whichever_axis_actually_survives():
    """Both halves of the axis are read, and the caveat is built from the answer.

    Only the DDL half used to be consulted, so a refused ``INSERT`` on CrateDB —
    where the row really does land — came back with a caveat about ``CREATE``
    and ``DROP``. To a caller that reads as *this was not DDL, so nothing
    happened*, which points away from the row instead of at it: worse than no
    caveat at all, since no caveat at least leaves them to check (issue #84).

    Driven through :func:`loader._not_a_read` rather than through a live
    connection, because what is under test is the composition and the three
    backends below give it its three distinct answers. That the row genuinely
    survives on CrateDB is the container test's job, and it asserts it.
    """
    from types import SimpleNamespace

    from localdata_mcp.loader import _not_a_read

    def refusal_for(name: str) -> str:
        return _not_a_read(SimpleNamespace(backend=backend_for(name)))

    # Neither axis: the plain refusal, with nothing to qualify it.
    plain = refusal_for("sqlite")
    assert "caveat" not in plain
    assert "already taken effect" not in plain

    # DDL only. The wording the container test pins is kept verbatim.
    oracle = refusal_for("oracle")
    assert "One caveat specific to oracle:" in oracle
    assert "If this was a CREATE or a DROP" in oracle
    assert "already taken effect" in oracle
    assert "rows" not in oracle.split("One caveat")[1]

    # Both. The row is what the caller has to be told about, and the remedy is
    # not `drop` on what a CREATE made — there is no verb here that deletes rows.
    crate = refusal_for("crate")
    assert "If this was a CREATE or a DROP" in crate
    assert "If it wrote rows, those rows have already taken effect" in crate
    assert "drop on the table and create again" in crate


# ---------------------------------------------------------------------------
# Exasol
# ---------------------------------------------------------------------------


def test_the_exasol_backend_is_keyed_on_its_url_and_named_after_its_engine():
    """The fifth way a dialect name is not an identity, and the narrowest yet.

    The four before this were a dialect borrowed from another engine
    (YugabyteDB), one named after its driver (Firebird), one named after a query
    language (YDB) and one whose name and engine simply agreed (Databend, as the
    control). This one is a single package disagreeing with itself:
    ``sqlalchemy-exasol`` registers the entry point ``exa``, so that is what a
    URL resolves to and what the registry must be keyed on, while the dialect it
    registers answers ``exasol`` when asked its name.

    Both halves are asserted, because either alone passes while the other is
    wrong: keyed on ``exasol`` the lookup misses and a caller is refused by "a
    generic datasource named exa", and named ``exa`` the refusal prints a URL
    scheme at somebody who opened Exasol.
    """
    from sqlalchemy.engine import make_url

    url = "exa+websocket://sys:exasol@127.0.0.1:18563/localdata"

    assert make_url(url).get_backend_name() == "exa"
    assert backend_for("exa").name == "exasol"
    # And the dialect's own answer is the third spelling, so nothing may key on
    # it either: it is a module path, not a driver name.
    assert "exasol" not in BACKENDS


def test_exasol_tells_only_the_read_engine_to_stop_committing():
    """Autocommit is the driver's default, so the floor has to be asked for.

    :meth:`Backend.read_posture`'s guarantee is transactional — open a
    connection, never commit, close it — and it is worth nothing against a driver
    that commits every statement as it runs. This DBAPI does: ``autocommit``
    defaults to ``True``, and an ``INSERT`` followed by ``rollback()`` leaves the
    row behind. ``AUTOCOMMIT=n`` on the read URL is what puts the floor back.

    The write engine must **not** carry it, which is the half worth asserting:
    the same parameter on both would make every write this server does depend on
    an explicit commit reaching a driver that was never asked to defer one.
    """
    exasol = backend_for("exa")

    assert dict(exasol.read_only_query) == {"AUTOCOMMIT": "n"}

    engines = exasol.open(
        "exa+websocket://sys:exasol@127.0.0.1:18563/localdata", writable=True
    )
    try:
        assert engines.read.url.query["AUTOCOMMIT"] == "n"
        assert "AUTOCOMMIT" not in engines.write.url.query
    finally:
        engines.dispose()


def test_exasol_asks_the_driver_for_the_type_mapper_its_dbapi_hardcodes_away(
    monkeypatch,
):
    """Without it a widened number arrives as text, which JSON cannot rescue.

    Exasol's WebSocket protocol sends a ``DECIMAL`` as a JSON **string** once its
    precision outgrows a double, so ``SUM`` over a ``DECIMAL(18,0)`` column —
    widened to ``DECIMAL(29,0)`` — reads back as ``'155000'``. ``_ON_THE_WIRE``
    spells a ``Decimal`` as a number on the way out and cannot help here: a
    ``str`` is indistinguishable from a column that really is text.

    The DBAPI hardcodes ``fetch_mapper`` to ``None`` and exposes no argument for
    it, which is why this goes through the ``connection_class`` its own
    ``connect()`` accepts. Asserted by driving the class rather than by reading
    the seam back: what matters is that the option reaches ``pyexasol``.
    """
    import pyexasol

    connection_class = backend_for("exa").connect_args()["connection_class"]
    connection = connection_class(
        dsn="127.0.0.1:18563", username="sys", password="exasol", schema="localdata"
    )

    captured: dict[str, object] = {}

    class _NeverOpened:
        """Enough of a connection for the wrapper's own destructor to run."""

        def __del__(self) -> None:
            return None

    def fake_connect(**options):
        captured.update(options)
        return _NeverOpened()

    monkeypatch.setattr(pyexasol, "connect", fake_connect)
    connection.connect()

    assert captured["fetch_mapper"] is pyexasol.exasol_mapper


def test_an_index_exasol_keeps_for_itself_is_refused_rather_than_reported():
    """This database indexes itself, so there is no name to hand back.

    Exasol creates and drops indexes from the queries it actually runs and offers
    no statement for making one: the dialect refuses ``CREATE INDEX`` at compile
    time with *"Exasol manages indexes internally"*, and the raw statement is
    refused by the server too. Reporting one as created would be the shape this
    server refuses elsewhere — an answer that reads as done and cannot be acted
    on — which is why ClickHouse, Trino and Databend all reach this same axis by
    different roads.
    """
    from sqlalchemy import Column, Integer, MetaData, Table

    backend = backend_for("exa")
    assert backend.builds_indexes() is False

    table = Table("orders", MetaData(), Column("region", Integer))
    with pytest.raises(UnsupportedOperation) as refused:
        backend.build_index("ix_orders_region", table, ["region"])

    message = str(refused.value)
    # The engine, not the URL scheme it is registered under.
    assert "exasol" in message
    assert "region" in message
    # And it says why, so the refusal does not read as this server's own limit.
    assert "maintains its own indexes" in message


def test_exasol_declines_the_one_type_its_dialect_will_not_compile():
    """One, and refused before anything is sent — the honest end of this axis.

    Exasol has no binary column type, and the dialect says so at *compile* time:
    ``BLOB is not supported by the Exasol dialect``, with no table made and no
    value bound. ClickHouse's binary column can be created and not written to,
    and Databend's works until the bytes stop being valid UTF-8; this one does
    not exist and nothing pretends it does.

    The rest is asserted rather than left to omission: ``Time`` is where three
    other backends here fail, and Exasol takes it.
    """
    unstorable = backend_for("exa").unstorable_column_types()

    assert unstorable == {"LargeBinary"}
    assert not {"Time", "Numeric", "Date", "DateTime", "Boolean", "Text"} & unstorable


def test_the_article_holds_for_a_name_this_file_has_never_seen():
    """Including the empty one — `"" in "aeiou"` is True, and would take `an`."""
    assert article_for("") == "a"
    assert article_for("exasol") == "an"
    assert article_for("postgresql") == "a"
    assert article_for("mssql") == "an"


# ---------------------------------------------------------------------------
# MySQL's index key budget
# ---------------------------------------------------------------------------


def _mysql_key_bytes(table, columns, lengths):
    """What InnoDB will charge for this key, on the code's own utf8mb4 basis.

    Four bytes a character, the figure `dialects.py` derives its prefix from —
    a prefixed column costs its prefix, a `VARCHAR(n)` costs all `n`, and the
    point of the test is that both land in the same key.
    """
    total = 0
    for column in columns:
        kind = table.c[column].type
        if column in lengths:
            total += 4 * lengths[column]
        elif getattr(kind, "length", None):
            total += 4 * kind.length
    return total


def test_a_mysql_prefix_is_budgeted_against_the_whole_key_not_its_unbounded_part():
    """A bounded column in the same index still spends the key, so it must count.

    Confirmed against MySQL 8.0.46 (localdata#93): a `VARCHAR(600)` beside a
    `TEXT` was indexed as `(wide_varchar, some_text(255))`, because the 3072-byte
    budget was divided among the unbounded columns *alone* and the 2400 bytes the
    `VARCHAR` costs were never subtracted. 2400 + 1020 = 3420, and InnoDB refused
    with error 1071.

    The assertion is the budget rather than a particular prefix: what has to hold
    is that the key fits, not that the arithmetic picked any one number.
    """
    from sqlalchemy import Column, MetaData, String, Table, Text

    backend = backend_for("mysql")
    table = Table(
        "wide",
        MetaData(),
        Column("wide_varchar", String(600)),
        Column("some_text", Text),
    )

    index, notes = backend.build_index("ix_wide", table, ["wide_varchar", "some_text"])

    lengths = index.kwargs["mysql_length"]
    assert _mysql_key_bytes(table, ["wide_varchar", "some_text"], lengths) <= 3072
    # And the note still says what was shortened, since that is why it exists.
    assert "some_text" in " ".join(notes)


def test_a_mysql_index_with_room_to_spare_still_gets_the_conventional_prefix():
    """The control: narrowing must not become the answer to every index.

    Without this, the fix above passes just as well by clamping every prefix to
    one character. A lone `TEXT` column has the whole budget and should keep the
    255 characters the dialect calls conventional and ample.
    """
    from sqlalchemy import Column, MetaData, Table, Text

    backend = backend_for("mysql")
    table = Table("narrow", MetaData(), Column("some_text", Text))

    index, _ = backend.build_index("ix_narrow", table, ["some_text"])

    assert index.kwargs["mysql_length"] == {"some_text": 255}


def test_a_mysql_index_with_no_room_left_is_refused_naming_the_column_to_drop():
    """When the bounded column eats the key, say which one, not how many bytes.

    What MySQL returns in this case is `1071, Specified key was too long` and a
    byte count, which does not name the column a caller would have to remove.
    Every other engine that cannot build an index refuses in this server's own
    words, and this is the one MySQL case where the arithmetic can see the
    answer before the statement is sent.
    """
    from sqlalchemy import Column, MetaData, String, Table, Text

    backend = backend_for("mysql")
    table = Table(
        "huge",
        MetaData(),
        Column("enormous", String(800)),
        Column("some_text", Text),
    )

    with pytest.raises(UnsupportedOperation) as refused:
        backend.build_index("ix_huge", table, ["enormous", "some_text"])

    message = str(refused.value)
    assert "enormous" in message, "the caller is not told which column to drop"
    assert "800" in message
    assert "3072" in message
