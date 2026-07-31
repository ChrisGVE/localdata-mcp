"""Where the endpoint databases are, and whether one is actually there.

An endpoint database is one this server *reaches* rather than holds — Postgres,
MySQL, MariaDB, SQL Server, Oracle. Each has a container in
``docker-compose.test.yml`` so no dialect is exercised blind, and this module is
the single place that knows how to address one.

**The compose file is the source of truth**, not this module. Ports and
credentials are read out of it rather than restated here, because the failure of
a restated port is a *skip* — the harness looks in the wrong place, finds
nothing, and reports the dialect as "not running" while it is running perfectly
well. That is the fail-open shape this project keeps being bitten by, so the
numbers are taken from the one file that also configures the container.

Three outcomes, and the difference between them is the whole point:

* **Skipped** — nothing is listening, or the driver this URL needs is not
  installed. Neither is a defect in the server, and neither should redden a run
  on a machine without Docker.
* **Failed** — the container *is* answering and something went wrong anyway.
  A drifted service name, a password the harness cannot use, a handshake that
  never completes. All of those are defects, in the harness or in the code, and
  they are loud.
* **Ran** — the dialect was exercised for real.
"""

from __future__ import annotations

import importlib
import os
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from sqlalchemy.engine import URL

COMPOSE = Path(__file__).resolve().parent.parent / "docker-compose.test.yml"

#: Every container publishes on the loopback interface. Named rather than
#: inlined because a remote harness would change this one value.
HOST = "127.0.0.1"


class Unavailable(Exception):
    """Why an endpoint cannot be exercised here. A skip, never a failure."""


@dataclass(frozen=True)
class Endpoint:
    """One endpoint dialect, and everything needed to decide whether to run it.

    ``dialect`` is SQLAlchemy's own backend name — the key
    :func:`localdata_mcp.dialects.backend_for` looks up — so a subclass added for
    this dialect is exercised by these tests without anything here changing.

    **``dialect`` is not an identity, and must never be used as one.** Several
    databases are reached through a dialect they did not write: TiDB and
    OceanBase speak MySQL's wire and have no dialect of their own, and
    YugabyteDB, Greenplum and OpenGauss are addressed as PostgreSQL. Keying
    anything per-endpoint on it means the second such endpoint silently reuses
    the first's container and reports a database as tested that was never
    reached. :attr:`name` is the identity; ``dialect`` says only which backend
    answers.
    """

    #: SQLAlchemy's backend name for this database. Not unique across endpoints.
    dialect: str
    #: The service in ``docker-compose.test.yml`` that provides it. Unique by
    #: construction, since compose services are.
    service: str
    #: The port the container listens on *inside* itself, which is the right-hand
    #: side of the compose port mapping. The published port is read from there.
    container_port: int
    #: The Python module the URL's driver needs, and the extra that installs it.
    driver: str
    extra: str
    #: Builds the URL from the service's own environment and published port.
    url: Callable[[dict[str, str], int], str]
    #: What :class:`localdata_mcp.dialects.Backend` should end up *called* for
    #: this endpoint — the engine actually answering, which is only sometimes
    #: what the dialect is named. ``None`` means the two agree, the ordinary
    #: case; YugabyteDB is reached as ``postgresql`` and is not PostgreSQL.
    #:
    #: Stated rather than derived, and it is the assertion that would have
    #: caught issue #45 the day YugabyteDB landed: keyed by dialect it was
    #: handed ``Backend(name="postgresql")``, and every test still passed,
    #: because nothing anywhere asked the backend what it thought it was.
    engine: str | None = None
    #: How long to keep trying the handshake once the port is open. A container
    #: publishes its port before it finishes initialising, and Oracle takes a
    #: minute and a half to come up; below this the answer is "still starting",
    #: above it something is actually wrong.
    warmup: float = 30.0
    #: Anything else that must hold before this dialect can be reached, raising
    #: :class:`Unavailable` when it does not. pyodbc needs a *system* ODBC
    #: driver, which importing it says nothing about.
    precondition: Callable[[], None] | None = None
    #: Ways of reaching this same database *other* than the credentialed URL
    #: above — see :class:`AuthMode`. Empty for an endpoint whose one mode is
    #: the only one it has, which is most of them.
    auth: tuple[AuthMode, ...] = ()

    @property
    def name(self) -> str:
        """What this endpoint is called, uniquely, in a cache key or a test id.

        Derived from the service rather than stored, so it cannot drift from it
        and so adding an endpoint cannot forget it. The prefix every service
        carries is dropped because it is the same on all of them and only makes
        the test ids harder to read.
        """
        return self.service.removeprefix("localdata-test-")

    @property
    def engine_name(self) -> str:
        """What the backend answering for this endpoint should call itself.

        The dialect's own name unless this endpoint borrowed it from another
        engine, in which case :attr:`engine` says whose it really is.
        """
        return self.engine or self.dialect


@dataclass(frozen=True)
class Reached:
    """How one authentication mode gets to a database: a URL, and a context.

    A mode is not always expressible as a URL. ``PGPASSWORD`` and ``PGPASSFILE``
    are read by libpq out of the process environment, and the URL that goes with
    them carries **no** password at all — so a builder that could only return a
    string would have nothing to say about half the modes here. Both halves come
    back together because they are one answer to one question, and reading them
    apart would let a mode set an environment its URL does not need or need one
    it does not set.
    """

    #: The URL to attach, credentials and all — or credentials absent, when the
    #: point of the mode is that they are somewhere else.
    url: str
    #: Process environment this mode needs while the connection is made. Empty
    #: for a mode that says everything in its URL.
    environ: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthMode:
    """A second way of *reaching* an endpoint that is already covered.

    Every endpoint above is addressed one way: a username and a password in the
    URL, in plaintext, over TCP to the loopback interface. That is one code path
    out of several a real caller uses, and the others had never run — which is
    what task 23 is about.

    **A mode varies the addressing, never the database.** It is not a new
    :class:`Endpoint`, and deliberately so: an ``Endpoint`` is identified by its
    compose service, and two entries sharing one would collide in the probe
    cache and run one container's suite while reporting the other's name — the
    defect issue #44 records. A mode hangs off the endpoint it varies, so the
    identity stays derived and stays unique.

    The credentialed mode is **not** listed here. It is the endpoint's own
    :attr:`Endpoint.url`, it stays exactly as it was, and every mode below is in
    addition to it. Weakening the ordinary path to add an unusual one would
    trade coverage rather than add it.
    """

    #: Short id for the mode, unique within its endpoint. Appears in the test id
    #: as ``postgres[env-password]``, so it says what ran rather than which
    #: number it was.
    mode: str
    #: Everything needed to reach the endpoint this way, built from the compose
    #: service's own environment, its published port, and a scratch directory
    #: for the credential files some modes keep outside the URL.
    reach: Callable[[dict[str, str], int, Path], Reached]
    #: The compose service configured for this mode, when the mode and the
    #: credentialed one cannot both hold on one server — an authentication
    #: method is a property of the server, not of the connection. ``None`` means
    #: this mode reuses the endpoint's own container, which is the cheaper and
    #: commoner case.
    service: str | None = None
    #: The port that service listens on inside itself. Only meaningful with
    #: :attr:`service`; ``None`` reuses the endpoint's.
    container_port: int | None = None
    #: Anything that must hold before this mode can be tried at all, raising
    #: :class:`Unavailable` when it does not — a driver feature, a file on the
    #: machine, an ODBC entry.
    precondition: Callable[[], None] | None = None
    #: Overrides the endpoint's warmup where a variant container starts at a
    #: different speed. ``None`` keeps the endpoint's.
    warmup: float | None = None


@dataclass(frozen=True)
class Target:
    """An endpoint reached one particular way — what a test actually runs against.

    The pair, rather than either half, because everything the harness does with
    an endpoint has to know which mode is meant: which service to read out of
    the compose file, which port it publishes, what to call the test, and which
    entry of the probe cache is this one's.

    ``auth is None`` is the endpoint's own credentialed mode, and it is the
    common case: every endpoint has one and most have nothing else.
    """

    endpoint: Endpoint
    auth: AuthMode | None = None

    @property
    def name(self) -> str:
        """Unique across every target, and readable as a test id.

        The endpoint's own name where the mode is the credentialed one, so the
        ids that existed before this axis are unchanged and a failure that was
        reported as ``postgres`` still is.
        """
        if self.auth is None:
            return self.endpoint.name
        return f"{self.endpoint.name}[{self.auth.mode}]"

    @property
    def service(self) -> str:
        """The compose service to read, which the mode may redirect."""
        if self.auth is not None and self.auth.service is not None:
            return self.auth.service
        return self.endpoint.service

    @property
    def container_port(self) -> int:
        """The port that service listens on inside the container."""
        if self.auth is not None and self.auth.container_port is not None:
            return self.auth.container_port
        return self.endpoint.container_port

    @property
    def warmup(self) -> float:
        if self.auth is not None and self.auth.warmup is not None:
            return self.auth.warmup
        return self.endpoint.warmup

    def build(self, environment: dict[str, str], port: int, scratch: Path) -> Reached:
        """How this target is reached, whichever half of the axis answers.

        The endpoint's own builder returns a bare URL and needs no scratch
        directory, so it is lifted into a :class:`Reached` here rather than every
        one of the sixteen builders being rewritten to return one. A mode that
        needed nothing but a URL would look identical from outside.
        """
        if self.auth is None:
            return Reached(url=self.endpoint.url(environment, port))
        return self.auth.reach(environment, port, scratch)


#: Driver libraries to fall back on when the driver manager has nothing
#: registered. FreeTDS is the one that comes from Homebrew and apt, and it
#: speaks TDS to SQL Server perfectly well. Named by path on purpose: putting it
#: in ``odbcinst.ini`` would be a change to the machine, and a test harness has
#: no business making one.
_DRIVER_LIBRARIES = (
    "/usr/local/lib/libtdsodbc.so",
    "/opt/homebrew/lib/libtdsodbc.so",
    "/usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so",
    "/usr/lib/aarch64-linux-gnu/odbc/libtdsodbc.so",
)


def _odbc_driver() -> str:
    """An ODBC driver that can reach SQL Server, named the way pyodbc wants it.

    pyodbc imports perfectly well with no drivers registered at all, so the
    import check says nothing about whether a connection can be made. This asks
    the driver manager what it has, and where it has nothing, looks for a driver
    library on disk — pyodbc takes a path in place of a name.
    """
    import pyodbc

    for candidate in pyodbc.drivers():
        if "SQL Server" in candidate:
            return candidate
    for library in _DRIVER_LIBRARIES:
        if Path(library).exists():
            return library
    raise Unavailable(
        "pyodbc is installed but there is no ODBC driver for SQL Server — none "
        "registered with the driver manager, and no FreeTDS library where one is "
        "usually found. On macOS: brew install freetds."
    )


def _url(
    drivername: str,
    *,
    username: str | None,
    password: str | None,
    port: int | None,
    database: str | None = None,
    query: dict[str, str] | None = None,
    host: str = HOST,
) -> str:
    """One endpoint URL, assembled from parts rather than formatted into text.

    **Every builder below goes through here, and none of them may interpolate a
    credential.** A URL is parsed, not concatenated: a password holding ``@``,
    ``:``, ``/``, ``?`` or ``#`` is re-read as structure, and an ``@`` in
    particular turns everything after it into a host — so the harness would
    connect somewhere nobody configured, or fail naming a host nobody wrote.

    ``URL.create`` takes each part as a **value** and renders whatever escaping
    that part needs. This is the same defect, and the same remedy, as
    ``Backend.open_file`` in :mod:`localdata_mcp.dialects`, whose docstring
    records a path re-read as syntax. The Postgres container carries a password
    full of delimiters so that this is proved rather than assumed.

    ``password=None`` means **no password at all**, which is not the same as an
    empty one: ``URL.create`` omits the ``:`` entirely, and that is the form a
    trust-authenticated server expects. CockroachDB in insecure mode is the one
    endpoint here reached that way. The difference is rendered, measured:
    ``None`` gives ``user@host`` and ``""`` gives ``user:@host``, and both parse
    back to what they were. ClickHouse's empty-password mode is the one place
    the second form is exercised.

    ``username=None`` is a third shape, and the modes are what need it: an
    option file may carry the user as well as the password, leaving the URL with
    no credential of any kind.

    ``host`` defaults to the loopback address every container publishes on and is
    named only by the ODBC DSN mode, where the host position holds **the name of
    a data source** rather than an address, and ``port`` is then absent because
    the file supplies it. It is a parameter rather than a second helper so that
    the one rule this function exists for — no builder interpolates — still has
    exactly one place it is enforced.
    """
    return URL.create(
        drivername,
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        query=query or {},
    ).render_as_string(hide_password=False)


def _postgres(env: dict[str, str], port: int) -> str:
    return _url(
        "postgresql+psycopg",
        username=env["POSTGRES_USER"],
        password=env["POSTGRES_PASSWORD"],
        port=port,
        database=env["POSTGRES_DB"],
    )


def _mysql(env: dict[str, str], port: int) -> str:
    return _url(
        "mysql+pymysql",
        username=env["MYSQL_USER"],
        password=env["MYSQL_PASSWORD"],
        port=port,
        database=env["MYSQL_DATABASE"],
    )


def _mariadb(env: dict[str, str], port: int) -> str:
    """MariaDB is addressed as MariaDB, not as MySQL.

    PyMySQL speaks to both, but the URL's scheme is what decides which dialect
    SQLAlchemy loads and therefore which backend answers for it — and the two
    have diverged enough that being told which one is on the other end is worth
    more than sharing a name.
    """
    return _url(
        "mariadb+pymysql",
        username=env["MARIADB_USER"],
        password=env["MARIADB_PASSWORD"],
        port=port,
        database=env["MARIADB_DATABASE"],
    )


def _mssql(env: dict[str, str], port: int) -> str:
    """SQL Server, through whichever ODBC driver is installed.

    ``master`` because the image creates no other database and there is no
    environment variable that would ask it to; the tests name their tables
    uniquely and drop what they made, so a shared database costs nothing. The
    certificate is self-signed, hence ``TrustServerCertificate`` — Microsoft's
    driver 18 encrypts by default and would otherwise refuse the container
    outright. FreeTDS ignores the setting, which is harmless.

    The driver name goes in as a query *value*: it holds spaces, and on some
    machines it is an absolute path, neither of which may reach the URL as text.
    """
    return _url(
        "mssql+pyodbc",
        username="sa",
        password=env["MSSQL_SA_PASSWORD"],
        port=port,
        database="master",
        query={"driver": _odbc_driver(), "TrustServerCertificate": "yes"},
    )


def _oracle(env: dict[str, str], port: int) -> str:
    """Oracle Free, reached through python-oracledb in its thin mode.

    ``FREEPDB1`` is the pluggable database the image creates and the one
    ``APP_USER`` is created in; the container's own root service would need
    privileged credentials and holds nothing a test wants. It is named as a
    query parameter rather than as the database, which is how the thin mode
    distinguishes a service from a SID.
    """
    return _url(
        "oracle+oracledb",
        username=env["APP_USER"],
        password=env["APP_USER_PASSWORD"],
        port=port,
        query={"service_name": "FREEPDB1"},
    )


def _clickhouse(env: dict[str, str], port: int) -> str:
    """ClickHouse over HTTP, through the dialect inside ``clickhouse-connect``.

    ``clickhousedb`` is the dialect's registered name and the key
    :func:`localdata_mcp.dialects.backend_for` looks up; the third-party
    ``clickhouse-sqlalchemy`` registers ``clickhouse`` and is a different
    project. The port is the HTTP one — this dialect does not speak the native
    protocol on 9000, and only the port it speaks on is published.
    """
    return _url(
        "clickhousedb",
        username=env["CLICKHOUSE_USER"],
        password=env["CLICKHOUSE_PASSWORD"],
        port=port,
        database=env["CLICKHOUSE_DB"],
    )


def _cockroachdb(env: dict[str, str], port: int) -> str:
    """CockroachDB, on the PostgreSQL wire but as its own dialect.

    ``cockroachdb+psycopg`` rather than ``postgresql+psycopg``: psycopg speaks to
    it either way, but the scheme is what decides which dialect SQLAlchemy loads
    and therefore which backend answers — the same reason MariaDB is addressed as
    MariaDB. Addressing it as PostgreSQL would test PostgreSQL's answers against
    CockroachDB's behaviour, which is precisely the question this endpoint exists
    to ask.

    ``root`` with **no password**, because the container runs ``--insecure``.
    ``defaultdb`` is the database a fresh single node creates.
    """
    return _url(
        "cockroachdb+psycopg",
        username="root",
        password=None,
        port=port,
        database="defaultdb",
    )


def _yugabytedb(env: dict[str, str], port: int) -> str:
    """YugabyteDB's YSQL layer, addressed as PostgreSQL — and that is a choice.

    The other two PostgreSQL-wire endpoints are addressed as themselves, so this
    one breaking the pattern needs its reason stated. YugabyteDB *is* eligible
    under the adapter rule: ``sqlalchemy-yugabytedb`` exists and is Apache-2.0,
    Yugabyte's own. What that adapter cannot do is be reached from here.

    It registers **psycopg2 drivers only**, and hard-requires
    ``psycopg2-yugabytedb`` — a fork of psycopg2 pinned at 2.9.3 that publishes
    wheels for macOS arm64 and nothing else, so every Linux and Windows user
    compiles it against libpq. Adopting it would put a second PostgreSQL driver
    family in this project for one database, next to the psycopg 3 the
    ``postgres`` extra already carries.

    Against that, the dialect itself is 81 lines and none of what it adds is
    something this server's seam asks about: it narrows the isolation-level
    lookup, and overrides ``initialize`` in a way that calls
    ``super(PGDialect, self)`` — *skipping* PGDialect's own initialisation
    rather than extending it. Plain ``postgresql+psycopg`` connects and reads
    the version correctly, measured: ``PostgreSQL 15.12-YB-2.25.2.0-b0`` parses
    to ``(15, 12)``, where CockroachDB's banner could not be parsed at all and
    is why *that* endpoint genuinely needs its own dialect.

    **The driver's distribution is a quality judgement, and it decides only the
    addressing, never the eligibility** — those are different tests, and
    answering one with the other is the mistake ClickHouse's removal and
    restoration already recorded (``CONSTRAINTS.md`` §11).

    The cost is real and is the finding: reached as ``postgresql``, this
    database gets ``Backend(name="postgresql")``, so a refusal names PostgreSQL
    to someone who opened YugabyteDB. That is issue #45 — a dialect name is not
    an identity — landing on a second item.

    ``yugabyte`` with **no password**: a fresh cluster authenticates by trust,
    the same URL shape CockroachDB's insecure mode needs.
    """
    return _url(
        "postgresql+psycopg",
        username="yugabyte",
        password=None,
        port=port,
        database="yugabyte",
    )


def _trino(env: dict[str, str], port: int) -> str:
    """Trino, which is a query engine and therefore names a *catalog*, not a database.

    Trino stores nothing of its own. Every table it can see belongs to a
    catalog — a configured connector pointing at some other system — so the
    thing a URL has to name here is which catalog and which schema inside it,
    and ``memory/default`` is both. The ``memory`` connector is the one writable
    catalog the stock image ships that needs nothing behind it; ``tpch`` and
    ``tpcds`` are read-only generators and ``jmx`` exposes the JVM.

    That makes this the first endpoint whose database component is **two path
    segments**, which ``URL.create`` renders as given — a catalog and a schema
    are a path, not a name to be escaped into one.

    ``test`` with **no password**, and any username would do: with no
    authenticator configured Trino takes whoever the client says it is. This is
    the third no-auth endpoint here, after CockroachDB's ``--insecure`` and
    YugabyteDB's trust.
    """
    return _url(
        "trino",
        username="test",
        password=None,
        port=port,
        database="memory/default",
    )


def _monetdb(env: dict[str, str], port: int) -> str:
    """MonetDB, a column store, through the dialect its own vendor publishes.

    ``monetdb`` three times over, and they are three different things: the
    dialect ``sqlalchemy-monetdb`` registers, the user the image sets the admin
    password on, and the database ``MDB_CREATE_DBS`` defaults to creating. Only
    the last is read from the environment, because only the last is something
    the compose file could reasonably change.

    The password is the one the entrypoint demands — the image refuses to start
    without ``MDB_DB_ADMIN_PASS`` rather than coming up with an unreachable
    database — so it is read from there and never restated here.
    """
    return _url(
        "monetdb",
        username="monetdb",
        password=env["MDB_DB_ADMIN_PASS"],
        port=port,
        database="monetdb",
    )


def _cratedb(env: dict[str, str], port: int) -> str:
    """CrateDB over its own HTTP endpoint, not over the PostgreSQL wire it also speaks.

    The server publishes both: 4200 carries the HTTP protocol Crate.io's own
    driver and dialect speak, and 5432 is a PostgreSQL compatibility layer. Only
    the first is published by the compose entry, deliberately — addressing the
    compatibility layer would load PostgreSQL's dialect and test PostgreSQL's
    answers against CrateDB's behaviour, the mistake CockroachDB's entry exists
    not to make.

    ``crate`` with **no password**: a fresh node has no users configured and its
    HTTP endpoint accepts any client. This is the fifth no-auth endpoint here,
    after CockroachDB, YugabyteDB, Trino and MonetDB — which is why task 23's
    first item keeps being covered by accident.

    No database component at all. CrateDB has no databases to choose between;
    tables live in schemas, and an unqualified name lands in ``doc``.
    """
    return _url(
        "crate",
        username="crate",
        password=None,
        port=port,
    )


def _firebird(env: dict[str, str], port: int) -> str:
    """Firebird, whose database component is a **path on the server's filesystem**.

    Every other endpoint here names a database the server looks up in a catalogue.
    Firebird opens a file: the container's entrypoint resolves
    ``FIREBIRD_DATABASE`` against ``FIREBIRD_DATA`` and creates it there, so what
    a URL has to carry is that whole absolute path, rendered by ``URL.create`` as
    a second slash after the port. The directory is the image's own and is not
    something the compose file sets, so it is the one part of this URL not read
    from the environment.

    The dialect is ``firebirdsql``, which is the name of the **driver** — the pure
    Python one, from ``sqlalchemy-firebirdsql``. The mature ``sqlalchemy-firebird``
    registers ``firebird`` and rides on ``libfbclient``, a native library this
    machine has no way to obtain: measured, ``firebird-driver`` raises ``The
    location of Firebird Client Library could not be determined.`` Adapter
    *quality* would have argued for the older dialect; adapter *reach* decided it,
    and under standing instruction 10 that is a question about addressing rather
    than eligibility — the same distinction YugabyteDB's entry above turns on.

    Hence ``engine="firebird"`` on the entry below. This is the first endpoint
    whose dialect is named after neither the engine nor another engine, and the
    identity assertion is what keeps a refusal from naming a Python package to
    someone who opened a database.
    """
    return _url(
        "firebirdsql",
        username=env["FIREBIRD_USER"],
        password=env["FIREBIRD_PASSWORD"],
        port=port,
        database=f"/var/lib/firebird/data/{env['FIREBIRD_DATABASE']}",
    )


def _opengauss(env: dict[str, str], port: int) -> str:
    """openGauss, on PostgreSQL's wire and reached through its **own** dialect.

    The third engine here speaking that wire, and the first that cannot be
    addressed as PostgreSQL at all — which is the opposite of what its lineage
    suggests and was settled by measurement rather than by expectation.

    psycopg connects and authenticates perfectly well. What fails is SQLAlchemy's
    own ``PGDialect`` immediately afterwards, because openGauss answers
    ``version()`` with ``(openGauss 7.0.0-RC3 build 01b7e318) compiled at …`` — a
    banner that does not begin with ``PostgreSQL x.y``. ``PGDialect`` asserts on
    that pattern while *initialising the connection*, so
    ``AssertionError: Could not determine version from string`` arrives before any
    statement runs. YugabyteDB's banner parses (``PostgreSQL 15.12-YB-…``) and is
    why that endpoint could take the cheap route; this one has no cheap route.

    So the cost YugabyteDB's entry above declined is paid here, and it is paid
    because there is no alternative rather than because it got cheaper:
    ``opengauss-sqlalchemy`` — the openGauss project's own, MIT — registers
    psycopg2 drivers only, making this the second PostgreSQL driver family in the
    project. It is a milder cost than the one that decided YugabyteDB:
    ``psycopg2-binary`` publishes ordinary wheels, where ``psycopg2-yugabytedb``
    was a fork pinned at 2.9.3 with wheels for one platform.

    No ``engine=`` is needed, and that is worth stating because the lineage
    predicts otherwise. The dialect registers under **its own** name, so the
    backend answering already calls itself ``opengauss`` and there is no impostor
    to resolve — the identity problem #45 describes simply does not arise when a
    database ships its own dialect.

    The password is read from the environment like every other, and here it
    carries an ``@`` for a reason that is not stylistic: the image's entrypoint
    enforces a complexity rule that *requires* one of ``#?!@$%^&*-``, so every
    password this database will accept contains a URL delimiter. An endpoint that
    formatted credentials into a URL could not reach openGauss at all.
    """
    return _url(
        "opengauss+psycopg2",
        username=env["GS_USERNAME"],
        password=env["GS_PASSWORD"],
        port=port,
        database=env["GS_DB"],
    )


def _ydb(env: dict[str, str], port: int) -> str:
    """YDB, anonymous, on the database the image creates at ``/local``.

    Two things here are not what the other builders do, and both were measured.

    **No credentials at all.** The image configures no authentication, so this
    passes ``password=None`` — which ``_url`` renders by omitting the ``:``
    entirely — and a username that is never checked. CockroachDB's insecure mode
    is the only other endpoint reached this way.

    **The port is not offset.** Every other service here publishes on a port well
    away from its engine's default so that a locally-installed copy cannot be
    reached by accident. YDB's client resolves the endpoint it is given into the
    cluster's *own* advertised node addresses and connects to those, so the
    published port has to equal the advertised one or nothing after the handshake
    can be reached. The compose entry carries the other half of this.

    The dialect is ``yql+ydb``: ``yql`` is what the dialect calls itself — after
    the query language — and ``ydb`` is the driver entry point. Both spellings
    resolve to the same class, and ``yql`` alone would too, but naming the driver
    keeps this URL saying which of the dialect's two drivers is meant, since the
    package also registers an async one. Hence ``engine="ydb"`` on the entry
    below: the backend must call itself after the database, not after its SQL.
    """
    return _url(
        "yql+ydb",
        username="root",
        password=None,
        port=port,
        database="local",
    )


def _databend(env: dict[str, str], port: int) -> str:
    """Databend over its own HTTP query handler, with a credential.

    The dialect and the driver are both called ``databend`` and so is the engine,
    which after YugabyteDB, Firebird and YDB is worth stating rather than passing
    over: this is the first endpoint in a while that needs no ``engine=``
    override, because the database ships its own dialect and named it after
    itself.

    ``default`` is the database the image creates, and ``sslmode=disable`` is
    required rather than tidy — the driver defaults to TLS and this container
    serves plaintext, so without it the handshake fails.

    The credentials are the container's, read from its environment like every
    other builder's. What is specific here is that the image needs **both**
    variables to make a user at all: given only one, its entrypoint writes a
    passwordless ``root`` instead and the user this URL names does not exist. So
    a missing password is not a weaker endpoint, it is a *different* one, and
    that is why the compose entry sets the pair together.
    """
    return _url(
        "databend",
        username=env["QUERY_DEFAULT_USER"],
        password=env["QUERY_DEFAULT_PASSWORD"],
        port=port,
        database="default",
        query={"sslmode": "disable"},
    )


def _exasol(env: dict[str, str], port: int) -> str:
    """Exasol over its WebSocket protocol, as a user its healthcheck made.

    Three things here are unlike the builders above, and each was measured.

    **The URL's database component is a schema.** The dialect's
    ``create_connect_args`` calls ``translate_connect_args(database="schema")``,
    so what every other builder spells as a database is what Exasol opens as the
    current schema — and a fresh cluster has none. Naming one that does not exist
    fails at connect, not at the first statement.

    **The credential is provisioned rather than configured.** This image reads no
    password variable at all: ``exadt init-sc --sys-passwd`` wants a hash, and
    passing one — hash or cleartext — stops the database from starting, which
    also rules out its ``--init-sql``. So the compose entry declares a user and a
    password, its healthcheck creates them along with the schema, and this reads
    them back out of that same environment. The variables are ours rather than
    the image's, and they are read here for the reason every other builder reads
    its own: a credential stated twice is a credential that can disagree.

    **``SSLCertificate=SSL_VERIFY_NONE`` is required, not tidy.** The protocol is
    TLS-only and the cluster serves a certificate it signed itself, so
    ``pyexasol`` — which since 1.0.0 verifies by default — refuses the handshake
    with ``CERTIFICATE_VERIFY_FAILED: self-signed certificate``. The dialect maps
    this query parameter onto its ``certificate_validation`` argument; the other
    accepted spelling, ``FINGERPRINT``, pins the certificate instead and would
    make the URL depend on a value the container regenerates. Databend's
    ``sslmode=disable`` is the same shape of concession made for the opposite
    reason — there the server offers no TLS, here it offers nothing else.
    """
    return _url(
        "exa+websocket",
        username=env["EXASOL_APP_USER"],
        password=env["EXASOL_APP_PASSWORD"],
        port=port,
        database="localdata",
        query={"SSLCertificate": "SSL_VERIFY_NONE"},
    )


# ---------------------------------------------------------------------------
# Authentication modes — the same databases, reached other ways
# ---------------------------------------------------------------------------


def _postgres_trust(env: dict[str, str], port: int, scratch: Path) -> Reached:
    """A server that authenticates nobody, reached with a username and nothing else.

    Five endpoints here already pass ``password=None``, and none of them is this
    case: CockroachDB, YugabyteDB, Trino, CrateDB and YDB have **no
    authentication to configure**, so a passwordless URL is the only URL they
    have. PostgreSQL has authentication and is told not to use it, which is the
    posture a developer's own machine is usually in and the one a caller most
    easily reaches by accident.

    The user is real and the database is real; only the check is absent. That
    makes it the one mode where a wrong password cannot be sent — there is no
    password for it to be wrong against — which is why the failed-open test skips
    here rather than being weakened to accommodate it.
    """
    return Reached(
        url=_url(
            "postgresql+psycopg",
            username=env["POSTGRES_USER"],
            password=None,
            port=port,
            database=env["POSTGRES_DB"],
        )
    )


def _postgres_pgpassword(env: dict[str, str], port: int, scratch: Path) -> Reached:
    """The password in the environment, and the URL carrying none.

    An everyday case rather than an exotic one: a caller who has exported
    ``PGPASSWORD`` — or inherited it from a shell profile, or a CI secret — and
    attaches a URL that names only the user. Everything in this server's path
    then has to keep working while the credential is somewhere it never sees.

    libpq reads the variable itself, beneath psycopg and beneath SQLAlchemy, so
    nothing here passes it on. **That is the point**: the URL is genuinely
    passwordless, and the bare URL with no variable set is refused —
    ``fe_sendauth: no password supplied``, measured — which is what makes this a
    test of the environment rather than of a server that was not asking.
    """
    return Reached(
        url=_url(
            "postgresql+psycopg",
            username=env["POSTGRES_USER"],
            password=None,
            port=port,
            database=env["POSTGRES_DB"],
        ),
        environ={"PGPASSWORD": env["POSTGRES_PASSWORD"]},
    )


def _pgpass_field(value: str) -> str:
    """One ``.pgpass`` field, with the two characters libpq treats as syntax escaped.

    Its rule, and only its rule: a colon separates fields and a backslash escapes
    the next character, so both are written with a leading backslash and nothing
    else is special. The backslash goes first — doing it second would escape the
    backslashes the colon rule had just added.
    """
    return value.replace("\\", "\\\\").replace(":", "\\:")


def _postgres_pgpassfile(env: dict[str, str], port: int, scratch: Path) -> Reached:
    """The password in a file libpq reads, which this endpoint's password breaks.

    ``.pgpass`` is five colon-separated fields — host, port, database, user,
    password — so a password **containing a colon** has to escape it, and this
    endpoint's password is ``p@ss:w/rd?x#y``. Unescaped, libpq reads the
    password as ``p@ss`` and takes ``w/rd?x#y`` as a sixth field it ignores; the
    connection then fails with ``password authentication failed``, which points
    at the credential rather than at the file that mangled it. Measured both
    ways round.

    That is the same defect as a password formatted into a URL and re-read as
    syntax — the one ``_url`` above exists to prevent — arriving through a
    different file format. A harness whose password held no delimiters would
    pass either way and prove nothing.

    **Every** field is escaped, not only the password, and that is not
    defensiveness: the separator is the same one in all five, so a database or a
    user whose name holds a colon splits the line exactly as badly. Escaping only
    the field whose value happened to be hostile would be fixing the instance
    rather than the format.

    The mode is also refused for a reason that has nothing to do with the
    password: libpq **ignores the file entirely** if it is group- or
    world-readable, warns on stderr, and then reports ``fe_sendauth: no password
    supplied`` to the client. Hence the explicit ``chmod`` — measured, at 0644
    this mode does not connect.
    """
    passfile = scratch / "pgpass"
    passfile.write_text(
        ":".join(
            _pgpass_field(field)
            for field in (
                HOST,
                str(port),
                env["POSTGRES_DB"],
                env["POSTGRES_USER"],
                env["POSTGRES_PASSWORD"],
            )
        )
        + "\n"
    )
    passfile.chmod(0o600)
    return Reached(
        url=_url(
            "postgresql+psycopg",
            username=env["POSTGRES_USER"],
            password=None,
            port=port,
            database=env["POSTGRES_DB"],
        ),
        environ={"PGPASSFILE": str(passfile)},
    )


#: Where the compose CA service leaves the material a *client* needs. The server
#: half lives in a named volume where this process cannot read it and does not
#: need to.
TLS = COMPOSE.parent / "tests" / "tls"


def _tls_material(*names: str) -> None:
    """Refuse the TLS modes with a route out when the CA has not run.

    A skip rather than a failure: an absent certificate means the compose
    service that makes them has not been brought up, which is the same kind of
    absence as a container that is not running.
    """
    missing = [name for name in names if not (TLS / name).is_file()]
    if missing:
        raise Unavailable(
            f"{', '.join(missing)} missing from {TLS}. They are made by the CA "
            f"service, which runs as a dependency of the TLS endpoint: "
            f"docker compose -f {COMPOSE.name} up -d localdata-test-postgres-tls"
        )


def _postgres_tls_verify_full(env: dict[str, str], port: int, scratch: Path) -> Reached:
    """TLS actually verified, which no other endpoint here does.

    ``sslmode=verify-full`` is the only mode that checks both halves: that the
    certificate chains to a CA the client trusts, **and** that the name on it is
    the name the client asked for. Everything weaker is decoration —
    ``sslmode=require`` encrypts against an eavesdropper and not against the
    server being someone else, which is the property people believe they are
    getting.

    The contrast worth naming is inside this harness: the SQL Server endpoint
    passes ``TrustServerCertificate=yes``, which is the *opposite* of this, and
    until this mode existed no endpoint here verified anything at all.

    Both refusals were measured, and they are what make this a test rather than a
    connection: ``sslmode=disable`` against this server is refused outright —
    there is no plain ``host`` line in its ``pg_hba.conf``, so TLS is not an
    option it offers — and ``verify-full`` *without* ``sslrootcert`` fails
    looking for ``~/.postgresql/root.crt``, so the CA named here is genuinely the
    one doing the verifying.

    The certificate's SAN carries ``IP:127.0.0.1`` for the same reason: the URL
    asks for an address, so an address is what ``verify-full`` compares.
    """
    _tls_material("ca.crt")
    return Reached(
        url=_url(
            "postgresql+psycopg",
            username=env["POSTGRES_USER"],
            password=env["POSTGRES_PASSWORD"],
            port=port,
            database=env["POSTGRES_DB"],
            query={"sslmode": "verify-full", "sslrootcert": str(TLS / "ca.crt")},
        )
    )


def _postgres_client_cert(env: dict[str, str], port: int, scratch: Path) -> Reached:
    """A certificate instead of a password, and the username it may claim.

    The first mode here that sends **no** credential at all: PostgreSQL's
    ``cert`` method takes the common name out of the client certificate and logs
    that role in. So the username in the URL is not something the client
    asserts, it is something the certificate has to agree with — measured, the
    same certificate offered as ``tlsuser`` falls through to the password line
    and is refused with ``fe_sendauth: no password supplied``.

    ``certuser`` is therefore read from neither the compose environment nor this
    file but from the certificate's subject, which is where it is decided. It is
    named in the CA service that makes the certificate and in the ``pg_hba.conf``
    that same service writes, and the role is created by a ``.sql`` the image
    runs — three places, one name, and the connection is what proves they agree.
    """
    _tls_material("ca.crt", "client.crt", "client.key")
    return Reached(
        url=_url(
            "postgresql+psycopg",
            username="certuser",
            password=None,
            port=port,
            database=env["POSTGRES_DB"],
            query={
                "sslmode": "verify-full",
                "sslrootcert": str(TLS / "ca.crt"),
                "sslcert": str(TLS / "client.crt"),
                "sslkey": str(TLS / "client.key"),
            },
        )
    )


#: The one compose service that is not a database and not a one-shot. Named here
#: because the Kerberos mode has to read *two* services — the endpoint it
#: authenticates to, and the realm it authenticates against — which is the only
#: place in this module a builder looks outside its own service.
KDC = "localdata-test-kdc"


def _other_service(name: str, container_port: int) -> tuple[dict[str, str], int]:
    """Another service's environment and published port, by name.

    The same reading every builder gets for its own service, reached through the
    same helpers so a restated port cannot creep in here either.
    """
    services = _services()
    try:
        service = services[name]
    except KeyError:
        raise Unavailable(
            f"{COMPOSE.name} has no service {name!r}, which this mode needs"
        ) from None
    for mapping in service.get("ports", ()):
        published, _, inside = str(mapping).partition(":")
        # A published port may carry a protocol suffix — `1088:88/udp` — which
        # is part of the mapping and not of the number.
        if inside.partition("/")[0] == str(container_port):
            return _environment(service), int(published)
    raise Unavailable(
        f"{name} publishes {service.get('ports')}, none of which maps to "
        f"{container_port} inside the container"
    )


def _postgres_kerberos(env: dict[str, str], port: int, scratch: Path) -> Reached:
    """A ticket from a third party, and no credential in the connection at all.

    The last of the six modes and the only one where the client authenticates
    to something that is **not** the database: it gets a ticket from the KDC
    first and presents that. The URL carries a username and nothing else — the
    proof of it is in a credential cache this function fills.

    Two measurements shaped it, and both were surprises worth writing down.

    **The service principal names an address, not a host.** Asked for
    ``localhost``, the client library canonicalises the name through DNS before
    building the principal, lands on this machine's Tailscale domain, derives a
    realm from it and asks for a **cross-realm** ticket:
    ``Server krbtgt/<TAILNET>.TS.NET@LOCALDATA.TEST not found in Kerberos
    database``. That failure is about the DNS suffix of the machine the suite
    runs on, which no harness should depend on. A literal address is not
    canonicalised — the principal requested is exactly ``postgres/127.0.0.1`` —
    so that is what the KDC issues and the keytab holds.

    **``include_realm=0`` in the server's rules is load-bearing.** Without it the
    database user is ``krbuser@LOCALDATA.TEST``, which is not a role, so the
    authentication succeeds and the login is refused immediately afterwards —
    the confusing order of events.

    ``gssencmode`` is stated rather than left to the default because psycopg
    warns, in as many words, that the libpq it bundles may default it to
    ``disable``; a mode whose whole subject is GSSAPI should not be one build
    away from silently not using it.
    """
    if not shutil.which("kinit"):
        raise Unavailable(
            "Kerberos needs a `kinit` on PATH to obtain a ticket. macOS ships "
            "one at /usr/bin/kinit; on Debian it is in krb5-user."
        )
    realm_env, kdc_port = _other_service(KDC, 88)
    realm = realm_env["KRB5_REALM"]
    principal = f"{realm_env['KRB5_PRINCIPAL']}@{realm}"

    config = scratch / "krb5.conf"
    config.write_text(
        "[libdefaults]\n"
        f"  default_realm = {realm}\n"
        "  dns_lookup_realm = false\n"
        "  dns_lookup_kdc = false\n"
        "  rdns = false\n"
        "[realms]\n"
        f"  {realm} = {{\n"
        f"    kdc = {HOST}:{kdc_port}\n"
        "  }\n"
    )
    # Nothing in this file is a secret, and it is still 0600: everything written
    # into the scratch directory is private to this process, so that no file
    # there has to be judged individually — which is the judgement that gets a
    # credential left readable one day.
    config.chmod(0o600)
    ccache = scratch / "krb5cc"
    _kinit(principal, realm_env["KRB5_PASSWORD"], config, ccache, scratch)

    return Reached(
        url=_url(
            "postgresql+psycopg",
            username=realm_env["KRB5_PRINCIPAL"],
            password=None,
            port=port,
            database=env["POSTGRES_DB"],
            query={"krbsrvname": "postgres", "gssencmode": "prefer"},
        ),
        environ={"KRB5_CONFIG": str(config), "KRB5CCNAME": f"FILE:{ccache}"},
    )


def _kinit(
    principal: str, password: str, config: Path, ccache: Path, scratch: Path
) -> None:
    """Fill a credential cache, or say why the ticket could not be had.

    The password goes in through a file rather than an argument, because an
    argument is visible in the process list to everyone on the machine. Two
    spellings are tried because the two Kerberos families disagree: macOS ships
    Heimdal, whose ``kinit`` takes ``--password-file``, and MIT's reads standard
    input instead.
    """
    passfile = scratch / "krbpw"
    passfile.write_text(password)
    passfile.chmod(0o600)
    environment = {
        **os.environ,
        "KRB5_CONFIG": str(config),
        "KRB5CCNAME": f"FILE:{ccache}",
    }
    attempts = (
        (["kinit", f"--password-file={passfile}", principal], None),
        (["kinit", principal], password + "\n"),
    )
    complaints = []
    for command, stdin in attempts:
        done = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        if done.returncode == 0:
            return
        complaints.append(
            f"{' '.join(command[:2])}: {(done.stderr or done.stdout).strip()[:200]}"
        )
    raise Unavailable(
        "kinit could not get a ticket for "
        f"{principal}, so the Kerberos mode cannot run. " + "; ".join(complaints)
    )


def _mssql_dsn(env: dict[str, str], port: int, scratch: Path) -> Reached:
    """SQL Server named by a **DSN** rather than by an address.

    The oldest way to address an ODBC database and still the common one in
    places that have an ODBC estate: the URL names an entry in a file, and the
    file says where the server is and which driver reaches it. It is the one
    URL shape here with no host, no port and no database — ``mssql+pyodbc``
    reads the host position as the data-source name — so it exercises a branch
    of SQLAlchemy's own dialect that the ordinary form never touches.

    The file is written here rather than installed, and pointed at with
    ``ODBCINI``, because registering a DSN is a change to the machine and a test
    harness has no business making one. That is the same judgement the driver
    lookup above already records for ``odbcinst.ini``.

    The driver goes in by **path**, from the same lookup the TCP builder uses, so
    the two forms cannot disagree about which driver is meant — and on this
    machine there is no registered driver at all, which is exactly the case a
    hard-coded driver name would get wrong. ``TrustServerCertificate`` is carried
    over from the TCP builder for the same reason it is there: Microsoft's driver
    18 encrypts by default and would refuse a self-signed container. FreeTDS
    ignores it.

    A wrong password through this DSN is refused by the server, measured, so the
    mode is a real authentication and not a file that happens to open.
    """
    driver = _odbc_driver()
    dsn = "localdata-test-mssql-dsn"
    odbcini = scratch / "odbc.ini"
    odbcini.write_text(
        f"[{dsn}]\n"
        f"Driver = {driver}\n"
        f"Server = {HOST}\n"
        f"Port = {port}\n"
        f"Database = master\n"
        f"TrustServerCertificate = yes\n"
    )
    odbcini.chmod(0o600)
    return Reached(
        # `host=` is the DSN name: this is the one URL here whose host component
        # is not an address.
        url=_url(
            "mssql+pyodbc",
            username="sa",
            password=env["MSSQL_SA_PASSWORD"],
            port=None,
            host=dsn,
        ),
        environ={"ODBCINI": str(odbcini)},
    )


def _clickhouse_empty_password(
    env: dict[str, str], port: int, scratch: Path
) -> Reached:
    """A user whose password is the **empty string**, which is not the same as none.

    A third URL shape rather than a repeat of the second: an absent password
    renders ``user@host``, an empty one renders ``user:@host``, and both parse
    back to what they were. A server that accepts the first would not
    necessarily accept the second, and until now nothing here sent the second at
    a database that checks. It *is* checked — a wrong password on this same user
    is refused, measured — so this mode is not quietly a no-auth endpoint under
    another name.

    The obvious route to it does not work, and that is why the compose entry
    names a user rather than leaving the image alone. With ``CLICKHOUSE_USER``
    unset, the entrypoint leaves a ``default`` user restricted to ``::1`` and
    ``127.0.0.1`` — the **container's** loopback, not the host's — so from here
    it is refused, and refused as ``password is incorrect, or there is no user
    with such name``, which names the wrong cause entirely. Naming a user with
    an empty password is what gets ``<ip>::/0</ip>`` written alongside
    ``<password><![CDATA[]]></password>``.

    Its own container, because the entrypoint's answer to ``CLICKHOUSE_USER`` is
    ``<default remove="remove">`` — read out of the running container rather than
    assumed. One instance holds one of these users.

    An empty password is also the boundary of the redaction rule. It is a secret
    that was supplied, so it is rendered as ``***`` like any other; it is also
    the empty string, so "the password does not appear in the payload" is not a
    question that can be asked of it. The test says so rather than asserting
    something that is true of every string.
    """
    return Reached(
        url=_url(
            "clickhousedb",
            username=env["CLICKHOUSE_USER"],
            password=env["CLICKHOUSE_PASSWORD"],
            port=port,
            database=env["CLICKHOUSE_DB"],
        )
    )


def _mysql_option_file(env: dict[str, str], port: int, scratch: Path) -> Reached:
    """Both credentials in an option file, and a URL with neither.

    MySQL's ``~/.my.cnf`` is the oldest of these conventions and the one a
    long-lived installation is most likely to be leaning on. PyMySQL reads it
    when handed ``read_default_file``, so unlike ``PGPASSWORD`` the mode *is*
    expressible in the URL — as the **path to** the credentials rather than the
    credentials themselves.

    That makes this the only mode here whose URL carries no username either, and
    it is worth having one: ``URL.create`` renders an absent user by omitting the
    whole userinfo section, and nothing else in this harness exercises that.
    Measured, the same URL without the option file is refused — ``Access denied
    for user 'testuser'`` — so the file is doing the work.

    The path goes in as a query *value*, never as text: it is a temporary
    directory whose name this harness does not choose.
    """
    optionfile = scratch / "my.cnf"
    optionfile.write_text(
        f"[client]\nuser={env['MYSQL_USER']}\npassword={env['MYSQL_PASSWORD']}\n"
    )
    optionfile.chmod(0o600)
    return Reached(
        url=_url(
            "mysql+pymysql",
            username=None,
            password=None,
            port=port,
            database=env["MYSQL_DATABASE"],
            query={"read_default_file": str(optionfile)},
        )
    )


#: Every endpoint dialect this server is tested against, in the order they were
#: taken on. A dialect is here because it has a container; nothing about the
#: server enumerates dialects, so this list is a statement about *coverage*, not
#: about what is reachable.
ENDPOINTS = (
    Endpoint(
        dialect="postgresql",
        service="localdata-test-postgres",
        container_port=5432,
        driver="psycopg",
        extra="postgres",
        url=_postgres,
        # PostgreSQL carries three of the modes because it is the endpoint that
        # can: libpq is the richest credential-resolution path any driver here
        # has, and this container's password is the hostile one, so a mode that
        # mangles a credential is caught here rather than somewhere it would
        # look like the server's fault.
        auth=(
            AuthMode(
                mode="trust",
                reach=_postgres_trust,
                service="localdata-test-postgres-trust",
                container_port=5432,
            ),
            AuthMode(mode="env-password", reach=_postgres_pgpassword),
            AuthMode(mode="pgpass-file", reach=_postgres_pgpassfile),
            AuthMode(
                mode="tls-verify-full",
                reach=_postgres_tls_verify_full,
                service="localdata-test-postgres-tls",
                container_port=5432,
            ),
            AuthMode(
                mode="client-cert",
                reach=_postgres_client_cert,
                service="localdata-test-postgres-tls",
                container_port=5432,
            ),
            AuthMode(
                mode="kerberos",
                reach=_postgres_kerberos,
                service="localdata-test-postgres-krb",
                container_port=5432,
            ),
        ),
    ),
    Endpoint(
        dialect="mysql",
        service="localdata-test-mysql",
        container_port=3306,
        driver="pymysql",
        extra="mysql",
        url=_mysql,
        auth=(AuthMode(mode="option-file", reach=_mysql_option_file),),
    ),
    Endpoint(
        dialect="mariadb",
        service="localdata-test-mariadb",
        container_port=3306,
        driver="pymysql",
        extra="mysql",
        url=_mariadb,
    ),
    Endpoint(
        dialect="mssql",
        service="localdata-test-mssql",
        container_port=1433,
        driver="pyodbc",
        extra="mssql",
        url=_mssql,
        warmup=60.0,
        precondition=_odbc_driver,
        # Same container, same credentials, addressed through a data-source name
        # instead of an address. No `service` override: what changes is the URL,
        # not the server.
        auth=(AuthMode(mode="odbc-dsn", reach=_mssql_dsn),),
    ),
    Endpoint(
        dialect="oracle",
        service="localdata-test-oracle",
        container_port=1521,
        driver="oracledb",
        extra="oracle",
        url=_oracle,
        warmup=120.0,
    ),
    Endpoint(
        dialect="clickhousedb",
        service="localdata-test-clickhouse",
        container_port=8123,
        driver="clickhouse_connect",
        extra="clickhouse",
        url=_clickhouse,
        auth=(
            AuthMode(
                mode="empty-password",
                reach=_clickhouse_empty_password,
                service="localdata-test-clickhouse-noauth",
                container_port=8123,
            ),
        ),
    ),
    Endpoint(
        dialect="cockroachdb",
        service="localdata-test-cockroachdb",
        container_port=26257,
        driver="sqlalchemy_cockroachdb",
        extra="cockroachdb",
        url=_cockroachdb,
    ),
    # Shares PostgreSQL's dialect, driver and extra, and is the first endpoint
    # here to share any of them. That is what makes it the first real exercise
    # of `name` being the identity rather than `dialect`: keyed the old way,
    # this entry would read Postgres's URL out of the probe cache and run its
    # whole suite against the wrong container while reporting green. See #44.
    Endpoint(
        dialect="postgresql",
        service="localdata-test-yugabytedb",
        container_port=5433,
        driver="psycopg",
        extra="postgres",
        url=_yugabytedb,
        engine="yugabytedb",
        warmup=60.0,
    ),
    Endpoint(
        dialect="trino",
        service="localdata-test-trino",
        container_port=8080,
        driver="trino.sqlalchemy",
        extra="trino",
        url=_trino,
        warmup=60.0,
    ),
    Endpoint(
        dialect="monetdb",
        service="localdata-test-monetdb",
        container_port=50000,
        driver="pymonetdb",
        extra="monetdb",
        url=_monetdb,
    ),
    Endpoint(
        dialect="crate",
        service="localdata-test-cratedb",
        container_port=4200,
        driver="crate",
        extra="cratedb",
        url=_cratedb,
        warmup=60.0,
    ),
    # `engine` differs from `dialect` for the second time here, and for a new
    # reason: YugabyteDB borrows PostgreSQL's dialect, whereas Firebird's own
    # dialect is named after the driver that speaks to it. Both end up asserting
    # the same thing — that the backend answering knows which engine it is.
    Endpoint(
        dialect="firebirdsql",
        service="localdata-test-firebird",
        container_port=3050,
        driver="firebirdsql",
        extra="firebird",
        url=_firebird,
        engine="firebird",
    ),
    # No `engine` override: openGauss ships its own dialect, so the name the
    # backend answers to is already the engine's. `opengauss_sqlalchemy` rather
    # than `psycopg2` as the driver to import — both arrive with the same extra, so
    # either would skip correctly, and naming the dialect says which distribution
    # is missing rather than which library.
    Endpoint(
        dialect="opengauss",
        service="localdata-test-opengauss",
        container_port=5432,
        driver="opengauss_sqlalchemy",
        extra="opengauss",
        url=_opengauss,
        warmup=60.0,
    ),
    # `engine` differs from `dialect` for the third time, and for a third distinct
    # reason: YugabyteDB borrows another engine's dialect, Firebird's is named
    # after its driver, and YDB's is named after its **query language**. A
    # refusal that said `yql` would name a syntax to someone who opened a
    # database.
    Endpoint(
        dialect="yql",
        service="localdata-test-ydb",
        container_port=2136,
        driver="ydb_sqlalchemy",
        extra="ydb",
        url=_ydb,
        engine="ydb",
    ),
    # No `engine` override, and the entry after three consecutive ones that
    # needed it: `databend-sqlalchemy` registers `databend`, calls itself
    # `databend`, names its driver `databend` and answers for Databend. Measured
    # (`dialect.name`, `dialect.driver`) rather than read off the entry points,
    # because that is exactly the reading that got YDB wrong.
    Endpoint(
        dialect="databend",
        service="localdata-test-databend",
        container_port=8000,
        driver="databend_sqlalchemy",
        extra="databend",
        url=_databend,
    ),
    # `engine` differs from `dialect` for the fourth time, and this one is
    # narrower than the three before it: not another engine's dialect, not a
    # driver's name and not a query language, but **one package disagreeing with
    # itself**. `sqlalchemy-exasol` registers the entry point `exa`, which is
    # what a URL resolves to and therefore what `backend_for` is keyed on, while
    # the dialect it registers calls itself `exasol` — and reports its driver as
    # `exasol.driver.websocket.dbapi2`, a module path rather than a driver name.
    # A refusal has to say `exasol`.
    #
    # `warmup` is generous because this container builds a cluster before it
    # serves anything: the port is published while `exadt` is still at stage 2.
    Endpoint(
        dialect="exa",
        service="localdata-test-exasol",
        container_port=8563,
        driver="sqlalchemy_exasol",
        extra="exasol",
        url=_exasol,
        engine="exasol",
        warmup=90.0,
    ),
)


#: Every endpoint, once per way of reaching it. This is what the endpoint suite
#: runs over, so a mode added above is exercised by all of it without a test
#: being written — and a mode that breaks a verb reddens that verb rather than a
#: connection test somebody has to remember to read.
#:
#: The credentialed mode comes first for each endpoint, so a run that is
#: interrupted has covered the ordinary path before the unusual ones.
TARGETS = tuple(
    target
    for endpoint in ENDPOINTS
    for target in (Target(endpoint), *(Target(endpoint, m) for m in endpoint.auth))
)


# ---------------------------------------------------------------------------
# Reading the compose file
# ---------------------------------------------------------------------------


def _services() -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the env
        raise Unavailable(
            "Reading docker-compose.test.yml needs PyYAML: "
            "uv sync --extra dev --extra yaml."
        ) from exc
    return yaml.safe_load(COMPOSE.read_text())["services"]


def _service(target: Target) -> dict:
    """One service's compose entry, or a failure naming the drift.

    A missing service is not a skip. It means this table and the compose file
    disagree about what exists, and a harness that quietly skipped would report
    the dialect as untested rather than as unconfigured.
    """
    services = _services()
    try:
        return services[target.service]
    except KeyError:
        raise RuntimeError(
            f"{COMPOSE.name} has no service {target.service!r}. Known: "
            f"{', '.join(sorted(services))}. The endpoint table in "
            f"{Path(__file__).name} and the compose file have drifted apart."
        ) from None


def _published_port(target: Target, service: dict) -> int:
    """The host-side port this service publishes for its database."""
    for mapping in service.get("ports", ()):
        host, _, container = str(mapping).partition(":")
        if container == str(target.container_port):
            return int(host)
    raise RuntimeError(
        f"{target.service} publishes {service.get('ports')}, none of which "
        f"maps to port {target.container_port} inside the container."
    )


def _environment(service: dict) -> dict[str, str]:
    """A service's environment as a mapping, whichever form compose uses.

    Compose accepts both a mapping and a ``KEY=value`` list, and a harness that
    handled only one would break on an ordinary edit to the compose file.
    """
    raw = service.get("environment", {})
    if isinstance(raw, dict):
        return {key: str(value) for key, value in raw.items()}
    pairs = (str(item).partition("=") for item in raw)
    return {key: value for key, _, value in pairs}


# ---------------------------------------------------------------------------
# Is it there?
# ---------------------------------------------------------------------------


def _listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((HOST, port)) == 0


def _handshake(url: str, warmup: float) -> None:
    """Wait until the database answers, or say why it never did.

    Deliberately a **failure** rather than a skip when it runs out: the port is
    open, so a container is there and this harness could not use it. Treating
    that as "not running" would hide a wrong password or a missing schema behind
    the same green run as a machine with no Docker at all.

    **The probe is a Core expression, not a string** (issue #54). It used to be
    ``SELECT 1`` with a branch for Oracle, which needs a FROM clause — and that
    branch was two defects at once: a dialect fact stated in a test fixture, which
    standing instruction 1 forbids as firmly as one in shared code, and a dialect
    sniffed out of a URL, which #45 established is not an identity. Firebird needs
    ``FROM RDB$DATABASE`` and would have been the second entry.

    None was needed. ``select(literal(1))`` compiles per dialect and already
    emits ``FROM DUAL`` on Oracle and ``FROM rdb$database`` on Firebird, so the
    fix deleted the special case rather than joining it. A per-dialect table whose
    every row is a *spelling* of one operation is usually a portable expression
    somebody has not looked for yet.
    """
    from sqlalchemy import create_engine, literal, select

    probe = select(literal(1))
    deadline = time.monotonic() + warmup
    last: Exception | None = None
    while True:
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                conn.execute(probe)
            return
        except Exception as exc:  # noqa: BLE001 - every driver has its own
            last = exc
        finally:
            engine.dispose()
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"A container is listening but never completed a handshake "
                f"within {warmup:.0f}s. Last complaint: {type(last).__name__}: "
                f"{str(last).splitlines()[0] if last else ''}"
            )
        time.sleep(1.0)


@contextmanager
def applied(environ: dict[str, str]) -> Iterator[None]:
    """Hold an authentication mode's environment for the length of a block.

    Modes that put the credential outside the URL need it in the process
    environment at the moment the driver connects, and **not** afterwards: a
    ``PGPASSWORD`` left set would make the next target's passwordless URL work
    for a reason that has nothing to do with the mode under test, which is the
    quietest way a suite can report coverage it does not have.

    Restores rather than deletes, because a variable this sets may already have
    a value on the machine the suite is running on.
    """
    before = {name: os.environ.get(name) for name in environ}
    os.environ.update(environ)
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


#: Where modes that keep a credential in a file put it. One directory for the
#: whole session, because the probe below is cached for the session and the file
#: has to outlive the connection that proved it works.
#:
#: Created on demand rather than at import, so a run that never reaches an
#: endpoint — no Docker, or ``-k`` selecting something else — leaves nothing
#: behind at all.
_scratch: Path | None = None


def _scratch_dir() -> Path:
    global _scratch
    if _scratch is None:
        _scratch = Path(tempfile.mkdtemp(prefix="localdata-endpoint-auth-"))
    return _scratch


#: One probe per *target* per session: either how it answered, or the reason it
#: was skipped. Eighteen tests against one database must not pay Oracle's warmup
#: eighteen times, and must not each print a different reason for the same
#: absence.
#:
#: **Keyed by :attr:`Target.name`, not by dialect and not by endpoint.** Keyed by
#: dialect, a second endpoint sharing one — TiDB on MySQL's, YugabyteDB on
#: PostgreSQL's — would read the first's URL out of this cache and run its whole
#: suite against a container it never named, reporting green for a database that
#: was never reached (issue #44). Keyed by endpoint, every authentication mode
#: would do the same to the mode before it, which is the same defect one axis
#: further out.
_probed: dict[str, Reached | Exception] = {}


def reach(target: Target) -> Reached:
    """How this target answers, or the reason it does not.

    **Both outcomes are cached, not only the skip.** A misconfigured target
    fails its handshake by waiting out the whole warmup, and until this cached
    it did that once per test — nineteen times thirty seconds for one endpoint
    whose credentials were wrong, which is ten minutes of a run spent
    rediscovering a single fact. The authentication axis multiplies the number of
    targets, so it multiplied that cost too, which is how it was noticed.

    Nothing is quieter for it: the same complaint is raised for every test that
    asked, so the failure is still reported against each one. What is lost is the
    chance for a container that becomes healthy mid-run to be picked up
    half-way through, and a run whose result depends on when a container
    finished starting is not one to want.
    """
    if target.name not in _probed:
        try:
            _probed[target.name] = _probe(target)
        except Exception as exc:  # noqa: BLE001 - a skip and a failure alike
            _probed[target.name] = exc
    answer = _probed[target.name]
    if isinstance(answer, Exception):
        raise answer
    return answer


def _probe(target: Target) -> Reached:
    endpoint = target.endpoint
    try:
        importlib.import_module(endpoint.driver)
    except ModuleNotFoundError as exc:
        raise Unavailable(
            f"{endpoint.dialect} needs the {endpoint.driver} driver: "
            f"uv sync --extra {endpoint.extra}."
        ) from exc
    if endpoint.precondition is not None:
        endpoint.precondition()
    if target.auth is not None and target.auth.precondition is not None:
        target.auth.precondition()

    service = _service(target)
    port = _published_port(target, service)
    if not _listening(port):
        raise Unavailable(
            f"Nothing is listening on {HOST}:{port}. Start it with: "
            f"docker compose -f {COMPOSE.name} up -d {target.service}"
        )

    reached = target.build(_environment(service), port, _scratch_dir())
    # The handshake runs under the mode's own environment for the same reason
    # the tests do: for half these modes the credential is *only* there, so a
    # probe that connected without it would be proving the server was not
    # asking.
    with applied(reached.environ):
        _handshake(reached.url, target.warmup)
    return reached
