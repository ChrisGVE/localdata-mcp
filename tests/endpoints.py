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
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    """

    #: SQLAlchemy's backend name for this database.
    dialect: str
    #: The service in ``docker-compose.test.yml`` that provides it.
    service: str
    #: The port the container listens on *inside* itself, which is the right-hand
    #: side of the compose port mapping. The published port is read from there.
    container_port: int
    #: The Python module the URL's driver needs, and the extra that installs it.
    driver: str
    extra: str
    #: Builds the URL from the service's own environment and published port.
    url: Callable[[dict[str, str], int], str]
    #: How long to keep trying the handshake once the port is open. A container
    #: publishes its port before it finishes initialising, and Oracle takes a
    #: minute and a half to come up; below this the answer is "still starting",
    #: above it something is actually wrong.
    warmup: float = 30.0
    #: Anything else that must hold before this dialect can be reached, raising
    #: :class:`Unavailable` when it does not. pyodbc needs a *system* ODBC
    #: driver, which importing it says nothing about.
    precondition: Callable[[], None] | None = None


def _odbc_driver() -> str:
    """The installed ODBC driver for SQL Server, or a reason there is none.

    pyodbc imports perfectly well with no drivers registered at all, so the
    import check says nothing about whether a connection can be made. This asks
    the driver manager what it actually has.
    """
    import pyodbc

    for candidate in pyodbc.drivers():
        if "SQL Server" in candidate:
            return candidate
    raise Unavailable(
        "pyodbc is installed but no ODBC driver for SQL Server is registered. "
        "On macOS: brew tap microsoft/mssql-release && brew install msodbcsql18."
    )


def _postgres(env: dict[str, str], port: int) -> str:
    return (
        f"postgresql+psycopg://{env['POSTGRES_USER']}:{env['POSTGRES_PASSWORD']}"
        f"@{HOST}:{port}/{env['POSTGRES_DB']}"
    )


def _mysql(env: dict[str, str], port: int) -> str:
    return (
        f"mysql+pymysql://{env['MYSQL_USER']}:{env['MYSQL_PASSWORD']}"
        f"@{HOST}:{port}/{env['MYSQL_DATABASE']}"
    )


def _mariadb(env: dict[str, str], port: int) -> str:
    """MariaDB is addressed as MariaDB, not as MySQL.

    PyMySQL speaks to both, but the URL's scheme is what decides which dialect
    SQLAlchemy loads and therefore which backend answers for it — and the two
    have diverged enough that being told which one is on the other end is worth
    more than sharing a name.
    """
    return (
        f"mariadb+pymysql://{env['MARIADB_USER']}:{env['MARIADB_PASSWORD']}"
        f"@{HOST}:{port}/{env['MARIADB_DATABASE']}"
    )


def _mssql(env: dict[str, str], port: int) -> str:
    """SQL Server, through whichever ODBC driver is installed.

    ``master`` because the image creates no other database and there is no
    environment variable that would ask it to; the tests name their tables
    uniquely and drop what they made, so a shared database costs nothing. The
    certificate is self-signed, hence ``TrustServerCertificate`` — driver 18
    encrypts by default and would otherwise refuse the container outright.
    """
    driver = _odbc_driver().replace(" ", "+")
    return (
        f"mssql+pyodbc://sa:{env['MSSQL_SA_PASSWORD']}@{HOST}:{port}/master"
        f"?driver={driver}&TrustServerCertificate=yes"
    )


def _oracle(env: dict[str, str], port: int) -> str:
    """Oracle Free, reached through python-oracledb in its thin mode.

    ``FREEPDB1`` is the pluggable database the image creates and the one
    ``APP_USER`` is created in; the container's own root service would need
    privileged credentials and holds nothing a test wants.
    """
    return (
        f"oracle+oracledb://{env['APP_USER']}:{env['APP_USER_PASSWORD']}"
        f"@{HOST}:{port}/?service_name=FREEPDB1"
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
    ),
    Endpoint(
        dialect="mysql",
        service="localdata-test-mysql",
        container_port=3306,
        driver="pymysql",
        extra="mysql",
        url=_mysql,
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


def _service(endpoint: Endpoint) -> dict:
    """One service's compose entry, or a failure naming the drift.

    A missing service is not a skip. It means this table and the compose file
    disagree about what exists, and a harness that quietly skipped would report
    the dialect as untested rather than as unconfigured.
    """
    services = _services()
    try:
        return services[endpoint.service]
    except KeyError:
        raise RuntimeError(
            f"{COMPOSE.name} has no service {endpoint.service!r}. Known: "
            f"{', '.join(sorted(services))}. The endpoint table in "
            f"{Path(__file__).name} and the compose file have drifted apart."
        ) from None


def _published_port(endpoint: Endpoint, service: dict) -> int:
    """The host-side port this service publishes for its database."""
    for mapping in service.get("ports", ()):
        host, _, container = str(mapping).partition(":")
        if container == str(endpoint.container_port):
            return int(host)
    raise RuntimeError(
        f"{endpoint.service} publishes {service.get('ports')}, none of which "
        f"maps to port {endpoint.container_port} inside the container."
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
    """
    from sqlalchemy import create_engine, text

    deadline = time.monotonic() + warmup
    last: Exception | None = None
    while True:
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                conn.execute(
                    text("SELECT 1 FROM dual" if ":oracle" in url else "SELECT 1")
                )
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


#: One probe per dialect per session: either the URL it answered on, or the
#: reason it was skipped. Ten tests against five dialects must not pay Oracle's
#: warmup ten times, and must not each print a different reason for the same
#: absence.
_probed: dict[str, str | Unavailable] = {}


def url_for(endpoint: Endpoint) -> str:
    """The URL this endpoint answers on, or :class:`Unavailable` saying why not."""
    if endpoint.dialect not in _probed:
        try:
            _probed[endpoint.dialect] = _probe(endpoint)
        except Unavailable as exc:
            _probed[endpoint.dialect] = exc
    answer = _probed[endpoint.dialect]
    if isinstance(answer, Unavailable):
        raise answer
    return answer


def _probe(endpoint: Endpoint) -> str:
    try:
        importlib.import_module(endpoint.driver)
    except ModuleNotFoundError as exc:
        raise Unavailable(
            f"{endpoint.dialect} needs the {endpoint.driver} driver: "
            f"uv sync --extra {endpoint.extra}."
        ) from exc
    if endpoint.precondition is not None:
        endpoint.precondition()

    service = _service(endpoint)
    port = _published_port(endpoint, service)
    if not _listening(port):
        raise Unavailable(
            f"Nothing is listening on {HOST}:{port}. Start it with: "
            f"docker compose -f {COMPOSE.name} up -d {endpoint.service}"
        )

    url = endpoint.url(_environment(service), port)
    _handshake(url, endpoint.warmup)
    return url
