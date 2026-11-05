"""A misspelled first word is answered as the server's syntax error (#143).

On the PostgreSQL family a streamed read is a cursor declaration, and the
server will not declare a cursor over a write, so ``INSERT …`` arrives as
``syntax error at or near "INSERT"`` beside ``CURSOR FOR INSERT``. The explainer
reads that as a refused kind of statement (#58, #141). ``SELEC 1`` arrives in
exactly the same words, and was answered "it is not a read … use create,
update or drop", which sends somebody with a typo to the wrong verb.

So before reporting a refused kind, the explainer asks the server to *parse*
the statement: the extended protocol's Parse message, ``PQprepare`` with the
unnamed statement, which parses and analyses and never plans, binds or
executes. A syntax error there (42601) is the answer, in the server's words;
a statement that parses keeps the refused-kind answer. On PostgreSQL,
CockroachDB and YugabyteDB it is psycopg 3's connection that is asked. openGauss
is reached through psycopg 2, which sends no Parse of its own, so the same
``PQprepare`` is asked through the libpq psycopg 2 loaded, on its connection.

The live half needs the containers and skips without them, like every
endpoint test. It proves the parse cannot run a write: an ``INSERT`` and a
``CREATE TABLE`` asked about leave the table's count and the catalog as they
were, on a slot attached writable too.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from psycopg import pq
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import Refusal, backend_for
from localdata_mcp.loader import Workspace

_CURSOR_MESSAGE = (
    'syntax error at or near "{verb}"\n'
    'LINE 1: DECLARE "c_7f8316b95bd0_94" CURSOR FOR {sql}'
)


# ---------------------------------------------------------------------------
# The explainer asks, offline
# ---------------------------------------------------------------------------


class _AskedBackend:
    """A backend standing in for the server's parse, recording what it was asked."""

    name = "postgresql"

    def __init__(self, answer):
        self.answer = answer
        self.asked: list[str] = []

    def denies_write(self, exc):
        return False

    def ddl_survives_refusal(self):
        return False

    def syntax_error(self, engine, sql, message):
        self.asked.append(sql)
        return self.answer


def _explained(sql: str, verb: str, backend) -> str:
    entry = SimpleNamespace(
        engines=SimpleNamespace(refusal=Refusal(), read=object()),
        backend=backend,
        tag="endpoint",
    )
    message = _CURSOR_MESSAGE.format(verb=verb, sql=sql)
    error = ProgrammingError(sql, None, Exception(message))
    return str(Workspace._explain(None, entry, error, sql))


def test_a_first_word_the_server_cannot_parse_is_its_syntax_error():
    backend = _AskedBackend('syntax error at or near "SELEC"\nLINE 1: SELEC 1')

    error = _explained("SELEC 1", "SELEC", backend)

    assert error.startswith('syntax error at or near "SELEC"'), error
    assert "DECLARE" not in error and "CURSOR" not in error, error
    assert "create" not in error and "not a read" not in error, error
    assert backend.asked == ["SELEC 1"]


def test_a_statement_that_parses_is_still_a_refused_kind():
    backend = _AskedBackend(None)

    error = _explained("INSERT INTO t VALUES (1)", "INSERT", backend)

    assert "create" in error and "drop" in error, error
    assert backend.asked == ["INSERT INTO t VALUES (1)"]


def test_the_server_is_not_asked_about_an_ordinary_error():
    """A malformed SELECT objects past its first word; no parse is needed."""
    backend = _AskedBackend("never used")

    error = _explained("SELECT , FROM t", ",", backend)

    assert backend.asked == []
    assert 'syntax error at or near ","' in error, error


# ---------------------------------------------------------------------------
# The parse, against stand-ins for psycopg's connection
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, status, code=b"", message=""):
        self.status = status
        self._code = code
        self._message = message

    def error_field(self, field):
        return self._code if field == pq.DiagnosticField.SQLSTATE else None

    def get_error_message(self, encoding="utf-8"):
        return self._message


class _PGconn:
    def __init__(self, result):
        self.result = result
        self.prepared: list[tuple[bytes, bytes]] = []

    def prepare(self, name, command, param_types=None):
        self.prepared.append((name, command))
        return self.result


class _Engine:
    """``engine.connect()`` handing back one driver connection."""

    def __init__(self, driver):
        self.driver = driver

    def connect(self):
        engine = self

        class _Connection:
            connection = SimpleNamespace(driver_connection=engine.driver)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Connection()


def _psycopg3(result):
    pgconn = _PGconn(result)
    return pgconn, _Engine(
        SimpleNamespace(pgconn=pgconn, info=SimpleNamespace(encoding="utf-8"))
    )


def test_a_syntax_error_from_the_parse_is_returned_in_the_servers_words():
    pgconn, engine = _psycopg3(
        _Result(pq.ExecStatus.FATAL_ERROR, b"42601", 'syntax error at or near "SELEC"')
    )

    answer = backend_for("postgresql").syntax_error(engine, "SELEC 1", "…")

    assert answer == 'syntax error at or near "SELEC"'
    assert pgconn.prepared == [(b"", b"SELEC 1")]  # the unnamed statement


@pytest.mark.parametrize(
    "result",
    [
        _Result(pq.ExecStatus.COMMAND_OK),
        _Result(pq.ExecStatus.FATAL_ERROR, b"42P01", 'relation "t" does not exist'),
    ],
    ids=["parses", "parses-but-names-a-missing-table"],
)
def test_anything_but_a_syntax_error_leaves_the_refused_kind_reading(result):
    _, engine = _psycopg3(result)

    assert (
        backend_for("postgresql").syntax_error(engine, "INSERT INTO t VALUES (1)", "…")
        is None
    )


def test_a_driver_without_a_parse_of_its_own_is_not_asked():
    """Neither psycopg's ``pgconn`` nor psycopg 2's ``pgconn_ptr``: no answer, nothing sent."""
    engine = _Engine(SimpleNamespace())

    assert backend_for("postgresql").syntax_error(engine, "SELEC 1", "…") is None


def test_generic_backends_have_no_one_to_ask():
    assert backend_for("sqlite").syntax_error(None, "SELEC 1", "…") is None


# ---------------------------------------------------------------------------
# Live, on the PostgreSQL family
# ---------------------------------------------------------------------------

_PG_FAMILY = {"postgres", "cockroachdb", "yugabytedb", "opengauss"}
_SERVERS = [t for t in TARGETS if t.name in _PG_FAMILY and t.auth is None]


def _call(tool: str, **arguments):
    """The in-process protocol client, as ``test_endpoints`` has it."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


@pytest.fixture(params=[False, True], ids=["read-only", "writable"])
def writable(request):
    return request.param


@pytest.fixture(params=_SERVERS, ids=lambda target: target.name)
def server(request, tmp_path, writable):
    """A slot over a one-row table; yields ``(table, admin, engine_name)``."""
    try:
        reached = reach(request.param)
    except Unavailable as exc:
        pytest.skip(str(exc))
    table = f"t{uuid.uuid4().hex[:8]}"
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    with applied(reached.environ):
        admin = create_engine(reached.url)
        try:
            with admin.begin() as conn:
                conn.execute(text(f"CREATE TABLE {table} (a INT)"))
                conn.execute(text(f"INSERT INTO {table} VALUES (1)"))
            grant = {"writable": True} if writable else {}
            attached = _call(
                "attach", database=reached.url, nickname="endpoint", **grant
            )
            assert attached["ok"] is True, attached
            yield table, admin, request.param.name
        finally:
            server_module._reset()
            with admin.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                conn.execute(text(f"DROP TABLE IF EXISTS {table}_made"))
            admin.dispose()


def _count(admin, table: str) -> int:
    with admin.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()


def _exists(admin, table: str) -> bool:
    with admin.connect() as conn:
        return bool(
            conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.tables WHERE table_name = :t"
                ),
                {"t": table},
            ).scalar()
        )


@pytest.mark.endpoint
@pytest.mark.parametrize("sql", ["SELEC 1", "INSERTT INTO {t} VALUES (2)"])
def test_a_misspelled_first_word_is_the_servers_syntax_error(server, sql):
    table, _, _ = server
    word = sql.split()[0]

    answer = _call("query", nickname="endpoint", sql=sql.format(t=table))

    assert answer["ok"] is False, answer
    assert "syntax error" in answer["error"], answer
    assert word in answer["error"], answer
    assert "create" not in answer["error"], answer
    assert "CURSOR" not in answer["error"], answer


@pytest.mark.endpoint
def test_asking_the_server_to_parse_a_write_does_not_run_it(server):
    """The parse itself, and query around it: the count and the catalog are unchanged."""
    table, admin, name = server
    backend = backend_for("postgresql")
    engines = server_module._session()._workspace.entry("endpoint").engines

    for sql in (
        f"INSERT INTO {table} VALUES (2)",
        f"CREATE TABLE {table}_made (a INT)",
        f"DELETE FROM {table}",
    ):
        assert backend.syntax_error(engines.read, sql, "…") is None, sql
        answer = _call("query", nickname="endpoint", sql=sql)
        assert answer["ok"] is False, answer
        assert "create" in answer["error"], answer  # still the refused kind

    assert _count(admin, table) == 1
    assert not _exists(admin, f"{table}_made")


# ---------------------------------------------------------------------------
# openGauss asks the same Parse, through psycopg 2's own libpq (#143 residual)
# ---------------------------------------------------------------------------
#
# psycopg 2 sends no Parse message of its own, but it is a libpq client and
# hands out its connection's ``PGconn *`` as ``pgconn_ptr``. Asked through the
# libpq psycopg 2 itself loaded, ``PQprepare`` is the same Parse the psycopg 3
# engines are asked, so on openGauss a keyword that cannot begin a statement
# (``FROM t``) is answered with the server's syntax error too, where asking the
# keyword list could not tell it from ``CREATE``.


class _Libpq:
    """libpq's five calls, answering one prepared result, recording the calls."""

    def __init__(self, status, code=None, message=b"", severity=b"ERROR"):
        self.status = status
        self.code = code
        self.message = message
        self.severity = severity
        self.prepared: list[tuple] = []
        self.cleared: list[object] = []

    def PQprepare(self, conn, name, command, count, types):
        self.prepared.append((conn, name, command, count, types))
        return "result"

    def PQresultStatus(self, result):
        return self.status

    def PQresultErrorField(self, result, field):
        return {ord("C"): self.code, ord("S"): self.severity}.get(field)

    def PQresultErrorMessage(self, result):
        return self.message

    def PQclear(self, result):
        self.cleared.append(result)


def _psycopg2(monkeypatch, libpq):
    from localdata_mcp import dialects

    monkeypatch.setattr(dialects, "_psycopg2_libpq", lambda: libpq)
    return _Engine(SimpleNamespace(pgconn_ptr=4242, encoding="UTF8"))


_PGRES_COMMAND_OK = 1
_PGRES_FATAL_ERROR = 7


@pytest.mark.parametrize("name", ["opengauss", "postgresql"])
def test_psycopg2s_connection_is_asked_to_parse_with_the_unnamed_statement(
    monkeypatch, name
):
    libpq = _Libpq(
        _PGRES_FATAL_ERROR,
        b"42601",
        b'ERROR:  syntax error at or near "FROM"\nLINE 1: FROM t\n        ^\n',
    )
    engine = _psycopg2(monkeypatch, libpq)

    answer = backend_for(name).syntax_error(engine, "FROM t", "the cursor's words")

    assert answer == 'syntax error at or near "FROM"\nLINE 1: FROM t\n        ^'
    assert libpq.prepared == [(4242, b"", b"FROM t", 0, None)]
    assert libpq.cleared == ["result"]  # the result is freed, whatever it said


@pytest.mark.parametrize(
    "libpq",
    [
        _Libpq(_PGRES_COMMAND_OK),
        _Libpq(_PGRES_FATAL_ERROR, b"42P01", b'ERROR:  relation "t" does not exist\n'),
    ],
    ids=["parses", "parses-but-names-a-missing-table"],
)
def test_psycopg2_anything_but_a_syntax_error_leaves_the_refused_kind_reading(
    monkeypatch, libpq
):
    engine = _psycopg2(monkeypatch, libpq)

    answer = backend_for("opengauss").syntax_error(
        engine, "CREATE TABLE t (a INT)", "…"
    )

    assert answer is None
    assert libpq.cleared == ["result"]


def test_the_libpq_asked_is_the_one_psycopg2_itself_loaded():
    """Resolved through psycopg 2's own extension, so the PGconn is the one it made."""
    psycopg2 = pytest.importorskip("psycopg2")
    from localdata_mcp import dialects

    libpq = dialects._psycopg2_libpq()

    assert libpq.PQlibVersion() == psycopg2.__libpq_version__


@pytest.mark.endpoint
@pytest.mark.parametrize("sql", ["FROM {t}", "WHERE a = 1"])
def test_a_keyword_that_cannot_begin_a_statement_is_the_servers_syntax_error(
    server, sql
):
    table, admin, _ = server
    word = sql.split()[0]

    answer = _call("query", nickname="endpoint", sql=sql.format(t=table))

    assert answer["ok"] is False, answer
    # Each engine's own wording: PostgreSQL, YugabyteDB and openGauss say
    # `syntax error at or near "FROM"`, CockroachDB `at or near "from": syntax
    # error`. Both name the word the server stopped at.
    assert "syntax error" in answer["error"], answer
    assert f'at or near "{word.lower()}"' in answer["error"].lower(), answer
    assert "CURSOR" not in answer["error"], answer
    assert "create" not in answer["error"], answer
    assert _count(admin, table) == 1
