"""A write on CockroachDB is bounded at 30 seconds, and a timed-out one says what it left (#148).

On CockroachDB a schema change waits, inside the statement, until every other
session's open transaction on the table has ended: CockroachDB is waiting for
the old version of the table's descriptor to drain. A plain ``SELECT`` left
open in another session is enough. With no bound the verb answered only once
that transaction ended, and because every verb runs under the server's one
lock, no other call on any datasource answered in the meantime.

Decided by Chris on 2026-11-05: writes on CockroachDB get a 30 second bound.
Measured on CockroachDB v26.3.4 (docs/CONSTRAINTS.md §15, amendment of
2026-11-05), with another session holding a ``SELECT`` or an uncommitted
``UPDATE`` open on the table:

* ``lock_timeout`` never fired: 8 of 8 DDL statements waited until the other
  transaction ended. ``statement_timeout`` fired every time, inside the DDL,
  with SQLSTATE 57014, 8 of 8.
* At that moment, read from a fresh connection, a rename, a ``DROP TABLE``
  and a ``DROP INDEX`` were already applied (6 of 6). A ``CREATE INDEX`` was
  not yet usable (2 of 2): its build carried on as a background job, a resend
  was refused because the index already existed, and the index appeared once
  the other transaction ended.

So the refusal says what the call waited on, what is in place, and what to do
next, per verb. The offline half builds each refusal and checks the write
engine is given the bound; the live half needs the CockroachDB container and
skips without it.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import Index, Table, create_engine, text
from sqlalchemy.exc import OperationalError

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import (
    COCKROACHDB_WRITE_BOUND_SECONDS,
    Backend,
    CockroachDBBackend,
    PostgreSQLBackend,
)
from localdata_mcp.loader import LoadError, Workspace

# ---------------------------------------------------------------------------
# Offline: the bound is set on the write engine, on CockroachDB only
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, sent: list[str]) -> None:
        self._sent = sent

    def execute(self, statement: str) -> None:
        self._sent.append(statement)

    def close(self) -> None:
        return None


class _Connection:
    """Just enough of a DBAPI connection for a ``connect`` listener to run on."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.commits = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self.sent)

    def commit(self) -> None:
        self.commits += 1


def _connected(engine, url: str) -> _Connection:
    """Run the ``connect`` listeners the backend added, on a fake connection.

    Only those: the ones SQLAlchemy and the dialect install for themselves are
    present on a generic open of the same URL as well, and set driver hooks a
    fake connection has no business imitating. What is left is this
    backend's own posture.
    """
    generic = Backend().open(url, writable=True)
    try:
        theirs = {
            getattr(listener, "__qualname__", None)
            for listener in generic.write.pool.dispatch.connect
        }
    finally:
        generic.dispose()
    connection = _Connection()
    for listener in engine.pool.dispatch.connect:
        if getattr(listener, "__qualname__", None) in theirs:
            continue
        listener(connection, None)
    return connection


_URL = "cockroachdb+psycopg://root@localhost:1/defaultdb"


def test_the_bound_is_thirty_seconds():
    # Chris's value (2026-11-05, #148), not a tuning knob of this test.
    assert COCKROACHDB_WRITE_BOUND_SECONDS == 30


def test_a_cockroachdb_write_engine_sets_the_statement_timeout_on_connect():
    engines = CockroachDBBackend().open(_URL, writable=True)
    try:
        connection = _connected(engines.write, _URL)
    finally:
        engines.dispose()

    assert "SET statement_timeout = '30s'" in connection.sent, connection.sent
    # psycopg opens a transaction around a SET; it is ended, so the pooled
    # connection is idle when the server's own transaction begins.
    assert connection.commits >= 1


def test_a_cockroachdb_read_engine_is_not_bounded():
    # The decision bounds writes. The reader keeps its posture and nothing else.
    engines = CockroachDBBackend().open(_URL, writable=True)
    try:
        connection = _connected(engines.read, _URL)
    finally:
        engines.dispose()

    assert not any("statement_timeout" in sent for sent in connection.sent)


def test_a_postgresql_write_engine_is_not_bounded():
    url = "postgresql+psycopg://u@localhost:1/db"
    engines = PostgreSQLBackend().open(url, writable=True)
    try:
        connection = _connected(engines.write, url)
    finally:
        engines.dispose()

    assert connection.sent == []


def _timed_out(code: str = "57014") -> OperationalError:
    original = type("DriverFailure", (Exception,), {"sqlstate": code})(
        "query execution canceled due to statement timeout"
    )
    return OperationalError("ALTER TABLE people RENAME TO staff", {}, original)


def test_cockroachdb_recognises_its_own_bound_firing():
    assert CockroachDBBackend().write_bound(_timed_out()) == 30


@pytest.mark.parametrize(
    "backend", [Backend(), PostgreSQLBackend()], ids=lambda b: b.name
)
def test_a_backend_that_sets_no_bound_recognises_none(backend):
    # A 57014 there comes from somebody else's timeout or a cancel, and what it
    # left behind was never measured, so it keeps the words it always had.
    assert backend.write_bound(_timed_out()) is None


def test_cockroachdb_does_not_take_another_failure_for_its_bound():
    assert CockroachDBBackend().write_bound(_timed_out("40001")) is None


# ---------------------------------------------------------------------------
# Offline: what a timed-out write says, per verb
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path):
    config_module.use(Config(roots=(tmp_path,)))
    ws = Workspace.in_memory()
    ws.attach_memory("main")
    with ws.entry("main").engines.write.begin() as conn:
        conn.execute(text("CREATE TABLE people (name TEXT, salary INTEGER)"))
        conn.execute(text("INSERT INTO people VALUES ('a', 1), ('b', 2)"))
        conn.execute(text("CREATE INDEX ix_people_name ON people (name)"))
    yield ws
    ws.close()


#: Which call each write makes to the database, so the timeout can be put there.
_SENDS = {
    "rename": ("backend", "rename_table"),
    "drop table": (Table, "drop"),
    "create index": (Index, "create"),
    "drop index": (Index, "drop"),
}


def _writes(ws: Workspace) -> dict[str, Callable[[], object]]:
    return {
        "rename": lambda: ws.rename_table("main", "people", "staff"),
        "drop table": lambda: ws.drop_table("main", "people"),
        "create index": lambda: ws.create_index("main", "people", ["salary"]),
        "drop index": lambda: ws.drop_index("main", "ix_people_name"),
    }


#: What each refusal must say is in place, measured on CockroachDB v26.3.4.
_IN_PLACE = {
    "rename": "The rename is already made: main.people is now main.staff.",
    "drop table": "The drop is already made: main.people is gone.",
    "create index": (
        "The index is not usable yet: CockroachDB keeps building "
        "ix_people_salary after this call"
    ),
    "drop index": "The drop is already made: main.ix_people_name is gone.",
}


def _bounded(monkeypatch, ws: Workspace, write: str, code: str = "57014") -> list[int]:
    """Make this write time out with ``code``, on a backend that reports the bound."""
    backend = type(ws.entry("main").backend)
    monkeypatch.setattr(
        backend,
        "write_bound",
        lambda _self, exc: CockroachDBBackend().write_bound(exc),
    )
    owner_name, method = _SENDS[write]
    owner = backend if owner_name == "backend" else owner_name
    calls: list[int] = []

    def time_out(*_args, **_kwargs):
        calls.append(1)
        raise _timed_out(code)

    monkeypatch.setattr(owner, method, time_out)
    return calls


@pytest.mark.parametrize("write", list(_SENDS))
def test_a_timed_out_write_says_what_it_waited_on_and_what_is_in_place(
    workspace, monkeypatch, write
):
    calls = _bounded(monkeypatch, workspace, write)

    with pytest.raises(LoadError) as raised:
        _writes(workspace)[write]()

    message = str(raised.value)
    assert "within 30 seconds" in message, message
    assert "another session's open transaction on the table" in message, message
    assert _IN_PLACE[write] in message, message
    assert "directory(nickname='main'" in message, message
    # The driver's own words still follow, so nothing it said is lost.
    assert "canceled due to statement timeout" in message, message
    # Sent once. It is not resent, by this server or on its advice.
    assert len(calls) == 1


def test_a_timed_out_index_says_a_resend_would_be_refused(workspace, monkeypatch):
    _bounded(monkeypatch, workspace, "create index")

    with pytest.raises(LoadError) as raised:
        workspace.create_index("main", "people", ["salary"])

    message = str(raised.value)
    # Measured: a resend while the build runs answered 42P07 "already exists".
    assert "Do not send this create again" in message, message
    # The cause is not always another session: a build over 2,000,000 rows took
    # 53 s with nobody else connected, and its bound fired the same way.
    assert "a large table takes longer than that to index" in message, message


@pytest.mark.parametrize("write", ["rename", "drop table", "drop index"])
def test_an_applied_write_says_nothing_needs_sending_again(
    workspace, monkeypatch, write
):
    _bounded(monkeypatch, workspace, write)

    with pytest.raises(LoadError) as raised:
        _writes(workspace)[write]()

    assert "Do not send it again" in str(raised.value)


def _loaded(ws: Workspace, tmp_path) -> str:
    """A table loaded from a file, so the workspace holds a description of it."""
    source = tmp_path / "crew.csv"
    source.write_text("name,salary\na,1\nb,2\n")
    (info,) = ws.load_file(str(source), "main", delimiter=",")
    assert info.qualified in ws._tables
    return info.name


def test_a_timed_out_rename_forgets_the_old_name(workspace, monkeypatch, tmp_path):
    table = _loaded(workspace, tmp_path)
    _bounded(monkeypatch, workspace, "rename")

    with pytest.raises(LoadError):
        workspace.rename_table("main", table, "staff")

    # The rename is in place, so a cached description under the old name
    # would describe a table that no longer exists.
    assert "main.people" not in workspace._tables


def test_a_timed_out_drop_forgets_the_table(workspace, monkeypatch, tmp_path):
    table = _loaded(workspace, tmp_path)
    _bounded(monkeypatch, workspace, "drop table")

    with pytest.raises(LoadError):
        workspace.drop_table("main", table)

    assert f"main.{table}" not in workspace._tables


@pytest.mark.parametrize("write", list(_SENDS))
def test_a_57014_where_no_bound_is_set_keeps_its_old_words(
    workspace, monkeypatch, write
):
    # SQLite reports no bound, so the 57014 is not this server's.
    owner_name, method = _SENDS[write]
    backend = type(workspace.entry("main").backend)
    owner = backend if owner_name == "backend" else owner_name
    monkeypatch.setattr(
        owner, method, lambda *_a, **_k: (_ for _ in ()).throw(_timed_out())
    )

    with pytest.raises(LoadError) as raised:
        _writes(workspace)[write]()

    message = str(raised.value)
    assert message.startswith("Could not "), message
    assert "within 30 seconds" not in message, message


# ---------------------------------------------------------------------------
# Live: another session holds a transaction open on the table
# ---------------------------------------------------------------------------


_COCKROACH = [target for target in TARGETS if target.name == "cockroachdb"]

#: How long a timed-out call may take beyond the bound before the test calls
#: it unbounded: the server's own round-trips on either side of the DDL.
_SLACK = 15.0


def _call(tool: str, **arguments):
    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


@contextmanager
def _server(target, tmp_path) -> Iterator[tuple[str, object]]:
    try:
        reached = reach(target)
    except Unavailable as exc:
        pytest.skip(str(exc))
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    admin = create_engine(reached.url)
    try:
        with applied(reached.environ):
            attached = _call(
                "attach", database=reached.url, nickname="endpoint", writable=True
            )
            assert attached["ok"] is True, attached
            yield reached.url, admin
    finally:
        server_module._reset()
        admin.dispose()


@contextmanager
def _held(url: str, table: str) -> Iterator[None]:
    """Another session, not this server, with a read of ``table`` left open."""
    engine = create_engine(url)
    conn = engine.connect()
    try:
        conn.execute(text(f"SELECT count(*) FROM {table}"))
        yield
    finally:
        conn.rollback()
        conn.close()
        engine.dispose()


def _table(admin, *, index: bool = False) -> str:
    name = f"t{uuid.uuid4().hex[:8]}"
    with admin.begin() as conn:
        conn.execute(text(f"CREATE TABLE {name} (id INT PRIMARY KEY, salary INT)"))
        conn.execute(
            text(f"INSERT INTO {name} SELECT g, g % 97 FROM generate_series(1, 1000) g")
        )
        if index:
            conn.execute(text(f"CREATE INDEX ix_{name}_salary ON {name} (salary)"))
    return name


def _tables(admin, *names: str) -> set[str]:
    with admin.connect() as conn:
        return {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
            )
            if row[0] in names
        }


def _public_indexes(admin, table: str) -> set[str]:
    with admin.connect() as conn:
        return {
            row[0]
            for row in conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = :t"),
                {"t": table},
            )
        }


def _drop(admin, *tables: str) -> None:
    with admin.begin() as conn:
        for table in tables:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


def _timed(tool: str, **arguments) -> tuple[dict, float]:
    began = time.monotonic()
    answer = _call(tool, **arguments)
    return answer, time.monotonic() - began


def _assert_bounded(answer: dict, took: float) -> str:
    assert answer["ok"] is False, answer
    assert (
        COCKROACHDB_WRITE_BOUND_SECONDS
        <= took
        < (COCKROACHDB_WRITE_BOUND_SECONDS + _SLACK)
    ), took
    error = answer["error"]
    assert "within 30 seconds" in error, error
    assert "another session's open transaction on the table" in error, error
    return error


@pytest.mark.endpoint
@pytest.mark.slow
@pytest.mark.parametrize("target", _COCKROACH, ids=lambda t: t.name)
def test_a_rename_held_up_by_an_open_reader_answers_at_the_bound_and_is_made(
    target, tmp_path
):
    with _server(target, tmp_path) as (url, admin):
        table = _table(admin)
        try:
            with _held(url, table):
                answer, took = _timed(
                    "update",
                    nickname="endpoint",
                    type="table",
                    name=table,
                    to=f"{table}_r",
                )
                error = _assert_bounded(answer, took)
                assert f"endpoint.{table} is now endpoint.{table}_r." in error, error
                # Read on a connection of the test's own while the other
                # session is still open: the rename is in place.
                assert _tables(admin, table, f"{table}_r") == {f"{table}_r"}
                # The next step the refusal names works, and sees it.
                listed = _call("directory", nickname="endpoint")
                assert listed["ok"] is True, listed
                assert f"{table}_r" in json.dumps(listed), listed
        finally:
            _drop(admin, table, f"{table}_r")


@pytest.mark.endpoint
@pytest.mark.slow
@pytest.mark.parametrize("target", _COCKROACH, ids=lambda t: t.name)
def test_a_drop_held_up_by_an_open_reader_answers_at_the_bound_and_is_made(
    target, tmp_path
):
    with _server(target, tmp_path) as (url, admin):
        table = _table(admin)
        try:
            with _held(url, table):
                answer, took = _timed(
                    "drop", nickname="endpoint", type="table", name=table
                )
                error = _assert_bounded(answer, took)
                assert f"endpoint.{table} is gone." in error, error
                assert _tables(admin, table) == set()
        finally:
            _drop(admin, table)


@pytest.mark.endpoint
@pytest.mark.slow
@pytest.mark.parametrize("target", _COCKROACH, ids=lambda t: t.name)
def test_an_index_drop_held_up_by_an_open_reader_answers_at_the_bound_and_is_made(
    target, tmp_path
):
    with _server(target, tmp_path) as (url, admin):
        table = _table(admin, index=True)
        index = f"ix_{table}_salary"
        try:
            with _held(url, table):
                answer, took = _timed(
                    "drop", nickname="endpoint", type="index", name=index
                )
                error = _assert_bounded(answer, took)
                assert f"endpoint.{index} is gone." in error, error
                assert index not in _public_indexes(admin, table)
        finally:
            _drop(admin, table)


@pytest.mark.endpoint
@pytest.mark.slow
@pytest.mark.parametrize("target", _COCKROACH, ids=lambda t: t.name)
def test_an_index_held_up_by_an_open_reader_answers_at_the_bound_and_is_built_after(
    target, tmp_path
):
    with _server(target, tmp_path) as (url, admin):
        table = _table(admin)
        index = f"ix_{table}_salary"
        try:
            with _held(url, table):
                answer, took = _timed(
                    "create",
                    nickname="endpoint",
                    type="index",
                    table=table,
                    columns=["salary"],
                )
                error = _assert_bounded(answer, took)
                assert "The index is not usable yet" in error, error
                assert index not in _public_indexes(admin, table)
                # A resend is refused rather than building a second index.
                again = _call(
                    "create",
                    nickname="endpoint",
                    type="index",
                    table=table,
                    columns=["salary"],
                )
                assert again["ok"] is False, again
                assert index in again["error"], again
            # The other session has ended: the build completes by itself.
            deadline = time.monotonic() + 60
            while index not in _public_indexes(admin, table):
                assert time.monotonic() < deadline, "the index was never built"
                time.sleep(1)
        finally:
            _drop(admin, table)


@pytest.mark.endpoint
@pytest.mark.slow
@pytest.mark.parametrize("target", _COCKROACH, ids=lambda t: t.name)
def test_the_server_answers_other_calls_again_once_the_bound_has_fired(
    target, tmp_path
):
    # What #148 was about: while the write waited, no other call answered. The
    # lock is still held for the bound (#150), and released when it fires.
    with _server(target, tmp_path) as (url, admin):
        table = _table(admin)
        answered: list[float] = []
        began = time.monotonic()

        def other_call() -> None:
            time.sleep(2)
            listed = _call("directory", nickname="endpoint")
            assert listed["ok"] is True, listed
            answered.append(time.monotonic() - began)

        try:
            with _held(url, table):
                worker = threading.Thread(target=other_call)
                worker.start()
                answer, _ = _timed(
                    "drop", nickname="endpoint", type="table", name=table
                )
                worker.join(COCKROACHDB_WRITE_BOUND_SECONDS + _SLACK)
                assert answer["ok"] is False, answer
                assert answered, "the other call never answered"
                assert answered[0] < COCKROACHDB_WRITE_BOUND_SECONDS + _SLACK
        finally:
            _drop(admin, table)
