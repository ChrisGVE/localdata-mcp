"""A write the database refused with SQLSTATE 40001 is explained, not retried (#47).

40001 is ``serialization_failure``: another session's transaction conflicted
with this one and the database aborted this one rather than let the two
collide. Measured on YugabyteDB 2.25 with a second connection changing the
same table in a loop (docs/CONSTRAINTS.md §15, amendment of 2026-11-05):
``drop(type='table')`` and ``create(type='index')`` were refused with 40001,
nothing had been applied, and the same call sent again once the other session
stopped went through.

The caller used to receive the driver's text and nothing else — a Python
exception class, a hybrid timestamp and a key dump — with no word on whether
anything changed or what to do. So the refusal now says, in this server's
words, that another session's change collided with this one, that nothing was
changed, and that the same call sent again goes through once that change is
done. The driver's own text follows, so nothing it said is lost.

The write is still sent exactly once. Whether a write is safe to send again
is the caller's judgement (docs/CONSTRAINTS.md §15.2); what changes here is
that the caller now has what it needs to make it.

The offline half builds the refusal, as ``test_loader`` does for the catalog
read, because provoking it needs a distributed engine. The live half needs
the YugabyteDB container and skips without it.
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
from localdata_mcp.loader import LoadError, Workspace

# ---------------------------------------------------------------------------
# Offline: the refusal's words, and that the write is sent once
# ---------------------------------------------------------------------------


def _refused(code: str) -> OperationalError:
    """A driver failure carrying a SQLSTATE, wrapped the way SQLAlchemy wraps one."""
    original = type("DriverFailure", (Exception,), {"sqlstate": code})(
        "could not serialize access due to concurrent update"
    )
    return OperationalError("DROP TABLE people", {}, original)


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


def _refusing(monkeypatch, owner: object, method: str, code: str) -> list[int]:
    """Make ``owner.method`` raise a driver failure with ``code``; count the calls."""
    calls: list[int] = []

    def refuse(*_args, **_kwargs):
        calls.append(1)
        raise _refused(code)

    monkeypatch.setattr(owner, method, refuse)
    return calls


def _writes(ws: Workspace) -> dict[str, Callable[[], object]]:
    return {
        "rename": lambda: ws.rename_table("main", "people", "staff"),
        "drop table": lambda: ws.drop_table("main", "people"),
        "create index": lambda: ws.create_index("main", "people", ["salary"]),
        "drop index": lambda: ws.drop_index("main", "ix_people_name"),
    }


#: Which call each write makes to the database, so the refusal can be put there.
_SENDS = {
    "rename": ("backend", "rename_table"),
    "drop table": (Table, "drop"),
    "create index": (Index, "create"),
    "drop index": (Index, "drop"),
}

#: What each write's refusal must say was left as it was.
_UNCHANGED = {
    "rename": "main.people still has its old name",
    "drop table": "main.people is still there",
    "create index": "main.people has no new index",
    "drop index": "main.ix_people_name is still there",
}


def _owner(ws: Workspace, write: str) -> object:
    owner, _ = _SENDS[write]
    return type(ws.entry("main").backend) if owner == "backend" else owner


@pytest.mark.parametrize("write", list(_SENDS))
def test_a_write_refused_with_40001_says_what_happened_and_what_to_do(
    workspace, monkeypatch, write
):
    calls = _refusing(monkeypatch, _owner(workspace, write), _SENDS[write][1], "40001")

    with pytest.raises(LoadError) as raised:
        _writes(workspace)[write]()

    message = str(raised.value)
    assert "another session" in message, message
    assert "40001" in message, message
    assert f"Nothing was changed: {_UNCHANGED[write]}." in message, message
    assert "Send the same call again" in message, message
    # The driver's own words still follow, so nothing it said is lost.
    assert "could not serialize access due to concurrent update" in message, message
    # Sent once. Resending a write is the caller's call, not this server's.
    assert len(calls) == 1


@pytest.mark.parametrize("write", list(_SENDS))
def test_a_write_refused_for_another_reason_keeps_its_old_words(
    workspace, monkeypatch, write
):
    calls = _refusing(monkeypatch, _owner(workspace, write), _SENDS[write][1], "42501")

    with pytest.raises(LoadError) as raised:
        _writes(workspace)[write]()

    message = str(raised.value)
    assert message.startswith("Could not "), message
    assert "another session" not in message, message
    assert "Send the same call again" not in message, message
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Offline: reading the table before writing to it
# ---------------------------------------------------------------------------
#
# ``create(type='index')`` reads the table's columns before it builds anything.
# That read used to answer every failure with "No such table", so a 40001 on it
# — measured on CockroachDB 26.3 with the engine's own retry-error injection on
# the server's write session — told the caller that a table it could see did
# not exist. The read is a catalog read, so it is run again once on 40001, as
# every catalog read is (§15.2); and a 40001 that survives that is reported as
# a read that failed, with the database's words, never as a missing table.


def _schema_refused(monkeypatch, codes: list[str]) -> list[int]:
    """Make the column read raise each of ``codes`` in turn, then read normally."""
    from localdata_mcp import loader as loader_module

    real = loader_module._schema_of
    calls: list[int] = []

    def read(entry, table):
        calls.append(1)
        if len(calls) <= len(codes):
            raise _refused(codes[len(calls) - 1])
        return real(entry, table)

    monkeypatch.setattr(loader_module, "_schema_of", read)
    return calls


def test_a_column_read_refused_once_with_40001_is_run_again_and_the_index_built(
    workspace, monkeypatch
):
    calls = _schema_refused(monkeypatch, ["40001"])

    made = workspace.create_index("main", "people", ["salary"])

    assert made.name == "ix_people_salary"
    assert len(calls) == 2


def test_a_column_read_refused_twice_with_40001_is_not_called_a_missing_table(
    workspace, monkeypatch
):
    calls = _schema_refused(monkeypatch, ["40001", "40001"])

    with pytest.raises(LoadError) as raised:
        workspace.create_index("main", "people", ["salary"])

    message = str(raised.value)
    assert "No such table" not in message, message
    assert message.startswith("Could not read the columns of main.people"), message
    assert "could not serialize access" in message, message
    assert len(calls) == 2


def test_a_column_read_that_fails_otherwise_still_says_no_such_table(
    workspace, monkeypatch
):
    calls = _schema_refused(monkeypatch, ["42P01"])

    with pytest.raises(LoadError) as raised:
        workspace.create_index("main", "people", ["salary"])

    assert str(raised.value) == "No such table: main.people"
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Live: a second connection, opened by the test, collides with the server's write
# ---------------------------------------------------------------------------


_YUGABYTE = [target for target in TARGETS if target.name == "yugabytedb"]

#: How long the other session keeps changing the table before the server's
#: write is sent, and between its own statements. Measured: 4 of 4 drops and
#: 2 of 2 index builds were refused at these timings on YugabyteDB 2.25.
_HEAD_START = 0.3
_BETWEEN = 0.02

#: Attempts before the test concludes the collision could not be produced. A
#: write that goes through on an attempt is still checked against the
#: datasource; the test fails only if no attempt was refused at all, because a
#: probe that never saw the case proves nothing about it.
_ATTEMPTS = 5


def _call(tool: str, **arguments):
    """The in-process protocol client, as ``test_endpoints`` has it."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


@contextmanager
def _changing(url: str, statement: str) -> Iterator[list[str]]:
    """Another session, not this server, running ``statement`` until the block ends."""
    stop = threading.Event()
    outcomes: list[str] = []
    engine = create_engine(url, isolation_level="AUTOCOMMIT")

    def loop() -> None:
        n = 0
        with engine.connect() as conn:
            while not stop.is_set():
                try:
                    conn.execute(text(statement.format(n=n)))
                    outcomes.append("ok")
                except Exception as exc:  # noqa: BLE001 - only counted
                    outcomes.append(type(exc).__name__)
                n += 1
                time.sleep(_BETWEEN)

    worker = threading.Thread(target=loop, daemon=True)
    worker.start()
    time.sleep(_HEAD_START)
    try:
        yield outcomes
    finally:
        stop.set()
        worker.join(30)
        engine.dispose()


@contextmanager
def _server(target, tmp_path) -> Iterator[tuple[str, object]]:
    """A writable slot over the target; yields its URL and an engine of the test's own."""
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


def _table(admin, rows: int = 2000) -> str:
    name = f"t{uuid.uuid4().hex[:8]}"
    with admin.begin() as conn:
        conn.execute(text(f"CREATE TABLE {name} (id INT PRIMARY KEY, salary INT)"))
        conn.execute(
            text(
                f"INSERT INTO {name} SELECT g, g % 97 FROM generate_series(1, {rows}) g"
            )
        )
    return name


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


def _indexes(admin, table: str) -> set[str]:
    with admin.connect() as conn:
        return {
            row[0]
            for row in conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = :t"),
                {"t": table},
            )
        }


def _drop_if_there(admin, table: str) -> None:
    with admin.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


@pytest.mark.endpoint
@pytest.mark.parametrize("target", _YUGABYTE, ids=lambda t: t.name)
def test_a_drop_refused_by_a_concurrent_change_left_the_table_and_goes_through_again(
    target, tmp_path
):
    refused = 0
    with _server(target, tmp_path) as (url, admin):
        for _ in range(_ATTEMPTS):
            table = _table(admin)
            try:
                with _changing(url, f"COMMENT ON TABLE {table} IS 'c{{n}}'") as other:
                    answer = _call(
                        "drop", nickname="endpoint", type="table", name=table
                    )
                    assert "ok" in other, other  # the other session really was writing
                if answer["ok"]:
                    # Truthful either way: a drop that went through dropped it.
                    assert not _exists(admin, table), answer
                    continue
                refused += 1
                error = answer["error"]
                assert "another session" in error and "40001" in error, error
                assert f"Nothing was changed: endpoint.{table} is still there." in error
                assert _exists(admin, table), answer

                again = _call("drop", nickname="endpoint", type="table", name=table)
                assert again["ok"] is True, again
                assert not _exists(admin, table), again
            finally:
                _drop_if_there(admin, table)
    assert refused > 0, (
        f"no drop of {_ATTEMPTS} was refused: the collision never happened"
    )


@pytest.mark.endpoint
@pytest.mark.parametrize("target", _YUGABYTE, ids=lambda t: t.name)
def test_an_index_refused_by_a_concurrent_change_was_not_built_and_builds_again(
    target, tmp_path
):
    refused = 0
    with _server(target, tmp_path) as (url, admin):
        for _ in range(_ATTEMPTS):
            table = _table(admin, rows=100_000)
            index = f"ix_{table}_salary"
            try:
                with _changing(
                    url, f"ALTER TABLE {table} ADD COLUMN z{{n}} INT"
                ) as other:
                    answer = _call(
                        "create",
                        nickname="endpoint",
                        type="index",
                        table=table,
                        columns=["salary"],
                    )
                    assert "ok" in other, other
                if answer["ok"]:
                    assert index in _indexes(admin, table), answer
                    continue
                refused += 1
                error = answer["error"]
                assert "another session" in error and "40001" in error, error
                assert (
                    f"Nothing was changed: endpoint.{table} has no new index." in error
                )
                assert index not in _indexes(admin, table), answer

                again = _call(
                    "create",
                    nickname="endpoint",
                    type="index",
                    table=table,
                    columns=["salary"],
                )
                assert again["ok"] is True, again
                assert index in _indexes(admin, table), again
            finally:
                _drop_if_there(admin, table)
    assert refused > 0, (
        f"no index of {_ATTEMPTS} was refused: the collision never happened"
    )
