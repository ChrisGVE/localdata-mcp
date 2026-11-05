"""A failed ``create(type='table')`` that left its table behind says so (#147).

``insert_source`` makes the table and fills it in one transaction wherever the
backend sees a new table inside the transaction that made it, on the premise
that a failed insert takes the ``CREATE`` with it. Measured, the premise does
not hold everywhere:

* **SQLite** (2026-11-05, SQLite 3.47.1 through pysqlite, SQLAlchemy 2.1.6):
  the driver opens a transaction only before DML, so the ``CREATE TABLE`` runs
  on its own and is committed before the insert fails. A file whose third row
  holds ``18446744073709551615`` left the table there, empty.
* **YugabyteDB** 2.25: the ``CREATE TABLE`` survives the aborted transaction,
  for the same bad value or for a collision with another session (SQLSTATE
  40001) — the issue's two triggers.
* **CockroachDB** 26.3.4: the same bad value leaves the same empty table.
* **DuckDB**: the table is gone, as the premise says.

The caller used to hear only why the insert failed, so the same call sent again
was refused with "already exists, holding 0 rows". The refusal now reads back,
on a fresh connection, whether the table is there; if it is, it names the table
and its row count and says to drop it before creating it again. Nothing is
dropped by this server: another session can write into the table while it is
being filled (measured on YugabyteDB), so a compensating drop could destroy rows
that are not this server's.

The offline half uses SQLite and DuckDB files, the one engine that leaves the
table and the one that does not. The live half needs the YugabyteDB or the
CockroachDB container and skips without it.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import create_engine, text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.loader import Workspace

#: The largest unsigned 64-bit value: pandas reads it as ``uint64`` and no
#: signed 64-bit column holds it, so the insert is refused with no other
#: session involved — the trigger the issue measured without a second session.
TOO_WIDE = "18446744073709551615"

#: The row the bad value sits on, as in the issue: late enough that rows before
#: it have already been sent when the insert fails.
BAD_ROW = 5001


def call(tool: str, **arguments):
    """Call a tool over the protocol and return its decoded payload."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


def _failing_file(root: Path, bad_row: int = BAD_ROW) -> Path:
    """A CSV whose row ``bad_row`` holds a value no 64-bit column can store."""
    path = root / "failing.csv"
    good = "".join(f"{i},{i}\n" for i in range(1, bad_row))
    path.write_text(f"id,v\n{good}{bad_row},{TOO_WIDE}\n")
    return path


def _good_file(root: Path) -> Path:
    path = root / "good.csv"
    path.write_text("id,v\n1,1\n2,2\n3,3\n")
    return path


# ---------------------------------------------------------------------------
# Offline: SQLite leaves the table, DuckDB does not
# ---------------------------------------------------------------------------


@pytest.fixture()
def root(tmp_path):
    """An allowed root and a fresh registry for every test."""
    allowed = (tmp_path / "root").resolve()
    allowed.mkdir()
    config_module.use(Config(roots=(allowed,)))
    server_module._reset()
    yield allowed
    server_module._reset()


def _sqlite(root: Path) -> Path:
    import sqlite3

    path = root / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE seed (a INTEGER)")
    conn.commit()
    conn.close()
    return path


def _duckdb(root: Path) -> Path:
    path = root / "shop.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE seed (a INTEGER)")
    conn.close()
    return path


def _tables(nickname: str) -> list[str]:
    listed = call("directory", nickname=nickname)
    assert listed["ok"] is True, listed
    return listed["tables"]


def test_a_failed_create_on_sqlite_says_it_left_the_table_and_how_to_clear_it(root):
    attached = call(
        "attach", database=str(_sqlite(root)), nickname="shop", writable=True
    )
    assert attached["ok"] is True, attached

    answer = call(
        "create",
        nickname="shop",
        type="table",
        source=str(_failing_file(root)),
        delimiter=",",
        table="t",
    )

    assert answer["ok"] is False, answer
    error = answer["error"]
    # Why it failed comes first, unchanged.
    assert "cannot be stored in a 64-bit column without corrupting it." in error
    # Then what it left, with the count read back and the call that clears it.
    assert (
        "The failed create left shop.t behind, holding 0 rows: drop it with "
        "drop(nickname='shop', type='table', name='t') before creating it again."
    ) in error, error
    # The datasource matches the answer.
    assert "t" in _tables("shop")

    # The next step the refusal gives works: drop, then create.
    dropped = call("drop", nickname="shop", type="table", name="t")
    assert dropped["ok"] is True, dropped
    made = call(
        "create",
        nickname="shop",
        type="table",
        source=str(_good_file(root)),
        delimiter=",",
        table="t",
    )
    assert made["ok"] is True, made
    assert made["rows"] == 3, made


def test_a_failed_create_that_left_nothing_does_not_claim_a_table(root):
    attached = call(
        "attach", database=str(_duckdb(root)), nickname="shop", writable=True
    )
    assert attached["ok"] is True, attached

    answer = call(
        "create",
        nickname="shop",
        type="table",
        source=str(_failing_file(root)),
        delimiter=",",
        table="t",
    )

    assert answer["ok"] is False, answer
    assert "cannot be stored in a 64-bit column" in answer["error"], answer
    assert "behind" not in answer["error"], answer
    # The probe saw the case it names: DuckDB really did leave nothing.
    assert _tables("shop") == ["seed"]


def test_a_read_back_that_fails_says_the_table_could_not_be_checked(root, monkeypatch):
    attached = call(
        "attach", database=str(_sqlite(root)), nickname="shop", writable=True
    )
    assert attached["ok"] is True, attached

    real = Workspace.resolve_table
    asked: list[str] = []

    def unreachable(self, tag, table):
        # Only the read-back after the failure is made to fail; the check
        # create makes before it starts goes through, so the insert is reached.
        asked.append(table)
        if len(asked) > 1:
            raise OSError("connection reset by peer")
        return real(self, tag, table)

    monkeypatch.setattr(Workspace, "resolve_table", unreachable)

    answer = call(
        "create",
        nickname="shop",
        type="table",
        source=str(_failing_file(root, bad_row=3)),
        delimiter=",",
        table="t",
    )

    assert answer["ok"] is False, answer
    error = answer["error"]
    assert "cannot be stored in a 64-bit column without corrupting it." in error
    assert (
        "Whether the failed create left shop.t behind could not be checked "
        "(connection reset by peer); directory(nickname='shop') lists what "
        "shop holds."
    ) in error, error
    assert len(asked) == 2, asked


def test_a_count_that_fails_still_names_the_table_left_behind(root, monkeypatch):
    attached = call(
        "attach", database=str(_sqlite(root)), nickname="shop", writable=True
    )
    assert attached["ok"] is True, attached

    def refused(self, tag, table, source=""):
        raise OSError("could not serialize access")

    monkeypatch.setattr(Workspace, "describe", refused)

    answer = call(
        "create",
        nickname="shop",
        type="table",
        source=str(_failing_file(root, bad_row=3)),
        delimiter=",",
        table="t",
    )

    assert answer["ok"] is False, answer
    assert (
        "The failed create left shop.t behind, holding rows that could not be "
        "counted just now: drop it with drop(nickname='shop', type='table', "
        "name='t') before creating it again."
    ) in answer["error"], answer


# ---------------------------------------------------------------------------
# Live: YugabyteDB and CockroachDB, read back on a connection of the test's own
# ---------------------------------------------------------------------------

#: Whether each engine leaves the table behind after this failure, as measured
#: on 2026-11-05 (docs/CONSTRAINTS.md §15, amendment of that date). Pinned, so a
#: run that stops producing the case fails instead of passing over it.
_LEAVES_IT = {"yugabytedb": True, "cockroachdb": True}

_LIVE = [target for target in TARGETS if target.name in _LEAVES_IT]


@contextmanager
def _server(target, tmp_path) -> Iterator[tuple[Path, object]]:
    """A writable slot over the target; yields the allowed root and an engine."""
    try:
        reached = reach(target)
    except Unavailable as exc:
        pytest.skip(str(exc))
    allowed = (tmp_path / "root").resolve()
    allowed.mkdir(exist_ok=True)
    config_module.use(Config(roots=(allowed,), network_enabled=True))
    server_module._reset()
    admin = create_engine(reached.url)
    try:
        with applied(reached.environ):
            attached = _call_attach(reached.url)
            assert attached["ok"] is True, attached
            yield allowed, admin
    finally:
        server_module._reset()
        admin.dispose()


def _call_attach(url: str):
    return call("attach", database=url, nickname="endpoint", writable=True)


def _rows_if_there(admin, table: str) -> int | None:
    """The table's row count read by the test's own session, or None if absent."""
    with admin.connect() as conn:
        present = conn.execute(
            text(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = :t"
            ),
            {"t": table},
        ).scalar()
        if not present:
            return None
        return int(conn.execute(text(f"SELECT count(*) FROM {table}")).scalar())


@pytest.mark.endpoint
@pytest.mark.parametrize("target", _LIVE, ids=lambda t: t.name)
def test_a_failed_create_reports_the_table_it_left_on_a_distributed_engine(
    target, tmp_path
):
    table = f"t{uuid.uuid4().hex[:8]}"
    with _server(target, tmp_path) as (allowed, admin):
        try:
            answer = call(
                "create",
                nickname="endpoint",
                type="table",
                source=str(_failing_file(allowed)),
                delimiter=",",
                table=table,
            )
            assert answer["ok"] is False, answer
            error = answer["error"]
            assert "cannot be stored in a 64-bit column" in error, error

            left = _rows_if_there(admin, table)
            assert (left is not None) == _LEAVES_IT[target.name], (left, error)
            if left is None:
                assert "behind" not in error, error
                return
            assert (
                f"The failed create left endpoint.{table} behind, holding "
                f"{left} rows: drop it with drop(nickname='endpoint', "
                f"type='table', name='{table}') before creating it again."
            ) in error, error

            dropped = call("drop", nickname="endpoint", type="table", name=table)
            assert dropped["ok"] is True, dropped
            assert _rows_if_there(admin, table) is None
        finally:
            with admin.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


#: The issue's other trigger: a second session alters the new table as soon as
#: it appears, and YugabyteDB refuses the insert with 40001 (schema version
#: mismatch). Racy by nature, so tried a few times; the test fails only if no
#: attempt was refused, because a probe that never saw the case proves nothing.
_ATTEMPTS = 5
_ROWS = 50_000


def _altering_when_it_appears(
    url: str, table: str
) -> tuple[threading.Event, list[str]]:
    """Another session that adds a column to ``table`` the moment it exists."""
    stop = threading.Event()
    outcomes: list[str] = []
    engine = create_engine(url, isolation_level="AUTOCOMMIT")

    def loop() -> None:
        n = 0
        with engine.connect() as conn:
            while not stop.is_set():
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN z{n} INT"))
                    outcomes.append("ok")
                    n += 1
                except Exception as exc:  # noqa: BLE001 - only counted
                    outcomes.append(type(exc).__name__)
                time.sleep(0.01)
        engine.dispose()

    threading.Thread(target=loop, daemon=True).start()
    return stop, outcomes


@pytest.mark.endpoint
@pytest.mark.parametrize(
    "target",
    [target for target in TARGETS if target.name == "yugabytedb"],
    ids=lambda t: t.name,
)
def test_a_create_refused_by_a_concurrent_change_reports_the_table_it_left(
    target, tmp_path
):
    refused = 0
    with _server(target, tmp_path) as (allowed, admin):
        big = allowed / "big.csv"
        big.write_text("id,v\n" + "".join(f"{i},{i}\n" for i in range(_ROWS)))
        url = admin.url.render_as_string(hide_password=False)
        for _ in range(_ATTEMPTS):
            table = f"t{uuid.uuid4().hex[:8]}"
            stop, other = _altering_when_it_appears(url, table)
            try:
                answer = call(
                    "create",
                    nickname="endpoint",
                    type="table",
                    source=str(big),
                    delimiter=",",
                    table=table,
                )
                stop.set()
                if answer["ok"]:
                    continue
                refused += 1
                error = answer["error"]
                assert "Could not insert rows" in error, error
                left = _rows_if_there(admin, table)
                assert left is not None, error  # the case the issue measured
                # The count is read while the other session may still be
                # altering the table, and YugabyteDB can refuse it with 40001
                # (measured); the table is named either way.
                held = (
                    f"{left} rows"
                    if f"holding {left} rows:" in error
                    else "rows that could not be counted just now"
                )
                assert (
                    f"The failed create left endpoint.{table} behind, holding "
                    f"{held}: drop it with drop(nickname='endpoint', "
                    f"type='table', name='{table}') before creating it again."
                ) in error, error
                assert "ok" in other, other  # the other session really altered it
            finally:
                stop.set()
                time.sleep(0.1)
                with admin.begin() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    assert refused > 0, (
        f"no create of {_ATTEMPTS} was refused: the collision never happened"
    )
