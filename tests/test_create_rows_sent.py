"""``rows`` on a successful create is what the call sent, not a read-back.

``insert_source`` used to answer ``rows`` with a ``COUNT(*)`` read after the
commit. On YugabyteDB the new table was visible to another session while the
create was filling it; that session inserted 462 rows, and a 50 000-row create
answered ``rows: 50462`` (issue #147, measured 2026-11-05).

``rows`` now reports the rows this call sent. The read-back stays: it is the
only check that every row sent landed, and when the table holds a different
number the answer says both, in a warning, and gives the query that counts the
table as it is now. ``directory`` is unchanged: it still describes the table,
so its ``rows`` is the count read when the create finished.

The offline half uses a SQLite file. pysqlite runs the ``CREATE TABLE`` on its
own and opens the write transaction only at the first insert, so a second
connection can write into the new table before the fill starts. That is the
window the hook below uses. The live half drives the issue's own case on
YugabyteDB and skips without the container.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest
from endpoints import TARGETS
from sqlalchemy import create_engine, text
from test_create_left_behind import _server, call

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.loader import Workspace

#: Rows in the file the offline creates read.
_SENT = 1000


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
    path = root / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE seed (a INTEGER)")
    conn.commit()
    conn.close()
    return path


def _file(root: Path, rows: int = _SENT) -> Path:
    path = root / "sales.csv"
    path.write_text("id,v\n" + "".join(f"{i},{i}\n" for i in range(rows)))
    return path


def _attached(root: Path) -> Path:
    database = _sqlite(root)
    attached = call("attach", database=str(database), nickname="shop", writable=True)
    assert attached["ok"] is True, attached
    return database


def _count(nickname: str, table: str) -> int:
    counted = call("query", nickname=nickname, sql=f"SELECT COUNT(*) FROM {table}")
    assert counted["ok"] is True, counted
    return counted["rows"][0][0]


def test_rows_is_what_the_create_sent_when_another_session_wrote_into_the_table(
    root, monkeypatch
):
    database = _attached(root)
    other_rows = 7
    real = Workspace._blocks
    wrote: list[int] = []

    def another_session_first(frame, columns, key=None, base=0):
        # Once, before the first block goes in: by then the CREATE TABLE has
        # committed and the write transaction has not begun, so a second
        # connection can insert into the new table.
        if not wrote:
            other = sqlite3.connect(database)
            other.executemany(
                "INSERT INTO t (id, v) VALUES (?, ?)",
                [(-n, -n) for n in range(1, other_rows + 1)],
            )
            other.commit()
            other.close()
            wrote.append(other_rows)
        yield from real(frame, columns, key, base)

    monkeypatch.setattr(Workspace, "_blocks", staticmethod(another_session_first))

    answer = call(
        "create",
        nickname="shop",
        type="table",
        source=str(_file(root)),
        delimiter=",",
        table="t",
    )

    assert answer["ok"] is True, answer
    assert wrote == [other_rows]  # the other session really wrote
    assert answer["rows"] == _SENT, answer
    assert answer["warnings"] == [
        f"This create sent {_SENT} rows into t, and the table now holds "
        f"{_SENT + other_rows}: another session wrote {other_rows} rows into it "
        f"after this create made it. rows is what this create sent; "
        f"query(nickname='shop', sql='SELECT COUNT(*) FROM t') counts what the "
        f"table holds now."
    ], answer

    # The query the warning gives counts what the table holds.
    assert _count("shop", "t") == _SENT + other_rows
    # directory still describes the table, and does not repeat a warning that
    # is about one create call.
    detail = call("directory", nickname="shop", table="t")
    assert detail["ok"] is True, detail
    assert detail["rows"] == _SENT + other_rows, detail
    assert "warnings" not in detail, detail


def test_rows_is_what_the_create_sent_when_the_table_holds_fewer(root, monkeypatch):
    database = _attached(root)
    removed = 10
    real = Workspace._count

    def another_session_deletes_first(entry, table):
        # Between the commit and the read-back, another session removes rows.
        other = sqlite3.connect(database)
        other.execute(f"DELETE FROM t WHERE id < {removed}")
        other.commit()
        other.close()
        return real(entry, table)

    monkeypatch.setattr(
        Workspace, "_count", staticmethod(another_session_deletes_first)
    )

    answer = call(
        "create",
        nickname="shop",
        type="table",
        source=str(_file(root)),
        delimiter=",",
        table="t",
    )

    assert answer["ok"] is True, answer
    assert answer["rows"] == _SENT, answer
    assert answer["warnings"] == [
        f"This create sent {_SENT} rows into t, and the table now holds "
        f"{_SENT - removed}, {removed} fewer than it sent. rows is what this "
        f"create sent; query(nickname='shop', sql='SELECT COUNT(*) FROM t') "
        f"counts what the table holds now."
    ], answer
    monkeypatch.undo()
    assert _count("shop", "t") == _SENT - removed


def test_a_create_whose_table_holds_what_it_sent_says_nothing_about_it(root):
    _attached(root)

    answer = call(
        "create",
        nickname="shop",
        type="table",
        source=str(_file(root)),
        delimiter=",",
        table="t",
    )

    assert answer["ok"] is True, answer
    assert answer["rows"] == _SENT, answer
    assert "warnings" not in answer, answer


# ---------------------------------------------------------------------------
# Live: the issue's case on YugabyteDB
# ---------------------------------------------------------------------------

#: The issue's size. Racy by nature, so tried a few times; the test fails only
#: if no attempt saw the other session's rows, because a probe that never saw
#: the case proves nothing.
_ROWS = 50_000
_ATTEMPTS = 5


def _inserting_when_it_appears(
    url: str, table: str
) -> tuple[threading.Event, list[str]]:
    """Another session inserting one row at a time into ``table`` once it exists."""
    stop = threading.Event()
    outcomes: list[str] = []
    engine = create_engine(url, isolation_level="AUTOCOMMIT")

    def loop() -> None:
        with engine.connect() as conn:
            while not stop.is_set():
                try:
                    conn.execute(text(f"INSERT INTO {table} (id, v) VALUES (-1, -1)"))
                    outcomes.append("ok")
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
def test_rows_is_what_the_create_sent_on_yugabytedb_with_another_session_writing(
    target, tmp_path
):
    seen: list[tuple[int, int]] = []
    with _server(target, tmp_path) as (allowed, admin):
        big = allowed / "big.csv"
        big.write_text("id,v\n" + "".join(f"{i},{i}\n" for i in range(_ROWS)))
        url = admin.url.render_as_string(hide_password=False)
        for _ in range(_ATTEMPTS):
            table = f"t{uuid.uuid4().hex[:8]}"
            stop, other = _inserting_when_it_appears(url, table)
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
                time.sleep(0.2)
                if not answer["ok"]:
                    # A collision refused the create; #147's path, not this one.
                    continue
                assert answer["rows"] == _ROWS, answer
                warnings = answer.get("warnings", [])
                stated = [w for w in warnings if w.startswith("This create sent")]
                if not stated:
                    continue
                held = int(stated[0].split("the table now holds ")[1].split(":")[0])
                extra = held - _ROWS
                assert extra > 0, stated
                assert stated == [
                    f"This create sent {_ROWS} rows into {table}, and the table "
                    f"now holds {held}: another session wrote {extra} rows into "
                    f"it after this create made it. rows is what this create "
                    f"sent; query(nickname='endpoint', sql='SELECT COUNT(*) FROM "
                    f"{table}') counts what the table holds now."
                ], warnings
                # The other session's committed rows are all in the table now.
                final = _count("endpoint", table)
                assert final == _ROWS + other.count("ok"), (final, other.count("ok"))
                seen.append((held, final))
                break
            finally:
                stop.set()
                time.sleep(0.1)
                with admin.begin() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    assert seen, f"no create of {_ATTEMPTS} saw another session's rows"
    print(f"yugabytedb: sent {_ROWS}, held at answer and now: {seen}")
