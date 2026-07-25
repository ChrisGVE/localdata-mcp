"""Concurrency: prove the workspace does not lose rows under parallel tool calls.

This suite exists because of one measured failure. With a shared DBAPI connection
under a pool that rolls back on return, an unrelated reader checking the
connection out and closing it **discarded an in-flight load** — 79,807 of 200,000
rows gone, with no exception raised anywhere, and the writer's own
``in_transaction()`` reporting ``True`` throughout. ``docs/CONSTRAINTS.md`` §3.5.

The workspace runs in exactly that configuration, because an in-memory SQLite
database only exists for the life of its connection and ``StaticPool`` is
therefore obligatory. Two things are supposed to close the hole: the session
holds its raw connection for its whole life rather than checking one out per
operation, and the server serialises tool calls behind a lock. Neither is
self-evidently sufficient, so both are tested here rather than asserted in a
comment.

**The GIL does not make any of this safe, and it is worth being exact about
why.** It prevents two threads executing Python bytecode simultaneously, so
pure-Python CPU work gets no parallel speedup — measured here at ~2x for two
threads. But it is released during I/O and inside C extensions, so two SQLite
scans on *separate* connections run in 1.02x the time of one: genuinely
parallel. On a *shared* connection they take 2.14x, serialised — by SQLite's own
per-connection mutex, not by the GIL.

Neither mechanism prevents **interleaving at transaction granularity**, and that
is what destroys data. A writer holding an open transaction, a reader running one
statement and calling ``rollback()`` on the same connection — which is exactly
what a pool does on check-in — loses the writer's work entirely: 1,000 rows
inserted, **0 landed**, no exception. Reproduced in
``test_an_unrelated_rollback_destroys_an_in_flight_transaction`` below.

So the hazard needs no parallelism, only a thread switch, and threads are real
here whatever the GIL is doing. The corollary is that the lock is close to free:
a shared connection serialises statements anyway, so serialising them ourselves
costs throughput we never had, and buys transaction-level safety we otherwise
lack.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config

LOAD_ROWS = 40_000


@pytest.fixture(autouse=True)
def session(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,)))
    server_module._reset()
    yield root
    server_module._reset()


def _write_csv(path: Path, rows: int, start: int = 0) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("id,payload\n")
        for index in range(start, start + rows):
            handle.write(f"{index},value_{index}\n")


def _payload(result):
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def test_tool_dispatch_is_genuinely_parallel():
    """Guard the guard: prove the tests below are exercising real concurrency.

    Every other test in this file is meaningless if the framework happens to
    serialise tool calls — they would pass trivially, and a regression in the
    workspace's own locking would sail through. So assert the premise directly:
    two blocking tool bodies must overlap, on different OS threads.

    Measured here at 0.504 s of overlap across two threads, 0.52 s wall against
    1.0 s if serialised.
    """
    import threading
    import time

    probe = FastMCP("dispatch-probe")
    events: list[tuple[str, int, float]] = []

    @probe.tool
    def block(tag: str) -> str:
        """Occupy a worker for a fixed interval."""
        events.append(("start", threading.get_ident(), time.perf_counter()))
        time.sleep(0.4)
        events.append(("end", threading.get_ident(), time.perf_counter()))
        return tag

    async def _run():
        async with Client(probe) as client:
            await asyncio.gather(
                client.call_tool("block", {"tag": "a"}),
                client.call_tool("block", {"tag": "b"}),
            )

    asyncio.run(_run())

    threads = {ident for _, ident, _ in events}
    latest_start = max(t for kind, _, t in events if kind == "start")
    earliest_end = min(t for kind, _, t in events if kind == "end")

    assert len(threads) == 2, f"tool bodies shared a thread: {threads}"
    assert earliest_end - latest_start > 0.1, (
        "tool bodies did not overlap — dispatch is serialised, so every "
        "concurrency test in this file is passing vacuously."
    )


def test_an_unrelated_rollback_destroys_an_in_flight_transaction():
    """The hazard itself, isolated — the reason the lock above exists.

    Not a test of our code: a demonstration that the danger is real in this
    interpreter, so that removing the lock is never mistaken for free. A
    connection has exactly one transaction, so a second user of that connection
    rolling back discards the first's uncommitted work — silently, with no
    exception on either side.
    """
    import sqlite3
    import threading

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.execute("CREATE TABLE t (a INTEGER)")
    connection.commit()

    writing = threading.Event()
    reader_done = threading.Event()

    def writer():
        connection.execute("BEGIN")
        for value in range(1_000):
            connection.execute("INSERT INTO t VALUES (?)", (value,))
        writing.set()
        reader_done.wait(timeout=5)
        try:
            connection.commit()
        except sqlite3.Error:
            pass

    def reader():
        writing.wait(timeout=5)
        connection.execute("SELECT count(*) FROM t").fetchone()
        connection.rollback()  # what a pool does when a checkout is returned
        reader_done.set()

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    landed = connection.execute("SELECT count(*) FROM t").fetchone()[0]
    connection.close()

    assert landed == 0, (
        "Expected the reader's rollback to discard all 1,000 uncommitted rows. "
        f"{landed} survived — if this now passes, sqlite3's transaction "
        "semantics changed and the workspace's locking should be revisited."
    )


def test_readers_running_against_a_load_do_not_lose_rows(session):
    """The measured failure, reproduced as a test: readers during a write.

    A load of 40,000 rows runs while eight read calls hammer the same workspace.
    Every row must land. Under the unmitigated configuration this is the shape
    that silently dropped 40% of them.
    """
    big = session / "big.csv"
    _write_csv(big, LOAD_ROWS)
    seed = session / "seed.csv"
    _write_csv(seed, 10)

    async def _run():
        async with Client(server_module.mcp) as client:
            # A table for the readers to hit, so they are doing real work
            # rather than erroring immediately on a missing table.
            await client.call_tool(
                "attach", {"database": str(seed), "nickname": "seed"}
            )

            tasks = [
                client.call_tool(
                    "attach", {"database": str(big), "nickname": "big"}
                )
            ]
            tasks += [
                client.call_tool(
                    "query",
                    {"nickname": "seed", "sql": "SELECT count(*) FROM seed.seed"},
                )
                for _ in range(8)
            ]
            results = await asyncio.gather(*tasks)

            verify = await client.call_tool(
                "query",
                {"nickname": "big", "sql": "SELECT count(*) FROM big.big", "limit": 0},
            )
            return [_payload(r) for r in results], _payload(verify)

    results, verify = asyncio.run(_run())

    load_result = results[0]
    assert load_result["ok"] is True, load_result
    assert load_result["loaded"][0]["rows"] == LOAD_ROWS

    # Every reader answered, and none of them corrupted the load.
    for reader in results[1:]:
        assert reader["ok"] is True, reader
        assert reader["rows"][0][0] == 10

    # The authoritative check: count from the database, not from the load report.
    assert verify["rows"][0][0] == LOAD_ROWS


def test_parallel_loads_all_land(session):
    """Concurrent writes to distinct tables must not drop each other's rows."""
    files = []
    for index in range(4):
        path = session / f"part{index}.csv"
        _write_csv(path, 5_000, start=index * 5_000)
        files.append((path, f"part{index}"))

    async def _run():
        async with Client(server_module.mcp) as client:
            results = await asyncio.gather(
                *[
                    client.call_tool(
                        "attach",
                        {"database": str(path), "nickname": table},
                    )
                    for path, table in files
                ]
            )
            counts = await asyncio.gather(
                *[
                    client.call_tool(
                        "query",
                        {
                            "nickname": table,
                            "sql": f"SELECT count(*) FROM {table}.{table}",
                            "limit": 0,
                        },
                    )
                    for _, table in files
                ]
            )
            return [_payload(r) for r in results], [_payload(c) for c in counts]

    results, counts = asyncio.run(_run())

    for result in results:
        assert result["ok"] is True, result
        assert result["loaded"][0]["rows"] == 5_000

    for count in counts:
        assert count["rows"][0][0] == 5_000

    # And the union is intact — no table clobbered another's contents.
    async def _union():
        async with Client(server_module.mcp) as client:
            return _payload(
                await client.call_tool(
                    "query",
                    {
                        "nickname": "part0",
                        # Four slots in one statement: separate databases on one
                        # connection, so this is a plain query, not four trips.
                        "sql": (
                            "SELECT count(DISTINCT id) FROM ("
                            "SELECT id FROM part0.part0 "
                            "UNION ALL SELECT id FROM part1.part1 "
                            "UNION ALL SELECT id FROM part2.part2 "
                            "UNION ALL SELECT id FROM part3.part3)"
                        ),
                        "limit": 0,
                    },
                )
            )

    assert asyncio.run(_union())["rows"][0][0] == 20_000


def test_a_failing_load_does_not_damage_an_existing_table(session):
    """A rollback must be scoped to the load that failed."""
    good = session / "good.csv"
    _write_csv(good, 1_000)
    missing = session / "absent.csv"

    async def _run():
        async with Client(server_module.mcp) as client:
            await client.call_tool(
                "attach", {"database": str(good), "nickname": "good"}
            )
            failures = await asyncio.gather(
                *[
                    client.call_tool(
                        "attach",
                        {"database": str(missing), "nickname": "bad"},
                    )
                    for _ in range(4)
                ]
            )
            survivor = await client.call_tool(
                "query",
                {
                    "nickname": "good",
                    "sql": "SELECT count(*) FROM good.good",
                    "limit": 0,
                },
            )
            return [_payload(f) for f in failures], _payload(survivor)

    failures, survivor = asyncio.run(_run())

    for failure in failures:
        assert failure["ok"] is False
        assert "No such file" in failure["error"]

    assert survivor["rows"][0][0] == 1_000
