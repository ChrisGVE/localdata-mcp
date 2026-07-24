"""tests/v3/test_execution_adapter.py — E6.1's wire over the native shape.

The SQLAlchemy shape is exercised end to end by test_guard.py; this
file drives the OTHER `EngineHandle` connection shape (DuckDB's native
connection — `execute` returning a cursor-like self with `description`
and `fetchmany`, no `exec_driver_sql`) through the same adapter with a
faithful fake, since importing `duckdb` outside nexus/persistence is
exactly what FR-105 forbids a test to normalize.
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.chokepoint.execution import (
    execute_mutation,
    fetch_bounded,
    iter_frames,
)
from localdata_mcp.nexus.chokepoint.resource_bounds import (
    ResourceBounds,
    ResourceRefusedError,
)
from localdata_mcp.nexus.config.models import ConfigModel, ResourcesConfig


class NativeConnection:
    """The DuckDB-native surface the adapter relies on: `execute`
    returns the cursor-like connection itself; `description` names the
    columns; `fetchmany` pages; no `exec_driver_sql` attribute."""

    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows
        self._cursor = 0
        self.seen_sql: str | None = None
        self.seen_parameters: dict[str, object] | None = None
        self.description = [("id", None), ("label", None)]

    def execute(
        self, sql: str, parameters: dict[str, object] | None = None
    ) -> "NativeConnection":
        self.seen_sql = sql
        self.seen_parameters = parameters
        self._cursor = 0
        return self

    def fetchmany(self, size: int) -> list[tuple[object, ...]]:
        batch = self._rows[self._cursor : self._cursor + size]
        self._cursor += size
        return batch


def test_fetch_bounded_over_the_native_shape() -> None:
    connection = NativeConnection([(1, "a"), (2, "b"), (3, "c")])
    columns, rows = fetch_bounded(
        connection,
        "SELECT * FROM t",
        {"p": 1},
        ResourceBounds(ConfigModel()),
        batch_size=2,
        registry_id="q-1",
    )
    assert columns == ("id", "label")
    assert rows == ((1, "a"), (2, "b"), (3, "c"))
    assert connection.seen_parameters == {"p": 1}


def test_execute_mutation_over_the_native_shape_reports_no_count() -> None:
    """The native shape exposes no reliable rowcount — the adapter
    answers None, never a guess, and never calls a commit the
    autocommitting connection does not need."""
    connection = NativeConnection([])
    assert execute_mutation(connection, "INSERT ...", None) is None


def test_iter_frames_over_the_native_shape() -> None:
    connection = NativeConnection([(1, "a"), (2, "b"), (3, "c")])
    frames = list(iter_frames(connection, "SELECT * FROM t", None, batch_size=2))
    assert [len(frame) for frame in frames] == [2, 1]
    assert list(frames[0].columns) == ["id", "label"]
    assert frames[1]["label"].iloc[0] == "c"


def test_native_parameters_pass_none_when_absent() -> None:
    connection = NativeConnection([])
    fetch_bounded(
        connection,
        "SELECT 1",
        None,
        ResourceBounds(ConfigModel()),
        batch_size=2,
        registry_id="q-2",
    )
    assert connection.seen_parameters is None


def _bounds(ceiling: int) -> ResourceBounds:
    return ResourceBounds(
        ConfigModel(resources=ResourcesConfig(memory_ceiling_bytes=ceiling))
    )


def test_fetch_bounded_charges_and_holds_the_retained_working_set() -> None:
    """CR-007: the retained analytical result is charged to the aggregate
    residency ledger under its registry_id and HELD, so a concurrent
    bounded query can see it. Before the fix nothing charged the ledger
    for a completed bounded query, so live_residency stayed at zero."""
    bounds = _bounds(1_000_000)
    connection = NativeConnection([(i, "x" * 100) for i in range(40)])
    fetch_bounded(
        connection,
        "SELECT * FROM t",
        None,
        bounds,
        batch_size=100,
        registry_id="analysis-A",
    )
    assert bounds.live_residency() > 0


def test_concurrent_bounded_queries_jointly_refuse() -> None:
    """CR-007: two concurrent bounded queries whose combined working set
    exceeds the ceiling get a fail-safe refusal. Query A holds its
    working set on the ledger (as guarded_query holds it across its
    lifetime); query B's fetch then refuses rather than jointly blowing
    the ceiling — the residency the headroom check reads now learns
    about finished/in-flight bounded results."""
    bounds = _bounds(1_000_000)
    bounds.charge("analysis-A", 950_000)  # query A's held working set
    connection = NativeConnection([(i, "x" * 3000) for i in range(60)])
    with pytest.raises(ResourceRefusedError):
        fetch_bounded(
            connection,
            "SELECT * FROM t",
            None,
            bounds,
            batch_size=100,
            registry_id="analysis-B",
        )
