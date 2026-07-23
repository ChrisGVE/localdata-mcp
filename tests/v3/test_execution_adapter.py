"""tests/v3/test_execution_adapter.py — E6.1's wire over the native shape.

The SQLAlchemy shape is exercised end to end by test_guard.py; this
file drives the OTHER `EngineHandle` connection shape (DuckDB's native
connection — `execute` returning a cursor-like self with `description`
and `fetchmany`, no `exec_driver_sql`) through the same adapter with a
faithful fake, since importing `duckdb` outside nexus/persistence is
exactly what FR-105 forbids a test to normalize.
"""

from __future__ import annotations

from localdata_mcp.nexus.chokepoint.execution import (
    execute_mutation,
    fetch_bounded,
    iter_frames,
)
from localdata_mcp.nexus.chokepoint.resource_bounds import ResourceBounds
from localdata_mcp.nexus.config.models import ConfigModel


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
        connection, "SELECT 1", None, ResourceBounds(ConfigModel()), batch_size=2
    )
    assert connection.seen_parameters is None
