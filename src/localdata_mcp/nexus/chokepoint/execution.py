"""localdata_mcp/nexus/chokepoint/execution.py — the guarded wire (E6.1 half).

The one place a validated statement meets a live NX-5 connection: both
`EngineHandle` connection shapes (a SQLAlchemy `Connection`, DuckDB's
native connection — told apart by SQLAlchemy's own `exec_driver_sql`
attribute, never by importing `duckdb`, which FR-105 confines to
nexus/persistence) execute through here, and rows come back as plain
tuples — the capability-narrow material guard.py freezes into a
`Result` (GP3: no cursor, no connection, nothing live escapes).
Retention is admission-gated: `fetch_bounded` measures the first batch,
then re-admits through the E6.5 dynamic gate BEFORE retaining each
further batch, so an over-cap or over-headroom result refuses mid-fetch
instead of materializing (S8 row 13 applied on the wire). It also
CHARGES the retained working set to the aggregate residency ledger
under a caller-owned `registry_id` (CR-007), so concurrent bounded
queries are visible to each other's headroom check; the caller releases
that id once the result is handed off.
`iter_frames` is the streaming counterpart: the same fetch loop as a
pull source of DataFrames for the E6.6 ChunkRegistry. Neighbors:
guard.py is the only caller; resource_bounds.py rules on every
retention step.
"""

from __future__ import annotations

import sys
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd
from sqlalchemy import text

from .resource_bounds import ResourceBounds


def _is_sqlalchemy(connection: Any) -> bool:
    """SQLAlchemy `Connection` carries `exec_driver_sql`; the native
    DuckDB connection does not — a structural probe, not an import."""
    return hasattr(connection, "exec_driver_sql")


def _execute(connection: Any, sql: str, parameters: Mapping[str, Any] | None) -> Any:
    """Run `sql` with bound parameters, returning the cursor-like
    result the shape-specific fetch helpers below read from."""
    if _is_sqlalchemy(connection):
        return connection.execute(text(sql), dict(parameters or {}))
    return connection.execute(sql, dict(parameters) if parameters else None)


def _columns_of(cursor: Any) -> tuple[str, ...]:
    if hasattr(cursor, "keys"):  # SQLAlchemy CursorResult
        return tuple(str(key) for key in cursor.keys())
    description = getattr(cursor, "description", None) or []
    return tuple(str(entry[0]) for entry in description)


def _returns_rows(cursor: Any) -> bool:
    if hasattr(cursor, "returns_rows"):  # SQLAlchemy CursorResult
        return bool(cursor.returns_rows)
    return bool(getattr(cursor, "description", None))


def _fetch_batch(cursor: Any, size: int) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in cursor.fetchmany(size)]


def _per_row_bytes(batch: Sequence[tuple[Any, ...]]) -> float:
    """A conservative first-batch estimate: per-value `sys.getsizeof`
    plus the tuple's own size — cheap, measured once (the pandas
    `memory_usage(deep=True)` pattern's tuple-world analogue)."""
    total = sum(
        sys.getsizeof(row) + sum(sys.getsizeof(value) for value in row) for row in batch
    )
    return max(total / len(batch), 1.0)


def fetch_bounded(
    connection: Any,
    sql: str,
    parameters: Mapping[str, Any] | None,
    bounds: ResourceBounds,
    batch_size: int,
    *,
    registry_id: str,
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
    """Execute and fetch ALL rows — each batch admitted through the
    dynamic gate before it is retained, so the refusal (over the row
    cap, or over live headroom) lands before the memory does.

    The retained working set is CHARGED to the aggregate residency
    ledger under `registry_id` as it accrues (CR-007), so a concurrent
    bounded query's headroom check sees this query's memory and the two
    cannot jointly exceed the ceiling. The caller owns the registry_id's
    lifetime and MUST release it once the result is handed off (the
    charge is left standing on return — this is not a self-contained
    admission). To keep the query's own accumulating set from
    double-counting against its own headroom check, its prior charge is
    released immediately before each admission and re-established
    immediately after (`charge` is the joint ceiling gate; the release
    is idempotent on the first batch)."""
    cursor = _execute(connection, sql, parameters)
    columns = _columns_of(cursor)
    rows: list[tuple[Any, ...]] = []
    per_row: float | None = None
    while True:
        batch = _fetch_batch(cursor, batch_size)
        if not batch:
            break
        if per_row is None:
            per_row = _per_row_bytes(batch)
        total = len(rows) + len(batch)
        bounds.release(registry_id)
        bounds.admit_analysis(rows=total, per_row_bytes=per_row)
        bounds.charge(registry_id, int(total * per_row))
        rows.extend(batch)
    return columns, tuple(rows)


def execute_mutation(
    connection: Any, sql: str, parameters: Mapping[str, Any] | None
) -> int | None:
    """Execute a write and commit where the shape needs it (SQLAlchemy
    2.0 connections are transactional; the native DuckDB connection
    autocommits). Returns the affected-row count when the driver
    reports one, else None — never a guess."""
    cursor = _execute(connection, sql, parameters)
    affected = getattr(cursor, "rowcount", None)
    if _is_sqlalchemy(connection):
        connection.commit()
    return affected if isinstance(affected, int) and affected >= 0 else None


def iter_frames(
    connection: Any,
    sql: str,
    parameters: Mapping[str, Any] | None,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    """The streaming pull source (E6.6): execute once, then yield one
    DataFrame per fetched batch. The caller owns the connection's
    lifetime — the ChunkRegistry's `on_close` releases it (§5)."""
    cursor = _execute(connection, sql, parameters)
    columns = _columns_of(cursor)
    while True:
        batch = _fetch_batch(cursor, batch_size)
        if not batch:
            return
        yield pd.DataFrame(batch, columns=list(columns))
