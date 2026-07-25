"""The MCP tool surface.

There is one idea to learn here, and everything else follows from it: **a
datasource is attached under a nickname, and the nickname is a database.** A CSV,
a SQLite file and a service URL all become the same kind of thing, so tables are
always addressed as ``nickname.table`` and a join across two datasources is
ordinary SQL:

    SELECT * FROM shop.sales JOIN warehouse.products ON sales.sku = products.sku

Every operation carries the nickname. That is not ceremony — it names the engine
the statement is executed against, and it is what lets an evicted slot be
reported as an eviction instead of as ``no such table``.

Concurrency note: this server dispatches tool calls concurrently, and a
synchronous tool body runs on a worker OS thread — measured, two blocking calls
overlapped for 1.2 s on two different threads. The registry's host connection is
therefore guarded by a lock rather than assumed to be reached from one thread.
"""

from __future__ import annotations

import threading
from typing import Any

from fastmcp import FastMCP

from . import config
from .export import export_csv
from .loader import TableInfo
from .paths import PathNotAllowed, allowed_paths
from .slots import Registry, SlotError

mcp = FastMCP(
    "localdata",
    instructions=(
        "SQL over local data files and databases. Attach each datasource with "
        "attach_datasource(database, nickname) — a CSV/TSV file, a SQLite database "
        "file, or a database URL. Every datasource becomes a database named by its "
        "nickname, so its tables are addressed as nickname.table, and one query can "
        "join across datasources: SELECT ... FROM shop.sales JOIN wh.products ... . "
        "Pass the nickname to every other tool; it selects which datasource the "
        "statement runs against. Slots are limited and the oldest is evicted when "
        "the limit is reached, so check the 'evicted' field an attach returns."
    ),
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

# ponytail: one global lock serialises every tool call. Correct, and cheap at
# the scale of interactive use. A process-wide lock was measured starving a
# concurrent reader under a 200k-row load (one completed read per run), so if
# throughput under load matters, this is the thing to replace — with separate
# reader and writer connections over a WAL file-backed workspace, which measured
# best on every axis. See docs/CONSTRAINTS.md §3.6.
_lock = threading.Lock()
_registry: Registry | None = None


def _session() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def _reset() -> None:
    """Drop all session state. Used by tests, not exposed as a tool."""
    global _registry
    if _registry is not None:
        _registry.close()
    _registry = None


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def _column_payload(column: Any) -> dict[str, Any]:
    return {
        "name": column.name,
        "type": column.declared_type,
        **(
            {"temporal": column.temporal_kind, "unit": "nanoseconds_since_epoch"}
            if column.temporal_kind
            else {}
        ),
        **(
            {
                "storage_classes": column.storage_classes,
                "numeric_values": column.numeric_values,
                "non_numeric_values": column.non_numeric_values,
            }
            if column.is_mixed
            else {}
        ),
    }


def _table_payload(info: TableInfo) -> dict[str, Any]:
    return {
        "table": info.qualified,
        "rows": info.row_count,
        "source": info.source,
        "columns": [_column_payload(column) for column in info.columns],
        "mixed_columns": info.mixed_columns,
    }


def _mixed_column_warning(info: TableInfo) -> str:
    return (
        f"In {info.qualified}, columns {', '.join(info.mixed_columns)} hold more "
        f"than one storage class. Aggregates over such a column silently coerce "
        f"text to 0 and keep it in the denominator, so avg() and sum() will be "
        f"wrong. Filter with typeof(col)='integer' or CAST explicitly."
    )


def _slot_payload(nickname: str, registry: Registry) -> dict[str, Any]:
    slot = registry.slot(nickname)
    return {
        "nickname": slot.nickname,
        "kind": slot.kind,
        "source": slot.source,
        "tables": [f"{slot.nickname}.{table}" for table in registry.tables(nickname)],
    }


def _failed(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
def attach_datasource(database: str, nickname: str) -> dict[str, Any]:
    """Attach a datasource as a database named by ``nickname``.

    Args:
        database: A tabular file (.csv, .tsv, .txt), a SQLite database file, or a
            database URL. A file becomes a new database holding one table named
            after the file; a SQLite database arrives with the tables it has.
        nickname: How to address it in SQL, as ``nickname.table``. Must be usable
            as a SQL name: a letter or underscore, then letters, digits or
            underscores.

    Returns the slot's tables. If the slot limit was reached, ``evicted``
    describes the datasource that was dropped to make room, including every table
    it held, so it can be attached again.
    """
    with _lock:
        registry = _session()
        try:
            attachment = registry.attach(database, nickname)
        except (SlotError, PathNotAllowed) as exc:
            return _failed(exc)

        slot = attachment.slot
        payload: dict[str, Any] = {
            "ok": True,
            "nickname": slot.nickname,
            "kind": slot.kind,
            "source": slot.source,
            "tables": [f"{slot.nickname}.{table}" for table in slot.tables],
        }

        # A slot we built ourselves was described as it was read, so its shape
        # and its mixed-column signal are already known — returning them here
        # saves the caller an immediate describe_table round trip.
        if slot.kind == "file":
            described = [
                registry.describe(slot.nickname, table) for table in slot.tables
            ]
            payload["loaded"] = [_table_payload(info) for info in described]
            warnings = [
                _mixed_column_warning(info) for info in described if info.mixed_columns
            ]
            if warnings:
                payload["warnings"] = warnings

        # Always present, never merely omitted: a caller that has to tell an
        # absent key from an empty one will eventually get it wrong.
        evicted = attachment.evicted
        payload["evicted"] = (
            None
            if evicted is None
            else {
                "nickname": evicted.nickname,
                "kind": evicted.kind,
                "source": evicted.source,
                "tables": list(evicted.tables),
                "reason": evicted.reason,
            }
        )
        return payload


@mcp.tool
def list_tables(nickname: str | None = None) -> dict[str, Any]:
    """List the attached datasources and the tables inside them.

    Args:
        nickname: Restrict to one datasource. Omit for all of them.
    """
    with _lock:
        registry = _session()
        try:
            names = [nickname] if nickname else [s.nickname for s in registry.slots()]
            datasources = [_slot_payload(name, registry) for name in names]
        except SlotError as exc:
            return _failed(exc)

        return {
            "ok": True,
            "datasources": datasources,
            "slots_used": len(registry.slots()),
            "slots_available": registry.capacity(),
            # The posture the LLM is working under, where it will read it.
            "roots": [str(path) for path in allowed_paths()],
            "path_limited": config.active().path_limited,
        }


@mcp.tool
def describe_table(nickname: str, table: str) -> dict[str, Any]:
    """Describe one table's columns, types and row count.

    Args:
        nickname: The datasource holding it.
        table: The table name inside that datasource, unqualified.
    """
    with _lock:
        try:
            info = _session().describe(nickname, table)
        except SlotError as exc:
            return _failed(exc)

        payload = _table_payload(info)
        payload["ok"] = True
        if info.mixed_columns:
            payload["warnings"] = [_mixed_column_warning(info)]
        return payload


@mcp.tool
def query(nickname: str, sql: str, limit: int = 100) -> dict[str, Any]:
    """Run a SQL query against a datasource.

    Tables are addressed as ``nickname.table``. A single statement may join
    across every datasource attached from a file or a SQLite database; a
    datasource opened from a URL is a separate engine and cannot be joined
    against the others without copying the rows in first.

    Args:
        nickname: Which datasource executes the statement.
        sql: The SQL statement.
        limit: Maximum rows to return. Use 0 for no limit.
    """
    with _lock:
        try:
            columns, rows = _session().query(nickname, sql, limit=limit or None)
        except SlotError as exc:
            return _failed(exc)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": True,
            "columns": columns,
            "rows": [list(row) for row in rows],
            "row_count": len(rows),
            "truncated": bool(limit) and len(rows) == limit,
        }


@mcp.tool
def export_query(
    nickname: str, sql: str, path: str, overwrite: bool = False
) -> dict[str, Any]:
    """Run a query and write the full result to a CSV file.

    Refuses to replace an existing file unless ``overwrite`` is set. The file is
    created readable by its owner only.

    Args:
        nickname: Which datasource executes the statement.
        sql: The SQL statement.
        path: Destination CSV path, within the allowed paths.
        overwrite: Replace the file if it already exists.
    """
    with _lock:
        try:
            columns, rows = _session().query(nickname, sql)
        except SlotError as exc:
            return _failed(exc)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        try:
            result = export_csv(columns, rows, path, overwrite=overwrite)
        except PathNotAllowed as exc:
            return _failed(exc)
        except OSError as exc:
            return {"ok": False, "error": f"Could not write {path}: {exc}"}

        return {
            "ok": True,
            "path": result.path,
            "rows_written": result.row_count,
            "columns": result.columns,
            "replaced_existing": result.replaced_existing,
        }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
