"""The MCP tool surface.

There is one idea to learn here, and everything else follows from it: **a
datasource is attached under a nickname, and the nickname is a database.** A CSV,
a SQLite file and a service URL all become the same kind of thing, so tables are
always addressed as ``nickname.table`` and a join across two datasources is
ordinary SQL:

    SELECT * FROM shop.sales JOIN warehouse.products ON sales.sku = products.sku

Because a slot is a database rather than a view over a file, it has the verbs a
database has: lifecycle (``attach``, ``detach``, ``save``), composition
(``add_table``, ``drop_table``), and introspection (``info``). Seven in total,
several of them multi-faceted — **few and multi-faceted beats many and narrow**,
because the model has less to choose between and each choice is obvious.

**A query reads.** Every write — ``INSERT``, ``CREATE TABLE``, ``CREATE VIEW``,
``PRAGMA`` — is refused by ``query`` whatever the datasource itself permits, so
mutation happens only through the composition verbs and there is exactly one way
to change a slot. SQLite's own authorizer enforces it while the statement is
being prepared, so nothing here parses SQL to decide.

Two behaviours are deliberately invisible from out here. A database that outgrows
the memory budget is moved to a temp file and re-attached under the same nickname
between one call and the next, and no result mentions it. And write access is not
a flag this module honours by being careful: a read-only slot carries ``mode=ro``
in its own connection URI, so SQLite is what refuses the write.

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
from .slots import AddedTable, Attachment, JoinReport, Registry, Slot, SlotError

mcp = FastMCP(
    "localdata",
    instructions=(
        "SQL over local data files and databases.\n\n"
        "Attach each datasource with attach(database) — a CSV/TSV file, a SQLite "
        "database file, or a database URL. **Every datasource becomes a database, "
        "named by a nickname**, so even a single CSV holds its rows in a table: "
        "address them as nickname.table, never as the filename. attach returns the "
        "nickname it actually used, which may not be the one you asked for — use "
        "what it returns.\n\n"
        "One query can join across datasources: SELECT ... FROM shop.sales JOIN "
        "wh.products ... . But save writes one database, not the join, so when a "
        "second file is meant to be looked up against one already open, use "
        "add_table(nickname, source=...) to land it *inside* that database rather "
        "than attaching it separately. Pass join_on to be told which key values "
        "have no match on the other side.\n\n"
        "**query only reads.** INSERT, UPDATE, CREATE TABLE, CREATE VIEW and every "
        "other write are refused there whatever the datasource allows — composition "
        "has its own verbs, add_table and drop_table, and those are the ones the "
        "writable grant governs.\n\n"
        "Files you attach are read-only unless you pass writable=true; a database "
        "built from a flat file is yours and is always writable. Slots are limited "
        "and the oldest is evicted when the limit is reached, so check the "
        "'evicted' field an attach returns, and detach what you are done with. "
        "Nothing survives the session unless you save it."
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
    """The session registry, with any memory pressure relieved first.

    Relieving it *here* is what gives the spill its deferred timing, and doing
    it in one place is what stops a tool being added later that quietly skips
    it: every tool body reaches the registry through this function, exactly
    once, under the lock. The operation that crossed the budget has already
    returned; this is the next one, paying for it before doing its own work.
    """
    global _registry
    if _registry is None:
        _registry = Registry()
    else:
        _registry.relieve_memory()
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


def _slot_payload(slot: Slot, registry: Registry) -> dict[str, Any]:
    return {
        "nickname": slot.nickname,
        "kind": slot.kind,
        "source": slot.source,
        "writable": slot.writable,
        "tables": [
            f"{slot.nickname}.{table}" for table in registry.tables(slot.nickname)
        ],
    }


def _join_payload(report: JoinReport) -> dict[str, Any]:
    """The facts about a join, for the skill to say out loud in the user's words."""
    return {
        "key": report.key,
        "existing_table": report.existing_table,
        "added_table": report.added_table,
        "complete": report.complete,
        "matched_keys": report.matched_keys,
        "missing_from_added": {
            "values": list(report.missing_from_added),
            "total": report.missing_from_added_total,
        },
        "missing_from_existing": {
            "values": list(report.missing_from_existing),
            "total": report.missing_from_existing_total,
        },
    }


def _described(registry: Registry, nickname: str, tables: tuple[str, ...]) -> tuple:
    """Describe several tables, and collect whatever is worth warning about."""
    described = [registry.describe(nickname, table) for table in tables]
    warnings = [_mixed_column_warning(info) for info in described if info.mixed_columns]
    return described, warnings


def _attachment_payload(attachment: Attachment, registry: Registry) -> dict[str, Any]:
    slot = attachment.slot
    payload: dict[str, Any] = {"ok": True, **_slot_payload(slot, registry)}

    # A slot we built ourselves was described as it was read, so its shape and
    # its mixed-column signal are already known — returning them here saves the
    # caller an immediate info() round trip.
    if slot.kind == "file":
        described, warnings = _described(registry, slot.nickname, slot.tables)
        payload["loaded"] = [_table_payload(info) for info in described]
        if warnings:
            payload["warnings"] = warnings

    # Always present, never merely omitted: a caller that has to tell an absent
    # key from an empty one will eventually get it wrong.
    collision = attachment.collided_with
    payload["collided_with"] = (
        None
        if collision is None
        else {"nickname": collision.nickname, "source": collision.source}
    )
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


def _failed(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
def attach(
    database: str, nickname: str | None = None, writable: bool = False
) -> dict[str, Any]:
    """Attach a datasource as a database, and return the nickname it got.

    A flat file becomes a new database holding one table named after the file; a
    SQLite database arrives with the tables it already has.

    Args:
        database: A tabular file (.csv, .tsv, .txt), a SQLite database file, or a
            database URL.
        nickname: How to address it in SQL, as ``nickname.table``. Derived from
            the filename when omitted. If it collides with a slot already open,
            a numeric suffix is added — so always use the nickname that comes
            back rather than the one you asked for.
        writable: Allow writes to a datasource that came from outside. Ignored
            for a flat file, whose database is built here and always writable.

    Refuses a datasource that is already attached, naming the slot holding it.
    ``collided_with`` says which live slot forced a suffix, and ``evicted``
    describes the datasource dropped to make room, if any.
    """
    with _lock:
        registry = _session()
        try:
            attachment = registry.attach(database, nickname, writable=writable)
        except (SlotError, PathNotAllowed) as exc:
            return _failed(exc)
        return _attachment_payload(attachment, registry)


@mcp.tool
def detach(nickname: str) -> dict[str, Any]:
    """Close a datasource and free its slot.

    Everything the slot held is gone: tables added to it, and any rows not
    written out with ``save`` first.

    Args:
        nickname: The datasource to close.
    """
    with _lock:
        registry = _session()
        try:
            slot = registry.detach(nickname)
        except SlotError as exc:
            return _failed(exc)
        return {
            "ok": True,
            "nickname": slot.nickname,
            "source": slot.source,
            "tables": list(slot.tables),
            "slots_used": len(registry.slots()),
            "slots_available": registry.capacity(),
        }


@mcp.tool
def info(nickname: str | None = None, table: str | None = None) -> dict[str, Any]:
    """Describe what is attached, at whichever altitude you need.

    Three forms, and the arguments choose between them:

    * neither — every attached datasource, and the path posture in force;
    * ``nickname`` — that datasource and the tables inside it;
    * ``nickname`` and ``table`` — that table's columns, types and row count.

    Args:
        nickname: Restrict to one datasource.
        table: With a nickname, describe this one table in full.
    """
    with _lock:
        registry = _session()
        try:
            if nickname is not None and table is not None:
                return _table_detail(registry, nickname, table)
            if nickname is not None:
                return _slot_detail(registry, nickname)
            return _session_detail(registry)
        except SlotError as exc:
            return _failed(exc)


def _table_detail(registry: Registry, nickname: str, table: str) -> dict[str, Any]:
    described = registry.describe(nickname, table)
    payload = {"ok": True, **_table_payload(described)}
    if described.mixed_columns:
        payload["warnings"] = [_mixed_column_warning(described)]
    return payload


def _slot_detail(registry: Registry, nickname: str) -> dict[str, Any]:
    slot = registry.slot(nickname)
    tables = registry.tables(nickname)
    described, warnings = _described(registry, nickname, tables)
    payload: dict[str, Any] = {
        "ok": True,
        **_slot_payload(slot, registry),
        "contents": [
            {"table": info.qualified, "rows": info.row_count} for info in described
        ],
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


def _session_detail(registry: Registry) -> dict[str, Any]:
    return {
        "ok": True,
        "datasources": [_slot_payload(slot, registry) for slot in registry.slots()],
        "slots_used": len(registry.slots()),
        "slots_available": registry.capacity(),
        # The posture the LLM is working under, where it will read it.
        "roots": [str(path) for path in allowed_paths()],
        "path_limited": config.active().path_limited,
    }


@mcp.tool
def query(
    nickname: str,
    sql: str,
    limit: int = 100,
    path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run SQL against a datasource, returning rows or writing them to a file.

    Tables are addressed as ``nickname.table``. A single statement may join
    across every datasource attached from a file or a SQLite database; a
    datasource opened from a URL is a separate engine and cannot be joined
    against the others without copying the rows in first.

    Writes (INSERT, UPDATE, CREATE TABLE) go through here too, and succeed only
    where the datasource is writable.

    Args:
        nickname: Which datasource executes the statement.
        sql: The SQL statement.
        limit: Maximum rows to return. Use 0 for no limit. Ignored when writing
            to a file, which always receives the whole result.
        path: Write the full result to this CSV file instead of returning rows.
        force: Replace the file if it is already there. Set this only after the
            user has said to — the path is theirs, so the refusal you get
            without it is a question to put to them, not a retry to make. A file
            an attached datasource is sitting on is refused either way.
    """
    with _lock:
        registry = _session()
        try:
            columns, rows = registry.query(
                nickname, sql, limit=None if path else (limit or None)
            )
        except SlotError as exc:
            return _failed(exc)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        if path is None:
            return {
                "ok": True,
                "columns": columns,
                "rows": [list(row) for row in rows],
                "row_count": len(rows),
                "truncated": bool(limit) and len(rows) == limit,
            }

        try:
            result = export_csv(
                columns, rows, path, force=force, claimed=registry.claimed_paths()
            )
        except PathNotAllowed as exc:
            return _failed(exc)
        except OSError as exc:
            return {"ok": False, "error": f"Could not write {path}: {exc}"}

        return {
            "ok": True,
            "path": result.path,
            "rows_written": result.row_count,
            "columns": result.columns,
        }


@mcp.tool
def add_table(
    nickname: str,
    source: str,
    table: str | None = None,
    join_on: str | None = None,
    join_table: str | None = None,
) -> dict[str, Any]:
    """Add another table inside a datasource that is already open.

    Use this — rather than attaching a second slot — when a new file is meant to
    be looked up against one already loaded. ``save`` writes one database rather
    than a join, so landing both sides in the same slot is what makes the lookup
    outlive the session.

    Args:
        nickname: The datasource to add to. Must be writable.
        source: A file to read in.
        table: Name for the new table. Derived from the filename when omitted.
        join_on: A column shared with a table already in this datasource. Given
            one, the result reports which key values have no match on the other
            side, in both directions.
        join_table: Which existing table ``join_on`` refers to. Only needed when
            the datasource holds more than one.
    """
    with _lock:
        registry = _session()
        try:
            added = registry.add_table(
                nickname,
                source=source,
                table=table,
                join_on=join_on,
                join_table=join_table,
            )
        except (SlotError, PathNotAllowed) as exc:
            return _failed(exc)
        return _added_payload(added)


def _added_payload(added: AddedTable) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": True, **_table_payload(added.info)}
    if added.info.mixed_columns:
        payload["warnings"] = [_mixed_column_warning(added.info)]
    if added.join is not None:
        payload["join"] = _join_payload(added.join)
    return payload


@mcp.tool
def drop_table(nickname: str, table: str) -> dict[str, Any]:
    """Remove a table from a datasource.

    Args:
        nickname: The datasource holding it. Must be writable.
        table: The table name inside that datasource, unqualified.
    """
    with _lock:
        registry = _session()
        try:
            registry.drop_table(nickname, table)
            remaining = registry.tables(nickname)
        except SlotError as exc:
            return _failed(exc)
        return {
            "ok": True,
            "nickname": nickname,
            "dropped": f"{nickname}.{table}",
            "tables": [f"{nickname}.{name}" for name in remaining],
        }


@mcp.tool
def save(nickname: str, path: str, force: bool = False) -> dict[str, Any]:
    """Write a datasource out to a SQLite file the user keeps.

    Everything attached is otherwise ephemeral — it dies on detach and when this
    server stops. This is how a session's work survives, including tables added
    to a slot.

    The datasource stays open and unchanged; this writes a copy. Attaching that
    copy later is an ordinary attach, so it comes back read-only unless write is
    granted again.

    Args:
        nickname: The datasource to write out.
        path: Destination file, within the allowed paths.
        force: Replace the file if it is already there. Set this only after the
            user has said to — the name is theirs, so the refusal you get
            without it is a question to put to them, not a retry to make. A file
            an attached datasource is sitting on is refused either way.
    """
    with _lock:
        registry = _session()
        try:
            written = registry.save(nickname, path, force=force)
            tables = registry.tables(nickname)
        except (SlotError, PathNotAllowed) as exc:
            return _failed(exc)
        return {
            "ok": True,
            "nickname": nickname,
            "path": str(written),
            "tables": [f"{nickname}.{name}" for name in tables],
        }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
