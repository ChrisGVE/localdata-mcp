"""The MCP tool surface.

One in-memory SQLite database is the workspace for the whole session. Files are
loaded into it as tables; existing SQLite databases are ``ATTACH``-ed read-only
alongside. That is what makes the flagship capability work — a CSV and a
database file are two schemas in one connection, so joining across them is
ordinary SQL rather than machinery we have to build:

    SELECT * FROM sales JOIN warehouse.products ON sales.sku = products.sku

Concurrency note: this server dispatches tool calls concurrently, and a
synchronous tool body runs on a worker OS thread — measured, two blocking calls
overlapped for 1.2 s on two different threads. The single connection is
therefore guarded by a lock rather than assumed to be reached from one thread.
"""

from __future__ import annotations

import threading
from typing import Any

from fastmcp import FastMCP

from .export import export_csv
from .loader import LoadError, TableInfo, Workspace, _quote
from .paths import PathNotAllowed, allowed_root, resolve_read_path

mcp = FastMCP(
    "localdata",
    instructions=(
        "SQL access to local tabular files and SQLite databases. Load files with "
        "load_file, attach existing databases with attach_database, then query "
        "across all of them with one SQL statement. Attached databases are "
        "namespaced by their alias (alias.table_name)."
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
_workspace: Workspace | None = None
_attached: dict[str, str] = {}


def _session() -> Workspace:
    global _workspace
    if _workspace is None:
        _workspace = Workspace.in_memory()
    return _workspace


def _reset() -> None:
    """Drop all session state. Used by tests, not exposed as a tool."""
    global _workspace
    if _workspace is not None:
        _workspace.close()
    _workspace = None
    _attached.clear()


def _table_payload(info: TableInfo) -> dict[str, Any]:
    return {
        "table": info.name,
        "rows": info.row_count,
        "source": info.source,
        "columns": [
            {
                "name": column.name,
                "type": column.declared_type,
                **(
                    {
                        "temporal": column.temporal_kind,
                        "unit": "nanoseconds_since_epoch",
                    }
                    if column.temporal_kind
                    else {}
                ),
                **(
                    {"storage_classes": column.storage_classes}
                    if column.is_mixed
                    else {}
                ),
            }
            for column in info.columns
        ],
        "mixed_columns": info.mixed_columns,
    }


def _mixed_column_warning(info: TableInfo) -> list[str]:
    if not info.mixed_columns:
        return []
    return [
        f"Columns {', '.join(info.mixed_columns)} hold more than one storage class. "
        f"Aggregates over such a column silently coerce text to 0 and keep it in "
        f"the denominator, so avg() and sum() will be wrong. Filter with "
        f"typeof(col)='integer' or CAST explicitly."
    ]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
def load_file(path: str, table_name: str | None = None) -> dict[str, Any]:
    """Load a tabular file (.csv, .tsv, .txt) into the workspace as a table.

    Args:
        path: Path to the file, within the allowed root.
        table_name: Name for the resulting table. Defaults to the file stem.

    Returns the table's shape and column types, plus a warning for any column
    holding mixed storage classes.
    """
    with _lock:
        try:
            info = _session().load_file(path, table_name)
        except (LoadError, PathNotAllowed) as exc:
            return {"ok": False, "error": str(exc)}

        payload = _table_payload(info)
        payload["ok"] = True
        warnings = _mixed_column_warning(info)
        if warnings:
            payload["warnings"] = warnings
        return payload


@mcp.tool
def attach_database(path: str, alias: str) -> dict[str, Any]:
    """Attach an existing SQLite database read-only, under a namespace alias.

    Its tables are then addressable as ``alias.table_name`` in any query, and can
    be joined against loaded files.

    Args:
        path: Path to the SQLite database file.
        alias: Namespace to reach its tables through.
    """
    with _lock:
        try:
            resolved = resolve_read_path(path)
        except PathNotAllowed as exc:
            return {"ok": False, "error": str(exc)}

        safe_alias = "".join(c for c in alias if c.isalnum() or c == "_")
        if not safe_alias:
            return {"ok": False, "error": f"Alias {alias!r} is not a usable name."}
        if safe_alias in _attached:
            return {"ok": False, "error": f"Alias {safe_alias!r} is already attached."}

        workspace = _session()
        try:
            # mode=ro is carried by the connection itself, so there is no
            # writable path to reach around.
            workspace._conn.execute(
                f"ATTACH DATABASE ? AS {_quote(safe_alias)}",
                (f"file:{resolved}?mode=ro",),
            )
        except Exception as exc:
            return {"ok": False, "error": f"Could not attach {resolved}: {exc}"}

        _attached[safe_alias] = str(resolved)
        tables = [
            row[0]
            for row in workspace._conn.execute(
                f"SELECT name FROM {_quote(safe_alias)}.sqlite_master "
                f"WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            "ok": True,
            "alias": safe_alias,
            "path": str(resolved),
            "tables": [f"{safe_alias}.{name}" for name in tables],
        }


@mcp.tool
def list_tables() -> dict[str, Any]:
    """List every table available to query, loaded and attached."""
    with _lock:
        workspace = _session()
        loaded = [_table_payload(info) for info in workspace.tables.values()]
        attached: list[dict[str, Any]] = []
        for alias, source in _attached.items():
            names = [
                row[0]
                for row in workspace._conn.execute(
                    f"SELECT name FROM {_quote(alias)}.sqlite_master "
                    f"WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            attached.append(
                {
                    "alias": alias,
                    "path": source,
                    "tables": [f"{alias}.{n}" for n in names],
                }
            )
        return {
            "ok": True,
            "root": str(allowed_root()),
            "loaded": loaded,
            "attached": attached,
        }


@mcp.tool
def describe_table(table: str) -> dict[str, Any]:
    """Describe a table's columns, types, and row count.

    Args:
        table: Table name, optionally qualified as ``alias.table``.
    """
    with _lock:
        workspace = _session()
        info = workspace.tables.get(table)
        if info is not None:
            payload = _table_payload(info)
            payload["ok"] = True
            warnings = _mixed_column_warning(info)
            if warnings:
                payload["warnings"] = warnings
            return payload

        # Attached tables are not in our own registry; ask SQLite directly.
        try:
            qualified = ".".join(_quote(part) for part in table.split(".", 1))
            columns = [
                {"name": row[1], "type": row[2] or ""}
                for row in workspace._conn.execute(f"PRAGMA table_info({qualified})")
            ]
            if not columns:
                return {"ok": False, "error": f"No such table: {table}"}
            rows = workspace._conn.execute(
                f"SELECT count(*) FROM {qualified}"
            ).fetchone()
            return {"ok": True, "table": table, "rows": rows[0], "columns": columns}
        except Exception as exc:
            return {"ok": False, "error": f"Could not describe {table}: {exc}"}


@mcp.tool
def query(sql: str, limit: int = 100) -> dict[str, Any]:
    """Run a SQL query across every loaded file and attached database.

    Args:
        sql: The SQL statement.
        limit: Maximum rows to return. Use 0 for no limit.
    """
    with _lock:
        try:
            columns, rows = _session().query(sql, limit=limit or None)
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
def export_query(sql: str, path: str, overwrite: bool = False) -> dict[str, Any]:
    """Run a query and write the full result to a CSV file.

    Refuses to replace an existing file unless ``overwrite`` is set. The file is
    created readable by its owner only.

    Args:
        sql: The SQL statement.
        path: Destination CSV path, within the allowed root.
        overwrite: Replace the file if it already exists.
    """
    with _lock:
        try:
            columns, rows = _session().query(sql)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        try:
            result = export_csv(columns, rows, path, overwrite=overwrite)
        except PathNotAllowed as exc:
            return {"ok": False, "error": str(exc)}
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
