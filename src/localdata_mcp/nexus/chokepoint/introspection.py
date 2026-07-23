"""localdata_mcp/nexus/chokepoint/introspection.py — schema discovery (E9.1).

X-1's privileged half: reading a backend's catalog needs the live
engine (SQLAlchemy's inspector) or dialect catalog queries — exactly
the capabilities FR-802 confines to this side of the guard seam.
`guard.py` calls in here with the resolved record; tool modules only
ever see the returned plain data. Dispatch is by DECLARED kind first:
the store families answer with their SEMANTIC summaries (a kv/tree
store's key-space shape, a graph store's node/edge shape, an rdf
store's triple shape — PRD X-1's per-family contract), never their
raw carrier tables; SQL kinds answer with the inspector's table
catalog plus row counts (identifiers quoted through the dialect's own
preparer — catalog names are data, not trusted SQL). Neighbors:
guard.py is the only caller; persistence supplies records.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import inspect as sqla_inspect
from sqlalchemy import text

from localdata_mcp.nexus.persistence.record import ConnectionRecord

# Kinds answered semantically (the E8.3 store families).
_TREE_KINDS = frozenset({"kv", "tree"})


def endpoint_schema(record: ConnectionRecord) -> dict[str, Any]:
    """The endpoint's schema summary, by declared kind."""
    kind = record.backend_kind
    if kind in _TREE_KINDS:
        return _tree_summary(record)
    if kind == "graph":
        return _graph_summary(record)
    if kind == "rdf":
        return _rdf_summary(record)
    if kind == "duckdb":
        return _duckdb_summary(record)
    return _sql_summary(record)


def table_names(record: ConnectionRecord) -> tuple[str, ...]:
    """The endpoint's table catalog (SQL kinds only — the guard refuses
    store kinds before calling here)."""
    if record.backend_kind == "duckdb":
        with record.pool.connect() as connection:
            rows = connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        return tuple(row[0] for row in rows)
    inspector = sqla_inspect(_engine_of(record))
    return tuple(inspector.get_table_names())


def table_schema(record: ConnectionRecord, table: str) -> dict[str, Any] | None:
    """One table's columns + row count, or None when absent."""
    if table not in table_names(record):
        return None
    if record.backend_kind == "duckdb":
        return _duckdb_table(record, table)
    inspector = sqla_inspect(_engine_of(record))
    return _inspected_table(record, inspector, table)


# -- SQL kinds --------------------------------------------------------


def _engine_of(record: ConnectionRecord) -> Any:
    # SqlAlchemyHandle's engine attribute; the duckdb/rdf kinds never
    # route here (their branches use the connection protocol directly).
    return getattr(record.pool, "engine")


def _row_count(record: ConnectionRecord, quoted_table: str) -> int:
    with record.pool.connect() as connection:
        if hasattr(connection, "exec_driver_sql"):
            result = connection.execute(
                text(f"SELECT COUNT(*) FROM {quoted_table}")  # nosec B608
            )
            return int(result.scalar() or 0)
        return int(
            connection.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0]
        )  # nosec B608


def _inspected_table(
    record: ConnectionRecord, inspector: Any, table: str
) -> dict[str, Any]:
    quoted = inspector.bind.dialect.identifier_preparer.quote(table)
    columns = [
        {
            "name": column["name"],
            "type": str(column["type"]),
            "nullable": bool(column.get("nullable", True)),
        }
        for column in inspector.get_columns(table)
    ]
    primary = inspector.get_pk_constraint(table).get("constrained_columns") or []
    return {
        "name": table,
        "columns": columns,
        "primary_key": list(primary),
        "row_count": _row_count(record, quoted),
    }


def _sql_summary(record: ConnectionRecord) -> dict[str, Any]:
    engine = _engine_of(record)
    inspector = sqla_inspect(engine)
    return {
        "endpoint": record.name,
        "backend_kind": record.backend_kind,
        "storage": "sql",
        "dialect": engine.dialect.name,
        "tables": [
            _inspected_table(record, inspector, table)
            for table in inspector.get_table_names()
        ],
    }


def _duckdb_summary(record: ConnectionRecord) -> dict[str, Any]:
    tables = []
    for table in table_names(record):
        tables.append(_duckdb_table(record, table))
    return {
        "endpoint": record.name,
        "backend_kind": "duckdb",
        "storage": "sql",
        "dialect": "duckdb",
        "tables": tables,
    }


def _duckdb_table(record: ConnectionRecord, table: str) -> dict[str, Any]:
    with record.pool.connect() as connection:
        columns = [
            {"name": row[0], "type": row[1], "nullable": row[2] == "YES"}
            for row in connection.execute(
                "SELECT column_name, data_type, is_nullable "
                "FROM information_schema.columns WHERE table_name = ? "
                "ORDER BY ordinal_position",
                [table],
            ).fetchall()
        ]
        quoted = '"' + table.replace('"', '""') + '"'
        count = connection.execute(
            f"SELECT COUNT(*) FROM {quoted}"  # nosec B608
        ).fetchone()[0]
    return {
        "name": table,
        "columns": columns,
        "primary_key": [],
        "row_count": int(count),
    }


# -- store kinds (semantic summaries, PRD X-1) ------------------------


def _one_scalar(record: ConnectionRecord, sql: str) -> int:
    with record.pool.connect() as connection:
        return int(connection.execute(text(sql)).fetchone()[0])


def _tree_summary(record: ConnectionRecord) -> dict[str, Any]:
    """The kv/tree key-space shape: node and property counts, depth."""
    return {
        "endpoint": record.name,
        "backend_kind": record.backend_kind,
        "storage": "tree",
        "total_nodes": _one_scalar(record, "SELECT COUNT(*) FROM nodes"),
        "total_properties": _one_scalar(record, "SELECT COUNT(*) FROM properties"),
        "root_count": _one_scalar(
            record, "SELECT COUNT(*) FROM nodes WHERE parent_id IS NULL"
        ),
        "max_depth": _one_scalar(record, "SELECT COALESCE(MAX(depth), -1) FROM nodes"),
        "hint": "Browse with get_children / get_node; properties via list_keys.",
    }


def _graph_summary(record: ConnectionRecord) -> dict[str, Any]:
    node_count = _one_scalar(record, "SELECT COUNT(*) FROM graph_nodes")
    edge_count = _one_scalar(record, "SELECT COUNT(*) FROM graph_edges")
    density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0.0
    return {
        "endpoint": record.name,
        "backend_kind": "graph",
        "storage": "graph",
        "node_count": node_count,
        "edge_count": edge_count,
        "property_count": _one_scalar(record, "SELECT COUNT(*) FROM graph_properties"),
        "is_directed": True,
        "density": min(density, 1.0),
        "hint": "Browse with get_node / get_edges / get_graph_stats.",
    }


def _rdf_summary(record: ConnectionRecord) -> dict[str, Any]:
    with record.pool.connect() as connection:
        triple_count = int(
            connection.execute("SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }").fetchmany(
                1
            )[0][0]
        )
        predicate_count = int(
            connection.execute(
                "SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE { ?s ?p ?o }"
            ).fetchmany(1)[0][0]
        )
        subject_count = int(
            connection.execute(
                "SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE { ?s ?p ?o }"
            ).fetchmany(1)[0][0]
        )
    return {
        "endpoint": record.name,
        "backend_kind": "rdf",
        "storage": "rdf",
        "triple_count": triple_count,
        "subject_count": subject_count,
        "predicate_count": predicate_count,
        "hint": "Query with query(endpoint, SPARQL).",
    }


def quoted_select(record: ConnectionRecord, table: str) -> str:
    """A whole-table SELECT with the identifier quoted by the
    dialect's own rules — `table` has already been verified against
    the catalog (guard.read_table), so quoting guards injection via a
    crafted catalog name, not membership."""
    if record.backend_kind == "duckdb":
        quoted = '"' + table.replace('"', '""') + '"'
    else:
        engine = _engine_of(record)
        quoted = engine.dialect.identifier_preparer.quote(table)
    return f"SELECT * FROM {quoted}"  # nosec B608
