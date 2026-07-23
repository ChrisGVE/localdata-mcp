"""renderers/schema.py — a schema mapping as a Markdown document (FR-902).

The harvested `schema_export` design decoupled from any engine: v3's
introspection lives with the Explore tools (E9), which hand THIS
renderer a plain mapping `{table: {"columns": [{name, type, nullable,
primary_key}, ...]}}` — NX-8 renders declared data, it never touches a
backend (§8 NX-8's boundary).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..interface import ExportError

FORMAT = "schema"


def render(payload: Any) -> bytes:
    if not isinstance(payload, Mapping):
        raise ExportError(
            f"schema payload must be a table mapping, got {type(payload).__name__}"
        )
    sections: list[str] = []
    for table_name, table in payload.items():
        sections.append(_table_section(str(table_name), table))
    return ("\n\n".join(sections) + "\n").encode("utf-8")


def _table_section(name: str, table: Any) -> str:
    columns = table.get("columns") if isinstance(table, Mapping) else None
    if not isinstance(columns, (list, tuple)):
        raise ExportError(f"schema table {name!r} carries no columns list")
    lines = [
        f"## {name}",
        "",
        "| column | type | nullable | primary key |",
        "| --- | --- | --- | --- |",
    ]
    for column in columns:
        if not isinstance(column, Mapping) or "name" not in column:
            raise ExportError(f"schema table {name!r} has a malformed column entry")
        lines.append(
            "| {name} | {type} | {nullable} | {pk} |".format(
                name=column["name"],
                type=column.get("type", ""),
                nullable=column.get("nullable", True),
                pk=column.get("primary_key", False),
            )
        )
    return "\n".join(lines)
