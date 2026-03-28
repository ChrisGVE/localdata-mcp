"""Database schema export for LocalData MCP."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool = True
    default: Optional[str] = None
    primary_key: bool = False
    comment: Optional[str] = None


@dataclass
class ForeignKeyInfo:
    column: str
    references_table: str
    references_column: str
    constraint_name: Optional[str] = None


@dataclass
class IndexInfo:
    name: str
    columns: List[str] = field(default_factory=list)
    unique: bool = False


@dataclass
class TableSchema:
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    primary_key: List[str] = field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)
    indexes: List[IndexInfo] = field(default_factory=list)


class SchemaIntrospector:
    """Extract schema metadata via SQLAlchemy Inspector."""

    def __init__(self, engine):
        from sqlalchemy import inspect as sa_inspect

        self._engine = engine
        self._inspector = sa_inspect(engine)
        self._dialect = engine.dialect.name

    def get_all_tables(self, schema=None) -> List[str]:
        try:
            return self._inspector.get_table_names(schema=schema)
        except Exception:
            return []

    def get_table_schema(self, table: str, schema=None) -> TableSchema:
        pk = self._inspector.get_pk_constraint(table, schema=schema)
        pk_cols = pk.get("constrained_columns", []) if pk else []

        columns = []
        for col in self._inspector.get_columns(table, schema=schema):
            columns.append(
                ColumnInfo(
                    name=col["name"],
                    type=str(col.get("type", "VARCHAR")),
                    nullable=col.get("nullable", True),
                    default=str(col["default"]) if col.get("default") else None,
                    primary_key=col["name"] in pk_cols,
                    comment=col.get("comment"),
                )
            )

        fks = []
        for fk in self._inspector.get_foreign_keys(table, schema=schema):
            for i, col in enumerate(fk.get("constrained_columns", [])):
                ref_cols = fk.get("referred_columns", [])
                fks.append(
                    ForeignKeyInfo(
                        column=col,
                        references_table=fk.get("referred_table", ""),
                        references_column=ref_cols[i] if i < len(ref_cols) else "",
                        constraint_name=fk.get("name"),
                    )
                )

        indexes = []
        for idx in self._inspector.get_indexes(table, schema=schema):
            indexes.append(
                IndexInfo(
                    name=idx.get("name", ""),
                    columns=idx.get("column_names", []),
                    unique=idx.get("unique", False),
                )
            )

        return TableSchema(
            name=table,
            columns=columns,
            primary_key=pk_cols,
            foreign_keys=fks,
            indexes=indexes,
        )


# Type mappings
_JSON_SCHEMA_TYPES = {
    "INTEGER": {"type": "integer"},
    "BIGINT": {"type": "integer"},
    "SMALLINT": {"type": "integer"},
    "FLOAT": {"type": "number"},
    "REAL": {"type": "number"},
    "NUMERIC": {"type": "number"},
    "DECIMAL": {"type": "number"},
    "BOOLEAN": {"type": "boolean"},
    "VARCHAR": {"type": "string"},
    "TEXT": {"type": "string"},
    "CHAR": {"type": "string"},
    "NVARCHAR": {"type": "string"},
    "DATE": {"type": "string", "format": "date"},
    "TIMESTAMP": {"type": "string", "format": "date-time"},
    "DATETIME": {"type": "string", "format": "date-time"},
    "JSON": {"type": "object"},
    "JSONB": {"type": "object"},
    "BLOB": {"type": "string", "contentEncoding": "base64"},
}

_PYTHON_TYPES = {
    "INTEGER": "int",
    "BIGINT": "int",
    "SMALLINT": "int",
    "FLOAT": "float",
    "REAL": "float",
    "NUMERIC": "float",
    "DECIMAL": "float",
    "BOOLEAN": "bool",
    "VARCHAR": "str",
    "TEXT": "str",
    "CHAR": "str",
    "NVARCHAR": "str",
    "DATE": "date",
    "TIMESTAMP": "datetime",
    "DATETIME": "datetime",
    "JSON": "Dict[str, Any]",
    "JSONB": "Dict[str, Any]",
    "BLOB": "bytes",
}

_TS_TYPES = {
    "INTEGER": "number",
    "BIGINT": "number",
    "SMALLINT": "number",
    "FLOAT": "number",
    "REAL": "number",
    "NUMERIC": "number",
    "BOOLEAN": "boolean",
    "VARCHAR": "string",
    "TEXT": "string",
    "CHAR": "string",
    "DATE": "string",
    "TIMESTAMP": "string",
    "DATETIME": "string",
    "JSON": "Record<string, unknown>",
    "BLOB": "string",
}


def _base_type(col_type: str) -> str:
    return col_type.split("(")[0].strip().upper()


def export_json_schema(introspector: SchemaIntrospector, tables=None) -> Dict[str, Any]:
    if tables is None:
        tables = introspector.get_all_tables()
    result: Dict[str, Any] = {"tables": {}}
    for t in tables:
        ts = introspector.get_table_schema(t)
        props: Dict[str, Any] = {}
        required: List[str] = []
        for col in ts.columns:
            bt = _base_type(col.type)
            props[col.name] = dict(_JSON_SCHEMA_TYPES.get(bt, {"type": "string"}))
            if not col.nullable:
                required.append(col.name)
            if "(" in col.type and bt in ("VARCHAR", "CHAR", "NVARCHAR"):
                try:
                    length = int(col.type.split("(")[1].rstrip(")"))
                    props[col.name]["maxLength"] = length
                except ValueError:
                    pass
        schema: Dict[str, Any] = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        if ts.primary_key:
            schema["x-primary-key"] = ts.primary_key
        if ts.foreign_keys:
            schema["x-foreign-keys"] = [
                {
                    "column": fk.column,
                    "references": f"{fk.references_table}.{fk.references_column}",
                }
                for fk in ts.foreign_keys
            ]
        if ts.indexes:
            schema["x-indexes"] = [idx.name for idx in ts.indexes]
        result["tables"][t] = schema
    return result


def export_python_dataclass(introspector: SchemaIntrospector, tables=None) -> str:
    if tables is None:
        tables = introspector.get_all_tables()
    lines = [
        "from dataclasses import dataclass",
        "from typing import Any, Dict, Optional",
        "from datetime import date, datetime",
        "",
    ]
    for t in tables:
        ts = introspector.get_table_schema(t)
        class_name = "".join(w.capitalize() for w in t.replace("-", "_").split("_"))
        lines.append(f"\n@dataclass\nclass {class_name}:")
        required_cols = [c for c in ts.columns if not c.nullable]
        optional_cols = [c for c in ts.columns if c.nullable]
        for col in required_cols + optional_cols:
            bt = _base_type(col.type)
            py_type = _PYTHON_TYPES.get(bt, "str")
            if col.nullable:
                lines.append(f"    {col.name}: Optional[{py_type}] = None")
            else:
                lines.append(f"    {col.name}: {py_type}")
    return "\n".join(lines)


def export_typescript(introspector: SchemaIntrospector, tables=None) -> str:
    if tables is None:
        tables = introspector.get_all_tables()
    lines: List[str] = []
    for t in tables:
        ts = introspector.get_table_schema(t)
        class_name = "".join(w.capitalize() for w in t.replace("-", "_").split("_"))
        lines.append(f"export interface {class_name} {{")
        for col in ts.columns:
            bt = _base_type(col.type)
            ts_type = _TS_TYPES.get(bt, "string")
            optional = "?" if col.nullable else ""
            lines.append(f"  {col.name}{optional}: {ts_type};")
        lines.append("}\n")
    return "\n".join(lines)


def export_sql_ddl(introspector: SchemaIntrospector, tables=None) -> str:
    if tables is None:
        tables = introspector.get_all_tables()
    lines: List[str] = []
    for t in tables:
        ts = introspector.get_table_schema(t)
        lines.append(f"CREATE TABLE {t} (")
        col_defs: List[str] = []
        for col in ts.columns:
            parts = [f"  {col.name} {col.type}"]
            if not col.nullable:
                parts.append("NOT NULL")
            if col.default:
                parts.append(f"DEFAULT {col.default}")
            col_defs.append(" ".join(parts))
        if ts.primary_key:
            col_defs.append(f"  PRIMARY KEY ({', '.join(ts.primary_key)})")
        for fk in ts.foreign_keys:
            col_defs.append(
                f"  FOREIGN KEY ({fk.column}) REFERENCES "
                f"{fk.references_table}({fk.references_column})"
            )
        lines.append(",\n".join(col_defs))
        lines.append(");\n")
    return "\n".join(lines)
