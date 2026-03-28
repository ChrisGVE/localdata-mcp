"""Tests for schema export module."""

import pytest
from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    Boolean,
    Float,
    create_engine,
)

from localdata_mcp.schema_export import (
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    SchemaIntrospector,
    TableSchema,
    export_json_schema,
    export_python_dataclass,
    export_sql_ddl,
    export_typescript,
)


@pytest.fixture
def engine():
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def populated_engine(engine):
    metadata = MetaData()
    Table(
        "users",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(255), nullable=False),
        Column("email", String(255), nullable=True),
        Column("active", Boolean, nullable=False, default=True),
    )
    Table(
        "orders",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
        Column("total", Float, nullable=False),
        Column("notes", Text, nullable=True),
    )
    Index("ix_users_email", metadata.tables["users"].c.email, unique=True)
    Index("ix_orders_user_id", metadata.tables["orders"].c.user_id)
    metadata.create_all(engine)
    return engine


@pytest.fixture
def introspector(populated_engine):
    return SchemaIntrospector(populated_engine)


def test_column_info_creation():
    col = ColumnInfo(name="id", type="INTEGER", nullable=False, primary_key=True)
    assert col.name == "id"
    assert col.type == "INTEGER"
    assert col.nullable is False
    assert col.primary_key is True
    assert col.default is None
    assert col.comment is None


def test_table_schema_creation():
    col = ColumnInfo(name="id", type="INTEGER")
    fk = ForeignKeyInfo(
        column="user_id", references_table="users", references_column="id"
    )
    idx = IndexInfo(name="ix_test", columns=["id"], unique=True)
    ts = TableSchema(
        name="test",
        columns=[col],
        primary_key=["id"],
        foreign_keys=[fk],
        indexes=[idx],
    )
    assert ts.name == "test"
    assert len(ts.columns) == 1
    assert ts.primary_key == ["id"]
    assert len(ts.foreign_keys) == 1
    assert ts.foreign_keys[0].references_table == "users"
    assert len(ts.indexes) == 1
    assert ts.indexes[0].unique is True


def test_introspector_get_tables(introspector):
    tables = introspector.get_all_tables()
    assert sorted(tables) == ["orders", "users"]


def test_introspector_get_table_schema(introspector):
    ts = introspector.get_table_schema("users")
    assert ts.name == "users"

    col_names = [c.name for c in ts.columns]
    assert "id" in col_names
    assert "name" in col_names
    assert "email" in col_names

    assert "id" in ts.primary_key

    id_col = next(c for c in ts.columns if c.name == "id")
    assert id_col.primary_key is True

    name_col = next(c for c in ts.columns if c.name == "name")
    assert name_col.nullable is False

    email_col = next(c for c in ts.columns if c.name == "email")
    assert email_col.nullable is True

    # Check FK on orders
    orders_ts = introspector.get_table_schema("orders")
    assert len(orders_ts.foreign_keys) == 1
    assert orders_ts.foreign_keys[0].references_table == "users"
    assert orders_ts.foreign_keys[0].references_column == "id"

    # Check indexes
    assert len(ts.indexes) >= 1
    idx_names = [i.name for i in ts.indexes]
    assert "ix_users_email" in idx_names


def test_json_schema_export(introspector):
    result = export_json_schema(introspector)
    assert "tables" in result
    assert "users" in result["tables"]

    users = result["tables"]["users"]
    assert users["type"] == "object"
    assert "id" in users["properties"]
    assert users["properties"]["id"]["type"] == "integer"

    # name is NOT NULL so should be required
    assert "name" in users.get("required", [])
    assert "id" in users.get("required", [])

    # email is nullable so should NOT be required
    assert "email" not in users.get("required", [])

    assert "x-primary-key" in users


def test_json_schema_varchar_maxlength(introspector):
    result = export_json_schema(introspector)
    users = result["tables"]["users"]
    # SQLite reports VARCHAR(255) in type string
    name_prop = users["properties"]["name"]
    assert name_prop["type"] == "string"
    assert name_prop.get("maxLength") == 255


def test_python_dataclass_export(introspector):
    output = export_python_dataclass(introspector)
    assert "from dataclasses import dataclass" in output
    assert "@dataclass" in output
    assert "class Users:" in output

    # Required fields should not have Optional
    assert "name: str" in output

    # Nullable fields should be Optional
    assert "email: Optional[str] = None" in output


def test_typescript_export(introspector):
    output = export_typescript(introspector)
    assert "export interface Users {" in output
    assert "  id: number;" in output
    assert "  name: string;" in output
    # email is nullable so should be optional
    assert "  email?: string;" in output


def test_sql_ddl_export(introspector):
    output = export_sql_ddl(introspector)
    assert "CREATE TABLE users" in output
    assert "PRIMARY KEY" in output
    assert "CREATE TABLE orders" in output
    assert "FOREIGN KEY (user_id) REFERENCES users(id)" in output


def test_export_empty_database(engine):
    introspector = SchemaIntrospector(engine)
    assert export_json_schema(introspector) == {"tables": {}}
    assert "class " not in export_python_dataclass(introspector)
    assert export_typescript(introspector) == ""
    assert export_sql_ddl(introspector) == ""


def test_export_table_filter(introspector):
    result = export_json_schema(introspector, tables=["users"])
    assert "users" in result["tables"]
    assert "orders" not in result["tables"]

    py_output = export_python_dataclass(introspector, tables=["orders"])
    assert "class Orders:" in py_output
    assert "class Users:" not in py_output

    ts_output = export_typescript(introspector, tables=["users"])
    assert "interface Users" in ts_output
    assert "interface Orders" not in ts_output
