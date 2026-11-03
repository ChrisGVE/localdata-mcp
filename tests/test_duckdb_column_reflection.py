"""DuckDB column reflection works on SQLAlchemy 2.1 (issue #70).

SQLAlchemy 2.1 changed the PostgreSQL column query: each column now carries a
``collate`` subquery that joins ``pg_catalog.pg_collation``. DuckDB has no such
table, and DuckDB binds every table a statement names before it runs, so the
whole query fails whether or not the subquery is ever taken. ``duckdb-engine``
0.17.0 inherits that query for ``get_multi_columns``, and its own source says
so in a TODO. On SQLAlchemy 2.0 the same reflection works, which is why this
only shows up once the pin moves.

Our fix replaces the ``collate`` column with a ``NULL`` on DuckDB's dialect
instance only. DuckDB has no non-default collations for this server to report,
so the value it loses is one nothing reads.
"""

from __future__ import annotations

import duckdb
import pytest
from sqlalchemy import inspect

from localdata_mcp.dialects import backend_for


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "reflect.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE t (id INTEGER NOT NULL, name VARCHAR)")
    connection.execute("INSERT INTO t VALUES (1, 'a')")
    connection.close()
    return path


@pytest.mark.parametrize("writable", [False, True])
def test_duckdb_reflects_its_columns_through_the_backend(database, writable):
    engines = backend_for("duckdb").open_file(database, writable=writable)
    try:
        columns = inspect(engines.read).get_columns("t")
    finally:
        for engine in (engines.read, engines.write):
            engine.dispose()

    assert [column["name"] for column in columns] == ["id", "name"]
    assert [column["nullable"] for column in columns] == [False, True]
    assert str(columns[0]["type"]).upper() == "INTEGER"
