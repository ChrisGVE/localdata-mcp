"""Foreign databases — the kind this server attaches rather than makes.

Every test that needs an existing database to attach builds it here, through
SQLAlchemy Core with **portable** types, for the same reason
``test_endpoints._build_typed_table`` does: a fixture that reaches for a driver
directly is writing code around the library the server is built on, and it is
then stuck on one dialect — the same table cannot be asked for on a second one.
``build_database(..., dialect="duckdb")`` is that difference made visible.

Not a test module: pytest collects ``test_*.py`` only, and ``tests/`` has no
``__init__.py``, so this is imported as ``from foreign import build_database``
exactly as ``endpoints`` is.

**Two things deliberately do not come through here**, and both are SQLite
speaking about itself rather than a database anyone could have made:

* a column with **no declared type at all** (``CREATE TABLE t (v)``), which is
  what gives SQLite per-value storage classes. Core has no way to spell "no
  type", and the concept means nothing on another backend — which is exactly the
  test for what belongs in a dialect and not in shared code.
* a file whose **view names its own schema**, built by ``ATTACH`` and
  ``VACUUM … INTO``. It is a pathological artifact, constructed to be broken;
  there is nothing portable about it and nothing to gain from pretending there
  is.

Those two keep their raw ``sqlite3`` where they are used, and say so there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from sqlalchemy import Column, MetaData, Table, create_engine
from sqlalchemy.engine import URL
from sqlalchemy.types import TypeEngine


def build_database(
    path: Path,
    table: str,
    columns: Sequence[tuple[str, TypeEngine]],
    rows: Sequence[tuple] = (),
    *,
    dialect: str = "sqlite",
) -> Path:
    """Create ``table`` in a database file at ``path`` and fill it with ``rows``.

    ``columns`` pairs a name with a Core type — the generic ones (``Text``,
    ``Integer``), so each dialect renders its own spelling rather than being
    handed a token to emit verbatim.

    The engine is disposed before returning: the file has to be closed for the
    server to open it, and on a fixture that leaks an engine the failure arrives
    much later and somewhere else.
    """
    metadata = MetaData()
    defined = Table(table, metadata, *[Column(name, kind) for name, kind in columns])
    # URL.create, not an f-string: a path is a *value* here, and one holding a
    # `?` (there is a test for exactly that) would otherwise be re-read as the
    # start of a query string and the file created somewhere else entirely.
    engine = create_engine(URL.create(dialect, database=str(path)))
    try:
        metadata.create_all(engine)
        if rows:
            names = [name for name, _ in columns]
            with engine.begin() as conn:
                conn.execute(defined.insert(), [dict(zip(names, row)) for row in rows])
    finally:
        engine.dispose()
    return path
