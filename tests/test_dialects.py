"""What a backend has to supply, and what it gets for free.

The rule these defend is a negative one: **a database does not have to be
enumerated here to be usable.** Reaching a datasource is ``create_engine``'s job
and it already works for everything SQLAlchemy speaks, so :func:`backend_for`
answers for every dialect and refuses none. A dialect earns an entry in
``BACKENDS`` only when it can say something the generic answer cannot.

The failure this replaces is worth naming, because it is the one that made the
"anything SQLAlchemy speaks" claim false: ``backend_for`` used to raise for any
dialect nobody had subclassed, so a Postgres URL could not be opened at all —
not because SQLAlchemy could not reach it, but because this module had not been
told about it.
"""

from __future__ import annotations

from pathlib import Path

import foreign
import pytest
from sqlalchemy import Text, text

from localdata_mcp.dialects import (
    BACKENDS,
    Backend,
    SQLiteBackend,
    UnsupportedOperation,
    backend_for,
)


def build_database(path: Path, *, dialect: str = "sqlite") -> Path:
    return foreign.build_database(
        path,
        "products",
        [("sku", Text), ("name", Text)],
        [("a", "Widget")],
        dialect=dialect,
    )


# ---------------------------------------------------------------------------
# An unregistered dialect is usable, not refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", ["postgresql", "mysql", "duckdb", "wobble"])
def test_every_dialect_gets_a_backend(dialect):
    """No dialect is turned away. Being unlisted is the ordinary case."""
    assert isinstance(backend_for(dialect), Backend)


def test_a_registered_dialect_gets_its_own_answers():
    assert isinstance(backend_for("sqlite"), SQLiteBackend)
    assert BACKENDS["sqlite"] is backend_for("sqlite")


def test_the_generic_backend_actually_opens_a_datasource(tmp_path):
    """The generic path is real code, not a placeholder that raises.

    Exercised against SQLite because it is the only driver present without a
    server — but through :class:`Backend`, not :class:`SQLiteBackend`, so what
    runs here is the same ``create_engine`` call any other dialect would get.
    """
    database = build_database(tmp_path / "generic.db")
    engines = Backend().open(f"sqlite:///{database}", writable=False)
    try:
        with engines.read.connect() as connection:
            rows = connection.execute(text("SELECT sku FROM products")).fetchall()
        assert [row[0] for row in rows] == ["a"]
    finally:
        engines.dispose()


def test_the_generic_read_engine_does_not_persist_a_write(tmp_path):
    """The generic read floor: a write may execute, but none of it survives.

    Transactional rather than refusing-at-preparation, which is what a backend
    with no dialect-specific posture can honestly promise. SQLite raises that
    floor and refuses outright; this is what everything else still guarantees.
    """
    database = build_database(tmp_path / "generic.db")
    engines = Backend().open(f"sqlite:///{database}", writable=False)
    try:
        with engines.read.connect() as connection:
            connection.execute(text("INSERT INTO products VALUES ('b', 'Gadget')"))
            # No commit, and the connection closes here.
        with engines.read.connect() as connection:
            total = connection.execute(
                text("SELECT count(*) FROM products")
            ).scalar_one()
        assert total == 1
    finally:
        engines.dispose()


# ---------------------------------------------------------------------------
# What an unregistered dialect loses — honestly, and only the extras
# ---------------------------------------------------------------------------


def test_residency_is_unknown_rather_than_zero(tmp_path):
    """``None`` means the question does not apply; ``0`` would read as empty."""
    database = build_database(tmp_path / "generic.db")
    engines = Backend().open(f"sqlite:///{database}", writable=False)
    try:
        assert Backend().resident_bytes(engines.write) is None
    finally:
        engines.dispose()


def test_a_generic_datasource_says_why_it_cannot_be_saved(tmp_path):
    """Refused with a route out, rather than silently producing a wrong file."""
    database = build_database(tmp_path / "generic.db")
    engines = Backend().open(f"sqlite:///{database}", writable=False)
    try:
        with pytest.raises(UnsupportedOperation) as raised:
            Backend().snapshot(engines.write, tmp_path / "copy.db")
        assert "with create" in str(raised.value)
    finally:
        engines.dispose()


def test_storage_classes_are_empty_where_the_question_is_meaningless(tmp_path):
    """A backend with real column types has one class per column by construction."""
    database = build_database(tmp_path / "generic.db")
    engines = Backend().open(f"sqlite:///{database}", writable=False)
    try:
        with engines.read.connect() as connection:
            assert Backend().storage_classes(connection, "products", "sku") == {}
    finally:
        engines.dispose()


def test_the_refusal_names_the_database_the_caller_opened(tmp_path):
    """Not "a generic datasource" — that names our fallback, not their database.

    An unregistered dialect gets a plain :class:`Backend`, and it used to be one
    shared instance called "generic", so this sentence reached a PostgreSQL user
    naming a database that does not exist. Same defect MariaDB being registered
    in its own right already avoids.
    """
    with pytest.raises(UnsupportedOperation) as raised:
        backend_for("postgresql").snapshot(None, tmp_path / "copy.db")
    assert "postgresql" in str(raised.value)
    assert "generic" not in str(raised.value)


# ---------------------------------------------------------------------------
# Opening a file: the generic answer, and what a path may contain
# ---------------------------------------------------------------------------


def test_a_question_mark_in_a_filename_survives_the_generic_file_open(tmp_path):
    """A path is a value, not URL syntax — on every dialect, not just SQLite.

    ``loader`` used to build this URL by formatting a string, so ``why? not``
    split at the ``?`` and the database became ``why``. SQLite never showed it
    because its own override goes through ``as_uri()``; every *other* file-based
    dialect took the string path, and nothing exercised one. This is that gap:
    DuckDB, through the generic :meth:`Backend.open_file`.
    """
    awkward = build_database(
        tmp_path / "why? not.duckdb",
        dialect="duckdb",
    )
    engines = backend_for("duckdb").open_file(awkward, writable=False)
    try:
        with engines.read.connect() as connection:
            rows = connection.execute(text("SELECT sku FROM products")).fetchall()
        assert [row[0] for row in rows] == ["a"]
    finally:
        engines.dispose()


def test_the_generic_file_open_carries_the_read_only_posture(tmp_path):
    """DuckDB is *told* read-only in the URL, rather than merely not committed.

    The fact lives on the backend now. It used to be a dictionary in ``loader``
    keyed by dialect name, which is a dispatch on dialect name in shared code.
    """
    database = build_database(tmp_path / "warehouse.duckdb", dialect="duckdb")
    engines = backend_for("duckdb").open_file(database, writable=False)
    try:
        assert engines.read.url.query["access_mode"] == "read_only"
        with engines.read.connect() as connection:
            with pytest.raises(Exception):
                connection.execute(text("INSERT INTO products VALUES ('b', 'Gadget')"))
    finally:
        engines.dispose()

    writable = backend_for("duckdb").open_file(database, writable=True)
    try:
        # Writable is the absence of the posture, not a different code path.
        assert "access_mode" not in writable.write.url.query
    finally:
        writable.dispose()
