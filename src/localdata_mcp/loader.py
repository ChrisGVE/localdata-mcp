"""Reading sources into a queryable SQL workspace.

A :class:`Workspace` is a SQLAlchemy engine over the session's SQLite connection
plus what we know about the tables reachable through it. It is the *host*: every
datasource is an attached database on this one connection, so joining across two
of them is ordinary SQL rather than machinery we have to build. ``main`` stays
empty in normal operation — slots attach beside it, never into it.

The engine, rather than a raw driver connection, is what keeps the backend
replaceable: everything above this module works in SQL and table metadata, so
reaching a different database means a different URL, not different code. The
places that deliberately drop to the DBAPI cursor are marked and justified —
``ATTACH``, ``PRAGMA``, and the insert path below, none of which survive being
expressed through the Core layer with the properties we measured.

The insert path is the part with a measured constraint behind it. Handing pandas
a frame via ``to_sql`` peaks at **35×** the frame's own size — 3.20 MB of data
allocating 113.75 MB — and sub-batching does not bound it, because pandas
materialises the whole frame into insert-ready sequences *before* it chunks.
Feeding ``executemany`` a lazy row iterator instead holds a flat **0.008 MB**
peak from 100,000 rows through 1,600,000, and runs 4-5× faster.

The property is one sentence: **the row sequence handed to executemany is never
materialised.** Wrapping the iterator in ``list()`` puts the 40 MB straight back.
``docs/CONSTRAINTS.md`` §3 has the numbers.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from . import binding
from .paths import resolve_read_path

__all__ = ["ColumnInfo", "TableInfo", "Workspace", "LoadError"]


class LoadError(RuntimeError):
    """A source that could not be loaded."""


# ---------------------------------------------------------------------------
# What we record about what we loaded
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    declared_type: str
    #: ``"timestamp"`` / ``"duration"`` when the column holds integer ticks whose
    #: meaning is not recoverable from the value alone, otherwise ``None``.
    temporal_kind: str | None = None
    #: Storage classes actually present, as ``{"integer": 120, "text": 3}``.
    #: Measured with a GROUP BY after the load rather than predicted from dtypes.
    storage_classes: dict[str, int] = field(default_factory=dict)
    #: Among non-null values in a TEXT column, how many parse as a number and how
    #: many do not. Both zero for a column that is already numerically typed.
    numeric_values: int = 0
    non_numeric_values: int = 0

    @property
    def is_mixed(self) -> bool:
        """True when the column holds values of more than one kind.

        Worth surfacing, because **aggregates over a mixed column silently
        coerce text to 0 and keep it in the denominator** — the average of
        1..5 plus two text rows returns 2.14, not 3.0. No choice of column
        affinity fixes that; it is a property of the aggregate. A caller told
        about it can work around it with ``WHERE typeof(col)='integer'`` or an
        explicit ``CAST``.

        Two signals, because one of them alone misses the common case. The
        storage-class count catches genuinely heterogeneous storage. But a CSV
        column mixing ``1``, ``2`` and ``3a`` is read by pandas as ``object``,
        declared ``TEXT``, and stored entirely as text — so its storage classes
        read as *one* class and the histogram says nothing. The numeric-parse
        split is what catches that, and it is the shape most real files take.
        """
        distinct_classes = [c for c in self.storage_classes if c != "null"]
        if len(distinct_classes) > 1:
            return True
        return self.numeric_values > 0 and self.non_numeric_values > 0


@dataclass(frozen=True)
class TableInfo:
    name: str
    row_count: int
    columns: list[ColumnInfo]
    source: str
    #: The attached database holding this table — a slot nickname in normal use.
    schema: str = "main"

    @property
    def qualified(self) -> str:
        """How a caller addresses this table in SQL."""
        return f"{self.schema}.{self.name}"

    @property
    def mixed_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_mixed]


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

_UNSAFE = re.compile(r"\W+")


def _sanitize(name: str, fallback: str) -> str:
    cleaned = _UNSAFE.sub("_", str(name).strip()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    return cleaned.lower()


def _unique_columns(raw_names: list[Any]) -> list[str]:
    """Make column names safe and unique, preserving order.

    CSV headers arrive duplicated, blank, and full of punctuation. Silently
    dropping a duplicate would lose a column, so collisions get a numeric suffix.
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(raw_names):
        name = _sanitize(raw, f"column_{index + 1}")
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def _quote(identifier: str) -> str:
    """Quote an identifier for DDL. Doubling embedded quotes is the escape."""
    return '"' + identifier.replace('"', '""') + '"'


def _target(schema: str, table: str) -> str:
    """A schema-qualified table reference, both halves quoted."""
    return f"{_quote(schema)}.{_quote(table)}"


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


def _declared_type(dtype: Any) -> str:
    """Map a pandas dtype to a SQLite column affinity.

    An ``object`` column becomes ``TEXT`` rather than being left un-affined.
    That matters most in joins: ``INTEGER`` affinity on *either* side of a join
    collapses ``1``, ``1.0``, ``01`` and ``1.00`` into one value and fans the
    result out, and only all-``TEXT`` returns the truthful row count.
    """
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if binding.temporal_kind(dtype) is not None:
        return "INTEGER"
    return "TEXT"


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


#: Rows per block when scanning a column for numeric-ness. Bounds the scan's
#: peak allocation without changing its result — see :func:`_numeric_split`.
_SCAN_BLOCK = 50_000


def _numeric_split(series: pd.Series) -> tuple[int, int]:
    """Count how many non-null values in a text column parse as numbers.

    A column where both counts are non-zero is the ordinary "mostly numbers,
    some junk" CSV column — the one whose ``avg()`` is silently wrong and whose
    storage-class histogram shows nothing, because every value was stored as
    text.

    Scanned in blocks. ``pd.to_numeric`` over a whole column allocates a second
    array the length of the column, which made loading a 200,000-row file peak
    four times higher than a 50,000-row one — turning an otherwise flat insert
    path into a linear one. Blocking bounds the peak while keeping the count
    exact; sampling would bound it too, and would miss the single odd value that
    is the entire point of the check.
    """
    numeric = non_numeric = 0
    for start in range(0, len(series), _SCAN_BLOCK):
        block = series.iloc[start : start + _SCAN_BLOCK].dropna()
        if block.empty:
            continue
        parsed = pd.to_numeric(block, errors="coerce")
        block_numeric = int(parsed.notna().sum())
        numeric += block_numeric
        non_numeric += len(block) - block_numeric
    return numeric, non_numeric


def _read_csv(path: Path) -> pd.DataFrame:
    # `keep_default_na` is left on: pandas' blank/NA handling is what turns an
    # empty cell into NULL rather than the string "".
    return pd.read_csv(path)


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


#: Extension to reader. The seam through which new formats arrive — a new entry
#: is the whole change, because everything downstream works from the DataFrame.
READERS = {
    ".csv": _read_csv,
    ".tsv": _read_tsv,
    ".txt": _read_csv,
}


def read_frame(path: Path) -> pd.DataFrame:
    """Read a tabular file into a frame, touching no database state.

    Separate from the insert so a caller can find out whether a file is readable
    *before* committing to it. That ordering matters once slots are limited: a
    file that cannot be parsed must not cost a live datasource its place.
    """
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        supported = ", ".join(sorted(READERS))
        raise LoadError(f"No reader for {path.suffix!r}. Supported: {supported}")

    try:
        frame = reader(path)
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    if frame.empty and len(frame.columns) == 0:
        raise LoadError(f"{path.name} contains no columns.")
    return frame


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class Workspace:
    """The host connection: a SQLAlchemy engine, and the tables reached through it.

    The engine is the abstraction that lets a backend change without the code
    above it changing. Datasources arrive as attached databases on this one
    connection — that is what makes a join across two of them a plain statement.

    **Pool configuration is load-bearing, not incidental.** An in-memory SQLite
    database exists only for as long as its connection does, so a fresh checkout
    would find an empty database; ``StaticPool`` is therefore obligatory, and it
    means every checkout is the *same* DBAPI connection. That is precisely the
    configuration measured losing **79,807 of 200,000 rows** silently: the pool's
    default rollback-on-return means an unrelated reader closing its checkout
    discards a load in flight. Two things close it here — the session holds one
    raw connection for its whole life, so the pool never resets anything
    underneath a load, and callers serialise access (see the lock in
    ``server.py``). ``docs/CONSTRAINTS.md`` §3.5 has the measurement, and §3.6
    the posture to adopt if this stops being enough under load.
    """

    def __init__(self, engine: Engine) -> None:
        binding.install()
        self._engine = engine
        # Held for the session rather than checked out per operation. See the
        # class docstring: releasing it is what lets the pool roll back a load.
        self._raw = engine.raw_connection()
        self._tables: dict[str, TableInfo] = {}

    @property
    def _conn(self):
        """The underlying DBAPI connection.

        Used where SQLite-specific work is unavoidable — ``ATTACH``, ``PRAGMA``,
        and the non-materialising ``executemany`` — none of which SQLAlchemy's
        Core layer expresses without giving up the properties we measured.
        """
        return self._raw.driver_connection

    @property
    def engine(self) -> Engine:
        return self._engine

    # -- construction ------------------------------------------------------

    @classmethod
    def in_memory(cls) -> "Workspace":
        engine = create_engine(
            "sqlite://",
            poolclass=StaticPool,
            connect_args={
                # URI mode is a connection-level flag that also governs ATTACH,
                # so `ATTACH DATABASE 'file:...?mode=ro'` needs it set here or
                # the argument is read as a literal filename.
                "uri": True,
                # Tool bodies are dispatched to worker OS threads, so the
                # connection is legitimately reached from more than one.
                "check_same_thread": False,
            },
        )
        return cls(engine)

    # -- attaching ---------------------------------------------------------

    def attach_memory(self, schema: str) -> None:
        """Attach a fresh, writable, empty database under ``schema``.

        This is what a flat file becomes: its own database, so a later table can
        be added beside the first one under the same nickname.
        """
        self._conn.execute(f"ATTACH DATABASE ':memory:' AS {_quote(schema)}")

    def attach_readonly(self, schema: str, path: Path) -> None:
        """Attach an existing SQLite file, read-only.

        Read-only is carried by the connection's own URI, not by a wrapper
        callers are expected to route through. A guarantee implemented as an
        interception point can be walked around by reaching the intercepted
        object; there is nothing here to reach around.
        """
        self._conn.execute(
            f"ATTACH DATABASE ? AS {_quote(schema)}", (f"file:{path}?mode=ro",)
        )

    def detach(self, schema: str) -> None:
        """Release a schema, freeing its slot against SQLite's ATTACH limit."""
        self._conn.execute(f"DETACH DATABASE {_quote(schema)}")
        for key in [k for k in self._tables if k.startswith(f"{schema}.")]:
            del self._tables[key]

    # -- loading -----------------------------------------------------------

    def load_file(
        self, raw_path: str, schema: str = "main", table_name: str | None = None
    ) -> TableInfo:
        """Read a tabular file into a new table and describe what landed.

        ``schema`` is the attached database to land in — a slot nickname in
        normal operation. It defaults to ``main`` so the reader and typing
        behaviour can be exercised without slot ceremony.
        """
        path = resolve_read_path(raw_path)
        frame = read_frame(path)
        name = _sanitize(table_name or path.stem, "table")
        return self._insert_frame(frame, name, source=str(path), schema=schema)

    def _insert_frame(
        self, frame: pd.DataFrame, table: str, *, source: str, schema: str = "main"
    ) -> TableInfo:
        columns = _unique_columns(list(frame.columns))
        declared = [_declared_type(frame[original].dtype) for original in frame.columns]

        target = _target(schema, table)
        column_ddl = ", ".join(
            f"{_quote(name)} {sql_type}" for name, sql_type in zip(columns, declared)
        )
        self._conn.execute(f"DROP TABLE IF EXISTS {target}")
        self._conn.execute(f"CREATE TABLE {target} ({column_ddl})")

        placeholders = ", ".join("?" * len(columns))
        statement = f"INSERT INTO {target} VALUES ({placeholders})"

        try:
            # The iterator is passed through, never wrapped in list(). See the
            # module docstring: materialising it is a 40 MB peak instead of 0.02.
            self._conn.executemany(statement, self._rows(frame))
        except binding.UnrepresentableValue as exc:
            self._conn.rollback()
            raise LoadError(
                f"{source}: {exc.reason}. Value {exc.value!r} cannot be stored in "
                f"SQLite without corrupting it."
            ) from exc
        except (sqlite3.Error, OverflowError) as exc:
            self._conn.rollback()
            raise LoadError(f"Could not insert rows from {source}: {exc}") from exc

        self._conn.commit()

        info = TableInfo(
            name=table,
            row_count=self._scalar(f"SELECT count(*) FROM {target}"),
            columns=self._describe_columns(schema, table, columns, declared, frame),
            source=source,
            schema=schema,
        )
        self._tables[info.qualified] = info
        return info

    @staticmethod
    def _rows(frame: pd.DataFrame) -> Iterator[tuple]:
        return frame.itertuples(index=False, name=None)

    # -- inspection --------------------------------------------------------

    def table_names(self, schema: str) -> tuple[str, ...]:
        """Every ordinary table in an attached database, in name order."""
        return tuple(
            row[0]
            for row in self._conn.execute(
                f"SELECT name FROM {_quote(schema)}.sqlite_master "
                f"WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )

    def describe(self, schema: str, table: str, source: str = "") -> TableInfo:
        """Describe a table we did not load ourselves, by asking SQLite."""
        known = self._tables.get(f"{schema}.{table}")
        if known is not None:
            return known

        columns, declared = [], []
        for row in self._conn.execute(
            f"PRAGMA {_quote(schema)}.table_info({_quote(table)})"
        ):
            columns.append(row[1])
            declared.append(row[2] or "")
        if not columns:
            raise LoadError(f"No such table: {schema}.{table}")

        return TableInfo(
            name=table,
            row_count=self._scalar(f"SELECT count(*) FROM {_target(schema, table)}"),
            columns=self._describe_columns(schema, table, columns, declared, None),
            source=source,
            schema=schema,
        )

    def _describe_columns(
        self,
        schema: str,
        table: str,
        columns: list[str],
        declared: list[str],
        frame: pd.DataFrame | None,
    ) -> list[ColumnInfo]:
        described = []
        for index, (name, sql_type) in enumerate(zip(columns, declared)):
            kind = None
            numeric = non_numeric = 0
            if frame is not None:
                series = frame[frame.columns[index]]
                kind = binding.temporal_kind(series.dtype)
                if sql_type == "TEXT":
                    numeric, non_numeric = _numeric_split(series)
            described.append(
                ColumnInfo(
                    name=name,
                    declared_type=sql_type,
                    temporal_kind=kind,
                    storage_classes=self._storage_classes(schema, table, name),
                    numeric_values=numeric,
                    non_numeric_values=non_numeric,
                )
            )
        return described

    def _storage_classes(self, schema: str, table: str, column: str) -> dict[str, int]:
        """Count actual storage classes present. Measured, not inferred."""
        rows = self._conn.execute(
            f"SELECT typeof({_quote(column)}), count(*) "
            f"FROM {_target(schema, table)} GROUP BY 1"
        ).fetchall()
        return {storage_class: count for storage_class, count in rows}

    def _scalar(self, sql: str) -> int:
        return int(self._conn.execute(sql).fetchone()[0])

    # -- public surface ----------------------------------------------------

    @property
    def tables(self) -> dict[str, TableInfo]:
        """Tables this workspace loaded, keyed by their qualified name."""
        return dict(self._tables)

    def query(
        self, sql: str, limit: int | None = None
    ) -> tuple[list[str], list[tuple]]:
        """Run a read query and return ``(column_names, rows)``.

        No statement parsing happens here. Attached datasources carry their own
        read-only URI, and the writable schemas hold only what this session
        loaded, so there is nothing to protect from a write.
        """
        cursor = self._conn.execute(sql)
        if cursor.description is None:
            return [], []
        names = [description[0] for description in cursor.description]
        rows = cursor.fetchmany(limit) if limit else cursor.fetchall()
        return names, rows

    def close(self) -> None:
        self._raw.close()
        self._engine.dispose()
