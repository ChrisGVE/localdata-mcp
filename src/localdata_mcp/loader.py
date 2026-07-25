"""Loading sources into a queryable SQLite workspace.

A :class:`Workspace` is one SQLite connection plus what we know about the tables
in it. Files are loaded into an in-memory database; an existing SQLite file is
opened directly, read-only.

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


def _numeric_split(series: pd.Series) -> tuple[int, int]:
    """Count how many non-null values in a text column parse as numbers.

    A column where both counts are non-zero is the ordinary "mostly numbers,
    some junk" CSV column — the one whose ``avg()`` is silently wrong and whose
    storage-class histogram shows nothing, because every value was stored as
    text.
    """
    non_null = series.dropna()
    if non_null.empty:
        return 0, 0
    parsed = pd.to_numeric(non_null, errors="coerce")
    numeric = int(parsed.notna().sum())
    return numeric, len(non_null) - numeric


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


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class Workspace:
    """One SQLite connection and the tables loaded into it."""

    def __init__(self, connection: sqlite3.Connection, *, writable: bool) -> None:
        binding.install()
        self._conn = connection
        self._writable = writable
        self._tables: dict[str, TableInfo] = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def in_memory(cls) -> "Workspace":
        connection = sqlite3.connect(
            ":memory:",
            # URI mode is a connection-level flag and applies to ATTACH as well,
            # so it must be set here for `ATTACH DATABASE 'file:...?mode=ro'` to
            # be honoured rather than read as a literal filename.
            uri=True,
            # Tool bodies are dispatched to worker OS threads, so the connection
            # is legitimately reached from more than one. Callers serialise
            # access themselves; see the lock in server.py.
            check_same_thread=False,
        )
        return cls(connection, writable=True)

    @classmethod
    def open_sqlite_file(cls, raw_path: str) -> "Workspace":
        """Open an existing SQLite database, read-only.

        Read-only is carried by the connection's own state via the URI, not by a
        wrapper object that callers are expected to route through. A guarantee
        implemented as an interception point can be walked around by reaching
        the intercepted object; there is nothing to reach around here.
        """
        path = resolve_read_path(raw_path)
        connection = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, check_same_thread=False
        )
        workspace = cls(connection, writable=False)
        workspace._adopt_existing_tables(str(path))
        return workspace

    # -- loading -----------------------------------------------------------

    def load_file(self, raw_path: str, table_name: str | None = None) -> TableInfo:
        """Read a tabular file into a new table and describe what landed."""
        if not self._writable:
            raise LoadError(
                "This workspace is read-only; open a file workspace to load."
            )

        path = resolve_read_path(raw_path)
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

        name = _sanitize(table_name or path.stem, "table")
        return self._insert_frame(frame, name, source=str(path))

    def _insert_frame(
        self, frame: pd.DataFrame, table: str, *, source: str
    ) -> TableInfo:
        columns = _unique_columns(list(frame.columns))
        declared = [_declared_type(frame[original].dtype) for original in frame.columns]

        column_ddl = ", ".join(
            f"{_quote(name)} {sql_type}" for name, sql_type in zip(columns, declared)
        )
        self._conn.execute(f"DROP TABLE IF EXISTS {_quote(table)}")
        self._conn.execute(f"CREATE TABLE {_quote(table)} ({column_ddl})")

        placeholders = ", ".join("?" * len(columns))
        statement = f"INSERT INTO {_quote(table)} VALUES ({placeholders})"

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
            row_count=self._scalar(f"SELECT count(*) FROM {_quote(table)}"),
            columns=self._describe_columns(table, columns, declared, frame),
            source=source,
        )
        self._tables[table] = info
        return info

    @staticmethod
    def _rows(frame: pd.DataFrame) -> Iterator[tuple]:
        return frame.itertuples(index=False, name=None)

    # -- inspection --------------------------------------------------------

    def _describe_columns(
        self,
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
                    storage_classes=self._storage_classes(table, name),
                    numeric_values=numeric,
                    non_numeric_values=non_numeric,
                )
            )
        return described

    def _storage_classes(self, table: str, column: str) -> dict[str, int]:
        """Count actual storage classes present. Measured, not inferred."""
        rows = self._conn.execute(
            f"SELECT typeof({_quote(column)}), count(*) FROM {_quote(table)} GROUP BY 1"
        ).fetchall()
        return {storage_class: count for storage_class, count in rows}

    def _adopt_existing_tables(self, source: str) -> None:
        names = [
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for name in names:
            columns, declared = [], []
            for row in self._conn.execute(f"PRAGMA table_info({_quote(name)})"):
                columns.append(row[1])
                declared.append(row[2] or "")
            self._tables[name] = TableInfo(
                name=name,
                row_count=self._scalar(f"SELECT count(*) FROM {_quote(name)}"),
                columns=self._describe_columns(name, columns, declared, None),
                source=source,
            )

    def _scalar(self, sql: str) -> int:
        return int(self._conn.execute(sql).fetchone()[0])

    # -- public surface ----------------------------------------------------

    @property
    def tables(self) -> dict[str, TableInfo]:
        return dict(self._tables)

    def query(
        self, sql: str, limit: int | None = None
    ) -> tuple[list[str], list[tuple]]:
        """Run a read query and return ``(column_names, rows)``.

        No statement parsing happens here. On a workspace opened from a file the
        connection itself is read-only, and an in-memory workspace holds only
        what this session loaded, so there is nothing to protect from a write.
        """
        cursor = self._conn.execute(sql)
        if cursor.description is None:
            return [], []
        names = [description[0] for description in cursor.description]
        rows = cursor.fetchmany(limit) if limit else cursor.fetchall()
        return names, rows

    def close(self) -> None:
        self._conn.close()
