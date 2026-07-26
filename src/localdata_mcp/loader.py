"""Reading sources into queryable databases, one per tag.

A :class:`Workspace` is a **dict keyed by tag**. Each entry holds the URI the
caller gave, where that data currently lives — which may be the URI, may be
memory, may be a temp file it was spilled to — and the SQLAlchemy engines the
database is reached through. Every access takes that tag's engines and runs a
transaction.

**A tag is a database, not a schema on a shared connection.** The earlier design
made every datasource an ``ATTACH``ed schema on one SQLite connection, which is
what made ``SELECT … FROM shop.sales JOIN wh.products …`` work, and which tied
the whole layer to one backend's statement vocabulary. Tables are now addressed
bare inside the tag the caller named — ``FROM sales``, not ``FROM shop.sales`` —
because the tag parameter has already chosen the database, exactly as connecting
to a database does everywhere else in SQL.

Everything here is SQLAlchemy **Core**, never the ORM: there are no mapped
classes, because there is no fixed schema to map — the tables arrive at runtime
from whatever file was read. What Core buys is that reaching a different backend
is a different URL rather than different code. The handful of things that cannot
be said portably live behind the seam in :mod:`dialects`, and the test for
belonging there is whether the *meaning* changes on another backend, not whether
the SQL is awkward.

The insert path is the part with a measured constraint behind it. Handing pandas
a frame via ``to_sql`` peaks at **35×** the frame's own size — 3.20 MB of data
allocating 113.75 MB — and sub-batching does not bound it, because pandas
materialises the whole frame into insert-ready sequences *before* it chunks.
Core will not take a lazy iterator at all (``ArgumentError: mapping or list
expected for parameters``) and materialising 800,000 rows for it costs **511 MB**.
Chunking a lazy iterator *ourselves* is what bounds it, and unlike pandas there is
no eager step upstream to defeat the chunking:

===========================  ==============  ==============
Rows                         100,000         800,000
===========================  ==============  ==============
Core, one chunk of 100,000   63.74 MB        68.27 MB
Core, chunks of 20,000       13.79 MB        13.82 MB
Core, chunks of 1,000        **0.84 MB**     **0.81 MB**
===========================  ==============  ==============

The peak tracks the **chunk size and nothing else** — 800,000 rows cost what
100,000 do. The smallest chunk measured is also the fastest, so there is no
memory-for-speed trade to weigh here. Core costs a flat ~2.6× wall clock against
handing the driver an iterator directly; that is the price of the abstraction and
it is paid once per load, not per query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import uuid4

import pandas as pd
from sqlalchemy import (
    INTEGER,
    REAL,
    TEXT,
    Column,
    Engine,
    Index,
    MetaData,
    Table,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

from . import binding
from .dialects import Backend, Engines, backend_for
from .paths import resolve_read_path

__all__ = [
    "ColumnInfo",
    "IndexInfo",
    "TableInfo",
    "Tagged",
    "Workspace",
    "LoadError",
]


class LoadError(RuntimeError):
    """A source that could not be loaded."""


#: Rows handed to Core in one execute. Bounds the insert's peak allocation; see
#: the module docstring for the measurement it comes from. Not a config knob:
#: 1,000 rows measured both smallest and fastest, so there is nothing to tune.
_INSERT_CHUNK = 1_000

#: Rows the read path pulls from the driver at a time. The point of streaming is
#: that a caller taking ten rows out of a million-row result never materialises
#: the rest.
_YIELD_PER = 1_000


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
    #: Empty on a backend whose columns carry a single real type.
    storage_classes: dict[str, int] = field(default_factory=dict)
    #: Among non-null values in a TEXT column, how many parse as a number and how
    #: many do not. Both zero for a column that is already numerically typed.
    numeric_values: int = 0
    non_numeric_values: int = 0
    #: A few distinct values from ``non_numeric_values``, so a caller can write
    #: the filter without first going to look. See :func:`_numeric_split`.
    non_numeric_examples: tuple[str, ...] = ()

    @property
    def mixed_kind(self) -> str | None:
        """Which signal made this column mixed, or ``None`` if it is not.

        The two are not interchangeable, and the difference decides the remedy:

        * ``"storage"`` — the values are stored under genuinely different
          classes, so ``typeof(col)`` tells them apart and can filter them.
        * ``"text"`` — every value is stored as text and only some of them
          *read* as numbers. ``typeof(col)`` answers ``'text'`` for all of them,
          so that filter separates nothing; the values themselves are what a
          filter has to name, which is why they are carried in
          ``non_numeric_examples``.

        Two signals, because one of them alone misses the common case. The
        storage-class count catches genuinely heterogeneous storage. But a CSV
        column mixing ``1``, ``2`` and ``3a`` is read by pandas as ``object``,
        declared ``TEXT``, and stored entirely as text — so its storage classes
        read as *one* class and the histogram says nothing. The numeric-parse
        split is what catches that, and it is the shape most real files take.
        """
        distinct_classes = [c for c in self.storage_classes if c != "null"]
        if len(distinct_classes) > 1:
            return "storage"
        if self.numeric_values > 0 and self.non_numeric_values > 0:
            return "text"
        return None

    @property
    def is_mixed(self) -> bool:
        """True when the column holds values of more than one kind.

        Worth surfacing, because **aggregates over a mixed column silently
        coerce text to 0 and keep it in the denominator** — the average of
        1..5 plus two text rows returns 2.14, not 3.0. No choice of column
        affinity fixes that; it is a property of the aggregate.
        """
        return self.mixed_kind is not None


@dataclass(frozen=True)
class TableInfo:
    name: str
    row_count: int
    columns: list[ColumnInfo]
    source: str
    #: The tag whose database holds this table.
    tag: str = "main"

    @property
    def qualified(self) -> str:
        """``tag.table`` — how this table is *identified*, not how it is addressed.

        A caller querying it writes the bare name, because the tag has already
        been chosen by the call. This form exists so two tables of the same name
        in different tags stay distinguishable in listings and in our own
        bookkeeping.
        """
        return f"{self.tag}.{self.name}"

    @property
    def mixed_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_mixed]


@dataclass(frozen=True)
class IndexInfo:
    """An index that exists on a table, as the database reports it.

    Read back by inspection rather than remembered from creation, because a
    datasource can arrive with indexes this server never made — an attached
    SQLite file usually has at least one.
    """

    name: str
    table: str
    #: The indexed columns, in index order. A component that is an *expression*
    #: rather than a plain column reflects as ``None``; it is kept in place so
    #: the position of the columns around it stays honest.
    columns: tuple[str | None, ...]
    unique: bool = False


@dataclass
class Tagged:
    """One tag's database: where it came from, where it is now, how to reach it.

    ``uri`` is what the caller asked for and never changes — it is the identity
    of the datasource. ``location`` is where the data actually sits *now*, and
    moves when a database is spilled from memory to a temp file. Keeping both is
    what lets a spilled tag keep answering under its own name while still
    reporting honestly where it came from.
    """

    tag: str
    uri: str
    location: str
    engines: Engines
    backend: Backend


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

_UNSAFE = re.compile(r"\W+")


def _sanitize(name: str, fallback: str) -> str:
    cleaned = _UNSAFE.sub("_", str(name).strip()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}" if cleaned else fallback
    return cleaned.lower()


def _index_name(table: str, columns: Sequence[str]) -> str:
    """What an index on these columns is called.

    Derived rather than chosen so that asking for the same index twice produces
    the same name, and so the collision is what tells a caller it is already
    there. Sanitised because the columns it is built from already were.
    """
    return _sanitize(f"ix_{table}_{'_'.join(columns)}", "ix")


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


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

#: Declared type to the Core type that renders it. The uppercase spellings are
#: SQLAlchemy's "exactly this SQL type" forms — ``Float`` would render ``FLOAT``
#: and quietly change what ``info`` reports a column to be.
_CORE_TYPES = {"INTEGER": INTEGER, "REAL": REAL, "TEXT": TEXT}


def _declared_type(dtype: Any) -> str:
    """Map a pandas dtype to a column type.

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

#: How many distinct non-numeric values to carry out of the scan. Enough to
#: write a filter from — real files use one or two sentinels — and bounded so a
#: column of unique junk cannot return a copy of itself.
MAX_NON_NUMERIC_EXAMPLES = 5


def _numeric_split(series: pd.Series) -> tuple[int, int, tuple[str, ...]]:
    """Count how many non-null values in a text column parse as numbers.

    A column where both counts are non-zero is the ordinary "mostly numbers,
    some junk" CSV column — the one whose ``avg()`` is silently wrong and whose
    storage-class histogram shows nothing, because every value was stored as
    text.

    Also returns the first few distinct values that did *not* parse. The counts
    alone say a filter is needed without saying what it must exclude, and the
    caller's next move is always to go and look — measured on live agents, every
    one of them spent a round trip on a ``GROUP BY`` to learn what this scan had
    already seen.

    Scanned in blocks. ``pd.to_numeric`` over a whole column allocates a second
    array the length of the column, which made loading a 200,000-row file peak
    four times higher than a 50,000-row one — turning an otherwise flat insert
    path into a linear one. Blocking bounds the peak while keeping the count
    exact; sampling would bound it too, and would miss the single odd value that
    is the entire point of the check.
    """
    numeric = non_numeric = 0
    examples: dict[str, None] = {}  # insertion-ordered, and deduplicating
    for start in range(0, len(series), _SCAN_BLOCK):
        block = series.iloc[start : start + _SCAN_BLOCK].dropna()
        if block.empty:
            continue
        parsed = pd.to_numeric(block, errors="coerce")
        block_numeric = int(parsed.notna().sum())
        numeric += block_numeric
        non_numeric += len(block) - block_numeric
        if len(examples) < MAX_NON_NUMERIC_EXAMPLES:
            for value in block[parsed.isna()]:
                examples.setdefault(str(value))
                if len(examples) == MAX_NON_NUMERIC_EXAMPLES:
                    break
    return numeric, non_numeric, tuple(examples)


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
# Helpers
# ---------------------------------------------------------------------------


def _blocks(rows: Iterator[dict], size: int) -> Iterator[list[dict]]:
    """Pull ``size`` rows at a time from a lazy iterator, never more.

    The whole insert-path memory property lives in this function: ``rows`` is
    never materialised, only ``size`` of it exists at once, and the peak is
    therefore a function of ``size`` rather than of the file.
    """
    while True:
        block = list(islice(rows, size))
        if not block:
            return
        yield block


def _unrepresentable(exc: BaseException) -> binding.UnrepresentableValue | None:
    """Find an unrepresentable-value error anywhere in a raised chain.

    Adapters run inside the driver, several frames below Core, so by the time the
    error surfaces SQLAlchemy has wrapped it. Walking the chain is what keeps the
    caller's message about *their value* rather than about a statement.
    """
    seen = exc
    while seen is not None:
        if isinstance(seen, binding.UnrepresentableValue):
            return seen
        seen = seen.__cause__ or seen.__context__
    return None


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class Workspace:
    """Tags, and the databases they name.

    Nothing is shared between tags: each has its own engines, its own
    transactions, and its own lifetime. Detaching one cannot disturb another, and
    a backend change is confined to the tag that uses it.
    """

    def __init__(self) -> None:
        binding.install()
        self._tags: dict[str, Tagged] = {}
        self._tables: dict[str, TableInfo] = {}
        #: Distinguishes this workspace's memory databases from any other
        #: workspace's in the same process. Shared-cache memory databases are
        #: addressed by name, so two workspaces using the same tag would
        #: otherwise silently share one database — which tests, running many
        #: workspaces per process, would hit immediately.
        self._token = uuid4().hex[:12]

    # -- construction ------------------------------------------------------

    @classmethod
    def in_memory(cls) -> "Workspace":
        """A workspace holding no tags yet.

        Kept as a named constructor because callers read better for it, and
        because what it once meant — *one* in-memory database that everything
        attaches to — is exactly the thing that no longer exists.
        """
        return cls()

    # -- the tag dict ------------------------------------------------------

    def tags(self) -> tuple[str, ...]:
        return tuple(self._tags)

    def entry(self, tag: str) -> Tagged:
        """The tag's entry, or an error naming what is actually here."""
        found = self._tags.get(tag)
        if found is None:
            known = ", ".join(self._tags) or "none"
            raise LoadError(f"No database is tagged {tag!r}. Tagged: {known}.")
        return found

    def location(self, tag: str) -> str:
        return self.entry(tag).location

    def uri(self, tag: str) -> str:
        return self.entry(tag).uri

    # -- opening -----------------------------------------------------------

    def attach_memory(self, tag: str) -> None:
        """Open a fresh, writable, empty database under ``tag``.

        This is what a flat file becomes: its own database, so a later table can
        be added beside the first one under the same tag.
        """
        backend = backend_for("sqlite")
        self._install(
            tag,
            uri=":memory:",
            location=":memory:",
            backend=backend,
            engines=backend.open_memory(tag, self._token),
        )

    def attach_file(self, tag: str, path: Path, *, readonly: bool = True) -> None:
        """Open an existing database file, read-only unless told otherwise.

        Writable is not a flag this module honours by being careful — it is a
        different URI, and the database itself is what refuses the write.

        ``sqlite`` is named here rather than derived, and that is not a dispatch:
        the caller has already established that this path *is* a SQLite database
        by reading its header. Naming a fact is not the same as guessing a type.
        """
        backend = backend_for("sqlite")
        location = str(path.resolve())
        self._install(
            tag,
            uri=location,
            location=location,
            backend=backend,
            engines=backend.open_file(path, writable=not readonly),
        )

    def attach(self, url: str | URL, tag: str, *, writable: bool = False) -> None:
        """Open any datasource SQLAlchemy can reach, under ``tag``.

        **The URL is the abstraction.** It carries which database this is,
        SQLAlchemy parses it, and :func:`dialects.backend_for` looks up whatever
        that dialect adds — finding nothing, most of the time, which is the
        ordinary case and not a failure. Nothing in this method knows or asks
        what is on the other end.

        A tag opened this way is a tag like any other: the same ``query``, the
        same ``describe``, the same ``insert_frame``. There is deliberately no
        second code path for "remote" datasources, because a second path is how
        one of them silently stops supporting a verb the other has.
        """
        parsed = make_url(url)
        safe = parsed.render_as_string(hide_password=True)
        backend = backend_for(parsed.get_backend_name())
        self._install(
            tag,
            uri=safe,
            location=safe,
            backend=backend,
            engines=backend.open(parsed, writable=writable),
        )

    def _install(
        self,
        tag: str,
        *,
        uri: str,
        location: str,
        backend: Backend,
        engines: Engines,
    ) -> None:
        if tag in self._tags:
            self.detach(tag)
        self._tags[tag] = Tagged(
            tag=tag, uri=uri, location=location, engines=engines, backend=backend
        )

    def relocate(self, tag: str, path: Path) -> None:
        """Point a tag at a file holding what it used to hold in memory.

        The tag keeps its identity — its ``uri`` still says where the data came
        from — while ``location`` follows the data to disk. The old engines are
        disposed, which is what actually frees the memory the spill was for.
        """
        entry = self.entry(tag)
        origin = entry.uri
        entry.engines.dispose()
        engines = entry.backend.open_file(path, writable=True)
        self._tags[tag] = Tagged(
            tag=tag,
            uri=origin,
            location=str(path),
            engines=engines,
            backend=entry.backend,
        )

    def detach(self, tag: str) -> None:
        """Close a tag's database and forget everything we knew about it."""
        entry = self._tags.pop(tag, None)
        if entry is not None:
            entry.engines.dispose()
        for key in [k for k in self._tables if k.startswith(f"{tag}.")]:
            del self._tables[key]

    # -- measuring ---------------------------------------------------------

    def resident_bytes(self, tag: str) -> int | None:
        """What this tag holds in our process, or ``None`` if that is not a thing.

        ``None`` is not zero. A tag backed by a file or a server holds nothing
        *here*, and reporting zero would invite a caller to treat it as an empty
        database rather than as a question that does not apply.
        """
        entry = self.entry(tag)
        return entry.backend.resident_bytes(entry.engines.write)

    def snapshot(self, tag: str, path: Path) -> None:
        """Write a consistent, compacted copy of a tag's database to a file."""
        entry = self.entry(tag)
        entry.backend.snapshot(entry.engines.write, path)

    # -- composing ---------------------------------------------------------

    def has_table(self, tag: str, table: str) -> bool:
        return table in self.table_names(tag)

    def table_names(self, tag: str) -> tuple[str, ...]:
        """Everything in this tag's database that can be selected from.

        Views included, and deliberately: a caller who cannot see one in the
        listing has no way to learn it is there. They describe like tables and
        are queried like tables, so telling them apart here would be a
        distinction without a use.
        """
        inspector = inspect(self.entry(tag).engines.read)
        names = set(inspector.get_table_names()) | set(inspector.get_view_names())
        return tuple(sorted(n for n in names if not n.startswith("sqlite_")))

    def drop_table(self, tag: str, table: str) -> None:
        entry = self.entry(tag)
        target = Table(table, MetaData())
        try:
            with entry.engines.write.begin() as conn:
                target.drop(conn)
        except SQLAlchemyError as exc:
            raise LoadError(f"Could not drop {tag}.{table}: {exc}") from exc
        self._tables.pop(f"{tag}.{table}", None)

    # -- indexes -----------------------------------------------------------

    def indexes(self, tag: str, table: str | None = None) -> tuple[IndexInfo, ...]:
        """Every index in this tag's database, or only those on one table.

        Inspected over the *write* engine for the reason :meth:`describe` gives:
        the inspector is PRAGMA underneath, and the read engine's authorizer
        refuses PRAGMA so that a caller's SQL cannot reach one.
        """
        entry = self.entry(tag)
        inspector = inspect(entry.engines.write)
        tables = (table,) if table is not None else self.table_names(tag)

        found: list[IndexInfo] = []
        for name in tables:
            try:
                reported = inspector.get_indexes(name)
            except SQLAlchemyError:
                # A view has no indexes and some dialects say so by raising.
                continue
            found.extend(
                IndexInfo(
                    name=str(index["name"]),
                    table=name,
                    columns=tuple(index.get("column_names") or ()),
                    unique=bool(index.get("unique")),
                )
                for index in reported
                if index.get("name")
            )
        return tuple(found)

    def create_index(self, tag: str, table: str, columns: Sequence[str]) -> IndexInfo:
        """Index ``columns`` on ``table``, under a name derived from both.

        The name is ours to generate rather than the caller's to choose: it is
        bookkeeping, and one less thing for a caller to have to invent, remember
        and get wrong. It comes back in the result and appears in ``indexes``,
        so it is always in hand before anything needs to drop it.
        """
        entry = self.entry(tag)
        target = self._reflect(entry, tag, table)

        missing = [column for column in columns if column not in target.c]
        if missing:
            known = ", ".join(target.c.keys())
            raise LoadError(
                f"{tag}.{table} has no column "
                f"{', '.join(repr(m) for m in missing)}. Its columns are: {known}."
            )

        name = _index_name(table, columns)
        index = Index(name, *[target.c[column] for column in columns])
        try:
            with entry.engines.write.begin() as conn:
                index.create(conn)
        except SQLAlchemyError as exc:
            raise LoadError(f"Could not create {name} on {tag}.{table}: {exc}") from exc
        return IndexInfo(name=name, table=table, columns=tuple(columns))

    def drop_index(self, tag: str, name: str) -> IndexInfo:
        """Remove an index by name, and say what went.

        The table is looked up rather than asked for, because ``DROP INDEX`` is
        one of the places dialects disagree — MySQL wants the table named, SQLite
        and PostgreSQL refuse it. Reflecting the index off its table and letting
        SQLAlchemy emit the statement means that difference is not ours to know.
        """
        entry = self.entry(tag)
        existing = next((i for i in self.indexes(tag) if i.name == name), None)
        if existing is None:
            known = ", ".join(sorted(i.name for i in self.indexes(tag))) or "none"
            raise LoadError(f"No such index: {tag}.{name}. In {tag}: {known}.")

        target = self._reflect(entry, tag, existing.table)
        index = next((i for i in target.indexes if i.name == name), None)
        if index is None:  # pragma: no cover - reflection disagreeing with itself
            raise LoadError(f"No such index: {tag}.{name}.")
        try:
            with entry.engines.write.begin() as conn:
                index.drop(conn)
        except SQLAlchemyError as exc:
            raise LoadError(f"Could not drop {tag}.{name}: {exc}") from exc
        return existing

    @staticmethod
    def _reflect(entry: Tagged, tag: str, table: str) -> Table:
        """The live schema of one table, columns and indexes both."""
        try:
            return Table(table, MetaData(), autoload_with=entry.engines.write)
        except SQLAlchemyError as exc:
            raise LoadError(f"No such table: {tag}.{table}") from exc

    # -- loading -----------------------------------------------------------

    def load_file(
        self, raw_path: str, tag: str, table_name: str | None = None
    ) -> TableInfo:
        """Read a tabular file into a new table in ``tag`` and describe what landed."""
        path = resolve_read_path(raw_path)
        frame = read_frame(path)
        name = _sanitize(table_name or path.stem, "table")
        return self.insert_frame(frame, name, source=str(path), tag=tag)

    def insert_frame(
        self, frame: pd.DataFrame, table: str, *, source: str, tag: str
    ) -> TableInfo:
        entry = self.entry(tag)
        columns = _unique_columns(list(frame.columns))
        declared = [_declared_type(frame[original].dtype) for original in frame.columns]

        target = Table(
            table,
            MetaData(),
            *[
                Column(name, _CORE_TYPES[sql_type]())
                for name, sql_type in zip(columns, declared)
            ],
        )

        try:
            with entry.engines.write.begin() as conn:
                target.drop(conn, checkfirst=True)
                target.create(conn)
                statement = target.insert()
                # The row iterator is never materialised — only one chunk of it
                # exists at a time. See the module docstring for the numbers.
                for block in _blocks(self._rows(frame, columns), _INSERT_CHUNK):
                    conn.execute(statement, block)
        except Exception as exc:
            unrepresentable = _unrepresentable(exc)
            if unrepresentable is not None:
                raise LoadError(
                    f"{source}: {unrepresentable.reason}. Value "
                    f"{unrepresentable.value!r} cannot be stored in SQLite "
                    f"without corrupting it."
                ) from exc
            if isinstance(exc, (SQLAlchemyError, OverflowError)):
                raise LoadError(f"Could not insert rows from {source}: {exc}") from exc
            raise

        info = TableInfo(
            name=table,
            row_count=self._count(entry, table),
            columns=self._describe_columns(entry, table, columns, declared, frame),
            source=source,
            tag=tag,
        )
        self._tables[info.qualified] = info
        return info

    @staticmethod
    def _rows(frame: pd.DataFrame, columns: list[str]) -> Iterator[dict]:
        """Frame rows as bind-parameter mappings, lazily.

        A generator, not a list comprehension: the difference between the two is
        0.81 MB and 511 MB at 800,000 rows.
        """
        for row in frame.itertuples(index=False, name=None):
            yield dict(zip(columns, row))

    # -- inspection --------------------------------------------------------

    @staticmethod
    def _count(entry: Tagged, table: str) -> int:
        """How many rows the table holds, asked in Core rather than in text."""
        target = Table(table, MetaData())
        with entry.engines.read.connect() as conn:
            return int(
                conn.execute(select(func.count()).select_from(target)).scalar_one()
            )

    def describe(self, tag: str, table: str, source: str = "") -> TableInfo:
        """Describe a table we did not load ourselves, by asking the database."""
        known = self._tables.get(f"{tag}.{table}")
        if known is not None:
            return known

        entry = self.entry(tag)
        # Inspected over the *write* engine, as residency is, and for the same
        # reason: this is the server asking about the schema, not the caller's
        # SQL running. The read engine's authorizer refuses PRAGMA — rightly, it
        # is what stops ``query`` reaching one — and SQLAlchemy's inspector is
        # PRAGMA underneath, so inspecting there refuses every table the server
        # did not load itself. No write ability is implied: a read-only
        # datasource carries ``mode=ro`` on both engines and SQLite refuses the
        # write whichever one asks.
        described = inspect(entry.engines.write).get_columns(table)
        if not described:
            raise LoadError(f"No such table: {tag}.{table}")

        columns = [column["name"] for column in described]
        declared = [str(column["type"]) for column in described]
        return TableInfo(
            name=table,
            row_count=self._count(entry, table),
            columns=self._describe_columns(entry, table, columns, declared, None),
            source=source,
            tag=tag,
        )

    def _describe_columns(
        self,
        entry: Tagged,
        table: str,
        columns: list[str],
        declared: list[str],
        frame: pd.DataFrame | None,
    ) -> list[ColumnInfo]:
        with entry.engines.read.connect() as conn:
            described = []
            for index, (name, sql_type) in enumerate(zip(columns, declared)):
                kind = None
                numeric = non_numeric = 0
                examples: tuple[str, ...] = ()
                if frame is not None:
                    series = frame[frame.columns[index]]
                    kind = binding.temporal_kind(series.dtype)
                    if sql_type == "TEXT":
                        numeric, non_numeric, examples = _numeric_split(series)
                described.append(
                    ColumnInfo(
                        name=name,
                        declared_type=sql_type,
                        temporal_kind=kind,
                        storage_classes=entry.backend.storage_classes(
                            conn, table, name
                        ),
                        numeric_values=numeric,
                        non_numeric_values=non_numeric,
                        non_numeric_examples=examples,
                    )
                )
        return described

    # -- public surface ----------------------------------------------------

    @property
    def tables(self) -> dict[str, TableInfo]:
        """Tables this workspace loaded, keyed by their qualified name."""
        return dict(self._tables)

    def engine(self, tag: str) -> Engine:
        """The writing engine for a tag, for callers that need the engine itself."""
        return self.entry(tag).engines.write

    def query(self, tag: str, sql: str) -> tuple[list[str], list[tuple]]:
        """Run a **read** query against one tag and return ``(column_names, rows)``.

        A query reads. Anything that would change the database — ``INSERT``,
        ``CREATE TABLE``, ``CREATE VIEW``, ``ATTACH``, a ``PRAGMA`` — is refused
        here, whatever rights the datasource itself carries. Mutation has its own
        verbs (``create``, ``drop``) which do not come through this method, so
        the refusal costs the surface nothing.

        Enforced by the connection's posture rather than by reading the SQL: this
        engine's connections are read-only from the moment they are opened, so
        there is no text to parse and mis-parse and no window in which the
        posture is briefly something else.

        **The whole result comes back.** Bounding it here was tried and removed:
        a row cap measures the wrong dimension, since a hundred rows of a
        two-hundred-column table is the flood it was meant to prevent. The SQL
        already has ``LIMIT`` for a caller who wants fewer rows, and ``path``
        writes an oversized result to a file instead of into the answer.

        Streamed nonetheless. ``yield_per`` bounds what the driver hands back at
        a time, which is what keeps the *server's* memory flat while the rows
        accumulate.
        """
        entry = self.entry(tag)
        entry.engines.refusal.take()
        try:
            with entry.engines.read.connect() as conn:
                result = conn.execution_options(
                    stream_results=True, yield_per=_YIELD_PER
                ).execute(text(sql))
                if not result.returns_rows:
                    return [], []
                names = list(result.keys())
                return names, [tuple(row) for row in result]
        except SQLAlchemyError as exc:
            raise self._explain(entry, exc, sql) from exc

    def _explain(self, entry: Tagged, exc: SQLAlchemyError, sql: str) -> LoadError:
        """Turn a driver error into something an agent can act on.

        Two cases are worth naming. A refused write should say *what* was
        attempted and where the verb for it lives. And a table addressed as
        ``tag.table`` — the spelling the previous ``ATTACH``-based design took —
        fails as a plain "no such table", which tells an agent its table is
        missing when in fact its addressing is stale.
        """
        denied = entry.engines.refusal.take()
        if denied is not None:
            return LoadError(
                f"query reads; it does not write. This statement asks to "
                f"{denied}, which is refused here even on a writable "
                f"datasource. To add a table or an index use create, to remove "
                f"one use drop; there is no verb for arbitrary DDL by design."
            )

        message = str(exc.orig) if getattr(exc, "orig", None) else str(exc)
        stale = re.search(rf"no such table:\s*{re.escape(entry.tag)}\.(\w+)", message)
        if stale is not None:
            table = stale.group(1)
            return LoadError(
                f"No such table: {entry.tag}.{table}. Tables are addressed by "
                f"their own name inside the datasource you named — write "
                f"FROM {table}, not FROM {entry.tag}.{table}. Available here: "
                f"{', '.join(self.table_names(entry.tag)) or 'none'}."
            )
        return LoadError(message)

    def close(self) -> None:
        for tag in list(self._tags):
            self.detach(tag)
