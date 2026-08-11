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

import importlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TypeVar
from uuid import uuid4
from xml.etree import ElementTree

import pandas as pd
from sqlalchemy import (
    Column,
    Connection,
    Engine,
    Integer,
    MetaData,
    Table,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

from . import binding, temporal
from .dialects import Backend, Engines, backend_for, backend_for_url
from .paths import resolve_read_path

__all__ = [
    "ColumnInfo",
    "ColumnMeasurement",
    "IndexInfo",
    "ReadResult",
    "SourceRead",
    "SourceTable",
    "TableInfo",
    "Tagged",
    "Workspace",
    "LoadError",
    "measure_frame",
    "read_file",
    "read_source",
]

#: What a retried read gives back, whatever that happens to be. Only
#: :func:`_run_again_once` uses it, to keep its caller's return type rather than
#: flattening every catalog read to ``Any``.
_T = TypeVar("_T")


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

#: Rows pulled out of a *file* at a time, while measuring it and again while
#: inserting it. Distinct from :data:`_INSERT_CHUNK`, which bounds what one
#: ``execute`` hands the driver: this bounds what one chunk of the file costs to
#: hold. Matched to :data:`_SCAN_BLOCK` so the numeric scan sees one block per
#: chunk rather than blocking a block.
_READ_CHUNK = 50_000

#: What to call the primary key added to a loaded table on a backend that will
#: not make one without a key — see ``Backend.requires_primary_key``. Held here
#: rather than in :mod:`dialects` because the backend states the *fact* and this
#: module chooses the *response*, and the response must be the same for every
#: backend that ever states it.
#:
#: The leading underscore is what keeps it from colliding with a column the file
#: brought: :func:`_sanitize` strips leading underscores from every header it
#: cleans, and its fallback names are ``column_N``, so no data column can be
#: called this.
_SURROGATE_KEY = "_row"

#: What a caller is told when the statement they sent was not a query. Written
#: out here because it has to name the verb to use instead: an agent that is only
#: told "no" retries the same statement.
_NOT_A_READ = (
    "That statement returns no rows, so it is not a read, and query only reads — "
    "whatever the datasource itself permits. Composition has its own verbs: "
    "create adds a table or an index, update renames a table, drop removes "
    "either. A statement that does read returns rows even when it matches none."
)

#: Added where the refusal arrives too late to be the whole truth. Only Oracle
#: needs it today; the wording is the backend's name and this sentence, so a
#: second such dialect says the same thing without a second message.
_DDL_ALREADY_RAN = (
    "One caveat specific to {name}: it commits a CREATE or a DROP as it runs it, "
    "before anything here can object, so if that is what this was then it has "
    "already taken effect and this refusal did not undo it. Check with info, and "
    "use drop to remove what it made."
)


# ---------------------------------------------------------------------------
# What we record about what we loaded
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NamedFrame:
    """One table out of a file, with the name the file gave it if it had one.

    ``name`` is ``None`` for a format that holds a single unnamed table — a CSV
    is just rows, and what to call them is the caller's or the filename's. A
    spreadsheet sheet names itself, and that name is the one to use.
    """

    frame: pd.DataFrame
    name: str | None = None


@dataclass(frozen=True)
class ReadResult:
    """The tables in a file, and anything about the reading the data cannot show.

    **A file may hold more than one table**, and a workbook is the obvious case:
    three sheets are three tables. Reading only the first and ignoring the rest
    would leave data that is present in the file unreachable through the server,
    which is the same silent loss as dropping a nested value — so the reader
    returns all of them and the datasource, being a database, holds all of them.

    Most formats need no second field either: a CSV's rows are the whole story.
    Some cannot say everything in the data. A JSON file whose tables hang under
    a key was read from *one* of those keys; a fixed-width file was read under
    inferred boundaries. Each is a fact about the source that the caller can act
    on and would otherwise have to infer from a shape that looks perfectly
    ordinary — so it is carried out rather than dropped here.

    A note is a sentence, and the contract is the same as the warnings the load
    path already produces: state what happened and what to do about it. It is
    not a place to guess.
    """

    tables: tuple[NamedFrame, ...]
    notes: tuple[str, ...] = ()


def _one(frame: pd.DataFrame, notes: tuple[str, ...] = ()) -> ReadResult:
    """A result for the common case: a format holding a single unnamed table."""
    return ReadResult((NamedFrame(frame),), notes)


#: A reader turns a path into a frame plus whatever it had to assume or choose.
Reader = Callable[[Path], ReadResult]


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
    #: ``"iso8601_utc"`` for a date column held in the one canonical spelling,
    #: whose comparisons are therefore chronological. ``None`` for everything
    #: else, including a date column in no standard — see below.
    temporal_standard: str | None = None
    #: Values from a column that reads as dates but is not in a standard this
    #: server recognises, so it is compared as text. Non-empty means ``ORDER
    #: BY``, ``min``/``max`` and range filters on this column are unreliable.
    #: See :func:`temporal.unparsed_temporal_examples`.
    unparsed_temporal_examples: tuple[str, ...] = ()

    @property
    def is_unparsed_temporal(self) -> bool:
        """True when this column holds dates that will not compare correctly."""
        return bool(self.unparsed_temporal_examples)

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
    #: What the reader had to assume or choose to produce this table. Carried
    #: from :class:`ReadResult` and surfaced beside the other load warnings, so
    #: a choice made on the caller's behalf is one they get told about.
    notes: tuple[str, ...] = ()

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

    @property
    def unparsed_temporal_columns(self) -> list[str]:
        """Columns holding dates in no standard, so compared as text."""
        return [c.name for c in self.columns if c.is_unparsed_temporal]


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
    #: What the database had to do differently to build this, in words. Empty
    #: for the ordinary case, and only ever populated at creation — an index
    #: read back by inspection says nothing about how it came to be.
    notes: tuple[str, ...] = ()


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


#: How the backends we have seen say "that table is not here". Matching driver
#: prose is **enrichment, not detection**: a phrasing we do not recognise simply
#: falls through to the driver's own message, which is never worse than what the
#: caller would have got. Nothing branches on the outcome, so an unrecognised
#: dialect loses a listing of table names and nothing else.
_NO_SUCH_TABLE = re.compile(
    r"""(?x)
    no\ such\ table:\s*(?P<sqlite>[\w.]+)
    | relation\ "(?P<postgres>[^"]+)"\ does\ not\ exist
    | Table\ '[^.']*\.?(?P<mysql>[^.']+)'\ doesn't\ exist
    """,
    re.IGNORECASE,
)


def _driver_failures(entry: Tagged) -> tuple[type[BaseException], ...]:
    """The exception types a failure from this datasource can arrive as.

    ``SQLAlchemyError`` covers every driver that lets SQLAlchemy wrap it, which
    is nearly all of them. The backend adds any its driver raises unwrapped —
    see :meth:`dialects.Backend.driver_errors` for why that happens and why the
    answer is a named set rather than catching ``Exception`` here.
    """
    return (SQLAlchemyError, *entry.backend.driver_errors())


#: SQLSTATE 40001, ``serialization_failure``: this transaction was aborted and
#: the statement should simply be run again. A standard code with one standard
#: meaning, not any one database's invention.
_SERIALIZATION_FAILURE = "40001"


def _asks_to_be_retried(exc: BaseException) -> bool:
    """Whether the database said this statement should just be run again.

    The code is standard; the attribute carrying it is not. psycopg 3 publishes
    ``sqlstate``, psycopg 2 ``pgcode``, and a driver doing neither simply fails
    to match — which costs nothing, because the answer is then the error the
    caller would have received anyway.
    """
    original = getattr(exc, "orig", None)
    return any(
        getattr(original, attribute, None) == _SERIALIZATION_FAILURE
        for attribute in ("sqlstate", "pgcode")
    )


def _run_again_once(read: Callable[[], _T]) -> _T:
    """Run a catalog read, once more if the database asked for exactly that.

    A distributed engine caches the catalog per connection, so a DDL committed
    on one connection leaves every *other* pooled connection holding a snapshot
    the cluster has moved past. The next catalog read on such a connection is
    refused with SQLSTATE 40001 — YugabyteDB's wording is "A DDL occurred while
    processing this query. Try again." — and the read after it succeeds, because
    being refused is what refreshes the snapshot.

    Measured on YugabyteDB, where ``update`` renamed a table and then reported
    that it had failed. The DDL had committed and the rows were under the new
    name; it was the *describe* afterwards that met a stale snapshot on another
    engine. Reporting a change as failed when it succeeded is the worst answer
    available — worse than the raw error, because the obvious next move is to
    rename again and be told there is no such table.

    **Not a** :class:`~localdata_mcp.dialects.Backend` **axis, deliberately.**
    40001 means one thing everywhere: PostgreSQL raises it under SERIALIZABLE
    and every distributed SQL engine raises it routinely, so there is no
    per-dialect answer to override. A dialect branch here would state a dialect
    fact in shared code, which this design forbids — and it would have nowhere
    to live in any case, since YugabyteDB is reached through PostgreSQL's
    dialect and could only get an answer of its own by changing PostgreSQL's
    (issue #45).

    **Once, and only for a read.** A catalog read is idempotent, so running it
    again carries no consequence, and one retry is all the refusal costs — this
    is a snapshot being refreshed, not contention being waited out, so there is
    nothing for a backoff to help with. A *write* that fails this way is not
    retried here: whether it is safe to send again depends on what it was, and
    that is the caller's judgement rather than this server's to make.
    """
    try:
        return read()
    except SQLAlchemyError as exc:
        if not _asks_to_be_retried(exc):
            raise
    return read()


def _not_a_read(entry: Tagged) -> str:
    """The refusal, plus the caveat where the refusal cannot be the whole truth.

    On a backend that commits DDL as it runs it, a ``CREATE`` sent here has
    already happened by the time anything can object, and saying only "refused"
    would be the same lie in the other direction as calling a rolled-back write
    a success.
    """
    if entry.backend.ddl_survives_refusal():
        return f"{_NOT_A_READ} {_DDL_ALREADY_RAN.format(name=entry.backend.name)}"
    return _NOT_A_READ


def _objected_to_the_leading_verb(message: str, sql: str) -> bool:
    """Whether the database refused the *kind* of statement, not its wording.

    A streamed read is sent as a server-side cursor declaration, and PostgreSQL
    will not declare a cursor over anything but a query — so ``INSERT`` arrives
    as ``syntax error at or near "INSERT"``, which sends an agent hunting for a
    typo it does not have. The refusal is real and correct; only the diagnosis is
    wrong, and this is what corrects it.

    Narrow on purpose. It fires only when the word the database objected to is
    the statement's *own first word*, which is a complaint about what kind of
    statement it is. A malformed query objects at whichever token is actually
    wrong — never at its leading ``SELECT`` — so it falls through to the
    driver's message, exactly as an unrecognised phrasing does.

    **The cursor marker is two words that must both appear, not one phrase**
    (issue #58). This used to look for the literal ``cursor for``, which is
    PostgreSQL's exact wording and nobody else's: openGauss declares the same
    cursor as ``DECLARE "c_1" CURSOR WITHOUT HOLD FOR INSERT INTO …``, where the
    two words are separated by modifiers, so the test missed it and a refused
    write was reported as a syntax error the caller could not act on. Requiring
    ``declare`` *and* ``cursor`` keeps the context this needs — it is still only
    ever a cursor declaration being described — without pinning one engine's
    phrasing of the words in between.
    """
    words = sql.strip().split(None, 1)
    folded = message.lower()
    if not words or "declare" not in folded or "cursor" not in folded:
        return False
    return f'"{words[0].lower()}"' in folded


def _missing_table(message: str) -> str | None:
    """The table name a driver is complaining about, if we recognise the phrasing."""
    found = _NO_SUCH_TABLE.search(message)
    if found is None:
        return None
    name = next((value for value in found.groupdict().values() if value), None)
    return name.rsplit(".", 1)[-1] if name else None


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


def _longest_value(values: pd.Series, declared: str) -> int | None:
    """How wide the widest value in a text column actually is, in characters.

    ``None`` for a column that holds no text, because the question does not
    arise there. Measured rather than assumed for the one dialect that has to
    size the column up front: Oracle's usable text type is ``VARCHAR2``, and a
    number picked out of the air would either waste the row or truncate it.
    """
    if declared != "TEXT":
        return None
    present = values.dropna()
    if present.empty:
        return 0
    return int(present.astype(str).str.len().max())


def _declared_type(values: pd.Series) -> str:
    """Map a column to the type it is declared as.

    Takes the column rather than its dtype because one case cannot be decided
    from a dtype at all: temporal values inside an ``object`` column are stored
    as integer ticks, so declaring them ``TEXT`` would have SQLite's affinity
    turn each tick back into a string (:func:`binding.column_temporal_kind`).

    Otherwise an ``object`` column becomes ``TEXT`` rather than being left
    un-affined. That matters most in joins: ``INTEGER`` affinity on *either*
    side of a join collapses ``1``, ``1.0``, ``01`` and ``1.00`` into one value
    and fans the result out, and only all-``TEXT`` returns the truthful row
    count.
    """
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if binding.column_temporal_kind(values) is not None:
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


#: Delimiters worth naming in a warning. The character that separated the file
#: is never among the candidates, because a column name cannot contain it.
_COMMON_DELIMITERS = {";": "';'", "\t": "a tab", "|": "'|'", ",": "','"}


def _delimited(separator: str) -> Reader:
    """A reader for character-separated text, at a given separator.

    The default separator comes from the extension — comma for ``.csv`` and
    ``.txt``, tab for ``.tsv`` — and an explicit ``delimiter`` replaces the
    reader rather than being threaded through every other format's signature.
    """

    def read(path: Path) -> ReadResult:
        # `keep_default_na` is left on: pandas' blank/NA handling is what turns
        # an empty cell into NULL rather than the string "".
        frame = pd.read_csv(path, sep=separator)
        return _one(frame, _fat_column_note(frame, separator, path.name))

    return read


def _fat_column_note(frame: pd.DataFrame, separator: str, name: str) -> tuple[str, ...]:
    """Say when a file has plainly been read at the wrong separator.

    The parameter alone does not fix the silent failure: a caller who does not
    know the file is semicolon-separated gets one column holding every field and
    no signal at all — the whole header becomes the column's name. One column
    whose *name* still contains a common delimiter is that, and nothing else, so
    it is worth saying and worth naming the parameter that fixes it.

    This states what it found; it does not re-read the file at the guessed
    separator. Sniffing is the fail-open shape this project keeps being bitten
    by, and a guess that is usually right is the worst kind.
    """
    if len(frame.columns) != 1:
        return ()

    column = str(frame.columns[0])
    found = [
        spelling
        for character, spelling in _COMMON_DELIMITERS.items()
        if character != separator and character in column
    ]
    if not found:
        return ()

    return (
        f"{name} loaded as a single column whose name contains "
        f"{' and '.join(found)}, which is what a file separated by something "
        f"other than {_COMMON_DELIMITERS[separator]} looks like when read at "
        f"{_COMMON_DELIMITERS[separator]}. If that is the case, attach it again "
        f"with delimiter set to the right character. Nothing here guesses it.",
    )


def _read_json(path: Path) -> ReadResult:
    """Read a JSON document that holds one table.

    A JSON file is only *sometimes* tabular, so this reader says which shapes it
    takes rather than reshaping whatever it finds:

    * **An array of objects** is the table, and nothing is assumed.
    * **An object with exactly one non-empty array of objects under it** — the
      shape an API dump takes, ``{"count": 2, "employees": [...]}`` — is that
      array, and the note says which key it came from. There is no choice to
      make when there is one candidate, and refusing would be a dead end: an
      agent holding this file has no way to lift the array out of it, so a
      refusal it cannot act on is worse than a load it is told about.
    * **Anything else is refused**, naming what was found. Two candidate arrays
      is a genuine choice between tables, and choosing is the caller's.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    records, notes = _table_within(document, path.name)
    return _frame_of_records(records, notes)


def _read_jsonl(path: Path) -> ReadResult:
    """Read JSON Lines: one object per line, blank lines ignored.

    No shape ambiguity exists here — the format *is* a sequence of records — so
    unlike ``.json`` this reader never has anything to report.
    """
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise LoadError(
                    f"Could not read {path.name}: line {number} is not JSON ({exc})"
                ) from exc
            if not isinstance(value, dict):
                raise LoadError(
                    f"Could not read {path.name}: line {number} is "
                    f"{_json_kind(value)}, and JSON Lines is one object per line."
                )
            records.append(value)
    return _frame_of_records(records, ())


def _require(module: str, extra: str, doing: str):
    """Import an optional format library, or say how to install it.

    Every format is *known* to this server whether or not its library is here,
    so a caller asking for one that is not installed gets an instruction rather
    than a mystery. Listing only the installed formats would make the tool's own
    description vary by environment, which is worse: the agent could not learn
    what this server does without discovering what it happens to have.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise LoadError(
            f"{doing} needs {module}, which is not installed. Install it with: "
            f"pip install 'localdata-mcp[{extra}]' (or [all] for every format)."
        ) from exc


def _read_yaml(path: Path) -> ReadResult:
    """YAML parses to the same structures JSON does, so it gets the same rules.

    ``safe_load``, never ``load``: the full loader constructs arbitrary Python
    objects from a document, and every document here arrives from outside.
    """
    yaml = _require("yaml", "yaml", "Reading YAML")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    records, notes = _table_within(document, path.name)
    return _frame_of_records(records, notes)


def _read_fwf(path: Path) -> ReadResult:
    """Fixed-width text, whose column boundaries are inferred from alignment.

    Nothing in the file states where the columns are, so pandas finds them by
    looking at which character positions stay blank. That is the only way to
    read the format without a declared layout — and it is still a guess, so the
    note says so rather than letting an inferred split pass as a fact.
    """
    return _one(
        pd.read_fwf(path),
        (
            f"{path.name} is fixed-width, so its column boundaries were inferred "
            f"from which character positions are blank on every line — nothing "
            f"in the file declares them. Check the columns are the ones you "
            f"expect before relying on the split.",
        ),
    )


def _read_columnar(path: Path) -> ReadResult:
    """Parquet, Feather/Arrow and ORC — typed formats, so nothing is inferred.

    Feather goes through ``pyarrow.ipc`` rather than ``pandas.read_feather``,
    which routes to a ``pyarrow.feather`` entry point deprecated as of pyarrow
    24 and warns on every call. Same file format either way.
    """
    suffix = path.suffix.lower()
    _require("pyarrow", "parquet", f"Reading {suffix}")
    try:
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif suffix == ".orc":
            frame = pd.read_orc(path)
        else:
            import pyarrow as pa

            with pa.memory_map(str(path), "rb") as source:
                frame = pa.ipc.open_file(source).read_all().to_pandas()
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc
    return _one(frame)


#: Which library reads which workbook, and which extra installs it. Every one of
#: them is reached through ``pandas.read_excel``, which dispatches on the engine.
_WORKBOOKS = {
    ".xlsx": ("openpyxl", "openpyxl", "excel"),
    ".xlsm": ("openpyxl", "openpyxl", "excel"),
    ".xls": ("xlrd", "xlrd", "xls"),
    ".ods": ("odf", "odfpy", "ods"),
}


def _read_workbook(path: Path) -> ReadResult:
    """Every sheet of a workbook, each as a table under its own sheet name.

    ``sheet_name=None`` rather than the default ``0``: the default reads the
    first sheet and says nothing about the others, which leaves data that is
    present in the file unreachable through the server. A workbook is a database
    and its sheets are its tables, so all of them land.
    """
    suffix = path.suffix.lower()
    module, package, extra = _WORKBOOKS[suffix]
    _require(module, extra, f"Reading {suffix}")

    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    # A workbook may carry a sheet that is entirely empty; it is a sheet with no
    # table in it, and dropping it is not loss. Refusing the whole file over one
    # would be, so only a workbook with nothing in any sheet is refused.
    tables = tuple(
        NamedFrame(frame, name)
        for name, frame in sheets.items()
        if not frame.empty or len(frame.columns)
    )
    if not tables:
        raise LoadError(f"Could not read {path.name}: every sheet in it is empty.")
    return ReadResult(tables)


def _read_numbers(path: Path) -> ReadResult:
    """Apple Numbers, whose sheets each hold their own named tables.

    Two levels rather than one: a Numbers sheet is a canvas that may carry
    several tables, so the name here is the table's, qualified by its sheet only
    when the same table name appears on more than one.
    """
    parser = _require("numbers_parser", "numbers", "Reading Apple Numbers")
    try:
        document = parser.Document(str(path))
        found = [
            (sheet.name, table.name, table.rows(values_only=True))
            for sheet in document.sheets
            for table in sheet.tables
        ]
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    counts: dict[str, int] = {}
    for _, table_name, _ in found:
        counts[table_name] = counts.get(table_name, 0) + 1

    tables = []
    for sheet_name, table_name, rows in found:
        trimmed = _without_grid_padding(rows)
        if trimmed is None:
            continue
        header, body = trimmed
        frame = pd.DataFrame(body, columns=header)
        name = table_name if counts[table_name] == 1 else f"{sheet_name}_{table_name}"
        tables.append(NamedFrame(_inferred_types(frame), name))

    if not tables:
        raise LoadError(f"Could not read {path.name}: it holds no table with rows.")
    return ReadResult(tuple(tables))


def _without_grid_padding(rows: Sequence[Sequence[Any]]):
    """Strip a Numbers table's empty grid, and nothing that holds a value.

    A Numbers table is a fixed canvas — a new one is 8 columns by 12 rows —
    so the cells beyond the data come back as ``None`` and would otherwise
    become columns named ``None`` and a tail of all-null rows.

    Emptiness is tested on the *whole* column, not on its header: a column with
    values but no header is real data that happens to be unlabelled, and
    dropping it on the strength of a blank header would be exactly the silent
    loss this reader is meant to avoid. Returns ``None`` for a table that is
    entirely empty.
    """
    if not rows:
        return None

    header, *body = rows
    keep = [
        index
        for index in range(len(header))
        if header[index] is not None
        or any(index < len(row) and row[index] is not None for row in body)
    ]
    if not keep:
        return None

    kept_body = [[row[index] for index in keep] for row in body]
    kept_body = [row for row in kept_body if any(cell is not None for cell in row)]
    if not kept_body:
        return None

    return [header[index] for index in keep], kept_body


def _read_xml(path: Path) -> ReadResult:
    """Read an XML document whose root holds one repeated element per row.

    Written rather than delegated to ``pandas.read_xml``, which is fail-open on
    two shapes this corpus contains. Measured against the stdlib parser: a row
    holding a nested element comes back with that column ``NaN`` — the subtree
    silently dropped — and a row with a tag repeated twice keeps only the last
    one. Both are data loss with nothing reported, which is the class this
    server exists to refuse.

    Columns are the row's attributes and the tags of its direct children. A
    child with children of its own is kept as its XML text (as a nested JSON
    value is kept as JSON text); a tag appearing twice in one row is a list
    rather than a column, and is refused by name.
    """
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    rows, notes = _rows_within(root, path.name)

    records = []
    nested: dict[str, None] = {}  # insertion-ordered, and deduplicating
    for element in rows:
        record, held = _record_of(element, path.name)
        records.append(record)
        nested.update(dict.fromkeys(held))

    if nested:
        notes = notes + (
            f"Nested elements in {', '.join(nested)} were kept as XML text, "
            f"because SQL has no nested type. The text is the element as it "
            f"stood in the file, so nothing was lost.",
        )

    result = _frame_of_records(records, notes)
    frame = result.tables[0].frame
    return _one(_inferred_types(frame), result.notes)


def _inferred_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Read numbers out of text, since every value in XML arrives as text.

    The same inference ``read_csv`` performs, applied here because this reader
    builds its frame by hand. A column is converted only when **every** non-null
    value in it parses, so a column mixing numbers and text stays text and the
    mixed-column signal downstream still has something to find.

    The skip test asks what a column *is* rather than comparing its dtype to
    ``object``: pandas 3 infers a dedicated ``str`` dtype for text, so an
    ``!= object`` guard here skipped every column it was meant to convert.
    """
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(
            series
        ) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        present = series.notna().sum()
        if not present:
            continue
        candidate = pd.to_numeric(series, errors="coerce")
        if candidate.notna().sum() == present:
            frame[name] = candidate
    return frame


def _rows_within(
    root: ElementTree.Element, name: str
) -> tuple[list[ElementTree.Element], tuple[str, ...]]:
    """Pick the repeated element that is the table, on JSON's rules.

    One kind of child is the table. Several kinds, one of which repeats, is the
    wrapped shape — a ``<generated>`` beside the rows — and the note names what
    was left out. Two kinds that both repeat are two tables, which is a choice,
    so it is refused.
    """
    groups: dict[str, list[ElementTree.Element]] = {}
    for child in root:
        groups.setdefault(child.tag, []).append(child)

    if not groups:
        raise LoadError(f"Could not read {name}: <{root.tag}> holds no elements.")
    if len(groups) == 1:
        return next(iter(groups.values())), ()

    repeated = [tag for tag, members in groups.items() if len(members) > 1]
    if len(repeated) == 1:
        tag = repeated[0]
        others = ", ".join(f"<{other}>" for other in groups if other != tag)
        return groups[tag], (
            f"{name} holds <{tag}> repeated among other elements, and <{tag}> "
            f"was loaded as the table. The rest ({others}) are not part of it.",
        )
    if repeated:
        listed = ", ".join(f"<{tag}>" for tag in repeated)
        raise LoadError(
            f"Could not read {name}: it holds more than one table ({listed}), "
            f"and which one you want is not something this server should "
            f"decide. Split the file, or attach it as one table per file."
        )
    raise LoadError(
        f"Could not read {name}: <{root.tag}> holds one each of "
        f"{', '.join(f'<{tag}>' for tag in groups)}, so nothing in it repeats "
        f"as rows do."
    )


def _record_of(element: ElementTree.Element, name: str) -> tuple[dict, list[str]]:
    """One row — attributes then children by tag — and which parts were nested."""
    record: dict[str, object] = dict(element.attrib)
    nested = []

    for child in element:
        if child.tag in record:
            raise LoadError(
                f"Could not read {name}: <{child.tag}> appears more than once "
                f"in a single <{element.tag}>, which makes it a list rather "
                f"than a column. A table cannot hold it without choosing which "
                f"one to keep, and choosing would lose the rest."
            )
        if len(child):
            # Kept whole. Dropping it is what pandas does, and it reports nothing.
            record[child.tag] = ElementTree.tostring(child, encoding="unicode").strip()
            nested.append(child.tag)
        else:
            record[child.tag] = child.text

    if not record:
        raise LoadError(
            f"Could not read {name}: <{element.tag}> has no attributes and no "
            f"child elements, so it names no columns. A row needs named parts."
        )
    return record, nested


def _table_within(document: object, name: str) -> tuple[list[dict], tuple[str, ...]]:
    """Find the one table in a parsed JSON document, or refuse and say why."""
    if isinstance(document, list):
        offender = next(
            (v for v in document if not isinstance(v, dict)),
            None,
        )
        if offender is not None:
            raise LoadError(
                f"Could not read {name}: it is an array of "
                f"{_json_kind(offender)}, and a table needs an array of objects "
                f"— each one a row, its keys the columns."
            )
        return document, ()

    if isinstance(document, dict):
        # Empty arrays are not candidates, which is what makes the common
        # `{"data": [...], "errors": []}` unambiguous rather than a refusal.
        candidates = [
            key
            for key, value in document.items()
            if isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value)
        ]
        if len(candidates) == 1:
            key = candidates[0]
            return document[key], (
                f"{name} is an object rather than an array, and the array under "
                f"{key!r} was the only table in it — that is what was loaded. "
                f"The other keys ({', '.join(k for k in document if k != key)}) "
                f"are not part of this table.",
            )
        if candidates:
            raise LoadError(
                f"Could not read {name}: it holds more than one table "
                f"({', '.join(repr(k) for k in candidates)}), and which one you "
                f"want is not something this server should decide. Split the "
                f"file, or attach it as one table per file."
            )

    raise LoadError(
        f"Could not read {name}: it holds no table. This reader takes an array "
        f"of objects, or an object with exactly one array of objects under it."
    )


def _frame_of_records(records: list[dict], notes: tuple[str, ...]) -> ReadResult:
    """Build a frame from JSON records, encoding anything SQL cannot hold.

    A nested value has no SQL type, so it is written as its JSON text. That is
    lossless and reversible, and unlike dropping or flattening it invents
    nothing — but the resulting column looks like ordinary text, so the note
    names the columns and the function that reads back into them.
    """
    frame = pd.DataFrame(records)

    encoded = []
    for name in frame.columns:
        series = frame[name]
        if not series.map(lambda value: isinstance(value, (dict, list))).any():
            continue
        frame[name] = series.map(
            lambda value: (
                json.dumps(value) if isinstance(value, (dict, list)) else value
            )
        )
        encoded.append(str(name))

    if encoded:
        notes = notes + (
            f"Nested values in {', '.join(encoded)} were stored as JSON text, "
            f"because SQL has no nested type. Read into them with "
            f"json_extract(column, '$.key'); the text is exactly what was in the "
            f"file, so nothing was lost.",
        )
    return _one(frame, notes)


def _json_kind(value: object) -> str:
    """What a JSON value is, in JSON's own words rather than Python's."""
    if value is None:
        return "null"
    return {
        bool: "a boolean",
        int: "a number",
        float: "a number",
        str: "a string",
        list: "an array",
        dict: "an object",
    }.get(type(value), f"a {type(value).__name__}")


#: Extension to reader. The seam through which new formats arrive — a new entry
#: is the whole change, because everything downstream works from the DataFrame.
READERS: dict[str, Reader] = {
    ".csv": _delimited(","),
    ".tsv": _delimited("\t"),
    ".txt": _delimited(","),
    ".json": _read_json,
    ".jsonl": _read_jsonl,
    ".ndjson": _read_jsonl,
    ".xml": _read_xml,
    ".yaml": _read_yaml,
    ".yml": _read_yaml,
    ".fwf": _read_fwf,
    ".parquet": _read_columnar,
    ".feather": _read_columnar,
    ".orc": _read_columnar,
    ".xlsx": _read_workbook,
    ".xlsm": _read_workbook,
    ".xls": _read_workbook,
    ".ods": _read_workbook,
    ".numbers": _read_numbers,
}

#: The formats a delimiter means anything for. Everything else carries its own
#: structure, so being handed a separator for one is a caller's mistake.
DELIMITED = {".csv", ".tsv", ".txt"}


def _readable_suffix(path: Path) -> str:
    """The suffix, if anything here reads it."""
    suffix = path.suffix.lower()
    if suffix not in READERS:
        supported = ", ".join(sorted(READERS))
        raise LoadError(f"No reader for {path.suffix!r}. Supported: {supported}")
    return suffix


def _check_delimiter(path: Path, suffix: str, delimiter: str | None) -> None:
    """Refuse a delimiter that cannot mean anything for this file.

    Refused rather than ignored: a caller who set it believes it did something.
    """
    if delimiter is None:
        return
    if suffix not in DELIMITED:
        listed = ", ".join(sorted(DELIMITED))
        raise LoadError(
            f"delimiter does not apply to {suffix} — only to character-"
            f"separated text ({listed}). {path.name} has its own structure "
            f"and nothing here needs to be told how to split it."
        )
    if len(delimiter) != 1:
        raise LoadError(f"delimiter must be a single character, not {delimiter!r}.")


def read_file(path: Path, *, delimiter: str | None = None) -> ReadResult:
    """Read a tabular file into frames, touching no database state.

    Separate from the insert so a caller can find out whether a file is readable
    *before* committing to it. That ordering matters once slots are limited: a
    file that cannot be parsed must not cost a live datasource its place.

    ``delimiter`` replaces the separator the extension implied, and applies only
    to the delimited formats. Passing it for a format that has no separator is
    refused rather than ignored: a caller who set it believes it did something.
    """
    suffix = _readable_suffix(path)
    _check_delimiter(path, suffix, delimiter)
    reader = _delimited(delimiter) if delimiter is not None else READERS[suffix]

    try:
        result = reader(path)
    except LoadError:
        # A reader that refused for a reason of its own has already said what it
        # was; wrapping it again would bury the specific message under a generic
        # one. Only an unexpected failure needs the file named.
        raise
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    # Recognise the temporal columns before anything downstream sees types.
    # `read_csv` infers numbers and leaves everything else as text, so without
    # this a date is compared as text and `ORDER BY` runs backwards on any
    # spelling whose lexical order is not its chronological one.
    standardized = []
    for table in result.tables:
        frame = temporal.standardize(table.frame)
        if frame.empty and len(frame.columns) == 0:
            named = f" ({table.name})" if table.name else ""
            raise LoadError(f"{path.name}{named} contains no columns.")
        standardized.append(NamedFrame(frame, table.name))

    return ReadResult(tuple(standardized), result.notes)


# ---------------------------------------------------------------------------
# Measuring a source before its table exists
# ---------------------------------------------------------------------------
#
# `read_file` above builds every reader's whole frame before a row is inserted,
# so the load's peak tracks the *file* — 3.0 GB resident against a 1.22 GB CSV
# (CONSTRAINTS §10.6). What stops that being a chunk-size change is that
# everything deciding the *table* is a whole-column measurement made before the
# first insert: the declared type, the width of the widest text value, the
# numeric split of a mixed column, and whether a text column is dates. pandas
# infers dtypes per chunk, so a naive chunked insert declares a column from
# chunk one and meets a value it cannot hold in chunk five.
#
# So the file is read twice. The first pass measures, accumulating each of those
# answers across chunks; the second coerces every chunk to what was measured and
# inserts it. Two properties are kept that are easy to lose:
#
#   * A file that cannot be parsed must not cost a live slot its place
#     (slots.py). Pass one reads the whole file before `_make_room` is called,
#     so a parse error still arrives before any eviction.
#   * The notes are part of the read — the fat-column note, the mixed-column
#     report, the surrogate-key note — and they are produced from measurements,
#     so they belong to pass one.


@dataclass(frozen=True)
class ColumnMeasurement:
    """One column, as it was measured before there was a table to put it in.

    Also everything :meth:`Workspace._described_from` reports, so a column is
    measured once rather than once for the table and again for the description.
    """

    name: str
    declared_type: str
    #: What the values are, so a chunk of raw text can be turned back into them:
    #: ``"int"``, ``"float"``, ``"bool"`` or ``"text"``. ``None`` where the frame
    #: was already typed and there is nothing to turn back.
    family: str | None = None
    longest: int | None = None
    #: The spelling to rewrite this column's dates in, or ``None`` to leave its
    #: text alone — which covers a column that is not temporal *and* one already
    #: canonical throughout, since :func:`temporal.standardize` rewrites neither.
    spelling: temporal.Spelling | None = None
    temporal_kind: str | None = None
    temporal_standard: str | None = None
    numeric_values: int = 0
    non_numeric_values: int = 0
    non_numeric_examples: tuple[str, ...] = ()
    unparsed_temporal_examples: tuple[str, ...] = ()


def measure_frame(frame: pd.DataFrame) -> tuple[ColumnMeasurement, ...]:
    """Measure a frame already in memory, a whole column at a time.

    What every format that is not read in chunks goes through, and the reference
    the chunked measurement is held to: :class:`_ColumnScan` accumulates these
    same fields from parts of a column, and a test loads every fixture both ways
    and compares the schema, the notes and every value.
    """
    measured = []
    for name, label in zip(_unique_columns(list(frame.columns)), frame.columns):
        series = frame[label]
        declared = _declared_type(series)
        numeric = non_numeric = 0
        examples: tuple[str, ...] = ()
        dates: tuple[str, ...] = ()
        standard: str | None = None
        if declared == "TEXT":
            if temporal.is_standard(series):
                standard = "iso8601_utc"
            else:
                numeric, non_numeric, examples = _numeric_split(series)
                dates = temporal.unparsed_temporal_examples(series)
        measured.append(
            ColumnMeasurement(
                name=name,
                declared_type=declared,
                longest=_longest_value(series, declared),
                temporal_kind=binding.column_temporal_kind(series),
                temporal_standard=standard,
                numeric_values=numeric,
                non_numeric_values=non_numeric,
                non_numeric_examples=examples,
                unparsed_temporal_examples=dates,
            )
        )
    return tuple(measured)


#: What a column of these, and nothing else, is read as. ``read_csv`` gives such
#: a column a real boolean dtype, and gives it up the moment anything else — a
#: gap included — appears in it.
_BOOLEANS = ("true", "false")

#: The declared type each family of values gets. Anything not named is text,
#: which is also what an unrecognised family would have to be.
_DECLARED = {"bool": "INTEGER", "int": "INTEGER", "float": "REAL"}


class _ColumnScan:
    """One column's measurements, accumulated a chunk of raw text at a time.

    **Raw text, deliberately.** A chunked read cannot let pandas infer the
    column's type, because it infers per chunk: a column holding ``1`` and ``2``
    in chunk one and ``3a`` in chunk five is read as integers and then as text,
    and the two chunks disagree about what the column is. Reading every chunk as
    strings and rebuilding the verdict from them is what makes the answer a
    property of the column rather than of where the chunk boundaries fell.

    It is also what keeps the string-shaped measurements exact. A column of
    ``007`` inferred as integers measures one character wide instead of three,
    which is a ``VARCHAR2`` too narrow for the values pass two then reads.

    The rebuilt verdict was checked against ``read_csv``'s own inference over
    thirty-nine column shapes — leading zeroes, underscores, hex, ``inf``,
    ``nan``, whitespace, wide integers, booleans with and without gaps — and
    agrees on all of them (CONSTRAINTS §28).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        #: Families seen in chunks that held values, widened once at the end
        #: rather than pairwise as they arrive: the rule reads as a rule that
        #: way, and applying it pairwise would state it four times.
        self._families: set[str] = set()
        self._has_gap = False
        self._had_values = False
        self._rows = 0
        self._longest = 0
        # Three all-or-nothing temporal questions. Each stops being asked as
        # soon as a chunk answers no, because no later chunk can answer yes.
        self._canonical_throughout = True
        self._parses_throughout = True
        self._date_shaped_throughout = True
        #: Which canonical spellings the column arrived in, unioned rather than
        #: folded: two chunks each written one way are each uniform and the
        #: column is not, so a flag per chunk cannot answer this (#75).
        self._canonical_widths: set[int] = set()
        self._spelling: temporal.Spelling | None = None
        self._date_examples: dict[str, None] = {}
        # The numeric split, for the mixed-column report.
        self._numeric = 0
        self._non_numeric = 0
        self._numeric_examples: dict[str, None] = {}

    def observe(self, raw: pd.Series) -> None:
        """Fold one chunk of this column's raw text into the measurements."""
        self._rows += len(raw)
        if bool(raw.isna().any()):
            self._has_gap = True
        present = temporal.text_values(raw)
        if present is None:
            # A chunk holding no values says nothing about the column, and must
            # not be able to answer an all-or-nothing question either — which is
            # why every question below is asked of `present` and not of `raw`.
            return
        self._had_values = True
        self._longest = max(self._longest, int(present.str.len().max()))
        self._observe_family(present)
        self._observe_temporal(present, raw)

    def _observe_family(self, present: pd.Series) -> None:
        """Which family this chunk's values belong to, and their numeric split.

        The two are measured together because the first answers the second for
        free in the common case: if every value parses as a number then the
        split is all-numeric with nothing to give as an example, and the second
        scan is only paid for by a chunk that actually holds something mixed.
        """
        if bool(present.str.lower().isin(_BOOLEANS).all()):
            self._families.add("bool")
            self._split(present)
            return
        try:
            parsed = pd.to_numeric(present, errors="raise")
        except (ValueError, TypeError, OverflowError):
            self._families.add("text")
            self._split(present)
            return
        if pd.api.types.is_object_dtype(parsed.dtype):
            # An integer too wide for int64. `to_numeric` falls back to object
            # and hands back Python ints, and so does `read_csv` — which is why
            # this is its own family rather than text: the column is *declared*
            # text either way, but what the values are matters to what binds.
            self._families.add("wide")
        elif pd.api.types.is_integer_dtype(parsed.dtype):
            self._families.add("int")
        else:
            self._families.add("float")
        self._numeric += len(present)

    def _split(self, present: pd.Series) -> None:
        """Count this chunk's numeric and non-numeric values, and keep examples."""
        numeric, non_numeric, examples = _numeric_split(present)
        self._numeric += numeric
        self._non_numeric += non_numeric
        for value in examples:
            if len(self._numeric_examples) == MAX_NON_NUMERIC_EXAMPLES:
                break
            self._numeric_examples.setdefault(value)

    def _observe_temporal(self, present: pd.Series, raw: pd.Series) -> None:
        if self._canonical_throughout:
            if not temporal.is_canonical(present):
                self._canonical_throughout = False
            else:
                self._canonical_widths |= temporal.canonical_widths(present)

        if self._parses_throughout:
            parsed = temporal.parse(present, raw)
            if parsed is None:
                self._parses_throughout = False
            else:
                found = temporal.spelling_of(parsed)
                self._spelling = (
                    found
                    if self._spelling is None
                    else self._spelling.merged_with(found)
                )

        if self._date_shaped_throughout:
            if not temporal.is_date_shaped(present):
                self._date_shaped_throughout = False
            else:
                for value in present:
                    if len(self._date_examples) == temporal.MAX_TEMPORAL_EXAMPLES:
                        break
                    self._date_examples.setdefault(value)

    def _family(self) -> str:
        """Which family the whole column belongs to, from the ones its parts do.

        A gap is what widens int to float and takes a boolean column away
        altogether, because that is what ``read_csv`` does with one: there is no
        missing marker in either dtype, so the column becomes the one that has
        one. Boolean beside numeric is not a column either can hold, and reads
        as text.

        Every rule here is ``read_csv``'s rather than this module's, and each
        was measured against it rather than reasoned about (CONSTRAINTS §28).
        """
        if not self._families:
            # No value anywhere. A column of *rows* that are all missing reads
            # as float64; a column with no rows at all — a file that is a header
            # and nothing else — is left as object, which is text.
            return "float" if self._rows else "text"
        if "text" in self._families:
            return "text"
        if self._families == {"bool"}:
            return "text" if self._has_gap else "bool"
        if "bool" in self._families:
            return "text"
        if "wide" in self._families:
            # A too-wide integer keeps the column as Python ints for as long as
            # everything in it is an integer, gaps included. One real number in
            # it and `read_csv` gives up and reads the whole column as text.
            return "wide" if self._families <= {"wide", "int"} else "text"
        if "float" in self._families or self._has_gap:
            return "float"
        return "int"

    def finish(self) -> ColumnMeasurement:
        """The column's measurements, now that every chunk has been seen."""
        family = self._family()
        declared = _DECLARED.get(family, "TEXT")

        # Canonical throughout by the end, either because it arrived that way or
        # because pass two will rewrite it into that. `standardize` skips a
        # column that arrived canonical *and in one spelling*, so only the other
        # cases have a spelling to apply — and only those are sized from the
        # spelling, since a column left alone keeps the text it came with.
        #
        # A column mixing canonical spellings is rewritten like any other: it
        # arrived canonical, but not in the one spelling that makes it sort
        # (#75), and pass two settles it on the merged one.
        arrived_in_one_spelling = (
            self._canonical_throughout and len(self._canonical_widths) == 1
        )
        standard = declared == "TEXT" and self._had_values and self._parses_throughout
        spelling = self._spelling if standard and not arrived_in_one_spelling else None

        if declared != "TEXT":
            longest = None
        elif spelling is not None:
            longest = temporal.canonical_width(spelling)
        else:
            longest = self._longest

        return ColumnMeasurement(
            name=self.name,
            declared_type=declared,
            family=family,
            longest=longest,
            spelling=spelling,
            temporal_standard="iso8601_utc" if standard else None,
            numeric_values=0 if standard or declared != "TEXT" else self._numeric,
            non_numeric_values=(
                0 if standard or declared != "TEXT" else self._non_numeric
            ),
            non_numeric_examples=(
                () if standard or declared != "TEXT" else tuple(self._numeric_examples)
            ),
            unparsed_temporal_examples=(
                tuple(self._date_examples)
                if declared == "TEXT"
                and not standard
                and self._had_values
                and self._date_shaped_throughout
                else ()
            ),
        )


#: How a chunk of raw text becomes the values the measurements settled on.
#: ``float`` is cast rather than left as ``to_numeric`` returns it, because a
#: chunk holding only whole numbers comes back as integers while the column was
#: measured to hold reals — the same column read whole is float throughout.
_COERCE: dict[str, Callable[[pd.Series], pd.Series]] = {
    "int": lambda values: pd.to_numeric(values),
    "float": lambda values: pd.to_numeric(values).astype("float64"),
    "bool": lambda values: values.str.lower().map({"true": True, "false": False}),
    # Python ints in an object column, which is what `read_csv` gives a column
    # of integers too wide for int64. Reproduced rather than improved on: the
    # value is then refused by `binding.adapt_value` for being unrepresentable,
    # and a file the materialised path refuses must not load here (issue #72).
    "wide": lambda values: pd.to_numeric(values),
    "text": lambda values: values,
}


def _as_measured(
    chunk: pd.DataFrame, columns: tuple[ColumnMeasurement, ...]
) -> pd.DataFrame:
    """Turn one chunk of raw text into the values the measurements settled on."""
    converted = {
        label: _COERCE[column.family or "text"](chunk[label])
        for label, column in zip(chunk.columns, columns)
    }
    frame = pd.DataFrame(converted, index=chunk.index)
    spellings = {
        label: column.spelling
        for label, column in zip(chunk.columns, columns)
        if column.spelling is not None
    }
    return temporal.standardize_as(frame, spellings) if spellings else frame


@dataclass(frozen=True)
class SourceTable:
    """One table out of a source: what it holds, and how to read it again.

    ``chunks`` is a callable rather than an iterator because it is called more
    than once. A backend that cannot see a table it made inside the transaction
    that made it has its rows written in a second one, and an iterator already
    walked would write nothing into it — silently, since an empty insert is not
    an error. Calling it re-reads the source from the beginning.
    """

    name: str | None
    columns: tuple[ColumnMeasurement, ...]
    chunks: Callable[[], Iterator[pd.DataFrame]]


@dataclass(frozen=True)
class SourceRead:
    """Every table in a source, measured, with whatever the reader had to assume."""

    tables: tuple[SourceTable, ...]
    notes: tuple[str, ...] = ()


#: The formats read a chunk at a time. Each is read by a pandas entry point that
#: takes a ``chunksize``, and each is a format whose rows are *lines* — which is
#: what makes reading part of one meaningful. A workbook, a JSON document, XML,
#: YAML and ``.numbers`` are parsed whole by the libraries that read them, so
#: they stay materialised and the limit is stated rather than worked around.
STREAMED = {".csv", ".tsv", ".txt", ".fwf"}


def _chunk_reader(
    path: Path, suffix: str, delimiter: str | None
) -> Callable[[], Iterator[pd.DataFrame]]:
    """A callable giving fresh chunks of this file, every column as raw text."""

    def chunks() -> Iterator[pd.DataFrame]:
        if suffix == ".fwf":
            reader = pd.read_fwf(path, dtype=str, chunksize=_READ_CHUNK)
        else:
            separator = delimiter or ("\t" if suffix == ".tsv" else ",")
            reader = pd.read_csv(path, sep=separator, dtype=str, chunksize=_READ_CHUNK)
        with reader as opened:
            yield from opened

    return chunks


def _streamed_notes(
    path: Path, suffix: str, first: pd.DataFrame, sep: str
) -> tuple[str, ...]:
    """What the reader had to assume, which the header alone is enough to say."""
    if suffix == ".fwf":
        return (
            f"{path.name} is fixed-width, so its column boundaries were inferred "
            f"from which character positions are blank on every line — nothing "
            f"in the file declares them. Check the columns are the ones you "
            f"expect before relying on the split.",
        )
    return _fat_column_note(first, sep, path.name)


def read_source(path: Path, *, delimiter: str | None = None) -> SourceRead:
    """Measure a source, in chunks where the format allows it.

    The one entry point the load path uses, so that a format read in chunks and
    a format read whole reach the insert as the same thing. What differs between
    them is only where the measurements came from.

    Like :func:`read_file`, this touches no database state and reads the whole
    source before returning — which is what lets a caller find out that a file
    is unreadable before it costs a live datasource its place.
    """
    suffix = _readable_suffix(path)
    _check_delimiter(path, suffix, delimiter)

    if suffix not in STREAMED:
        read = read_file(path, delimiter=delimiter)
        return SourceRead(
            tables=tuple(
                SourceTable(
                    name=table.name,
                    columns=measure_frame(table.frame),
                    chunks=(lambda frame=table.frame: iter([frame])),
                )
                for table in read.tables
            ),
            notes=read.notes,
        )

    chunks = _chunk_reader(path, suffix, delimiter)
    separator = delimiter or ("\t" if suffix == ".tsv" else ",")
    scans: list[_ColumnScan] | None = None
    notes: tuple[str, ...] = ()
    try:
        for chunk in chunks():
            if scans is None:
                scans = [
                    _ColumnScan(name) for name in _unique_columns(list(chunk.columns))
                ]
                notes = _streamed_notes(path, suffix, chunk, separator)
            for scan, label in zip(scans, chunk.columns):
                scan.observe(chunk[label])
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    if not scans:
        # A file whose header names no columns. `read_csv` raises for a wholly
        # empty one, so this is the header-only-and-empty case, which the
        # materialised path refuses in the same words.
        raise LoadError(f"{path.name} contains no columns.")

    measured = tuple(scan.finish() for scan in scans)

    def coerced() -> Iterator[pd.DataFrame]:
        """Pass two: the same chunks, as the values pass one settled on."""
        for chunk in _chunk_reader(path, suffix, delimiter)():
            yield _as_measured(chunk, measured)

    return SourceRead(
        tables=(SourceTable(name=None, columns=measured, chunks=coerced),),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    def attach_file(
        self, tag: str, path: Path, *, dialect: str = "sqlite", readonly: bool = True
    ) -> None:
        """Open an existing database file, read-only unless told otherwise.

        Writable is not a flag this module honours by being careful — it is a
        different URI, and the database itself is what refuses the write.

        ``dialect`` is passed in rather than derived, and that is not a dispatch:
        the caller has already established *which* database this file is by
        reading its header (``slots.FILE_SIGNATURES``). Naming a fact is not the
        same as guessing a type.

        **How a file becomes a URL is the backend's answer, not this method's.**
        Every dialect has an ``open_file``, generic unless it has earned an
        override, so nothing here asks which database it is holding.
        """
        backend = backend_for(dialect)
        location = str(path.resolve())
        engines = backend.open_file(path, writable=not readonly)

        self._install(
            tag,
            uri=location,
            location=location,
            backend=backend,
            engines=engines,
        )

    def attach(self, url: str | URL, tag: str, *, writable: bool = False) -> None:
        """Open any datasource SQLAlchemy can reach, under ``tag``.

        **The URL is the abstraction.** It carries which database this is,
        SQLAlchemy parses it, and :func:`dialects.backend_for_url` looks up
        whatever that dialect adds — finding nothing, most of the time, which is
        the ordinary case and not a failure.

        What this method does *not* assume is that the dialect names the engine.
        Several do not: YugabyteDB, Greenplum and OpenGauss all answer on
        PostgreSQL's, TiDB and OceanBase on MySQL's. So the lookup goes through
        ``backend_for_url``, which asks the server which it is — and asks only
        where the dialect is one somebody else borrows. See issue #45.

        A tag opened this way is a tag like any other: the same ``query``, the
        same ``describe``, the same ``insert_frame``. There is deliberately no
        second code path for "remote" datasources, because a second path is how
        one of them silently stops supporting a verb the other has.
        """
        parsed = make_url(url)
        safe = parsed.render_as_string(hide_password=True)
        backend = backend_for_url(parsed)
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

    def resolve_table(self, tag: str, table: str) -> str | None:
        """What this tag calls the table a caller named, or ``None`` for none.

        Existence is the database's question and not Python's. SQLite resolves
        an unquoted identifier case-insensitively, so a table stored as
        ``MyTable`` is one ``query`` reads as ``mytable`` — and a verb that
        takes the same name as an *argument* and compares it exactly would
        refuse a table the slot beside it can read, while listing that table as
        what the slot holds.

        Exact match wins outright, so a backend that stores two names differing
        only in case still resolves each of them to itself. A fold resolves only
        when exactly one name matches: two candidates is genuine ambiguity on a
        case-sensitive backend, and guessing between them would drop or rename
        the wrong table.
        """
        names = self.table_names(tag)
        if table in names:
            return table
        folded = [name for name in names if name.lower() == table.lower()]
        return folded[0] if len(folded) == 1 else None

    def has_table(self, tag: str, table: str) -> bool:
        return self.resolve_table(tag, table) is not None

    def table_names(self, tag: str) -> tuple[str, ...]:
        """Everything in this tag's database that can be selected from.

        Views included, and deliberately: a caller who cannot see one in the
        listing has no way to learn it is there. They describe like tables and
        are queried like tables, so telling them apart here would be a
        distinction without a use.
        """

        def read() -> tuple[str, ...]:
            inspector = inspect(self.entry(tag).engines.read)
            names = set(inspector.get_table_names()) | set(inspector.get_view_names())
            return tuple(sorted(n for n in names if not n.startswith("sqlite_")))

        return _run_again_once(read)

    def landed_as(self, tag: str, wanted: str) -> str:
        """What a table just created or renamed is *actually* called.

        Asked rather than assumed, and that is the whole of it: a database may
        store a name in a case it chose itself. Trino lower-cases every
        identifier at the connector, quoted or not, so a table asked for as
        ``Mixed`` is called ``mixed`` — and reporting ``Mixed`` back names a
        table the caller will not find in the listing they get in the same
        payload.

        Deliberately observed rather than looked up per dialect. A name is a
        fact the database will state if asked, so asking costs one reflection
        and needs no dialect knowledge; a table of which backends fold would be
        a dispatch on dialect name, and one that went stale would fail silently.

        ``wanted`` is returned unchanged when nothing matches it — which is what
        every non-folding backend hits on the first line, and what a fold this
        does not recognise (a truncation, say) degrades to. That is the previous
        behaviour, so this can only improve an answer, never break one.
        """
        return self.resolve_table(tag, wanted) or wanted

    def rename_table(self, tag: str, table: str, to: str) -> str:
        """Rename a table, moving its cached description with it, and say to what.

        The name comes back because it is not always the name that was asked
        for — see :meth:`landed_as`. The bookkeeping matters as much as the DDL:
        ``_tables`` is keyed on ``tag.table`` and a stale entry would leave the
        old name describable after it stopped existing — the same class of
        defect as the slot listing that reported a table it no longer had. Keyed
        under the *wanted* name on a backend that folded it, the entry would be
        just as stale, and this is where that is settled for every caller.
        """
        entry = self.entry(tag)
        try:
            with entry.engines.write.begin() as conn:
                entry.backend.rename_table(conn, table, to)
        except SQLAlchemyError as exc:
            raise LoadError(f"Could not rename {tag}.{table}: {exc}") from exc

        landed = self.landed_as(tag, to)
        known = self._tables.pop(f"{tag}.{table}", None)
        if known is not None:
            moved = replace(known, name=landed)
            self._tables[moved.qualified] = moved
        return landed

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
        this is the server asking about the schema, not the caller's statement
        running, so it does not go through the posture the read engine carries
        on the caller's behalf.
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
        # How to index a column is the backend's to answer: MySQL will not key
        # on a whole TEXT column at all, and returns an index over a prefix of
        # it together with the words for what that cost.
        index, notes = entry.backend.build_index(name, target, columns)
        try:
            with entry.engines.write.begin() as conn:
                index.create(conn)
        except _driver_failures(entry) as exc:
            raise LoadError(f"Could not create {name} on {tag}.{table}: {exc}") from exc
        return IndexInfo(name=name, table=table, columns=tuple(columns), notes=notes)

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
        except _driver_failures(entry) as exc:
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
        self,
        raw_path: str,
        tag: str,
        table_name: str | None = None,
        delimiter: str | None = None,
    ) -> list[TableInfo]:
        """Read a tabular file into ``tag`` and describe every table that landed.

        A list, because a file is not always one table: a workbook's sheets are
        each a table, and returning only the first would leave the rest present
        in the file and unreachable through the server.
        """
        path = resolve_read_path(raw_path)
        read = read_source(path, delimiter=delimiter)

        if len(read.tables) > 1 and table_name is not None:
            named = ", ".join(str(table.name) for table in read.tables)
            raise LoadError(
                f"{path.name} holds {len(read.tables)} tables ({named}), so one "
                f"name cannot cover them — they keep the names the file gives "
                f"them. Drop table_name, or point at a file holding one table."
            )

        return [
            self.insert_source(
                table,
                _sanitize(table_name or table.name or path.stem, "table"),
                source=str(path),
                tag=tag,
                # The notes describe reading the file, so they belong to every
                # table that came out of it.
                notes=read.notes,
            )
            for table in read.tables
        ]

    def insert_frame(
        self,
        frame: pd.DataFrame,
        table: str,
        *,
        source: str,
        tag: str,
        notes: tuple[str, ...] = (),
    ) -> TableInfo:
        """Load a frame that is already in memory.

        A source of exactly one chunk, so that a frame and a file read in chunks
        travel the same path from here on. Kept as its own method because a
        caller holding a frame should not have to build the wrapper.
        """
        return self.insert_source(
            SourceTable(
                name=None,
                columns=measure_frame(frame),
                chunks=lambda: iter([frame]),
            ),
            table,
            source=source,
            tag=tag,
            notes=notes,
        )

    def insert_source(
        self,
        source_table: SourceTable,
        table: str,
        *,
        source: str,
        tag: str,
        notes: tuple[str, ...] = (),
    ) -> TableInfo:
        """Make the table these measurements describe and stream the rows in.

        The one insert path. A file read a chunk at a time and a frame held
        whole differ only in where their :class:`ColumnMeasurement` came from
        and how many chunks arrive; everything about making the table, splitting
        the transaction, filling the surrogate key and describing the result is
        the same, and a second copy of it would be free to drift.
        """
        entry = self.entry(tag)
        measured = source_table.columns
        # The file's own columns. Distinct from `columns` below, which describes
        # the *table* — the two differ by exactly the surrogate key.
        from_file = [column.name for column in measured]

        # A backend that will not make a table without a primary key gets a
        # surrogate one, because a file has none to offer: nothing in a CSV is
        # guaranteed unique, so any column nominated here would be a constraint
        # this server invented on the caller's data. True only for YDB.
        #
        # The *fact* is the backend's and the *response* is this module's — one
        # response, the same for any dialect that ever states the fact, which is
        # what keeps the answer from being written twice. The added column is
        # **reported**: it is described in `info` like any other column and the
        # notes say why it is there, rather than surprising somebody who later
        # runs `SELECT *`.
        keyed = entry.backend.requires_primary_key()

        if keyed:
            notes = (
                *notes,
                f"A {entry.backend.name} table must declare a primary key and a "
                f"file has none to offer, so the column {_SURROGATE_KEY} was "
                f"added, holding each row's position in the file. The file's own "
                f"columns and values are unchanged.",
            )

        target = Table(
            table,
            MetaData(),
            # The surrogate first, so a `SELECT *` reads as the file does with an
            # ordinal in front of it rather than one tacked on the end.
            *([Column(_SURROGATE_KEY, Integer, primary_key=True)] if keyed else []),
            *[
                # What a declared type is *called* in SQL is the backend's to
                # say: SQLite's affinity depends on the exact token, Oracle's
                # portable text type cannot be grouped on, and PostgreSQL's REAL
                # is only four bytes wide.
                Column(
                    column.name,
                    entry.backend.column_type(
                        column.declared_type, longest=column.longest
                    ),
                )
                for column in measured
            ],
            # Anything this dialect's CREATE TABLE cannot be written without.
            # Empty for all but ClickHouse, which has no default table engine —
            # and passed through without being read here, so that what it
            # contains stays the backend's business rather than this module's.
            **entry.backend.table_options(),
        )

        # Whether the schema and the rows may travel together. True everywhere
        # but Firebird, which keeps DDL transactional *and* prepares statements
        # against committed metadata — so the table it has just made is invisible
        # to the transaction that made it, and the insert fails with `-204 Table
        # unknown` (issue #53). Asked once and named, because the two branches
        # below have to agree about the answer.
        #
        # Splitting is the worse shape and is used only where the alternative is
        # that the write cannot happen: with the CREATE committed first, a failure
        # part-way through the rows leaves an empty table where a single
        # transaction would have left nothing. The drop-and-create pair stays
        # together either way — it is only the DDL→DML boundary that has to give.
        together = entry.backend.sees_new_tables_in_transaction()

        chunks = source_table.chunks
        try:
            with entry.engines.write.begin() as conn:
                target.drop(conn, checkfirst=True)
                target.create(conn)
                if together:
                    self._fill(entry, conn, target, chunks, from_file, table, keyed)
            if not together:
                with entry.engines.write.begin() as conn:
                    self._fill(entry, conn, target, chunks, from_file, table, keyed)
        except Exception as exc:
            unrepresentable = _unrepresentable(exc)
            if unrepresentable is not None:
                raise LoadError(
                    f"{source}: {unrepresentable.reason}. Value "
                    f"{unrepresentable.value!r} cannot be stored in a 64-bit "
                    f"column without corrupting it."
                ) from exc
            if isinstance(exc, (*_driver_failures(entry), OverflowError)):
                raise LoadError(f"Could not insert rows from {source}: {exc}") from exc
            raise

        info = TableInfo(
            # Under the name the database gave it, which is not always the one
            # asked for — see landed_as. Reporting the requested name would put
            # a table in this payload that is absent from the listing in the
            # next one, on the backend that folds.
            name=self.landed_as(tag, table),
            row_count=self._count(entry, table),
            columns=self._described_from(entry, table, measured, keyed),
            source=source,
            tag=tag,
            notes=notes,
        )
        self._tables[info.qualified] = info
        return info

    def _fill(
        self,
        entry: Tagged,
        conn: Connection,
        target: Table,
        chunks: Callable[[], Iterator[pd.DataFrame]],
        columns: list[str],
        table: str,
        keyed: bool = False,
    ) -> None:
        """Put the source's rows into a table that already exists, then settle it.

        Extracted from :meth:`insert_frame` when Firebird made the transaction
        boundary a per-backend question (#53), and extracted rather than
        duplicated so that both answers write rows the *same* way. The thing that
        varies between them is which transaction this runs in; nothing about the
        writing itself does, and a second copy of this loop would be free to
        drift.

        ``chunks`` is *called* here rather than iterated by the caller, so that
        the split-transaction branch re-reads the source instead of walking an
        iterator the first branch already consumed.

        ``keyed`` says the table carries the surrogate key
        :meth:`insert_source` adds where a backend refuses a keyless table, and
        the value it must be given is the row's position in the file. ``columns``
        stays the file's own columns either way — the key is filled from the row
        offset rather than from the data, because the data does not have it.
        """
        statement = target.insert()
        # Nothing is ever materialised as rows — only one chunk of the source and
        # one block of that chunk exist at a time. See the module docstring.
        rows_so_far = 0
        for chunk in chunks():
            for block in self._blocks(
                chunk, columns, _SURROGATE_KEY if keyed else None, rows_so_far
            ):
                conn.execute(statement, block)
            rows_so_far += len(chunk)
        # Nothing for every backend that makes a committed write readable, which
        # is all of them but the search-engine lineage. Inside this block on
        # purpose: a write and the visibility of that write must not be separable
        # by a failure between them.
        entry.backend.settle(conn, table)

    @staticmethod
    def _blocks(
        frame: pd.DataFrame,
        columns: list[str],
        key: str | None = None,
        base: int = 0,
    ) -> Iterator[list[dict]]:
        """Frame rows as bind-parameter mappings, one insertable chunk at a time.

        A generator, not a list comprehension: the difference between the two is
        0.81 MB and 511 MB at 800,000 rows.

        Each chunk is converted **by column** rather than by value
        (:func:`binding.adapt_column`), which is both the driver-independence and
        the reason the conversion is close to free — a typed column becomes a
        list of Python natives in one numpy pass. Chunking before converting is
        what keeps the peak flat: converting the whole column first would
        materialise exactly what the generator exists to avoid.

        ``key``, where given, names a surrogate primary-key column that is filled
        with the row's position in the file — ``base``, how many rows of the file
        came before this frame. Both are offsets rather than a running counter,
        so the ordinal is a property of the row and not of how anything happened
        to chunk it: the same file always produces the same keys, whatever
        ``_INSERT_CHUNK`` and ``_READ_CHUNK`` are.
        """
        for start in range(0, len(frame), _INSERT_CHUNK):
            chunk = frame.iloc[start : start + _INSERT_CHUNK]
            adapted = [binding.adapt_column(chunk[name]) for name in chunk.columns]
            rows = [dict(zip(columns, values)) for values in zip(*adapted)]
            if key is not None:
                for offset, row in enumerate(rows):
                    row[key] = base + start + offset
            yield rows

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
        # SQL running, so it must not be subject to the read posture the
        # backend installs on the caller's behalf. That posture is a
        # ``Backend.read_posture`` hook and each dialect decides how strict it
        # is; SQLite's is strict enough to refuse introspection itself. No write
        # ability is implied either way — a read-only datasource is opened
        # read-only on *both* engines, so the database refuses a write whichever
        # one asks.
        # Bound to a *connection* rather than to the engine. Both are ordinary
        # SQLAlchemy, but a dialect supplying its own inspector may use the bind
        # directly, and one that does raises on an engine — ClickHouse's does.
        # Opening the connection here is what the engine form would have done
        # anyway, so nothing is paid for it.
        # The connection is opened *inside* the retried read, not around it: a
        # transaction refused for a stale snapshot stays refused, so running the
        # inspection again on the same connection would meet the same answer.
        def read() -> list[Any]:
            with entry.engines.write.connect() as conn:
                return inspect(conn).get_columns(table)

        described = _run_again_once(read)
        if not described:
            raise LoadError(f"No such table: {tag}.{table}")

        columns = [column["name"] for column in described]
        declared = [str(column["type"]) for column in described]
        return TableInfo(
            name=table,
            row_count=self._count(entry, table),
            columns=self._describe_columns(entry, table, columns, declared),
            source=source,
            tag=tag,
        )

    def _describe_columns(
        self, entry: Tagged, table: str, columns: list[str], declared: list[str]
    ) -> list[ColumnInfo]:
        """Describe a table nothing here loaded, from its declared types alone.

        Everything the load path also reports — whether a column is dates, how
        its values split between numbers and junk — was measured *from the
        source*, and a table that arrived in the database rather than through a
        load has no source to measure. So this says what the declared type says
        and nothing more, which is all there is to say about it truthfully.
        """
        with entry.engines.read.connect() as conn:
            return [
                ColumnInfo(
                    name=name,
                    declared_type=sql_type,
                    storage_classes=entry.backend.storage_classes(conn, table, name),
                )
                for name, sql_type in zip(columns, declared)
            ]

    def _described_from(
        self,
        entry: Tagged,
        table: str,
        measured: tuple[ColumnMeasurement, ...],
        keyed: bool,
    ) -> list[ColumnInfo]:
        """Describe a table just loaded, from what its source was measured to hold.

        The storage classes are the one thing asked of the database rather than
        of the measurements, and deliberately: they are what the values *became*
        once stored, which is a different question from what the file held and
        is the question the mixed-column report exists to answer.

        The surrogate key, where a backend demanded one, has no measurement
        behind it — it is a column this server added rather than read — so it is
        described from its declared type, which is the whole truth about it.
        """
        with entry.engines.read.connect() as conn:
            described = []
            if keyed:
                described.append(
                    ColumnInfo(
                        name=_SURROGATE_KEY,
                        declared_type="INTEGER",
                        storage_classes=entry.backend.storage_classes(
                            conn, table, _SURROGATE_KEY
                        ),
                    )
                )
            described.extend(
                ColumnInfo(
                    name=column.name,
                    declared_type=column.declared_type,
                    temporal_kind=column.temporal_kind,
                    storage_classes=entry.backend.storage_classes(
                        conn, table, column.name
                    ),
                    numeric_values=column.numeric_values,
                    non_numeric_values=column.non_numeric_values,
                    non_numeric_examples=column.non_numeric_examples,
                    temporal_standard=column.temporal_standard,
                    unparsed_temporal_examples=column.unparsed_temporal_examples,
                )
                for column in measured
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
        verbs (``create``, ``update``, ``drop``) which do not come through this
        method, so the refusal costs the surface nothing.

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

        **A statement that returns no rows — or returns rows of no columns — is
        not a read**, and is refused on that ground alone; no SQL is parsed to
        decide it. A ``SELECT`` returns rows even when it matches none, and it
        always projects at least one column, so the distinction is exact rather
        than heuristic. This is the floor under every dialect: where the database
        itself refuses a write on a read-only connection (SQLite's authorizer,
        DuckDB's ``access_mode``, a read-only session on the servers that have
        one) the refusal arrives before this, and where it does not, a write that
        was quietly rolled back would otherwise be reported as a statement that
        succeeded and returned nothing.

        **The column half was added because ``returns_rows`` alone let a write
        through.** CrateDB answers an ``INSERT`` with a result its driver reports
        as returning rows — one row, of *zero* columns — so the statement passed
        the floor and came back ``ok`` while the row it inserted stayed inserted,
        there being no transaction to withhold. That is the exact lie this
        function exists to prevent, and it was not a CrateDB fact so much as an
        assumption in shared code: that a driver saying "rows" means a caller was
        reading. Asking for a column as well is the same question asked
        completely, and it stays generic — no dialect is named to enforce it.
        """
        with self.query_stream(tag, sql) as (names, rows):
            return names, list(rows)

    @contextmanager
    def query_stream(
        self, tag: str, sql: str
    ) -> Iterator[tuple[list[str], Iterator[tuple]]]:
        """The same read as :meth:`query`, handed back **unmaterialised**.

        Yields ``(column_names, rows)`` where ``rows`` is a lazy iterator over
        the open cursor, valid only inside the ``with`` block — the connection
        closes on the way out, so a caller who wants the rows afterwards wants
        :meth:`query` instead.

        This exists for the one caller that never needs the rows in memory: a
        result being written to a file. ``query`` already streamed from the
        driver, but it then built a list of every row and handed *that* to the
        writer, so a result on its way to disk existed twice — once as a list of
        tuples, and again in whatever the writer itself builds. Nine of the
        fifteen export suffixes write row by row and need neither copy.

        ``query`` is this method plus a ``list`` rather than the two sharing
        copied code: one read path, and one place where the read-only posture
        and the returns-rows refusal are decided.

        **A failure part-way through the result is explained the same way**, by
        ``_rows`` below rather than by the ``except`` here: the rows are pulled
        from the caller's frame, so a driver error raised on the way to the
        third million rows never passes through this function at all.
        """
        entry = self.entry(tag)
        entry.engines.refusal.take()
        try:
            with entry.engines.read.connect() as conn:
                result = conn.execution_options(
                    stream_results=True, yield_per=_YIELD_PER
                ).execute(text(sql))
                names = list(result.keys()) if result.returns_rows else []
                if not names:
                    raise LoadError(_not_a_read(entry))
                yield names, self._rows(entry, result, sql)
        except _driver_failures(entry) as exc:
            raise self._explain(entry, exc, sql) from exc

    def _rows(self, entry: Tagged, result: Any, sql: str) -> Iterator[tuple]:
        """Pull rows off an open cursor, explaining a failure that arrives late.

        A generator expression would do everything this does except the last
        part. A cursor streaming a large result can fail after it has already
        handed back rows — a lost connection, a server-side timeout — and that
        failure is raised inside whichever frame is iterating. Written as its
        own generator, the ``try`` lives in the frame the rows come from, so the
        error is translated wherever it is consumed.
        """
        try:
            for row in result:
                yield tuple(row)
        except _driver_failures(entry) as exc:
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
                f"one use drop, to rename one use update; there is no verb for "
                f"arbitrary DDL by design."
            )

        if entry.backend.denies_write(exc):
            # The backend refused the statement itself and said so in its own
            # words, which name no verb the caller could use instead.
            return LoadError(_not_a_read(entry))

        message = str(exc.orig) if getattr(exc, "orig", None) else str(exc)
        if _objected_to_the_leading_verb(message, sql):
            return LoadError(_not_a_read(entry))

        stale = re.search(rf"no such table:\s*{re.escape(entry.tag)}\.(\w+)", message)
        if stale is not None:
            table = stale.group(1)
            return LoadError(
                f"No such table: {entry.tag}.{table}. Tables are addressed by "
                f"their own name inside the datasource you named — write "
                f"FROM {table}, not FROM {entry.tag}.{table}. Available here: "
                f"{', '.join(self.table_names(entry.tag)) or 'none'}."
            )

        missing = _missing_table(message)
        if missing is not None:
            return LoadError(
                f"No such table: {entry.tag}.{missing}. In {entry.tag}: "
                f"{', '.join(self.table_names(entry.tag)) or 'none'}."
            )
        return LoadError(message)

    def close(self) -> None:
        for tag in list(self._tags):
            self.detach(tag)
