"""The MCP tool surface.

There is one idea to learn here, and everything else follows from it: **a
datasource is attached under a nickname, and the nickname is a database.** A CSV,
a SQLite file and a service URL all become the same kind of thing. Each call names
the datasource it is for, and the SQL then addresses tables inside it by their own
names:

    query(nickname="shop", sql="SELECT * FROM sales WHERE qty > 10")

One statement reaches one datasource. Looking two of them up against each other is
``create``, which copies the second *into* the first — a named act, rather than
something that falls out of how the databases happen to be connected.

Because a slot is a database rather than a view over a file, it has the verbs a
database has: lifecycle (``attach``, ``detach``, ``save``), composition
(``create``, ``update``, ``drop``), introspection (``directory``) and the profile that
says whether the values inside can be trusted (``stats``). Nine in total, several
of them multi-faceted — **few and multi-faceted beats many and narrow**, because
the model has less to choose between and each choice is obvious.

``stats`` is the newest and is a verb rather than a flag on ``directory`` because they
answer different questions: ``directory`` is a directory — what is attached, what
tables, what shape — and a directory that reports statistics stops being one.
Nothing else here reports a missing value, which is what made an empty
``mixed_columns`` readable as a clean bill of health (issue #94).

``update`` completes the composition triad and arrived with the formats that name
their own tables: a workbook's sheets land under the names the spreadsheet chose,
and renaming one beats re-reading the file to get a different name.

**These are raw capabilities, not a workflow.** Nothing here infers a join key,
decides that an index would help, or turns an anti-join into a sentence. Those
are acts of judgement that need to know what the user asked and what they called
things, and the caller is the only party holding either. What this module owes
them is primitives that compose and refusals that say what to do instead.

**A query reads.** Every write — ``INSERT``, ``CREATE TABLE``, ``CREATE VIEW``,
``PRAGMA`` — is refused by ``query`` whatever the datasource itself permits, so
mutation happens only through the composition verbs and there is exactly one way
to change a slot. Enforced by the connection: reads go over an engine whose
connections are read-only from the moment they are opened, so nothing here parses
SQL to decide and there is no window in which the posture is anything else.

Two behaviours are deliberately invisible from out here. A database that outgrows
the memory budget is moved to a temp file and goes on answering under the same nickname
between one call and the next, and no result mentions it. And write access is not
a flag this module honours by being careful: a read-only slot carries ``mode=ro``
in its own connection URI, so SQLite is what refuses the write.

Concurrency note: this server dispatches tool calls concurrently, and a
synchronous tool body runs on a worker OS thread — measured, two blocking calls
overlapped for 1.2 s on two different threads. The registry's host connection is
therefore guarded by a lock rather than assumed to be reached from one thread.
"""

from __future__ import annotations

import datetime as _dt
import functools
import threading
from decimal import Decimal
from collections.abc import Callable
from typing import Any
from uuid import UUID

from fastmcp import FastMCP

from . import config
from .export import ExportError, export_rows
from .loader import ColumnProfile, IndexInfo, TableInfo
from .paths import PathNotAllowed, allowed_paths
from .slots import Attachment, Registry, Slot, SlotError, SlotNotAvailable

mcp = FastMCP(
    "localdata",
    instructions=(
        "SQL over local data files and databases.\n\n"
        "Attach each datasource with attach(database) — a tabular file (CSV, "
        "TSV, JSON, YAML, XML, Parquet and more), a SQLite or DuckDB "
        "database file, or a database URL. **Every datasource becomes a database, "
        "named by a nickname**, so even a single CSV holds its rows in a table. "
        "attach returns the nickname it actually used, which may not be the one "
        "you asked for — use what it returns.\n\n"
        "**Each call names one datasource, and the SQL addresses tables inside it "
        "by their own names**: query(nickname='shop', sql='SELECT * FROM sales'), "
        "not FROM shop.sales. One statement reaches one datasource. To look a "
        "second file up against one already open, use create(nickname, "
        "type='table', source=...) to land it *inside* that database, then join "
        "the two tables there in an ordinary statement. Whether the join is "
        "complete is an anti-join you write yourself; if it is slow, "
        "create(nickname, type='index', table=..., columns=[...]) first — "
        "refused on ClickHouse, Trino, CrateDB, Databend and Exasol, each for "
        "its own reason, which the refusal names — and directory(nickname, table) "
        "says which indexes are already there.\n\n"
        "**query only reads.** INSERT, UPDATE, CREATE TABLE and every other write "
        "are refused there whatever the datasource allows — composition has its "
        "own verbs, create, update and drop, and those are the ones the writable "
        "grant governs. A table keeps the name its file gave it until you "
        "update(nickname, type='table', name=..., to=...), which matters for a "
        "workbook whose sheets are named Sheet1; that one is refused on "
        "Firebird, which has no rename-table statement. The whole result comes "
        "back, so ask for what you want: use SQL LIMIT, name your columns "
        "instead of SELECT *, or pass path= to write a large result to a file "
        "rather than into the answer.\n\n"
        "Files you attach are read-only unless you pass writable=true; a database "
        "built from a flat file is yours and is always writable. Slots are limited "
        "and the oldest is evicted when the limit is reached, so check the "
        "'evicted' field an attach returns, and detach what you are done with. "
        "Nothing survives the session unless you save it — and save writes out "
        "only a database this server holds, so it is refused on every backend "
        "but SQLite — a DuckDB file and a URL-attached database included. To "
        "keep rows from one of those, write them to a file with "
        "query(nickname, sql, path=...), attach that file — a file-derived "
        "datasource is yours and is writable — and save that.\n\n"
        "**stats(nickname, table) says whether the data is any good.** It is the "
        "only call reporting missing values: every column comes back with nulls "
        "and non_nulls, a numeric one adds min, max and avg, and median and "
        "stddev appear only where the database computes them itself. Read nulls "
        "before reporting an average — avg() skips them silently. A column "
        "answering with 'withheld' and no statistics is one where an aggregate "
        "would be wrong rather than approximate, and it says which.\n\n"
        "Every call answers with a payload, and a refusal is an ordinary answer "
        'rather than an error: it comes back as {"ok": false, "error": "…"} '
        "with the reason in plain words and, where a name was wrong, the names "
        "that were right. Branch on 'ok'."
    ),
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

# ponytail: one global lock serialises every tool call. Correct, and cheap at
# the scale of interactive use. A process-wide lock was measured starving a
# concurrent reader under a 200k-row load (one completed read per run), so if
# throughput under load matters, this is the thing to replace — with separate
# reader and writer connections over a WAL file-backed workspace, which measured
# best on every axis. See docs/CONSTRAINTS.md §3.6.
_lock = threading.Lock()
_registry: Registry | None = None


def _session() -> Registry:
    """The session registry, with any memory pressure relieved first.

    Relieving it *here* is what gives the spill its deferred timing, and doing
    it in one place is what stops a tool being added later that quietly skips
    it: every tool body reaches the registry through this function, exactly
    once, under the lock. The operation that crossed the budget has already
    returned; this is the next one, paying for it before doing its own work.
    """
    global _registry
    if _registry is None:
        _registry = Registry()
    else:
        _registry.relieve_memory()
    return _registry


def _reset() -> None:
    """Drop all session state. Used by tests, not exposed as a tool."""
    global _registry
    if _registry is not None:
        _registry.close()
    _registry = None


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def _iso_duration(value: _dt.timedelta) -> str:
    """A duration in ISO 8601, the only spelling this server recognises.

    MySQL hands a ``TIME`` column back as a duration rather than a time of day,
    and PostgreSQL does the same for ``INTERVAL``, so this is not an exotic case.
    Rendered rather than turned into a number because a bare number needs its
    unit said somewhere, and there is nowhere in a row to say it.
    """
    seconds = value.days * 86_400 + value.seconds
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    fraction = f".{value.microseconds:06d}".rstrip("0") if value.microseconds else ""
    return f"PT{hours}H{minutes}M{seconds}{fraction}S"


def _wire_datetime(value: _dt.datetime) -> str:
    """An instant as canonical UTC ISO 8601 text, per ``CONSTRAINTS.md`` §8.1.

    An aware instant is converted rather than reported in the offset it arrived
    in, so the same instant written two ways compares equal. A naive one is
    reported as it stands and gains no ``Z``: nothing in the database said which
    zone it was in, and stamping one on would be an invention.
    """
    if value.tzinfo is not None:
        return value.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return value.isoformat()


#: How a value that JSON cannot hold is spelled on the way out, by exact type.
#:
#: **A server-side database returns far more than the three storage classes
#: SQLite has**, and every one of these came back from a live container: a
#: ``NUMERIC`` as ``Decimal``, a ``DATE`` as ``date``, a MySQL ``TIME`` as
#: ``timedelta``, a ``BYTEA`` as ``bytes``, a Postgres ``UUID`` as ``UUID``.
#: Left alone they reach the serialiser, which renders each in whatever way it
#: happens to — a number arriving as the *string* ``"155000"`` was the defect
#: that asked for this table, because an agent then compares and sums text.
#:
#: Keyed on exact type rather than ``isinstance``, as :mod:`binding` is, because
#: ``datetime`` subclasses ``date`` and ``bool`` subclasses ``int``.
_ON_THE_WIRE: dict[type, Any] = {
    # As float, matching what `binding` already decided a Decimal becomes on the
    # way *in*: lossy past float64's 53-bit mantissa, and worth it because the
    # alternative is text, and text does not add up.
    Decimal: float,
    _dt.datetime: _wire_datetime,
    _dt.date: _dt.date.isoformat,
    _dt.time: _dt.time.isoformat,
    _dt.timedelta: _iso_duration,
    # Hex, prefixed as SQL writes a blob literal. Legible for the small binary
    # values that actually turn up — a hash, a checksum, a packed identifier —
    # and the prefix is what stops it reading as ordinary text.
    bytes: lambda value: f"0x{value.hex()}",
    bytearray: lambda value: f"0x{bytes(value).hex()}",
    # The third spelling of "binary", and the one the fallback below handles worst
    # (issue #57). psycopg2 returns a ``BYTEA`` as ``memoryview`` where psycopg 3
    # returns ``bytes``, so which of these three arrives is a property of the
    # *driver* rather than of the database — and this project now carries two
    # PostgreSQL drivers, so both answers turn up. Unspelled, a memoryview reaches
    # the caller as ``<memory at 0x1192c7640>``: not a lossy rendering of the value
    # but a process address, carrying none of the bytes and differing every run.
    memoryview: lambda value: f"0x{bytes(value).hex()}",
    UUID: str,
}


def _wire_value(value: Any) -> Any:
    """One value from a result row, as something JSON can carry.

    Numbers, strings, booleans, ``None`` and the containers a JSON column
    already returns pass through untouched. Everything else is spelled by
    :data:`_ON_THE_WIRE`, and a type not named there arrives as its text form —
    a PostGIS geometry, say. Stringifying the unknown case beats refusing the
    whole result over one column, and beats guessing at its structure.
    """
    spelling = _ON_THE_WIRE.get(type(value))
    if spelling is not None:
        return spelling(value)
    if value is None or isinstance(value, (bool, int, float, str, list, dict)):
        return value
    return str(value)


def _column_payload(column: Any) -> dict[str, Any]:
    return {
        "name": column.name,
        "type": column.declared_type,
        **(
            {"temporal": column.temporal_kind, "unit": "nanoseconds_since_epoch"}
            if column.temporal_kind
            else {}
        ),
        **(
            {
                "storage_classes": column.storage_classes,
                "numeric_values": column.numeric_values,
                "non_numeric_values": column.non_numeric_values,
                # The values, not only their count: the count says a filter is
                # needed, these say what it must exclude.
                "non_numeric_examples": list(column.non_numeric_examples),
            }
            if column.is_mixed
            else {}
        ),
        # Said positively so a caller can tell "this compares
        # chronologically" from "nothing is known about this column".
        **(
            {"temporal": column.temporal_standard, "normalized": "UTC"}
            if column.temporal_standard
            else {}
        ),
        **(
            {"dates_not_a_standard": list(column.unparsed_temporal_examples)}
            if column.is_unparsed_temporal
            else {}
        ),
    }


def _table_payload(info: TableInfo) -> dict[str, Any]:
    return {
        # The bare name, because it is what a statement against this datasource
        # uses. The payload already carries the nickname it belongs to.
        "table": info.name,
        "rows": info.row_count,
        "source": info.source,
        "columns": [_column_payload(column) for column in info.columns],
        "mixed_columns": info.mixed_columns,
        **(
            {"unparsed_temporal_columns": info.unparsed_temporal_columns}
            if info.unparsed_temporal_columns
            else {}
        ),
    }


def _mixed_column_warning(info: TableInfo) -> str:
    """Say which columns are mixed, and how to work around each one.

    The remedy differs by *why* the column is mixed, and getting that wrong is
    worse than saying nothing, because the instruction is followed. This warning
    used to prescribe ``typeof(col)='integer'`` for both kinds; on the kind a CSV
    always produces, every value is stored as text and that filter separates
    nothing. Four of six agents in live validation ran it, got the whole column
    back, and had to go and find the sentinel value themselves — so this now
    names the values instead.
    """
    lines = []
    for column in (c for c in info.columns if c.is_mixed):
        if column.mixed_kind == "storage":
            remedy = (
                "values are stored under different types here, so filter with "
                f"typeof({column.name})='integer' or CAST explicitly"
            )
        else:
            listed = ", ".join(repr(value) for value in column.non_numeric_examples)
            remedy = (
                f"{column.non_numeric_values} of its values do not read as "
                f"numbers ({listed}) while the rest do, and every one of them is "
                f"stored as text — so typeof() cannot tell them apart. Exclude "
                f"them by value, as in WHERE {column.name} NOT IN ({listed}), "
                f"and CAST the rest"
            )
        lines.append(f"{column.name}: {remedy}.")

    return (
        f"In {info.tag}, table {info.name}: aggregates over a mixed column "
        f"silently coerce text to 0 and keep it in the denominator, so avg() and "
        f"sum() will be wrong. " + " ".join(lines)
    )


def _unparsed_temporal_warning(info: TableInfo) -> str:
    """Say which columns hold dates that will not compare correctly.

    The counterpart to the mixed-column warning, for the other silent
    wrong-answer class — and the larger of the two. A date column is
    *uniformly* text, so no storage-class signal exists to trip and nothing
    else in the payload would ever mention it.

    It says which order the comparison actually is, because the failure is not
    that the column is unusable but that it lies convincingly: ``max()``
    returns a real value from the column, and on four of the spellings measured
    it is the earliest instant in the table (CONSTRAINTS §8.1).
    """
    lines = []
    for column in (c for c in info.columns if c.is_unparsed_temporal):
        listed = ", ".join(repr(v) for v in column.unparsed_temporal_examples)
        lines.append(f"{column.name} (for example {listed})")

    return (
        f"In {info.tag}, table {info.name}: {', '.join(lines)} "
        f"read as dates but are in no standard this server recognises, so they "
        f"are stored and compared as text — ORDER BY, min(), max() and range "
        f"filters on them follow alphabetical order, not chronological, and "
        f"will quietly return the wrong row. Only ISO 8601 "
        f"(2024-03-01, 2024-03-01T14:30:00Z) and Unix timestamps are "
        f"recognised; a spelling like 01/03/2025 cannot be, because nothing in "
        f"the file says whether it is March or January. Sort or compare with "
        f"an expression that reorders the parts, or have the file written in "
        f"ISO 8601."
    )


def _slot_payload(slot: Slot, registry: Registry) -> dict[str, Any]:
    return {
        "nickname": slot.nickname,
        "kind": slot.kind,
        "source": slot.source,
        "writable": slot.writable,
        # Bare names: this payload already says which nickname it describes, and
        # these are the names a statement against that nickname actually uses.
        "tables": list(registry.tables(slot.nickname)),
    }


def _index_payload(index: IndexInfo) -> dict[str, Any]:
    """One index, named so that ``drop`` can be handed the name verbatim."""
    return {
        "index": index.name,
        "table": index.table,
        "columns": list(index.columns),
        "unique": index.unique,
    }


def _table_warnings(info: TableInfo) -> list[str]:
    """Everything worth saying about one table, in one place.

    The two column signals are silent-wrong-answer classes: a value comes back,
    it looks like an answer, and it is not one. The reader's notes are the same
    problem one step earlier — a choice the reader had to make about the file
    itself, which the loaded table then looks perfectly ordinary despite. All of
    them are gathered here so a new class reaches every payload at once rather
    than the one whose call site was remembered.
    """
    warnings = []
    if info.mixed_columns:
        warnings.append(_mixed_column_warning(info))
    if info.unparsed_temporal_columns:
        warnings.append(_unparsed_temporal_warning(info))
    warnings.extend(info.notes)
    return warnings


def _described(registry: Registry, nickname: str, tables: tuple[str, ...]) -> tuple:
    """Describe several tables, and collect whatever is worth warning about."""
    described = [registry.describe(nickname, table) for table in tables]
    warnings = [w for info in described for w in _table_warnings(info)]
    return described, warnings


def _attachment_payload(attachment: Attachment, registry: Registry) -> dict[str, Any]:
    slot = attachment.slot
    payload: dict[str, Any] = {"ok": True, **_slot_payload(slot, registry)}

    # A slot we built ourselves was described as it was read, so its shape and
    # its mixed-column signal are already known — returning them here saves the
    # caller an immediate directory() round trip.
    if slot.kind == "file":
        described, warnings = _described(registry, slot.nickname, slot.tables)
        payload["loaded"] = [_table_payload(info) for info in described]
        if warnings:
            payload["warnings"] = warnings

    # Always present, never merely omitted: a caller that has to tell an absent
    # key from an empty one will eventually get it wrong.
    collision = attachment.collided_with
    payload["collided_with"] = (
        None
        if collision is None
        else {"nickname": collision.nickname, "source": collision.source}
    )
    evicted = attachment.evicted
    payload["evicted"] = (
        None
        if evicted is None
        else {
            "nickname": evicted.nickname,
            "kind": evicted.kind,
            "source": evicted.source,
            "tables": list(evicted.tables),
            "reason": evicted.reason,
        }
    )
    return payload


def _failed(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(exc)}


def _answers(tool: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Make a verb answer whatever happens, since that is what was promised.

    The instructions this server ships tell the caller that a refusal is an
    ordinary answer and to branch on ``ok``. An agent reads that once. So the
    promise has to hold for the failures nobody enumerated too — a driver that
    raises something new, a filesystem that goes away mid-call, a bug in here —
    or the one shape an agent was told would never arrive is the one it gets.

    Each verb still catches the failures it *expects* and refuses them in its
    own words; those are the answers a caller can act on. This is the outer net
    under all eight, and it exists as one decorator rather than as eight
    ``except`` clauses because the same promise stated eight times is a promise
    that will soon be stated seven ways: ``query`` carried this catch-all alone
    for a release while its seven siblings let the same failures out as protocol
    errors (issue #92).

    The exception type is named in the message. An unexpected failure that reads
    like an ordinary refusal is worse than one that escapes — the caller cannot
    fix it, will try, and nobody learns it happened. ``BaseException`` is
    deliberately not caught: an interrupt is not this server's to answer.
    """

    @functools.wraps(tool)
    def answered(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return tool(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the envelope is the point
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return answered


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
@_answers
def attach(
    database: str,
    nickname: str | None = None,
    writable: bool = False,
    delimiter: str | None = None,
) -> dict[str, Any]:
    """Attach a datasource as a database, and return the nickname it got.

    A flat file becomes a new database holding one table named after the file; a
    SQLite or DuckDB database arrives with the tables it already has. A file that
    holds several tables — a workbook's sheets, a ``.numbers`` document's tables —
    becomes a database holding all of them, under the names the file gives them.

    For a file, the answer already carries what it loaded — columns, types, row
    count and any warning — so calling ``directory`` straight afterwards returns the
    same thing again. Go and ask the question instead.

    Args:
        database: A tabular file (.csv, .tsv, .txt, .json, .jsonl, .ndjson,
            .xml, .yaml, .yml, .parquet, .feather, .orc, .xlsx, .xlsm,
            .xls, .ods, .numbers), a SQLite or DuckDB database file, or a
            database URL. A database file is told apart from a flat file by its
            header, not by its suffix.
        nickname: The name this datasource answers to — pass it to every later
            call. Derived from the filename when omitted. If it collides with a
            slot already open, a numeric suffix is added — so always use the
            nickname that comes back rather than the one you asked for.
        writable: Allow writes to a datasource that came from outside. Ignored
            for a flat file, whose database is built here and always writable.
            The grant belongs to this attach, and the same source attached twice
            is refused — so granting write to something already open means
            ``detach`` first, then attach it again with this set.
        delimiter: The character separating fields. **Required** for .csv,
            .tsv and .txt, which are one format here — character-separated
            text — and refused for every other, which carries its own
            structure. There is no default and the extension supplies nothing:
            files called .csv are shipped separated by ';' or '|' or a tab as
            readily as by ',', and reading one at the wrong character changes
            what the data is while the answer looks perfectly ordinary. Ask the
            user or look at the file; do not guess on their behalf.

    Refuses a datasource that is already attached, naming the slot holding it.
    ``collided_with`` says which live slot forced a suffix, and ``evicted``
    describes the datasource dropped to make room, if any.
    """
    with _lock:
        registry = _session()
        try:
            attachment = registry.attach(
                database, nickname, writable=writable, delimiter=delimiter
            )
        except (SlotError, PathNotAllowed) as exc:
            return _failed(exc)
        return _attachment_payload(attachment, registry)


@mcp.tool
@_answers
def detach(nickname: str) -> dict[str, Any]:
    """Close a datasource and free its slot.

    Everything the slot held is gone: tables added to it, and any rows not
    written out with ``save`` first.

    The answer names what was closed — ``nickname``, ``source`` and the
    ``tables`` it held — and the session's ``slots_used`` and
    ``slots_available`` afterwards. ``slots_available`` is the slot **limit**
    and not the number free, the same as in ``directory``; the free slots are the
    subtraction.

    Args:
        nickname: The datasource to close.
    """
    with _lock:
        registry = _session()
        try:
            slot = registry.detach(nickname)
        except SlotError as exc:
            return _failed(exc)
        return {
            "ok": True,
            "nickname": slot.nickname,
            "source": slot.source,
            "tables": list(slot.tables),
            "slots_used": len(registry.slots()),
            "slots_available": registry.capacity(),
        }


@mcp.tool
@_answers
def directory(nickname: str | None = None, table: str | None = None) -> dict[str, Any]:
    """Describe what is attached, at whichever altitude you need.

    Three forms, and the arguments choose between them:

    * neither — every attached datasource, and the path posture in force;
    * ``nickname`` — that datasource, the tables inside it, and its indexes;
    * ``nickname`` and ``table`` — that table's columns, types, row count and
      indexes.

    The indexes are what to consult before asking ``create`` for one: they are
    reported whoever made them, so an attached database arrives describing the
    indexes it already had.

    ``slots_available`` is the slot **limit**, not the number of free slots — it
    is constant for the session, so a full one reports ``slots_used`` and
    ``slots_available`` equal rather than zero. Free slots are the subtraction.

    A ``table`` with no ``nickname`` is refused rather than dropped: it names
    the third form and reaches the first, and answering the session listing
    under ``ok: true`` would look like the question was understood.

    Args:
        nickname: Restrict to one datasource.
        table: With a nickname, describe this one table in full.
    """
    with _lock:
        registry = _session()
        try:
            if nickname is None and table is not None:
                raise SlotNotAvailable(
                    f"directory was asked for table {table!r} with no nickname, "
                    f"and a table is addressed inside one datasource. Name the "
                    f"datasource too, or call directory() with neither argument "
                    f"for the session."
                )
            if nickname is not None and table is not None:
                return _table_detail(registry, nickname, table)
            if nickname is not None:
                return _slot_detail(registry, nickname)
            return _session_detail(registry)
        except SlotError as exc:
            return _failed(exc)


@mcp.tool
@_answers
def stats(
    nickname: str, table: str, columns: list[str] | None = None
) -> dict[str, Any]:
    """Profile a table's columns — how many values are missing, and their range.

    This is the verb that answers *"can I trust this data?"*. ``directory`` is a
    directory — what is attached, what tables it holds, what shape they are —
    and it says nothing about the values inside them. In particular **nothing
    else here reports a missing value**: a column can be half empty and appear
    in ``directory`` looking perfectly ordinary.

    Every column reports ``nulls`` and ``non_nulls``. A numeric column also
    reports ``min``, ``max`` and ``avg``, and a date column normalised to ISO
    8601 reports ``min`` and ``max``. ``median`` and ``stddev`` appear only where
    the database computes them itself — nothing here approximates a statistic the
    engine does not have, so their absence is a fact about the datasource rather
    than about the column.

    Two kinds of column are deliberately reported with their null count and
    nothing else, and each says why in ``withheld``: a **mixed** column, where an
    average silently coerces text to 0 and keeps it in the denominator, and a
    column of **dates in no recognised standard**, where ``min`` and ``max``
    compare alphabetically and return the wrong instant. Both would answer with a
    real number that is not the number asked for.

    Args:
        nickname: The datasource holding the table.
        table: The table to profile, by the name ``directory`` lists.
        columns: Restrict the profile to these columns. All of them by default;
            a name matching nothing is refused, naming the columns that exist.

    Refuses an unattached nickname, a missing table and a missing column, each
    naming what was available instead.
    """
    with _lock:
        registry = _session()
        try:
            described, profiled = registry.profile(nickname, table, columns)
        except SlotError as exc:
            return _failed(exc)

        payload: dict[str, Any] = {
            "ok": True,
            "nickname": nickname,
            "table": described.name,
            "rows": described.row_count,
            "columns": [_profile_payload(column) for column in profiled],
        }
        if warnings := _table_warnings(described):
            payload["warnings"] = warnings
        return payload


def _profile_payload(profile: ColumnProfile) -> dict[str, Any]:
    """One column's statistics, carrying only what was actually computed.

    A statistic that was not asked for is **absent** rather than ``null``. The
    two would be indistinguishable to a caller otherwise, and they mean opposite
    things here: a missing ``avg`` says this server did not compute one, while
    ``avg: null`` would say the engine computed one over no rows.
    """
    payload: dict[str, Any] = {
        "name": profile.name,
        "type": profile.declared_type,
        "nulls": profile.nulls,
        "non_nulls": profile.non_nulls,
    }
    optional = {
        "min": profile.minimum,
        "max": profile.maximum,
        "avg": profile.average,
        **profile.optional,
    }
    payload.update(
        {
            key: _wire_value(value)
            for key, value in optional.items()
            if value is not None
        }
    )
    if profile.withheld:
        payload["withheld"] = profile.withheld
    return payload


def _table_detail(registry: Registry, nickname: str, table: str) -> dict[str, Any]:
    described = registry.describe(nickname, table)
    payload = {"ok": True, **_table_payload(described)}
    payload["indexes"] = [
        _index_payload(index) for index in registry.indexes(nickname, table)
    ]
    if warnings := _table_warnings(described):
        payload["warnings"] = warnings
    return payload


def _slot_detail(registry: Registry, nickname: str) -> dict[str, Any]:
    slot = registry.slot(nickname)
    tables = registry.tables(nickname)
    described, warnings = _described(registry, nickname, tables)
    payload: dict[str, Any] = {
        "ok": True,
        **_slot_payload(slot, registry),
        "contents": [
            {"table": info.name, "rows": info.row_count} for info in described
        ],
        "indexes": [_index_payload(index) for index in registry.indexes(nickname)],
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


def _session_detail(registry: Registry) -> dict[str, Any]:
    return {
        "ok": True,
        "datasources": [_slot_payload(slot, registry) for slot in registry.slots()],
        "slots_used": len(registry.slots()),
        "slots_available": registry.capacity(),
        # The posture the LLM is working under, where it will read it.
        "roots": [str(path) for path in allowed_paths()],
        "path_limited": config.active().path_limited,
    }


@mcp.tool
@_answers
def query(
    nickname: str,
    sql: str,
    path: str | None = None,
    force: bool = False,
    delimiter: str | None = None,
) -> dict[str, Any]:
    """Run SQL against a datasource, returning rows or writing them to a file.

    ``nickname`` chooses the datasource; the SQL then names tables inside it
    directly — ``SELECT * FROM sales``, not ``FROM shop.sales``. One statement
    reaches one datasource. To query two of them together, ``create`` copies one
    into the other first, and the join is then ordinary SQL over two tables in
    the same database.

    **A query reads.** INSERT, UPDATE, CREATE TABLE, CREATE VIEW and PRAGMA are
    refused here whatever the datasource permits — the connection this runs on
    is read-only from the moment it opens. To change a datasource use ``create``,
    ``update`` and ``drop``; those are what ``writable=true`` governs.

    **The whole result comes back.** There is no row cap, because a row cap
    measures the wrong thing — a hundred rows of a two-hundred-column table is
    the flood it would be meant to prevent. Ask for what you want instead: SQL
    ``LIMIT`` for fewer rows, named columns rather than ``SELECT *`` for fewer
    of those, and ``path`` when the whole result is genuinely wanted but does
    not belong in an answer.

    Args:
        nickname: Which datasource executes the statement.
        sql: The SQL statement.
        path: Write the full result to this file instead of returning rows. The
            suffix chooses the format (.csv, .tsv, .txt, .json, .jsonl,
            .ndjson, .xml, .yaml, .yml, .md, .parquet, .feather, .orc, .xlsx,
            .ods); one this server cannot write is refused by name rather than
            written as something else. Passing this changes what comes back:
            the response carries rows_written and the column names instead of
            the rows themselves. For a very large result the suffix is worth
            choosing rather than defaulting to. .csv and .jsonl write row by row
            and never hold the result; .parquet does hold it. Which format is
            smallest depends on the data far more than on the format: over seven
            shapes driven, .parquet was smallest on three — by 100x or more
            where a column repeats few distinct values — while on high-entropy
            text it was eighth of fifteen and .orc was smallest. Where .orc wins
            it wins by about 1.2x; where .parquet wins it can win by 190x. The
            two write within 8% of each other. .feather writes faster and was
            larger than .parquet on every text shape driven but smaller on both
            float shapes. .yaml is slow (4.3x .csv on a million-row result)
            though it holds nothing, and .md holds the whole table because a
            Markdown column is only as wide as its widest value. A spreadsheet
            (.xlsx, .ods) refuses more than 65,535 rows outright, and of those
            two .ods is 5.8x slower than .xlsx at 20,000 rows of eleven
            ordinary-width columns. The gap widens with rows: a separate drive
            on its own corpus puts the pair at 6.45x at 20,000 and 12.7x at
            50,000, and two corpora at one nominal shape is why that drive's
            20,000-row figure is not the 5.8x above — cell widths fix a corpus,
            row and column counts do not. On a forty-column result
            (50,000 x 40) .ods ran for over half an hour without producing a
            file — ask for it when OpenDocument is what was wanted, not by
            default.
        force: Replace the file if it is already there. Set this only after the
            user has said to — the path is theirs, so the refusal you get
            without it is a question to put to them, not a retry to make. A file
            an attached datasource is sitting on is refused either way.
        delimiter: The character to separate fields with. **Required** when
            ``path`` names .csv, .tsv or .txt, for the same reason it is
            required on the way in: the suffix decides the format and never the
            separator, so writing without saying would put commas in a file
            called .tsv on request. Ignored for a format that has no separator,
            so one value can be carried across a mix of destinations.
    """
    with _lock:
        registry = _session()

        if path is None:
            try:
                columns, rows = registry.query(nickname, sql)
            except SlotError as exc:
                return _failed(exc)
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            return {
                "ok": True,
                "columns": columns,
                # Spelled for JSON here and not in `loader`, because a result
                # bound for a file goes to `export_rows` instead, and a file is
                # better served by the native value — a Parquet timestamp column
                # wants a timestamp, not the text an answer needs.
                "rows": [[_wire_value(value) for value in row] for row in rows],
                "row_count": len(rows),
            }

        # A result bound for a file is never assembled. `query_stream` holds the
        # cursor open for as long as the writer is pulling from it, so the rows
        # go from the database to the disk without a list of them existing —
        # which for the nine suffixes that write row by row means the peak does
        # not move with the size of the result.
        try:
            with registry.query_stream(nickname, sql) as (columns, rows):
                result = export_rows(
                    columns,
                    rows,
                    path,
                    force=force,
                    claimed=registry.claimed_paths(),
                    delimiter=delimiter,
                )
        except (ExportError, PathNotAllowed) as exc:
            return _failed(exc)
        except SlotError as exc:
            return _failed(exc)
        except OSError as exc:
            return {"ok": False, "error": f"Could not write {path}: {exc}"}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        return {
            "ok": True,
            "path": result.path,
            "rows_written": result.row_count,
            "columns": result.columns,
        }


@mcp.tool
@_answers
def create(
    nickname: str,
    type: str,
    table: str | None = None,
    source: str | None = None,
    columns: list[str] | None = None,
    delimiter: str | None = None,
) -> dict[str, Any]:
    """Create a table or an index inside a datasource that is already open.

    ``type="table"`` reads a file in beside the tables already there. Use this —
    rather than attaching a second slot — when a new file is meant to be looked
    up against one already loaded: ``save`` writes one database rather than a
    join, so landing both sides in the same slot is what makes the lookup
    outlive the session. The answer describes the table it read, so ``directory``
    straight afterwards tells you nothing new.

    It makes **one** table, so a source holding more than one is refused, naming
    them — a two-sheet workbook that ``attach`` opens happily as a database of
    two tables cannot be read in here, and ``table`` does not select a sheet.
    To land one sheet of a workbook beside tables you already have: ``attach``
    the workbook as its own datasource, ``query(path=…)`` the sheet you want out
    to a flat file, then ``create`` that file in.

    ``type="index"`` indexes columns of a table already there. Ask for one when
    you are about to join or filter on those columns and the table is large;
    nothing here guesses that for you, because which query is coming is yours to
    know. The index is named for you and the name comes back — that is the name
    ``drop`` wants. ``directory(nickname, table)`` lists the indexes that already
    exist, which is the cheaper way to find out than asking twice. Five engines
    refuse ``type="index"`` outright — ClickHouse, Trino, CrateDB, Databend and
    Exasol — each for its own reason: indexes that cannot be reflected, no
    storage to index, every column indexed already, a statement that compiles to
    nothing, or an engine that maintains its own. The refusal says which, and
    arrives before any work is done.

    Whether a join actually lines up is not reported here. It is an anti-join
    over two tables in one database — ordinary SQL you can write, and better
    said in the user's own words than in a payload field.

    Args:
        nickname: The datasource to create in. Must be writable.
        type: ``"table"`` or ``"index"``.
        table: For a table, its name — derived from the filename when omitted.
            For an index, the existing table to index; required.
        source: For a table, the file to read in — one holding a single table.
            Required for ``type="table"``.
        columns: For an index, the columns to index, in order. Required for
            ``type="index"``.
        delimiter: For a table read from .csv, .tsv or .txt, the character
            separating fields. Means the same here as on ``attach``, including
            that it is required and that nothing sniffs for it.
    """
    with _lock:
        registry = _session()
        try:
            if type == "table":
                if source is None:
                    raise SlotError(
                        "create(type='table') reads a file in, so it needs "
                        "source=. To index an existing table, use type='index'."
                    )
                return _table_created(
                    nickname,
                    registry.create_table(
                        nickname, source=source, table=table, delimiter=delimiter
                    ),
                )
            if type == "index":
                if table is None or not columns:
                    raise SlotError(
                        "create(type='index') needs table= and columns= — which "
                        "table, and which of its columns to index."
                    )
                made = registry.create_index(nickname, table=table, columns=columns)
                payload = {"ok": True, "nickname": nickname, **_index_payload(made)}
                if made.notes:
                    # An index that covers less than it was asked to is a fact
                    # about the answer, and it is said the same way every other
                    # assumption here is said.
                    payload["warnings"] = list(made.notes)
                return payload
        except (SlotError, PathNotAllowed) as exc:
            return _failed(exc)
        return _failed(
            SlotError(f"create has no type {type!r}. It is 'table' or 'index'.")
        )


def _table_created(nickname: str, info: TableInfo) -> dict[str, Any]:
    # `nickname` is carried here so both of `create`'s answers have the same
    # keys at the front. The index branch always named the slot; this one did
    # not, so one verb returned two shapes differing in a key.
    payload: dict[str, Any] = {"ok": True, "nickname": nickname, **_table_payload(info)}
    if warnings := _table_warnings(info):
        payload["warnings"] = warnings
    return payload


@mcp.tool
@_answers
def drop(nickname: str, type: str, name: str) -> dict[str, Any]:
    """Remove a table or an index from a datasource.

    Dropping a table takes its indexes with it, and they are not listed in the
    answer — so an index name that came back from ``create`` stops naming
    anything the moment its table goes, and ``drop`` on it afterwards refuses as
    a missing index rather than reporting one already gone.

    Args:
        nickname: The datasource holding it. Must be writable.
        type: ``"table"`` or ``"index"``.
        name: The table name, unqualified — or the index name, as ``create``
            returned it and as ``directory`` lists it.
    """
    with _lock:
        registry = _session()
        try:
            if type == "table":
                registry.drop_table(nickname, name)
                return {
                    "ok": True,
                    "nickname": nickname,
                    "dropped": name,
                    "tables": list(registry.tables(nickname)),
                }
            if type == "index":
                gone = registry.drop_index(nickname, name)
                return {
                    "ok": True,
                    "nickname": nickname,
                    "dropped": gone.name,
                    "table": gone.table,
                }
        except SlotError as exc:
            return _failed(exc)
        return _failed(
            SlotError(f"drop has no type {type!r}. It is 'table' or 'index'.")
        )


@mcp.tool
@_answers
def update(nickname: str, type: str, name: str, to: str) -> dict[str, Any]:
    """Rename something inside a datasource, keeping what it holds.

    The third of create/update/drop, and the first verb here that changes a
    table without rebuilding it: the rows, the types and the indexes all stay
    where they are.

    What asked for it is a file that names its own tables. A workbook's sheets
    arrive under the names the *spreadsheet* chose — ``Sheet1``, or a label with
    a year in it — and those are frequently not the names you want to write SQL
    against for the rest of the session. Renaming beats re-reading the file
    under a different name, which would cost the load again and lose any index
    already built.

    Renaming onto a name that is taken is refused rather than allowed to replace
    it, and the refusal says how many rows the other table holds. So is renaming
    to the same name in another case: an engine that folds case holds one name
    for both, so that is not a free name.

    The answer carries ``renamed`` — the name it had, as the database stored it,
    which is not always the spelling you passed — alongside ``table``, the name
    it has now.

    One engine has no rename statement this server can issue, so
    ``type="table"`` is refused outright on Firebird.

    Args:
        nickname: The datasource holding it. Must be writable.
        type: ``"table"``. Indexes are named by ``create`` and dropped by name,
            so there is nothing to rename there.
        name: What it is called now.
        to: What to call it. Same rule as any table name — a letter or
            underscore, then letters, digits or underscores.
    """
    with _lock:
        registry = _session()
        try:
            if type == "table":
                # Asked before the rename, and not echoed from `name`: the
                # database may store the old name in a case the caller did not
                # use, and `renamed` is the field that has to match the listing
                # this same payload replaces.
                was = registry.stored_name(nickname, name)
                info = registry.rename_table(nickname, name, to)
                return {
                    "ok": True,
                    "nickname": nickname,
                    "renamed": was,
                    **_table_payload(info),
                    "tables": list(registry.tables(nickname)),
                }
        except SlotError as exc:
            return _failed(exc)
        return _failed(
            SlotError(
                f"update has no type {type!r}. It is 'table' — an index is named "
                f"by create and removed by drop, so there is nothing to rename."
            )
        )


@mcp.tool
@_answers
def save(nickname: str, path: str, force: bool = False) -> dict[str, Any]:
    """Write a datasource out to a SQLite file the user keeps.

    Everything attached is otherwise ephemeral — it dies on detach and when this
    server stops. This is how a session's work survives, including tables added
    to a slot.

    The datasource stays open and unchanged; this writes a copy. Attaching that
    copy later is an ordinary attach, so it comes back read-only unless write is
    granted again.

    Only a database this server holds is one it can write out, so this is
    refused on every backend but SQLite — a database reached over a URL, and a
    DuckDB file, included. Keeping rows from one of those takes three calls:
    ``query(nickname, sql, path=…)`` writes them to a file, ``attach`` on that
    file makes a datasource of your own (file-derived, so writable), and
    ``save`` writes that one out. ``create`` is for adding further tables to
    that new slot, not for making it. Offer that route rather than this verb
    when the datasource is not SQLite.

    Args:
        nickname: The datasource to write out.
        path: Destination file, within the allowed paths.
        force: Replace the file if it is already there. Set this only after the
            user has said to — the name is theirs, so the refusal you get
            without it is a question to put to them, not a retry to make. A file
            an attached datasource is sitting on is refused either way.
    """
    with _lock:
        registry = _session()
        try:
            written = registry.save(nickname, path, force=force)
            tables = registry.tables(nickname)
        except (SlotError, PathNotAllowed) as exc:
            return _failed(exc)
        return {
            "ok": True,
            "nickname": nickname,
            "path": str(written),
            "tables": list(tables),
        }


def main() -> None:
    """Read the configuration, then serve.

    Reading it *here* is what makes a mistyped setting refuse to start. The
    configuration decides what this server may reach on the filesystem and the
    network, and :mod:`config` refuses an unknown key rather than ignoring it
    for exactly that reason — but it loads on first use, so without this line
    the refusal surfaced inside whichever tool call first happened to read it,
    as a protocol error, with the server already serving under a boundary the
    user did not set. Failing here puts the message in the client's server log,
    which is where the person who wrote the file will look.
    """
    config.active()
    mcp.run()


if __name__ == "__main__":
    main()
