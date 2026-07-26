"""Datasource slots: the one idea the tool surface is built on.

**A slot is always a database, addressed by a nickname.** Three things can fill
one, and none of them is a special case:

* a **flat file** becomes a brand-new in-memory database holding one table named
  after the file — so a later ``add_table`` can put a second table beside it
  under the same nickname;
* a **SQLite file** is attached read-only, arriving with the tables it already
  has;
* a **URL** becomes an engine SQLAlchemy resolved from the URL itself.

Because every slot is a database, nothing above this module needs to know which
kind it is holding. Each slot is reached on its own engines, and a statement
addresses the tables inside one slot by their own names.

**"Not a special case" is meant literally, and is the property most worth
defending here.** All three live in the same :class:`~.loader.Workspace`, under
the same kind of tag, and every verb has exactly one implementation. There is no
second code path for a datasource reached over a URL — which is how one of them
previously came to support three of the seven verbs while the other supported
all seven, with nothing in the type system to notice.

**A statement reaches one slot.** Slots do not share a connection, so there is no
join across them; putting two datasources together means copying one into the
other with ``add_table``, which says so in the call rather than depending on how
the databases happen to be wired underneath.

**Ten slots, and the number is now chosen rather than forced.** It used to be
SQLite's own ceiling — every slot was an attached database and the eleventh
``ATTACH`` raised ``too many attached databases - max 10``. Per-slot engines
remove that limit entirely, so the number survives on its own merits: each slot
costs live connections and, while it is in memory, memory. Slots are evicted
oldest-first, and the eviction is *reported*, because a caller told only ``no
such table`` a minute later has to re-plan blind.

A slot that cannot be reached is explained by **enriching an error that already
happened** rather than by inspecting SQL in advance. A substring check over a
statement would eventually refuse a legitimate query whose column happened to be
named like a slot; the same check applied only after the engine has already
refused cannot cause a false positive.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sqlalchemy.engine import make_url

from . import config
from .dialects import UnsupportedOperation
from .loader import (
    READERS,
    LoadError,
    TableInfo,
    Workspace,
    _sanitize,
    read_frame,
)
from .paths import PathNotAllowed, resolve_read_path, resolve_write_path

__all__ = [
    "AddedTable",
    "AttachRefused",
    "Attachment",
    "Collision",
    "Eviction",
    "JoinReport",
    "NotWritable",
    "Registry",
    "Slot",
    "SlotNotAvailable",
    "url_scheme",
]

#: The first sixteen bytes of every SQLite database, header magic included.
SQLITE_MAGIC = b"SQLite format 3\x00"

#: A scheme must be at least two characters, so a Windows drive letter is never
#: mistaken for one, and the ``://`` must be present so a bare path never is.
_URL = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]+)://")

#: A nickname names the slot's own database and is carried in the URI that
#: opens it, so it must be a plain identifier: punctuation would be read as URI
#: syntax and open a database nobody chose.
_NICKNAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: How many evictions to remember, so a later reference can still be explained.
_EVICTION_MEMORY = 64


class SlotError(RuntimeError):
    """Something wrong with a slot or the request to make one."""


class AttachRefused(SlotError):
    """A datasource this server will not open, and why."""


class SlotNotAvailable(SlotError):
    """A nickname that names no live slot."""


class NotWritable(SlotError):
    """A slot this server will not change, and how to make it changeable."""


def url_scheme(database: str) -> str | None:
    """The scheme of a datasource URL, or ``None`` for a filesystem path."""
    match = _URL.match(database)
    return match.group("scheme") if match else None


@dataclass(frozen=True)
class Slot:
    """One attached database, and how to reach it."""

    nickname: str
    #: How this slot was opened: ``"file"`` (a flat file, now its own in-memory
    #: database), ``"database"`` (a local database file), or ``"engine"`` (a URL
    #: SQLAlchemy resolved). **Descriptive only** — every kind is one tag in the
    #: workspace, reached the same way, and nothing branches on this to decide
    #: how to do its job.
    kind: str
    #: Where it came from, with any password removed.
    source: str
    tables: tuple[str, ...]
    #: Whether this database may be written to. Only databases the server
    #: created are writable by default; anything attached from outside arrives
    #: read-only unless the caller granted write at attach time.
    writable: bool = False
    #: Where this database was moved to when it outgrew the memory budget. The
    #: file is ours and is deleted when the slot goes.
    spill_path: Path | None = None


@dataclass(frozen=True)
class Eviction:
    """A slot that was dropped, described well enough to be rebuilt."""

    nickname: str
    kind: str
    source: str
    #: Every table the slot held — a slot can be composed of several files, and
    #: re-reading the original source alone would restore only one of them.
    tables: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Collision:
    """The live slot a nickname ran into, and where its data came from.

    Reported so a caller told ``sales_2`` can see *why* — and can offer the user
    a name that will mean more later, which the source is what makes possible.
    """

    nickname: str
    source: str


#: How many unmatched key values a join report carries. Enough to name them in
#: a sentence; the counts beside them say how many more there are.
_UNMATCHED_SAMPLE = 10


@dataclass(frozen=True)
class JoinReport:
    """Whether two tables in one database actually line up on a shared key.

    Stated as facts rather than as prose: which values on each side have no
    partner on the other, and how many. Turning that into *"Acme, Globex and
    Initech have no match in suppliers.csv"* is the skill's job — it knows what
    the user called these things, and this module does not.
    """

    key: str
    #: Bare table names, as a statement against this slot would write them.
    existing_table: str
    added_table: str
    matched_keys: int
    #: Key values present in the existing table with no partner in the new one.
    missing_from_added: tuple[Any, ...]
    missing_from_added_total: int
    #: And the other direction, which is just as often the interesting one.
    missing_from_existing: tuple[Any, ...]
    missing_from_existing_total: int

    @property
    def complete(self) -> bool:
        return self.missing_from_added_total == self.missing_from_existing_total == 0


@dataclass(frozen=True)
class AddedTable:
    """A table that landed in an existing slot, and how it lines up."""

    info: TableInfo
    join: JoinReport | None = None


@dataclass(frozen=True)
class Attachment:
    """The result of attaching: the new slot, and whatever it displaced."""

    slot: Slot
    evicted: Eviction | None = None
    #: What the derived or requested nickname collided with, if anything.
    collided_with: Collision | None = None


class Registry:
    """The live slots, oldest first, over one host connection."""

    def __init__(self) -> None:
        self._workspace = Workspace.in_memory()
        self._slots: dict[str, Slot] = {}
        self._evicted: deque[Eviction] = deque(maxlen=_EVICTION_MEMORY)
        #: Where spilled databases live. Made on first need, and only then.
        self._temp_dir: Path | None = None

    # -- introspection ------------------------------------------------------

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def slots(self) -> list[Slot]:
        """Live slots in attachment order — which is also eviction order."""
        return list(self._slots.values())

    def capacity(self) -> int:
        return config.active().slots

    def slot(self, nickname: str) -> Slot:
        """The live slot for a nickname, or an explanation of its absence."""
        found = self._slots.get(nickname)
        if found is not None:
            return found

        for eviction in self._evicted:
            if eviction.nickname == nickname:
                raise SlotNotAvailable(
                    f"Slot {nickname!r} was evicted ({eviction.reason}). It held "
                    f"{', '.join(eviction.tables) or 'no tables'} from "
                    f"{eviction.source}. Attach it again to use it."
                )

        known = ", ".join(self._slots) or "none"
        raise SlotNotAvailable(
            f"No datasource is attached as {nickname!r}. Attached: {known}."
        )

    # -- attaching ----------------------------------------------------------

    def attach(
        self, database: str, nickname: str | None = None, *, writable: bool = False
    ) -> Attachment:
        """Open a datasource as a slot, evicting the oldest if the shelf is full.

        The nickname is derived from the filename when none is given, and is
        disambiguated with a numeric suffix when it runs into a live slot. The
        name actually used comes back in the :class:`Attachment`, because it may
        not be the one that was asked for and a caller who assumes otherwise
        addresses the wrong database.
        """
        scheme = url_scheme(database)
        if scheme is not None and not scheme.startswith("sqlite"):
            return self._attach_url(database, nickname, writable=writable)

        path = self._resolve(database, scheme)
        self._refuse_duplicate(str(path))
        chosen, collision = self._choose_nickname(nickname, path.stem)

        if path.suffix.lower() in READERS:
            # Read before making room. A file that cannot be parsed must not
            # cost a live datasource its place — every refusal above and here
            # happens while the shelf is still untouched.
            try:
                frame = read_frame(path)
            except LoadError as exc:
                raise AttachRefused(str(exc)) from exc
            evicted = self._make_room()
            slot = self._attach_frame(frame, path, chosen)
        elif self._is_sqlite(path):
            evicted = self._make_room()
            # An outside database is read-only unless the caller granted write.
            slot = self._attach_database(path, chosen, writable=writable)
        else:
            supported = ", ".join(sorted(READERS))
            raise AttachRefused(
                f"{path} is neither a SQLite database nor a supported file "
                f"({supported}). Its first bytes are not the SQLite header."
            )

        self._slots[chosen] = slot
        return Attachment(slot=slot, evicted=evicted, collided_with=collision)

    # -- naming a slot ------------------------------------------------------

    def _refuse_duplicate(self, source: str) -> None:
        """Refuse a datasource that is already attached, wherever it landed.

        A second copy of identical data burns one of ten slots for nothing, and
        the answer the caller needs is not a new slot but the name of the one
        already holding it. The check is on the *source*, so asking for a
        different nickname does not get around it.
        """
        for slot in self._slots.values():
            if slot.source == source:
                raise AttachRefused(
                    f"{source} is already attached as {slot.nickname!r}, holding "
                    f"{', '.join(slot.tables) or 'no tables'}. Query it there, or "
                    f"detach it first if you want to re-read the file."
                )

    def _choose_nickname(
        self, requested: str | None, stem: str
    ) -> tuple[str, Collision | None]:
        """Settle on a usable nickname, and report what it displaced.

        A requested nickname is validated rather than rewritten — handing back a
        silently corrected handle is the same class of defect as a config that
        keeps its default after a typo. A *derived* one is sanitised, because
        ``2024 Sales Report.csv`` has no legal spelling the caller chose.

        Either way a collision is resolved with a numeric suffix rather than by
        refusing: ``~/q1/sales.csv`` and ``~/q2/sales.csv`` are two real
        datasources and both deserve a slot.
        """
        if requested is None:
            base = _sanitize(stem, "db")
        else:
            self._validate_nickname(requested)
            base = requested

        if not self._nickname_taken(base):
            return base, None

        collided = self._slots.get(base)
        index = 2
        while self._nickname_taken(f"{base}_{index}"):
            index += 1
        return (
            f"{base}_{index}",
            None if collided is None else Collision(collided.nickname, collided.source),
        )

    def _nickname_taken(self, nickname: str) -> bool:
        return nickname in self._slots

    def _resolve(self, database: str, scheme: str | None) -> Path:
        """A datasource path, containment-checked.

        A ``sqlite:`` URL is another way of spelling a local file, so it is
        resolved and contained exactly as a bare path is — naming it as a URL
        must not route around the trust boundary.
        """
        raw = database
        if scheme is not None:
            raw = make_url(database).database or ""
            if not raw:
                raise AttachRefused(f"{database} names no database file.")
        try:
            return resolve_read_path(raw)
        except PathNotAllowed as exc:
            raise AttachRefused(str(exc)) from exc

    @staticmethod
    def _is_sqlite(path: Path) -> bool:
        """Ask the file, rather than believing its extension.

        ``.db``, ``.sqlite``, ``.sqlite3``, ``.db3`` and no extension at all are
        all real in the wild, and ``.db`` is used by unrelated formats too.
        """
        try:
            with path.open("rb") as handle:
                return handle.read(16) == SQLITE_MAGIC
        except OSError:
            return False

    def _attach_frame(self, frame: Any, path: Path, nickname: str) -> Slot:
        """Give an already-read frame its own database, named after the file."""
        self._workspace.attach_memory(nickname)
        table = _sanitize(path.stem, "table")
        try:
            info = self._workspace.insert_frame(
                frame, table, source=str(path), tag=nickname
            )
        except LoadError as exc:
            self._workspace.detach(nickname)
            raise AttachRefused(str(exc)) from exc
        return Slot(
            nickname=nickname,
            kind="file",
            source=str(path),
            tables=(info.name,),
            # This database is one we built. Nothing outside it is at risk from
            # a write, so composition needs no grant.
            writable=True,
        )

    def _attach_database(self, path: Path, nickname: str, *, writable: bool) -> Slot:
        try:
            self._workspace.attach_file(nickname, path, readonly=not writable)
            # Listing forces the schema to be read, which is where a file with an
            # unusable view fails. Opening alone would succeed and leave the slot
            # to fail later, on somebody else's query.
            tables = self._workspace.table_names(nickname)
        except Exception as exc:
            # Nothing half-open is left behind: the tag either works or is absent.
            self._workspace.detach(nickname)
            complaint = str(exc)
            if (
                "cannot reference objects in database" in complaint
                or "malformed database schema" in complaint
            ):
                # One bad view takes the whole database down, and SQLite's own
                # wording ("malformed database schema") points at corruption
                # rather than at the recoverable thing that actually happened.
                raise AttachRefused(
                    f"Could not attach {path}: it holds a view that names its "
                    f"tables with the name of the database it was built in, and "
                    f"no such name exists here — so SQLite rejects the whole file "
                    f"rather than just that view. Original complaint: {complaint}. "
                    f"Every datasource is opened as a database in its own right, "
                    f"so there is no name to attach it under that would resolve "
                    f"the view; it has to be rebuilt naming its tables "
                    f"unqualified (FROM sales, not FROM shop.sales)."
                ) from exc
            raise AttachRefused(f"Could not attach {path}: {complaint}") from exc
        return Slot(
            nickname=nickname,
            kind="database",
            source=str(path),
            tables=tables,
            writable=writable,
        )

    def _attach_url(
        self, database: str, nickname: str | None, *, writable: bool
    ) -> Attachment:
        """Open a service URL as its own engine, once the gate allows it."""
        try:
            url = make_url(database)
        except Exception as exc:
            raise AttachRefused(f"Not a usable datasource URL: {exc}") from exc

        safe = url.render_as_string(hide_password=True)
        if not config.active().network_enabled:
            raise AttachRefused(
                f"Opening {safe} would reach a network service, and network access "
                f"is off. Set network.enabled = true in the configuration file to "
                f"allow it."
            )

        self._refuse_duplicate(safe)
        chosen, collision = self._choose_nickname(
            nickname, url.database or url.drivername
        )
        evicted = self._make_room()
        slot = self._attach_engine(database, chosen, writable=writable)
        return Attachment(slot=slot, evicted=evicted, collided_with=collision)

    def _attach_engine(self, database: str, nickname: str, *, writable: bool) -> Slot:
        """Create a slot for a datasource addressed by URL.

        It goes into the same workspace, under the same kind of tag, as a flat
        file and a local database do. That is the whole of what makes the seven
        verbs work on it: there is one implementation of each, so a URL-addressed
        database cannot quietly support fewer of them than a file does.
        """
        url = make_url(database)
        safe = url.render_as_string(hide_password=True)
        try:
            self._workspace.attach(url, nickname, writable=writable)
            # Listing forces the schema to be read, so a datasource that opens
            # but cannot be inspected fails here rather than on someone's query.
            tables = self._workspace.table_names(nickname)
        except Exception as exc:
            # Nothing half-open is left behind: the tag either works or is absent.
            self._workspace.detach(nickname)
            # The message may carry the URL, so report the class and the
            # redacted form rather than the driver's own text.
            raise AttachRefused(
                f"Could not open {safe}: {type(exc).__name__}. "
                f"{self._first_line(exc, url)}"
            ) from exc

        slot = Slot(
            nickname=nickname,
            kind="engine",
            source=safe,
            tables=tables,
            writable=writable,
        )
        # Registered here rather than by the caller: this is the only path that
        # creates an engine slot, and it is reached directly by tests that
        # exercise the engine machinery without a server to route through.
        self._slots[nickname] = slot
        return slot

    @staticmethod
    def _first_line(exc: Exception, url: Any) -> str:
        """The driver's complaint, with the password scrubbed out of it."""
        message = str(exc).splitlines()[0] if str(exc) else ""
        if url.password:
            message = message.replace(str(url.password), "***")
        return message

    def _validate_nickname(self, nickname: str) -> None:
        if not _NICKNAME.match(nickname or ""):
            raise AttachRefused(
                f"{nickname!r} cannot be a nickname. Use a letter or underscore "
                f"followed by letters, digits or underscores — the nickname "
                f"is carried in the URI that opens its database, and quietly "
                f"rewriting it would hand back a handle you did not ask for."
            )

    # -- lifecycle and composition ------------------------------------------

    def detach(self, nickname: str) -> Slot:
        """Drop a slot on purpose, rather than waiting for FIFO to guess.

        The deliberate counterpart to eviction: it frees the slot against
        SQLite's ten-attachment ceiling, and takes the temp file with it if the
        database had been moved out to disk.
        """
        self.slot(nickname)  # Explains an evicted or unknown nickname.
        return self._release(nickname)

    def add_table(
        self,
        nickname: str,
        *,
        source: str,
        table: str | None = None,
        join_on: str | None = None,
        join_table: str | None = None,
    ) -> AddedTable:
        """Land another table inside a database that is already open.

        This is what makes the lookup arc work. Adding beside the existing
        tables rather than attaching a second slot is not only the friendlier
        mental model — it is what makes the result *keepable*, because ``save``
        writes one database rather than a join, so both sides have to live in
        the slot being saved.
        """
        slot = self._writable(nickname, "add a table to")
        name = self._table_name(table, source)
        if self._workspace.has_table(nickname, name):
            raise SlotError(
                f"{nickname}.{name} already exists, holding "
                f"{self._workspace.describe(nickname, name).row_count} rows. Drop "
                f"it first if you meant to replace it."
            )

        info = self._read_into(slot, name, source)

        report = None
        if join_on is not None:
            report = self._join_report(nickname, name, join_on, join_table)
        return AddedTable(info=info, join=report)

    def _read_into(self, slot: Slot, table: str, source: str) -> TableInfo:
        """Read a datasource into an existing slot as one more table."""
        try:
            path = resolve_read_path(source)
        except PathNotAllowed as exc:
            raise SlotError(str(exc)) from exc
        try:
            frame = read_frame(path)
            return self._workspace.insert_frame(
                frame, table, source=str(path), tag=slot.nickname
            )
        except LoadError as exc:
            raise SlotError(str(exc)) from exc

    def _table_name(self, table: str | None, source: str) -> str:
        """Settle on the table's name: given verbatim, or derived from the file."""
        if table is not None:
            if not _NICKNAME.match(table):
                raise SlotError(
                    f"{table!r} cannot be a table name. Use a letter or underscore "
                    f"followed by letters, digits or underscores."
                )
            return table
        return _sanitize(Path(source).stem, "table")

    def drop_table(self, nickname: str, table: str) -> None:
        """Remove a table from a slot. Composition needs both directions."""
        self._writable(nickname, "drop a table from")
        if not self._workspace.has_table(nickname, table):
            known = ", ".join(self._workspace.table_names(nickname)) or "none"
            raise SlotNotAvailable(
                f"No such table: {nickname}.{table}. In {nickname}: {known}."
            )
        try:
            self._workspace.drop_table(nickname, table)
        except LoadError as exc:
            raise SlotError(str(exc)) from exc

    def claimed_paths(self) -> dict[Path, str]:
        """Every file a live slot is sitting on, mapped to its nickname.

        Two kinds, and both matter for the same reason — something is reading
        them right now. A slot's original source (the SQLite file it attached,
        or the flat file it was built from and would be rebuilt from after an
        eviction), and the temp file a spilled slot moved into.

        Sources that are not filesystem paths — a database URL behind an engine
        slot — resolve to nothing here and simply do not appear.
        """
        claimed: dict[Path, str] = {}
        for slot in self._slots.values():
            if slot.spill_path is not None:
                claimed[slot.spill_path] = slot.nickname
            if url_scheme(slot.source) is not None:
                continue
            try:
                claimed[Path(slot.source).expanduser().resolve()] = slot.nickname
            except (OSError, ValueError):
                continue
        return claimed

    def save(self, nickname: str, path: str, *, force: bool = False) -> Path:
        """Write a slot's database out to a path the caller chose.

        The escape from ephemerality: a database built in memory, or moved to a
        temp file under memory pressure, becomes a file the user owns and can
        attach again another day.

        The slot is *copied*, not moved — it keeps answering under the same
        nickname, with the same write access it had. Re-attaching the saved file
        later is an ordinary attach, which is why it comes back read-only unless
        write is granted again.

        ``force`` replaces a file already at the path, and carries the *user's*
        decision to lose it. It does not extend to a file some live slot is
        sitting on, including this one's own source — that stays refused.
        """
        self.slot(nickname)
        try:
            target = resolve_write_path(path, force=force, claimed=self.claimed_paths())
        except PathNotAllowed as exc:
            raise SlotError(str(exc)) from exc

        try:
            self._workspace.snapshot(nickname, target)
        except UnsupportedOperation as exc:
            # Not a failure to save — a datasource that is not ours to save.
            # Nothing was written, so there is nothing to clean up, and the
            # backend's own words say what to do instead.
            raise SlotError(str(exc)) from exc
        except Exception as exc:
            # A partial file looks like a complete save, which is worse than none.
            target.unlink(missing_ok=True)
            raise SlotError(f"Could not save {nickname} to {target}: {exc}") from exc

        # SQLite creates the file 0o644 — world-readable, holding the user's
        # actual data. Narrowed immediately; the window is small and known.
        os.chmod(target, 0o600)
        return target

    # -- does the join actually line up? ------------------------------------

    def _join_report(
        self, nickname: str, added: str, key: str, join_table: str | None
    ) -> JoinReport:
        """Anti-join both ways, and say what has no partner on the other side.

        Both directions, because which one matters is not ours to guess: rows in
        the original file with nothing to look up are the usual worry, and rows
        in the new file that nothing refers to are the usual surprise.
        """
        existing = self._join_partner(nickname, added, join_table)
        column = self._shared_column(nickname, existing, added, key)

        missing_from_added, added_total = self._unmatched(
            nickname, existing, added, column
        )
        missing_from_existing, existing_total = self._unmatched(
            nickname, added, existing, column
        )
        return JoinReport(
            key=column,
            existing_table=existing,
            added_table=added,
            matched_keys=self._matched(nickname, existing, added, column),
            missing_from_added=missing_from_added,
            missing_from_added_total=added_total,
            missing_from_existing=missing_from_existing,
            missing_from_existing_total=existing_total,
        )

    def _join_partner(self, nickname: str, added: str, requested: str | None) -> str:
        """Which table the new one is being checked against."""
        others = [t for t in self._workspace.table_names(nickname) if t != added]
        if requested is not None:
            if requested not in others:
                known = ", ".join(others) or "no other table"
                raise SlotNotAvailable(
                    f"No such table to join against: {nickname}.{requested}. "
                    f"In {nickname}: {known}."
                )
            return requested
        if len(others) == 1:
            return others[0]
        if not others:
            raise SlotError(
                f"{nickname}.{added} is the only table in {nickname!r}, so there "
                f"is nothing to check the join against."
            )
        raise SlotError(
            f"{nickname!r} holds {', '.join(others)}, so which one "
            f"{nickname}.{added} should line up with has to be said explicitly."
        )

    def _shared_column(self, nickname: str, existing: str, added: str, key: str) -> str:
        """The key column, present on both sides or the check means nothing."""
        for table in (existing, added):
            names = [c.name for c in self._workspace.describe(nickname, table).columns]
            if key not in names:
                raise SlotError(
                    f"{nickname}.{table} has no column {key!r}. Its columns are: "
                    f"{', '.join(names)}."
                )
        return key

    def _unmatched(
        self, nickname: str, left: str, right: str, key: str
    ) -> tuple[tuple[Any, ...], int]:
        """Distinct key values in ``left`` with no partner in ``right``.

        ``NOT EXISTS`` rather than a ``LEFT JOIN ... IS NULL``, because it says
        what it means and stays right when the key column itself holds nulls —
        which are excluded, since a null key has no partner anywhere and saying
        so is noise rather than a finding.
        """
        absent = (
            f'FROM "{left}" l WHERE l."{key}" IS NOT NULL '
            f'AND NOT EXISTS (SELECT 1 FROM "{right}" r '
            f'WHERE r."{key}" = l."{key}")'
        )
        _, counted = self._workspace.query(
            nickname, f'SELECT count(*) FROM (SELECT DISTINCT l."{key}" {absent})'
        )
        _, sampled = self._workspace.query(
            nickname,
            f'SELECT DISTINCT l."{key}" {absent} ORDER BY 1 LIMIT {_UNMATCHED_SAMPLE}',
        )
        return tuple(row[0] for row in sampled), int(counted[0][0])

    def _matched(self, nickname: str, left: str, right: str, key: str) -> int:
        _, rows = self._workspace.query(
            nickname,
            f'SELECT count(*) FROM (SELECT DISTINCT l."{key}" '
            f'FROM "{left}" l WHERE EXISTS '
            f'(SELECT 1 FROM "{right}" r WHERE r."{key}" = l."{key}"))',
        )
        return int(rows[0][0])

    def _writable(self, nickname: str, action: str) -> Slot:
        """The slot for a nickname, refusing if it may not be changed."""
        slot = self.slot(nickname)
        if not slot.writable:
            raise NotWritable(
                f"{nickname!r} was attached read-only, so this server will not "
                f"{action} it. Attach {slot.source} again with writable=true if "
                f"you meant to change the file itself."
            )
        return slot

    # -- staying inside the memory budget -----------------------------------

    def relieve_memory(self) -> tuple[Slot, ...]:
        """Move databases out to disk until the session is inside its budget.

        **Called on the way *into* an operation, and that placement is the whole
        design.** The operation that crossed the budget has already finished by
        the time this runs, so an overshoot is tolerated exactly once: going to
        1.2 GB probably does not kill us, and unloading mid-load would. The
        *next* operation, whatever it is, pays the cost first.

        There is deliberately no pre-flight estimate. Deciding from a file's
        size or metadata what it *will* cost is the fail-open pattern that has
        already bitten this project; every number here is read from a database
        that already holds the data.

        Transparent, including to the LLM: nothing announces the move, and the
        slot answers afterwards under the same nickname with the same rights.
        """
        budget = config.active().memory_budget_mb * 1024 * 1024
        # ``resident_bytes`` answers ``None`` for a database whose data is not in
        # this process. Those are dropped rather than counted as zero: a tag that
        # cannot be measured is not a tag that is holding nothing, and spilling
        # it would move data that is already on disk.
        measured = {
            slot.nickname: self._workspace.resident_bytes(slot.nickname)
            for slot in self._slots.values()
            if self._in_memory(slot)
        }
        resident = {
            nickname: held for nickname, held in measured.items() if held is not None
        }

        total = sum(resident.values())
        if total <= budget:
            return ()

        # Largest first: it buys the most room per move, and the database that
        # pushed us over is usually the one that just arrived and is biggest.
        moved = []
        for nickname in sorted(resident, key=resident.__getitem__, reverse=True):
            if total <= budget:
                break
            moved.append(self._spill(self._slots[nickname]))
            total -= resident[nickname]
        return tuple(moved)

    @staticmethod
    def _in_memory(slot: Slot) -> bool:
        """Whether this slot's data is ours and sitting in RAM.

        A slot already backed by a file is not a candidate — moving it would buy
        nothing, and how much of it SQLite is holding in its page cache cannot
        be observed from Python anyway (``docs/CONSTRAINTS.md`` §6).
        """
        return slot.kind == "file" and slot.spill_path is None

    def _spill(self, slot: Slot) -> Slot:
        """Write one database to a temp file and re-attach it in the same slot.

        Re-assigning the existing key keeps the slot's place in insertion order,
        which is also eviction order — a database that moved to disk must not
        become the youngest and outlive the ones that were there before it.
        """
        target = self._scratch() / f"{slot.nickname}.sqlite"
        target.unlink(missing_ok=True)

        self._workspace.snapshot(slot.nickname, target)
        # Writable, because this database is still one we made: moving it must
        # not quietly take away rights the caller already had. Relocating rather
        # than detach-and-reattach keeps the tag's identity — it still reports
        # the source it came from, while its data now lives in the temp file.
        self._workspace.relocate(slot.nickname, target)

        moved = replace(slot, spill_path=target)
        self._slots[slot.nickname] = moved
        return moved

    def _scratch(self) -> Path:
        """The directory spilled databases live in, made on first need.

        OS-assigned, so nothing else knows the name and nothing collides with
        it. It goes away with the session.
        """
        if self._temp_dir is None:
            # Resolved, because the attach below records the resolved path and a
            # slot whose remembered file does not match the one SQLite is
            # holding is a temp file nobody will ever delete. On macOS the two
            # differ: /var is a symlink to /private/var.
            self._temp_dir = Path(tempfile.mkdtemp(prefix="localdata-")).resolve()
        return self._temp_dir

    # -- eviction -----------------------------------------------------------

    def _make_room(self) -> Eviction | None:
        """Drop the oldest slot if the incoming one would not fit.

        Called only once the nickname is settled, and the nickname is settled to
        one that is free — so this never has to consider replacing a slot in
        place. Dropping a slot on purpose is what ``detach`` is for.
        """
        if len(self._slots) < self.capacity():
            return None

        oldest = next(iter(self._slots))
        slot = self._slots[oldest]
        eviction = Eviction(
            nickname=slot.nickname,
            kind=slot.kind,
            source=slot.source,
            tables=slot.tables,
            reason=f"the slot limit of {self.capacity()} was reached",
        )
        self._release(oldest)
        self._evicted.append(eviction)
        return eviction

    def _release(self, nickname: str) -> Slot:
        slot = self._slots.pop(nickname)
        self._workspace.detach(nickname)
        self._discard_spill(slot)
        return slot

    @staticmethod
    def _discard_spill(slot: Slot) -> None:
        """Delete the temp file a spilled slot was living in.

        The file is ours — nobody else knows its name — so a slot going away is
        the end of it. Failing to remove it is not worth failing the operation
        the caller actually asked for.
        """
        if slot.spill_path is None:
            return
        try:
            slot.spill_path.unlink()
        except OSError:
            pass

    # -- using --------------------------------------------------------------

    def query(
        self, nickname: str, sql: str, limit: int | None = None
    ) -> tuple[list[str], list[tuple]]:
        """Run a statement against the database the nickname names."""
        slot = self.slot(nickname)
        try:
            return self._workspace.query(nickname, sql, limit=limit)
        except Exception as exc:
            self._explain(exc, sql, slot)
            raise

    def describe(self, nickname: str, table: str) -> TableInfo:
        """Describe one table inside a slot."""
        slot = self.slot(nickname)
        try:
            return self._workspace.describe(nickname, table, source=slot.source)
        except LoadError as exc:
            raise SlotNotAvailable(str(exc)) from exc

    def tables(self, nickname: str) -> tuple[str, ...]:
        """Refresh and return the table names inside a slot.

        Asked of the database every time rather than read from the ``Slot``,
        whose ``tables`` is the snapshot taken at attach time. A slot that has
        since been composed with ``add_table`` would otherwise answer with what
        it held when it arrived.
        """
        self.slot(nickname)
        return self._workspace.table_names(nickname)

    # -- explaining a failure -----------------------------------------------

    def _explain(self, exc: Exception, sql: str, routed: Slot) -> None:
        """Turn an engine's refusal into something a caller can act on.

        Applied only to a statement that has *already* failed, so a nickname
        appearing coincidentally in the text cannot cause a false refusal.
        """
        referenced = self._names_in(sql)

        for eviction in self._evicted:
            if eviction.nickname in referenced and eviction.nickname not in self._slots:
                raise SlotNotAvailable(
                    f"{sql.strip()[:60]}… refers to {eviction.nickname!r}, which was "
                    f"evicted ({eviction.reason}). It held "
                    f"{', '.join(eviction.tables) or 'no tables'} from "
                    f"{eviction.source}. Attach it again and retry."
                ) from exc

        elsewhere = [
            slot.nickname
            for slot in self._slots.values()
            if slot.nickname in referenced and slot.nickname != routed.nickname
        ]
        if elsewhere:
            raise SlotError(
                f"{routed.nickname!r} and {', '.join(sorted(elsewhere))} are separate "
                f"databases, and one statement cannot span two of them. Copy the "
                f"tables you need into one slot with add_table, then join there."
            ) from exc

    @staticmethod
    def _names_in(sql: str) -> set[str]:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql))

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        self._slots.clear()
        self._workspace.close()

        # The third exit for a spilled database, beside eviction and detach:
        # the session ending with connections still live. Nothing outside this
        # process knows these files, so leaving them behind is pure litter.
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
