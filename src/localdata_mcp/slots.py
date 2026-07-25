"""Datasource slots: the one idea the tool surface is built on.

**A slot is always a database, addressed by a nickname.** Three things can fill
one, and none of them is a special case:

* a **flat file** becomes a brand-new in-memory database holding one table named
  after the file — so a later ``create_table`` can add a second table beside it
  under the same nickname;
* a **SQLite file** is attached read-only, arriving with the tables it already
  has;
* a **URL naming a service** becomes its own engine.

Because every slot is a database, addressing is uniformly ``nickname.table`` and
nothing above this module needs to know which kind it is holding.

**Ten slots, and the number is measured rather than chosen.** Every slot is an
attached database — a file-born slot attaches ``:memory:`` exactly as a database
file attaches itself — and SQLite raises ``too many attached databases - max 10``
on the eleventh. Slots are evicted oldest-first, and the eviction is *reported*,
because a caller told only ``no such table`` a minute later has to re-plan blind.

**One engine per slot; SQL joins within an engine, never across it.** ``ATTACH``
puts another SQLite database into the *same* connection, so all the SQLite-backed
slots are mutually joinable in one statement. A service reached over a URL cannot
be attached to a SQLite connection, so its slot is a separate engine. Crossing
that line needs the rows copied, which is a named act and not a side effect of
querying.

Both failures above are handled by **enriching an error that already happened**
rather than by inspecting SQL in advance. A substring check over a statement
would eventually refuse a legitimate query whose column happened to be named
like a slot; the same check applied only after the engine has already refused
cannot cause a false positive.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url

from . import config
from .loader import (
    READERS,
    ColumnInfo,
    LoadError,
    TableInfo,
    Workspace,
    _sanitize,
    read_frame,
)
from .paths import PathNotAllowed, resolve_read_path

__all__ = [
    "AttachRefused",
    "Attachment",
    "Collision",
    "Eviction",
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

#: A nickname becomes a SQL schema name, so it must be usable as an identifier.
_NICKNAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Schema names SQLite keeps for itself.
_RESERVED = {"main", "temp"}

#: How many evictions to remember, so a later reference can still be explained.
_EVICTION_MEMORY = 64


class SlotError(RuntimeError):
    """Something wrong with a slot or the request to make one."""


class AttachRefused(SlotError):
    """A datasource this server will not open, and why."""


class SlotNotAvailable(SlotError):
    """A nickname that names no live slot."""


def url_scheme(database: str) -> str | None:
    """The scheme of a datasource URL, or ``None`` for a filesystem path."""
    match = _URL.match(database)
    return match.group("scheme") if match else None


@dataclass(frozen=True)
class Slot:
    """One attached database, and how to reach it."""

    nickname: str
    #: ``"file"`` (a flat file, now its own in-memory database), ``"database"``
    #: (a SQLite file attached read-only), or ``"engine"`` (its own connection).
    kind: str
    #: Where it came from, with any password removed.
    source: str
    tables: tuple[str, ...]
    #: Set only for ``"engine"`` slots; the others live on the host connection.
    engine: Engine | None = None
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
        lowered = nickname.lower()
        return (
            nickname in self._slots
            or lowered in _RESERVED
            or lowered.startswith("sqlite_")
        )

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
                frame, table, source=str(path), schema=nickname
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
        except Exception as exc:
            raise AttachRefused(f"Could not attach {path}: {exc}") from exc
        return Slot(
            nickname=nickname,
            kind="database",
            source=str(path),
            tables=self._workspace.table_names(nickname),
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
        """Create a slot backed by its own engine.

        Separate from :meth:`_attach_url` so the engine machinery can be
        exercised without a server: the only driver present without one is
        SQLite, and a ``sqlite:`` URL routes to ``ATTACH`` at the surface.
        """
        url = make_url(database)
        safe = url.render_as_string(hide_password=True)
        try:
            engine = create_engine(url)
            tables = tuple(sorted(inspect(engine).get_table_names()))
        except Exception as exc:
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
            engine=engine,
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
                f"becomes a SQL name, and quietly rewriting it would hand back a "
                f"handle you did not ask for."
            )
        lowered = nickname.lower()
        if lowered in _RESERVED or lowered.startswith("sqlite_"):
            raise AttachRefused(f"{nickname!r} is a name SQLite reserves.")

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
        if slot.engine is not None:
            slot.engine.dispose()
        else:
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
        """Run a statement against the engine the nickname names."""
        slot = self.slot(nickname)
        try:
            if slot.engine is None:
                return self._workspace.query(sql, limit=limit)
            return self._query_engine(slot.engine, sql, limit)
        except Exception as exc:
            self._explain(exc, sql, slot)
            raise

    @staticmethod
    def _query_engine(
        engine: Engine, sql: str, limit: int | None
    ) -> tuple[list[str], list[tuple]]:
        with engine.connect() as connection:
            result = connection.execute(text(sql))
            if result.returns_rows is False:
                return [], []
            names = list(result.keys())
            rows = result.fetchmany(limit) if limit else result.fetchall()
            return names, [tuple(row) for row in rows]

    def describe(self, nickname: str, table: str) -> TableInfo:
        """Describe one table inside a slot."""
        slot = self.slot(nickname)
        if slot.engine is not None:
            return self._describe_engine_table(slot, table)
        try:
            return self._workspace.describe(nickname, table, source=slot.source)
        except LoadError as exc:
            raise SlotNotAvailable(str(exc)) from exc

    def _describe_engine_table(self, slot: Slot, table: str) -> TableInfo:
        assert slot.engine is not None
        if table not in slot.tables:
            raise SlotNotAvailable(f"No such table: {slot.nickname}.{table}")
        inspector = inspect(slot.engine)
        columns = [
            ColumnInfo(name=column["name"], declared_type=str(column["type"]))
            for column in inspector.get_columns(table)
        ]
        with slot.engine.connect() as connection:
            rows = connection.execute(text(f'SELECT count(*) FROM "{table}"')).scalar()
        return TableInfo(
            name=table,
            row_count=int(rows or 0),
            columns=columns,
            source=slot.source,
            schema=slot.nickname,
        )

    def tables(self, nickname: str) -> tuple[str, ...]:
        """Refresh and return the table names inside a slot."""
        slot = self.slot(nickname)
        if slot.engine is not None:
            return slot.tables
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
            if slot.nickname in referenced
            and slot.nickname != routed.nickname
            and (slot.engine is None) != (routed.engine is None)
        ]
        if elsewhere:
            raise SlotError(
                f"{routed.nickname!r} and {', '.join(sorted(elsewhere))} are held by "
                f"different engines, and one statement cannot span two engines. "
                f"Copy the tables you need into one slot first, then join there."
            ) from exc

    @staticmethod
    def _names_in(sql: str) -> set[str]:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql))

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        for slot in list(self._slots.values()):
            if slot.engine is not None:
                slot.engine.dispose()
        self._slots.clear()
        self._workspace.close()
