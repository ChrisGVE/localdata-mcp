"""Every verb against a real endpoint database, one dialect at a time.

An endpoint database is one this server reaches over a connection rather than
holds on disk. :mod:`localdata_mcp.dialects` claims that the generic ``Backend``
is the whole answer for such a database — that opening, querying, composing and
inspecting need no per-dialect code — and this module is where that claim is
either true or found out. Every test runs against each dialect that has a
container answering, and the ones that have none are skipped with a reason
(``pytest -rs`` prints them), never quietly passed over.

The tests drive the tool surface over the protocol, as :mod:`test_server` does,
because that surface is what an agent touches. A verb that works when called as
a Python function and fails as a tool is still broken.

Containers come from ``docker-compose.test.yml`` and are **not** started here:
starting a two-gigabyte Oracle image inside a test run would make a suite that
takes a minute take five. :mod:`tests.endpoints` says how to start one in the
skip reason.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import pytest
from fastmcp import Client
from sqlalchemy import DDL, create_engine, inspect
from sqlalchemy.engine import make_url

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import backend_for_url
from endpoints import TARGETS, Target, Unavailable, applied, reach

pytestmark = pytest.mark.endpoint

#: What every test loads. Lower-case names throughout, deliberately: half these
#: dialects fold an unquoted identifier and the other half do not, so a column
#: called ``Name`` would turn every statement below into a quoting exercise
#: instead of a test of the verb. Quoting gets its own test, where it is the
#: subject rather than the noise.
PEOPLE = """\
name,department,salary
alice,engineering,75000
bob,sales,65000
carol,marketing,70000
dave,engineering,80000
eve,hr,55000
"""


def call(tool: str, **arguments):
    """Call a tool over the protocol and return its decoded payload.

    The same in-process client :mod:`test_server` uses. Duplicated rather than
    imported: importing one test module from another makes the first one's
    fixtures collect twice.
    """

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# One live dialect, cleaned up after
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Live:
    """A dialect that is answering, and a name prefix nothing else will use."""

    target: Target
    url: str
    root: Path
    prefix: str

    @property
    def endpoint(self):
        """The database behind this target, whichever way it was reached."""
        return self.target.endpoint

    def table(self, name: str = "people") -> str:
        """A table name unique to this test, so a leftover cannot confuse it."""
        return f"{self.prefix}_{name}"

    @property
    def csv(self) -> str:
        return str(self.root / "people.csv")


@pytest.fixture(params=TARGETS, ids=lambda target: target.name)
def live(request, tmp_path):
    """A reachable endpoint, a fresh session, and no tables left behind.

    Parametrised over **targets** rather than endpoints: an endpoint reached a
    second way is the same database and the same verbs, so every test here runs
    against every authentication mode without one being written for it. Ids are
    unchanged for the credentialed mode — ``postgres`` is still ``postgres`` —
    and a variant reads as ``postgres[env-password]``.

    A mode that keeps its credential outside the URL needs its environment held
    for the whole test and not a moment longer, so it is applied here and undone
    with the rest of the teardown.

    Cleanup drops every table carrying this test's prefix, and it goes through
    SQLAlchemy directly rather than through ``drop``: the test may have detached
    the slot, renamed the table or left the datasource read-only, and cleanup
    that depended on the state under test would leak exactly when a test failed.
    """
    target: Target = request.param
    try:
        reached = reach(target)
    except Unavailable as exc:
        pytest.skip(str(exc))

    root = tmp_path / "root"
    root.mkdir()
    (root / "people.csv").write_text(PEOPLE)
    # Network access is off by default and an endpoint URL is exactly what that
    # gate is about, so a test of endpoints has to grant it explicitly.
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()

    prefix = f"t{uuid.uuid4().hex[:8]}"
    with applied(reached.environ):
        try:
            yield Live(target=target, url=reached.url, root=root, prefix=prefix)
        finally:
            server_module._reset()
            # Inside `applied`, deliberately: for a mode whose credential lives
            # in the environment, cleanup is a connection like any other and
            # would fail to authenticate outside it — leaving every table behind
            # and reddening the *next* test instead of this one.
            _drop_everything_named(reached.url, prefix)


def _drop_everything_named(url: str, prefix: str) -> None:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            names = [
                name
                for name in inspect(conn).get_table_names()
                if name.lower().startswith(prefix)
            ]
        if not names:
            # Deliberately not "open a transaction and commit nothing": FreeTDS
            # answers a commit with no transaction behind it by raising.
            return
        with engine.begin() as conn:
            preparer = conn.dialect.identifier_preparer
            for name in names:
                # `DDL`, not `text`, for the reason `Backend.rename_table`
                # records (#59): only the DDL construct carries the "this is
                # schema" signal a dialect can route on, and YDB refuses any
                # schema statement that arrives inside a transaction. As `text`
                # this cleanup raised `Scheme operations cannot be executed
                # inside transaction` and left every test's tables behind.
                conn.execute(DDL(f"DROP TABLE {preparer.quote(name)}"))
    finally:
        engine.dispose()


def attach_writable(live: Live, nickname: str = "endpoint") -> dict:
    payload = call("attach", database=live.url, nickname=nickname, writable=True)
    assert payload["ok"] is True, payload
    return payload


def land_people(live: Live, nickname: str = "endpoint") -> str:
    """Put the fixture CSV into the endpoint through ``create``, and name it.

    Through the server's own verb rather than through the driver: a fixture that
    built the table another way would be testing the fixture, and ``create`` on
    an endpoint is one of the things in question.
    """
    table = live.table()
    made = call("create", nickname=nickname, type="table", source=live.csv, table=table)
    assert made["ok"] is True, made
    return table


# ---------------------------------------------------------------------------
# Opening one
# ---------------------------------------------------------------------------


def test_an_endpoint_attaches_as_an_engine_and_keeps_its_password(live):
    """``kind`` says how it is reached, and the payload must not carry secrets."""
    attached = call("attach", database=live.url, nickname="endpoint")

    assert attached["ok"] is True, attached
    assert attached["kind"] == "engine"
    # Read-only is the default for a datasource that came from outside, whatever
    # rights the credentials themselves carry.
    assert attached["writable"] is False

    secret = _password(live)
    if secret is None:
        # Several endpoints take no password at all, and so does every mode that
        # keeps the credential outside the URL. There is nothing to redact, and
        # inventing a `***` for an absent credential would tell the caller a
        # secret was carried when none was.
        assert "***" not in attached["source"]
        return
    assert "***" in attached["source"]
    if secret == "":
        # An empty password is supplied and checked, so it is redacted like any
        # other — but "the secret does not appear in the payload" cannot be asked
        # of it: `"" in anything` is true, so the assertion below would fail on a
        # payload that is perfectly clean. Stated rather than skipped, because
        # the redaction above is the half that can be asserted and is asserted.
        return
    assert secret not in json.dumps(attached)


def test_a_failed_open_does_not_echo_the_password(live):
    """The driver's own complaint frequently quotes the whole URL back."""
    secret = _password(live)
    if secret is None:
        pytest.skip(
            "reached with no password in the URL, so there is no wrong one to "
            "send and no failed open to inspect. Either the database has no "
            "authentication to configure, or the mode keeps the credential "
            "somewhere the URL does not carry"
        )
    wrong = make_url(live.url).set(password="definitely-not-the-password")

    refused = call("attach", database=wrong.render_as_string(hide_password=False))

    assert refused["ok"] is False, refused
    assert "definitely-not-the-password" not in refused["error"]


def _password(live: Live) -> str | None:
    """The password this endpoint is reached with, or ``None`` where there is none.

    ``None`` rather than the string ``"None"``, which is what ``str()`` of an
    absent password gives and which would then be searched for in the payload —
    an assertion that passes for the wrong reason.

    ``""`` is a third answer and a real one: an empty password was supplied and
    is checked, which is neither "no password" nor a password that can be
    searched for. Callers here distinguish all three.
    """
    return make_url(live.url).password


def test_the_backend_is_chosen_by_the_engine_answering_not_by_its_dialect(live):
    """Issue #45: a dialect names a wire protocol and a driver, never an engine.

    Reached by dialect alone, YugabyteDB was handed ``Backend(name="postgresql")``
    — so a refusal named PostgreSQL to somebody who had opened YugabyteDB, and
    an answer of its own could not have been given without also changing real
    PostgreSQL's. The whole endpoint suite passed throughout, because nothing
    asked the backend what it thought it was. This asks.

    Meaningful for every entry and not only the one that shares: the resolution
    must leave the eight that own their dialect exactly where they were.
    """
    assert backend_for_url(live.url).name == live.endpoint.engine_name


def test_residency_does_not_apply_to_a_server_side_database(live):
    """``None``, not ``0`` — the difference decides whether it gets spilled.

    A database on a server is holding nothing in *this* process, and there is no
    way to ask it how much it holds in its own. Answering ``0`` would read as an
    empty database and take it out of the memory budget's sight for the right
    reason by accident; ``None`` says the question does not apply.
    """
    engine = create_engine(live.url)
    try:
        assert backend_for_url(live.url).resident_bytes(engine) is None
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Composing in one
# ---------------------------------------------------------------------------


def test_a_file_lands_in_the_endpoint_and_reads_back(live):
    attach_writable(live)
    table = land_people(live)

    answer = call(
        "query",
        nickname="endpoint",
        sql=f"SELECT department, sum(salary) AS total FROM {table} "
        f"GROUP BY department ORDER BY department",
    )

    assert answer["ok"] is True, answer
    # Numbers, as numbers. MySQL answers a SUM over an integer column with a
    # DECIMAL, and a Decimal left alone reaches the wire as the string
    # "155000" — which an agent then compares and adds as text.
    assert answer["rows"] == [
        ["engineering", 155000],
        ["hr", 55000],
        ["marketing", 70000],
        ["sales", 65000],
    ]


def test_a_gap_in_a_text_column_arrives_as_null(live):
    """A blank cell is NULL, not whatever a bound NaN renders as (#71).

    Here rather than in the local-slot suite because the local slot cannot be
    asked: SQLite has no NaN and stores a bound one as NULL, so it answered
    correctly while PostgreSQL stored the string ``'NaN'`` and SQL Server
    ``'nan'`` — each its own driver's rendering of a value this server should
    never have bound. ``IS NULL`` matched neither, so every gap in a text column
    silently counted as present.

    Both halves are asserted. That the count is right is not enough on its own:
    a backend could answer 1 while holding a string that merely *sorts* like a
    gap, so the stored values are read back and compared as well.
    """
    attach_writable(live)
    gapped = live.root / "gapped.csv"
    gapped.write_text("name,note\nalice,first\nbob,\ncarol,third\n")
    table = live.table("gapped")
    made = call(
        "create",
        nickname="endpoint",
        type="table",
        source=str(gapped),
        table=table,
    )
    assert made["ok"] is True, made

    counted = call(
        "query",
        nickname="endpoint",
        sql=f"SELECT count(*) AS gaps FROM {table} WHERE note IS NULL",
    )
    assert counted["ok"] is True, counted
    assert counted["rows"][0][0] == 1

    read_back = call(
        "query",
        nickname="endpoint",
        sql=f"SELECT note FROM {table} ORDER BY name",
    )
    assert read_back["ok"] is True, read_back
    assert [row[0] for row in read_back["rows"]] == ["first", None, "third"]


def _build_typed_table(live: Live) -> str:
    """A table of types SQLite does not have, built the way the user's would be.

    Through SQLAlchemy's *generic* types, so each dialect renders its own: this
    is standing in for a database somebody else made, which is the only kind an
    endpoint ever is. The three storage classes a loaded file produces would
    never reach ``Decimal`` or ``bytes``, so a fixture that went through
    ``create`` could not exercise this at all.
    """
    from sqlalchemy import (
        Boolean,
        Column,
        Date,
        DateTime,
        LargeBinary,
        MetaData,
        Numeric,
        Table,
        Time,
    )

    table = live.table("typed")
    metadata = MetaData()
    values = {
        "amount": Decimal("12345.6789"),
        "day": date(2024, 3, 1),
        "moment": datetime(2024, 3, 1, 14, 30),
        "clock": time(14, 30),
        "blob": b"\x00\xff",
        "flag": True,
    }
    columns = [
        Column("amount", Numeric(12, 4)),
        Column("day", Date),
        Column("moment", DateTime),
        Column("clock", Time),
        Column("blob", LargeBinary),
        Column("flag", Boolean),
    ]
    # A column this backend cannot be given is left out rather than asserted
    # around: this test is about what comes *back*, and nothing can come back
    # from a column that was never made. Which ones those are is the backend's
    # to say — Oracle has no time-of-day type, and ClickHouse's driver cannot
    # bind bytes — because a dialect fact stated in a fixture is the same defect
    # as one stated in shared code.
    unstorable = backend_for_url(live.url).unstorable_column_types()
    for column in [c for c in columns if type(c.type).__name__ in unstorable]:
        columns.remove(column)
        values.pop(column.name)

    backend = backend_for_url(live.url)

    # A backend that will not make a table without a primary key gets one here
    # too. `insert_frame` adds a surrogate for the same reason and does not cover
    # this fixture, which writes below the verbs — and the key is declared on a
    # column the table already has rather than added as a seventh, because this
    # table holds exactly one row and any of its columns is unique across it.
    # `amount` is nominated because it is the one column every backend in the
    # catalogue can hold, so the choice cannot interact with
    # `unstorable_column_types` above.
    if backend.requires_primary_key():
        columns[0].primary_key = True

    # Whatever this dialect's CREATE TABLE cannot be written without — asked of
    # the backend rather than branched on here, because a dialect fact stated in
    # a fixture is the same defect as one stated in shared code.
    defined = Table(table, metadata, *columns, **backend.table_options())
    # Whether the schema and the row may travel in one transaction. False only on
    # Firebird, which prepares statements against committed metadata and so
    # cannot address the table it has just made (issue #53). Asked of the seam for
    # the same reason the three lines above are: a fixture that branched on the
    # dialect name would be a dialect fact stated in a fixture, and this
    # fixture writes below the verbs so `insert_frame`'s own split does not cover
    # it.
    together = backend.sees_new_tables_in_transaction()

    def _fill(conn) -> None:
        conn.execute(defined.insert(), values)
        # This fixture writes below the verbs, so nothing has asked the backend
        # to make the row readable. On every transactional database the commit
        # does it; on CrateDB the row is durable and invisible until the index
        # refreshes, and the test would read an empty table.
        backend.settle(conn, table)

    engine = create_engine(live.url)
    try:
        with engine.begin() as conn:
            metadata.create_all(conn)
            if together:
                _fill(conn)
        if not together:
            with engine.begin() as conn:
                _fill(conn)
    finally:
        engine.dispose()
    return table


#: Every column :func:`_build_typed_table` defines, in one place so the test
#: below can state which of them a backend dropped rather than only checking the
#: ones that happened to survive.
_TYPED_COLUMNS = ("amount", "day", "moment", "clock", "blob", "flag")


def test_every_value_reaches_the_wire_as_something_json_can_hold(live):
    """The value space of an endpoint is much wider than SQLite's three classes.

    Each of these came back from a live container as a Python object JSON has no
    form for, and each has one spelling here rather than whatever the serialiser
    would have reached for. The failure this prevents is quiet: a number that
    arrives as text still looks like an answer.
    """
    table = _build_typed_table(live)
    call("attach", database=live.url, nickname="endpoint")

    answer = call("query", nickname="endpoint", sql=f"SELECT * FROM {table}")

    assert answer["ok"] is True, answer
    row = dict(zip(answer["columns"], answer["rows"][0]))

    # Which columns the fixture left out is the backend's statement, so it is
    # asserted rather than tolerated. `if "blob" in row` — what this replaces —
    # passes just as quietly when a column is missing for a reason nobody
    # declared, which is the fail-open shape this project keeps meeting.
    absent = {
        name
        for name, declared in (
            ("amount", "Numeric"),
            ("blob", "LargeBinary"),
            ("clock", "Time"),
        )
        if declared in backend_for_url(live.url).unstorable_column_types()
    }
    assert set(_TYPED_COLUMNS) - absent == set(row), (row, absent)

    # Three spellings, and each is the truth its dialect can tell. Oracle's DATE
    # carries a time of day whether or not one was given, so it comes back as
    # the instant it actually is rather than as a bare date. CrateDB has no date
    # type at all — a date is a TIMESTAMP, and its instants are UTC, so the `Z`
    # is information rather than noise. All three are ISO 8601, which is what
    # the rule asks for; none is an epoch integer, which is what it forbids.
    assert row["day"] in (
        "2024-03-01",
        "2024-03-01T00:00:00",
        "2024-03-01T00:00:00Z",
    )
    assert row["moment"] in ("2024-03-01T14:30:00", "2024-03-01T14:30:00Z")
    # Booleans are the one case a dialect may answer with an integer, and both
    # spellings are JSON numbers or literals, so both are usable as they stand.
    assert row["flag"] in (True, 1)
    if "amount" in row:
        assert row["amount"] == pytest.approx(12345.6789)
    if "blob" in row:
        assert row["blob"] == "0x00ff"
    # A time of day is a *duration* on MySQL, which is the dialect's own reading
    # of the column and not something to paper over. Either spelling is ISO 8601.
    if "clock" in row:
        assert row["clock"] in ("14:30:00", "PT14H30M0S")


def test_directory_describes_a_table_the_endpoint_holds(live):
    """The file's columns, and anything the backend made this server add.

    The list was once exactly the CSV's three columns, which was true of every
    backend until one refused to make a table without a primary key. A file has
    no key to offer, so ``create`` supplies a surrogate — a real column, which
    ``SELECT *`` will return and which ``directory`` must therefore name. Asked of the
    seam rather than of the dialect, for the reason the other fixtures here are.

    **The note is asserted, not just the column.** An added column that ``directory``
    lists but does not explain is a table shape the caller has to
    reverse-engineer; a wrong explanation attached to a right outcome passes
    every test that only checks the outcome.
    """
    attach_writable(live)
    table = land_people(live)

    described = call("directory", nickname="endpoint", table=table)

    assert described["ok"] is True, described
    assert described["rows"] == 5

    names = [column["name"] for column in described["columns"]]
    from_file = ["name", "department", "salary"]

    if backend_for_url(live.url).requires_primary_key():
        assert names == ["_row", *from_file]
        # The surrogate holds each row's position in the file, so over five rows
        # it is 0..4 — the column is not merely present, it is populated.
        keys = call("query", nickname="endpoint", sql=f"SELECT _row FROM {table}")
        assert sorted(row[0] for row in keys["rows"]) == [0, 1, 2, 3, 4]
        assert any(
            "_row" in note and "primary key" in note
            for note in described.get("warnings", ())
        ), described.get("warnings")
    else:
        assert names == from_file
        assert not any("_row" in note for note in described.get("warnings", ()))


def test_the_slot_lists_the_table_that_was_added_to_it(live):
    attach_writable(live)
    table = land_people(live)

    listed = call("directory", nickname="endpoint")

    assert table in listed["tables"]
    assert {entry["table"]: entry["rows"] for entry in listed["contents"]}[table] == 5


def test_an_index_can_be_created_and_dropped(live):
    """Indexing a text column is where the dialects stop agreeing.

    MySQL and MariaDB will not key on a whole ``TEXT`` column — and every text
    column a loaded file produces is ``TEXT``, so this verb did not work there at
    all. What they can do is index a prefix, which still answers a lookup on the
    whole value, and saying so is the difference between an index that covers
    less than asked and one that quietly does.
    """
    attach_writable(live)
    table = land_people(live)

    made = call(
        "create", nickname="endpoint", type="index", table=table, columns=["department"]
    )

    backend = backend_for_url(live.url)
    if not backend.builds_indexes():
        # ClickHouse, whose secondary indexes are data-skipping indexes that
        # cannot be reflected and do not answer a lookup; and Trino, which holds
        # no data and so has nothing of its own to index. On both the verb does
        # not apply, and the refusal has to name the database that declined —
        # a caller told "no" by "a generic datasource" has been told nothing.
        # What each says instead is the backend's own words, not this fixture's.
        assert made["ok"] is False, made
        assert backend.name in made["error"]
        assert call("directory", nickname="endpoint", table=table)["indexes"] == []
        return

    assert made["ok"] is True, made
    assert made["columns"] == ["department"]
    for warning in made.get("warnings", []):
        assert "first" in warning and "characters" in warning

    listed = call("directory", nickname="endpoint", table=table)
    assert made["index"] in [index["index"] for index in listed["indexes"]]

    gone = call("drop", nickname="endpoint", type="index", name=made["index"])
    assert gone["ok"] is True, gone
    assert call("directory", nickname="endpoint", table=table)["indexes"] == []


def test_a_table_can_be_renamed_and_keeps_its_rows(live):
    attach_writable(live)
    table = land_people(live)
    renamed = live.table("renamed")

    answer = call("update", nickname="endpoint", type="table", name=table, to=renamed)

    backend = backend_for_url(live.url)
    if not backend.renames_tables():
        # Firebird, which has no rename-table statement in any version. The verb
        # does not apply, and the same three things have to hold as for an index
        # a backend cannot build: it says no, it names the database that declined
        # rather than "a generic datasource", and — the part that makes this a
        # test rather than a formality — the table it would not rename is still
        # there under its original name with its rows intact. A refusal that had
        # half-renamed something would pass the first two assertions.
        assert answer["ok"] is False, answer
        assert backend.name in answer["error"]
        listed = call("directory", nickname="endpoint")["tables"]
        assert table in listed and renamed not in listed
        assert call(
            "query", nickname="endpoint", sql=f"SELECT count(*) AS n FROM {table}"
        )["rows"] == [[5]]
        return

    assert answer["ok"] is True, answer
    assert answer["rows"] == 5
    assert renamed in answer["tables"] and table not in answer["tables"]
    assert call(
        "query", nickname="endpoint", sql=f"SELECT count(*) AS n FROM {renamed}"
    )["rows"] == [[5]]


def test_a_rename_onto_a_name_that_needs_quoting_keeps_the_case(live):
    """Half these dialects fold an unquoted identifier; none may be allowed to.

    The name goes through the dialect's own preparer, so ``Mixed`` arrives as
    ``Mixed`` rather than as whatever the database would have folded it to. A
    dialect that folded it would report the rename as done and then have no such
    table, which is the shape of a silently wrong answer rather than an error.

    Trino is where quoting stops being enough: it lower-cases every identifier
    at the connector, so ``Mixed`` is *stored* as ``mixed`` however it is
    written. There the guarantee cannot be that the case survives, so it is the
    other one — that the name reported back is the name the database actually
    has. Both halves are asserted for every dialect, and which one applies is
    the backend's to say rather than this fixture's.

    Firebird cannot rename at all, so there is no case for its case to survive;
    the sibling test above is where that refusal is asserted.
    """
    attach_writable(live)
    table = land_people(live)
    mixed = live.table("Mixed")

    answer = call("update", nickname="endpoint", type="table", name=table, to=mixed)

    if not backend_for_url(live.url).renames_tables():
        assert answer["ok"] is False, answer
        return

    assert answer["ok"] is True, answer
    if backend_for_url(live.url).folds_identifiers():
        assert answer["table"] == mixed.lower()
    else:
        assert answer["table"] == mixed
    # The part that holds everywhere, and the one an agent's next statement
    # depends on: what came back is findable under exactly that spelling.
    listed = call("directory", nickname="endpoint")["tables"]
    assert answer["table"] in listed
    assert table not in listed


def test_a_table_can_be_dropped(live):
    attach_writable(live)
    table = land_people(live)

    gone = call("drop", nickname="endpoint", type="table", name=table)

    assert gone["ok"] is True, gone
    assert table not in gone["tables"]


# ---------------------------------------------------------------------------
# What a query may not do
# ---------------------------------------------------------------------------


def test_query_refuses_a_write_even_where_the_datasource_permits_it(live):
    """``writable=true`` governs ``create`` and ``drop``, never ``query``.

    The tool says so plainly, so an agent believes it: a statement that came
    back ``ok`` is a statement that happened. On a server-side database the
    write reaches a connection that never commits and is rolled back — which
    from outside looks exactly like a statement that succeeded and returned no
    rows. Reporting that as success is the lie this refuses.
    """
    attach_writable(live)
    table = land_people(live)
    backend = backend_for_url(live.url)

    # **The statement has to be one this database would otherwise accept**, or
    # the refusal proves nothing: a statement the parser rejects never reaches
    # the posture under test, and the assertions below then pass on a refusal
    # that has nothing to do with the read connection. Both halves of this were
    # measured on YDB, and each was a way for this test to pass while the posture
    # it defends was absent.
    #
    # The columns are named rather than left to positional order — valid SQL
    # everywhere, and required there: a bare `INSERT ... VALUES` is answered with
    # `requires specification of table columns`.
    #
    # And a surrogate primary key, where the backend demanded one, must be given
    # a value: an `INSERT` that omits the key column is refused for *that*, so
    # the row could never land whatever the posture said. With `read_posture`
    # removed the row did land, at 5 -> 6 rows, and this test still passed.
    columns = ["name", "department", "salary"]
    values = ["'x'", "'y'", "1"]
    if backend.requires_primary_key():
        columns.insert(0, "_row")
        # Past the five the file landed, so it cannot collide with one of them
        # and be refused as a duplicate key instead of as a write.
        values.insert(0, "99")
    written = call(
        "query",
        nickname="endpoint",
        sql=(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(values)})"
        ),
    )

    assert written["ok"] is False, written
    # The refusal has to name where mutation lives, or the same statement is
    # simply sent again.
    assert "create" in written["error"]
    if backend.dml_survives_refusal():
        # A database with no transactions and no read-only session applies the
        # write before there is anything to refuse it with — the same shape as
        # the DDL caveat below, asked about rows. What is tested here is that
        # the refusal is still issued and the row really did land; claiming it
        # did not would send the caller looking for a row that exists.
        #
        # Settled first, because on such a backend the row is durable before it
        # is readable, and counting straight away would report the refresh
        # timer's state rather than the database's.
        engine = create_engine(live.url)
        try:
            with engine.begin() as conn:
                backend.settle(conn, table)
        finally:
            engine.dispose()
        expected = [[6]]
    else:
        expected = [[5]]

    after = call(
        "query", nickname="endpoint", sql=f"SELECT count(*) AS n FROM {table}"
    )["rows"]
    assert after == expected, after


def test_ddl_through_query_never_reaches_the_database(live):
    """Refusing after the fact is not enough where a rollback cannot undo it.

    MySQL and MariaDB commit DDL implicitly, so a ``CREATE TABLE`` sent through
    a read connection that never commits was *permanent* — the table was still
    there on the next connection. The transactional floor assumes a write can be
    left uncommitted and thereby undone, and on those two dialects it cannot be,
    so the read posture refuses the statement rather than declining to keep it.

    Oracle is where that stops being possible: its implicit commit runs *before*
    the statement is considered and ends the read-only transaction, and it has no
    session-level equivalent. So there the table really does appear, and what is
    tested is that the refusal says so — a refusal claiming a statement did not
    happen, when it did, is the same lie as calling a rolled-back write a
    success.
    """
    attach_writable(live)
    orphan = live.table("nope")

    # A key where the backend demands one, asked of the seam rather than branched
    # on: YDB refuses a keyless `CREATE TABLE` at parse time, and a statement the
    # parser rejects never reaches the read posture — so this would have asserted
    # a refusal that had nothing to do with the connection it was sent through.
    key = (
        " , PRIMARY KEY (a)" if backend_for_url(live.url).requires_primary_key() else ""
    )
    made = call("query", nickname="endpoint", sql=f"CREATE TABLE {orphan} (a INT{key})")

    assert made["ok"] is False, made
    assert "create" in made["error"]

    engine = create_engine(live.url)
    try:
        with engine.connect() as conn:
            landed = orphan.lower() in [
                name.lower() for name in inspect(conn).get_table_names()
            ]
    finally:
        engine.dispose()

    if backend_for_url(live.url).ddl_survives_refusal():
        assert landed, "the caveat is only honest if the statement really ran"
        assert "already taken effect" in made["error"]
    else:
        assert not landed
        assert "already taken effect" not in made["error"]


def test_a_read_only_attach_refuses_create_and_drop(live):
    """The grant is the server's, not the credentials'."""
    attach_writable(live, nickname="writable")
    table = land_people(live, nickname="writable")
    call("detach", nickname="writable")

    call("attach", database=live.url, nickname="endpoint")

    refused = call("create", nickname="endpoint", type="table", source=live.csv)
    assert refused["ok"] is False
    dropped = call("drop", nickname="endpoint", type="table", name=table)
    assert dropped["ok"] is False
    # And the table is still there, which is the part that matters.
    assert table in call("directory", nickname="endpoint")["tables"]


# ---------------------------------------------------------------------------
# Getting rows out of one
# ---------------------------------------------------------------------------


def test_save_refuses_and_says_how_to_keep_the_rows_instead(live):
    """There is no local database to write out, and saying so is the answer."""
    attach_writable(live)
    land_people(live)

    refused = call("save", nickname="endpoint", path=str(live.root / "kept.db"))

    assert refused["ok"] is False
    assert "create" in refused["error"]
    assert not (live.root / "kept.db").exists()


def test_rows_can_be_copied_into_a_slot_that_saves(live):
    """The documented route for a datasource this server reaches but cannot hold."""
    attach_writable(live)
    table = land_people(live)
    call("attach", database=live.csv, nickname="local")

    exported = call(
        "query",
        nickname="endpoint",
        sql=f"SELECT * FROM {table}",
        path=str(live.root / "out.csv"),
    )
    assert exported["ok"] is True, exported

    added = call(
        "create",
        nickname="local",
        type="table",
        source=str(live.root / "out.csv"),
        table="copied",
    )
    assert added["ok"] is True, added
    assert added["rows"] == 5

    kept = call("save", nickname="local", path=str(live.root / "kept.db"))
    assert kept["ok"] is True, kept
    assert sorted(kept["tables"]) == ["copied", "people"]


def test_a_result_can_be_written_straight_to_a_columnar_file(live):
    """A format with no delimiter, so the writer table is exercised too."""
    pytest.importorskip("pyarrow")
    attach_writable(live)
    table = land_people(live)
    target = live.root / "out.parquet"

    written = call(
        "query", nickname="endpoint", sql=f"SELECT * FROM {table}", path=str(target)
    )

    assert written["ok"] is True, written
    assert written["rows_written"] == 5
    assert target.exists()


def test_detach_frees_the_slot_and_leaves_the_database_alone(live):
    attach_writable(live)
    table = land_people(live)

    closed = call("detach", nickname="endpoint")
    assert closed["ok"] is True, closed
    assert closed["slots_used"] == 0

    # Detaching closed a connection; it did not drop anything.
    again = attach_writable(live)
    assert table in again["tables"]
