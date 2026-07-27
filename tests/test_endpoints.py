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
from sqlalchemy import create_engine, inspect, text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import backend_for
from endpoints import ENDPOINTS, Endpoint, Unavailable, url_for

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

    endpoint: Endpoint
    url: str
    root: Path
    prefix: str

    def table(self, name: str = "people") -> str:
        """A table name unique to this test, so a leftover cannot confuse it."""
        return f"{self.prefix}_{name}"

    @property
    def csv(self) -> str:
        return str(self.root / "people.csv")


@pytest.fixture(params=ENDPOINTS, ids=lambda endpoint: endpoint.dialect)
def live(request, tmp_path):
    """A reachable endpoint, a fresh session, and no tables left behind.

    Cleanup drops every table carrying this test's prefix, and it goes through
    SQLAlchemy directly rather than through ``drop``: the test may have detached
    the slot, renamed the table or left the datasource read-only, and cleanup
    that depended on the state under test would leak exactly when a test failed.
    """
    endpoint: Endpoint = request.param
    try:
        url = url_for(endpoint)
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
    try:
        yield Live(endpoint=endpoint, url=url, root=root, prefix=prefix)
    finally:
        server_module._reset()
        _drop_everything_named(url, prefix)


def _drop_everything_named(url: str, prefix: str) -> None:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            names = [
                name
                for name in inspect(conn).get_table_names()
                if name.lower().startswith(prefix)
            ]
            preparer = conn.dialect.identifier_preparer
            for name in names:
                conn.execute(text(f"DROP TABLE {preparer.quote(name)}"))
            conn.commit()
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
    assert "***" in attached["source"]
    assert _password(live) not in json.dumps(attached)


def test_a_failed_open_does_not_echo_the_password(live):
    """The driver's own complaint frequently quotes the whole URL back."""
    wrong = live.url.replace(_password(live), "definitely-not-the-password")

    refused = call("attach", database=wrong, nickname="endpoint")

    assert refused["ok"] is False
    assert "definitely-not-the-password" not in refused["error"]


def _password(live: Live) -> str:
    from sqlalchemy.engine import make_url

    return str(make_url(live.url).password)


def test_residency_does_not_apply_to_a_server_side_database(live):
    """``None``, not ``0`` — the difference decides whether it gets spilled.

    A database on a server is holding nothing in *this* process, and there is no
    way to ask it how much it holds in its own. Answering ``0`` would read as an
    empty database and take it out of the memory budget's sight for the right
    reason by accident; ``None`` says the question does not apply.
    """
    engine = create_engine(live.url)
    try:
        assert backend_for(live.endpoint.dialect).resident_bytes(engine) is None
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
    defined = Table(
        table,
        metadata,
        Column("amount", Numeric(12, 4)),
        Column("day", Date),
        Column("moment", DateTime),
        Column("clock", Time),
        Column("blob", LargeBinary),
        Column("flag", Boolean),
    )
    engine = create_engine(live.url)
    try:
        with engine.begin() as conn:
            metadata.create_all(conn)
            conn.execute(
                defined.insert(),
                {
                    "amount": Decimal("12345.6789"),
                    "day": date(2024, 3, 1),
                    "moment": datetime(2024, 3, 1, 14, 30),
                    "clock": time(14, 30),
                    "blob": b"\x00\xff",
                    "flag": True,
                },
            )
    finally:
        engine.dispose()
    return table


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
    assert row["amount"] == pytest.approx(12345.6789)
    assert row["day"] == "2024-03-01"
    assert row["moment"] == "2024-03-01T14:30:00"
    assert row["blob"] == "0x00ff"
    # Booleans are the one case a dialect may answer with an integer, and both
    # spellings are JSON numbers or literals, so both are usable as they stand.
    assert row["flag"] in (True, 1)
    # A time of day is a *duration* on MySQL, which is the dialect's own reading
    # of the column and not something to paper over. Either spelling is ISO 8601.
    assert row["clock"] in ("14:30:00", "PT14H30M0S")


def test_info_describes_a_table_the_endpoint_holds(live):
    attach_writable(live)
    table = land_people(live)

    described = call("info", nickname="endpoint", table=table)

    assert described["ok"] is True, described
    assert described["rows"] == 5
    assert [column["name"] for column in described["columns"]] == [
        "name",
        "department",
        "salary",
    ]


def test_the_slot_lists_the_table_that_was_added_to_it(live):
    attach_writable(live)
    table = land_people(live)

    listed = call("info", nickname="endpoint")

    assert table in listed["tables"]
    assert {entry["table"]: entry["rows"] for entry in listed["contents"]}[table] == 5


def test_a_table_can_be_renamed_and_keeps_its_rows(live):
    attach_writable(live)
    table = land_people(live)
    renamed = live.table("renamed")

    answer = call("update", nickname="endpoint", type="table", name=table, to=renamed)

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
    """
    attach_writable(live)
    table = land_people(live)
    mixed = live.table("Mixed")

    answer = call("update", nickname="endpoint", type="table", name=table, to=mixed)

    assert answer["ok"] is True, answer
    assert answer["table"] == mixed
    assert mixed in call("info", nickname="endpoint")["tables"]


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

    written = call(
        "query", nickname="endpoint", sql=f"INSERT INTO {table} VALUES ('x', 'y', 1)"
    )

    assert written["ok"] is False, written
    # The refusal has to name where mutation lives, or the same statement is
    # simply sent again.
    assert "create" in written["error"]
    assert call("query", nickname="endpoint", sql=f"SELECT count(*) AS n FROM {table}")[
        "rows"
    ] == [[5]]


def test_query_refuses_ddl(live):
    attach_writable(live)

    made = call(
        "query", nickname="endpoint", sql=f"CREATE TABLE {live.table('nope')} (a INT)"
    )

    assert made["ok"] is False, made


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
    assert table in call("info", nickname="endpoint")["tables"]


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
