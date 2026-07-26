"""End-to-end tests over the MCP protocol itself.

These drive the server the way a client does — listing tools, calling them by
name with JSON arguments, reading structured results back — rather than calling
the Python functions underneath. That distinction matters: a tool can be
perfectly correct as a function and still be unusable over the wire because its
signature does not serialise, its docstring never reaches the client, or its
return value is not JSON.

Everything runs in-process against the same server object ``main()`` serves, so
no subprocess, no registration and no port are involved.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pathlib import Path

import pytest
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config

ASSETS = Path(__file__).parent / "assets"

#: The whole surface. Named here so a tool added or removed without thinking
#: about the shape of the surface fails a test rather than passing quietly.
TOOLS = {"attach", "detach", "info", "query", "add_table", "drop_table", "save"}


@pytest.fixture(autouse=True)
def session(monkeypatch, tmp_path):
    """A fresh registry and a fresh allowed root for every test."""
    root = tmp_path / "root"
    root.mkdir()
    for asset in ASSETS.iterdir():
        (root / asset.name).write_bytes(asset.read_bytes())
    config_module.use(Config(roots=(root,)))
    server_module._reset()
    yield root
    server_module._reset()


def call(tool: str, **arguments):
    """Call a tool over the protocol and return its decoded payload."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


def listed_tools():
    async def _run():
        async with Client(server_module.mcp) as client:
            return await client.list_tools()

    return asyncio.run(_run())


def _build_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE departments (department TEXT, floor INTEGER)")
    connection.executemany(
        "INSERT INTO departments VALUES (?, ?)",
        [("Engineering", 3), ("Sales", 1), ("Marketing", 2)],
    )
    connection.commit()
    connection.close()


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_the_surface_is_seven_verbs_each_with_a_description():
    tools = listed_tools()
    assert {tool.name for tool in tools} == TOOLS

    # A tool with no description is invisible to an agent choosing between them.
    for tool in tools:
        assert tool.description and len(tool.description) > 20, tool.name

    by_name = {tool.name: tool for tool in tools}
    for name in TOOLS:
        assert by_name[name].inputSchema["properties"], name


def test_the_nickname_is_required_everywhere_it_routes():
    """It names the engine to execute against, so it cannot be optional."""
    by_name = {tool.name: tool for tool in listed_tools()}
    for name in ("detach", "query", "add_table", "drop_table", "save"):
        assert "nickname" in by_name[name].inputSchema["required"], name

    # The two exceptions, and both are deliberate: attach *derives* a nickname,
    # and info with none reports the whole session.
    assert "nickname" not in by_name["attach"].inputSchema.get("required", [])
    assert "nickname" not in by_name["info"].inputSchema.get("required", [])


def test_the_instructions_teach_the_premise_where_the_model_reads_it():
    """A CSV is a table inside a database, and that has to be said up front.

    The addressing has to be there too, and stated as the thing an agent will
    otherwise get wrong: it will reach for ``nickname.table``, which is what the
    previous design taught and what every listing used to print.
    """
    instructions = server_module.mcp.instructions
    assert "Each call names one datasource" in instructions
    assert "not FROM shop.sales" in instructions
    assert "add_table" in instructions


# ---------------------------------------------------------------------------
# A datasource is a database named by its nickname
# ---------------------------------------------------------------------------


def test_a_file_attaches_as_a_database_holding_one_table(session):
    attached = call("attach", database=str(session / "simple.csv"), nickname="staff")

    assert attached["ok"] is True
    assert attached["kind"] == "file"
    assert attached["nickname"] == "staff"
    assert attached["tables"] == ["simple"]
    assert attached["writable"] is True
    assert attached["evicted"] is None
    assert attached["collided_with"] is None


def test_a_nickname_is_derived_when_none_is_given(session):
    attached = call("attach", database=str(session / "simple.csv"))
    assert attached["nickname"] == "simple"
    assert attached["tables"] == ["simple"]


def test_attach_then_query_over_the_protocol(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    answer = call("query", nickname="staff", sql="SELECT sum(salary) AS t FROM simple")
    assert answer["ok"] is True
    assert answer["columns"] == ["t"]
    assert answer["rows"][0][0] == 345000


def test_the_same_source_twice_is_refused_and_names_where_it_lives(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    again = call("attach", database=str(session / "simple.csv"), nickname="other")

    assert again["ok"] is False
    assert "'staff'" in again["error"]


def test_a_colliding_nickname_is_disambiguated_and_both_slots_survive(session):
    """Pointing a handle at a second datasource must not take the first away.

    The caller is told which name it actually got, and what it collided with,
    because a caller that assumes it got the name it asked for addresses the
    wrong database.
    """
    call("attach", database=str(session / "simple.csv"), nickname="slot")
    second = call("attach", database=str(session / "mixed_tabs.tsv"), nickname="slot")

    assert second["ok"] is True
    assert second["nickname"] == "slot_2"
    assert second["tables"] == ["mixed_tabs"]
    assert second["collided_with"] == {
        "nickname": "slot",
        "source": str(session / "simple.csv"),
    }
    assert call("info")["slots_used"] == 2


# ---------------------------------------------------------------------------
# info: one verb, three altitudes
# ---------------------------------------------------------------------------


def test_info_with_nothing_reports_every_datasource_and_the_posture(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    listing = call("info")

    assert listing["ok"] is True
    assert listing["datasources"] == [
        {
            "nickname": "staff",
            "kind": "file",
            "source": str(session / "simple.csv"),
            "writable": True,
            "tables": ["simple"],
        }
    ]
    assert listing["slots_used"] == 1
    assert listing["slots_available"] == 10
    # The posture is reported so the LLM can see what it may reach.
    assert str(session) in listing["roots"]
    assert listing["path_limited"] is True


def test_info_with_a_nickname_reports_that_datasources_tables(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    call("add_table", nickname="staff", source=str(session / "mixed_tabs.tsv"))

    detail = call("info", nickname="staff")

    assert detail["ok"] is True
    assert detail["nickname"] == "staff"
    assert sorted(detail["tables"]) == ["mixed_tabs", "simple"]
    assert {entry["table"]: entry["rows"] for entry in detail["contents"]} == {
        "simple": 5,
        "mixed_tabs": 5,
    }


def test_info_with_a_table_describes_its_columns_and_types(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    described = call("info", nickname="staff", table="simple")

    assert described["ok"] is True
    assert described["table"] == "simple"
    assert described["rows"] == 5
    types = {column["name"]: column["type"] for column in described["columns"]}
    assert types["salary"] == "INTEGER"
    assert types["department"] == "TEXT"


def test_info_reports_an_unknown_nickname_as_an_answer(session):
    answer = call("info", nickname="nothing")
    assert answer["ok"] is False
    assert "nothing" in answer["error"]


# ---------------------------------------------------------------------------
# query: rows back, or the whole result written out
# ---------------------------------------------------------------------------


def test_query_limit_is_reported_as_truncation(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    answer = call("query", nickname="staff", sql="SELECT * FROM simple", limit=2)
    assert answer["row_count"] == 2
    assert answer["truncated"] is True

    full = call("query", nickname="staff", sql="SELECT * FROM simple", limit=0)
    assert full["row_count"] == 5
    assert full["truncated"] is False


def test_a_path_turns_the_same_query_into_an_export(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "out.csv"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple ORDER BY name",
        path=str(target),
    )
    assert result["ok"] is True
    assert result["rows_written"] == 5
    assert target.read_text().splitlines()[0] == "name,salary"


def test_the_export_ignores_the_display_limit(session):
    """A limit is about what fits in an answer, never about what lands in a file."""
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "full.csv"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT * FROM simple",
        limit=2,
        path=str(target),
    )
    assert result["rows_written"] == 5


def test_the_export_refuses_to_clobber_then_takes_the_users_answer(session):
    """The refusal has to read as a question, or the agent just retries.

    It relayed a path the user chose rather than choosing one, so replacing what
    is there is the user's call — and force is how that answer comes back.
    """
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "out.csv"
    target.write_text("existing content\n")

    refused = call(
        "query",
        nickname="staff",
        sql="SELECT name FROM simple",
        path=str(target),
    )
    assert refused["ok"] is False
    assert "Ask the user" in refused["error"]
    assert target.read_text() == "existing content\n"

    forced = call(
        "query",
        nickname="staff",
        sql="SELECT name FROM simple",
        path=str(target),
        force=True,
    )
    assert forced["ok"] is True
    assert target.read_text().splitlines()[0] == "name"


def test_an_export_will_not_be_forced_over_an_attached_file(session):
    """force is the user's consent to lose a spare file, nothing more."""
    source = session / "simple.csv"
    call("attach", database=str(source), nickname="staff")
    before = source.read_text()

    refused = call(
        "query",
        nickname="staff",
        sql="SELECT name FROM simple",
        path=str(source),
        force=True,
    )
    assert refused["ok"] is False
    assert "'staff'" in refused["error"]
    assert source.read_text() == before


# ---------------------------------------------------------------------------
# Joining across datasources — the capability the tool exists for
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The lookup arc: add a table here, and say whether it lines up
# ---------------------------------------------------------------------------


def test_a_second_file_lands_inside_the_open_database_and_joins(session):
    write_csv(session / "sales.csv", "sku,qty\na,3\nb,4\n")
    write_csv(session / "prices.csv", "sku,price\na,10\nb,20\n")
    call("attach", database=str(session / "sales.csv"), nickname="shop")

    added = call("add_table", nickname="shop", source=str(session / "prices.csv"))

    assert added["ok"] is True
    assert added["table"] == "prices"
    assert added["rows"] == 2

    answer = call(
        "query",
        nickname="shop",
        sql=(
            "SELECT s.sku, s.qty * p.price FROM sales s "
            "JOIN prices p ON s.sku = p.sku ORDER BY s.sku"
        ),
    )
    assert answer["rows"] == [["a", 30], ["b", 80]]


def test_an_incomplete_join_is_reported_with_the_values_that_do_not_match(session):
    write_csv(session / "sales.csv", "sku,qty\na,3\nb,4\nc,5\n")
    write_csv(session / "prices.csv", "sku,price\na,10\nz,99\n")
    call("attach", database=str(session / "sales.csv"), nickname="shop")

    added = call(
        "add_table",
        nickname="shop",
        source=str(session / "prices.csv"),
        join_on="sku",
    )

    report = added["join"]
    assert report["complete"] is False
    assert report["key"] == "sku"
    assert report["existing_table"] == "sales"
    assert report["added_table"] == "prices"
    assert report["matched_keys"] == 1
    assert report["missing_from_added"] == {"values": ["b", "c"], "total": 2}
    assert report["missing_from_existing"] == {"values": ["z"], "total": 1}


def test_a_complete_join_says_so(session):
    write_csv(session / "sales.csv", "sku,qty\na,3\n")
    write_csv(session / "prices.csv", "sku,price\na,10\n")
    call("attach", database=str(session / "sales.csv"), nickname="shop")

    added = call(
        "add_table",
        nickname="shop",
        source=str(session / "prices.csv"),
        join_on="sku",
    )
    assert added["join"]["complete"] is True
    assert added["join"]["missing_from_added"]["values"] == []


def test_adding_to_a_read_only_datasource_says_how_to_allow_it(session):
    _build_database(session / "hr.db")
    call("attach", database=str(session / "hr.db"), nickname="hr")

    answer = call("add_table", nickname="hr", source=str(session / "simple.csv"))
    assert answer["ok"] is False
    assert "writable=true" in answer["error"]


def test_the_write_grant_is_honoured_over_the_protocol(session):
    _build_database(session / "hr.db")
    call("attach", database=str(session / "hr.db"), nickname="hr", writable=True)

    added = call("add_table", nickname="hr", source=str(session / "simple.csv"))
    assert added["ok"] is True
    assert added["table"] == "simple"


def test_dropping_a_table_leaves_the_rest_answering(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    call("add_table", nickname="staff", source=str(session / "mixed_tabs.tsv"))

    dropped = call("drop_table", nickname="staff", table="mixed_tabs")

    assert dropped["ok"] is True
    assert dropped["dropped"] == "mixed_tabs"
    assert dropped["tables"] == ["simple"]
    assert call("query", nickname="staff", sql="SELECT count(*) FROM simple")[
        "rows"
    ] == [[5]]


# ---------------------------------------------------------------------------
# detach and save: the ends of a slot's life
# ---------------------------------------------------------------------------


def test_detaching_frees_the_slot_and_reports_what_went(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    closed = call("detach", nickname="staff")

    assert closed["ok"] is True
    assert closed["tables"] == ["simple"]
    assert closed["slots_used"] == 0
    assert call("info")["datasources"] == []


def test_saving_then_attaching_again_brings_the_whole_session_back(session):
    write_csv(session / "sales.csv", "sku,qty\na,3\nb,4\n")
    write_csv(session / "prices.csv", "sku,price\na,10\nb,20\n")
    call("attach", database=str(session / "sales.csv"), nickname="shop")
    call("add_table", nickname="shop", source=str(session / "prices.csv"))
    target = session / "keep.db"

    saved = call("save", nickname="shop", path=str(target))
    assert saved["ok"] is True
    assert sorted(saved["tables"]) == ["prices", "sales"]

    call("detach", nickname="shop")
    restored = call("attach", database=str(target), nickname="kept")

    assert sorted(restored["tables"]) == ["prices", "sales"]
    # Like any other outside database, it comes back read-only.
    assert restored["writable"] is False
    answer = call(
        "query",
        nickname="kept",
        sql=(
            "SELECT s.sku, s.qty * p.price FROM sales s "
            "JOIN prices p ON s.sku = p.sku ORDER BY s.sku"
        ),
    )
    assert answer["rows"] == [["a", 30], ["b", 80]]


def test_saving_refuses_an_existing_file_until_the_user_says_replace_it(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "keep.db"
    call("save", nickname="staff", path=str(target))

    refused = call("save", nickname="staff", path=str(target))
    assert refused["ok"] is False
    assert "Ask the user" in refused["error"]

    assert call("save", nickname="staff", path=str(target), force=True)["ok"] is True


def test_saving_will_not_be_forced_over_the_file_it_came_from(session):
    """The sharpest case: the slot's own source. Unlinking it would succeed
    while the slot carried on answering from an inode with no name."""
    source = session / "simple.csv"
    call("attach", database=str(source), nickname="staff")

    refused = call("save", nickname="staff", path=str(source), force=True)

    assert refused["ok"] is False
    assert "'staff'" in refused["error"]
    assert source.exists()


# ---------------------------------------------------------------------------
# Outgrowing memory, invisibly
# ---------------------------------------------------------------------------


def test_a_database_moved_to_disk_says_nothing_and_answers_the_same(session):
    """The spill is transparent: no field appears, no wording changes."""
    config_module.use(Config(roots=(session,), memory_budget_mb=1))
    rows = "\n".join(f"{index},label{index},{index * 2}" for index in range(60_000))
    write_csv(session / "big.csv", f"id,label,amount\n{rows}\n")

    call("attach", database=str(session / "big.csv"), nickname="big")
    before = call("query", nickname="big", sql="SELECT count(*), sum(amount) FROM big")
    described_before = call("info", nickname="big")

    # The next call is the one that pays for the overshoot, whatever it is.
    call("info")

    after = call("query", nickname="big", sql="SELECT count(*), sum(amount) FROM big")
    assert after["rows"] == before["rows"]
    assert after["rows"][0][0] == 60_000

    # Described exactly as before: same kind, same source, same rights, same
    # tables, same row counts. A caller has no way to tell the move happened.
    assert call("info", nickname="big") == described_before
    assert "spill" not in json.dumps(call("info"))


# ---------------------------------------------------------------------------
# The slot limit, and what an eviction has to say
# ---------------------------------------------------------------------------


def test_the_eviction_is_reported_in_the_attachment_that_caused_it(session):
    config_module.use(Config(roots=(session,), slots=2))
    call("attach", database=str(session / "simple.csv"), nickname="a")
    call("attach", database=str(session / "mixed_tabs.tsv"), nickname="b")

    third = call("attach", database=str(session / "no_header.csv"), nickname="c")

    assert third["ok"] is True
    assert third["evicted"]["nickname"] == "a"
    assert third["evicted"]["tables"] == ["simple"]
    assert third["evicted"]["source"] == str(session / "simple.csv")


def test_querying_an_evicted_datasource_says_it_was_evicted(session):
    config_module.use(Config(roots=(session,), slots=1))
    call("attach", database=str(session / "simple.csv"), nickname="a")
    call("attach", database=str(session / "mixed_tabs.tsv"), nickname="b")

    answer = call("query", nickname="a", sql="SELECT * FROM simple")
    assert answer["ok"] is False
    assert "evicted" in answer["error"]
    # Enough to rebuild it without guessing.
    assert str(session / "simple.csv") in answer["error"]


# ---------------------------------------------------------------------------
# Errors reach the caller as answers, not as exceptions
# ---------------------------------------------------------------------------


def test_a_bad_path_is_answered_not_raised(session, tmp_path):
    answer = call("attach", database=str(tmp_path / "elsewhere.csv"), nickname="x")
    assert answer["ok"] is False
    assert "outside the allowed paths" in answer["error"]


def test_a_missing_file_is_answered(session):
    answer = call("attach", database=str(session / "absent.csv"), nickname="x")
    assert answer["ok"] is False
    assert "No such file" in answer["error"]


def test_an_unusable_nickname_is_answered(session):
    answer = call("attach", database=str(session / "simple.csv"), nickname="my-data")
    assert answer["ok"] is False
    assert "my-data" in answer["error"]


def test_an_unknown_nickname_is_answered(session):
    answer = call("query", nickname="nothing", sql="SELECT 1")
    assert answer["ok"] is False
    assert "nothing" in answer["error"]


def test_a_network_url_is_answered_while_the_network_is_closed(session):
    answer = call(
        "attach",
        database="postgresql://user:hunter2@db.example.com/sales",
        nickname="pg",
    )
    assert answer["ok"] is False
    assert "network" in answer["error"]
    assert "hunter2" not in answer["error"]


def test_invalid_sql_returns_the_engine_message(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    answer = call("query", nickname="staff", sql="SELECT * FROM nonexistent")
    assert answer["ok"] is False
    assert "no such table" in answer["error"].lower()


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM simple",
        "INSERT INTO simple (name) VALUES ('x')",
        "UPDATE simple SET salary = 0",
        "CREATE TABLE t (a TEXT)",
        "CREATE VIEW v AS SELECT name FROM simple",
        "DROP TABLE simple",
        "ALTER TABLE simple RENAME TO other",
        "CREATE INDEX i ON simple (name)",
        "PRAGMA journal_mode = WAL",
    ],
)
def test_query_reads_and_refuses_every_way_of_writing(session, sql):
    """A query is a query. The datasource here is fully writable, which is the
    point: the refusal is a property of the verb, not of the grant."""
    attached = call("attach", database=str(session / "simple.csv"), nickname="staff")
    assert attached["writable"] is True

    answer = call("query", nickname="staff", sql=sql)

    assert answer["ok"] is False, sql
    assert "does not write" in answer["error"], answer["error"]
    # Still intact, and still answering.
    rows = call("query", nickname="staff", sql="SELECT count(*) FROM simple")
    assert rows["rows"][0][0] == 5


def test_the_refusal_says_what_was_attempted_and_where_to_go(session):
    """An agent told only "denied" cannot tell which clause offended.

    DDL is described as a schema change rather than as the INSERT that SQLite
    happens to refuse first — see ``_describe_action``. Reporting the raw first
    refusal would tell an agent that wrote CREATE VIEW it attempted an INSERT.
    """
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    ddl = call(
        "query",
        nickname="staff",
        sql="CREATE VIEW v AS SELECT name FROM simple",
    )
    assert "change the database schema" in ddl["error"]
    assert "add_table" in ddl["error"]

    plain = call(
        "query", nickname="staff", sql="INSERT INTO simple (name) VALUES ('x')"
    )
    assert "asks to INSERT" in plain["error"]


def test_a_read_only_datasource_is_still_readable(session):
    _build_database(session / "hr.db")
    call("attach", database=str(session / "hr.db"), nickname="hr")

    answer = call("query", nickname="hr", sql="SELECT count(*) FROM departments")
    assert answer["ok"] is True
    assert answer["rows"][0][0] == 3


def test_mixed_columns_are_flagged_on_attach(session):
    """The signal that keeps an agent from trusting a wrong average."""
    attached = call(
        "attach", database=str(session / "messy_mixed_types.csv"), nickname="messy"
    )
    assert attached["ok"] is True
    assert any("coerce text to 0" in warning for warning in attached["warnings"])
    assert any("In messy, table messy_mixed_types" in w for w in attached["warnings"])


def test_the_mixed_column_detail_is_available_from_info(session):
    call("attach", database=str(session / "messy_mixed_types.csv"), nickname="messy")
    described = call("info", nickname="messy", table="messy_mixed_types")
    assert "id" in described["mixed_columns"]


# ---------------------------------------------------------------------------
# The README is the one document everybody reads, and it drifted before
# ---------------------------------------------------------------------------


def test_every_tool_the_readme_names_actually_exists():
    """A README claiming tools that do not exist is worse than none.

    This branch inherited one advertising seventy-one tools across thirteen
    database types, none of which were still there. Checking it mechanically is
    what stops that happening quietly a second time.
    """
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    real = {tool.name for tool in listed_tools()}

    # The verb table, and every call written at the start of a code-block line.
    # Deliberately not every backticked `name(...)` in the prose: that also
    # matches SQL functions like avg(), which are not ours to provide.
    claimed = set(re.findall(r"^\| `([a-z_]+)\(", readme, re.MULTILINE))
    claimed |= set(re.findall(r"^([a-z_]+)\(", readme, re.MULTILINE))
    invented = claimed - real
    assert not invented, f"README names tools that do not exist: {sorted(invented)}"

    # And the surface it documents is the whole surface, not a flattering slice.
    assert real <= claimed, f"README omits: {sorted(real - claimed)}"


def test_mixed_column_warning_prescribes_a_filter_that_works(session):
    """The remedy has to fit the column that triggered it.

    Live-agent validation found this the hard way: four of six agents were told
    to ``filter with typeof(col)='integer'``, tried exactly that, and got every
    row back — a CSV column is declared TEXT, so every value's storage class is
    text whatever it contains. An instruction that cannot work on the case that
    produced it is worse than none, because it is followed.
    """
    (session / "sentinels.csv").write_text("v\n1\n2\n3\npending\n")
    attached = call("attach", database=str(session / "sentinels.csv"))

    warning = attached["warnings"][0]
    # Not prescribed — and said not to work, because an agent reaches for it
    # unprompted and needs telling why the whole column comes back.
    assert "typeof(v)='integer'" not in warning
    assert "typeof() cannot tell them apart" in warning
    # It names the values, so the caller can write the filter without a round trip.
    assert "pending" in warning

    column = attached["loaded"][0]["columns"][0]
    assert column["non_numeric_examples"] == ["pending"]

    # And the filter the warning does prescribe returns the numbers, nothing else.
    result = call(
        "query",
        nickname=attached["nickname"],
        sql="SELECT avg(CAST(v AS REAL)) FROM sentinels WHERE v NOT IN ('pending')",
    )
    assert result["rows"][0][0] == pytest.approx(2.0)


def test_storage_class_mixture_still_gets_the_typeof_remedy(session):
    """Where ``typeof`` does discriminate, it is still the right answer."""
    external = session / "heterogeneous.sqlite"
    with sqlite3.connect(external) as conn:
        conn.execute("CREATE TABLE t (v)")
        conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), ("pending",)])

    call("attach", database=str(external), nickname="ext")
    warning = call("info", nickname="ext", table="t")["warnings"][0]
    assert "typeof" in warning


def test_query_does_not_advertise_writes_it_refuses(session):
    """The docstring is what an agent without the skill reads, and it lied.

    It said writes 'go through here too, and succeed only where the datasource is
    writable', while the tool refuses every one of them on any datasource. An
    agent found the contradiction against the server instructions unprompted.
    """
    documented = {tool.name: tool.description for tool in listed_tools()}["query"]
    assert "reads" in documented
    assert not re.search(r"[Ww]rites.*(go through|succeed)", documented)

    attached = call("attach", database=str(session / "simple.csv"))
    refused = call(
        "query",
        nickname=attached["nickname"],
        sql="INSERT INTO simple (id) VALUES (1)",
    )
    assert refused["ok"] is False
    assert "does not write" in refused["error"]


def test_attach_and_add_table_say_their_answer_needs_no_info_call(session):
    """Three of three skill-less agents called ``info`` straight after ``attach``.

    Every one of them reported the call as waste: the payload was identical to
    what they already held. The skill says not to; the docstrings — all a bare
    agent gets — did not.
    """
    documented = {tool.name: tool.description for tool in listed_tools()}
    for name in ("attach", "add_table"):
        assert "info" in documented[name], f"{name} never mentions the redundant call"


def test_readme_never_teaches_a_qualified_table_name(session):
    """The addressing that ``ATTACH`` made possible, outliving ``ATTACH``.

    This is the fourth time the project has written an implementation
    side-effect up as a specification, and the second time this particular one
    survived its own removal — in the README's two worked examples, hours after
    every other document had been corrected. A grep is cheaper than a fifth time.
    """
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    # `FROM shop.sales`, `JOIN sales.prices s` — a dotted name in shipped SQL.
    taught = re.findall(r"(?:FROM|JOIN)\s+([a-z_]+\.[a-z_]+)", readme)
    assert not taught, f"README still teaches qualified table names: {taught}"


def test_info_describes_a_table_inside_an_attached_database(session):
    """The whole "attach a SQLite file" datasource class had no describe at all.

    Found by driving the surface as an agent rather than by a test: every table
    the server loaded itself is described from what it remembered loading, so the
    suite never reached the other branch — the one that asks the database. That
    branch inspected over the read engine, whose authorizer refuses PRAGMA, and
    SQLAlchemy's inspector speaks PRAGMA. It did not fail politely either: the
    driver error escaped as a protocol error rather than as a refusal an agent
    could read.
    """
    external = session / "kept.sqlite"
    with sqlite3.connect(external) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER, note TEXT)")
        conn.executemany("INSERT INTO notes VALUES (?, ?)", [(1, "a"), (2, "b")])

    call("attach", database=str(external), nickname="kept")

    detail = call("info", nickname="kept", table="notes")
    assert detail["ok"] is True
    assert [column["name"] for column in detail["columns"]] == ["id", "note"]
    assert detail["rows"] == 2

    # And at the altitude above it, which describes every table in the slot.
    slot = call("info", nickname="kept")
    assert slot["ok"] is True
    assert slot["contents"] == [{"table": "notes", "rows": 2}]


def test_a_saved_database_can_be_described_when_it_comes_back(session):
    """Arc 2's return journey: keep it this week, open it next week.

    ``save`` then ``attach`` is the sequence the whole "keep this" promise rests
    on, and the first thing anyone does with a database they have just reopened
    is ask what is in it.
    """
    attached = call("attach", database=str(session / "simple.csv"))
    call("save", nickname=attached["nickname"], path=str(session / "kept.db"))
    call("detach", nickname=attached["nickname"])

    reopened = call("attach", database=str(session / "kept.db"), nickname="lastweek")
    described = call("info", nickname="lastweek", table=reopened["tables"][0])
    assert described["ok"] is True
    assert described["columns"]
