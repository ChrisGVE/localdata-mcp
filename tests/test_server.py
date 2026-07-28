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
import subprocess
import zipfile
from pathlib import Path

import foreign
import pytest
from fastmcp import Client
from sqlalchemy import Integer, Text

from localdata_mcp import config as config_module
from localdata_mcp import export as export_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config

ASSETS = Path(__file__).parent / "assets"

#: The whole surface. Named here so a tool added or removed without thinking
#: about the shape of the surface fails a test rather than passing quietly.
TOOLS = {"attach", "detach", "info", "query", "create", "update", "drop", "save"}


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
    foreign.build_database(
        path,
        "departments",
        [("department", Text), ("floor", Integer)],
        [("Engineering", 3), ("Sales", 1), ("Marketing", 2)],
    )


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_the_surface_is_eight_verbs_each_with_a_description():
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
    for name in ("detach", "query", "create", "drop", "save"):
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
    assert "create(nickname, type='table'" in instructions


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
    call(
        "create", nickname="staff", type="table", source=str(session / "mixed_tabs.tsv")
    )

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


def test_a_query_returns_its_whole_result(session):
    """No row cap, and no parameter offering one.

    A cap was tried and removed. It measured rows while the thing it was meant
    to bound is the size of the answer, so a hundred rows of a wide table walked
    straight past it. Fewer rows is what SQL ``LIMIT`` is for, and the caller is
    the one who knows how many it wants.
    """
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    answer = call("query", nickname="staff", sql="SELECT * FROM simple")
    assert answer["row_count"] == 5
    assert "truncated" not in answer

    asked = call("query", nickname="staff", sql="SELECT * FROM simple LIMIT 2")
    assert asked["row_count"] == 2

    schema = {tool.name: tool for tool in listed_tools()}["query"].inputSchema
    assert "limit" not in schema["properties"]


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


def test_the_export_writes_every_row_the_statement_selected(session):
    """A path is the answer to a result too large to return, so it truncates never."""
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "full.csv"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT * FROM simple",
        path=str(target),
    )
    assert result["rows_written"] == 5
    assert len(target.read_text().splitlines()) == 6  # header included


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


def test_the_export_suffix_chooses_the_format(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "out.tsv"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple ORDER BY name",
        path=str(target),
    )

    assert result["ok"] is True
    assert target.read_text().splitlines()[0] == "name\tsalary"


def test_an_ods_export_is_an_ods_file_and_not_a_workbook_under_a_false_name(session):
    """`.ods` wrote XLSX, and every test we had said it was fine.

    The read-back check that should have caught it could not: pandas sniffs a
    workbook's real format from its contents and reads the file happily, so the
    round trip returned the right rows out of a file of the wrong format. The
    magic-byte check could not either — ODS and XLSX are both Zip archives and
    both open `PK\\x03\\x04`. What distinguishes them is what is *inside* the
    archive, so that is what this asserts.

    The cause was one argument. `pandas.DataFrame.to_excel` infers its engine
    from the suffix of a `str` path but **not** of a `pathlib.Path` — measured
    on pandas 3.0.2, where a `Path` silently falls back to openpyxl — and the
    writer is handed a `Path`. The fix names the engine instead of inferring it.
    """
    pytest.importorskip("odf")
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "out.ods"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple",
        path=str(target),
    )
    assert result["ok"] is True

    inside = set(zipfile.ZipFile(target).namelist())
    assert "mimetype" in inside, f"not an OpenDocument package: {sorted(inside)}"
    assert "META-INF/manifest.xml" in inside, sorted(inside)
    assert "[Content_Types].xml" not in inside, "this is a workbook wearing .ods"

    # The property a caller actually depends on: the reader named by the suffix
    # can open it. Left as the last assertion because it is the slowest, and the
    # membership checks above say *why* when it fails.
    pandas = pytest.importorskip("pandas")
    assert not pandas.read_excel(target, engine="odf").empty


def test_an_xlsx_export_is_still_a_workbook(session):
    """The other half of the pair — the fix must not swap the two engines."""
    pytest.importorskip("openpyxl")
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "out.xlsx"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple",
        path=str(target),
    )
    assert result["ok"] is True

    inside = set(zipfile.ZipFile(target).namelist())
    assert "[Content_Types].xml" in inside, sorted(inside)
    assert "mimetype" not in inside, "this is an OpenDocument package wearing .xlsx"


#: One distinguishing mark per writer, and a module that has to import for the
#: case to mean anything. The mark is deliberately a property only the right
#: format has: a magic number where the format carries one, an archive member
#: where it is a Zip, a structural token otherwise.
FORMAT_MARKS = {
    ".csv": (None, lambda p: p.read_bytes().splitlines()[0] == b"name,salary"),
    ".tsv": (None, lambda p: p.read_bytes().splitlines()[0] == b"name\tsalary"),
    ".txt": (None, lambda p: p.read_bytes().splitlines()[0] == b"name,salary"),
    ".json": (None, lambda p: p.read_bytes().lstrip()[:1] == b"["),
    ".jsonl": (None, lambda p: json.loads(p.read_bytes().splitlines()[0])),
    ".ndjson": (None, lambda p: json.loads(p.read_bytes().splitlines()[0])),
    ".xml": (None, lambda p: p.read_bytes().lstrip()[:5] == b"<?xml"),
    ".html": (None, lambda p: b"<table>" in p.read_bytes()),
    ".htm": (None, lambda p: b"<table>" in p.read_bytes()),
    ".md": (None, lambda p: b"|" in p.read_bytes()),
    ".yaml": ("yaml", lambda p: p.read_bytes().lstrip()[:1] in (b"-", b"[")),
    ".yml": ("yaml", lambda p: p.read_bytes().lstrip()[:1] in (b"-", b"[")),
    ".parquet": ("pyarrow", lambda p: p.read_bytes()[:4] == b"PAR1"),
    ".feather": ("pyarrow", lambda p: p.read_bytes()[:6] == b"ARROW1"),
    ".orc": ("pyarrow", lambda p: p.read_bytes()[:3] == b"ORC"),
    ".xlsx": (
        "openpyxl",
        lambda p: "[Content_Types].xml" in zipfile.ZipFile(p).namelist(),
    ),
    ".ods": ("odf", lambda p: "META-INF/manifest.xml" in zipfile.ZipFile(p).namelist()),
}


@pytest.mark.parametrize("suffix", sorted(FORMAT_MARKS))
def test_every_writer_produces_the_format_its_suffix_names(session, suffix):
    """§10.1 says the suffix is the whole of the format decision. This asserts it.

    It was asserted nowhere until `.ods` was caught writing XLSX. The magic
    numbers for Parquet, Feather and ORC had been checked once by hand and
    written into `CONSTRAINTS.md`, which records that they were right on the day
    — an invariant kept in prose is not kept at all. Every writer in `WRITERS`
    is covered here, so the next one to drift takes a test with it.

    Deliberately not a round trip. A round trip through a reader that sniffs its
    input passes on a file of the wrong format, which is precisely how `.ods`
    survived; these marks are properties only the correct format has.
    """
    module, mark = FORMAT_MARKS[suffix]
    if module:
        pytest.importorskip(module)

    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / f"out{suffix}"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple ORDER BY name",
        path=str(target),
    )

    assert result["ok"] is True, result
    assert target.exists(), f"{suffix} reported ok and wrote nothing"
    assert mark(target), f"{target.name} is not {suffix}: {target.read_bytes()[:64]!r}"


@pytest.mark.parametrize("suffix", [".xlsx", ".ods"])
def test_a_spreadsheet_refuses_more_rows_than_it_is_worth_writing(session, suffix):
    """Refused, not truncated — a short file that reports success is a wrong answer.

    Both spreadsheet writers build the whole document before a byte reaches the
    disk, so the cost of the export scales with the result: a million rows of
    eleven columns cost 12.9 GB as `.xlsx`, and `.ods` crossed 16 GB without
    producing a file. The refusal is settled before the frame is built, so it
    does not pay the memory it is declining to spend.
    """
    module = {"xlsx": "openpyxl", "ods": "odf"}[suffix.lstrip(".")]
    pytest.importorskip(module)
    over = export_module.SPREADSHEET_ROW_LIMIT + 1
    call(
        "attach",
        database=str(session / "simple.csv"),
        nickname="staff",
    )
    # A generated result rather than a fixture of 65,536 rows: `sqlite_master`
    # is not big enough, so count the rows out in SQL.
    target = session / f"big{suffix}"
    refused = call(
        "query",
        nickname="staff",
        sql=(
            "WITH RECURSIVE n(i) AS ("
            f"  SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < {over}"
            ") SELECT i FROM n"
        ),
        path=str(target),
    )

    assert refused["ok"] is False
    assert f"{export_module.SPREADSHEET_ROW_LIMIT:,}" in refused["error"]
    assert ".csv" in refused["error"], "the refusal should name a format that works"
    assert not target.exists(), "a refused export must not leave a partial file"


@pytest.mark.parametrize("suffix", [".xlsx", ".ods"])
def test_a_spreadsheet_writes_right_up_to_the_limit(session, suffix):
    """The boundary is inclusive, and the row that trips it is the one after."""
    module = {"xlsx": "openpyxl", "ods": "odf"}[suffix.lstrip(".")]
    pytest.importorskip(module)
    at = export_module.SPREADSHEET_ROW_LIMIT
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / f"exact{suffix}"

    result = call(
        "query",
        nickname="staff",
        sql=(
            "WITH RECURSIVE n(i) AS ("
            f"  SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < {at}"
            ") SELECT i FROM n"
        ),
        path=str(target),
    )

    assert result["ok"] is True, result
    assert result["rows_written"] == at
    assert target.exists()


def test_the_identity_table_covers_every_writer():
    """A table of cases is only a guarantee while it is complete."""
    assert set(FORMAT_MARKS) == set(export_module.WRITERS), (
        "WRITERS and the format-identity table have diverged: "
        f"{set(export_module.WRITERS) ^ set(FORMAT_MARKS)}"
    )


def test_an_export_to_a_format_this_server_cannot_write_is_refused(session):
    """It used to answer ok:true with CSV inside, whatever the name said."""
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "out.wibble"

    refused = call(
        "query", nickname="staff", sql="SELECT name FROM simple", path=str(target)
    )

    assert refused["ok"] is False
    assert ".wibble" in refused["error"]
    assert not target.exists()


# ---------------------------------------------------------------------------
# The lookup arc: land a table here, index it, join it
# ---------------------------------------------------------------------------


def test_a_second_file_lands_inside_the_open_database_and_joins(session):
    write_csv(session / "sales.csv", "sku,qty\na,3\nb,4\n")
    write_csv(session / "prices.csv", "sku,price\na,10\nb,20\n")
    call("attach", database=str(session / "sales.csv"), nickname="shop")

    added = call(
        "create", nickname="shop", type="table", source=str(session / "prices.csv")
    )

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


def test_the_whole_lookup_arc_runs_over_the_protocol(session):
    """Attach, land the second file, index the key, join, check what did not match.

    Every step a caller takes, in the order it takes them — including the two
    the server no longer does on their behalf. It neither infers the join key
    nor decides an index is wanted; both are stated, and the completeness check
    is an anti-join the caller writes.
    """
    write_csv(session / "sales.csv", "sku,qty\na,3\nb,4\nc,5\n")
    write_csv(session / "prices.csv", "sku,price\na,10\nz,99\n")
    call("attach", database=str(session / "sales.csv"), nickname="shop")
    call("create", nickname="shop", type="table", source=str(session / "prices.csv"))

    made = call(
        "create", nickname="shop", type="index", table="prices", columns=["sku"]
    )
    assert made["ok"] is True
    assert made["index"] == "ix_prices_sku"

    # And the caller can see it is there without having to remember making it.
    assert call("info", nickname="shop", table="prices")["indexes"] == [
        {
            "index": "ix_prices_sku",
            "table": "prices",
            "columns": ["sku"],
            "unique": False,
        }
    ]

    unpriced = call(
        "query",
        nickname="shop",
        sql="SELECT sku FROM sales WHERE sku NOT IN (SELECT sku FROM prices)",
    )
    assert unpriced["rows"] == [["b"], ["c"]]


def test_an_index_is_dropped_by_the_name_creation_gave_it(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    made = call(
        "create", nickname="staff", type="index", table="simple", columns=["name"]
    )

    dropped = call("drop", nickname="staff", type="index", name=made["index"])

    assert dropped["ok"] is True
    assert dropped["dropped"] == made["index"]
    assert dropped["table"] == "simple"
    assert call("info", nickname="staff", table="simple")["indexes"] == []


def test_create_and_drop_name_the_two_types_when_given_another(session):
    """A refusal that lists the alternatives, rather than a schema-level rejection."""
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    made = call("create", nickname="staff", type="view", table="simple")
    assert made["ok"] is False
    assert "'table' or 'index'" in made["error"]

    gone = call("drop", nickname="staff", type="view", name="simple")
    assert gone["ok"] is False
    assert "'table' or 'index'" in gone["error"]


def test_creating_a_table_without_a_source_says_which_argument_is_missing(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    answer = call("create", nickname="staff", type="table")

    assert answer["ok"] is False
    assert "source=" in answer["error"]


def test_creating_an_index_without_columns_says_which_arguments_are_missing(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    answer = call("create", nickname="staff", type="index", table="simple")

    assert answer["ok"] is False
    assert "table= and columns=" in answer["error"]


def test_adding_to_a_read_only_datasource_says_how_to_allow_it(session):
    _build_database(session / "hr.db")
    call("attach", database=str(session / "hr.db"), nickname="hr")

    answer = call(
        "create", nickname="hr", type="table", source=str(session / "simple.csv")
    )
    assert answer["ok"] is False
    assert "writable=true" in answer["error"]


def test_the_write_grant_is_honoured_over_the_protocol(session):
    _build_database(session / "hr.db")
    call("attach", database=str(session / "hr.db"), nickname="hr", writable=True)

    added = call(
        "create", nickname="hr", type="table", source=str(session / "simple.csv")
    )
    assert added["ok"] is True
    assert added["table"] == "simple"


def test_dropping_a_table_leaves_the_rest_answering(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    call(
        "create", nickname="staff", type="table", source=str(session / "mixed_tabs.tsv")
    )

    dropped = call("drop", nickname="staff", type="table", name="mixed_tabs")

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
    call("create", nickname="shop", type="table", source=str(session / "prices.csv"))
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


def test_the_next_call_after_the_budget_is_crossed_is_the_one_that_spills(session):
    """The deferral is wired into the tool surface, and fires exactly once.

    The transparency test above asserts that nothing *changes* when a database
    moves to disk — which is the right claim, and is also true when the database
    never moves at all. Disabling ``relieve_memory`` in ``_session`` left the
    whole suite green (323/323): every other spill test drives ``Registry``
    directly and so proves the mechanism works without ever proving that
    anything calls it. This asserts the wiring.

    Both halves matter. Spilling *during* the load that crossed the budget would
    be a defect too — the overshoot is deliberately tolerated once, because
    unloading mid-load is worse than briefly holding too much.
    """
    config_module.use(Config(roots=(session,), memory_budget_mb=1))
    rows = "\n".join(f"{index},label{index},{index * 2}" for index in range(60_000))
    write_csv(session / "big.csv", f"id,label,amount\n{rows}\n")

    call("attach", database=str(session / "big.csv"), nickname="big")

    registry = server_module._registry
    assert registry.workspace.resident_bytes("big") > 1024 * 1024, (
        "the load did not cross the 1 MB budget, so this test would pass "
        "whether or not anything spills"
    )
    assert registry.slot("big").spill_path is None, (
        "spilled during the load that crossed the budget; the overshoot is "
        "meant to be tolerated exactly once"
    )

    call("info")

    spilled = registry.slot("big").spill_path
    assert spilled is not None, "the next tool call did not relieve the pressure"
    assert spilled.exists()
    # On disk now, so residency is no longer a question that applies to it.
    assert registry.workspace.resident_bytes("big") is None


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
    assert "use create" in ddl["error"]

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


def test_attach_and_create_say_their_answer_needs_no_info_call(session):
    """Three of three skill-less agents called ``info`` straight after ``attach``.

    Every one of them reported the call as waste: the payload was identical to
    what they already held. The skill says not to; the docstrings — all a bare
    agent gets — did not.
    """
    documented = {tool.name: tool.description for tool in listed_tools()}
    for name in ("attach", "create"):
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


#: Verbs of the v2 surface, which this branch does not implement. Named
#: explicitly rather than derived: the rule is not "every identifier must be a
#: tool" — a document may legitimately mention ``avg()`` or ``inspect()`` — it is
#: that no shipped document may offer a caller a tool that is not there.
DEPARTED = (
    "connect_database",
    "disconnect_database",
    "execute_query",
    "execute_query_json",
    "list_databases",
    "describe_database",
    "describe_table",
    "list_tables",
    "find_table",
    "read_text_file",
    "next_chunk",
    "get_query_chunk",
    "analyze_regression",
    "analyze_time_series",
    "analyze_clusters",
)

DEPARTED_CALL = re.compile(r"\b(" + "|".join(DEPARTED) + r")\s*\(")


def test_no_shipped_document_offers_a_tool_that_is_gone():
    """The README was not the only one, and it was not the worst one.

    Sweeping every tracked document rather than the README alone found the issue
    templates, the Docker guide and the troubleshooting guide still instructing
    people to call ``connect_database`` and ``execute_query`` — a reader who
    follows a troubleshooting page into a tool that does not exist has been sent
    somewhere by us, which is worse than being left to ask.

    ``CHANGELOG.md`` is exempt on purpose: naming a departed tool is what a
    record of the past is for. ``non_factual/`` is exempt because it is
    quarantined by construction and says so on its own front page.
    """
    root = Path(__file__).parent.parent
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=root, capture_output=True, text=True
    ).stdout.split()

    offenders = {}
    for name in tracked:
        if name.startswith("non_factual/") or name == "CHANGELOG.md":
            continue
        text = (root / name).read_text()
        # Call-shaped, so a document may still *refer* to a departed tool where
        # that is the point — LEVEL0 records that `info` absorbed `list_tables`,
        # which is history rather than an offer. Followed by an open paren, it
        # is being held out as callable.
        named = sorted(set(re.findall(DEPARTED_CALL, text)))
        if named:
            offenders[name] = named

    assert not offenders, f"documents offering tools that do not exist: {offenders}"


def test_the_tool_descriptions_name_exactly_the_formats_that_exist():
    """A hardcoded list in a docstring is what the agent chooses from.

    Both registries are meant to grow, and a format that lands without its tool
    description learning about it is invisible to the only reader that matters.
    Pinning each list to its registry makes adding a format without saying so a
    test failure rather than a silent omission.
    """
    from localdata_mcp.export import WRITERS
    from localdata_mcp.loader import READERS

    described = {tool.name: tool.description for tool in listed_tools()}

    def listed(pattern: str, text: str) -> set[str]:
        found = re.search(pattern, text)
        assert found, f"no format list matching {pattern!r} in:\n{text}"
        return {item.strip() for item in found.group(1).split(",")}

    assert listed(r"A tabular file \(([^)]*)\)", described["attach"]) == set(READERS)
    assert listed(r"suffix chooses the format \(([^)]*)\)", described["query"]) == set(
        WRITERS
    )


def test_a_readers_note_reaches_the_caller_as_a_warning(session):
    """The channel is worthless if it stops at TableInfo.

    A JSON object with one array under it loads from a key the caller never
    named, and the loaded table looks exactly like any other — so this is the
    only place that fact can be seen.
    """
    attached = call("attach", database=str(session / "wrapped.json"), nickname="staff")

    assert attached["ok"] is True
    assert attached["loaded"][0]["rows"] == 2
    assert any("employees" in warning for warning in attached["warnings"])


def test_a_json_file_with_two_tables_is_refused_over_the_wire(session):
    refused = call("attach", database=str(session / "two_tables.json"), nickname="x")

    assert refused["ok"] is False
    assert "employees" in refused["error"] and "departments" in refused["error"]
    assert call("info")["slots_used"] == 0


def test_the_delimiter_reaches_attach_over_the_wire(session):
    proper = call(
        "attach",
        database=str(session / "semicolons.csv"),
        nickname="staff",
        delimiter=";",
    )

    assert proper["ok"] is True
    assert [c["name"] for c in proper["loaded"][0]["columns"]] == [
        "name",
        "role",
        "salary",
    ]
    assert "warnings" not in proper


def test_without_it_the_same_file_warns_and_names_the_parameter(session):
    """The warning is what makes the parameter discoverable at all."""
    loaded = call("attach", database=str(session / "semicolons.csv"), nickname="staff")

    assert loaded["ok"] is True
    assert len(loaded["loaded"][0]["columns"]) == 1
    assert any("delimiter" in warning for warning in loaded["warnings"])


def test_a_delimiter_is_refused_on_a_datasource_that_has_none(session):
    target = session / "shop.db"
    _build_database(target)

    refused = call("attach", database=str(target), nickname="shop", delimiter=";")

    assert refused["ok"] is False
    assert "delimiter" in refused["error"]


def test_the_delimiter_is_on_create_too(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    added = call(
        "create",
        nickname="staff",
        type="table",
        source=str(session / "semicolons.csv"),
        table="extra",
        delimiter=";",
    )

    assert added["ok"] is True
    assert [c["name"] for c in added["columns"]] == ["name", "role", "salary"]


def test_a_workbook_attaches_as_a_database_of_sheets(session):
    attached = call("attach", database=str(session / "workbook.xlsx"), nickname="book")

    assert attached["ok"] is True
    # `tables` is what the database holds, listed as the database lists it;
    # `loaded` is what was read, in the order the sheets appear.
    assert sorted(attached["tables"]) == ["departments", "staff"]
    assert [one["table"] for one in attached["loaded"]] == ["staff", "departments"]

    answer = call("query", nickname="book", sql="SELECT sum(salary) AS t FROM staff")
    assert answer["rows"][0][0] == 353000


def test_create_refuses_a_multi_table_source_and_points_at_attach(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    refused = call(
        "create",
        nickname="staff",
        type="table",
        source=str(session / "workbook.xlsx"),
        table="everything",
    )

    assert refused["ok"] is False
    assert "ttach the file" in refused["error"]


# ---------------------------------------------------------------------------
# DuckDB — a second file-based engine, reached the same way as the first
# ---------------------------------------------------------------------------


def _build_duckdb(path: Path) -> None:
    """The same builder as the SQLite one, given a different dialect.

    Which is the point of it going through Core: a fixture holding
    ``duckdb.connect`` would have been a second, parallel way to make a table.
    """
    foreign.build_database(
        path,
        "sales",
        [("region", Text), ("amount", Integer)],
        [("north", 100), ("south", 250)],
        dialect="duckdb",
    )


def test_a_duckdb_file_attaches_as_a_database(session):
    """Recognised by its header, not its name — it is called .db as often as not."""
    target = session / "warehouse.db"
    _build_duckdb(target)

    attached = call("attach", database=str(target), nickname="wh")

    assert attached["ok"] is True
    assert attached["kind"] == "database"
    assert attached["tables"] == ["sales"]

    answer = call("query", nickname="wh", sql="SELECT sum(amount) AS t FROM sales")
    assert answer["rows"][0][0] == 350


def test_duckdb_and_sqlite_are_told_apart_by_their_headers(session):
    """Both are commonly .db, so the extension decides nothing."""
    duck = session / "duck.db"
    lite = session / "lite.db"
    _build_duckdb(duck)
    _build_database(lite)

    assert call("attach", database=str(duck), nickname="d")["tables"] == ["sales"]
    assert call("attach", database=str(lite), nickname="s")["tables"] == ["departments"]


def test_an_attached_duckdb_file_is_read_only_unless_granted(session):
    target = session / "warehouse.db"
    _build_duckdb(target)
    call("attach", database=str(target), nickname="wh")

    refused = call(
        "create", nickname="wh", type="table", source=str(session / "simple.csv")
    )
    assert refused["ok"] is False

    written = call("query", nickname="wh", sql="CREATE TABLE nope (a INTEGER)")
    assert written["ok"] is False


def test_a_duckdb_url_is_reached_like_any_other_url(session):
    target = session / "warehouse.db"
    _build_duckdb(target)

    attached = call("attach", database=f"duckdb:///{target}", nickname="wh")

    assert attached["ok"] is True
    assert attached["kind"] == "engine"
    answer = call("query", nickname="wh", sql="SELECT count(*) AS n FROM sales")
    assert answer["rows"][0][0] == 2


def test_rows_can_be_copied_out_of_duckdb_into_a_slot_that_saves(session):
    """The generic route for a datasource this server reaches but does not hold."""
    target = session / "warehouse.db"
    _build_duckdb(target)
    call("attach", database=str(target), nickname="wh")
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    exported = call(
        "query", nickname="wh", sql="SELECT * FROM sales", path=str(session / "s.csv")
    )
    assert exported["ok"] is True

    added = call(
        "create",
        nickname="staff",
        type="table",
        source=str(session / "s.csv"),
        table="sales",
    )
    assert added["ok"] is True
    assert call("save", nickname="staff", path=str(session / "kept.db"))["ok"] is True


def test_a_local_file_url_is_still_subject_to_the_path_gate(session, tmp_path):
    """Spelling a file as a URL must not walk around containment.

    The network gate does not apply to a host-less URL, because nothing on the
    network is being reached — so the path gate has to, or `duckdb:///` would be
    a way to open any file on the machine.
    """
    outside = tmp_path / "elsewhere.db"
    _build_duckdb(outside)

    refused = call("attach", database=f"duckdb:///{outside}", nickname="wh")

    assert refused["ok"] is False
    assert "outside the allowed paths" in refused["error"]


def test_the_output_delimiter_reaches_query_over_the_wire(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "out.csv"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple ORDER BY name",
        path=str(target),
        delimiter=";",
    )

    assert result["ok"] is True
    assert target.read_text().splitlines()[0] == "name;salary"


def test_an_output_delimiter_is_ignored_where_it_has_no_meaning(session):
    """Ignored on the way out, refused on the way in — see export.export_rows."""
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    target = session / "out.parquet"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name FROM simple",
        path=str(target),
        delimiter=";",
    )

    assert result["ok"] is True
    assert result["rows_written"] == 5


# ---------------------------------------------------------------------------
# update: the third of create / update / drop
# ---------------------------------------------------------------------------


def test_a_sheet_can_be_renamed_after_it_lands(session):
    """The case a workbook creates: the file chose the names, and you may not want them."""
    call("attach", database=str(session / "workbook.xlsx"), nickname="book")

    renamed = call("update", nickname="book", type="table", name="staff", to="people")

    assert renamed["ok"] is True
    assert renamed["table"] == "people"
    assert sorted(call("info", nickname="book")["tables"]) == ["departments", "people"]

    answer = call("query", nickname="book", sql="SELECT sum(salary) AS t FROM people")
    assert answer["rows"][0][0] == 353000


def test_the_rows_survive_the_rename(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    before = call("query", nickname="staff", sql="SELECT * FROM simple ORDER BY name")

    call("update", nickname="staff", type="table", name="simple", to="people")

    after = call("query", nickname="staff", sql="SELECT * FROM people ORDER BY name")
    assert after["rows"] == before["rows"]
    assert after["columns"] == before["columns"]


def test_renaming_onto_a_name_already_taken_is_refused(session):
    """Silently replacing the other table would lose it entirely."""
    call("attach", database=str(session / "workbook.xlsx"), nickname="book")

    refused = call(
        "update", nickname="book", type="table", name="staff", to="departments"
    )

    assert refused["ok"] is False
    assert "departments" in refused["error"]
    assert sorted(call("info", nickname="book")["tables"]) == ["departments", "staff"]


def test_renaming_a_table_that_is_not_there_names_what_is(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    refused = call("update", nickname="staff", type="table", name="nope", to="x")

    assert refused["ok"] is False
    assert "simple" in refused["error"]


def test_a_read_only_datasource_will_not_be_renamed(session):
    target = session / "shop.db"
    _build_database(target)
    call("attach", database=str(target), nickname="shop")

    refused = call(
        "update", nickname="shop", type="table", name="departments", to="teams"
    )

    assert refused["ok"] is False
    assert "writable" in refused["error"]


def test_a_rename_target_must_be_a_legal_identifier(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    refused = call(
        "update", nickname="staff", type="table", name="simple", to="not a name"
    )

    assert refused["ok"] is False


def test_update_names_what_it_can_update(session):
    call("attach", database=str(session / "simple.csv"), nickname="staff")

    refused = call("update", nickname="staff", type="index", name="i", to="j")

    assert refused["ok"] is False
    assert "table" in refused["error"]


def test_a_renamed_table_is_saved_under_its_new_name(session):
    """The rename has to reach the database, not only our bookkeeping."""
    call("attach", database=str(session / "simple.csv"), nickname="staff")
    call("update", nickname="staff", type="table", name="simple", to="people")
    kept = session / "kept.db"

    call("save", nickname="staff", path=str(kept))
    call("detach", nickname="staff")
    reattached = call("attach", database=str(kept), nickname="again")

    assert reattached["tables"] == ["people"]


def test_the_surface_is_eight_verbs_now(session):
    assert {tool.name for tool in listed_tools()} == TOOLS
    assert "update" in TOOLS
