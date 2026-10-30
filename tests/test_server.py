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
from localdata_mcp.formats import DELIMITED
from localdata_mcp.names import sanitize
from localdata_mcp.config import Config

ASSETS = Path(__file__).parent / "assets"

#: The whole surface. Named here so a tool added or removed without thinking
#: about the shape of the surface fails a test rather than passing quietly.
TOOLS = {
    "attach",
    "detach",
    "directory",
    "query",
    "create",
    "update",
    "drop",
    "save",
    "stats",
}


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


def separator_for(destination) -> dict[str, str]:
    """``delimiter=`` for a destination that is character-separated text.

    Empty for every other format, because a delimiter is inert there rather
    than required. The suffix no longer supplies a separator to the server —
    it has to be declared — so the tests state the one their fixtures use.
    """
    suffix = Path(str(destination)).suffix.lower()
    if suffix not in DELIMITED:
        return {}
    return {"delimiter": "\t" if suffix == ".tsv" else ","}


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


def test_the_surface_is_nine_verbs_each_with_a_description():
    tools = listed_tools()
    assert {tool.name for tool in tools} == TOOLS

    # A tool with no description is invisible to an agent choosing between them.
    for tool in tools:
        assert tool.description and len(tool.description) > 20, tool.name

    by_name = {tool.name: tool for tool in tools}
    for name in TOOLS:
        assert by_name[name].inputSchema["properties"], name


def test_the_shipped_skill_grants_exactly_the_tools_that_exist():
    """The skill travels with the server, so its tool list is part of the surface.

    It had drifted: the skill still granted ``add_table`` and ``drop_table``,
    two verbs replaced by ``create`` and ``drop``, and granted nothing for
    ``update`` at all — so the tools an agent was allowed to call and the tools
    there are to call were three names apart, and nothing said so. An invariant
    kept in prose is not kept.
    """
    skill = Path(__file__).parent.parent / "skills" / "local-data" / "SKILL.md"
    granted = re.search(r"^allowed-tools:(.*)$", skill.read_text(), re.MULTILINE)
    assert granted is not None, "the skill declares no allowed-tools"

    names = granted.group(1).split()
    prefix = "mcp__localdata__"
    assert all(name.startswith(prefix) for name in names), names
    assert {name[len(prefix) :] for name in names} == TOOLS


def test_the_nickname_is_required_everywhere_it_routes():
    """It names the engine to execute against, so it cannot be optional."""
    by_name = {tool.name: tool for tool in listed_tools()}
    for name in ("detach", "query", "create", "drop", "save"):
        assert "nickname" in by_name[name].inputSchema["required"], name

    # The two exceptions, and both are deliberate: attach *derives* a nickname,
    # and directory with none reports the whole session.
    assert "nickname" not in by_name["attach"].inputSchema.get("required", [])
    assert "nickname" not in by_name["directory"].inputSchema.get("required", [])


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
    attached = call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    assert attached["ok"] is True
    assert attached["kind"] == "file"
    assert attached["nickname"] == "staff"
    assert attached["tables"] == ["simple"]
    assert attached["writable"] is True
    assert attached["evicted"] is None
    assert attached["collided_with"] is None


def test_a_nickname_is_derived_when_none_is_given(session):
    attached = call("attach", database=str(session / "simple.csv"), delimiter=",")
    assert attached["nickname"] == "simple"
    assert attached["tables"] == ["simple"]


def test_attach_then_query_over_the_protocol(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    answer = call("query", nickname="staff", sql="SELECT sum(salary) AS t FROM simple")
    assert answer["ok"] is True
    assert answer["columns"] == ["t"]
    assert answer["rows"][0][0] == 345000


def test_the_same_source_twice_is_refused_and_names_where_it_lives(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    again = call(
        "attach", database=str(session / "simple.csv"), nickname="other", delimiter=","
    )

    assert again["ok"] is False
    assert "'staff'" in again["error"]


def test_a_colliding_nickname_is_disambiguated_and_both_slots_survive(session):
    """Pointing a handle at a second datasource must not take the first away.

    The caller is told which name it actually got, and what it collided with,
    because a caller that assumes it got the name it asked for addresses the
    wrong database.
    """
    call("attach", database=str(session / "simple.csv"), nickname="slot", delimiter=",")
    second = call(
        "attach",
        database=str(session / "mixed_tabs.tsv"),
        nickname="slot",
        delimiter="\t",
    )

    assert second["ok"] is True
    assert second["nickname"] == "slot_2"
    assert second["tables"] == ["mixed_tabs"]
    assert second["collided_with"] == {
        "nickname": "slot",
        "source": str(session / "simple.csv"),
    }
    assert call("directory")["slots_used"] == 2


# ---------------------------------------------------------------------------
# directory: one verb, three altitudes
# ---------------------------------------------------------------------------


def test_directory_with_nothing_reports_every_datasource_and_the_posture(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    listing = call("directory")

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


def test_directory_with_a_nickname_reports_that_datasources_tables(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    call(
        "create",
        nickname="staff",
        type="table",
        source=str(session / "mixed_tabs.tsv"),
        delimiter="\t",
    )

    detail = call("directory", nickname="staff")

    assert detail["ok"] is True
    assert detail["nickname"] == "staff"
    assert sorted(detail["tables"]) == ["mixed_tabs", "simple"]
    assert {entry["table"]: entry["rows"] for entry in detail["contents"]} == {
        "simple": 5,
        "mixed_tabs": 5,
    }


def test_directory_with_a_table_describes_its_columns_and_types(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    described = call("directory", nickname="staff", table="simple")

    assert described["ok"] is True
    assert described["table"] == "simple"
    assert described["rows"] == 5
    types = {column["name"]: column["type"] for column in described["columns"]}
    assert types["salary"] == "INTEGER"
    assert types["department"] == "TEXT"


def test_directory_reports_an_unknown_nickname_as_an_answer(session):
    answer = call("directory", nickname="nothing")
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    answer = call("query", nickname="staff", sql="SELECT * FROM simple")
    assert answer["row_count"] == 5
    assert "truncated" not in answer

    asked = call("query", nickname="staff", sql="SELECT * FROM simple LIMIT 2")
    assert asked["row_count"] == 2

    schema = {tool.name: tool for tool in listed_tools()}["query"].inputSchema
    assert "limit" not in schema["properties"]


def test_a_path_turns_the_same_query_into_an_export(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    target = session / "out.csv"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple ORDER BY name",
        path=str(target),
        **separator_for(target),
    )
    assert result["ok"] is True
    assert result["rows_written"] == 5
    assert target.read_text().splitlines()[0] == "name,salary"


def test_the_export_writes_every_row_the_statement_selected(session):
    """A path is the answer to a result too large to return, so it truncates never."""
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    target = session / "full.csv"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT * FROM simple",
        path=str(target),
        **separator_for(target),
    )
    assert result["rows_written"] == 5
    assert len(target.read_text().splitlines()) == 6  # header included


def test_a_writers_refusal_is_not_reported_as_a_slot_problem(session):
    """An export failure must answer the question the caller actually asked.

    The rows now reach the writer straight off the open cursor, so the writer
    raises from *inside* the block that runs the statement. The layer holding
    that block explains a failed statement by looking for another slot's
    nickname in the SQL — sound for a statement, and here it would read the
    column ``salary`` as the ``salary`` slot and answer a question about a file
    suffix with a sentence about joining across databases.
    """
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    call("attach", database=str(session / "records.json"), nickname="salary")

    refused = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple",
        path=str(session / "out.bogus"),
    )
    assert refused["ok"] is False
    assert "No writer for '.bogus'" in refused["error"]
    assert "separate databases" not in refused["error"]


def test_the_export_refuses_to_clobber_then_takes_the_users_answer(session):
    """The refusal has to read as a question, or the agent just retries.

    It relayed a path the user chose rather than choosing one, so replacing what
    is there is the user's call — and force is how that answer comes back.
    """
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    target = session / "out.csv"
    target.write_text("existing content\n")

    refused = call(
        "query",
        nickname="staff",
        sql="SELECT name FROM simple",
        path=str(target),
        **separator_for(target),
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
        **separator_for(target),
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    target = session / "out.tsv"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple ORDER BY name",
        path=str(target),
        **separator_for(target),
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    target = session / "out.ods"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple",
        path=str(target),
        **separator_for(target),
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    target = session / "out.xlsx"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple",
        path=str(target),
        **separator_for(target),
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

    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    target = session / f"out{suffix}"

    result = call(
        "query",
        nickname="staff",
        sql="SELECT name, salary FROM simple ORDER BY name",
        path=str(target),
        **separator_for(target),
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
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
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
        **separator_for(target),
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
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
        **separator_for(target),
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
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
    call("attach", database=str(session / "sales.csv"), nickname="shop", delimiter=",")

    added = call(
        "create",
        nickname="shop",
        type="table",
        source=str(session / "prices.csv"),
        delimiter=",",
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
    call("attach", database=str(session / "sales.csv"), nickname="shop", delimiter=",")
    call(
        "create",
        nickname="shop",
        type="table",
        source=str(session / "prices.csv"),
        delimiter=",",
    )

    made = call(
        "create", nickname="shop", type="index", table="prices", columns=["sku"]
    )
    assert made["ok"] is True
    assert made["index"] == "ix_prices_sku"

    # And the caller can see it is there without having to remember making it.
    assert call("directory", nickname="shop", table="prices")["indexes"] == [
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    made = call(
        "create", nickname="staff", type="index", table="simple", columns=["name"]
    )

    dropped = call("drop", nickname="staff", type="index", name=made["index"])

    assert dropped["ok"] is True
    assert dropped["dropped"] == made["index"]
    assert dropped["table"] == "simple"
    assert call("directory", nickname="staff", table="simple")["indexes"] == []


def test_create_and_drop_name_the_two_types_when_given_another(session):
    """A refusal that lists the alternatives, rather than a schema-level rejection."""
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    made = call("create", nickname="staff", type="view", table="simple")
    assert made["ok"] is False
    assert "'table' or 'index'" in made["error"]

    gone = call("drop", nickname="staff", type="view", name="simple")
    assert gone["ok"] is False
    assert "'table' or 'index'" in gone["error"]


def test_creating_a_table_without_a_source_says_which_argument_is_missing(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    answer = call("create", nickname="staff", type="table")

    assert answer["ok"] is False
    assert "source=" in answer["error"]


def test_creating_an_index_without_columns_says_which_arguments_are_missing(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    answer = call("create", nickname="staff", type="index", table="simple")

    assert answer["ok"] is False
    assert "table= and columns=" in answer["error"]


def test_adding_to_a_read_only_datasource_says_how_to_allow_it(session):
    _build_database(session / "hr.db")
    call("attach", database=str(session / "hr.db"), nickname="hr")

    answer = call(
        "create",
        nickname="hr",
        type="table",
        source=str(session / "simple.csv"),
        delimiter=",",
    )
    assert answer["ok"] is False
    assert "writable=true" in answer["error"]


def test_the_write_grant_is_honoured_over_the_protocol(session):
    _build_database(session / "hr.db")
    call("attach", database=str(session / "hr.db"), nickname="hr", writable=True)

    added = call(
        "create",
        nickname="hr",
        type="table",
        source=str(session / "simple.csv"),
        delimiter=",",
    )
    assert added["ok"] is True
    assert added["table"] == "simple"


def test_dropping_a_table_leaves_the_rest_answering(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    call(
        "create",
        nickname="staff",
        type="table",
        source=str(session / "mixed_tabs.tsv"),
        delimiter="\t",
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    closed = call("detach", nickname="staff")

    assert closed["ok"] is True
    assert closed["tables"] == ["simple"]
    assert closed["slots_used"] == 0
    assert call("directory")["datasources"] == []


def test_detach_reports_the_tables_the_slot_holds_when_it_closes(session):
    """Issue #90: not the ones it arrived with.

    ``detach``'s own docstring warns that tables added to the datasource are
    gone once it closes — so the tables it names are read at the moment it
    closes, the same way ``directory`` reads them. A table added with
    ``create``, one renamed with ``update`` and one removed with ``drop`` all
    have to show, or the answer is least accurate exactly where the caller is
    told to look.
    """
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    call(
        "create",
        nickname="staff",
        type="table",
        source=str(session / "mixed_tabs.tsv"),
        table="extra",
        delimiter="\t",
    )
    call(
        "create",
        nickname="staff",
        type="table",
        source=str(session / "semicolons.csv"),
        table="draft",
        delimiter=";",
    )
    call("update", nickname="staff", type="table", name="draft", to="final")
    call("drop", nickname="staff", type="table", name="simple")
    held = call("directory", nickname="staff")["tables"]
    assert sorted(held) == ["extra", "final"]

    closed = call("detach", nickname="staff")

    assert closed["ok"] is True
    assert closed["tables"] == held


def test_detach_names_a_table_added_to_an_outside_database(session):
    """The same for a database file attached writable: what it holds at the end."""
    target = session / "company.db"
    _build_database(target)
    call("attach", database=str(target), nickname="co", writable=True)
    call(
        "create",
        nickname="co",
        type="table",
        source=str(session / "simple.csv"),
        table="staff",
        delimiter=",",
    )

    closed = call("detach", nickname="co")

    assert closed["ok"] is True
    assert sorted(closed["tables"]) == ["departments", "staff"]


def test_saving_then_attaching_again_brings_the_whole_session_back(session):
    write_csv(session / "sales.csv", "sku,qty\na,3\nb,4\n")
    write_csv(session / "prices.csv", "sku,price\na,10\nb,20\n")
    call("attach", database=str(session / "sales.csv"), nickname="shop", delimiter=",")
    call(
        "create",
        nickname="shop",
        type="table",
        source=str(session / "prices.csv"),
        delimiter=",",
    )
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
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

    call("attach", database=str(session / "big.csv"), nickname="big", delimiter=",")
    before = call("query", nickname="big", sql="SELECT count(*), sum(amount) FROM big")
    described_before = call("directory", nickname="big")

    # The next call is the one that pays for the overshoot, whatever it is.
    call("directory")

    after = call("query", nickname="big", sql="SELECT count(*), sum(amount) FROM big")
    assert after["rows"] == before["rows"]
    assert after["rows"][0][0] == 60_000

    # Described exactly as before: same kind, same source, same rights, same
    # tables, same row counts. A caller has no way to tell the move happened.
    assert call("directory", nickname="big") == described_before
    assert "spill" not in json.dumps(call("directory"))


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

    call("attach", database=str(session / "big.csv"), nickname="big", delimiter=",")

    registry = server_module._registry
    assert registry.workspace.resident_bytes("big") > 1024 * 1024, (
        "the load did not cross the 1 MB budget, so this test would pass "
        "whether or not anything spills"
    )
    assert registry.slot("big").spill_path is None, (
        "spilled during the load that crossed the budget; the overshoot is "
        "meant to be tolerated exactly once"
    )

    call("directory")

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
    call("attach", database=str(session / "simple.csv"), nickname="a", delimiter=",")
    call(
        "attach", database=str(session / "mixed_tabs.tsv"), nickname="b", delimiter="\t"
    )

    third = call(
        "attach", database=str(session / "no_header.csv"), nickname="c", delimiter=","
    )

    assert third["ok"] is True
    assert third["evicted"]["nickname"] == "a"
    assert third["evicted"]["tables"] == ["simple"]
    assert third["evicted"]["source"] == str(session / "simple.csv")


def test_querying_an_evicted_datasource_says_it_was_evicted(session):
    config_module.use(Config(roots=(session,), slots=1))
    call("attach", database=str(session / "simple.csv"), nickname="a", delimiter=",")
    call(
        "attach", database=str(session / "mixed_tabs.tsv"), nickname="b", delimiter="\t"
    )

    answer = call("query", nickname="a", sql="SELECT * FROM simple")
    assert answer["ok"] is False
    assert "evicted" in answer["error"]
    # Enough to rebuild it without guessing.
    assert str(session / "simple.csv") in answer["error"]


# ---------------------------------------------------------------------------
# Errors reach the caller as answers, not as exceptions
# ---------------------------------------------------------------------------


def test_a_bad_path_is_answered_not_raised(session, tmp_path):
    answer = call(
        "attach", database=str(tmp_path / "elsewhere.csv"), nickname="x", delimiter=","
    )
    assert answer["ok"] is False
    assert "outside the allowed paths" in answer["error"]


def test_a_missing_file_is_answered(session):
    answer = call(
        "attach", database=str(session / "absent.csv"), nickname="x", delimiter=","
    )
    assert answer["ok"] is False
    assert "No such file" in answer["error"]


def test_an_unusable_nickname_is_answered(session):
    answer = call(
        "attach",
        database=str(session / "simple.csv"),
        nickname="my-data",
        delimiter=",",
    )
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
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
    attached = call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

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
        "attach",
        database=str(session / "messy_mixed_types.csv"),
        nickname="messy",
        delimiter=",",
    )
    assert attached["ok"] is True
    assert any("coerce text to 0" in warning for warning in attached["warnings"])
    assert any("In messy, table messy_mixed_types" in w for w in attached["warnings"])


def test_the_mixed_column_detail_is_available_from_directory(session):
    call(
        "attach",
        database=str(session / "messy_mixed_types.csv"),
        nickname="messy",
        delimiter=",",
    )
    described = call("directory", nickname="messy", table="messy_mixed_types")
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
    attached = call("attach", database=str(session / "sentinels.csv"), delimiter=",")

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
    warning = call("directory", nickname="ext", table="t")["warnings"][0]
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

    attached = call("attach", database=str(session / "simple.csv"), delimiter=",")
    refused = call(
        "query",
        nickname=attached["nickname"],
        sql="INSERT INTO simple (id) VALUES (1)",
    )
    assert refused["ok"] is False
    assert "does not write" in refused["error"]


def test_attach_and_create_say_their_answer_needs_no_directory_call(session):
    """Three of three skill-less agents called ``directory`` straight after ``attach``.

    Every one of them reported the call as waste: the payload was identical to
    what they already held. The skill says not to; the docstrings — all a bare
    agent gets — did not.
    """
    documented = {tool.name: tool.description for tool in listed_tools()}
    for name in ("attach", "create"):
        assert (
            "directory" in documented[name]
        ), f"{name} never mentions the redundant call"


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


def test_directory_describes_a_table_inside_an_attached_database(session):
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

    detail = call("directory", nickname="kept", table="notes")
    assert detail["ok"] is True
    assert [column["name"] for column in detail["columns"]] == ["id", "note"]
    assert detail["rows"] == 2

    # And at the altitude above it, which describes every table in the slot.
    slot = call("directory", nickname="kept")
    assert slot["ok"] is True
    assert slot["contents"] == [{"table": "notes", "rows": 2}]


def test_a_saved_database_can_be_described_when_it_comes_back(session):
    """Arc 2's return journey: keep it this week, open it next week.

    ``save`` then ``attach`` is the sequence the whole "keep this" promise rests
    on, and the first thing anyone does with a database they have just reopened
    is ask what is in it.
    """
    attached = call("attach", database=str(session / "simple.csv"), delimiter=",")
    call("save", nickname=attached["nickname"], path=str(session / "kept.db"))
    call("detach", nickname=attached["nickname"])

    reopened = call("attach", database=str(session / "kept.db"), nickname="lastweek")
    described = call("directory", nickname="lastweek", table=reopened["tables"][0])
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
        # that is the point — LEVEL0 records that `directory` absorbed `list_tables`,
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
    assert call("directory")["slots_used"] == 0


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


def test_a_wrongly_declared_separator_warns_and_names_the_parameter(session):
    """The warning is what turns a wrong declaration into something actionable.

    It used to catch a caller who had never heard of the parameter, because the
    separator was assumed. The separator is required now, so what it catches is
    a caller who declared the wrong one — a narrower case and a better-founded
    note, since the character actually used is known rather than guessed.
    """
    loaded = call(
        "attach",
        database=str(session / "semicolons.csv"),
        nickname="staff",
        delimiter=",",
    )

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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

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
        "create",
        nickname="wh",
        type="table",
        source=str(session / "simple.csv"),
        delimiter=",",
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    exported = call(
        "query",
        nickname="wh",
        sql="SELECT * FROM sales",
        path=str(session / "s.csv"),
        delimiter=",",
    )
    assert exported["ok"] is True

    added = call(
        "create",
        nickname="staff",
        type="table",
        source=str(session / "s.csv"),
        table="sales",
        delimiter=",",
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


# DuckDB reaches files by itself — `read_csv`, `read_text`, `COPY ... TO` — and
# never asks `paths.py`. Issue #105: on a file attached read-only, `query` read
# /etc/hosts and wrote a CSV to /tmp. These drive that through the protocol.

#: A line no datasource here contains, so finding it in an answer means the
#: statement read the file it was planted in.
_PLANTED = "planted-outside-the-allowed-paths"


def _attach_duckdb(session, **extra) -> None:
    target = session / "warehouse.db"
    _build_duckdb(target)
    assert call("attach", database=str(target), nickname="wh", **extra)["ok"]


def test_a_duckdb_query_cannot_read_a_file_through_read_text(session, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text(_PLANTED)
    _attach_duckdb(session)

    answer = call(
        "query", nickname="wh", sql=f"SELECT content FROM read_text('{outside}')"
    )

    assert answer["ok"] is False
    assert _PLANTED not in json.dumps(answer)


def test_a_duckdb_query_cannot_read_a_csv_outside_the_allowed_paths(session, tmp_path):
    outside = write_csv(tmp_path / "outside.csv", f"word\n{_PLANTED}\n")
    _attach_duckdb(session)

    answer = call("query", nickname="wh", sql=f"SELECT * FROM read_csv('{outside}')")

    assert answer["ok"] is False
    assert _PLANTED not in json.dumps(answer)


@pytest.mark.parametrize("where", ["outside", "inside"])
def test_a_duckdb_query_cannot_copy_rows_into_a_file(session, tmp_path, where):
    """``query`` only reads — inside the allowed paths as much as outside them.

    The inside case is the read-only contract rather than the path boundary:
    writing a file is ``query(..., path=...)``'s job, done by this server, and
    a statement writing one itself is a write that verb never granted.
    """
    destination = (tmp_path if where == "outside" else session) / "copied.csv"
    _attach_duckdb(session)

    answer = call(
        "query", nickname="wh", sql=f"COPY (SELECT * FROM sales) TO '{destination}'"
    )

    assert answer["ok"] is False
    assert not destination.exists()


def test_a_duckdb_query_cannot_switch_file_access_back_on(session, tmp_path):
    """Try to undo the lock, then read in a separate call. The read is refused.

    The attempt is its own call, and its answer is not what is asserted: a
    ``SET`` is refused as "not a read" whether or not it worked, and DuckDB
    rejects ``SET enable_external_access = true`` even where access was never
    off. What proves anything is the read that follows on the same slot.
    """
    outside = tmp_path / "secret.txt"
    outside.write_text(_PLANTED)
    _attach_duckdb(session)

    call(
        "query",
        nickname="wh",
        sql="SET lock_configuration = false; SET GLOBAL enable_external_access = true",
    )
    answer = call(
        "query", nickname="wh", sql=f"SELECT content FROM read_text('{outside}')"
    )

    assert answer["ok"] is False
    assert _PLANTED not in json.dumps(answer)


def test_a_duckdb_query_cannot_unlock_the_configuration(session):
    """The lock is what keeps every *other* setting where the server put it.

    ``enable_external_access`` refuses to be switched back on by itself, so a
    test on that setting alone would stay green with the lock removed. This
    asks the lock directly: either the ``SET`` is refused, or it did nothing.
    """
    _attach_duckdb(session)

    answer = call(
        "query",
        nickname="wh",
        sql=(
            "SET lock_configuration = false; "
            "SELECT current_setting('lock_configuration') AS locked"
        ),
    )

    assert answer["ok"] is False or answer["rows"] == [[True]]


def test_a_duckdb_url_cannot_ask_for_file_access_back(session, tmp_path):
    """A URL's own query parameters must not outrank the server's posture.

    The driver merges a URL's parameters over the connection arguments, so a
    setting passed only as a connection argument loses to
    ``?enable_external_access=true``.
    """
    outside = tmp_path / "secret.txt"
    outside.write_text(_PLANTED)
    target = session / "warehouse.db"
    _build_duckdb(target)
    attached = call(
        "attach",
        database=f"duckdb:///{target}?enable_external_access=true",
        nickname="wh",
    )
    assert attached["ok"] is True

    answer = call(
        "query", nickname="wh", sql=f"SELECT content FROM read_text('{outside}')"
    )

    assert answer["ok"] is False
    assert _PLANTED not in json.dumps(answer)


def test_a_duckdb_url_locked_with_file_access_on_is_refused(session):
    """A URL can lock the configuration before the server seals it.

    Then file access can no longer be switched off, and the datasource is
    refused at attach rather than handed out open.
    """
    target = session / "warehouse.db"
    _build_duckdb(target)

    refused = call(
        "attach",
        database=f"duckdb:///{target}?lock_configuration=true",
        nickname="wh",
    )

    assert refused["ok"] is False
    assert "lock_configuration" in refused["error"]


def test_a_duckdb_file_still_answers_with_file_access_switched_off(session):
    """The lock reaches only what is *not* the datasource: its own file still reads."""
    _attach_duckdb(session)

    answer = call(
        "query", nickname="wh", sql="SELECT region FROM sales ORDER BY region"
    )

    assert answer["ok"] is True
    assert answer["rows"] == [["north"], ["south"]]


def test_the_output_delimiter_reaches_query_over_the_wire(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
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
    assert sorted(call("directory", nickname="book")["tables"]) == [
        "departments",
        "people",
    ]

    answer = call("query", nickname="book", sql="SELECT sum(salary) AS t FROM people")
    assert answer["rows"][0][0] == 353000


def test_the_rows_survive_the_rename(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
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
    assert sorted(call("directory", nickname="book")["tables"]) == [
        "departments",
        "staff",
    ]


def test_renaming_a_table_that_is_not_there_names_what_is(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

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
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    refused = call(
        "update", nickname="staff", type="table", name="simple", to="not a name"
    )

    assert refused["ok"] is False


def test_update_names_what_it_can_update(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    refused = call("update", nickname="staff", type="index", name="i", to="j")

    assert refused["ok"] is False
    assert "table" in refused["error"]


def test_a_renamed_table_is_saved_under_its_new_name(session):
    """The rename has to reach the database, not only our bookkeeping."""
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    call("update", nickname="staff", type="table", name="simple", to="people")
    kept = session / "kept.db"

    call("save", nickname="staff", path=str(kept))
    call("detach", nickname="staff")
    reattached = call("attach", database=str(kept), nickname="again")

    assert reattached["tables"] == ["people"]


def test_the_surface_is_nine_verbs_now(session):
    assert {tool.name for tool in listed_tools()} == TOOLS
    assert "update" in TOOLS


# ---------------------------------------------------------------------------
# The answer's own shape
# ---------------------------------------------------------------------------


def test_directory_refuses_a_table_with_no_nickname_rather_than_dropping_it(session):
    """The third form was asked for; answering the first would look understood."""
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    answer = call("directory", table="simple")

    assert answer["ok"] is False
    assert "simple" in answer["error"]
    assert "nickname" in answer["error"]
    assert "datasources" not in answer


def test_both_of_creates_answers_name_the_datasource(session):
    """One verb, two branches, one set of keys at the front."""
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )
    made_table = call(
        "create",
        nickname="staff",
        type="table",
        source=str(session / "mixed_tabs.tsv"),
        delimiter="\t",
    )
    made_index = call(
        "create", nickname="staff", type="index", table="simple", columns=["salary"]
    )

    assert made_table["ok"] is True
    assert made_table["nickname"] == "staff"
    assert made_index["nickname"] == "staff"


def test_update_reports_the_name_the_database_held_not_the_one_passed(session):
    """`renamed` has to match the listing this same payload replaces."""
    mixed = session / "mixed.sqlite"
    connection = sqlite3.connect(mixed)
    connection.execute('CREATE TABLE "MyTable" (id INTEGER)')
    connection.commit()
    connection.close()
    call("attach", database=str(mixed), nickname="held", writable=True)

    renamed = call("update", nickname="held", type="table", name="mytable", to="kept")

    assert renamed["ok"] is True
    assert renamed["renamed"] == "MyTable"
    assert renamed["table"] == "kept"
    assert renamed["tables"] == ["kept"]


# ---------------------------------------------------------------------------
# Sheet names that clean up to the same table name (#91)
# ---------------------------------------------------------------------------
#
# A table name is the sheet's name made safe for SQL, and two sheet names that
# differ only in punctuation or case make the same one. Before #91 the second
# sheet overwrote the first and `attach` reported two loads of the survivor.

_WORKBOOK_ENGINES = {".xlsx": "openpyxl", ".ods": "odf"}


def _workbook(path: Path, sheets: dict[str, dict[str, list]]) -> Path:
    """Write a workbook holding exactly these sheets, and check it does.

    The check is the point: a writer that renamed a sheet on the way out (Excel
    refuses two sheet names differing only in case, and openpyxl quietly renames
    the second) would leave a fixture with no collision in it, and every test
    below would pass against a file that never exercised the rule.
    """
    import pandas as pd

    pytest.importorskip(_WORKBOOK_ENGINES[path.suffix])
    with pd.ExcelWriter(path, engine=_WORKBOOK_ENGINES[path.suffix]) as writer:
        for name, columns in sheets.items():
            pd.DataFrame(columns).to_excel(writer, sheet_name=name, index=False)
    assert list(pd.read_excel(path, sheet_name=None)) == list(sheets)
    return path


def _rows(nickname: str, table: str) -> tuple[list, list]:
    answer = call("query", nickname=nickname, sql=f"SELECT * FROM {table}")
    assert answer["ok"] is True, answer
    return answer["columns"], answer["rows"]


@pytest.mark.parametrize("suffix", [".xlsx", ".ods"])
def test_two_sheets_that_clean_up_alike_each_land_as_a_table(session, suffix):
    book = _workbook(
        session / f"collide{suffix}",
        {"Q2 Prices": {"a": [1]}, "Q2-Prices": {"b": [2]}},
    )
    # The fixture has to produce the collision it is named for.
    assert sanitize("Q2 Prices", "t") == "q2_prices"
    assert sanitize("Q2-Prices", "t") == "q2_prices"

    attached = call("attach", database=str(book), nickname="book")

    assert attached["ok"] is True, attached
    assert sorted(attached["tables"]) == ["q2_prices", "q2_prices_1"]
    # Sheet order decides who keeps the bare name, and each entry describes its
    # own sheet rather than the survivor's.
    assert [
        (one["sheet"], one["table"], [c["name"] for c in one["columns"]])
        for one in attached["loaded"]
    ] == [("Q2 Prices", "q2_prices", ["a"]), ("Q2-Prices", "q2_prices_1", ["b"])]
    assert _rows("book", "q2_prices") == (["a"], [[1]])
    assert _rows("book", "q2_prices_1") == (["b"], [[2]])

    (warning,) = attached["warnings"]
    for named in ("'Q2 Prices'", "'Q2-Prices'", "q2_prices", "q2_prices_1"):
        assert named in warning, warning

    detached = call("detach", nickname="book")
    assert detached["ok"] is True
    assert sorted(detached["tables"]) == ["q2_prices", "q2_prices_1"]


@pytest.mark.parametrize("suffix", [".xlsx", ".ods"])
def test_three_sheets_that_clean_up_alike_each_land_as_a_table(session, suffix):
    sheets = {"Q2 Prices": {"a": [1]}, "Q2-Prices": {"b": [2]}, "q2.prices": {"c": [3]}}
    book = _workbook(session / f"three{suffix}", sheets)
    assert {sanitize(name, "t") for name in sheets} == {"q2_prices"}

    attached = call("attach", database=str(book), nickname="book")

    assert attached["ok"] is True, attached
    assert [(one["sheet"], one["table"]) for one in attached["loaded"]] == [
        ("Q2 Prices", "q2_prices"),
        ("Q2-Prices", "q2_prices_1"),
        ("q2.prices", "q2_prices_2"),
    ]
    assert _rows("book", "q2_prices") == (["a"], [[1]])
    assert _rows("book", "q2_prices_1") == (["b"], [[2]])
    assert _rows("book", "q2_prices_2") == (["c"], [[3]])
    (warning,) = attached["warnings"]
    for named in ("'Q2 Prices'", "'Q2-Prices'", "'q2.prices'", "q2_prices_2"):
        assert named in warning, warning


@pytest.mark.parametrize("suffix", [".xlsx", ".ods"])
@pytest.mark.parametrize(
    ("order", "expected", "clashing"),
    [
        # The suffix the second sheet is given is the third sheet's own name,
        # so the third is moved too: two clashes, a and then a_1.
        (
            ["a", "a!", "a_1"],
            ["a", "a_1", "a_1_1"],
            [["'a'", "'a!'"], ["'a!'", "'a_1'"]],
        ),
        # The other way round: the suffix already taken is skipped over, and
        # only a is contested.
        (["a_1", "a", "a!"], ["a_1", "a", "a_2"], [["'a'", "'a!'"]]),
    ],
)
def test_a_suffix_never_lands_on_a_name_another_sheet_has(
    session, suffix, order, expected, clashing
):
    sheets = {name: {f"c{i}": [i]} for i, name in enumerate(order)}
    book = _workbook(session / f"suffixed{suffix}", sheets)

    attached = call("attach", database=str(book), nickname="book")

    assert attached["ok"] is True, attached
    assert [(one["sheet"], one["table"]) for one in attached["loaded"]] == list(
        zip(order, expected)
    )
    assert len(set(attached["tables"])) == 3
    for i, table in enumerate(expected):
        assert _rows("book", table) == ([f"c{i}"], [[i]])
    # One warning per contested name, naming every sheet that wanted it.
    warnings = attached["warnings"]
    assert len(warnings) == len(clashing)
    for warning, named in zip(warnings, clashing):
        assert all(sheet in warning for sheet in named), warning


def test_a_workbook_whose_sheets_do_not_collide_carries_no_warning(session):
    attached = call("attach", database=str(session / "workbook.xlsx"), nickname="book")

    assert attached["ok"] is True
    assert "warnings" not in attached
    assert [(one["sheet"], one["table"]) for one in attached["loaded"]] == [
        ("staff", "staff"),
        ("departments", "departments"),
    ]


def test_a_flat_file_names_no_sheet(session):
    attached = call(
        "attach", database=str(session / "simple.csv"), nickname="flat", delimiter=","
    )

    assert attached["ok"] is True
    assert "sheet" not in attached["loaded"][0]


def test_create_names_the_tables_a_colliding_workbook_would_arrive_as(session):
    book = _workbook(
        session / "collide.xlsx", {"Q2 Prices": {"a": [1]}, "Q2-Prices": {"b": [2]}}
    )
    call("attach", database=str(session / "simple.csv"), nickname="s", delimiter=",")

    refused = call("create", nickname="s", type="table", source=str(book))

    assert refused["ok"] is False
    assert "arriving as q2_prices, q2_prices_1" in refused["error"]


def _numbers(path: Path, placed: list[tuple[str, str, str, int]]) -> Path:
    """A Numbers document with one single-column table per (sheet, table) entry."""
    parser = pytest.importorskip("numbers_parser")
    document = None
    for sheet, table, column, value in placed:
        if document is None:
            document = parser.Document(
                sheet_name=sheet, table_name=table, num_rows=2, num_cols=1
            )
            target = document.sheets[0].tables[0]
        elif sheet in [s.name for s in document.sheets]:
            target = document.sheets[sheet].add_table(
                table_name=table, num_rows=2, num_cols=1
            )
        else:
            document.add_sheet(sheet, table, num_rows=2, num_cols=1)
            target = document.sheets[-1].tables[0]
        target.write(0, 0, column)
        target.write(1, 0, value)
    document.save(str(path))
    reread = parser.Document(str(path))
    assert [(s.name, t.name) for s in reread.sheets for t in s.tables] == [
        (sheet, table) for sheet, table, _, _ in placed
    ]
    return path


def test_numbers_tables_that_clean_up_alike_are_qualified_by_their_sheet(session):
    """The sheet-prefix rule asks about the table name SQL will see, not the raw one."""
    document = _numbers(
        session / "collide.numbers",
        [("North", "Q2 Prices", "a", 1), ("South", "Q2-Prices", "b", 2)],
    )

    attached = call("attach", database=str(document), nickname="nums")

    assert attached["ok"] is True, attached
    assert [one["table"] for one in attached["loaded"]] == [
        "north_q2_prices",
        "south_q2_prices",
    ]
    assert _rows("nums", "north_q2_prices") == (["a"], [[1]])
    assert _rows("nums", "south_q2_prices") == (["b"], [[2]])
    # Qualifying by sheet is the reader's own rule and costs nobody a table, so
    # there is no collision left to warn about.
    assert "warnings" not in attached


def test_numbers_tables_on_one_sheet_that_clean_up_alike_each_land(session):
    """Qualifying by sheet cannot separate two tables on the same sheet."""
    document = _numbers(
        session / "one_sheet.numbers",
        [("North", "Q2 Prices", "a", 1), ("North", "Q2-Prices", "b", 2)],
    )

    attached = call("attach", database=str(document), nickname="nums")

    assert attached["ok"] is True, attached
    assert [one["table"] for one in attached["loaded"]] == [
        "north_q2_prices",
        "north_q2_prices_1",
    ]
    assert _rows("nums", "north_q2_prices") == (["a"], [[1]])
    assert _rows("nums", "north_q2_prices_1") == (["b"], [[2]])
    (warning,) = attached["warnings"]
    assert "north_q2_prices_1" in warning


def test_a_column_suffix_never_lands_on_a_name_another_column_has(session):
    """The same rule names columns, and had the same hole: a, A, a_1 made a_1 twice."""
    (session / "cols.csv").write_text("a,A,a_1\n1,2,3\n")

    attached = call(
        "attach", database=str(session / "cols.csv"), nickname="c", delimiter=","
    )

    assert attached["ok"] is True, attached
    assert _rows("c", "cols") == (["a", "a_1", "a_1_1"], [[1, 2, 3]])


# ---------------------------------------------------------------------------
# A password is masked where it is a credential, and nowhere else (#82)
# ---------------------------------------------------------------------------
#
# The masker used to replace every occurrence of the password's characters,
# so a short password rewrote the driver's own words: password `e` turned
# "connection refused" into "conn***ction r***fus***d". A refusal is read while
# the credentials are still wrong, which is when its wording matters most. The
# credential is masked where it sits as a credential — in a URL's userinfo, or
# under a password key in what a driver echoes — and the rest is left alone.
#
# Port 1 on the loopback refuses at once, so the two network drivers fail fast
# and in their own words without a container behind them.


def _network_open(session):
    config_module.use(Config(roots=(session,), network_enabled=True))


#: A driver URL whose open fails, and words its complaint must still carry.
_REFUSING_DRIVERS = [
    pytest.param(
        "postgresql+psycopg://u:{password}@127.0.0.1:1/db",
        "Connection refused",
        id="postgres",
    ),
    pytest.param(
        "mysql+pymysql://u:{password}@127.0.0.1:1/db",
        "Can't connect to MySQL server",
        id="mysql",
    ),
]


@pytest.mark.parametrize("password", ["e", "con", "r", "t"])
@pytest.mark.parametrize(("template", "driver_words"), _REFUSING_DRIVERS)
def test_a_short_password_does_not_corrupt_the_drivers_complaint(
    session, template, driver_words, password
):
    """A password that is also a common substring leaves the complaint intact."""
    _network_open(session)

    answer = call("attach", database=template.format(password=password))

    assert answer["ok"] is False
    assert driver_words in answer["error"], answer["error"]
    # One mask, in the URL's userinfo — none scattered through the prose.
    assert answer["error"].count("***") == 1, answer["error"]


@pytest.mark.parametrize("password", ["e", "con"])
def test_a_duckdb_url_with_a_short_password_keeps_the_drivers_words(session, password):
    """The repro in #82: duckdb's own complaint read `conn***ct()`."""
    target = session / "junk.db"
    target.write_bytes(b"")

    answer = call("attach", database=f"duckdb://u:{password}@/{target}")

    assert answer["ok"] is False
    assert "connect(): incompatible function arguments" in answer["error"]
    assert answer["error"].count("***") == 1, answer["error"]


#: A password made of the characters a URL gives meaning to, spelled as the
#: caller must spell it in the URL.
_SPECIAL = "p@ss/w:rd ?#%"
_SPECIAL_IN_URL = "p%40ss%2Fw%3Ard%20%3F%23%25"


@pytest.mark.parametrize(("template", "driver_words"), _REFUSING_DRIVERS)
def test_a_password_with_url_characters_is_masked_in_every_spelling(
    session, template, driver_words
):
    _network_open(session)

    answer = call("attach", database=template.format(password=_SPECIAL_IN_URL))

    assert answer["ok"] is False
    assert driver_words in answer["error"]
    for spelling in (_SPECIAL, _SPECIAL_IN_URL, "p@ss", "p%40ss"):
        assert spelling not in answer["error"], answer["error"]


def test_a_password_with_url_characters_is_masked_behind_the_closed_network(
    session,
):
    answer = call(
        "attach", database=f"postgresql://u:{_SPECIAL_IN_URL}@db.example.com/s"
    )

    assert answer["ok"] is False
    assert "network" in answer["error"]
    for spelling in (_SPECIAL, _SPECIAL_IN_URL, "p@ss", "p%40ss"):
        assert spelling not in answer["error"], answer["error"]


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        pytest.param(
            "query",
            {
                "nickname": "staff",
                "sql": "SELECT 1",
                "path": "postgresql://u:hunter2@localhost:5432/db.csv",
                "delimiter": ",",
            },
            id="query-path",
        ),
        pytest.param(
            "save",
            {
                "nickname": "staff",
                "path": "postgresql://u:hunter2@localhost:5432/db.sqlite",
            },
            id="save-path",
        ),
        pytest.param(
            "attach",
            {"database": "post gres://u:hunter2@db.example.com/sales"},
            id="attach-not-quite-a-url",
        ),
    ],
)
def test_a_credential_read_as_a_path_is_masked_in_the_refusal(session, tool, arguments):
    """A URL that never reaches the URL code is still a credential.

    Taken as a path, ``postgresql://u:hunter2@…`` is normalised to
    ``postgresql:/u:hunter2@…`` before any message is written, so the masking
    that ``attach`` applies to URLs never sees it. Every refusal leaving the
    server is masked on the way out, whichever path produced it.
    """
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    answer = call(tool, **arguments)

    assert answer["ok"] is False
    assert "hunter2" not in answer["error"], answer["error"]


# ---------------------------------------------------------------------------
# create reads a file, and a URL handed to it is refused as one (#88)
# ---------------------------------------------------------------------------
#
# create's source used to be taken as a path whatever it was, so a database
# URL came back as "No such file: <cwd>/postgresql:/u:hunter2@…" — the password
# in clear, and a message about a file nobody named. create reads a file; a
# database is read by attach. A URL is refused as a URL, its password masked,
# and the refusal names the verb that takes one.


@pytest.mark.parametrize(
    ("source", "secrets"),
    [
        pytest.param(
            "postgresql://u:hunter2@localhost:5432/db", ["hunter2"], id="postgres"
        ),
        pytest.param("postgresql://u:e@localhost:5432/db", ["u:e@"], id="short"),
        pytest.param(
            f"mysql+pymysql://u:{_SPECIAL_IN_URL}@localhost/db",
            [_SPECIAL, _SPECIAL_IN_URL, "p@ss", "p%40ss"],
            id="url-special",
        ),
        pytest.param(
            "postgresql://u:hunter2@localhost:notaport/db",
            ["hunter2"],
            id="unparseable",
        ),
        pytest.param("duckdb:///warehouse.duckdb", [], id="local-url"),
        pytest.param("sqlite:///shop.db", [], id="sqlite-url"),
    ],
)
def test_create_refuses_a_url_and_names_attach(session, source, secrets):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    answer = call("create", nickname="staff", type="table", source=source)

    assert answer["ok"] is False
    error = answer["error"]
    assert "attach" in error, error
    assert "No such file" not in error, error
    for secret in secrets:
        assert secret not in error, error
    # The URL is named, masked where it carried a password and nowhere else.
    assert error.count("***") == (1 if secrets else 0), error


def test_create_refusing_a_url_adds_nothing(session):
    call(
        "attach", database=str(session / "simple.csv"), nickname="staff", delimiter=","
    )

    call(
        "create",
        nickname="staff",
        type="table",
        source="postgresql://u:hunter2@localhost:5432/db",
    )

    assert call("directory", nickname="staff")["tables"] == ["simple"]


# ---------------------------------------------------------------------------
# A date from a typed format compares the way a date from a CSV does (#87)
# ---------------------------------------------------------------------------
#
# Parquet, Feather, ORC, the workbooks and YAML all *declare* their dates, so
# pandas hands them over already typed. Those columns used to land as INTEGER
# nanoseconds since the epoch, and the obvious query against them —
# `WHERE seen > '2024-03-02'` — compared an integer to text and returned zero
# rows with `ok: true`. The same frame written to a CSV answered correctly.
# Every assertion below is on what a query returns, never on the declaration
# alone (CONSTRAINTS §5.1).

#: Each typed writer, and the module it needs. Every one of them was driven in
#: the issue and behaved identically, so every one is asserted.
TYPED_WRITERS = {
    ".parquet": ("pyarrow", lambda frame, path: frame.to_parquet(path)),
    ".feather": ("pyarrow", lambda frame, path: _write_feather(frame, path)),
    ".orc": ("pyarrow", lambda frame, path: frame.to_orc(path)),
    # Named, so the workbook's one table is called what the other formats' is.
    ".xlsx": (
        "openpyxl",
        lambda frame, path: frame.to_excel(path, index=False, sheet_name="days"),
    ),
    ".ods": (
        "odf",
        lambda frame, path: frame.to_excel(path, index=False, sheet_name="days"),
    ),
}


def _write_feather(frame, path) -> None:
    """Feather through Arrow IPC, as the server's own writer does.

    ``DataFrame.to_feather`` goes through ``pyarrow.feather.write_feather``,
    which pyarrow deprecates, and the warning it raises is not this test's.
    """
    import pyarrow as pa

    table = pa.Table.from_pandas(frame, preserve_index=False)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _days_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "seen": pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03"]),
            "v": [10, 20, 30],
        }
    )


@pytest.mark.parametrize("suffix", sorted(TYPED_WRITERS))
def test_a_typed_date_filters_by_a_date_literal(session, suffix):
    """The repro from #87, for every typed format the issue drove."""
    module, write = TYPED_WRITERS[suffix]
    pytest.importorskip(module)
    write(_days_frame(), session / f"days{suffix}")

    attached = call("attach", database=str(session / f"days{suffix}"), nickname="d")
    assert attached["ok"] is True, attached

    answer = call(
        "query", nickname="d", sql="SELECT id FROM days WHERE seen > '2024-03-02'"
    )
    assert answer["ok"] is True, answer
    assert answer["rows"] == [[3]]


def test_a_typed_date_is_described_the_way_the_same_csv_date_is(session):
    """One representation for one kind of value, whichever file it came from.

    The two descriptions used to differ — ``INTEGER`` with a tick unit against
    ``TEXT`` in canonical UTC — and the difference was the bug: a caller reading
    either has to be able to write the same query against both.
    """
    pytest.importorskip("pyarrow")
    frame = _days_frame()
    frame.to_parquet(session / "days.parquet")
    frame.to_csv(session / "days.csv", index=False)

    typed = call("attach", database=str(session / "days.parquet"), nickname="p")
    text = call(
        "attach", database=str(session / "days.csv"), nickname="c", delimiter=","
    )

    def seen(payload):
        return next(c for c in payload["loaded"][0]["columns"] if c["name"] == "seen")

    assert seen(typed) == seen(text)
    assert seen(typed) == {
        "name": "seen",
        "type": "TEXT",
        "temporal": "iso8601_utc",
        "normalized": "UTC",
    }


def test_a_typed_datetime_orders_chronologically(session):
    """Shuffled, spanning three years, with a time of day on every value."""
    pytest.importorskip("pyarrow")
    import pandas as pd

    instants = [
        ("rank4", "2025-03-01 14:30:00"),
        ("rank1", "2023-11-30 00:01:00"),
        ("rank5", "2025-12-25 12:00:00"),
        ("rank2", "2024-01-05 09:15:00"),
        ("rank3", "2024-07-04 23:59:00"),
    ]
    pd.DataFrame(
        {
            "label": [label for label, _ in instants],
            "seen": pd.to_datetime([at for _, at in instants]),
        }
    ).to_parquet(session / "ranked.parquet")
    call("attach", database=str(session / "ranked.parquet"), nickname="r")

    ordered = call("query", nickname="r", sql="SELECT label FROM ranked ORDER BY seen")
    latest = call(
        "query",
        nickname="r",
        sql="SELECT label FROM ranked WHERE seen = (SELECT max(seen) FROM ranked)",
    )
    after = call(
        "query",
        nickname="r",
        sql="SELECT label FROM ranked WHERE seen >= '2025-01-01' ORDER BY seen",
    )

    assert [row[0] for row in ordered["rows"]] == [
        "rank1",
        "rank2",
        "rank3",
        "rank4",
        "rank5",
    ]
    assert latest["rows"] == [["rank5"]]
    assert after["rows"] == [["rank4"], ["rank5"]]


def test_a_tz_aware_datetime_keeps_its_instant(session):
    """An offset is honoured, not dropped: the value is the same instant, in UTC.

    23:30 in Zurich on the first of March is 22:30 UTC on the same day. Read
    with its offset thrown away it would be 23:30 UTC — an hour late — and read
    as local wall time it would sort after a UTC value it actually precedes.
    Asserted three ways: the stored value, a filter whose answer flips if the
    hour moves, and a join against a CSV writing the same instant in another
    offset.
    """
    pytest.importorskip("pyarrow")
    import pandas as pd

    pd.DataFrame(
        {
            "id": [1, 2],
            "seen": pd.to_datetime(
                ["2024-03-01 23:30:00", "2024-03-02 00:30:00"]
            ).tz_localize("Europe/Zurich"),
        }
    ).to_parquet(session / "zurich.parquet")
    write_csv(
        session / "newyork.csv",
        "id,seen\n1,2024-03-01T17:30:00-05:00\n2,2024-03-01T18:30:00-05:00\n",
    )
    call("attach", database=str(session / "zurich.parquet"), nickname="z")

    stored = call("query", nickname="z", sql="SELECT seen FROM zurich ORDER BY id")
    before_eleven = call(
        "query",
        nickname="z",
        sql="SELECT id FROM zurich WHERE seen < '2024-03-01T23:00:00Z'",
    )
    landed = call(
        "create",
        nickname="z",
        type="table",
        source=str(session / "newyork.csv"),
        delimiter=",",
    )
    joined = call(
        "query",
        nickname="z",
        sql="SELECT count(*) FROM zurich JOIN newyork ON zurich.seen = newyork.seen",
    )

    assert stored["rows"] == [["2024-03-01T22:30:00Z"], ["2024-03-01T23:30:00Z"]]
    assert before_eleven["rows"] == [[1]]
    assert landed["ok"] is True, landed
    assert joined["rows"] == [[2]]


def test_a_missing_typed_datetime_is_null(session):
    """NaT is pandas' gap; SQL's is NULL, and ``IS NULL`` has to find it.

    Not the string ``'NaT'`` that ``strftime`` renders, and not the smallest
    int64, which is what NaT views as.
    """
    pytest.importorskip("pyarrow")
    import pandas as pd

    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "seen": pd.to_datetime(["2024-03-01", None, "2024-03-03"]),
        }
    ).to_parquet(session / "gapped.parquet")
    call("attach", database=str(session / "gapped.parquet"), nickname="g")

    gaps = call("query", nickname="g", sql="SELECT id FROM gapped WHERE seen IS NULL")
    earliest = call("query", nickname="g", sql="SELECT min(seen) FROM gapped")

    assert gaps["rows"] == [[2]]
    assert earliest["rows"] == [["2024-03-01"]]


def test_a_parquet_date_column_filters_by_a_date_literal(session):
    """A Parquet ``date32`` reaches pandas as ``datetime.date`` objects, not a dtype.

    So it takes a different path from a ``datetime64`` column, and it was ticks
    as well: midnight UTC in nanoseconds, declared INTEGER.
    """
    pytest.importorskip("pyarrow")
    import datetime as dt

    import pandas as pd

    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "day": [dt.date(2024, 3, 1), None, dt.date(2024, 3, 3)],
        }
    ).to_parquet(session / "dates.parquet")
    attached = call("attach", database=str(session / "dates.parquet"), nickname="dd")

    after = call(
        "query", nickname="dd", sql="SELECT id FROM dates WHERE day > '2024-03-02'"
    )
    gaps = call("query", nickname="dd", sql="SELECT id FROM dates WHERE day IS NULL")

    day = next(c for c in attached["loaded"][0]["columns"] if c["name"] == "day")
    assert day["temporal"] == "iso8601_utc", day
    assert after["rows"] == [[3]]
    assert gaps["rows"] == [[2]]


def test_a_yaml_date_filters_by_a_date_literal(session):
    """YAML types an unquoted date as well, so it took the tick path too."""
    pytest.importorskip("yaml")
    (session / "events.yaml").write_text(
        "- id: 1\n  seen: 2024-03-01\n"
        "- id: 2\n  seen: 2024-03-02 10:00:00\n"
        "- id: 3\n  seen: 2024-03-03\n"
    )
    call("attach", database=str(session / "events.yaml"), nickname="y")

    answer = call(
        "query",
        nickname="y",
        sql="SELECT id FROM events WHERE seen > '2024-03-02' ORDER BY seen",
    )

    assert answer["ok"] is True, answer
    assert answer["rows"] == [[2], [3]]


def test_a_typed_date_keeps_its_description_through_an_export(session):
    """The tick tag did not survive a round trip; the canonical text does.

    Exported with ``query(path=…)`` and attached again, a tick column came back
    as a bare INTEGER with nothing to say it had ever been a date.
    """
    pytest.importorskip("pyarrow")
    _days_frame().to_parquet(session / "days.parquet")
    call("attach", database=str(session / "days.parquet"), nickname="d")
    exported = call(
        "query",
        nickname="d",
        sql="SELECT id, seen FROM days",
        path=str(session / "again.parquet"),
    )
    assert exported["ok"] is True, exported

    again = call("attach", database=str(session / "again.parquet"), nickname="a")
    after = call(
        "query", nickname="a", sql="SELECT id FROM again WHERE seen > '2024-03-02'"
    )

    seen = next(c for c in again["loaded"][0]["columns"] if c["name"] == "seen")
    assert seen.get("temporal") == "iso8601_utc", seen
    assert after["rows"] == [[3]]
