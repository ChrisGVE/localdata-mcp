"""The DuckDB reader runs a statement only if DuckDB's own parser calls it a read (#137).

``query`` only reads. On DuckDB that used to rest on three things: a read-only
datasource opened ``access_mode=read_only``, a writable one read in a
``READ ONLY`` transaction, and file access switched off (#105). None of them
stops DuckDB writing the database's own ``.wal``, which it exempts from the
switched-off file access (#106). With a WAL still holding a committed row —
a writer killed before its checkpoint — ``COPY (SELECT 'g') TO '<file>.wal'
(USE_TMP_FILE false)`` replaced it, answered ``ok``, and the next open of the
file had lost the row, with no error anywhere.

So the reader now asks DuckDB's parser (``extract_statements``) what the
statement is before running it, and runs it only if it is exactly one query.
These drive that over the protocol, because the refusal's wording is what
reaches the client, and once directly against the function that decides it.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import foreign
import pytest
from fastmcp import Client
from sqlalchemy import Integer, Text

from localdata_mcp import config as config_module
from localdata_mcp import dialects
from localdata_mcp import server as server_module


@pytest.fixture(autouse=True)
def session(tmp_path):
    """A fresh registry and a fresh allowed root for every test."""
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(config_module.Config(roots=(root,)))
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


def build_warehouse(root: Path) -> Path:
    """``warehouse.db``: one table ``sales`` of two rows, checkpointed."""
    return foreign.build_database(
        root / "warehouse.db",
        "sales",
        [("region", Text), ("amount", Integer)],
        [("north", 100), ("south", 250)],
        dialect="duckdb",
    )


def attach(root: Path, *, writable: bool) -> Path:
    target = build_warehouse(root)
    grant = {"writable": True} if writable else {}
    attached = call("attach", database=str(target), nickname="wh", **grant)
    assert attached["ok"] is True, attached
    return target


#: A writer that commits a third row and dies before checkpointing it, so the
#: row exists only in ``warehouse.db.wal``. Another process, with DuckDB
#: itself, because that is the situation being reproduced: a database file
#: this server attaches after somebody else's writer crashed.
_KILLED_WRITER = (
    "import duckdb, os, sys\n"
    "con = duckdb.connect(sys.argv[1])\n"
    "con.execute(\"SET checkpoint_threshold = '1GB'\")\n"
    "con.execute(\"INSERT INTO sales VALUES ('west', 7)\")\n"
    "os._exit(0)\n"
)


def rows_on_reopen(target: Path) -> int:
    """How many rows ``sales`` holds when DuckDB next opens the file."""
    connection = duckdb.connect(str(target), read_only=True)
    try:
        return connection.execute("SELECT count(*) FROM sales").fetchone()[0]
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# The data loss, end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reach", ["file", "url"])
def test_a_copy_onto_a_pending_wal_cannot_lose_a_committed_row(session, reach):
    """The repro in #137, through ``query`` on a datasource attached read-only."""
    target = build_warehouse(session)
    subprocess.run([sys.executable, "-c", _KILLED_WRITER, str(target)], check=True)
    wal = Path(f"{target}.wal")
    # The condition this test is named for: a committed row only the WAL holds.
    assert wal.exists() and wal.stat().st_size > 0
    database = str(target) if reach == "file" else f"duckdb:///{target}"
    assert call("attach", database=database, nickname="wh")["ok"] is True
    seen = call("query", nickname="wh", sql="SELECT count(*) FROM sales")
    assert seen["rows"] == [[3]], seen

    answer = call(
        "query",
        nickname="wh",
        sql=f"COPY (SELECT 'g') TO '{wal}' (USE_TMP_FILE false)",
    )

    assert answer["ok"] is False, answer
    assert answer["error"].startswith(
        "query reads; it does not write. This statement asks to COPY"
    ), answer
    assert call("detach", nickname="wh")["ok"] is True
    server_module._reset()
    assert rows_on_reopen(target) == 3


@pytest.mark.parametrize("writable", [False, True])
def test_explain_analyze_cannot_run_the_copy_it_explains(session, writable):
    """``EXPLAIN ANALYZE`` runs its statement; measured, this one wrote the WAL."""
    target = attach(session, writable=writable)
    wal = Path(f"{target}.wal")
    before = wal.read_bytes() if wal.exists() else None

    answer = call(
        "query",
        nickname="wh",
        sql=f"EXPLAIN ANALYZE COPY (SELECT 'g') TO '{wal}' (USE_TMP_FILE false)",
    )

    assert answer["ok"] is False, answer
    assert "asks to EXPLAIN a COPY statement" in answer["error"], answer
    assert (wal.read_bytes() if wal.exists() else None) == before


# ---------------------------------------------------------------------------
# What still reads, on either grant
# ---------------------------------------------------------------------------

#: Every spelling of a read measured working through ``query`` before #137,
#: with the first value it answers. DuckDB's parser calls each a ``SELECT``,
#: except ``EXPLAIN`` (which runs only over a query) and ``CALL`` (DuckDB's
#: spelling of ``SELECT * FROM f()``).
_READS = [
    ("SELECT region FROM sales ORDER BY region", "north"),
    ("SELECT region FROM sales ORDER BY region;", "north"),
    ("WITH q AS (SELECT region FROM sales) SELECT * FROM q ORDER BY 1", "north"),
    ("FROM sales SELECT region ORDER BY region", "north"),
    ("TABLE sales", "north"),
    ("VALUES ('v')", "v"),
    ("SUMMARIZE sales", "region"),
    ("DESCRIBE sales", "region"),
    ("SHOW sales", "region"),
    ("SHOW TABLES", "sales"),
    ("PRAGMA show_tables", "sales"),
    ("PRAGMA table_info('sales')", 0),
    ("CALL pragma_table_info('sales')", 0),
    ("EXPLAIN SELECT * FROM sales", "physical_plan"),
    ("EXPLAIN ANALYZE SELECT * FROM sales", "analyzed_plan"),
    ("EXPLAIN (FORMAT json) SELECT * FROM sales", "physical_plan"),
    ("PIVOT sales ON region IN ('north', 'south') USING sum(amount)", 100),
    ("UNPIVOT (SELECT 1 AS a) ON a INTO NAME k VALUE v", "a"),
]


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize(("sql", "first"), _READS)
def test_a_read_duckdb_parses_as_a_query_still_answers(session, writable, sql, first):
    attach(session, writable=writable)

    answer = call("query", nickname="wh", sql=sql)

    assert answer["ok"] is True, answer
    assert answer["rows"][0][0] == first, answer


# ---------------------------------------------------------------------------
# What is refused, and how it is worded
# ---------------------------------------------------------------------------

#: Statements that are not a query to DuckDB's parser, with the words the
#: refusal names them by. Most failed in DuckDB already; ``LOAD``, ``SET
#: VARIABLE``, ``PREPARE``, ``EXECUTE``, ``VACUUM``, ``ANALYZE``, a ``PRAGMA``
#: that is not a query and ``COPY`` onto the WAL answered ``ok`` before #137.
_REFUSED = [
    ("COPY sales TO '{root}/out.csv'", "COPY"),
    ("COPY sales FROM '{root}/in.csv'", "COPY"),
    ("EXPORT DATABASE '{root}/exported'", "EXPORT DATABASE"),
    ("ATTACH '{root}/other.db' AS other", "ATTACH"),
    ("DETACH other", "DETACH"),
    ("INSTALL json", "INSTALL or LOAD an extension"),
    ("LOAD json", "INSTALL or LOAD an extension"),
    ("SET threads = 1", "change a setting"),
    ("RESET threads", "change a setting"),
    ("SET VARIABLE v = 1", "change a setting"),
    ("USE memory", "change a setting"),
    ("PRAGMA enable_profiling", "run a PRAGMA that is not a query"),
    ("VACUUM", "VACUUM or ANALYZE"),
    ("ANALYZE", "VACUUM or ANALYZE"),
    ("PREPARE p AS SELECT 1", "PREPARE"),
    ("EXECUTE p", "EXECUTE"),
    ("INSERT INTO sales VALUES ('east', 1)", "INSERT"),
    ("CREATE TABLE extra (a INTEGER)", "CREATE"),
    ("CREATE MACRO m(x) AS x + 1", "CREATE"),
    ("ALTER TABLE sales ADD COLUMN c INTEGER", "ALTER"),
    ("DROP TABLE sales", "DROP"),
    ("EXPLAIN INSERT INTO sales VALUES ('east', 1)", "EXPLAIN an INSERT statement"),
    ("SELECT 1 AS a; SELECT 2 AS b", "run 2 statements (SELECT, SELECT) in one call"),
    ("SELECT 1 AS a; COPY sales TO '{root}/out.csv'", "COPY"),
]


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize(("template", "named"), _REFUSED)
def test_a_statement_that_is_not_a_query_is_refused_by_name(
    session, writable, template, named
):
    target = attach(session, writable=writable)
    (session / "in.csv").write_text("region,amount\neast,1\n")
    before = sorted(path.name for path in session.iterdir())

    answer = call("query", nickname="wh", sql=template.format(root=session))

    assert answer["ok"] is False, answer
    error = answer["error"]
    assert error.startswith("query reads; it does not write."), error
    assert f"This statement asks to {named}" in error, error
    for verb in ("create", "update", "drop"):
        assert verb in error, error
    assert sorted(path.name for path in session.iterdir()) == before
    call("detach", nickname="wh")
    server_module._reset()
    assert rows_on_reopen(target) == 2


def test_a_copy_is_pointed_at_query_s_own_way_of_writing_a_file(session):
    attach(session, writable=False)

    answer = call("query", nickname="wh", sql="COPY sales TO 'out.csv'")

    assert "path=" in answer["error"], answer


def test_a_parse_error_still_reads_as_duckdb_s_parse_error(session):
    """The gate asks the parser first, so a typo must not come back as a write."""
    attach(session, writable=False)

    answer = call("query", nickname="wh", sql="SELEC 1")

    assert answer["ok"] is False, answer
    assert "does not write" not in answer["error"], answer
    assert 'syntax error at or near "SELEC"' in answer["error"], answer


@pytest.mark.parametrize("writable", [False, True])
def test_an_import_read_while_parsing_still_lands_nothing(session, writable):
    """The one statement DuckDB's parser reads a file for.

    ``PRAGMA import_database`` is expanded while it is parsed, from the
    directory's ``schema.sql``, so asking the parser reads that file. On the
    reader that read is refused by the seal, before any statement exists, and
    the table the export holds never lands.
    """
    exporter = duckdb.connect()
    try:
        exporter.execute("CREATE TABLE planted AS SELECT 'planted' AS w")
        exporter.execute(f"EXPORT DATABASE '{session / 'exported'}'")
    finally:
        exporter.close()
    attach(session, writable=writable)

    answer = call(
        "query",
        nickname="wh",
        sql=f"PRAGMA import_database('{session / 'exported'}')",
    )

    assert answer["ok"] is False, answer
    tables = call("query", nickname="wh", sql="SHOW TABLES")
    assert tables["rows"] == [["sales"]], tables


# ---------------------------------------------------------------------------
# The function that decides, on a bare connection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "refused"),
    [
        ("SELECT 1", None),
        ("  select 1 ;  ", None),
        ("/* a */ EXPLAIN /* b */ ANALYZE SELECT 1", None),
        ("explain (analyze) select 1", None),
        ("EXPLAIN (FORMAT json, ANALYZE) SELECT 1", None),
        ("CALL pragma_version()", None),
        ("CHECKPOINT", None),
        ("COPY (SELECT 1) TO 'x.wal'", "COPY"),
        ("EXPLAIN (ANALYZE) COPY (SELECT 1) TO 'x.wal'", "EXPLAIN a COPY statement"),
        ("EXPLAIN (SELECT 1)", None),
        ("EXPLAIN (FORMAT json) (SELECT 1)", None),
        ("COMMIT", "begin, commit or roll back a transaction of its own"),
        ("SELECT 1; COMMIT", "begin, commit or roll back a transaction of its own"),
        ("PRAGMA force_checkpoint", "run a PRAGMA that is not a query"),
        ("CREATE FUNCTION f(x) AS x", "CREATE"),
    ],
)
def test_the_gate_names_what_duckdb_parses(sql, refused):
    connection = duckdb.connect()
    try:
        found = dialects._duckdb_read_refusal(connection, sql)
    finally:
        connection.close()

    if refused is None:
        assert found is None
    else:
        assert found is not None and found.startswith(refused), found


def test_two_statements_are_named_with_what_each_one_is():
    connection = duckdb.connect()
    try:
        found = dialects._duckdb_read_refusal(
            connection, "SELECT 1; INSERT INTO t VALUES (1)"
        )
    finally:
        connection.close()

    assert found == "INSERT (DuckDB reads this as 2 statements: SELECT, INSERT)"


def test_asking_the_parser_runs_nothing(tmp_path):
    """The gate's own question must not be the write it is guarding against.

    Pinned against DuckDB rather than taken on trust: ``extract_statements``
    only parses, so a refused ``CREATE`` leaves no table and a refused ``COPY``
    leaves no file.
    """
    copied = tmp_path / "copied.csv"
    connection = duckdb.connect()
    try:
        for sql in ("CREATE TABLE made (a INTEGER)", f"COPY (SELECT 1) TO '{copied}'"):
            assert dialects._duckdb_read_refusal(connection, sql) is not None
        tables = connection.execute("SELECT count(*) FROM duckdb_tables()").fetchone()
    finally:
        connection.close()

    assert tables == (0,)
    assert not copied.exists()
