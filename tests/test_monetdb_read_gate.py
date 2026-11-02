"""MonetDB's reader runs only what MonetDB's own ``EXPLAIN`` calls a read (#119).

MonetDB has no read-only transaction (``Readonly transactions not supported``),
its cursor describes ``INSERT … RETURNING`` exactly as it describes a
``SELECT``, and its grammar nests a write under ``WITH``. So before #119
``query`` answered ``ok`` with the row of an ``INSERT … RETURNING`` that the
read connection then rolled back: a row that does not exist.

The gate has two halves, and each is tested here on its own:

* **one statement**, decided without lexing: MonetDB needs a ``;`` between
  statements (``SELECT 1 SELECT 2`` is ``expecting SCOLON``), so any ``;``
  other than one trailing is refused. That costs a read with a ``;`` inside a
  literal and never admits a second statement;
* **a read**, decided by MonetDB: ``EXPLAIN`` prints the relational plan
  without running anything, and the statement runs only if every operator in
  it is one a read produces. The plans below are copied from
  ``monetdb/monetdb:latest`` (11.55.7, Dec2025-SP3).

The live half needs the ``localdata-test-monetdb`` service and skips without
it, like every endpoint test.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import DDL, create_engine, inspect

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import (
    _monetdb_plan_names,
    _monetdb_plan_operators,
    _monetdb_single_statement,
    backend_for,
)

# ---------------------------------------------------------------------------
# One statement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT 1;",
        "SELECT 1 ;  \n",
        "WITH q AS (SELECT 1 AS a) SELECT a FROM q",
    ],
)
def test_one_statement_with_at_most_a_trailing_semicolon_passes(sql):
    assert _monetdb_single_statement(sql) is True


@pytest.mark.parametrize(
    "sql",
    [
        "COMMIT; INSERT INTO t VALUES (1)",
        "SELECT 1; SELECT 2",
        "SELECT 1;;",
        "SELECT ';'",  # the read this rule costs: refused, never misread
        "SELECT 1 /* ; */",
    ],
)
def test_any_other_semicolon_is_refused_without_reading_the_sql(sql):
    assert _monetdb_single_statement(sql) is False


# ---------------------------------------------------------------------------
# A read, by MonetDB's own plan
# ---------------------------------------------------------------------------

#: Plans MonetDB printed for reads, verbatim, except ``every-read-operator``:
#: one line per operator a read was measured producing, assembled here so the
#: whole allowlist is exercised in one place.
_READ_PLANS = {
    "select": ["project (", '| table("sys"."t") [ "t"."name" ]', ') [ "t"."name" ]'],
    "constant": [
        "project (",
        '|  [ boolean(1) "true" as "%1"."%1" ]',
        ') [ tinyint(1) "1" ]',
    ],
    "every-read-operator": [
        "REF 1 (2)",
        "recursive munion (",
        "| & REF 1 ,",
        "| top N (",
        "| | group by (",
        "| | | select (",
        "| | | | semijoin (",
        "| | | | | left outer join (",
        "| | | | | | distinct except (",
        "| | | | | | | antijoin (",
        "| | | | | | | | crossproduct (",
        "| | | | | | | | | sample(",
        "| | | | | | | | | | distinct munion (",
        "| | | | | | | | | | | distinct intersect (",
        "| | | | | | | | | | | | right outer join (",
        "| | | | | | | | | | | | | full outer join (",
        "| | | | | | | | | | | | | | munion (",
        '| | | | | | | | | | | | | | | table("sys"."t") [ "t"."n" ]',
        ') [ "r"."i" ]',
    ],
}

#: Plans MonetDB printed for writes, verbatim, and the operator each is refused by.
_WRITE_PLANS = {
    "insert-returning": (
        [
            "project (",
            "| insert(",
            '| | table("sys"."t") [ "t"."name", "t"."n", "t"."%TID%" ]',
            '| |  [ varchar(10) "x" as "%1"."%1", int(31) "1" as "%2"."%2" ]',
            '| ) [ "%1"."%1" as "t"."name", "%2"."%2" as "t"."n" ]',
            ') [ "t"."name" ]',
        ],
        "insert",
    ),
    "update": (
        [
            "REF 1 (2)",
            'table("sys"."t") [ "t"."name", "t"."n", "t"."%TID%" ]',
            "project (",
            "| update(",
            "| | & REF 1 ",
        ],
        "update",
    ),
    "delete": (
        ["project (", "| delete(", '| | table("sys"."t") [ "t"."name" ]'],
        "delete",
    ),
    "create": (["ddl", '| table("sys"."u") [ "u"."%TID%" ]'], "ddl"),
    "copy-into-file": (["ddl", "| project ("], "ddl"),
    "call": (['ddl [ "sys"."querylog_enable"() ]'], "ddl"),
}


@pytest.mark.parametrize("shape", sorted(_READ_PLANS))
def test_every_operator_of_a_read_plan_is_a_read(shape):
    operators = _monetdb_plan_operators(_READ_PLANS[shape])
    assert operators and all(read for _, read in operators), operators


@pytest.mark.parametrize("shape", sorted(_WRITE_PLANS))
def test_a_write_plan_names_its_write_operator(shape):
    plan, operator = _WRITE_PLANS[shape]
    refused = [name for name, read in _monetdb_plan_operators(plan) if not read]
    assert refused and refused[0] == operator, refused


def test_an_operator_line_that_cannot_be_read_is_refused():
    """``TRUNCATE``'s line does not fit the operator shape; unreadable is refused."""
    plan = ["truncate continue identity, restrict(", '| table("sys"."t")', ")"]
    operators = _monetdb_plan_operators(plan)
    assert operators[0] == (None, False), operators


def test_an_operator_nobody_measured_is_refused():
    assert _monetdb_plan_operators(["merge ("]) == [("merge", False)]


def test_every_quoted_pair_in_a_plan_is_read_as_a_name():
    """Functions, aggregates, columns and aliases all print as ``"a"."b"``.

    Which pairs are functions is the catalog's to say, so all of them are
    collected; the backslash escape MonetDB prints is undone.
    """
    plan = [
        '| ) [  ] [ "sys"."max" no nil ("fx_t"."a") as "%1"."%1", "sys"."count"() ]',
        ') [ "sys"."fx_w"() ]',
        ') [ varchar(3) "q\\"x" as "we\\"ird"."col" ]',
    ]
    assert _monetdb_plan_names(plan) == {
        ("sys", "max"),
        ("fx_t", "a"),
        ("%1", "%1"),
        ("sys", "count"),
        ("sys", "fx_w"),
        ('we"ird', "col"),
    }


def test_monetdb_has_a_backend_of_its_own():
    assert type(backend_for("monetdb")).__name__ == "MonetDBBackend"
    assert backend_for("monetdb").name == "monetdb"


# ---------------------------------------------------------------------------
# Live, against MonetDB
# ---------------------------------------------------------------------------

_MONETDB = [target for target in TARGETS if target.endpoint.dialect == "monetdb"]


def _call(tool: str, **arguments):
    """The in-process protocol client, as ``test_endpoints`` has it."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


@pytest.fixture(params=_MONETDB, ids=lambda target: target.name)
def monetdb(request, tmp_path):
    """A writable MonetDB slot holding a five-row table; the table's name."""
    try:
        reached = reach(request.param)
    except Unavailable as exc:
        pytest.skip(str(exc))
    root = tmp_path / "root"
    root.mkdir()
    (root / "people.csv").write_text(
        "name,department,salary\n"
        "alice,engineering,75000\nbob,sales,65000\ncarol,marketing,70000\n"
        "dave,engineering,80000\neve,hr,55000\n"
    )
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    table = f"t{uuid.uuid4().hex[:8]}_people"
    with applied(reached.environ):
        try:
            attached = _call(
                "attach", database=reached.url, nickname="endpoint", writable=True
            )
            assert attached["ok"] is True, attached
            made = _call(
                "create",
                nickname="endpoint",
                type="table",
                source=str(root / "people.csv"),
                table=table,
                delimiter=",",
            )
            assert made["ok"] is True, made
            # Functions whose bodies write, which no verb can create. MonetDB
            # records both as side_effect = false, so only the gate's catalog
            # check (system and side-effect free) stands between them and a read.
            insert = (
                f"INSERT INTO {table} (name, department, salary) VALUES ('x', 'y', 1)"
            )
            engine = create_engine(reached.url)
            try:
                with engine.begin() as conn:
                    conn.execute(
                        DDL(
                            f"CREATE FUNCTION {table}_w() RETURNS int "
                            f"BEGIN {insert}; RETURN 1; END"
                        )
                    )
                    conn.execute(
                        DDL(
                            f"CREATE FUNCTION {table}_tw() RETURNS TABLE (n int) "
                            f"BEGIN {insert}; RETURN SELECT 1; END"
                        )
                    )
            finally:
                engine.dispose()
            yield table
        finally:
            server_module._reset()
            engine = create_engine(reached.url)
            try:
                with engine.begin() as conn:
                    for function in (f"{table}_w", f"{table}_tw"):
                        conn.execute(DDL(f"DROP FUNCTION IF EXISTS {function}"))
                    for name in inspect(conn).get_table_names():
                        if name.startswith(table[:9]):
                            conn.execute(DDL(f'DROP TABLE "{name}"'))
            finally:
                engine.dispose()


def _count(table: str) -> list:
    return _call("query", nickname="endpoint", sql=f"SELECT count(*) FROM {table}")[
        "rows"
    ]


_HIDDEN_WRITES = {
    "insert-returning": "INSERT INTO {t} (name, department, salary) VALUES ('x','y',1) RETURNING name",
    "with-insert-returning": "WITH q AS (SELECT 1 AS a) INSERT INTO {t} (name, department, salary) VALUES ('x','y',1) RETURNING name",
    "update-returning": "UPDATE {t} SET salary = 1 RETURNING name",
    "delete-returning": "DELETE FROM {t} RETURNING name",
    "commit-then-insert": "COMMIT; INSERT INTO {t} (name, department, salary) VALUES ('x','y',1)",
    "select-then-insert": "SELECT 1; INSERT INTO {t} (name, department, salary) VALUES ('x','y',1)",
    "truncate": "TRUNCATE {t}",
    "create-table": "CREATE TABLE {t}_copy (a int)",
    "writing-function": "SELECT {t}_w()",
    "writing-table-function": "SELECT * FROM {t}_tw()",
}


@pytest.mark.endpoint
@pytest.mark.parametrize("shape", sorted(_HIDDEN_WRITES))
def test_monetdb_query_refuses_a_write_before_it_runs(monetdb, shape):
    """Refused in ``query``'s own words, and the table is untouched."""
    written = _call(
        "query", nickname="endpoint", sql=_HIDDEN_WRITES[shape].format(t=monetdb)
    )

    assert written["ok"] is False, written
    assert "create" in written["error"], written
    assert _count(monetdb) == [[5]], written


_READS = {
    "select": "SELECT name FROM {t} ORDER BY name LIMIT 2",
    "trailing-semicolon": "SELECT count(*) FROM {t};",
    "cte": "WITH q AS (SELECT name FROM {t}) SELECT count(*) FROM q",
    "join": "SELECT a.name FROM {t} a JOIN {t} b ON a.salary = b.salary",
    "group": "SELECT department, count(*) FROM {t} GROUP BY department HAVING count(*) > 0",
    "set-ops": "SELECT name FROM {t} UNION SELECT name FROM {t} EXCEPT SELECT 'bob'",
    "window": "SELECT name, rank() OVER (ORDER BY salary) FROM {t}",
    "subquery": "SELECT * FROM {t} WHERE salary > (SELECT avg(salary) FROM {t})",
    "catalog": "SELECT name FROM sys.tables WHERE name = '{t}'",
    "values": "SELECT * FROM (VALUES (1), (2)) AS v(a)",
}


@pytest.mark.endpoint
@pytest.mark.parametrize("shape", sorted(_READS))
def test_monetdb_query_still_answers_a_read(monetdb, shape):
    """The gate costs no read a caller writes without a ``;`` inside it."""
    read = _call("query", nickname="endpoint", sql=_READS[shape].format(t=monetdb))
    assert read["ok"] is True, read


@pytest.mark.endpoint
def test_monetdb_directory_and_stats_still_work_through_the_gate(monetdb):
    """The server's own reads on the reader, reflection and aggregates, pass."""
    listed = _call("directory", nickname="endpoint")
    stats = _call("stats", nickname="endpoint", table=monetdb)
    assert listed["ok"] is True, listed
    assert stats["ok"] is True, stats


# ---------------------------------------------------------------------------
# What the ';' refusal says, and the rephrase it offers
# ---------------------------------------------------------------------------

#: The rephrase measured on monetdb/monetdb:latest (11.55.7): MonetDB has no
#: ``chr``, and ``char(59)`` is a syntax error there (``CHAR`` is a type
#: keyword), but ``code(59)`` answers ``;`` and is a system function recorded
#: free of side effects, so it passes the gate's catalog check.
_SEMICOLON_REPHRASE = "SELECT 'a' || code(59) || 'b'"


class _NoCursor:
    """A cursor the ';' half must never reach: refusing it needs no EXPLAIN."""

    def execute(self, *args, **kwargs):
        raise AssertionError(f"the ';' refusal sent SQL to MonetDB: {args}")


def _refused_text(sql: str) -> str:
    from pymonetdb.exceptions import ProgrammingError

    from localdata_mcp.dialects import Refusal, _refuse_monetdb_write

    refusal = Refusal()
    with pytest.raises(ProgrammingError) as raised:
        _refuse_monetdb_write(refusal, _NoCursor(), sql, None)
    assert refusal.what and refusal.what in str(raised.value), raised.value
    return refusal.what


@pytest.mark.parametrize(
    "sql",
    ["SELECT 'a;b'", "SELECT 1 /* ; */", "SELECT 1; SELECT 2", "SELECT 1;;"],
)
def test_the_semicolon_refusal_is_true_wherever_the_semicolon_sits(sql):
    """It names the ';' and says a quoted one is refused too — never "two statements".

    ``SELECT 'a;b'`` is one statement, so a refusal saying "more than one
    statement" is false for it; the gate cannot tell, by design, so the words
    must hold for a ';' in a literal, a comment, or between two statements.
    """
    what = _refused_text(sql)
    assert "more than one statement" not in what, what
    assert "';'" in what, what
    assert "quoted string" in what and "comment" in what, what
    assert "MonetDB" in what, what  # says why: what MonetDB would do after it


def test_the_semicolon_refusal_offers_the_measured_rephrase():
    assert _SEMICOLON_REPHRASE in _refused_text("SELECT 'a;b'")


@pytest.mark.endpoint
def test_monetdb_refuses_a_semicolon_in_a_literal_and_says_how_to_write_one(monetdb):
    refused = _call("query", nickname="endpoint", sql="SELECT 'a;b'")

    assert refused["ok"] is False, refused
    assert "more than one statement" not in refused["error"], refused
    assert "quoted string" in refused["error"], refused
    assert _SEMICOLON_REPHRASE in refused["error"], refused


@pytest.mark.endpoint
def test_the_offered_rephrase_passes_the_gate_and_returns_a_semicolon(monetdb):
    read = _call("query", nickname="endpoint", sql=_SEMICOLON_REPHRASE)

    assert read["ok"] is True, read
    assert read["rows"] == [["a;b"]], read
