"""MySQL's and MariaDB's reader runs only a statement that cannot reach past it (#123).

``START TRANSACTION READ ONLY`` refuses a row change and DDL, and nothing
else. Measured on MySQL 8.0 and MariaDB 11, a read-only session still ran
``SELECT … INTO OUTFILE`` and ``INTO DUMPFILE`` (a file on the server),
``SET GLOBAL`` and ``SET PERSIST`` (server-wide settings, the second written
to ``mysqld-auto.cnf``), an executable comment ``/*! SET GLOBAL … */``
(MariaDB also ``/*M! … */``), and a stored function called from a
``SELECT`` whose body runs ``SET GLOBAL`` or writes a file. ``query`` then
reported a refusal for a statement that had already happened.

MySQL's ``EXPLAIN`` answers an ordinary plan for ``INTO OUTFILE`` in every
format and refuses to explain ``SET``, so the server cannot be asked what a
statement does, as it is for MonetDB (#119) and SQL Server (#122). The gate
is a statement rule that fails closed, in two halves tested here apart:

* **the statement's shape**, decided without lexing: one statement; no
  executable comment; neither ``OUTFILE`` nor ``DUMPFILE`` anywhere; and a
  first keyword that only a read begins with;
* **what it names**, decided by the catalog: no stored or loadable function,
  and no view whose definition reaches one, appears in it as a word.

The live half needs ``localdata-test-mysql`` and ``localdata-test-mariadb``
and skips without them, like every endpoint test. It connects as ``root``,
because the writes in #123 need privileges the test user does not hold, and
a gate tested where the write could not have happened anyway proves nothing.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import DDL, create_engine, inspect, make_url, text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import (
    _mysql_named,
    _mysql_reachable_names,
    _mysql_shape_refusal,
    backend_for,
)

# ---------------------------------------------------------------------------
# The statement's shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select 1;",
        "  SELECT 1 ;  \n",
        "WITH q AS (SELECT 1 AS a) SELECT a FROM q",
        "(SELECT 1) UNION (SELECT 2)",
        "TABLE t",
        "VALUES ROW(1), ROW(2)",
        "SHOW TABLES",
        "DESCRIBE t",
        "desc t",
        "EXPLAIN SELECT 1",
        "-- a note\nSELECT 1",
        "# a note\nSELECT 1",
        "/* a note */ SELECT 1",
        "SELECT /*+ MAX_EXECUTION_TIME(1000) */ 1",
        "SELECT 1 INTO @x",
    ],
)
def test_a_read_has_an_acceptable_shape(sql):
    assert _mysql_shape_refusal(sql) is None


@pytest.mark.parametrize(
    ("sql", "words"),
    [
        ("SELECT 1; SELECT 2", "more than one statement"),
        ("SELECT 1;;", "more than one statement"),
        ("SELECT 1 INTO OUTFILE '/tmp/x'", "OUTFILE"),
        ("select 1 into dumpfile '/tmp/x'", "OUTFILE"),
        ("TABLE t INTO OUTFILE '/tmp/x'", "OUTFILE"),
        ("SELECT 'outfile'", "OUTFILE"),  # the read this rule costs, and says so
        ("/*!80000 SET GLOBAL max_connections = 1 */", "executable comment"),
        ("/*M!100000 SET GLOBAL max_connections = 1 */", "executable comment"),
        ("SELECT 1 /*!80000 , 2 */", "executable comment"),
        ("SET GLOBAL max_connections = 1", "SET"),
        ("SET PERSIST max_connections = 1", "SET"),
        ("set @@global.max_connections = 1", "SET"),
        ("DO RELEASE_ALL_LOCKS()", "DO"),
        ("CALL p()", "CALL"),
        ("HANDLER t OPEN", "HANDLER"),
        ("ANALYZE SELECT 1", "ANALYZE"),
        ("INSERT INTO t VALUES (1)", "INSERT"),
        ("--x\nSET GLOBAL max_connections = 1", "--"),
        ("", "begin"),
        ("/* only a note */", "begin"),
    ],
)
def test_anything_else_is_refused_by_shape_and_says_why(sql, words):
    refused = _mysql_shape_refusal(sql)
    assert refused is not None and words in refused, refused


def test_a_comment_that_is_not_a_comment_on_mysql_is_not_skipped():
    """``--`` opens a comment only when a space follows it; ``--x`` is SQL.

    Skipping it as a comment would read the keyword after it as the first one.
    """
    refused = _mysql_shape_refusal("--x\nSELECT 1")
    assert refused is not None


# ---------------------------------------------------------------------------
# What it names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT fset()",
        "SELECT FSET ()",
        "SELECT `fset`()",
        "SELECT testdb.fset()",
        "SELECT * FROM t WHERE a = fset()",
        "SELECT 'fset'",  # a literal costs the read, never admits the call
    ],
)
def test_a_statement_naming_an_unsafe_name_is_found(sql):
    assert _mysql_named(sql, {"fset"}) == "fset"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT fset_total FROM t",
        "SELECT 1 AS my_fset",
        "SELECT $fset FROM t",
        "SELECT 1",
    ],
)
def test_a_longer_word_containing_the_name_is_not_the_name(sql):
    assert _mysql_named(sql, {"fset"}) is None


def test_no_names_finds_nothing():
    assert _mysql_named("SELECT anything()", set()) is None


def test_a_view_reaching_a_function_is_unsafe_through_any_chain():
    names = _mysql_reachable_names(
        functions={"fset"},
        views={
            "v1": "select `fset`() AS `x`",
            "v2": "select * from v1",
            "v3": "select 1",
        },
    )
    assert names == {"fset", "v1", "v2"}


def test_a_view_whose_definition_is_hidden_is_unsafe_when_anything_is():
    """``VIEW_DEFINITION`` is empty to a user without ``SHOW VIEW``.

    An unread definition might call any function, so it is treated as calling
    one; with no function to call it cannot, and it stays readable.
    """
    assert _mysql_reachable_names(functions={"f"}, views={"v": ""}) == {"f", "v"}
    assert _mysql_reachable_names(functions=set(), views={"v": ""}) == set()


def test_mysql_and_mariadb_share_the_backend_and_keep_their_names():
    for name in ("mysql", "mariadb"):
        assert type(backend_for(name)).__name__ == "MySQLBackend"
        assert backend_for(name).name == name


# ---------------------------------------------------------------------------
# Live, against MySQL and MariaDB
# ---------------------------------------------------------------------------

_SERVERS = [
    target
    for target in TARGETS
    if target.endpoint.dialect in ("mysql", "mariadb") and target.auth is None
]


def _call(tool: str, **arguments):
    """The in-process protocol client, as ``test_endpoints`` has it."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


def _as_root(url: str) -> str:
    """The container's administrator, who holds FILE and SYSTEM_VARIABLES_ADMIN."""
    return (
        make_url(url)
        .set(username="root", password="testpass")
        .render_as_string(hide_password=False)
    )


@pytest.fixture(params=_SERVERS, ids=lambda target: target.name)
def server(request, tmp_path):
    """A read-only root slot over a five-row table, and what the test needs.

    Yields ``(table, admin)``: the table's name and an engine outside the
    server, as root, to read back what a refused statement would have changed.
    """
    try:
        reached = reach(request.param)
    except Unavailable as exc:
        pytest.skip(str(exc))
    root_url = _as_root(reached.url)
    admin = create_engine(root_url)
    table = f"t{uuid.uuid4().hex[:8]}_people"
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    try:
        with admin.begin() as conn:
            conn.execute(DDL(f"CREATE TABLE {table} (name VARCHAR(20), salary INT)"))
            conn.execute(
                text(
                    f"INSERT INTO {table} VALUES ('a',1),('b',2),('c',3),('d',4),('e',5)"
                )
            )
            # Functions whose bodies reach past a read-only transaction, which
            # no verb can create: one sets a server-wide variable, one writes a
            # file. A view calling the first hides the call behind a table name.
            conn.execute(text("SET GLOBAL log_bin_trust_function_creators = 1"))
            conn.execute(
                DDL(
                    f"CREATE FUNCTION {table}_set() RETURNS INT DETERMINISTIC "
                    f"BEGIN SET GLOBAL max_connections = 777; RETURN 1; END"
                )
            )
            conn.execute(DDL(f"CREATE VIEW {table}_v AS SELECT {table}_set() AS x"))
        with applied(reached.environ):
            attached = _call("attach", database=root_url, nickname="endpoint")
            assert attached["ok"] is True, attached
            yield table, admin
    finally:
        server_module._reset()
        with admin.begin() as conn:
            conn.execute(DDL(f"DROP VIEW IF EXISTS {table}_v"))
            conn.execute(DDL(f"DROP FUNCTION IF EXISTS {table}_set"))
            for name in inspect(conn).get_table_names():
                if name.startswith(table[:9]):
                    conn.execute(DDL(f"DROP TABLE {name}"))
        admin.dispose()


def _connections(admin) -> int:
    with admin.connect() as conn:
        return conn.execute(text("SELECT @@global.max_connections")).scalar()


def _file_dir(admin) -> str:
    with admin.connect() as conn:
        return conn.execute(text("SELECT @@global.secure_file_priv")).scalar()


def _file_exists(admin, path: str) -> bool:
    with admin.connect() as conn:
        return conn.execute(
            text("SELECT LOAD_FILE(:p) IS NOT NULL"), {"p": path}
        ).scalar()


_SETTINGS = {
    "set-global": "SET GLOBAL max_connections = 777",
    "set-persist": "SET PERSIST max_connections = 777",
    "executable-comment": "/*!50000 SET GLOBAL max_connections = 777 */",
    "function": "SELECT {t}_set()",
    "view": "SELECT * FROM {t}_v",
}


@pytest.mark.endpoint
@pytest.mark.parametrize("shape", sorted(_SETTINGS))
def test_query_refuses_a_server_setting_before_it_runs(server, shape):
    """Refused in ``query``'s own words, and the setting is as it was."""
    table, admin = server
    if shape == "set-persist" and "mariadb" in str(admin.url.drivername):
        pytest.skip("MariaDB has no SET PERSIST")
    before = _connections(admin)
    answer = _call("query", nickname="endpoint", sql=_SETTINGS[shape].format(t=table))
    after = _connections(admin)

    assert answer["ok"] is False, answer
    assert "query reads; it does not write" in answer["error"], answer
    assert after == before, (shape, before, after)


@pytest.mark.endpoint
@pytest.mark.parametrize("into", ["OUTFILE", "DUMPFILE"])
def test_query_refuses_a_file_write_before_it_runs(server, into):
    """No file appears where the server would have written it.

    A server whose ``secure_file_priv`` is NULL writes no file at all
    (MariaDB 11's image, measured), so there the refusal is asserted and the
    file check, which could not fail, is skipped rather than counted.
    """
    table, admin = server
    directory = _file_dir(admin)
    path = f"{(directory or '/tmp/').rstrip('/')}/{table}_{into.lower()}"
    answer = _call("query", nickname="endpoint", sql=f"SELECT 1 INTO {into} '{path}'")

    assert answer["ok"] is False, answer
    assert "query reads; it does not write" in answer["error"], answer
    if directory is None:
        pytest.skip("secure_file_priv is NULL: this server cannot write a file")
    assert not _file_exists(admin, path), path


_READS = {
    "select": "SELECT name FROM {t} ORDER BY name LIMIT 2",
    "trailing-semicolon": "SELECT count(*) FROM {t};",
    "cte": "WITH q AS (SELECT name FROM {t}) SELECT count(*) FROM q",
    "join": "SELECT a.name FROM {t} a JOIN {t} b ON a.salary = b.salary",
    "window": "SELECT name, rank() OVER (ORDER BY salary) FROM {t}",
    "builtin-functions": "SELECT upper(name), concat(name, '!'), round(avg(salary), 1) FROM {t} GROUP BY name",
    "comment-first": "-- top rows\nSELECT name FROM {t} LIMIT 1",
    "show": "SHOW TABLES",
    "explain": "EXPLAIN SELECT * FROM {t}",
}


@pytest.mark.endpoint
@pytest.mark.parametrize("shape", sorted(_READS))
def test_query_still_answers_a_read(server, shape):
    """The gate costs no read that names nothing it refuses."""
    table, _ = server
    read = _call("query", nickname="endpoint", sql=_READS[shape].format(t=table))
    assert read["ok"] is True, read


@pytest.mark.endpoint
def test_directory_and_stats_still_work_through_the_gate(server):
    """The server's own reads on the reader, reflection and aggregates, pass."""
    table, _ = server
    listed = _call("directory", nickname="endpoint")
    stats = _call("stats", nickname="endpoint", table=table)
    assert listed["ok"] is True, listed
    assert stats["ok"] is True, stats
    rows = {entry["table"]: entry["rows"] for entry in listed["contents"]}
    assert rows[table] == 5, rows


@pytest.mark.endpoint
def test_a_view_calling_an_unsafe_function_is_listed_but_not_counted(server):
    """Counting the view would run its function; one view must not sink the listing."""
    table, admin = server
    before = _connections(admin)
    listed = _call("directory", nickname="endpoint")
    detail = _call("directory", nickname="endpoint", table=f"{table}_v")

    assert listed["ok"] is True, listed
    rows = {entry["table"]: entry["rows"] for entry in listed["contents"]}
    assert rows[f"{table}_v"] is None, rows
    assert any(f"{table}_v was not counted" in w for w in listed["warnings"]), listed
    assert detail["ok"] is True and detail["rows"] is None, detail
    assert _connections(admin) == before
