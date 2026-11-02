"""PostgreSQL's reader runs only a statement that cannot act past the read (#124).

A read that never commits undoes what a transaction can undo, and on
PostgreSQL a function can act where a rollback does not reach. Measured on
PostgreSQL 16 as a superuser, through ``query`` on a read-only attach:
``lo_export`` wrote a file on the server, ``pg_reload_conf`` reloaded the
configuration, ``pg_advisory_lock`` left a lock on the pooled connection, and
``lo_import`` answered ``ok`` with an oid the rollback then removed. Opened
``READ ONLY``, the session refuses ``lo_import``, ``nextval`` and ``setval``,
and still runs the rest. YugabyteDB ran ``pg_reload_conf`` the same way;
CockroachDB ran ``setval`` and the sequence stayed moved.

So the reader is read-only *and* gated, in two halves tested here apart:

* **the statement's shape**: one statement, and no ``U&"…"`` identifier,
  whose escapes spell ``lo_export`` without the word (measured: it ran);
* **what it mentions**, decided by the catalog: no function that can act
  past the transaction or read a relation named at run time, no view or
  row-security policy calling one, no operator implemented by one.

The live half needs ``localdata-test-postgres``, ``localdata-test-cockroachdb``,
``localdata-test-yugabytedb`` and ``localdata-test-opengauss``, and skips without them, like every endpoint
test. Each connects as the container's superuser, because a gate tested where
the effect could not have happened anyway proves nothing.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import DDL, create_engine, text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import (
    PostgreSQLBackend,
    _named_as_word,
    _one_statement,
    _pg_refusal,
    _pg_refused_word,
    _pg_shape_refusal,
    _PGFunction,
    _PGUnsafe,
    _reachable_names,
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
        "SELECT 'u&' || 'x'",
        'SELECT "Name" FROM t',
        "SELECT U&'d\\0061t\\+000061'",  # a Unicode-escaped *string* is data
    ],
)
def test_a_read_has_an_acceptable_shape(sql):
    assert _pg_shape_refusal(sql) is None


@pytest.mark.parametrize(
    ("sql", "words"),
    [
        ("SELECT 1; SELECT 2", "more than one statement"),
        ("SELECT 1; COMMIT; SELECT 2", "more than one statement"),
        ("SELECT 1;;", "more than one statement"),
        ("SELECT ';'", "more than one statement"),  # the read this costs, said
        ("SELECT U&\"lo\\005fexport\"(1, '/tmp/x')", "Unicode escapes"),
        ('select u&"x"', "Unicode escapes"),
    ],
)
def test_anything_else_is_refused_by_shape_and_says_why(sql, words):
    refused = _pg_shape_refusal(sql)
    assert refused is not None and words in refused, refused


def test_one_statement_allows_only_a_trailing_semicolon():
    assert _one_statement("SELECT 1 ;\n")
    assert _one_statement("SELECT 1; ")
    assert not _one_statement("SELECT 1; ;")
    assert not _one_statement("SELECT 1;\nSELECT 2")


# ---------------------------------------------------------------------------
# Which functions a read may not reach
# ---------------------------------------------------------------------------


def _fn(schema, name, *, volatile=False, extension=False, on_path=False):
    return _PGFunction(schema, name, volatile, extension, on_path)


_SYSTEM = {"lower", "log", "random", "lo_export", "table_to_xml"}


@pytest.mark.parametrize(
    ("function", "word"),
    [
        # The server's own: volatile is refused unless its answer merely varies.
        (_fn("pg_catalog", "lo_export", volatile=True), "lo_export"),
        (_fn("pg_catalog", "pg_reload_conf", volatile=True), "pg_reload_conf"),
        (_fn("pg_catalog", "set_config", volatile=True), "set_config"),
        (
            _fn("information_schema", "crdb_pin_plan_gist", volatile=True),
            "crdb_pin_plan_gist",
        ),
        (_fn("pg_catalog", "random", volatile=True), None),
        (_fn("pg_catalog", "version", volatile=True), None),
        (_fn("pg_catalog", "pg_relation_size", volatile=True), None),
        (_fn("pg_catalog", "lower"), None),
        # ...and a stable one is refused when it reads what its argument names.
        (_fn("pg_catalog", "table_to_xml"), "table_to_xml"),
        (_fn("pg_catalog", "database_to_xmlschema"), "database_to_xmlschema"),
        # An extension's volatility is taken at its word.
        (_fn("public", "dblink_exec", volatile=True, extension=True), "dblink_exec"),
        (_fn("public", "st_distance", extension=True, on_path=True), None),
        # Anyone else's is refused whatever it declares.
        (_fn("public", "tidy", on_path=True), "tidy"),
        (_fn("public", "lower", on_path=True), "lower"),
        (_fn("crdb_internal", "write_file", volatile=True), "write_file"),
        # Off the path a colliding name can only be reached qualified.
        (_fn("crdb_internal", "log", volatile=True), "crdb_internal"),
    ],
)
def test_each_kind_of_function_is_refused_by_the_right_word(function, word):
    assert _pg_refused_word(function, _SYSTEM) == word


_UNSAFE = _PGUnsafe(
    words={
        "lo_export": "a function",
        "crdb_internal": "a schema",
        "salaries_v": "a view",
    },
    operators={"###": "add_and_log"},
)


@pytest.mark.parametrize(
    ("sql", "words"),
    [
        ("SELECT lo_export(1, '/tmp/x')", "use lo_export"),
        ("SELECT pg_catalog.LO_EXPORT(1, '/tmp/x')", "use lo_export"),
        ("SELECT \"lo_export\"(1, '/tmp/x')", "use lo_export"),
        ("SELECT crdb_internal.log(1)", "use crdb_internal"),
        ("SELECT * FROM salaries_v", "use salaries_v"),
        ("SELECT 'lo_export'", "use lo_export"),  # the read this costs, said
        ("SELECT 1 ### 2", "operator ###"),
    ],
)
def test_a_statement_mentioning_one_is_refused_and_says_which(sql, words):
    refused = _pg_refusal(sql, _UNSAFE)
    assert refused is not None and words in refused, refused


@pytest.mark.parametrize(
    "sql",
    ["SELECT log(10, 100)", "SELECT lo_exported FROM t", "SELECT 1 ## 2"],
)
def test_a_statement_mentioning_none_is_not(sql):
    assert _pg_refusal(sql, _UNSAFE) is None


# ---------------------------------------------------------------------------
# The word match both gates share, which runs over every catalog view
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT zeta(1), Beta(2), alpha_x FROM t",  # not the first mentioned
        "SELECT Beta(2), zeta(1), alpha_x FROM t",  # nor the last
    ],
)
def test_of_several_names_mentioned_the_first_in_sorted_order_is_named(sql):
    assert _named_as_word(sql, {"zeta", "beta", "alpha", "gamma"}) == "beta"


@pytest.mark.parametrize(
    ("sql", "names", "found"),
    [
        ("SELECT LO_EXPORT(1)", {"lo_export"}, "lo_export"),  # case-insensitive
        ("SELECT a$b FROM t", {"a"}, None),  # `$` is part of a word
        ("SELECT a$b FROM t", {"a$b"}, "a$b"),
        ('SELECT * FROM "my view"', {"my view"}, "my view"),  # not a plain word
        ('SELECT * FROM "my viewer"', {"my view"}, None),
        ("SELECT x.f(1)", {"f", "my view"}, "f"),
        # Unicode case-folding, as re.IGNORECASE has it: the Kelvin sign is a k.
        ("SELECT K(1)", {"k"}, "k"),
        ("SELECT set(1)", {"ſet"}, "ſet"),  # a long s is an s
        ("SELECT été(1)", {"ÉTÉ"}, "ÉTÉ"),
        ("SELECT 1", set(), None),
    ],
)
def test_a_name_is_matched_only_as_a_whole_word(sql, names, found):
    assert _named_as_word(sql, names) == found


def test_reachability_follows_views_through_views_whatever_their_order():
    views = {"c": "SELECT * FROM b", "b": "SELECT * FROM a", "a": "SELECT f(1)"}
    assert _reachable_names({"f"}, views) == {"f", "a", "b", "c"}


# ---------------------------------------------------------------------------
# Who gets the gate, and how its refusal is recognised
# ---------------------------------------------------------------------------

_YUGABYTEDB_BANNER = "PostgreSQL 15.12-YB-2.25.2.0-b0 on x86_64-pc-linux-gnu"


def test_postgresql_cockroachdb_and_yugabytedb_share_the_backend_and_keep_their_names():
    resolved = {
        "postgresql": backend_for("postgresql"),
        "cockroachdb": backend_for("cockroachdb"),
        "yugabytedb": backend_for("postgresql").named_by(_YUGABYTEDB_BANNER),
    }
    for name, backend in resolved.items():
        assert isinstance(backend, PostgreSQLBackend), name
        assert backend.name == name


def test_opengauss_gets_the_gate_and_asks_its_own_policy_catalog():
    """openGauss has no ``pg_policies``; asking for it would fail every read."""
    backend = backend_for("opengauss")

    assert isinstance(backend, PostgreSQLBackend)
    assert backend.name == "opengauss"
    assert "pg_rlspolicies" in backend.relations_query
    assert "pg_policies" not in backend.relations_query
    assert "pg_policies" in backend_for("postgresql").relations_query


@pytest.mark.parametrize("attribute", ["sqlstate", "pgcode"])
def test_a_read_only_transaction_refusal_is_recognised_under_either_driver(attribute):
    """psycopg 3 carries the code as ``sqlstate``, psycopg 2 as ``pgcode``."""
    backend = backend_for("postgresql")
    refused = SimpleNamespace(orig=SimpleNamespace(**{attribute: "25006"}))
    other = SimpleNamespace(orig=SimpleNamespace(**{attribute: "42P01"}))

    assert backend.denies_write(refused) is True
    assert backend.denies_write(other) is False


# ---------------------------------------------------------------------------
# Live, against PostgreSQL, CockroachDB, YugabyteDB and openGauss
# ---------------------------------------------------------------------------

_SERVERS = [
    target
    for target in TARGETS
    if target.endpoint.dialect in ("postgresql", "cockroachdb", "opengauss")
    and target.auth is None
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


def _engine_of(admin) -> str:
    with admin.connect() as conn:
        banner = conn.execute(text("SELECT version()")).scalar()
    if "CockroachDB" in banner:
        return "cockroachdb"
    if "openGauss" in banner:
        return "opengauss"
    return "yugabytedb" if "-YB-" in banner else "postgresql"


def _try(admin, statement: str) -> bool:
    """Run ``statement`` as the administrator, saying whether this engine took it."""
    try:
        with admin.begin() as conn:
            conn.execute(DDL(statement))
    except Exception:  # noqa: BLE001 - which engines support a feature is the question
        return False
    return True


@pytest.fixture(params=_SERVERS, ids=lambda target: target.name)
def server(request, tmp_path):
    """A read-only superuser slot over a five-row table, and what the test needs.

    Yields a namespace: ``t`` the table, ``admin`` an engine outside the
    server to read back what a refused statement would have changed,
    ``engine`` which of the four answered, and ``features``, the set of
    fixtures this engine accepted (operators and row security are not
    everywhere).
    """
    try:
        reached = reach(request.param)
    except Unavailable as exc:
        pytest.skip(str(exc))
    admin = create_engine(reached.url)
    t = f"t{uuid.uuid4().hex[:8]}"
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    features: set[str] = set()
    try:
        with admin.begin() as conn:
            conn.execute(DDL(f"CREATE TABLE {t} (name VARCHAR(20), salary INT)"))
            conn.execute(
                text(f"INSERT INTO {t} VALUES ('a',1),('b',2),('c',3),('d',4),('e',5)")
            )
            conn.execute(DDL(f"CREATE SEQUENCE {t}_seq"))
            # A function of nobody's in particular, which no verb can create,
            # and a view hiding a call to it behind a relation's name.
            conn.execute(
                DDL(
                    f"CREATE FUNCTION {t}_f(a INT, b INT) RETURNS INT "
                    f"LANGUAGE SQL AS 'SELECT a + b'"
                )
            )
            conn.execute(DDL(f"CREATE VIEW {t}_v AS SELECT {t}_f(1, 2) AS x"))
            # Off the search path, under a name the server also uses.
            conn.execute(DDL(f"CREATE SCHEMA {t}_s"))
            conn.execute(
                DDL(
                    f"CREATE FUNCTION {t}_s.lower(a TEXT) RETURNS TEXT "
                    f"LANGUAGE SQL AS 'SELECT a'"
                )
            )
        if _try(
            admin,
            f"CREATE OPERATOR ### (LEFTARG = INT, RIGHTARG = INT, FUNCTION = {t}_f)",
        ):
            features.add("operator")
        if _try(admin, f"CREATE TABLE {t}_guarded (salary INT)") and _try(
            admin, f"CREATE POLICY {t}_p ON {t}_guarded USING ({t}_f(salary, 0) > 0)"
        ):
            features.add("policy")
        with applied(reached.environ):
            attached = _call("attach", database=reached.url, nickname="endpoint")
            assert attached["ok"] is True, attached
            yield SimpleNamespace(
                t=t, admin=admin, engine=_engine_of(admin), features=features
            )
    finally:
        server_module._reset()
        for cleanup in [
            "DROP OPERATOR IF EXISTS ### (INT, INT)",
            f"DROP TABLE IF EXISTS {t}_guarded",
            f"DROP VIEW IF EXISTS {t}_v",
            f"DROP FUNCTION IF EXISTS {t}_f(INT, INT)",
            f"DROP FUNCTION IF EXISTS {t}_s.lower(TEXT)",
            f"DROP SCHEMA IF EXISTS {t}_s",
            f"DROP SEQUENCE IF EXISTS {t}_seq",
            f"DROP TABLE IF EXISTS {t}",
        ]:
            _try(admin, cleanup)
        admin.dispose()


def _scalar(admin, sql: str):
    with admin.connect() as conn:
        return conn.execute(text(sql)).scalar()


def _refused(answer) -> None:
    assert answer["ok"] is False, answer
    assert "query reads; it does not write" in answer["error"], answer


_REFUSED_EVERYWHERE = {
    "user-function": "SELECT {t}_f(1, 2)",
    "view-calling-it": "SELECT * FROM {t}_v",
    "qualified-shadow": "SELECT {t}_s.lower('A')",
    "set-config": "SELECT set_config('statement_timeout', '1', false)",
    "query-to-xml": "SELECT query_to_xml('SELECT 1', true, false, '')",
    "table-to-xml": "SELECT table_to_xml('{t}', true, false, '')",
    "unicode-escape": 'SELECT U&"{t}\\005ff"(1, 2)',
    "two-statements": "SELECT 1; SELECT {t}_f(1, 2)",
}

#: Shapes calling a function an engine does not have, measured: CockroachDB
#: 26.3 answers "unknown function", so there is nothing there to refuse, and
#: openGauss 7.0.0-RC3 has no ``gen_random_uuid``, so nothing there to read.
_ABSENT = {
    "cockroachdb": {"query-to-xml", "table-to-xml"},
    "opengauss": {"uuid"},
}


@pytest.mark.endpoint
@pytest.mark.parametrize("shape", sorted(_REFUSED_EVERYWHERE))
def test_query_refuses_what_could_act_past_the_read(server, shape):
    if shape in _ABSENT.get(server.engine, ()):
        pytest.skip(f"{server.engine} has no such function")
    answer = _call(
        "query", nickname="endpoint", sql=_REFUSED_EVERYWHERE[shape].format(t=server.t)
    )
    _refused(answer)


@pytest.mark.endpoint
def test_a_sequence_is_not_moved(server):
    """CockroachDB ran ``setval`` under a read-only session; the sequence stayed moved."""
    before = _scalar(server.admin, f"SELECT last_value FROM {server.t}_seq")
    for sql in (
        f"SELECT setval('{server.t}_seq', 50)",
        f"SELECT nextval('{server.t}_seq')",
    ):
        _refused(_call("query", nickname="endpoint", sql=sql))
    assert _scalar(server.admin, f"SELECT last_value FROM {server.t}_seq") == before


@pytest.mark.endpoint
def test_the_configuration_is_not_reloaded(server):
    if server.engine == "cockroachdb":
        pytest.skip("CockroachDB has no pg_reload_conf")
    before = _scalar(server.admin, "SELECT pg_conf_load_time()")
    _refused(_call("query", nickname="endpoint", sql="SELECT pg_reload_conf()"))
    assert _scalar(server.admin, "SELECT pg_conf_load_time()") == before


#: Engines with session advisory locks, measured: YugabyteDB answers 0A000
#: ("in preview"), CockroachDB has none. openGauss left one held before #124.
_ADVISORY_LOCKS = {"postgresql", "opengauss"}


@pytest.mark.endpoint
def test_no_file_is_written(server):
    """``lo_export`` writes a server file no rollback removes."""
    if server.engine != "postgresql":
        pytest.skip(f"{server.engine}: large objects not measured here")
    path = f"/tmp/{server.t}_lo"
    with server.admin.begin() as conn:  # committed, or the object is gone again
        oid = conn.execute(text("SELECT lo_from_bytea(0, 'x'::bytea)")).scalar()
    try:
        _refused(
            _call(
                "query", nickname="endpoint", sql=f"SELECT lo_export({oid}, '{path}')"
            )
        )
        assert _scalar(server.admin, f"SELECT pg_stat_file('{path}', true)") is None
    finally:
        with server.admin.begin() as conn:
            conn.execute(text(f"SELECT lo_unlink({oid})"))


@pytest.mark.endpoint
def test_no_lock_is_left(server):
    """A session advisory lock outlives the rollback on the pooled connection."""
    if server.engine not in _ADVISORY_LOCKS:
        pytest.skip(f"{server.engine} has no session advisory locks")
    _refused(_call("query", nickname="endpoint", sql="SELECT pg_advisory_lock(124)"))
    assert _scalar(server.admin, "SELECT count(*) FROM pg_locks WHERE objid = 124") == 0


@pytest.mark.endpoint
def test_an_operator_calling_a_refused_function_is_refused(server):
    if "operator" not in server.features:
        pytest.skip(f"{server.engine} has no CREATE OPERATOR")
    _refused(_call("query", nickname="endpoint", sql="SELECT 1 ### 2"))


@pytest.mark.endpoint
def test_a_table_whose_row_policy_calls_one_is_refused(server):
    if "policy" not in server.features:
        pytest.skip(f"{server.engine} has no CREATE POLICY")
    _refused(
        _call("query", nickname="endpoint", sql=f"SELECT * FROM {server.t}_guarded")
    )


@pytest.mark.endpoint
def test_a_write_in_a_cte_changes_nothing(server):
    """Whichever refuses it, the server or the gate, the table is as it was."""
    answer = _call(
        "query",
        nickname="endpoint",
        sql=f"WITH w AS (INSERT INTO {server.t} VALUES ('z', 9) RETURNING 1) "
        f"SELECT * FROM w",
    )
    assert answer["ok"] is False, answer
    assert _scalar(server.admin, f"SELECT count(*) FROM {server.t}") == 5


@pytest.mark.endpoint
def test_a_trigger_cannot_act_past_the_read(server):
    """A write fires functions the statement never names, so the gate cannot see them.

    What stops this one is the server-side cursor ``query`` streams through:
    PostgreSQL answers "DECLARE CURSOR must not contain data-modifying
    statements in WITH" before the ``INSERT`` runs, measured. Pinned because a
    trigger's session lock would outlive the rollback on the pooled
    connection, as ``pg_advisory_lock`` did when called by name.
    """
    if server.engine not in _ADVISORY_LOCKS:
        pytest.skip(f"{server.engine} has no session advisory locks")
    t = server.t
    with server.admin.begin() as conn:
        conn.execute(
            DDL(
                f"CREATE FUNCTION {t}_trg() RETURNS trigger LANGUAGE plpgsql AS "
                f"'BEGIN PERFORM pg_advisory_lock(125); RETURN NEW; END'"
            )
        )
        conn.execute(
            DDL(
                f"CREATE TRIGGER {t}_on_insert BEFORE INSERT ON {t} "
                f"FOR EACH ROW EXECUTE FUNCTION {t}_trg()"
            )
        )
    try:
        answer = _call(
            "query",
            nickname="endpoint",
            sql=f"WITH w AS (INSERT INTO {t} VALUES ('z', 9) RETURNING 1) "
            f"SELECT * FROM w",
        )
        assert answer["ok"] is False, answer
        assert (
            _scalar(server.admin, "SELECT count(*) FROM pg_locks WHERE objid = 125")
            == 0
        )
    finally:
        _try(server.admin, f"DROP TRIGGER IF EXISTS {t}_on_insert ON {t}")
        _try(server.admin, f"DROP FUNCTION IF EXISTS {t}_trg()")


@pytest.mark.endpoint
def test_a_function_called_without_its_name_cannot_move_a_sequence(server, request):
    """What the read-only session is for: a call the gate cannot see.

    A cast runs its function without the statement naming it, so the word
    match passes ``SELECT 1::<type>``. The function calls ``nextval``, whose
    effect a rollback does not undo. The read-only session refuses
    ``nextval`` wherever it is called from, which is the only thing between
    this statement and a moved sequence.

    **Not on openGauss**, whose read-only session runs ``nextval``, measured
    directly as well as through ``query`` (the sequence moved). There nothing
    stands between this statement and the sequence: the residual CONSTRAINTS
    §34 records (#125), pinned strict so that closing it turns this red.
    """
    if server.engine == "opengauss":
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="#125: openGauss's read-only session runs nextval, and an "
                "implicit call is the recorded residual",
            )
        )
    t = server.t
    if not (
        _try(server.admin, f"CREATE TYPE {t}_c AS (a BIGINT)")
        and _try(
            server.admin,
            f"CREATE FUNCTION {t}_cast(INT) RETURNS {t}_c LANGUAGE SQL AS "
            f"'SELECT ROW(nextval(''{t}_seq''))::{t}_c'",
        )
        and _try(
            server.admin,
            f"CREATE CAST (INT AS {t}_c) WITH FUNCTION {t}_cast(INT)",
        )
    ):
        _try(server.admin, f"DROP FUNCTION IF EXISTS {t}_cast(INT)")
        _try(server.admin, f"DROP TYPE IF EXISTS {t}_c")
        pytest.skip(f"{server.engine} has no user-defined casts")
    try:
        before = _scalar(server.admin, f"SELECT last_value FROM {t}_seq")
        answer = _call("query", nickname="endpoint", sql=f"SELECT (1::{t}_c).a")
        assert answer["ok"] is False, answer
        assert _scalar(server.admin, f"SELECT last_value FROM {t}_seq") == before
    finally:
        _try(server.admin, f"DROP CAST IF EXISTS (INT AS {t}_c)")
        _try(server.admin, f"DROP FUNCTION IF EXISTS {t}_cast(INT)")
        _try(server.admin, f"DROP TYPE IF EXISTS {t}_c")


_READS = {
    "select": "SELECT name FROM {t} ORDER BY name LIMIT 2",
    "trailing-semicolon": "SELECT count(*) FROM {t};",
    "cte": "WITH q AS (SELECT name FROM {t}) SELECT count(*) FROM q",
    "join": "SELECT a.name FROM {t} a JOIN {t} b ON a.salary = b.salary",
    "window": "SELECT name, rank() OVER (ORDER BY salary) FROM {t}",
    "builtins": "SELECT upper(name), lower(name), log(10.0, salary + 9.0), "
    "round(avg(salary), 1) FROM {t} GROUP BY name, salary",
    "answer-varies": "SELECT random() < 2, clock_timestamp() IS NOT NULL, "
    "version() IS NOT NULL",
    "uuid": "SELECT gen_random_uuid() IS NOT NULL",
    "size": "SELECT pg_relation_size('{t}'::regclass) >= 0",
    "comment-first": "-- top rows\nSELECT name FROM {t} LIMIT 1",
    "catalog": "SELECT count(*) FROM information_schema.tables",
}


@pytest.mark.endpoint
@pytest.mark.parametrize("shape", sorted(_READS))
def test_query_still_answers_a_read(server, shape):
    """The gate costs no read that names nothing it refuses."""
    if shape in _ABSENT.get(server.engine, ()):
        pytest.skip(f"{server.engine} has no such function")
    read = _call("query", nickname="endpoint", sql=_READS[shape].format(t=server.t))
    assert read["ok"] is True, read


@pytest.mark.endpoint
def test_directory_and_stats_still_work_through_the_gate(server):
    """The server's own reads on the reader, reflection and aggregates, pass."""
    listed = _call("directory", nickname="endpoint")
    stats = _call("stats", nickname="endpoint", table=server.t)
    assert listed["ok"] is True, listed
    assert stats["ok"] is True, stats
    rows = {entry["table"]: entry["rows"] for entry in listed["contents"]}
    assert rows[server.t] == 5, rows


@pytest.mark.endpoint
def test_a_view_calling_a_refused_function_is_listed_but_not_counted(server):
    """Counting the view would run its function; one view must not sink the listing."""
    listed = _call("directory", nickname="endpoint")
    detail = _call("directory", nickname="endpoint", table=f"{server.t}_v")

    assert listed["ok"] is True, listed
    rows = {entry["table"]: entry["rows"] for entry in listed["contents"]}
    assert rows[f"{server.t}_v"] is None, rows
    assert any(f"{server.t}_v was not counted" in w for w in listed["warnings"]), listed
    assert detail["ok"] is True and detail["rows"] is None, detail
