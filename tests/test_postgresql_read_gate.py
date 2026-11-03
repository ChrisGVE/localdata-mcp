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
from localdata_mcp import dialects as dialects_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import (
    PostgreSQLBackend,
    _named_as_word,
    _one_statement,
    _pg_refusal,
    _pg_refused_types,
    _pg_refused_word,
    _pg_shape_refusal,
    _PGFunction,
    _PGType,
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
        ("SELECT 1; SELECT 2", "put a ';' before its end"),
        ("SELECT 1; COMMIT; SELECT 2", "put a ';' before its end"),
        ("SELECT 1;;", "put a ';' before its end"),
        ("SELECT ';'", "put a ';' before its end"),  # the read this costs, said
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
# Types whose conversion calls a refused function (#125)
# ---------------------------------------------------------------------------


def _type(name, *, schema="public", on_path=True, collides=False, relation=False):
    return _PGType(schema, name, on_path, collides, relation)


#: oid -> type: a composite converted by a refused cast (1), an array of it
#: (2), a domain over it (3), a table with a column of it (4), a type nesting
#: the table's row (5), a view over nothing of the sort (6), and a table of
#: plain columns (7).
_TYPES = {
    1: _type("money_c"),
    2: _type("_money_c"),
    3: _type("money_d"),
    4: _type("ledger", relation=True),
    5: _type("audit_c"),
    6: _type("plain_v", relation=True),
    7: _type("plain_t", relation=True),
}
_PARTS = [(2, 1), (3, 1), (4, 1), (5, 4), (7, 23), (6, 25)]


def test_a_type_converted_by_a_refused_function_is_refused_with_all_it_holds():
    refused = _pg_refused_types({1}, _TYPES, _PARTS)
    assert set(refused) == {"money_c", "_money_c", "money_d", "ledger", "audit_c"}
    assert "column" in refused["ledger"]
    assert "column" not in refused["money_c"]


def test_no_cast_to_refuse_means_no_type_is_refused():
    assert _pg_refused_types(set(), _TYPES, _PARTS) == {}


def test_a_type_off_the_search_path_named_like_a_system_one_costs_its_schema():
    types = {1: _type("text", schema="vault", on_path=False, collides=True)}
    assert set(_pg_refused_types({1}, types, [])) == {"vault"}


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


#: Reads of the server's own files (#36, class B). Each is volatile, so the
#: name rule refuses it; ``PG_VERSION`` and ``.`` are in every data directory.
_SERVER_FILE_READS = {
    "pg_read_file": "SELECT pg_read_file('PG_VERSION')",
    "pg_ls_dir": "SELECT * FROM pg_ls_dir('.')",
    "pg_stat_file": "SELECT (pg_stat_file('PG_VERSION')).size",
}


@pytest.mark.endpoint
@pytest.mark.parametrize("function", sorted(_SERVER_FILE_READS))
def test_no_server_file_is_read(server, function):
    """A superuser could read the server's files through ``query``; the gate names the call."""
    if server.engine == "cockroachdb":
        pytest.skip("CockroachDB has no server-file functions")
    answer = _call("query", nickname="endpoint", sql=_SERVER_FILE_READS[function])
    _refused(answer)
    assert function in answer["error"], answer


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
def test_a_function_called_without_its_name_cannot_move_a_sequence(server):
    """A cast runs its function without the statement naming the function.

    ``SELECT 1::<type>`` names only the type; the cast's function calls
    ``nextval``, whose effect a rollback does not undo. On openGauss, whose
    read-only session runs ``nextval`` (measured, the sequence moved), this
    was the residual #125 recorded; since the gate refuses a type converted
    by a refused function, it is refused there too. A cast between two of the
    server's own types, which no word names, is
    ``test_a_cast_between_the_servers_own_types_is_refused_by_the_gate``.
    """
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


def _sequence_moves_from_here(admin, t: str):
    """Call ``nextval`` once as the administrator, so any later call moves ``last_value``.

    A fresh sequence's first ``nextval`` returns its start and leaves
    ``last_value`` where it was, which would hide exactly the call a test
    looks for.
    """
    with admin.begin() as conn:
        conn.execute(text(f"SELECT nextval('{t}_seq')"))
    return _scalar(admin, f"SELECT last_value FROM {t}_seq")


#: Reads reaching a cast whose function calls ``nextval`` (#125). The first
#: names the type; the others name only a table with a column of it, and the
#: cast runs implicitly — measured on PostgreSQL and openGauss, each moved the
#: sequence before the gate knew about casts.
_CAST_REACHED = {
    "explicit": "SELECT (1::{t}_k).a",
    "implicit-arithmetic": "SELECT c + 1 FROM {t}_kt",
    "implicit-comparison": "SELECT count(*) FROM {t}_kt WHERE c = 1",
}


@pytest.mark.endpoint
@pytest.mark.parametrize("shape", sorted(_CAST_REACHED))
def test_a_read_reaching_a_refused_cast_is_refused_by_the_gate(server, shape):
    """Refused before it is sent, so openGauss's read-only session is not asked."""
    t = server.t
    created = [
        f"CREATE TYPE {t}_k AS (a BIGINT)",
        (
            f"CREATE FUNCTION {t}_kin(INT) RETURNS {t}_k LANGUAGE SQL AS "
            f"'SELECT ROW(nextval(''{t}_seq''))::{t}_k'"
        ),
        f"CREATE CAST (INT AS {t}_k) WITH FUNCTION {t}_kin(INT)",
        (
            f"CREATE FUNCTION {t}_kout({t}_k) RETURNS INT LANGUAGE SQL AS "
            f"'SELECT nextval(''{t}_seq'')::INT'"
        ),
        f"CREATE CAST ({t}_k AS INT) WITH FUNCTION {t}_kout({t}_k) AS IMPLICIT",
        f"CREATE TABLE {t}_kt (c {t}_k)",
    ]
    try:
        if not all(_try(server.admin, statement) for statement in created):
            pytest.skip(f"{server.engine} has no user-defined casts")
        with server.admin.begin() as conn:
            conn.execute(text(f"INSERT INTO {t}_kt VALUES (ROW(1))"))
        before = _sequence_moves_from_here(server.admin, t)
        answer = _call(
            "query", nickname="endpoint", sql=_CAST_REACHED[shape].format(t=t)
        )
        _refused(answer)
        assert "converts through such a function" in answer["error"] or (
            "conversion calls such a function" in answer["error"]
        ), answer
        assert _scalar(server.admin, f"SELECT last_value FROM {t}_seq") == before
    finally:
        for cleanup in [
            f"DROP TABLE IF EXISTS {t}_kt",
            f"DROP CAST IF EXISTS ({t}_k AS INT)",
            f"DROP CAST IF EXISTS (INT AS {t}_k)",
            f"DROP FUNCTION IF EXISTS {t}_kout({t}_k)",
            f"DROP FUNCTION IF EXISTS {t}_kin(INT)",
            f"DROP TYPE IF EXISTS {t}_k",
        ]:
            _try(server.admin, cleanup)


@pytest.mark.endpoint
def test_a_domain_check_cannot_move_a_sequence(server):
    """A domain's ``CHECK`` runs its function without the statement naming it.

    ``SELECT 1::<domain>`` names only the domain; its ``CHECK`` calls a
    function that calls ``nextval``, whose effect a rollback does not undo.
    On openGauss, whose read-only session runs ``nextval`` (measured, the
    sequence moved), this was the residual #126 recorded; since the gate
    refuses a domain whose check reaches a refused function, it is refused
    there too. A cast between two of the server's own types, which no word
    names, is ``test_a_cast_between_the_servers_own_types_is_refused_by_the_gate``.
    The cleanup drops the domain with ``DROP TYPE`` where openGauss refuses
    ``DROP DOMAIN`` (#130).
    """
    t = server.t
    try:
        if not (
            _try(
                server.admin,
                f"CREATE FUNCTION {t}_chk(INT) RETURNS BOOLEAN LANGUAGE SQL AS "
                f"'SELECT nextval(''{t}_seq'') > 0'",
            )
            and _try(server.admin, f"CREATE DOMAIN {t}_d AS INT CHECK ({t}_chk(VALUE))")
        ):
            pytest.skip(f"{server.engine} has no domains with a function check")
        before = _sequence_moves_from_here(server.admin, t)
        answer = _call("query", nickname="endpoint", sql=f"SELECT 1::{t}_d")
        assert answer["ok"] is False, answer
        assert _scalar(server.admin, f"SELECT last_value FROM {t}_seq") == before
    finally:
        _drop_domain(server.admin, f"{t}_d")
        _try(server.admin, f"DROP FUNCTION IF EXISTS {t}_chk(INT)")


# ---------------------------------------------------------------------------
# Domains whose CHECK reaches a refused name (#126)
# ---------------------------------------------------------------------------


def test_a_domain_whose_check_reaches_one_is_refused_with_all_it_holds():
    """A domain checked, its array, a domain over it, a composite, a table: all refused."""
    types = {
        10: _type("d_fn"),
        11: _type("_d_fn"),
        12: _type("d_over"),
        13: _type("pair_c"),
        14: _type("ledger", relation=True),
        15: _type("plain_t", relation=True),
    }
    parts = [(11, 10), (12, 10), (13, 10), (14, 13), (15, 23)]

    refused = dialects_module._pg_refused_types(set(), types, parts, {10})

    assert set(refused) == {"d_fn", "_d_fn", "d_over", "pair_c", "ledger"}
    assert "domain whose CHECK" in refused["d_fn"]
    assert "column" in refused["ledger"] and "CHECK" in refused["ledger"]
    assert "CHECK" in refused["pair_c"] and "column" not in refused["pair_c"]


def test_a_type_reached_by_a_cast_and_a_check_is_explained_by_its_cast():
    types = {1: _type("money_c"), 2: _type("both_d")}
    refused = dialects_module._pg_refused_types({1}, types, [(2, 1)], {2})
    assert "conversion" in refused["both_d"], refused


class _Catalog:
    """A cursor answering the gate's catalog questions from fixed rows.

    Keyed by the exact question, and any other question fails the test: a gate
    that started asking something new would otherwise read an empty answer
    and pass. ``asked`` records the questions in order, for what a server
    pays per statement.
    """

    def __init__(self, answers):
        self.answers = answers
        self.asked = []
        self._rows = []

    def execute(self, sql):
        assert sql in self.answers, f"unexpected catalog question: {sql[:80]}"
        self.asked.append(sql)
        self._rows = list(self.answers[sql])

    def fetchall(self):
        return self._rows


_RELATIONS = "the backend's relations question"


def _catalog(*, checks, types=(), parts=(), views=(), casts=()):
    """A catalog with a user function ``chk``, and ``nextval`` and ``abs`` of the server's."""
    return _Catalog(
        {
            dialects_module._PG_FUNCTIONS: [
                ("public", "chk", False, False, True),
                ("pg_catalog", "nextval", True, False, False),
                ("pg_catalog", "abs", False, False, False),
            ],
            dialects_module._PG_CASTS: list(casts),
            dialects_module._PG_DOMAIN_CHECKS: list(checks),
            dialects_module._PG_TYPES: list(types),
            dialects_module._PG_TYPE_PARTS: list(parts),
            _RELATIONS: list(views),
            dialects_module._PG_OPERATORS: [],
        }
    )


def _domain(oid, name, check):
    return (oid, "public", name, True, False, check)


def _row(oid, name, *, relation=False):
    return (oid, "public", name, True, False, relation)


def test_the_gate_refuses_a_domain_by_what_its_check_reaches():
    """Through a function, through ``nextval``, through another such domain, in any order.

    And what follows from a refused domain: its array, a domain over it, a
    table with a column of it, a view naming it, and a view naming only that
    table's row, which is refused only once the domain's closure is known.
    What reaches nothing stays readable.
    """
    catalog = _catalog(
        checks=[
            _domain(13, "d_outer", "CHECK (((VALUE)::d_fn IS NOT NULL))"),
            _domain(10, "d_fn", "CHECK (chk(VALUE))"),
            _domain(11, "d_seq", "CHECK ((nextval('s'::regclass) > VALUE))"),
            _domain(12, "d_plain", "CHECK ((abs(VALUE) > 0))"),
        ],
        types=[
            _row(10, "d_fn"),
            _row(11, "d_seq"),
            _row(12, "d_plain"),
            _row(13, "d_outer"),
            _row(14, "_d_fn"),
            _row(15, "d_over"),
            _row(16, "ledger", relation=True),
            _row(17, "plain_t", relation=True),
        ],
        parts=[(14, 10), (15, 10), (16, 11), (17, 12)],
        views=[
            ("v_cast", "SELECT (1)::d_fn AS x"),
            ("v_row", "SELECT (ROW(1)::ledger).c AS x"),
            ("v_plain", "SELECT 1 AS x"),
        ],
    )

    words = dialects_module._pg_unsafe(catalog, _RELATIONS).words

    refused_here = ("d_fn", "d_seq", "d_outer", "_d_fn", "d_over", "ledger")
    for refused in (*refused_here, "v_cast", "v_row"):
        assert refused in words, (refused, words)
    for readable in ("d_plain", "plain_t", "v_plain", "abs"):
        assert readable not in words, (readable, words)
    assert "domain whose CHECK" in words["d_outer"], words
    assert "column of a domain" in words["ledger"], words
    assert "view" in words["v_cast"], words


def test_a_domain_named_by_a_refused_type_is_refused():
    """A check casting to a type a refused cast converts reaches that cast (#125)."""
    catalog = _catalog(
        casts=[(1, "public", "conv", False, False, True)],
        checks=[_domain(10, "d_money", "CHECK (((VALUE)::money_c IS NOT NULL))")],
        types=[_row(1, "money_c"), _row(10, "d_money")],
    )

    words = dialects_module._pg_unsafe(catalog, _RELATIONS).words

    assert "domain whose CHECK" in words["d_money"], words


def test_checks_reaching_nothing_cost_no_read_of_the_types():
    """The gate runs on every statement: without a cast or check to follow, no type is read."""
    catalog = _catalog(checks=[_domain(12, "d_plain", "CHECK ((VALUE > 0))")])

    words = dialects_module._pg_unsafe(catalog, _RELATIONS).words

    assert "d_plain" not in words
    assert dialects_module._PG_DOMAIN_CHECKS in catalog.asked
    assert dialects_module._PG_TYPES not in catalog.asked
    assert dialects_module._PG_TYPE_PARTS not in catalog.asked


def _drop_domain(admin, name: str) -> None:
    """Drop domain ``name``; openGauss refuses DROP DOMAIN, measured, and takes DROP TYPE."""
    if not _try(admin, f"DROP DOMAIN IF EXISTS {name}"):
        _try(admin, f"DROP TYPE IF EXISTS {name}")


#: Reads reaching a domain's CHECK that calls ``nextval`` (#126): directly
#: (``_dn``) or through a function (``_dd``), and through what holds a value
#: of it — a domain over it, its array, a composite and a table with a column
#: of it. Each statement names only the type it builds.
_DOMAIN_REACHED = {
    "function-check": "SELECT 1::{t}_dd",
    "nextval-check": "SELECT 1::{t}_dn",
    "domain-over-it": "SELECT 1::{t}_dover",
    "array-of-it": "SELECT '{{1}}'::_{t}_dd",
    "composite-holding-it": "SELECT (ROW(1)::{t}_dc).a",
    "table-holding-it": "SELECT (ROW(1)::{t}_dt).c",
}


@pytest.mark.endpoint
@pytest.mark.parametrize("shape", sorted(_DOMAIN_REACHED))
def test_a_read_reaching_a_refused_domain_check_is_refused_by_the_gate(server, shape):
    """Refused by the gate, by name, so openGauss's read-only session is never asked."""
    t = server.t
    created = [
        (
            f"CREATE FUNCTION {t}_dchk(INT) RETURNS BOOLEAN LANGUAGE SQL AS "
            f"'SELECT nextval(''{t}_seq'') > 0'"
        ),
        f"CREATE DOMAIN {t}_dd AS INT CHECK ({t}_dchk(VALUE))",
        f"CREATE DOMAIN {t}_dn AS BIGINT CHECK (nextval('{t}_seq') > VALUE - 1000000)",
        f"CREATE DOMAIN {t}_dover AS {t}_dd",
        f"CREATE TYPE {t}_dc AS (a {t}_dd)",
        f"CREATE TABLE {t}_dt (c {t}_dd)",
    ]
    try:
        if not all(_try(server.admin, statement) for statement in created):
            pytest.skip(f"{server.engine} has no domains with a function check")
        if shape == "array-of-it" and not _scalar(
            server.admin, f"SELECT count(*) FROM pg_type WHERE typname = '_{t}_dd'"
        ):
            pytest.skip(f"{server.engine} makes no array type for a domain")
        before = _sequence_moves_from_here(server.admin, t)
        answer = _call(
            "query", nickname="endpoint", sql=_DOMAIN_REACHED[shape].format(t=t)
        )
        _refused(answer)
        assert "CHECK" in answer["error"], answer
        assert _scalar(server.admin, f"SELECT last_value FROM {t}_seq") == before
    finally:
        _try(server.admin, f"DROP TABLE IF EXISTS {t}_dt")
        _try(server.admin, f"DROP TYPE IF EXISTS {t}_dc")
        for domain in (f"{t}_dover", f"{t}_dd", f"{t}_dn"):
            _drop_domain(server.admin, domain)
        _try(server.admin, f"DROP FUNCTION IF EXISTS {t}_dchk(INT)")


@pytest.mark.endpoint
def test_a_domain_whose_check_reaches_nothing_still_reads(server):
    """The gate costs no read of a domain whose check calls nothing it refuses."""
    t = server.t
    try:
        if not _try(server.admin, f"CREATE DOMAIN {t}_dok AS INT CHECK (VALUE > 0)"):
            pytest.skip(f"{server.engine} has no domains")
        answer = _call("query", nickname="endpoint", sql=f"SELECT 1::{t}_dok AS x")
        assert answer["ok"] is True, answer
    finally:
        _drop_domain(server.admin, f"{t}_dok")


@pytest.mark.endpoint
def test_a_cast_between_the_servers_own_types_is_refused_by_the_gate(server):
    """A cast no word names fails the gate closed (#129).

    ``CREATE CAST (INT AS BYTEA)`` through a function of the administrator's
    that calls ``nextval``: ``SELECT 1::bytea`` then names only the server's
    own types, which no word check can refuse without refusing every read of
    them. On openGauss, whose read-only session runs ``nextval``, it moved
    the sequence. So the gate refuses every statement on a server holding
    such a cast, naming it, before the read-only session is asked — on
    PostgreSQL and YugabyteDB too, where the session alone used to stop it.
    """
    t = server.t
    if _scalar(
        server.admin,
        "SELECT count(*) FROM pg_cast "
        "WHERE castsource = 'int4'::regtype AND casttarget = 'bytea'::regtype",
    ):
        pytest.skip(f"{server.engine} already has a cast from INT to BYTEA")
    try:
        if not (
            _try(
                server.admin,
                f"CREATE FUNCTION {t}_tobytes(INT) RETURNS BYTEA LANGUAGE SQL AS "
                f"'SELECT decode(nextval(''{t}_seq'')::TEXT, ''escape'')'",
            )
            and _try(
                server.admin,
                f"CREATE CAST (INT AS BYTEA) WITH FUNCTION {t}_tobytes(INT)",
            )
        ):
            pytest.skip(f"{server.engine} has no user-defined casts")
        before = _sequence_moves_from_here(server.admin, t)
        answer = _call("query", nickname="endpoint", sql="SELECT 1::bytea AS b")
        if "Catalog Version Mismatch" in answer.get("error", ""):
            # YugabyteDB, measured: a session opened before the cast existed
            # first answers "cannot cast type integer to bytea … Catalog
            # Version Mismatch … Try again", having run nothing. Asked again,
            # it sees the cast.
            answer = _call("query", nickname="endpoint", sql="SELECT 1::bytea AS b")
        _refused(answer)
        assert "DROP CAST (integer AS bytea)" in answer["error"], answer
        assert _scalar(server.admin, f"SELECT last_value FROM {t}_seq") == before
    finally:
        _try(server.admin, "DROP CAST IF EXISTS (INT AS BYTEA)")
        _try(server.admin, f"DROP FUNCTION IF EXISTS {t}_tobytes(INT)")
