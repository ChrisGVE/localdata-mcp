"""A reader gate's shape rule is not explained as a refused write (#135).

Some refusals are about how a statement is *written*, and a read meets them as
surely as a write: a ``;`` before the end (the PostgreSQL family, MySQL,
MariaDB and MonetDB decide "one statement" without lexing), a ``U&"…"`` name on
the PostgreSQL family, and a refused word that sits only inside a quoted
string. ``SELECT 'a;b'`` and ``SELECT 'outfile'`` are reads. Explaining them as
"query reads; it does not write", with ``create``/``update``/``drop`` advice,
tells the caller something false and sends them to the wrong verb.

The gate is unchanged: each rule still refuses whatever it refused. What
changed is the frame. :class:`Refusal` carries ``shape``, the explainer words a
shape refusal as the rule it is, and each such refusal gives a rephrase the
gate itself admits. Every rephrase offered here was run through ``query`` on
every engine it is offered for; the live half below is that measurement, and
skips without the containers, like every endpoint test.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import DDL, create_engine, make_url, text
from sqlalchemy.exc import ProgrammingError

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import (
    Refusal,
    _mysql_join,
    _mysql_named_refusal,
    _mysql_shape,
    _pg_join,
    _pg_quoted_refusal,
    _pg_refusal,
    _pg_shape_refusal,
    _PGUnsafe,
    _quoted_halves,
    _refuse_monetdb_write,
    backend_for,
)
from localdata_mcp.loader import Workspace

_FRAME = "query did not run this statement"
_WRITE_FRAME = "query reads; it does not write"

# ---------------------------------------------------------------------------
# The explainer, offline
# ---------------------------------------------------------------------------


def _explained(what: str, *, shape: bool) -> str:
    """What ``query`` answers for a refusal the gate recorded."""
    refusal = Refusal(what=what, shape=shape)
    entry = SimpleNamespace(
        engines=SimpleNamespace(refusal=refusal),
        backend=backend_for("postgresql"),
        tag="endpoint",
    )
    error = ProgrammingError("SELECT 'a;b'", None, Exception("query may not …"))
    return str(Workspace._explain(None, entry, error, "SELECT 'a;b'"))


def test_a_shape_refusal_is_not_explained_as_a_write():
    error = _explained("put a ';' before its end (…)", shape=True)

    assert error.startswith(_FRAME), error
    assert "does not write" not in error, error
    assert "how a statement is written" in error, error
    assert "put a ';' before its end" in error, error
    for advice in ("create", "drop", "update", "DDL"):
        assert advice not in error, error


def test_any_other_gate_refusal_keeps_the_refused_write_frame():
    error = _explained("run a SET statement", shape=False)

    assert error.startswith(_WRITE_FRAME), error
    assert "create" in error and "drop" in error, error


def test_taking_a_refusal_forgets_that_it_was_a_shape_rule():
    refusal = Refusal(what="x", shape=True)
    assert refusal.take() == "x"
    assert refusal.what is None and refusal.shape is False


# ---------------------------------------------------------------------------
# The rephrase is always one the gate admits
# ---------------------------------------------------------------------------


def test_halves_are_split_nearest_the_middle_where_the_gate_admits_them():
    assert _quoted_halves("set_config", _pg_join, lambda _: True) == (
        "'set_c' || 'onfig'"
    )

    def passes(statement: str) -> bool:  # as if 'set_c' were a refused word too
        return "'set_c'" not in statement

    assert _quoted_halves("set_config", _pg_join, passes) == "'set_' || 'config'"


def test_a_text_no_split_can_carry_gets_no_rephrase():
    assert _quoted_halves("f", _pg_join, lambda _: True) is None
    assert _quoted_halves("abc", _pg_join, lambda _: False) is None


# ---------------------------------------------------------------------------
# The PostgreSQL family, offline
# ---------------------------------------------------------------------------

_UNSAFE = _PGUnsafe(
    words={"set_config": "a function that can change what a rollback does not undo"},
    operators={"###": "t_f"},
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 'set_config'",
        "SELECT 'a note on set_config' AS n",
        "SELECT E'it\\'s set_config'",
        "SELECT $$a set_config b$$",
        "SELECT $q$ set_config $q$",
        "SELECT name FROM t WHERE note = 'set_config'",
    ],
)
def test_a_refused_word_only_in_a_string_is_a_shape_refusal(sql):
    what = _pg_quoted_refusal(sql, _UNSAFE)

    assert what is not None, sql
    assert what.startswith("hold set_config in a quoted string"), what
    assert "a quoted string included" in what, what
    rephrase = _pg_join("'set_c'", "'onfig'")
    assert f"as in SELECT {rephrase}" in what, what
    assert _pg_refusal(f"SELECT {rephrase}", _UNSAFE) is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT set_config('a', 'b', false)",
        "SELECT 'set_config', set_config('a', 'b', false)",
        "SELECT \"set_config\"('a', 'b', false)",
        "SELECT 1 ### 2",
    ],
)
def test_a_refused_word_outside_a_string_keeps_the_write_wording(sql):
    assert _pg_refusal(sql, _UNSAFE) is not None
    assert _pg_quoted_refusal(sql, _UNSAFE) is None


def test_a_refused_operator_only_in_a_string_is_a_shape_refusal():
    what = _pg_quoted_refusal("SELECT '1 ### 2'", _UNSAFE)

    assert what is not None and what.startswith("hold ### in a quoted string"), what
    assert "operator calling t_f" in what, what


def test_a_unicode_escaped_name_is_given_the_plain_spelling():
    what = _pg_shape_refusal('SELECT 1 AS U&"d\\0061ta"')

    assert "write the name itself" in what, what
    assert '"data" for U&"d\\0061ta"' in what, what
    assert _pg_shape_refusal('SELECT 1 AS "data"') is None


# ---------------------------------------------------------------------------
# MySQL and MariaDB, offline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "text_", "rephrase"),
    [
        ("SELECT 'outfile'", "outfile", "CONCAT('out', 'file')"),
        ("SELECT 'into DUMPFILE'", "DUMPFILE", "CONCAT('DUMP', 'FILE')"),
        ("SELECT 'a /*! b'", "/*!", "CONCAT('/', '*!')"),
        ("SELECT 'see load_file(x)'", "load_file(", "CONCAT('load_', 'file(')"),
    ],
)
def test_a_mysql_rule_met_only_in_a_string_is_a_shape_refusal(sql, text_, rephrase):
    what, shape = _mysql_shape(sql)

    assert shape is True, what
    assert what.startswith(f"hold {text_} in a quoted string"), what
    assert f"as in SELECT {rephrase}" in what, what
    assert _mysql_shape(f"SELECT {rephrase}") is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 INTO OUTFILE '/tmp/x'",
        "SELECT 1 INTO DUMPFILE 'outfile'",
        "/*!80000 SET GLOBAL max_connections = 1 */",
        "SELECT LOAD_FILE('/etc/passwd')",
        "SET GLOBAL max_connections = 1",
    ],
)
def test_a_mysql_rule_met_outside_a_string_keeps_the_write_wording(sql):
    what, shape = _mysql_shape(sql)

    assert shape is False, what
    assert not what.startswith("hold "), what


def test_a_mysql_semicolon_is_a_shape_refusal():
    what, shape = _mysql_shape("SELECT 'a;b'")

    assert shape is True
    assert "CHAR(59 USING utf8mb4)" in what, what


def test_a_stored_function_named_only_in_a_string_is_a_shape_refusal():
    what, shape = _mysql_named_refusal("SELECT 'fset'", "fset", {"fset"})

    assert shape is True, what
    assert what.startswith("hold fset in a quoted string"), what
    assert f"as in SELECT {_mysql_join(repr('fs'), repr('et'))}" in what, what


def test_a_stored_function_called_keeps_the_write_wording():
    what, shape = _mysql_named_refusal("SELECT fset(), 'fset'", "fset", {"fset"})

    assert shape is False
    assert what.startswith("use fset,"), what


# ---------------------------------------------------------------------------
# MonetDB, offline
# ---------------------------------------------------------------------------


class _NoCursor:
    def execute(self, *args, **kwargs):
        raise AssertionError(f"the ';' refusal sent SQL to MonetDB: {args}")


def test_a_monetdb_semicolon_is_a_shape_refusal():
    from pymonetdb.exceptions import ProgrammingError as MonetDBError

    refusal = Refusal()
    with pytest.raises(MonetDBError):
        _refuse_monetdb_write(refusal, _NoCursor(), "SELECT 'a;b'", None)
    assert refusal.shape is True
    assert "code(59)" in refusal.what


# ---------------------------------------------------------------------------
# Live: the frame, and every offered rephrase, through query
# ---------------------------------------------------------------------------

_PG_FAMILY = {"postgres", "cockroachdb", "yugabytedb", "opengauss"}
_MYSQL_FAMILY = {"mysql", "mariadb"}
_SERVERS = [
    target
    for target in TARGETS
    if target.name in _PG_FAMILY | _MYSQL_FAMILY | {"monetdb"} and target.auth is None
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
    return (
        make_url(url)
        .set(username="root", password="testpass")
        .render_as_string(hide_password=False)
    )


@pytest.fixture(params=_SERVERS, ids=lambda target: target.name)
def server(request, tmp_path):
    """A read-only slot on one engine; yields ``(family, function)``.

    On MySQL and MariaDB the slot is root's, with a stored function whose name
    a literal can hold; elsewhere ``function`` is ``None``.
    """
    try:
        reached = reach(request.param)
    except Unavailable as exc:
        pytest.skip(str(exc))
    name = request.param.name
    family = "pg" if name in _PG_FAMILY else "mysql" if name in _MYSQL_FAMILY else name
    url, admin, function = reached.url, None, None
    if family == "mysql":
        url = _as_root(reached.url)
        admin = create_engine(url)
        function = f"t{uuid.uuid4().hex[:8]}_fn"
        with admin.begin() as conn:
            conn.execute(text("SET GLOBAL log_bin_trust_function_creators = 1"))
            conn.execute(
                DDL(f"CREATE FUNCTION {function}() RETURNS INT DETERMINISTIC RETURN 1")
            )
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    try:
        with applied(reached.environ):
            attached = _call("attach", database=url, nickname="endpoint")
            assert attached["ok"] is True, attached
            yield family, function
    finally:
        server_module._reset()
        if admin is not None:
            with admin.begin() as conn:
                conn.execute(DDL(f"DROP FUNCTION IF EXISTS {function}"))
            admin.dispose()


def _shape_refused(sql: str) -> str:
    answer = _call("query", nickname="endpoint", sql=sql)
    assert answer["ok"] is False, answer
    error = answer["error"]
    assert error.startswith(_FRAME), error
    assert "does not write" not in error and "create" not in error, error
    return error


def _reads(sql: str, expected) -> None:
    answer = _call("query", nickname="endpoint", sql=sql)
    assert answer["ok"] is True, (sql, answer)
    assert answer["rows"] == [[expected]], (sql, answer)


@pytest.mark.endpoint
def test_a_semicolon_in_a_literal_is_framed_as_the_shape_rule(server):
    error = _shape_refused("SELECT 'a;b'")
    assert "put a ';' before its end" in error, error


@pytest.mark.endpoint
def test_a_pg_unicode_name_is_framed_and_its_plain_spelling_reads(server):
    family, _ = server
    if family != "pg":
        pytest.skip('U&"…" is refused on the PostgreSQL family only')
    error = _shape_refused('SELECT 1 AS U&"d\\0061ta"')
    assert '"data" for U&"d\\0061ta"' in error, error
    answer = _call("query", nickname="endpoint", sql='SELECT 1 AS "data"')
    assert answer["ok"] is True and answer["columns"] == ["data"], answer


@pytest.mark.endpoint
def test_a_pg_refused_word_in_a_literal_is_framed_and_its_rephrase_reads(server):
    family, _ = server
    if family != "pg":
        pytest.skip("the PostgreSQL family's word rule")
    rephrase = "'set_c' || 'onfig'"
    error = _shape_refused("SELECT 'set_config' AS w")
    assert f"as in SELECT {rephrase}" in error, error
    _reads(f"SELECT {rephrase}", "set_config")


@pytest.mark.endpoint
def test_a_pg_refused_call_keeps_the_refused_write_frame(server):
    family, _ = server
    if family != "pg":
        pytest.skip("the PostgreSQL family's word rule")
    answer = _call(
        "query", nickname="endpoint", sql="SELECT set_config('a.b', 'c', false)"
    )
    assert answer["ok"] is False and answer["error"].startswith(_WRITE_FRAME), answer


@pytest.mark.endpoint
@pytest.mark.parametrize(
    ("literal", "expected", "rephrase"),
    [
        ("outfile", "outfile", "CONCAT('out', 'file')"),
        ("a /*! b", "/*!", "CONCAT('/', '*!')"),
        ("see load_file(x)", "load_file(", "CONCAT('load_', 'file(')"),
    ],
)
def test_a_mysql_rule_met_in_a_literal_is_framed_and_its_rephrase_reads(
    server, literal, expected, rephrase
):
    family, _ = server
    if family != "mysql":
        pytest.skip("the MySQL family's shape rules")
    error = _shape_refused(f"SELECT '{literal}' AS w")
    assert f"as in SELECT {rephrase}" in error, error
    _reads(f"SELECT {rephrase}", expected)


@pytest.mark.endpoint
def test_a_mysql_function_named_in_a_literal_is_framed_and_its_rephrase_reads(server):
    family, function = server
    if family != "mysql":
        pytest.skip("the MySQL family's name rule")
    middle = len(function) // 2
    rephrase = f"CONCAT('{function[:middle]}', '{function[middle:]}')"
    error = _shape_refused(f"SELECT '{function}' AS w")
    assert f"as in SELECT {rephrase}" in error, error
    _reads(f"SELECT {rephrase}", function)
    called = _call("query", nickname="endpoint", sql=f"SELECT {function}()")
    assert called["ok"] is False and called["error"].startswith(_WRITE_FRAME), called
