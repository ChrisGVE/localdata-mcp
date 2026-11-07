"""``query`` sends the caller's SQL exactly as written: no colon in it is a bind parameter (#149).

``query`` takes no parameters, so nothing in the caller's SQL can be one. It used
to reach the database as ``text(sql)``, and SQLAlchemy reads any ``:word`` in a
``text()`` that is not preceded by a word character as a bind parameter, inside
a string literal or a comment included. Nothing binds it, so
``SELECT 'a :b' AS s FROM x`` was refused before it was sent, with "A value is
required for bind parameter 'b'" — a concept the caller never used. The same
held for Firebird PSQL's ``INTO :n`` in an ``EXECUTE BLOCK``.

The statement now goes through :func:`loader.as_written`, which escapes every
colon the way SQLAlchemy documents for ``text()`` (``\\:``), so the compiled
statement is the caller's string, character for character. That is checked here
by compiling for the dialect behind every backend this server registers, and
through the protocol on SQLite and DuckDB. The live half runs the same reads on
PostgreSQL, MySQL and Firebird, and skips without their containers, like every
endpoint test.
"""

from __future__ import annotations

import asyncio
import csv
import itertools
import json
from pathlib import Path

import foreign
import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import Text, event
from sqlalchemy.engine import make_url

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.dialects import BACKENDS
from localdata_mcp.loader import Workspace, as_written

#: Colons wherever a caller can put one: in a literal after a space, at the
#: start of a literal, in a time, after a backslash as typed (one and two),
#: in PostgreSQL's cast, in a comment, and in PSQL's variable syntax.
_COLONS = [
    "SELECT 'a :b'",
    "SELECT ':b'",
    "SELECT '10:30'",
    r"SELECT 'x\:y'",
    r"SELECT 'x\\:y'",
    "SELECT 'a::text'",
    "SELECT 1 -- :x",
    "SELECT 1 /* (:x) */",
    "SELECT ':a:b :c'",
    "FOR SELECT 1 FROM RDB$DATABASE INTO :n DO SUSPEND",
]

#: YugabyteDB is a backend of its own but arrives on PostgreSQL's dialect, so
#: there is no ``yugabytedb`` dialect to load; every other key is one.
_DIALECT_OF = {"yugabytedb": "postgresql"}


def _dialect(backend: str):
    """The dialect a URL for this registered backend resolves to, instantiated."""
    name = _DIALECT_OF.get(backend, backend)
    return make_url(f"{name}://").get_dialect()()


# ---------------------------------------------------------------------------
# The compiled statement, offline, for every registered backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("sql", _COLONS)
def test_the_compiled_statement_is_the_callers_sql(backend, sql):
    compiled = as_written(sql).compile(dialect=_dialect(backend))
    assert str(compiled) == sql
    assert not compiled.params, compiled.params


def test_every_short_string_of_colons_backslashes_and_words_compiles_to_itself():
    """Exhaustive over the characters the two SQLAlchemy patterns look at.

    The bind pattern and the un-escape pattern decide on a colon, a backslash,
    a word character and anything else; ``$`` is a non-word character an
    engine's own syntax uses next to a colon. Every string of up to six of
    them is checked, 19 531 strings, so a case the listed inputs miss — a
    backslash pair before a colon, a colon run between two others — is not
    left to an example someone thought of.
    """
    dialect = _dialect("sqlite")
    alphabet = [":", "\\", "a", " ", "$"]
    checked = 0
    for length in range(7):
        for chars in itertools.product(alphabet, repeat=length):
            sql = "".join(chars)
            assert str(as_written(sql).compile(dialect=dialect)) == sql, sql
            checked += 1
    assert checked == sum(len(alphabet) ** n for n in range(7))


def test_every_registered_backend_was_compiled_for():
    """The parametrisation above reads BACKENDS; this pins that it is not empty."""
    assert {"sqlite", "duckdb", "postgresql", "mysql", "firebirdsql"} <= set(BACKENDS)


# ---------------------------------------------------------------------------
# What reaches the driver
# ---------------------------------------------------------------------------


def test_a_colon_reaches_the_driver_as_written_with_no_parameters(tmp_path):
    path = foreign.build_database(tmp_path / "t.db", "t", [("a", Text)], [("x",)])
    workspace = Workspace()
    workspace.attach_file("t", path)
    sent: list[tuple[str, object]] = []

    @event.listens_for(workspace.entry("t").engines.read, "before_cursor_execute")
    def _record(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
        sent.append((statement, parameters))

    sql = "SELECT 'a :b' AS s FROM t"
    assert workspace.query("t", sql) == (["s"], [("a :b",)])
    mine = [parameters for statement, parameters in sent if statement == sql]
    assert len(mine) == 1, sent
    assert not mine[0], mine


# ---------------------------------------------------------------------------
# Through the protocol, on the two embedded engines
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path):
    """A fresh registry and a fresh allowed root for every test."""
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(config_module.Config(roots=(root,)))
    server_module._reset()
    yield root
    server_module._reset()


def _call(tool: str, **arguments):
    """Call a tool over the protocol and return its decoded payload."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


def _sqlite(root: Path) -> None:
    """``x.csv``, two columns, attached as ``x`` — a file-derived SQLite database."""
    path = root / "x.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([["a", "b"], ["1", "2"]])
    attached = _call("attach", database=str(path), nickname="x")
    assert attached["ok"] is True, attached


def _duckdb(root: Path) -> None:
    """``x.duckdb`` with one table ``x`` of one row, attached as ``x``."""
    path = foreign.build_database(
        root / "x.duckdb", "x", [("a", Text)], [("1",)], dialect="duckdb"
    )
    attached = _call("attach", database=str(path), nickname="x")
    assert attached["ok"] is True, attached


@pytest.mark.parametrize("attach", [_sqlite, _duckdb], ids=["sqlite", "duckdb"])
@pytest.mark.parametrize(
    ("sql", "value"),
    [
        ("SELECT 'a :b' AS s FROM x", "a :b"),
        ("SELECT ':b' AS s FROM x", ":b"),
        ("SELECT 'x\\:y' AS s FROM x", "x\\:y"),
        ("SELECT 'a :b' AS s FROM x -- :c", "a :b"),
    ],
)
def test_query_answers_a_colon_in_a_literal_as_written(root, attach, sql, value):
    attach(root)
    answer = _call("query", nickname="x", sql=sql)
    assert answer["ok"] is True, answer
    assert answer["rows"] == [[value]], answer


# ---------------------------------------------------------------------------
# Live, through query: PostgreSQL, MySQL, Firebird
# ---------------------------------------------------------------------------

#: Per engine, reads that carry a colon and the rows each answers. Firebird has
#: no FROM-less SELECT, and its block is the issue's own case: PSQL's ``INTO :n``.
_LIVE = {
    "postgres": [
        ("SELECT 'a :b' AS s", [["a :b"]]),
        ("SELECT '1'::int AS n", [[1]]),
        ("SELECT 'x\\:y' AS s", [["x\\:y"]]),
    ],
    "mysql": [
        ("SELECT 'a :b' AS s", [["a :b"]]),
        ("SELECT '10:30' AS s -- :x", [["10:30"]]),
    ],
    "firebird": [
        ("SELECT 'a :b' AS s FROM RDB$DATABASE", [["a :b"]]),
        (
            "EXECUTE BLOCK RETURNS (n INT) AS BEGIN "
            "FOR SELECT 1 FROM RDB$DATABASE INTO :n DO SUSPEND; END",
            [[1]],
        ),
    ],
}

_CASES = [
    pytest.param(target, sql, rows, id=f"{target.name}-{index}")
    for target in TARGETS
    if target.name in _LIVE and target.auth is None
    for index, (sql, rows) in enumerate(_LIVE[target.name])
]


@pytest.mark.endpoint
@pytest.mark.parametrize(("target", "sql", "rows"), _CASES)
def test_a_colon_reaches_a_live_engine_as_written(target, sql, rows, tmp_path):
    try:
        reached = reach(target)
    except Unavailable as exc:
        pytest.skip(str(exc))
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(config_module.Config(roots=(root,), network_enabled=True))
    server_module._reset()
    try:
        with applied(reached.environ):
            attached = _call("attach", database=reached.url, nickname="endpoint")
            assert attached["ok"] is True, attached
            answer = _call("query", nickname="endpoint", sql=sql)
            assert answer["ok"] is True, answer
            assert answer["rows"] == rows, answer
    finally:
        server_module._reset()
