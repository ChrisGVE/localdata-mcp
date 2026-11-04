"""MySQL's and MariaDB's reader refuses ``LOAD_FILE()`` by name (#36).

``LOAD_FILE(path)`` returns the contents of a file on the database server. It
is a read, so ``SET SESSION TRANSACTION READ ONLY`` lets it run, and no catalog
the #123 gate asks lists it, because it is a built-in. Whether it may read is
then left to the account's ``FILE`` privilege and the server's
``secure_file_priv``, and a ``query`` caller is handed whatever the account
behind the URL can read. So the reader refuses the call before sending it:

* **the call, by its shape** (:data:`_MYSQL_FILE_READ_CALL`): the word, then a
  parenthesis or a comment, case-insensitive, wherever it sits — a subquery,
  an expression, a backquoted name;
* **a view reading a file, by its name**: ``LOAD_FILE`` is followed into view
  definitions like a stored function (:func:`_mysql_unsafe_names`).

A column, an alias or a literal that merely holds the word is a read and is
answered.

The live half needs ``localdata-test-mysql`` and ``localdata-test-mariadb``
and skips without them, like every endpoint test. It connects as ``root``, who
holds ``FILE``, because a refusal tested where the file could not have been
read anyway proves nothing.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import DDL, create_engine, make_url, text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import _mysql_shape_refusal, _mysql_unsafe_names

# ---------------------------------------------------------------------------
# The call, offline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT LOAD_FILE('/etc/passwd')",
        "select load_file('/etc/passwd')",
        "SELECT LoAd_FiLe('/etc/passwd')",
        "SELECT LOAD_FILE ('/etc/passwd')",
        "SELECT LOAD_FILE\n('/etc/passwd')",
        "SELECT LOAD_FILE/**/('/etc/passwd')",
        "SELECT LOAD_FILE# a note\n('/etc/passwd')",
        "SELECT LOAD_FILE-- a note\n('/etc/passwd')",
        "SELECT `load_file`('/etc/passwd')",
        "SELECT length(LOAD_FILE('/etc/passwd'))",
        "SELECT CONCAT('x', load_file('/etc/passwd'), 'y')",
        "SELECT name FROM t WHERE name = (SELECT LOAD_FILE('/etc/passwd'))",
        "SELECT * FROM (SELECT LOAD_FILE('/etc/passwd') AS f) AS q",
        "WITH q AS (SELECT LOAD_FILE('/etc/passwd') AS f) SELECT f FROM q",
        "SELECT 1 FROM t WHERE LOAD_FILE('/etc/passwd') IS NOT NULL",
        "SELECT 1 UNION SELECT LOAD_FILE('/etc/passwd')",
    ],
)
def test_a_load_file_call_is_refused_wherever_it_sits(sql):
    refused = _mysql_shape_refusal(sql)
    assert refused is not None, sql
    assert "LOAD_FILE" in refused, refused
    assert "database server" in refused, refused
    assert "attach" in refused, refused  # the way to read a file of one's own


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT load_file FROM t",
        "SELECT `load_file` FROM t",
        "SELECT 1 AS load_file",
        "SELECT load_file_count FROM t",
        "SELECT my_load_file FROM t",
        "SELECT 'load_file'",
        "SELECT 'a note about load_file and nothing else'",
        "SELECT name FROM t WHERE note LIKE '%load_file%'",
    ],
)
def test_a_name_or_literal_merely_holding_the_word_is_a_read(sql):
    assert _mysql_shape_refusal(sql) is None, sql


class _Catalog:
    """A cursor answering the four catalog questions the name half asks."""

    def __init__(self, functions, views):
        self._answers = {
            "ROUTINES": [(name,) for name in functions],
            "VIEWS": list(views.items()),
        }
        self._last: list = []

    def execute(self, sql):
        import pymysql

        for key, rows in self._answers.items():
            if key in sql:
                self._last = rows
                return
        # The loadable-function listings, which an unprivileged user may not read.
        raise pymysql.err.OperationalError(1142, "SELECT command denied")

    def fetchall(self):
        return self._last


def test_a_view_reading_a_server_file_is_refused_by_its_name():
    unsafe = _mysql_unsafe_names(
        _Catalog(
            functions=[],
            views={
                "leak": "select load_file('/etc/passwd') AS `f`",
                "leak_again": "select `f` from `leak`",
                "plain": "select 1 AS `one`",
            },
        )
    )
    assert unsafe == {"leak", "leak_again"}


def test_the_word_itself_is_left_to_the_shape_half():
    """A statement that mentions ``load_file`` as a column is not refused by name."""
    unsafe = _mysql_unsafe_names(_Catalog(functions=["fset"], views={}))
    assert unsafe == {"fset"}


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
    """The container's administrator, who holds ``FILE``."""
    return (
        make_url(url)
        .set(username="root", password="testpass")
        .render_as_string(hide_password=False)
    )


@pytest.fixture(params=_SERVERS, ids=lambda target: target.name)
def server(request, tmp_path):
    """A read-only root slot, a file only the server can read, and a view reading it.

    Yields ``(path, view, admin)``. Where ``secure_file_priv`` names a
    directory (MySQL 8.0's image), a file is planted there, the one place
    ``LOAD_FILE`` reads. Where it is NULL (MariaDB 11's image), MariaDB reads
    any file its own user can, measured: root read ``/etc/passwd`` and
    ``/etc/hostname``, so the second is the file.
    """
    try:
        reached = reach(request.param)
    except Unavailable as exc:
        pytest.skip(str(exc))
    root_url = _as_root(reached.url)
    admin = create_engine(root_url)
    tag = f"t{uuid.uuid4().hex[:8]}"
    view = f"{tag}_leak"
    with admin.connect() as conn:
        directory = conn.execute(text("SELECT @@global.secure_file_priv")).scalar()
    path = "/etc/hostname"
    if directory is not None:
        path = f"{directory.rstrip('/') or '/tmp'}/{tag}_planted"
        with admin.connect() as conn:
            conn.execute(text(f"SELECT 'the planted text' INTO DUMPFILE '{path}'"))
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    try:
        with admin.begin() as conn:
            conn.execute(
                DDL(f"CREATE VIEW {view} AS SELECT LOAD_FILE('/etc/hostname') AS f")
            )
        with applied(reached.environ):
            attached = _call("attach", database=root_url, nickname="endpoint")
            assert attached["ok"] is True, attached
            yield path, view, admin
    finally:
        server_module._reset()
        with admin.begin() as conn:
            conn.execute(DDL(f"DROP VIEW IF EXISTS {view}"))
        admin.dispose()


def _read_by_admin(admin, path: str):
    with admin.connect() as conn:
        return conn.execute(text("SELECT LOAD_FILE(:p)"), {"p": path}).scalar()


@pytest.mark.endpoint
@pytest.mark.parametrize(
    "template",
    [
        "SELECT LOAD_FILE('{path}')",
        "select load_file ('{path}')",
        "SELECT length(LOAD_FILE('{path}'))",
        "SELECT 1 FROM (SELECT LOAD_FILE('{path}') AS f) AS q",
    ],
)
def test_query_refuses_a_load_file_call_that_the_account_could_make(server, template):
    """Refused through ``query``, where the same account, unrefused, reads the file."""
    path, _, admin = server
    content = _read_by_admin(admin, path)
    assert content, path  # what was at stake: the account reads the file

    answer = _call("query", nickname="endpoint", sql=template.format(path=path))

    assert answer["ok"] is False, answer
    assert "LOAD_FILE" in answer["error"], answer
    assert content.decode().strip() not in json.dumps(answer), answer


@pytest.mark.endpoint
def test_query_refuses_a_view_that_reads_a_server_file(server):
    _, view, _ = server

    answer = _call("query", nickname="endpoint", sql=f"SELECT f FROM {view}")

    assert answer["ok"] is False, answer
    assert view in answer["error"], answer


@pytest.mark.endpoint
@pytest.mark.parametrize(
    "sql",
    ["SELECT 'load_file' AS w", "SELECT 1 AS load_file"],
)
def test_query_answers_a_read_that_merely_holds_the_word(server, sql):
    answer = _call("query", nickname="endpoint", sql=sql)

    assert answer["ok"] is True, answer
