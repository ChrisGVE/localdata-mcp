"""A refused locking read is explained as a lock, with the read that works (#144).

``SELECT * FROM t FOR UPDATE`` passes PostgreSQL's reader gate, which decides
by what a statement mentions, and is stopped by the reader's read-only
session: the server answers 25006, the code it answers a write with. The
explainer used to read that code as a write and said the statement "returns
no rows, so it is not a read", which is false and names no way forward. The
statement does read; what it also asks for is a lock on every row it reads,
and the reader takes none. The same SELECT without its locking clause reads
the same rows.

So where the session refuses a statement holding a locking clause, the
refusal names the clause, says the reader takes no row locks, and gives the
statement with the clause removed. That rephrase is not argued, it is run:
the live half sends it back through ``query`` and counts the rows.

The same holds on MySQL and MariaDB for ``FOR UPDATE``, refused there with
1792. Oracle and SQL Server do not refuse a locking clause at all, measured,
so they have nothing to explain (docs/CONSTRAINTS.md §34).

Firebird refuses ``WITH LOCK`` with its read-only transaction's code and
is answered the same way.

The live half needs the PostgreSQL-family, MySQL, MariaDB and Firebird
containers and skips without them, like every endpoint test.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import DDL, create_engine, make_url, text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import (
    Refusal,
    _firebird_without_row_locks,
    _mysql_without_row_locks,
    _pg_without_row_locks,
    backend_for,
)
from localdata_mcp.loader import Workspace

# ---------------------------------------------------------------------------
# Finding the clause, offline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "clause", "rephrase"),
    [
        ("SELECT * FROM t FOR UPDATE", "FOR UPDATE", "SELECT * FROM t"),
        ("SELECT * FROM t FOR NO KEY UPDATE", "FOR NO KEY UPDATE", "SELECT * FROM t"),
        ("SELECT * FROM t FOR SHARE", "FOR SHARE", "SELECT * FROM t"),
        ("SELECT * FROM t FOR KEY SHARE", "FOR KEY SHARE", "SELECT * FROM t"),
        ("SELECT * FROM t for update", "for update", "SELECT * FROM t"),
        (
            "SELECT * FROM t FOR UPDATE SKIP LOCKED",
            "FOR UPDATE SKIP LOCKED",
            "SELECT * FROM t",
        ),
        ("SELECT * FROM t FOR UPDATE NOWAIT", "FOR UPDATE NOWAIT", "SELECT * FROM t"),
        ("SELECT * FROM t FOR SHARE OF t", "FOR SHARE OF t", "SELECT * FROM t"),
        (
            'SELECT * FROM s."T" FOR UPDATE OF "T" NOWAIT;',
            'FOR UPDATE OF "T" NOWAIT',
            'SELECT * FROM s."T";',
        ),
        (
            "SELECT * FROM a, b FOR UPDATE OF a FOR SHARE OF b",
            "FOR UPDATE OF a",
            "SELECT * FROM a, b",
        ),
        (
            "SELECT * FROM t ORDER BY 1 FOR UPDATE LIMIT 2",
            "FOR UPDATE",
            "SELECT * FROM t ORDER BY 1 LIMIT 2",
        ),
        (
            "SELECT * FROM t\n  FOR   UPDATE",
            "FOR UPDATE",
            "SELECT * FROM t",
        ),
        (
            "SELECT * FROM (SELECT * FROM t FOR UPDATE) AS q",
            "FOR UPDATE",
            "SELECT * FROM (SELECT * FROM t) AS q",
        ),
    ],
)
def test_a_locking_clause_is_found_and_the_read_without_it_given(sql, clause, rephrase):
    assert _pg_without_row_locks(sql) == (clause, rephrase)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t",
        "SELECT 'FOR UPDATE' FROM t",
        "SELECT * FROM t /* FOR UPDATE */",
        "SELECT * FROM t -- FOR SHARE",
        'SELECT "for update" FROM t',
        "SELECT $$FOR UPDATE$$",
        "SELECT substring(name FOR 2) FROM t",
        "INSERT INTO t VALUES (1)",
    ],
)
def test_a_statement_with_no_locking_clause_has_none(sql):
    assert _pg_without_row_locks(sql) is None


@pytest.mark.parametrize(
    ("sql", "clause", "rephrase"),
    [
        ("SELECT * FROM t FOR UPDATE", "FOR UPDATE", "SELECT * FROM t"),
        ("SELECT * FROM t FOR SHARE", "FOR SHARE", "SELECT * FROM t"),
        ("SELECT * FROM t LOCK IN SHARE MODE", "LOCK IN SHARE MODE", "SELECT * FROM t"),
        ("SELECT * FROM t FOR UPDATE WAIT 1", "FOR UPDATE WAIT 1", "SELECT * FROM t"),
        (
            "SELECT * FROM `t` FOR UPDATE OF `t` SKIP LOCKED",
            "FOR UPDATE OF `t` SKIP LOCKED",
            "SELECT * FROM `t`",
        ),
        (
            "SELECT * FROM t FOR UPDATE NOWAIT;",
            "FOR UPDATE NOWAIT",
            "SELECT * FROM t;",
        ),
    ],
)
def test_mysql_finds_its_locking_clauses(sql, clause, rephrase):
    assert _mysql_without_row_locks(sql) == (clause, rephrase)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t",
        "SELECT 'FOR UPDATE' FROM t",
        'SELECT "FOR UPDATE" FROM t',
        "SELECT `for update` FROM t",
        "SELECT * FROM t # FOR UPDATE",
        "SELECT * FROM t -- FOR UPDATE",
        "SELECT * FROM t /* LOCK IN SHARE MODE */",
    ],
)
def test_mysql_finds_none_where_there_is_none(sql):
    assert _mysql_without_row_locks(sql) is None


@pytest.mark.parametrize(
    ("sql", "clause", "rephrase"),
    [
        ("SELECT * FROM t WITH LOCK", "WITH LOCK", "SELECT * FROM t"),
        (
            "SELECT * FROM t WITH LOCK SKIP LOCKED",
            "WITH LOCK SKIP LOCKED",
            "SELECT * FROM t",
        ),
        (
            "SELECT * FROM t FOR UPDATE OF salary WITH LOCK",
            "FOR UPDATE OF salary WITH LOCK",
            "SELECT * FROM t",
        ),
    ],
)
def test_firebird_finds_its_locking_clause(sql, clause, rephrase):
    assert _firebird_without_row_locks(sql) == (clause, rephrase)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t",
        "SELECT 'WITH LOCK' FROM t",
        'SELECT "WITH LOCK" FROM t',
        "SELECT * FROM t -- WITH LOCK",
        "WITH lock AS (SELECT 1 AS a FROM rdb$database) SELECT * FROM lock",
        "WITH lock (a) AS (SELECT 1 FROM rdb$database) SELECT * FROM lock",
    ],
)
def test_firebird_finds_none_where_there_is_none(sql):
    assert _firebird_without_row_locks(sql) is None


def test_the_families_whose_session_refuses_a_lock_find_it_and_no_other_does():
    sql = "SELECT * FROM t FOR UPDATE"
    for dialect in ("postgresql", "cockroachdb", "opengauss"):
        assert backend_for(dialect).without_row_locks(sql) == (
            "FOR UPDATE",
            "SELECT * FROM t",
        ), dialect
    for dialect in ("mysql", "mariadb"):
        assert backend_for(dialect).without_row_locks(sql) == (
            "FOR UPDATE",
            "SELECT * FROM t",
        ), dialect
    assert backend_for("firebirdsql").without_row_locks(
        "SELECT * FROM t WITH LOCK"
    ) == (
        "WITH LOCK",
        "SELECT * FROM t",
    )
    for dialect in ("sqlite", "oracle", "mssql"):
        assert backend_for(dialect).without_row_locks(sql) is None, dialect


# ---------------------------------------------------------------------------
# What the explainer says, offline
# ---------------------------------------------------------------------------


class _ReadOnly:
    """A backend whose session refused the statement as a read-only one does."""

    name = "postgresql"

    def denies_write(self, exc):
        return True

    def ddl_survives_refusal(self):
        return False

    def without_row_locks(self, sql):
        return backend_for("postgresql").without_row_locks(sql)


def _explained(sql: str) -> str:
    entry = SimpleNamespace(
        engines=SimpleNamespace(refusal=Refusal(), read=object()),
        backend=_ReadOnly(),
        tag="endpoint",
    )
    error = Exception("cannot execute SELECT FOR UPDATE in a read-only transaction")
    return str(Workspace._explain(None, entry, error, sql))


def test_a_refused_locking_read_is_explained_as_a_lock():
    error = _explained("SELECT * FROM t FOR UPDATE")

    assert "returns no rows" not in error, error
    assert "not a read" not in error, error
    assert "FOR UPDATE" in error and "row locks" in error, error
    assert error.endswith("reads the same rows: SELECT * FROM t"), error


def test_a_refused_write_is_still_explained_as_one():
    error = _explained("INSERT INTO t VALUES (1)")

    assert "not a read" in error, error
    assert "row locks" not in error, error


# ---------------------------------------------------------------------------
# Live, against every engine whose read-only session refuses a locking clause
# ---------------------------------------------------------------------------

#: Each clause the session refuses, by dialect, measured on 2026-11-04
#: (docs/CONSTRAINTS.md §34). MariaDB has no ``FOR SHARE``, and on MySQL it
#: is not refused, nor is ``LOCK IN SHARE MODE`` on either. Firebird refuses
#: ``WITH LOCK`` only; its driver answers ``FOR UPDATE`` with no rows (#145).
_REFUSED_CLAUSES = {
    "postgresql": [
        "FOR UPDATE",
        "FOR NO KEY UPDATE",
        "FOR SHARE",
        "FOR KEY SHARE",
        "FOR UPDATE SKIP LOCKED",
        "FOR UPDATE NOWAIT",
    ],
    "mysql": ["FOR UPDATE", "FOR UPDATE NOWAIT", "FOR UPDATE SKIP LOCKED"],
    "mariadb": ["FOR UPDATE", "FOR UPDATE NOWAIT", "FOR UPDATE WAIT 1"],
    "firebirdsql": ["WITH LOCK", "WITH LOCK SKIP LOCKED"],
}
_REFUSED_CLAUSES["cockroachdb"] = _REFUSED_CLAUSES["opengauss"] = _REFUSED_CLAUSES[
    "postgresql"
]

_SERVERS = [
    target
    for target in TARGETS
    if target.endpoint.dialect in _REFUSED_CLAUSES and target.auth is None
]

_CASES = [
    pytest.param(target, clause, id=f"{target.name}-{clause}")
    for target in _SERVERS
    for clause in _REFUSED_CLAUSES[target.endpoint.dialect]
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


def _owner(url: str, dialect: str) -> str:
    """Who can take the lock: MySQL's ``FOR UPDATE`` needs UPDATE on the table."""
    if dialect not in ("mysql", "mariadb"):
        return url
    return (
        make_url(url)
        .set(username="root", password="testpass")
        .render_as_string(hide_password=False)
    )


@contextmanager
def _five_rows(target, tmp_path):
    """A five-row table on a read-only slot; yields its name."""
    try:
        reached = reach(target)
    except Unavailable as exc:
        pytest.skip(str(exc))
    url = _owner(reached.url, target.endpoint.dialect)
    admin = create_engine(url)
    t = f"t{uuid.uuid4().hex[:8]}"
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    # Two transactions and one row at a time, for Firebird: it cannot address
    # a table in the transaction that created it, and it has no multi-row
    # VALUES. Created before the ``try`` so a failed CREATE is not hidden by
    # a DROP of nothing; Firebird has no DROP TABLE IF EXISTS.
    with admin.begin() as conn:
        conn.execute(DDL(f"CREATE TABLE {t} (name VARCHAR(20), salary INT)"))
    try:
        with admin.begin() as conn:
            for salary, name in enumerate("abcde", 1):
                conn.execute(text(f"INSERT INTO {t} VALUES ('{name}', {salary})"))
        with applied(reached.environ):
            attached = _call("attach", database=url, nickname="endpoint")
            assert attached["ok"] is True, attached
            yield t
    finally:
        server_module._reset()
        with admin.begin() as conn:
            conn.execute(DDL(f"DROP TABLE {t}"))
        admin.dispose()


@pytest.mark.endpoint
@pytest.mark.parametrize(("target", "clause"), _CASES)
def test_a_refused_locking_read_names_the_lock_and_its_rephrase_reads(
    target, clause, tmp_path
):
    with _five_rows(target, tmp_path) as table:
        answer = _call(
            "query", nickname="endpoint", sql=f"SELECT * FROM {table} {clause}"
        )
        assert answer["ok"] is False, answer
        error = answer["error"]
        assert "returns no rows" not in error and "not a read" not in error, error
        assert clause in error and "row locks" in error, error
        rephrase = error.rsplit("reads the same rows: ", 1)[1]
        assert rephrase == f"SELECT * FROM {table}", error

        read = _call("query", nickname="endpoint", sql=rephrase)
        assert read["ok"] is True, read
        assert read["row_count"] == 5, read
