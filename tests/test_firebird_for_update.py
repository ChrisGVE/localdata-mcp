"""A Firebird select carrying ``FOR UPDATE`` is refused by name, not answered empty (#145).

``firebirdsql``, the driver this server reaches Firebird through, fetches rows
only for a statement the server reports as a plain select. Firebird reports a
select carrying ``FOR UPDATE`` as another kind, a select for update, so the
driver never asks for its rows and the cursor holds none. ``query`` used to
answer such a statement ``ok`` with no rows whatever the table held, on a
read-only connection and a read-write one alike. An agent reads that as "no row
matches", which is false.

The driver is not this server's to fix. So the reader refuses the clause before
the statement is sent, in the words #144 gave a refused locking read: it names
the clause, says why the answer would be false, and ends with the same SELECT
without the clause. That rephrase is not argued, it is run: the live half sends
it back through ``query`` and counts the rows.

``WITH LOCK`` alone is not this gate's: the read-only transaction refuses it
itself, and #144 explains that refusal (tests/test_locking_read.py).

The live half needs the Firebird container and skips without it, like every
endpoint test.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import firebirdsql
import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import DDL, create_engine, text
from sqlalchemy.exc import ProgrammingError

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import (
    Refusal,
    _firebird_for_update,
    _firebird_without_row_locks,
    _refuse_firebird_for_update,
    backend_for,
)
from localdata_mcp.loader import Workspace

# ---------------------------------------------------------------------------
# Finding the clause, offline
# ---------------------------------------------------------------------------

#: Every variant measured in #145, and a select behind a common table
#: expression, each answered with no rows before this gate, with the clause it
#: holds and the same select without it.
_VARIANTS = [
    ("SELECT * FROM t FOR UPDATE", "FOR UPDATE", "SELECT * FROM t"),
    (
        "SELECT * FROM t FOR UPDATE OF salary",
        "FOR UPDATE OF salary",
        "SELECT * FROM t",
    ),
    (
        "SELECT * FROM t WHERE salary > 2 FOR UPDATE",
        "FOR UPDATE",
        "SELECT * FROM t WHERE salary > 2",
    ),
    (
        "SELECT * FROM t FOR UPDATE WITH LOCK",
        "FOR UPDATE WITH LOCK",
        "SELECT * FROM t",
    ),
    (
        "WITH c AS (SELECT * FROM t) SELECT * FROM c FOR UPDATE",
        "FOR UPDATE",
        "WITH c AS (SELECT * FROM t) SELECT * FROM c",
    ),
]


@pytest.mark.parametrize(("sql", "clause", "rephrase"), _VARIANTS)
def test_firebird_finds_a_for_update_clause(sql, clause, rephrase):
    assert _firebird_without_row_locks(sql) == (clause, rephrase)
    assert _firebird_for_update(sql) == (clause, rephrase)


@pytest.mark.parametrize(
    ("sql", "clause", "rephrase"),
    [
        ("select * from t for update", "for update", "select * from t"),
        ("SELECT * FROM t\n  FOR   UPDATE;", "FOR UPDATE", "SELECT * FROM t;"),
        (
            'SELECT * FROM "T" FOR UPDATE OF "Salary", name',
            'FOR UPDATE OF "Salary", name',
            'SELECT * FROM "T"',
        ),
        (
            "SELECT * FROM t FOR UPDATE OF salary WITH LOCK SKIP LOCKED",
            "FOR UPDATE OF salary WITH LOCK SKIP LOCKED",
            "SELECT * FROM t",
        ),
        (
            "WITH c AS (SELECT * FROM t) SELECT * FROM c FOR UPDATE",
            "FOR UPDATE",
            "WITH c AS (SELECT * FROM t) SELECT * FROM c",
        ),
        (
            "/* why */ SELECT * FROM t FOR UPDATE",
            "FOR UPDATE",
            "/* why */ SELECT * FROM t",
        ),
    ],
)
def test_firebird_finds_for_update_however_it_is_written(sql, clause, rephrase):
    assert _firebird_for_update(sql) == (clause, rephrase)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t",
        "SELECT * FROM t WITH LOCK",
        "SELECT * FROM t WITH LOCK SKIP LOCKED",
        "SELECT 'FOR UPDATE' FROM t",
        'SELECT "FOR UPDATE" FROM t',
        "SELECT * FROM t -- FOR UPDATE",
        "SELECT * FROM t /* FOR UPDATE */",
        "SELECT SUBSTRING(name FROM 1 FOR 2) FROM t",
        "UPDATE OR INSERT INTO t VALUES ('a', 1)",
        # The block is a plain select to the server, and its rows are fetched:
        # measured, five rows (docs/CONSTRAINTS.md §34).
        "EXECUTE BLOCK RETURNS (n INT) AS BEGIN "
        "FOR SELECT salary FROM t FOR UPDATE INTO n DO SUSPEND; END",
    ],
)
def test_firebird_finds_no_for_update_where_there_is_none(sql):
    assert _firebird_for_update(sql) is None


# ---------------------------------------------------------------------------
# What the explainer says, offline
# ---------------------------------------------------------------------------


def _gated(sql: str) -> tuple[Refusal, BaseException | None]:
    """Send ``sql`` through Firebird's reader gate; what it recorded and raised."""
    refusal = Refusal()
    context = SimpleNamespace(dialect=SimpleNamespace(loaded_dbapi=firebirdsql))
    try:
        _refuse_firebird_for_update(refusal, sql, context)
    except firebirdsql.Error as exc:
        return refusal, exc
    return refusal, None


def _explained(sql: str) -> str:
    """What ``query`` answers once the Firebird gate has refused ``sql``."""
    refusal, raised = _gated(sql)
    assert raised is not None, sql
    entry = SimpleNamespace(
        engines=SimpleNamespace(refusal=refusal, read=object()),
        backend=backend_for("firebirdsql"),
        tag="endpoint",
    )
    error = ProgrammingError(sql, None, raised)
    return str(Workspace._explain(None, entry, error, sql))


@pytest.mark.parametrize(
    "sql", ["SELECT * FROM t", "SELECT * FROM t WITH LOCK", "SELECT 'FOR UPDATE'"]
)
def test_the_gate_lets_a_statement_without_for_update_through(sql):
    refusal, raised = _gated(sql)
    assert raised is None, raised
    assert refusal.what is None and refusal.lock is False


@pytest.mark.parametrize(("sql", "clause", "rephrase"), _VARIANTS)
def test_a_for_update_select_is_refused_with_why_and_the_read_without_it(
    sql, clause, rephrase
):
    error = _explained(sql)

    assert error.startswith(f"query did not run this statement: its {clause} "), error
    assert "no rows" in error and "driver" in error, error
    assert error.endswith(f"reads the same rows: {rephrase}"), error
    # Not the session's refusal of #144, nor a refused write.
    assert "read-only session refused" not in error, error
    assert "not a read" not in error and "does not write" not in error, error
    for advice in ("create", "drop", "DDL"):
        assert advice not in error, error


def test_taking_a_refusal_forgets_that_it_was_a_lock():
    refusal = Refusal(what="x", lock=True)
    assert refusal.take() == "x"
    assert refusal.what is None and refusal.lock is False


# ---------------------------------------------------------------------------
# Live, on Firebird
# ---------------------------------------------------------------------------

_SERVERS = [
    target
    for target in TARGETS
    if target.endpoint.dialect == "firebirdsql" and target.auth is None
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


@contextmanager
def _five_rows(target, tmp_path, *, writable: bool):
    """A five-row table on a slot of the posture asked for; yields its name."""
    try:
        reached = reach(target)
    except Unavailable as exc:
        pytest.skip(str(exc))
    admin = create_engine(reached.url)
    t = f"t{uuid.uuid4().hex[:8]}"
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    # Two transactions and one row at a time: Firebird cannot address a table
    # in the transaction that created it, and it has no multi-row VALUES.
    with admin.begin() as conn:
        conn.execute(DDL(f"CREATE TABLE {t} (name VARCHAR(20), salary INT)"))
    try:
        with admin.begin() as conn:
            for salary, name in enumerate("abcde", 1):
                conn.execute(text(f"INSERT INTO {t} VALUES ('{name}', {salary})"))
        with applied(reached.environ):
            attached = _call(
                "attach", database=reached.url, nickname="endpoint", writable=writable
            )
            assert attached["ok"] is True, attached
            yield t
    finally:
        server_module._reset()
        with admin.begin() as conn:
            conn.execute(DDL(f"DROP TABLE {t}"))
        admin.dispose()


_CASES = [
    pytest.param(target, sql, clause, rephrase, writable, id=f"{target.name}-{sql}-{w}")
    for target in _SERVERS
    for sql, clause, rephrase in _VARIANTS
    for writable, w in ((False, "ro"), (True, "rw"))
]


@pytest.mark.endpoint
@pytest.mark.parametrize(("target", "sql", "clause", "rephrase", "writable"), _CASES)
def test_a_for_update_select_is_refused_and_its_rephrase_reads_five_rows(
    target, sql, clause, rephrase, writable, tmp_path
):
    with _five_rows(target, tmp_path, writable=writable) as table:
        answer = _call(
            "query", nickname="endpoint", sql=sql.replace("FROM t", f"FROM {table}")
        )
        assert answer["ok"] is False, answer
        error = answer["error"]
        assert error.startswith(f"query did not run this statement: its {clause} "), (
            error
        )
        assert "no rows" in error, error
        sent = error.rsplit("reads the same rows: ", 1)[1]
        assert sent == rephrase.replace("FROM t", f"FROM {table}"), error

        read = _call("query", nickname="endpoint", sql=sent)
        assert read["ok"] is True, read
        assert read["row_count"] == (3 if "WHERE" in sent else 5), read


@pytest.mark.endpoint
@pytest.mark.parametrize("target", _SERVERS, ids=lambda target: target.name)
def test_a_block_selecting_for_update_is_not_refused_and_reads(target, tmp_path):
    """Firebird runs the block as a plain select, so its rows are fetched."""
    with _five_rows(target, tmp_path, writable=False) as table:
        answer = _call(
            "query",
            nickname="endpoint",
            sql=(
                "EXECUTE BLOCK RETURNS (n INT) AS BEGIN FOR SELECT salary FROM "
                f"{table} FOR UPDATE INTO n DO SUSPEND; END"
            ),
        )
        assert answer["ok"] is True, answer
        assert answer["row_count"] == 5, answer
