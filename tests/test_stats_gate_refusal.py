"""``stats`` answers a reader-gate refusal in the words ``query`` uses (#133).

A reader gate refuses before the statement runs: it records what was attempted
on the slot's :class:`~localdata_mcp.dialects.Refusal` and raises the driver's
own error, which ``query`` explains. ``stats`` sends its own aggregate through
the same reader, and before this its refusal fell through to the generic
fallback: ``ProgrammingError: (psycopg2.ProgrammingError) query may not … [SQL:
SELECT count(*) AS total, …] (Background on this error at:
https://sqlalche.me/…)``, which names the wrong verb, quotes SQL the caller
never wrote, and carries a link that helps nobody.

Offline, a stand-in gate on a SQLite slot does what every real one does: in
``do_execute``, it records the refusal and raises a DBAPI error, for the
profile's own statement only. The live half runs a real gate on the
PostgreSQL and MySQL families, where a view calling an unsafe function is
refused, and skips without the containers, like every endpoint test.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import DDL, create_engine, event, make_url, text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config


def _call(tool: str, **arguments):
    """The in-process protocol client, as ``test_endpoints`` has it."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


def _assert_in_this_servers_words(error: str, table: str) -> None:
    assert error.startswith(f"stats did not profile {table}: reading it would"), error
    assert "query refuses a read of it for the same reason" in error, error
    assert "[SQL:" not in error, error
    assert "sqlalche.me" not in error, error
    assert "ProgrammingError" not in error and "OperationalError" not in error, error
    assert "query may not" not in error, error


# ---------------------------------------------------------------------------
# Offline: a stand-in gate on a SQLite slot
# ---------------------------------------------------------------------------


@pytest.fixture
def session(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,)))
    server_module._reset()
    (root / "people.csv").write_text("name,salary\na,1\nb,2\n")
    attached = _call(
        "attach", database=str(root / "people.csv"), nickname="p", delimiter=","
    )
    assert attached["ok"] is True, attached
    yield server_module._session()._workspace.entry("p")
    server_module._reset()


def _gate_the_profile(entry, *, refuse: bool) -> None:
    """Refuse ``stats``' aggregate the way every reader gate refuses a statement."""

    @event.listens_for(entry.engines.read, "do_execute")
    def _gate(cursor, statement, parameters, context):  # noqa: ANN001, ANN202
        if "AS total" not in statement:
            return False
        if refuse:
            entry.engines.refusal.what = "run on this server while it holds a cast"
            raise sqlite3.OperationalError("query may not run on this server …")
        raise sqlite3.OperationalError("disk I/O error")


def test_a_gate_refusal_reaches_stats_in_this_servers_words(session):
    _gate_the_profile(session, refuse=True)

    answer = _call("stats", nickname="p", table="people")

    assert answer["ok"] is False, answer
    _assert_in_this_servers_words(answer["error"], "people")
    assert "while it holds a cast" in answer["error"], answer


def test_any_other_driver_failure_reaches_stats_in_the_drivers_words_alone(session):
    _gate_the_profile(session, refuse=False)

    answer = _call("stats", nickname="p", table="people")

    assert answer["ok"] is False, answer
    assert answer["error"] == "disk I/O error", answer


def test_the_refusal_stats_leaves_unread_is_not_attached_to_the_next_query(session):
    _gate_the_profile(session, refuse=True)
    _call("stats", nickname="p", table="people")

    answer = _call("query", nickname="p", sql="SELECT nope FROM people")

    assert answer["ok"] is False, answer
    assert "cast" not in answer["error"], answer


def test_stats_still_profiles_where_nothing_refuses(session):
    answer = _call("stats", nickname="p", table="people")

    assert answer["ok"] is True, answer


# ---------------------------------------------------------------------------
# Live: a real gate refusing a view that calls an unsafe function
# ---------------------------------------------------------------------------

_PG_FAMILY = {"postgres", "cockroachdb", "yugabytedb", "opengauss"}
_MYSQL_FAMILY = {"mysql", "mariadb"}
_SERVERS = [
    target
    for target in TARGETS
    if target.name in _PG_FAMILY | _MYSQL_FAMILY and target.auth is None
]


def _as_root(url: str) -> str:
    return (
        make_url(url)
        .set(username="root", password="testpass")
        .render_as_string(hide_password=False)
    )


@pytest.fixture(params=_SERVERS, ids=lambda target: target.name)
def refused_view(request, tmp_path):
    """A read-only slot holding a view whose definition calls an unsafe function.

    On the PostgreSQL family the function is a user-defined one, which the
    gate refuses whatever it does; on MySQL and MariaDB a stored function.
    Yields the view's name.
    """
    try:
        reached = reach(request.param)
    except Unavailable as exc:
        pytest.skip(str(exc))
    mysql = request.param.name in _MYSQL_FAMILY
    url = _as_root(reached.url) if mysql else reached.url
    t = f"t{uuid.uuid4().hex[:8]}"
    with applied(reached.environ):
        admin = create_engine(url)
        if mysql:
            created = [
                "SET GLOBAL log_bin_trust_function_creators = 1",
                f"CREATE FUNCTION {t}_f() RETURNS INT DETERMINISTIC RETURN 1",
                f"CREATE VIEW {t}_v AS SELECT {t}_f() AS x",
            ]
            dropped = [f"DROP VIEW IF EXISTS {t}_v", f"DROP FUNCTION IF EXISTS {t}_f"]
        else:
            created = [
                f"CREATE FUNCTION {t}_f() RETURNS INT LANGUAGE SQL AS 'SELECT 1'",
                f"CREATE VIEW {t}_v AS SELECT {t}_f() AS x",
            ]
            dropped = [f"DROP VIEW IF EXISTS {t}_v", f"DROP FUNCTION IF EXISTS {t}_f"]
        root = tmp_path / "root"
        root.mkdir()
        config_module.use(Config(roots=(root,), network_enabled=True))
        server_module._reset()
        try:
            with admin.begin() as conn:
                for statement in created:
                    conn.execute(
                        DDL(statement) if "CREATE" in statement else text(statement)
                    )
            attached = _call("attach", database=url, nickname="endpoint")
            assert attached["ok"] is True, attached
            yield f"{t}_v"
        finally:
            server_module._reset()
            with admin.begin() as conn:
                for statement in dropped:
                    conn.execute(DDL(statement))
            admin.dispose()


@pytest.mark.endpoint
def test_stats_on_a_refused_view_answers_as_query_does(refused_view):
    queried = _call("query", nickname="endpoint", sql=f"SELECT x FROM {refused_view}")
    answer = _call("stats", nickname="endpoint", table=refused_view)

    assert queried["ok"] is False, queried  # the gate refuses the read itself
    assert answer["ok"] is False, answer
    _assert_in_this_servers_words(answer["error"], refused_view)
    assert refused_view in answer["error"], answer
