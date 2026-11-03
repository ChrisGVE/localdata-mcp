"""A cast between two of the server's own types stops every read (#129).

The PostgreSQL-family reader gate refuses a statement by the words it
mentions (``test_postgresql_read_gate``). A cast between two of the server's
own types is the one call it cannot see by any word: after an administrator
runs ``CREATE CAST (INT AS BYTEA) WITH FUNCTION f(INT)``, ``SELECT 1::bytea``
names only ``bytea``, and an ``AS IMPLICIT`` cast would run with no type named
at all. Refusing ``bytea`` or ``integer`` by name would refuse every read of
them, and still miss the implicit form.

Measured on openGauss 7.0.0-RC3, whose read-only session runs ``nextval``,
before this change: ``SELECT 1::bytea`` through ``query`` answered ``ok`` and
the sequence ``f`` calls stayed moved. So the gate fails closed: on a server
holding such a cast, *every* statement on the reader is refused, and the
refusal names the cast and says an administrator must drop it.

The offline half drives the gate with a catalog of fixed rows. The live half
needs ``localdata-test-postgres``, ``localdata-test-yugabytedb`` or
``localdata-test-opengauss`` (CockroachDB has no ``CREATE CAST`` and skips),
and skips without them, like every endpoint test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# The live fixture and helpers of the read-gate suite, reused rather than
# copied: ``server`` attaches a read-only administrator slot over a table and
# a sequence, and the helpers read back what a refused statement changed.
from test_postgresql_read_gate import (  # noqa: F401 - ``server`` is a fixture
    _RELATIONS,
    _call,
    _catalog,
    _refused,
    _scalar,
    _sequence_moves_from_here,
    _try,
    server,
)

from localdata_mcp import dialects as dialects_module
from localdata_mcp.dialects import (
    Refusal,
    _pg_server_cast_refusal,
    _PGFunction,
    _PGServerCast,
    _refuse_postgresql_write,
)

# ---------------------------------------------------------------------------
# Offline: what the refusal says, and when the gate gives it
# ---------------------------------------------------------------------------


class _Refused(Exception):
    """Stands in for the driver's ``ProgrammingError``."""


def _gate(catalog, statement: str = "SELECT 1"):
    """Run the gate on ``statement`` over ``catalog``; return the refusal, or ``None``."""
    catalog.close = lambda: None
    cursor = SimpleNamespace(connection=SimpleNamespace(cursor=lambda: catalog))
    context = SimpleNamespace(
        dialect=SimpleNamespace(loaded_dbapi=SimpleNamespace(ProgrammingError=_Refused))
    )
    refusal = Refusal()
    try:
        _refuse_postgresql_write(refusal, cursor, statement, None, context, _RELATIONS)
    except _Refused as raised:
        assert refusal.what is not None and refusal.what in str(raised), raised
        return refusal.what
    assert refusal.what is None
    return None


def _server_cast(source, target, schema, name, *, volatile=False, extension=False):
    """A row of :data:`_PG_SERVER_CASTS`."""
    return (source, target, schema, name, volatile, extension, schema == "public")


def _with_server_casts(*rows):
    catalog = _catalog(checks=[])
    catalog.answers[dialects_module._PG_SERVER_CASTS] = list(rows)
    return catalog


def test_one_server_cast_refuses_every_statement_naming_the_cast():
    """Named by both types and the function, with the statement that drops it."""
    catalog = _with_server_casts(
        _server_cast("integer", "bytea", "public", "tobytes"),
    )

    refused = _gate(catalog, "SELECT 1")

    assert refused is not None
    assert "from integer to bytea through public.tobytes" in refused, refused
    assert "administrator" in refused and "DROP CAST (integer AS bytea)" in refused
    assert "Every statement on this datasource is refused while it exists" in refused
    assert "no word" in refused, refused


def test_a_refused_server_cast_costs_no_further_catalog_question():
    """Fail-closed decides on one question; the word catalog is not read at all."""
    catalog = _with_server_casts(
        _server_cast("integer", "bytea", "public", "tobytes"),
    )

    _gate(catalog)

    assert catalog.asked == [dialects_module._PG_SERVER_CASTS], catalog.asked


def test_several_server_casts_are_all_named_in_order():
    """An administrator has to drop every one, so the refusal lists every one."""
    refused = _pg_server_cast_refusal(
        [
            _PGServerCast(
                "text", "bytea", _PGFunction("public", "b", False, False, True)
            ),
            _PGServerCast(
                "integer", "bytea", _PGFunction("public", "a", False, False, True)
            ),
        ]
    )

    assert refused is not None
    first = refused.index("from integer to bytea through public.a")
    second = refused.index("from text to bytea through public.b")
    assert first < second, refused
    assert "DROP CAST (integer AS bytea), DROP CAST (text AS bytea)" in refused


def test_no_server_cast_refuses_nothing():
    assert _pg_server_cast_refusal([]) is None


@pytest.mark.parametrize(
    "row",
    [
        # The server's own, volatile only in that its answer varies.
        _server_cast("integer", "bytea", "pg_catalog", "random", volatile=True),
        # An extension's, taken at its word: not volatile.
        _server_cast("integer", "bytea", "ext", "pack", extension=True),
    ],
    ids=["answer-varies", "extension-not-volatile"],
)
def test_a_server_cast_the_gate_would_admit_by_name_refuses_nothing(row):
    """Only a cast whose function the gate refuses by name fails it closed."""
    assert _gate(_with_server_casts(row), "SELECT 1") is None


@pytest.mark.parametrize(
    "row",
    [
        _server_cast("integer", "bytea", "pg_catalog", "setval", volatile=True),
        _server_cast("integer", "bytea", "ext", "pack", volatile=True, extension=True),
        _server_cast("integer", "bytea", "tools", "pack"),
    ],
    ids=["servers-own-volatile", "extension-volatile", "off-the-search-path"],
)
def test_a_server_cast_the_gate_would_refuse_by_name_refuses_everything(row):
    assert _gate(_with_server_casts(row), "SELECT 1") is not None


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


def _query(sql: str):
    """``query`` on the fixture's slot, asked again once after a YugabyteDB catalog bump.

    YugabyteDB, measured for #126: the first statement after a DDL on a session
    opened before it answers "Catalog Version Mismatch … Try again", having run
    nothing.
    """
    answer = _call("query", nickname="endpoint", sql=sql)
    if "Catalog Version Mismatch" in answer.get("error", ""):
        answer = _call("query", nickname="endpoint", sql=sql)
    return answer


def _int_to_bytea_casts(admin) -> int:
    return _scalar(
        admin,
        "SELECT count(*) FROM pg_cast "
        "WHERE castsource = 'int4'::regtype AND casttarget = 'bytea'::regtype",
    )


@pytest.mark.endpoint
def test_a_cast_between_the_servers_own_types_stops_every_read_until_dropped(
    server,  # noqa: F811 - the read-gate suite's fixture, imported above
):
    """``SELECT 1`` is refused, naming the cast; dropping it restores reads.

    The cast's function calls ``nextval``, which openGauss's read-only
    session runs (measured, #125), so the sequence is read back too.
    Cleanup is part of the test: the cast is server-wide, and one left
    behind would refuse every later read on the server.
    """
    t = server.t
    if _int_to_bytea_casts(server.admin):
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

        for sql in ("SELECT 1", "SELECT 1::bytea AS b", f"SELECT count(*) FROM {t}"):
            answer = _query(sql)
            _refused(answer)
            assert "from integer to bytea through" in answer["error"], answer
            assert f"{t}_tobytes" in answer["error"], answer
            assert "DROP CAST (integer AS bytea)" in answer["error"], answer
        assert _scalar(server.admin, f"SELECT last_value FROM {t}_seq") == before

        assert _try(server.admin, "DROP CAST (INT AS BYTEA)")
        assert _try(server.admin, f"DROP FUNCTION {t}_tobytes(INT)")
        assert _int_to_bytea_casts(server.admin) == 0

        restored = _query("SELECT 1 AS one")
        assert restored["ok"] is True, restored
    finally:
        _try(server.admin, "DROP CAST IF EXISTS (INT AS BYTEA)")
        _try(server.admin, f"DROP FUNCTION IF EXISTS {t}_tobytes(INT)")
