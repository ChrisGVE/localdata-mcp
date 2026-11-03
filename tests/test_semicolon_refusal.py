"""What the PostgreSQL- and MySQL-family readers say when they refuse a ``;`` (#127).

Both gates decide "one statement" without lexing (:func:`_one_statement`), so
they refuse any ``;`` but a trailing one, a quoted one included:
``SELECT 'a;b'`` is refused as surely as ``SELECT 1; SELECT 2``. The refusal
must be true in every case it fires, so it names the ``;`` rather than a
second statement, says why a quoted one is refused as well, and offers a way
to put a ``;`` in the data. MonetDB was given the same treatment first
(``tests/test_monetdb_read_gate.py``).

Each rephrase was measured on every engine that offers it, through ``query``:

* PostgreSQL 16.15, CockroachDB 26.3.4, YugabyteDB 2.25.2.0 and openGauss
  7.0.0-RC3 answer
  ``SELECT 'a' || chr(59) || 'b'`` with ``a;b``;
* MySQL 8.0.46 and MariaDB 11.8.9 answer
  ``SELECT CONCAT('a', CHAR(59 USING utf8mb4), 'b')`` with ``a;b``. Without
  ``USING``, ``CHAR(59)`` has the charset ``binary`` on both, and ``query``
  answers it with the text ``0x3b`` rather than ``;``.

The live half needs ``localdata-test-postgres``, ``localdata-test-cockroachdb``,
``localdata-test-yugabytedb``, ``localdata-test-opengauss``,
``localdata-test-mysql`` and ``localdata-test-mariadb``, and skips without
them, like every endpoint test.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from endpoints import TARGETS, Unavailable, applied, reach
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import _mysql_shape_refusal, _pg_shape_refusal

#: The rephrase each family's refusal offers, measured as the module says.
_PG_REPHRASE = "SELECT 'a' || chr(59) || 'b'"
_MYSQL_REPHRASE = "SELECT CONCAT('a', CHAR(59 USING utf8mb4), 'b')"

_FAMILIES = {
    "postgresql": (_pg_shape_refusal, _PG_REPHRASE),
    "mysql": (_mysql_shape_refusal, _MYSQL_REPHRASE),
}

#: A ';' wherever it can sit: in a literal, in a comment, between statements.
_SEMICOLONS = [
    "SELECT 'a;b'",
    "SELECT 1 /* ; */",
    "-- a;b\nSELECT 1",
    "SELECT 1; SELECT 2",
    "SELECT 1;;",
]


# ---------------------------------------------------------------------------
# The words, offline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(_FAMILIES))
@pytest.mark.parametrize("sql", _SEMICOLONS)
def test_the_semicolon_refusal_is_true_wherever_the_semicolon_sits(family, sql):
    """It names the ';' and says a quoted one is refused too — never "two statements".

    ``SELECT 'a;b'`` is one statement, so "more than one statement" is false
    for it; the gate cannot tell, by design, so the words must hold for a
    ';' in a literal, a comment, or between two statements.
    """
    shape_refusal, _ = _FAMILIES[family]
    what = shape_refusal(sql)
    assert what is not None, sql
    assert "more than one statement" not in what, what
    assert "put a ';' before its end" in what, what
    assert "quoted string" in what and "comment" in what, what
    assert "does not parse SQL" in what, what  # says why a quoted one costs the read


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_the_semicolon_refusal_offers_its_familys_measured_rephrase(family):
    shape_refusal, rephrase = _FAMILIES[family]
    assert rephrase in shape_refusal("SELECT 'a;b'")


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_the_offered_rephrase_passes_the_shape_half_of_the_gate(family):
    """No ';' in it, and nothing else the statement's shape refuses."""
    shape_refusal, rephrase = _FAMILIES[family]
    assert shape_refusal(rephrase) is None


def test_mysql_is_offered_the_text_form_of_char():
    """``CHAR(59)`` alone is ``binary`` on MySQL and MariaDB: ``query`` answers ``0x3b``."""
    what = _mysql_shape_refusal("SELECT 'a;b'")
    assert "USING utf8mb4" in what, what
    assert "chr(59)" not in what, what


def test_postgresql_is_not_offered_mysqls_rephrase():
    what = _pg_shape_refusal("SELECT 'a;b'")
    assert "CHAR(59" not in what and "CONCAT" not in what, what


# ---------------------------------------------------------------------------
# Live, through query, on every engine that shares a family's wording
# ---------------------------------------------------------------------------

#: Which family each live target belongs to. YugabyteDB arrives on PostgreSQL's
#: dialect, so it is selected by name rather than by dialect.
_FAMILY_OF = {
    "postgres": "postgresql",
    "cockroachdb": "postgresql",
    "yugabytedb": "postgresql",
    "opengauss": "postgresql",
    "mysql": "mysql",
    "mariadb": "mysql",
}

_SERVERS = [
    target for target in TARGETS if target.name in _FAMILY_OF and target.auth is None
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


@pytest.fixture(params=_SERVERS, ids=lambda target: target.name)
def server(request, tmp_path):
    """A read-only slot on one engine; yields the rephrase its family offers."""
    try:
        reached = reach(request.param)
    except Unavailable as exc:
        pytest.skip(str(exc))
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    try:
        with applied(reached.environ):
            attached = _call("attach", database=reached.url, nickname="endpoint")
            assert attached["ok"] is True, attached
            yield _FAMILIES[_FAMILY_OF[request.param.name]][1]
    finally:
        server_module._reset()


@pytest.mark.endpoint
def test_a_semicolon_in_a_literal_is_refused_with_the_rephrase(server):
    refused = _call("query", nickname="endpoint", sql="SELECT 'a;b'")

    assert refused["ok"] is False, refused
    assert "more than one statement" not in refused["error"], refused
    assert "quoted string" in refused["error"], refused
    assert server in refused["error"], refused


@pytest.mark.endpoint
def test_the_offered_rephrase_returns_a_semicolon_through_query(server):
    read = _call("query", nickname="endpoint", sql=server)

    assert read["ok"] is True, read
    assert read["rows"] == [["a;b"]], read
