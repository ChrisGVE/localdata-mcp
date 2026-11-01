"""A driver error with no words of its own takes them from what it chained (#117).

The Exasol DBAPI driver (``exasol.driver.websocket``) raises its PEP 249
``Error`` with no arguments — ``raise Error() from ex`` — and keeps the server's
reason on ``__cause__``, a ``pyexasol`` ``ExaQueryError``. Anything that built a
caller's message from ``str(exc.orig)`` therefore handed back ``""``: on
Exasol, every failed statement answered ``{"ok": false, "error": ""}``.

The fix is generic rather than Exasol's: an empty driver message is filled from
the exception chain, and a non-empty one is left exactly as it was. The Exasol
shape is modelled with the driver's **own classes** rather than a look-alike, so
the test cannot pass on a summary of the shape that the real driver does not
have. Only the pyexasol connection is stubbed, and only for what its error's
``__str__`` reads.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pyexasol.exceptions
import pytest
from fastmcp import Client
from exasol.driver.websocket._errors import Error as ExasolError
from sqlalchemy import event
from sqlalchemy.exc import DBAPIError

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.loader import driver_message, failure_text


def call(tool: str, **arguments):
    """Call a tool over the protocol, as :mod:`test_server` does.

    Duplicated rather than imported: importing one test module from another
    makes the first one's fixtures collect twice.
    """

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


#: What Exasol 2026.1.0 answered ``SELECT 1 SELECT 2`` with, measured live
#: through ``query`` on ``localdata-test-exasol`` (#117): the cause's ``message``.
REASON = (
    "syntax error, unexpected SELECT_, expecting UNION_ or EXCEPT_ or MINUS_ or "
    "INTERSECT_ [line 1, column 10] (Session: 1880672662036807680)"
)


class _ExaConnection:
    """What ``pyexasol.ExaError.__str__`` reads off its connection, and no more.

    ``verbose_error`` is on because the DBAPI driver turns it on, and it is what
    makes ``str()`` of the cause a multi-line block rather than the reason.
    """

    options = {"verbose_error": True, "dsn": "127.0.0.1:18563", "user": "localdata"}

    def current_schema(self) -> str:
        return "LOCALDATA"

    def session_id(self) -> str:
        return "1880672662036807680"


def _exasol_failure(statement: str = "SELECT 1 SELECT 2") -> DBAPIError:
    """An Exasol failure as SQLAlchemy hands it over: wrapped, words on the cause."""
    cause = pyexasol.exceptions.ExaQueryError(
        _ExaConnection(), statement, "42000", REASON
    )
    try:
        try:
            raise cause
        except pyexasol.exceptions.ExaError as ex:
            # Exactly what exasol/driver/websocket/_cursor.py does.
            raise ExasolError() from ex
    except ExasolError as orig:
        return DBAPIError.instance(statement, None, orig, ExasolError)


def test_the_model_is_the_shape_the_driver_raises():
    exc = _exasol_failure()

    assert exc.orig.args == ()
    assert str(exc.orig) == ""
    assert isinstance(exc.orig.__cause__, pyexasol.exceptions.ExaQueryError)
    # Raised inside the handler, so the cause is the implicit context as well.
    assert exc.orig.__context__ is exc.orig.__cause__
    assert exc.orig.__cause__.message == REASON
    # The cause's own str() is pyexasol's verbose block, not the reason alone.
    assert str(exc.orig.__cause__) != REASON
    assert REASON in str(exc.orig.__cause__)


def test_an_empty_driver_message_takes_the_reason_from_its_cause():
    assert driver_message(_exasol_failure()) == REASON


def test_the_reason_is_the_message_not_the_verbose_block():
    # The block carries the DSN, the user and the schema; the reason is enough.
    words = driver_message(_exasol_failure())

    assert "dsn" not in words
    assert "localdata" not in words.lower()


def test_failure_text_puts_the_reason_where_the_driver_left_it_empty():
    exc = _exasol_failure()
    header = "(exasol.driver.websocket._errors.Error) "

    assert str(exc).startswith(header + "\n")
    assert failure_text(exc) == str(exc).replace(header, header + REASON, 1)


def _sqlite_failure() -> DBAPIError:
    try:
        sqlite3.connect(":memory:").execute("SELEC 1")
    except sqlite3.Error as orig:
        return DBAPIError.instance("SELEC 1", None, orig, sqlite3.Error)
    raise AssertionError("SQLite accepted SELEC")


def test_a_driver_with_words_of_its_own_is_left_exactly_as_it_was():
    exc = _sqlite_failure()

    assert driver_message(exc) == str(exc.orig)
    assert failure_text(exc) == str(exc)


def test_an_unwrapped_driver_error_with_words_is_left_as_it_was():
    plain = RuntimeError("connection reset by peer")

    assert driver_message(plain) == "connection reset by peer"
    assert failure_text(plain) == "connection reset by peer"


def test_an_unwrapped_driver_error_without_words_takes_its_cause():
    try:
        try:
            raise ValueError("the real reason")
        except ValueError as ex:
            raise ExasolError() from ex
    except ExasolError as unwrapped:
        assert driver_message(unwrapped) == "the real reason"
        assert failure_text(unwrapped) == "the real reason"


def test_the_context_is_read_when_there_is_no_cause():
    try:
        try:
            raise ValueError("implicitly chained")
        except ValueError:
            raise ExasolError()  # noqa: B904 - the implicit chain is the case
    except ExasolError as orig:
        exc = DBAPIError.instance("SELECT 1", None, orig, ExasolError)

    assert driver_message(exc) == "implicitly chained"


def test_a_cycle_in_the_chain_terminates():
    first, second = ExasolError(), ExasolError()
    first.__cause__ = second
    second.__cause__ = first
    exc = DBAPIError.instance("SELECT 1", None, first, ExasolError)

    # Nothing in the chain has words, so there are none to give — but it ends.
    assert driver_message(exc) == ""
    assert failure_text(exc) == str(exc)


def test_a_long_chain_is_walked_only_so_far():
    head = ExasolError()
    link = head
    for _ in range(1000):
        nxt = ExasolError()
        link.__cause__ = nxt
        link = nxt
    link.__cause__ = ValueError("far down")

    assert driver_message(head) == ""


# ---------------------------------------------------------------------------
# Through the tool surface: query's explainer, and a `Could not …` site
# ---------------------------------------------------------------------------


@pytest.fixture
def staff(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "staff.csv").write_text("name,salary\nalice,1\nbob,2\n")
    config_module.use(Config(roots=(root,)))
    server_module._reset()

    payload = call(
        "attach", database=str(root / "staff.csv"), nickname="staff", delimiter=","
    )
    assert payload["ok"] is True, payload
    yield call
    server_module._reset()


def _fail_like_exasol(engine, marker: str) -> None:
    """Make every statement containing ``marker`` fail the way Exasol's do."""

    @event.listens_for(engine, "before_cursor_execute")
    def _raise(conn, cursor, statement, parameters, context, executemany):
        if marker in statement:
            raise _exasol_failure(statement)


def test_query_answers_the_reason_not_an_empty_error(staff):
    call = staff
    entry = server_module._session()._workspace.entry("staff")
    _fail_like_exasol(entry.engines.read, "salary")

    answer = call("query", nickname="staff", sql="SELECT salary FROM staff")

    assert answer == {"ok": False, "error": REASON}


def test_a_failed_rename_answers_the_reason(staff):
    call = staff
    entry = server_module._session()._workspace.entry("staff")
    _fail_like_exasol(entry.engines.write, "RENAME")

    answer = call("update", nickname="staff", type="table", name="staff", to="people")

    assert answer["ok"] is False, answer
    assert answer["error"].startswith("Could not rename staff.staff: "), answer
    assert REASON in answer["error"], answer
