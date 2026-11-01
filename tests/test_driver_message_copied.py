"""A driver message that copies a verbose block keeps only the reason in it.

pyexasol 2.4.1 changed how its DBAPI layer raises: instead of an empty
``Error() from ex`` (#117) it raises ``ProgrammingError(str(ex)) from ex``.
With ``verbose_error`` on, ``str(ex)`` is a block that starts with a newline
and lists the DSN, the user, the schema and the query around ``ex.message``,
so the error was no longer empty and its first line was blank. Measured in
endpoint batch d with pyexasol 2.4.1 on Exasol 2026.1.0:
``test_exasol_answers_a_syntax_error_in_its_own_words`` and
``test_exasol_says_why_an_open_failed`` failed.
"""

from __future__ import annotations

import pyexasol
from exasol.driver.websocket._errors import translate_exception
from sqlalchemy.exc import DBAPIError

from localdata_mcp.loader import driver_message, failure_text

REASON = (
    "syntax error, unexpected SELECT_, expecting UNION_ or EXCEPT_ or MINUS_ or "
    "INTERSECT_ [line 1, column 10] (Session: 1880672662036807680)"
)


class _ExaConnection:
    """What ``pyexasol.ExaError.__str__`` reads off its connection, and no more."""

    options = {"verbose_error": True, "dsn": "127.0.0.1:18563", "user": "localdata"}

    def current_schema(self) -> str:
        return "LOCALDATA"

    def session_id(self) -> str:
        return "1880672662036807680"


def _copied_failure(statement: str = "SELECT 1 SELECT 2") -> DBAPIError:
    """An Exasol failure the 2.4.1 way: the driver's own translation, wrapped."""
    cause = pyexasol.exceptions.ExaQueryError(
        _ExaConnection(), statement, "42000", REASON
    )
    try:
        try:
            raise cause
        except pyexasol.exceptions.ExaError as ex:
            # The driver's own function, so the model cannot drift from it.
            raise translate_exception(ex) from ex
    except Exception as orig:  # noqa: BLE001 - whatever translate_exception chose
        return DBAPIError.instance(statement, None, orig, type(orig))


def test_the_model_carries_the_verbose_block_not_the_reason():
    exc = _copied_failure()

    assert str(exc.orig) == str(exc.orig.__cause__)
    assert str(exc.orig) != REASON
    assert REASON in str(exc.orig)
    assert str(exc.orig).startswith("\n")


def test_a_copied_block_gives_way_to_the_reason_inside_it():
    assert driver_message(_copied_failure()) == REASON


def test_failure_text_puts_the_reason_where_the_block_was():
    exc = _copied_failure()

    text = failure_text(exc)

    assert text == str(exc).replace(str(exc.orig), REASON, 1)
    assert "localdata" not in text.split(REASON)[0]


def test_a_driver_that_chose_its_own_words_keeps_them():
    cause = pyexasol.exceptions.ExaQueryError(
        _ExaConnection(), "SELECT 1", "42000", REASON
    )
    try:
        try:
            raise cause
        except pyexasol.exceptions.ExaError as ex:
            raise RuntimeError("the driver's own sentence") from ex
    except RuntimeError as orig:
        exc = DBAPIError.instance("SELECT 1", None, orig, RuntimeError)

    assert driver_message(exc) == "the driver's own sentence"
    assert failure_text(exc) == str(exc)
