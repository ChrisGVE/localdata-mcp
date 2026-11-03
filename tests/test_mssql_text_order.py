"""Putting punctuation first again on SQL Server text, after #128.

Text now lands on SQL Server as ``NVARCHAR``. Under the server's default
``SQL_`` collation an ``NVARCHAR`` column is ordered by the Unicode rules,
which pass over a hyphen and an apostrophe, where the ``VARCHAR`` it used to be
sorted punctuation first (docs/CONSTRAINTS.md §35.3). A caller who wants the
old order asks for it in the statement, and these tests measure the two ways
that work, through ``query`` against the live server:

* ``ORDER BY col COLLATE Latin1_General_BIN2`` puts punctuation first and
  loses nothing, but it is code-point order, so it is case-sensitive;
* ``ORDER BY CAST(col AS VARCHAR(n))`` is the old order exactly, and costs any
  character the column's code page lacks.

The first test is the control: it fails if the default order ever stops
ignoring punctuation, which would make both rephrases unnecessary.

The fixture and helpers are the ones ``test_mssql_text`` already has, imported
rather than copied; only its fixture is collected here.
"""

from __future__ import annotations

import pytest
from test_mssql_text import (  # noqa: F401 - ``mssql`` is a fixture
    _land,
    _rows,
    mssql,
)

#: The four values the issue's follow-up names, in file order.
_FOUR = ("a-b", "ab", "a'c", "ac")

#: The twelve values §35.3 sorted under ``VARCHAR`` and under ``NVARCHAR``.
_TWELVE = (
    "q",
    "'q",
    "_x",
    "x",
    "a-b",
    "Ab",
    "aB",
    "ab",
    "co op",
    "co-op",
    "coo",
    "coop",
)

#: What ``ORDER BY v`` gave when ``v`` was ``VARCHAR(10)``, copied from §35.3.
_VARCHAR_ORDER = (
    "'q",
    "_x",
    "a-b",
    "Ab",
    "aB",
    "ab",
    "co op",
    "co-op",
    "coo",
    "coop",
    "q",
    "x",
)


def _csv(values: tuple[str, ...]) -> str:
    return "n,v\n" + "".join(f"{i},{v}\n" for i, v in enumerate(values))


def _ordered(table: str, expression: str) -> list[str]:
    return [row[0] for row in _rows(f"SELECT v FROM {table} ORDER BY {expression}")]


@pytest.mark.endpoint
def test_the_default_order_of_nvarchar_text_passes_over_punctuation(
    mssql,  # noqa: F811 - the fixture imported above
):
    """The control: a hyphen and an apostrophe sort as if they were not there."""
    table, _ = _land(mssql, "four", _csv(_FOUR))

    assert _ordered(table, "v") == ["ab", "a-b", "ac", "a'c"]


@pytest.mark.endpoint
def test_a_binary_collation_puts_punctuation_first_again(
    mssql,  # noqa: F811 - the fixture imported above
):
    table, _ = _land(mssql, "four", _csv(_FOUR))

    assert _ordered(table, "v COLLATE Latin1_General_BIN2") == [
        "a'c",
        "a-b",
        "ab",
        "ac",
    ]


@pytest.mark.endpoint
def test_a_binary_collation_is_code_point_order_so_it_tells_the_cases_apart(
    mssql,  # noqa: F811 - the fixture imported above
):
    """It restores punctuation-first, not the old case-insensitive order."""
    table, _ = _land(mssql, "twelve", _csv(_TWELVE))

    by_code_point = _ordered(table, "v COLLATE Latin1_General_BIN2")

    assert by_code_point != list(_VARCHAR_ORDER)
    assert by_code_point.index("Ab") < by_code_point.index("_x")
    assert by_code_point.index("aB") < by_code_point.index("ab")


@pytest.mark.endpoint
def test_a_binary_collation_keeps_text_the_code_page_lacks(
    mssql,  # noqa: F811 - the fixture imported above
):
    table, _ = _land(mssql, "cjk", _csv(("日本", "😀", "été")))

    assert sorted(_ordered(table, "v COLLATE Latin1_General_BIN2")) == sorted(
        ["日本", "😀", "été"]
    )


@pytest.mark.endpoint
def test_a_cast_back_to_varchar_restores_the_old_order_exactly(
    mssql,  # noqa: F811 - the fixture imported above
):
    table, _ = _land(mssql, "twelve", _csv(_TWELVE))

    assert _ordered(table, "CAST(v AS VARCHAR(10))") == list(_VARCHAR_ORDER)


@pytest.mark.endpoint
def test_a_cast_back_to_varchar_costs_the_characters_the_code_page_lacks(
    mssql,  # noqa: F811 - the fixture imported above
):
    table, _ = _land(mssql, "cjk", _csv(("日本", "été")))

    cast = _rows(f"SELECT CAST(v AS VARCHAR(10)) FROM {table} ORDER BY n")

    assert cast == [["??"], ["été"]]
