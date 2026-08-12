"""``stats`` — the per-column profile, and the two columns it must stay quiet about.

This verb exists because of one measured failure. A cold agent, given only the
skill and a live server, read ``info``'s ``mixed_columns: []`` as a clean bill of
health and reported 3,000 rows as trustworthy; 52 values in a ``REAL`` column
were null, and nothing in ``attach``, ``info`` or the documented checklist would
ever have said so (issue #94). ``info`` is a directory — sources, tables, schema
— and a directory that reports statistics stops being one, so the profile is a
verb of its own.

The interesting half of this file is not that the numbers arrive. It is the two
column kinds where an aggregate is **worse than silence**, both of which this
server already knows how to recognise and already warns about elsewhere:

* a **mixed** column, where ``avg()`` silently coerces text to 0 and keeps it in
  the denominator, so the average of 1..5 plus two text rows is 2.14 rather
  than 3.0;
* an **unparsed-temporal** column, where ``min()`` and ``max()`` compare
  alphabetically and return the wrong instant while looking like an answer.

Reporting either would be the exact silent-wrong-answer class the rest of this
server is built to refuse, and it would arrive under a verb whose whole promise
is that the numbers are true. So both are profiled to their null count and no
further, and the payload says why.

The stats that are *not* universal — median and standard deviation — are
governed by one rule of Chris's: **free we take, expensive we leave.** A function
the engine lacks is simply not reported. It is never emulated, never
approximated, and never computed in a second pass in Python.
"""

from __future__ import annotations

from pathlib import Path

import foreign
import pytest
from sqlalchemy import Integer, Text

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config


@pytest.fixture
def root(tmp_path):
    """A fresh registry over a directory the server may reach.

    ``resolve()`` matters: on macOS ``tmp_path`` lives under ``/var``, a symlink
    to ``/private/var``, and a root spelled the unresolved way puts every file in
    it outside the boundary — a fixture failure that presents as a passing
    refusal.
    """
    workspace = (tmp_path / "root").resolve()
    workspace.mkdir()
    config_module.use(Config(roots=(workspace,)))
    server_module._reset()
    yield workspace
    server_module._reset()


def attach_csv(root: Path, name: str, text: str) -> str:
    """Write a CSV, attach it, and return the nickname it actually got."""
    path = root / name
    path.write_text(text)
    attached = server_module.attach(database=str(path))
    assert attached["ok"] is True, attached
    return attached["nickname"]


def column(answer: dict, name: str) -> dict:
    """One column out of a stats payload, failing loudly if it is absent."""
    named = {c["name"]: c for c in answer["columns"]}
    assert name in named, f"{name!r} not profiled; got {sorted(named)}"
    return named[name]


# ---------------------------------------------------------------------------
# The floor — what every engine can answer
# ---------------------------------------------------------------------------


def test_a_numeric_column_reports_its_range_and_average(root):
    nickname = attach_csv(root, "amounts.csv", "amount\n10\n20\n30\n")

    answer = server_module.stats(nickname=nickname, table="amounts")

    assert answer["ok"] is True
    assert answer["rows"] == 3
    amount = column(answer, "amount")
    assert amount["nulls"] == 0
    assert amount["non_nulls"] == 3
    assert amount["min"] == 10
    assert amount["max"] == 30
    assert amount["avg"] == 20


def test_the_null_count_that_info_never_reported(root):
    """Issue #94, stated as the measurement that was missing.

    The positive control is the whole point: ``info`` is asserted to call this
    table clean *first*. Without it a regression that made ``info`` report nulls
    would leave this passing while testing nothing, and the finding it encodes —
    that a caller following the documented checklist is told nothing — would
    quietly stop being true.
    """
    # Two columns deliberately: a wholly blank line is skipped by the CSV reader
    # rather than read as a row of nulls, so the gap has to sit in a real row.
    rows = "\n".join(f"s{n}," + ("" if n % 10 == 0 else "1.5") for n in range(1, 101))
    nickname = attach_csv(root, "readings.csv", f"sensor,reading\n{rows}\n")

    directory = server_module.info(nickname=nickname, table="readings")
    assert directory["ok"] is True
    assert directory["mixed_columns"] == [], (
        "positive control failed — info is expected to report this column as "
        "unmixed, which is the misreading #94 is about."
    )
    assert not any("null" in str(v).lower() for v in directory.get("warnings", []))

    profiled = server_module.stats(nickname=nickname, table="readings")

    assert profiled["ok"] is True
    assert column(profiled, "reading")["nulls"] == 10
    assert column(profiled, "reading")["non_nulls"] == 90


def test_a_text_column_reports_its_null_count_and_nothing_else(root):
    """Strings get the floor only — no length statistics, by design.

    ``count(*) - count(col)`` is stock SQL on every engine this server reaches,
    which is what makes the null count the floor the whole design rests on.
    Anything more about a string is level 1.
    """
    nickname = attach_csv(root, "people.csv", "name,seat\nada,1\n,2\ngrace,3\n")

    answer = server_module.stats(nickname=nickname, table="people")

    name = column(answer, "name")
    assert name["nulls"] == 1
    assert name["non_nulls"] == 2
    assert "min" not in name
    assert "max" not in name
    assert "avg" not in name


def test_columns_narrows_the_profile(root):
    nickname = attach_csv(root, "wide.csv", "a,b,c\n1,2,3\n4,5,6\n")

    answer = server_module.stats(nickname=nickname, table="wide", columns=["b"])

    assert [c["name"] for c in answer["columns"]] == ["b"]
    assert column(answer, "b")["avg"] == 3.5


# ---------------------------------------------------------------------------
# The two columns an aggregate would lie about
# ---------------------------------------------------------------------------


def test_a_mixed_column_is_profiled_to_its_null_count_and_says_why(root):
    """``avg()`` here would be arithmetically real and factually wrong.

    Every value is stored as text, so the aggregate coerces the non-numeric ones
    to 0 and keeps them in the denominator. The number that comes back is not an
    approximation of the average — it is a different quantity wearing its name.
    """
    # "pending" rather than "n/a": pandas reads the latter as a *missing* value,
    # which would make this column cleanly numeric with one null and test the
    # opposite of what it says on the tin.
    nickname = attach_csv(root, "mixed.csv", "amount\n1\n2\n3\npending\n")

    answer = server_module.stats(nickname=nickname, table="mixed")

    amount = column(answer, "amount")
    assert amount["nulls"] == 0
    assert amount["non_nulls"] == 4
    assert "avg" not in amount, (
        "an average over a mixed column coerces text to 0 and keeps it in the "
        "denominator — reporting it under a verb that promises true numbers is "
        "the silent wrong answer this server refuses elsewhere."
    )
    assert "min" not in amount
    assert "withheld" in amount
    assert "mixed" in amount["withheld"]
    assert answer["warnings"], "the existing mixed-column explanation must reach here"


def test_a_date_column_in_no_standard_gets_no_range(root):
    """``min``/``max`` on alphabetical dates return the wrong instant, convincingly.

    ``30.11.2023`` sorts after ``01.03.2025``, so ``max()`` answers with the
    earliest row in the table. It is a real value from the real column, which is
    exactly what makes it dangerous.
    """
    nickname = attach_csv(
        root, "orders.csv", "ordered\n30.11.2023\n01.03.2025\n17.06.2024\n"
    )

    answer = server_module.stats(nickname=nickname, table="orders")

    ordered = column(answer, "ordered")
    assert ordered["non_nulls"] == 3
    assert "min" not in ordered
    assert "max" not in ordered
    assert "withheld" in ordered
    assert answer["warnings"]


def test_a_canonical_date_column_does_get_its_range(root):
    """The counterpart control: a column normalised to ISO 8601 compares correctly.

    Without this, the test above would pass just as well against an
    implementation that withheld the range from *every* text column, and the
    distinction the withholding is meant to draw would be untested.
    """
    nickname = attach_csv(
        root, "shipped.csv", "shipped\n2023-11-30\n2025-03-01\n2024-06-17\n"
    )

    answer = server_module.stats(nickname=nickname, table="shipped")

    shipped = column(answer, "shipped")
    assert shipped["min"] == "2023-11-30"
    assert shipped["max"] == "2025-03-01"
    assert "avg" not in shipped, "an average of dates is not a date"
    assert "withheld" not in shipped


# ---------------------------------------------------------------------------
# What the engine has, and what it does not
# ---------------------------------------------------------------------------


def test_sqlite_reports_no_median_or_standard_deviation(root):
    """Stock SQLite has neither function, so neither is reported.

    Not emulated and not approximated. The alternative — pulling the column into
    Python to sort it — is the second pass the cost rule exists to forbid, and it
    would make the verb's cost depend on the row count rather than on the engine.
    """
    nickname = attach_csv(root, "amounts.csv", "amount\n10\n20\n30\n")

    amount = column(server_module.stats(nickname=nickname, table="amounts"), "amount")

    assert "median" not in amount
    assert "stddev" not in amount
    assert amount["avg"] == 20, "the floor is still reported"


def test_duckdb_reports_the_median_and_standard_deviation_it_has(root):
    """The same verb against an engine that has both, which is what makes the
    absence above a fact about SQLite rather than about this implementation."""
    target = root / "warehouse.db"
    foreign.build_database(
        target,
        "sales",
        [("region", Text), ("amount", Integer)],
        [("north", 10), ("south", 20), ("east", 30)],
        dialect="duckdb",
    )
    attached = server_module.attach(database=str(target), nickname="wh")
    assert attached["ok"] is True

    amount = column(server_module.stats(nickname="wh", table="sales"), "amount")

    assert amount["min"] == 10
    assert amount["max"] == 30
    assert amount["avg"] == 20
    assert amount["median"] == 20
    assert amount["stddev"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_unattached_nickname_is_refused_as_a_payload(root):
    answer = server_module.stats(nickname="nope", table="whatever")

    assert answer["ok"] is False
    assert "nope" in answer["error"]


def test_a_missing_table_is_refused_naming_the_tables_that_exist(root):
    nickname = attach_csv(root, "people.csv", "name\nada\n")

    answer = server_module.stats(nickname=nickname, table="absent")

    assert answer["ok"] is False
    assert "people" in answer["error"]


def test_a_missing_column_is_refused_naming_the_columns_that_exist(root):
    """A wrong column name is the likeliest input to a verb that narrows.

    Dropping it silently would answer with a profile of everything else under
    ``ok: true``, which looks like the question was understood.
    """
    nickname = attach_csv(root, "people.csv", "name,age\nada,36\n")

    answer = server_module.stats(nickname=nickname, table="people", columns=["nmae"])

    assert answer["ok"] is False
    assert "nmae" in answer["error"]
    assert "name" in answer["error"]
