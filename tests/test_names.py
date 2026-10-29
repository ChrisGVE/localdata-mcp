"""The one rule that turns a file's names into distinct SQL names.

Columns and tables both go through it, so a hole here is a hole in both: before
#91 the table side had no rule at all, and the column side gave ``a_1`` twice
for ``a, A, a_1``.
"""

from __future__ import annotations

from localdata_mcp.names import clashes, sanitize, unique


def test_the_first_holder_keeps_the_bare_name():
    assert unique(["a", "a", "a"]) == ["a", "a_1", "a_2"]


def test_a_suffix_skips_a_name_already_held():
    assert unique(["a", "a", "a_1"]) == ["a", "a_1", "a_1_1"]
    assert unique(["a_1", "a", "a"]) == ["a_1", "a", "a_2"]


def test_many_identical_names_all_come_out_distinct():
    names = unique(["x"] * 10_000)
    assert len(set(names)) == 10_000
    assert names[-1] == "x_9999"


def test_names_that_clean_up_alike_are_reported_with_who_holds_them():
    raw = ["Q2 Prices", "Q2-Prices", "Other", "q2.prices"]
    wanted = [sanitize(name, "table") for name in raw]
    given = unique(wanted)

    assert given == ["q2_prices", "q2_prices_1", "other", "q2_prices_2"]
    assert clashes(wanted, given) == [("q2_prices", [0, 1, 3])]


def test_distinct_names_report_no_clash():
    wanted = ["staff", "departments"]
    assert clashes(wanted, unique(wanted)) == []
