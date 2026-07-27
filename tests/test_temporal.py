"""Dates read from a file, asserted the way CONSTRAINTS §5.1 requires.

Every assertion is on what a query **returns** — the order rows come back in,
what ``max()`` answers, whether an equality join matches — and never on the
declared type or on the absence of an exception. A column can be declared
anything and still order backwards; the declaration is not the property.

The instants below span three years on purpose. ``MM/DD/YYYY`` and
``DD.MM.YYYY`` both sort correctly *by accident* inside a single year, so a
one-year fixture scores a broken format as working — which is how the live
sweep was designed and why it found what it found (CONSTRAINTS §8.1).
"""

from __future__ import annotations

import pandas as pd
import pytest

from localdata_mcp import temporal
from localdata_mcp.loader import Workspace

TAG = "bulk"

#: Five instants, and the label of each is its chronological rank.
INSTANTS = [
    ("rank1", "2023-11-30T00:01:00"),
    ("rank2", "2024-01-05T09:15:00"),
    ("rank3", "2024-07-04T23:59:00"),
    ("rank4", "2025-03-01T14:30:00"),
    ("rank5", "2025-12-25T12:00:00"),
]
#: Written in an order that is neither chronological nor reverse, so a column
#: that does not sort at all cannot be scored correct by the order it arrived.
SHUFFLED = [INSTANTS[i] for i in (3, 0, 4, 1, 2)]
CHRONOLOGICAL = [label for label, _ in INSTANTS]


@pytest.fixture()
def workspace():
    space = Workspace.in_memory()
    space.attach_memory(TAG)
    yield space
    space.close()


def load(workspace: Workspace, values: list[str], column: str = "v") -> None:
    frame = pd.DataFrame(
        {"label": [label for label, _ in SHUFFLED], column: values},
    )
    workspace.insert_frame(temporal.standardize(frame), "t", source="test", tag=TAG)


def labels_in_order(workspace: Workspace, column: str = "v") -> list[str]:
    _, rows = workspace.query(TAG, f"SELECT label FROM t ORDER BY {column}")
    return [row[0] for row in rows]


def spell(fmt: str) -> list[str]:
    """The five instants, in file order, written the given way."""
    return [pd.Timestamp(iso).strftime(fmt) for _, iso in SHUFFLED]


# ---------------------------------------------------------------------------
# The standards are recognised, and they answer correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt",
    [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",  # RFC 3339's space, what pandas and SQL emit
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S.%f",
    ],
)
def test_an_iso_8601_column_orders_chronologically(workspace, fmt):
    load(workspace, spell(fmt))
    assert labels_in_order(workspace) == CHRONOLOGICAL


@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"])
def test_max_of_an_iso_8601_column_is_the_latest_instant(workspace, fmt):
    """The failure this guards against returns a real value that is the wrong row."""
    load(workspace, spell(fmt))
    _, rows = workspace.query(
        TAG, "SELECT label FROM t WHERE v = (SELECT max(v) FROM t)"
    )
    assert [row[0] for row in rows] == ["rank5"]


def test_the_same_instant_in_two_offsets_compares_equal(workspace):
    """The join that returned zero rows through the live surface (§8.1).

    The test that already existed for this handed `insert_frame` two tz-aware
    Timestamps, which no reader can produce; these are the strings a file
    actually carries.
    """
    frame = pd.DataFrame(
        {
            "utc": ["2024-11-03T06:30:00+00:00"],
            "local": ["2024-11-03T01:30:00-05:00"],
        }
    )
    workspace.insert_frame(temporal.standardize(frame), "pair", source="test", tag=TAG)
    _, rows = workspace.query(TAG, "SELECT count(*) FROM pair WHERE utc = local")
    assert rows[0][0] == 1


def test_a_range_filter_on_an_iso_column_selects_the_right_rows(workspace):
    load(workspace, spell("%Y-%m-%d"))
    _, rows = workspace.query(TAG, "SELECT count(*) FROM t WHERE v > '2025-01-01'")
    assert rows[0][0] == 2


def test_an_iso_column_is_reported_as_the_standard_it_is_in(workspace):
    load(workspace, spell("%Y-%m-%dT%H:%M:%SZ"))
    described = workspace.describe(TAG, "t")
    column = next(c for c in described.columns if c.name == "v")
    assert column.temporal_standard == "iso8601_utc"
    assert not column.is_unparsed_temporal


def test_an_offset_is_normalized_to_utc_in_the_stored_value(workspace):
    """The transformation the contract has to declare, asserted on the value."""
    frame = pd.DataFrame({"v": ["2024-03-01T14:30:00+01:00"]})
    workspace.insert_frame(temporal.standardize(frame), "one", source="test", tag=TAG)
    _, rows = workspace.query(TAG, "SELECT v FROM one")
    assert rows[0][0] == "2024-03-01T13:30:00Z"


def test_a_date_only_column_keeps_the_date_form_it_arrived_in(workspace):
    """Nobody wants a date column to grow a midnight it never had."""
    load(workspace, spell("%Y-%m-%d"))
    _, rows = workspace.query(TAG, "SELECT v FROM t ORDER BY v LIMIT 1")
    assert rows[0][0] == "2023-11-30"


# ---------------------------------------------------------------------------
# Ambiguity is refused rather than guessed at
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt",
    ["%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%b %d, %Y", "%d %B %Y"],
)
def test_an_ambiguous_spelling_is_left_as_text_and_reported(workspace, fmt):
    """Not parsed — `01/03/2025` is March or January and the file does not say.

    What must not happen is silence: the column stays text, so it still orders
    alphabetically, and the caller has to be told that before it trusts a
    `max()`.
    """
    values = spell(fmt)
    load(workspace, values)

    described = workspace.describe(TAG, "t")
    column = next(c for c in described.columns if c.name == "v")
    assert column.declared_type == "TEXT"
    assert column.is_unparsed_temporal
    assert set(column.unparsed_temporal_examples) <= set(values)
    assert described.unparsed_temporal_columns == ["v"]


def test_iso_8601_basic_format_is_not_treated_as_a_date(workspace):
    """`20240301` is also an order number, and pandas would accept it as a date."""
    load(workspace, ["20240301", "20231130", "20251225", "20240105", "20250704"])
    described = workspace.describe(TAG, "t")
    column = next(c for c in described.columns if c.name == "v")
    assert column.temporal_standard is None
    _, rows = workspace.query(TAG, "SELECT v FROM t ORDER BY v LIMIT 1")
    assert rows[0][0] == "20231130"


def test_a_column_only_partly_iso_is_left_alone_entirely(workspace):
    """Parsing the part that matches would turn the rest into NULL."""
    values = ["2025-03-01", "30.11.2023", "2025-12-25", "2024-01-05", "2024-07-04"]
    load(workspace, values)
    _, rows = workspace.query(TAG, "SELECT count(*) FROM t WHERE v IS NOT NULL")
    assert rows[0][0] == 5


def test_a_spelling_that_orders_correctly_anyway_is_not_warned_about(workspace):
    """Quarters and ISO week dates sort chronologically as text.

    Warning about them would be noise, and noise in this channel costs the
    warnings that matter their credibility.
    """
    load(workspace, ["2025-Q1", "2023-Q4", "2025-Q4", "2024-Q1", "2024-Q3"])
    described = workspace.describe(TAG, "t")
    assert described.unparsed_temporal_columns == []


def test_an_ordinary_text_column_is_neither_parsed_nor_warned_about(workspace):
    load(workspace, ["north", "south", "east", "west", "north"])
    described = workspace.describe(TAG, "t")
    column = next(c for c in described.columns if c.name == "v")
    assert column.temporal_standard is None
    assert not column.is_unparsed_temporal


def test_an_empty_column_is_not_declared_temporal_on_no_evidence(workspace):
    load(workspace, [None, None, None, None, None])
    described = workspace.describe(TAG, "t")
    column = next(c for c in described.columns if c.name == "v")
    assert column.temporal_standard is None
    assert not column.is_unparsed_temporal
