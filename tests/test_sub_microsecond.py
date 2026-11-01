"""An instant cut to the microsecond is reported, not cut silently (#111).

The canonical date text keeps six fractional digits, because most backends this
server writes to store a timestamp to the microsecond (PostgreSQL `timestamp`,
MySQL `DATETIME(6)`, DuckDB `TIMESTAMP`), so a nine-digit spelling could not
survive a load in general. What was wrong was that the cut said nothing: two
instants a nanosecond apart became one value, and `count(DISTINCT v)`, an
equality join and `ORDER BY` between them all answered as if they were one,
with no warning anywhere.

The cut stays, and the answer now names the column it happened to. Every
fixture here is checked to carry the condition it is named for — a Parquet
column really `timestamp[ns]` with a non-zero nanosecond part — because a
fixture that quietly lost its nanoseconds on the way to disk would leave every
test green.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import loader as loader_module
from localdata_mcp import server as server_module
from localdata_mcp import temporal
from localdata_mcp.config import Config
from localdata_mcp.loader import Workspace, read_file, read_source

#: The issue's two instants, one nanosecond apart.
APART = ["2024-03-01T00:00:00.123456789Z", "2024-03-01T00:00:00.123456788Z"]
#: Both of them, as the canonical text keeps them.
CUT = "2024-03-01T00:00:00.123456Z"

#: The words every cut warning carries, and that no other warning does.
MARKER = "microsecond"


@pytest.fixture(autouse=True)
def session(tmp_path):
    """A fresh registry and a fresh allowed root for every test."""
    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,)))
    server_module._reset()
    yield root
    server_module._reset()


def call(tool: str, **arguments):
    """Call a tool over the protocol and return its decoded payload."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


def cut_warnings(payload: dict) -> list[str]:
    """The warnings in an answer that report a cut to the microsecond."""
    return [w for w in payload.get("warnings", []) if MARKER in w]


def write_csv(root: Path, name: str, values: list[str]) -> Path:
    path = root / name
    path.write_text("id,v\n" + "".join(f"{i},{v}\n" for i, v in enumerate(values)))
    return path


def write_parquet_ns(root: Path, name: str, nanoseconds: list[int]) -> Path:
    """A Parquet file whose `v` is `timestamp[ns]`, read back to prove it."""
    path = root / name
    table = pa.table(
        {
            "id": pa.array(range(len(nanoseconds)), pa.int64()),
            "v": pa.array(nanoseconds, pa.timestamp("ns", tz="UTC")),
        }
    )
    pq.write_table(table, path)
    stored = pq.read_table(path).column("v")
    assert stored.type == pa.timestamp("ns", tz="UTC"), stored.type
    assert stored.cast(pa.int64()).to_pylist() == nanoseconds
    return path


#: 2024-03-01T00:00:00Z as nanoseconds since the epoch.
MIDNIGHT_NS = 1_709_251_200_000_000_000


# ---------------------------------------------------------------------------
# temporal: the cut is measured where it is made
# ---------------------------------------------------------------------------


def test_the_issues_repro_still_cuts_and_now_names_the_column():
    frame = pd.DataFrame({"v": APART, "other": ["a", "b"]})

    written, cut = temporal.standardize_with_cuts(frame)

    assert written.v.tolist() == [CUT, CUT]
    assert cut == ("v",)
    # The plain entry point writes the same values, so nothing that already
    # called it changes.
    assert temporal.standardize(frame).v.tolist() == [CUT, CUT]


def test_a_column_at_the_microsecond_is_not_reported():
    frame = pd.DataFrame({"v": ["2024-03-01T00:00:00.123456Z", "2024-03-02"]})

    _, cut = temporal.standardize_with_cuts(frame)

    assert cut == ()


def test_a_typed_nanosecond_column_is_reported():
    frame = pd.DataFrame(
        {"v": pd.to_datetime([MIDNIGHT_NS + 123_456_789, MIDNIGHT_NS], utc=True)}
    )
    assert frame.v.dtype == "datetime64[ns, UTC]"

    written, cut = temporal.standardize_with_cuts(frame)

    assert written.v.tolist() == [CUT, "2024-03-01T00:00:00.000000Z"]
    assert cut == ("v",)


def test_a_typed_column_of_whole_microseconds_is_not_reported():
    """`timestamp[ns]` is the dtype, not the evidence: only a non-zero part is."""
    frame = pd.DataFrame(
        {"v": pd.to_datetime([MIDNIGHT_NS + 123_456_000, MIDNIGHT_NS], utc=True)}
    )

    _, cut = temporal.standardize_with_cuts(frame)

    assert cut == ()


def test_an_instant_beside_other_values_is_reported_when_cut():
    """The mixed column of #114 writes its instants through the same cut."""
    frame = pd.DataFrame(
        {"v": [pd.Timestamp(MIDNIGHT_NS + 123_456_789, tz="UTC"), "unknown"]},
        dtype=object,
    )

    written, cut = temporal.standardize_with_cuts(frame)

    assert written.v.tolist() == [CUT, "unknown"]
    assert cut == ("v",)


def test_a_fraction_only_below_the_microsecond_is_spelled_as_the_cut_value():
    """What is stored is the instant after the cut, in that instant's spelling.

    Cut to the microsecond these are whole seconds, and whole seconds are
    written `…:00Z` — the spelling a file holding the same instants at the
    second gets, so a join between the two still matches. Six zeroes would be
    a second spelling of the same instant (the `.000000` that #109 refuses for
    a time of day). The warning is what keeps the cut from being silent.
    """
    frame = pd.DataFrame(
        {"v": ["2024-03-01T10:00:00.000000005Z", "2024-03-02T10:00:00Z"]}
    )

    written, cut = temporal.standardize_with_cuts(frame)

    assert written.v.tolist() == ["2024-03-01T10:00:00Z", "2024-03-02T10:00:00Z"]
    assert cut == ("v",)


def test_instants_that_are_midnight_once_cut_are_written_as_dates():
    """The same rule one step further: all midnight after the cut is date-only."""
    frame = pd.DataFrame(
        {"v": ["2024-03-01T00:00:00.000000005Z", "2024-03-02T00:00:00Z"]}
    )

    written, cut = temporal.standardize_with_cuts(frame)

    assert written.v.tolist() == ["2024-03-01", "2024-03-02"]
    assert cut == ("v",)


# ---------------------------------------------------------------------------
# Over the protocol: attach and create say so
# ---------------------------------------------------------------------------


def test_attaching_the_issues_csv_warns_and_names_the_column(session):
    path = write_csv(session, "apart.csv", APART)

    attached = call("attach", database=str(path), nickname="d", delimiter=",")

    assert attached["ok"] is True, attached
    (warning,) = cut_warnings(attached)
    assert "Column v" in warning
    assert "compare equal" in warning
    (table,) = attached["tables"]
    answered = call("query", nickname="d", sql=f"SELECT DISTINCT v FROM {table}")
    assert answered["rows"] == [[CUT]]


def test_attaching_a_parquet_nanosecond_column_warns(session):
    path = write_parquet_ns(
        session, "apart.parquet", [MIDNIGHT_NS + 123_456_789, MIDNIGHT_NS + 123_456_788]
    )

    attached = call("attach", database=str(path), nickname="d")

    assert attached["ok"] is True, attached
    (warning,) = cut_warnings(attached)
    assert "Column v" in warning
    (table,) = attached["tables"]
    answered = call("query", nickname="d", sql=f"SELECT v FROM {table} ORDER BY id")
    assert answered["rows"] == [[CUT], [CUT]]


def test_a_parquet_nanosecond_column_of_whole_microseconds_does_not_warn(session):
    path = write_parquet_ns(
        session, "even.parquet", [MIDNIGHT_NS + 123_456_000, MIDNIGHT_NS]
    )

    attached = call("attach", database=str(path), nickname="d")

    assert attached["ok"] is True, attached
    assert cut_warnings(attached) == []


def test_a_csv_at_the_microsecond_does_not_warn(session):
    path = write_csv(session, "even.csv", [CUT, "2024-03-02T00:00:00Z"])

    attached = call("attach", database=str(path), nickname="d", delimiter=",")

    assert attached["ok"] is True, attached
    assert cut_warnings(attached) == []


def test_a_csv_whose_only_fraction_is_below_the_microsecond_warns(session):
    path = write_csv(
        session, "tiny.csv", ["2024-03-01T10:00:00.000000005Z", "2024-03-02T10:00:00Z"]
    )

    attached = call("attach", database=str(path), nickname="d", delimiter=",")

    assert attached["ok"] is True, attached
    assert len(cut_warnings(attached)) == 1
    (table,) = attached["tables"]
    answered = call("query", nickname="d", sql=f"SELECT v FROM {table} ORDER BY id")
    assert answered["rows"] == [["2024-03-01T10:00:00Z"], ["2024-03-02T10:00:00Z"]]


def test_create_from_the_file_warns_as_attach_does(session):
    base = session / "base.csv"
    base.write_text("id\n1\n")
    path = write_csv(session, "apart.csv", APART)
    assert call("attach", database=str(base), nickname="d", delimiter=",")["ok"]

    created = call(
        "create", nickname="d", type="table", source=str(path), delimiter=","
    )

    assert created["ok"] is True, created
    assert len(cut_warnings(created)) == 1, created


def test_a_workbook_warning_names_the_sheet_it_is_about(session):
    """A workbook's notes reach every sheet, so the note says which one it means."""
    path = session / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"v": APART}).to_excel(writer, sheet_name="sensors", index=False)
        pd.DataFrame({"v": ["a", "b"]}).to_excel(
            writer, sheet_name="plain", index=False
        )

    attached = call("attach", database=str(path), nickname="d")

    assert attached["ok"] is True, attached
    warned = set(cut_warnings(attached))
    assert len(warned) == 1, attached.get("warnings")
    (warning,) = warned
    assert "Column v" in warning and "sensors" in warning


# ---------------------------------------------------------------------------
# Both read paths say the same thing
# ---------------------------------------------------------------------------


@pytest.fixture()
def chunked(request, monkeypatch):
    monkeypatch.setattr(loader_module, "_READ_CHUNK", request.param)
    return request.param


@pytest.fixture()
def workspace():
    space = Workspace.in_memory()
    space.attach_memory("whole")
    space.attach_memory("parts")
    yield space
    space.close()


#: The nanosecond value first, last and alone in its chunk, so a fold that kept
#: only one chunk's answer would miss it in one of them.
BOTH_PATHS = {
    "cut_first.csv": [APART[0], "2024-03-02T00:00:00Z", "2024-03-03T00:00:00Z"],
    "cut_last.csv": ["2024-03-02T00:00:00Z", "2024-03-03T00:00:00Z", APART[1]],
    "cut_tiny.csv": ["2024-03-01T10:00:00Z", "2024-03-01T10:00:00.000000005Z"],
    "not_cut.csv": [CUT, "2024-03-02"],
}


@pytest.mark.parametrize("chunked", [1, 2, 5_000], indirect=True)
@pytest.mark.parametrize("name", sorted(BOTH_PATHS))
def test_both_read_paths_report_the_cut_identically(session, workspace, chunked, name):
    path = write_csv(session, name, BOTH_PATHS[name])

    read = read_file(path, delimiter=",")
    whole = workspace.insert_frame(
        read.tables[0].frame, "t", source=str(path), tag="whole", notes=read.notes
    )
    measured = read_source(path, delimiter=",")
    parts = workspace.insert_source(
        measured.tables[0], "t", source=str(path), tag="parts", notes=measured.notes
    )

    assert parts.notes == whole.notes
    assert parts.columns == whole.columns
    _, whole_rows = workspace.query("whole", "SELECT * FROM t ORDER BY rowid")
    _, part_rows = workspace.query("parts", "SELECT * FROM t ORDER BY rowid")
    assert part_rows == whole_rows
    reported = [n for n in whole.notes if MARKER in n]
    assert len(reported) == (0 if name == "not_cut.csv" else 1), whole.notes
