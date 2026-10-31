"""A typed date or time sharing a column with other values is stored as its text (#114).

A typed format can hand pandas a column that is part temporal and part not —
a YAML ``seen: 2024-03-01`` beside ``seen: unknown``, or a workbook column
holding a time cell and a datetime cell. That column is not one temporal kind,
so it is declared ``TEXT``, and each date or time in it used to reach the
database as its integer tick count written as digits: ``'1709251200000000000'``
for the first of March, with ``ok: true`` and nothing to say so.

Each such value is now written as the text it means, in the canonical spelling
of its kind — the spelling a column of only that kind is written in, and the
one a CSV holding the same text keeps. Driven over the protocol, because what
is stored is what an agent reads back.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path

import openpyxl
import pytest
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config


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


def attached_rows(path: Path, sql_column: str) -> tuple[list, dict]:
    """Attach ``path``, and return its one table's column values and description."""
    attached = call("attach", database=str(path), nickname="m")
    assert attached["ok"] is True, attached
    listed = call("directory", nickname="m")
    assert listed["ok"] is True, listed
    (table,) = listed["tables"]
    described = call("directory", nickname="m", table=table)
    assert described["ok"] is True, described
    (column,) = [c for c in described["columns"] if c["name"] == sql_column]
    answered = call(
        "query", nickname="m", sql=f"SELECT {sql_column} FROM {table} ORDER BY rowid"
    )
    assert answered["ok"] is True, answered
    return [row[0] for row in answered["rows"]], column


def test_a_yaml_date_beside_a_sentinel_is_stored_as_its_date(session):
    path = session / "mixed.yaml"
    path.write_text("- id: 1\n  seen: 2024-03-01\n- id: 2\n  seen: unknown\n")

    values, column = attached_rows(path, "seen")

    assert values == ["2024-03-01", "unknown"]
    assert column["type"] == "TEXT"
    assert "temporal" not in column


def test_a_yaml_instant_with_a_time_is_written_to_the_second_in_utc(session):
    """One spelling for the column's instants, measured over all of them.

    The date-only value becomes midnight because a neighbour carries a time of
    day, and the offset is normalised to UTC — the contract a column of only
    instants is written under.
    """
    path = session / "mixed.yaml"
    path.write_text(
        "- seen: 2024-03-01\n"
        "- seen: 2024-03-02T14:30:00+01:00\n"
        "- seen: n/a\n"
        "- seen: 7\n"
    )

    values, column = attached_rows(path, "seen")

    assert values == [
        "2024-03-01T00:00:00Z",
        "2024-03-02T13:30:00Z",
        "n/a",
        "7",
    ]
    assert column["type"] == "TEXT"


def test_the_stored_date_compares_against_a_date_literal(session):
    """The reason the spelling matters: a filter on the date finds it."""
    path = session / "mixed.yaml"
    path.write_text(
        "- seen: 2024-03-01\n- seen: 2024-03-05\n- seen: unknown\n- seen: 2024-02-01\n"
    )
    assert call("attach", database=str(path), nickname="m")["ok"] is True
    (table,) = call("directory", nickname="m")["tables"]

    found = call(
        "query",
        nickname="m",
        sql=f"SELECT seen FROM {table} WHERE seen BETWEEN '2024-03-01' AND "
        f"'2024-03-31' ORDER BY seen",
    )

    assert found["ok"] is True, found
    assert found["rows"] == [["2024-03-01"], ["2024-03-05"]]


def test_a_workbook_time_beside_a_datetime_is_stored_as_their_text(session):
    """The workbook case: a time cell and a cell openpyxl reads as a datetime."""
    path = session / "clock.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["clock"])
    sheet.append([dt.time(14, 30)])
    sheet.append([dt.datetime(1900, 1, 1, 6, 0)])
    book.save(path)

    values, column = attached_rows(path, "clock")

    assert values == ["14:30:00", "1900-01-01T06:00:00Z"]
    assert column["type"] == "TEXT"
