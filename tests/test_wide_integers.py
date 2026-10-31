"""An integer too wide for 64 bits loads as its digits, and says so (#72).

``read_csv`` reads a column of integers too wide for int64 as Python ints in an
``object`` column, which is declared ``TEXT``. The value used to be refused at
insert for being wider than a 64-bit column — but no 64-bit column was
involved, and a text column holds the digits losslessly. It now loads as the
digits the file held, and the answer warns that the column is text, because
``sum()`` over it converts each value to a float and loses its last digits.

Driven over the protocol with the issue's own file.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config

WIDE = "99999999999999999999"


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


def test_the_issues_file_loads_with_its_digits_as_text(session):
    path = session / "wide.csv"
    path.write_text(f"id,v\n1,1\n2,2\n3,{WIDE}\n")

    attached = call("attach", database=str(path), nickname="w", delimiter=",")

    assert attached["ok"] is True, attached
    (table,) = attached["tables"]
    described = call("directory", nickname="w", table=table)
    (column,) = [c for c in described["columns"] if c["name"] == "v"]
    assert column["type"] == "TEXT"
    answered = call("query", nickname="w", sql=f"SELECT v FROM {table} ORDER BY id")
    assert answered["rows"] == [["1"], ["2"], [WIDE]]
    found = call(
        "query", nickname="w", sql=f"SELECT id FROM {table} WHERE v = '{WIDE}'"
    )
    assert found["rows"] == [[3]]


def test_the_answer_warns_that_the_column_is_text(session):
    """An agent running ``sum()`` on the column has to know what it is summing."""
    path = session / "wide.csv"
    path.write_text(f"id,v\n1,1\n2,2\n3,{WIDE}\n")

    attached = call("attach", database=str(path), nickname="w", delimiter=",")

    warned = [w for w in attached.get("warnings", []) if "64-bit" in w]
    assert len(warned) == 1, attached.get("warnings")
    (warning,) = warned
    assert "v" in warning and WIDE in warning and "text" in warning
    assert "sum()" in warning


def test_a_column_of_ordinary_integers_carries_no_such_warning(session):
    path = session / "narrow.csv"
    path.write_text("id,v\n1,1\n2,2\n3,9223372036854775807\n")

    attached = call("attach", database=str(path), nickname="w", delimiter=",")

    assert attached["ok"] is True, attached
    assert not [w for w in attached.get("warnings", []) if "64-bit" in w]
    (table,) = attached["tables"]
    (column,) = [
        c
        for c in call("directory", nickname="w", table=table)["columns"]
        if c["name"] == "v"
    ]
    assert column["type"] == "INTEGER"


def test_create_from_the_file_warns_as_attach_does(session):
    base = session / "base.csv"
    base.write_text("id\n1\n")
    path = session / "wide.csv"
    path.write_text(f"id,v\n1,1\n2,2\n3,{WIDE}\n")
    assert call("attach", database=str(base), nickname="w", delimiter=",")["ok"]

    created = call(
        "create", nickname="w", type="table", source=str(path), delimiter=","
    )

    assert created["ok"] is True, created
    assert [w for w in created.get("warnings", []) if "64-bit" in w], created
