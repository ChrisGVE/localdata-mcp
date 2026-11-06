"""A UTF-8 byte-order mark at the start of a JSON file is skipped, not refused (#154).

Windows editors, Excel and PowerShell write UTF-8 with a byte-order mark, and
RFC 8259 §8.1 lets a JSON parser ignore one. ``.csv``, ``.tsv``, ``.yaml`` and
``.xml`` already read such a file with a clean header; ``.json``, ``.jsonl`` and
``.ndjson`` refused it with a codec message ("Unexpected UTF-8 BOM (decode using
utf-8-sig)") that names Python's decoder, not anything an agent can change about
the file.

Driven over the protocol, because the column name an agent reads back is the
thing a leftover mark would corrupt: ``\\ufeffa`` looks like ``a`` in every
answer and matches nothing in SQL.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config

BOM = b"\xef\xbb\xbf"


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


def columns_and_rows(path: Path) -> tuple[list[str], list]:
    """Attach ``path`` and return its one table's column names and rows."""
    attached = call("attach", database=str(path), nickname="b")
    assert attached["ok"] is True, attached
    listed = call("directory", nickname="b")
    assert listed["ok"] is True, listed
    (table,) = listed["tables"]
    described = call("directory", nickname="b", table=table)
    assert described["ok"] is True, described
    answered = call("query", nickname="b", sql=f"SELECT a, b FROM {table} ORDER BY a")
    assert answered["ok"] is True, answered
    return [c["name"] for c in described["columns"]], answered["rows"]


def test_a_json_array_with_a_byte_order_mark_attaches_with_a_clean_header(session):
    path = session / "marked.json"
    path.write_bytes(BOM + b'[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]')

    names, rows = columns_and_rows(path)

    assert names == ["a", "b"]
    assert rows == [[1, "x"], [2, "y"]]


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_json_lines_with_a_byte_order_mark_attaches_with_a_clean_header(
    session, suffix
):
    """The mark sits before the first record only, so line 1 is where it bit."""
    path = session / f"marked{suffix}"
    path.write_bytes(BOM + b'{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')

    names, rows = columns_and_rows(path)

    assert names == ["a", "b"]
    assert rows == [[1, "x"], [2, "y"]]


def test_a_mark_inside_a_json_lines_file_is_still_refused(session):
    """Only a mark at the start of the file is a byte-order mark.

    One at the start of line 2 is a stray character in the middle of the data,
    and JSON Lines has no rule allowing it, so it is refused by line number as
    before.
    """
    path = session / "stray.jsonl"
    path.write_bytes(b'{"a": 1, "b": "x"}\n' + BOM + b'{"a": 2, "b": "y"}\n')

    attached = call("attach", database=str(path), nickname="b")

    assert attached["ok"] is False, attached
    assert "line 2" in json.dumps(attached)


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("marked.csv", b"a,b\n1,x\n2,y\n"),
        ("marked.tsv", b"a\tb\n1\tx\n2\ty\n"),
        ("marked.yaml", b"- a: 1\n  b: x\n- a: 2\n  b: y\n"),
        (
            "marked.xml",
            b"<rows><row><a>1</a><b>x</b></row><row><a>2</a><b>y</b></row></rows>",
        ),
    ],
)
def test_the_other_text_formats_keep_skipping_a_byte_order_mark(session, name, body):
    """The behaviour the JSON readers were brought in line with, pinned so it stays."""
    path = session / name
    path.write_bytes(BOM + body)

    names, rows = columns_and_rows(path)

    assert names == ["a", "b"]
    assert rows == [[1, "x"], [2, "y"]]
