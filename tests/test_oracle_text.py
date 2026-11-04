"""How a loaded text column lands on Oracle, and what it reads back (#132).

A text column lands as ``VARCHAR2(n)``, with ``n`` the longest value measured in
characters. Written without a qualifier, ``VARCHAR2(n)`` takes the session's
``NLS_LENGTH_SEMANTICS``, which is ``BYTE`` unless somebody changed it, and in
an AL32UTF8 database ``été`` is three characters and five bytes. So the column
the server sized for a value may be too narrow, in bytes, to hold it.

The live tests here run only against the Oracle target; the unit tests at the
top need no container at all.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from endpoints import TARGETS, Target, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import DDL, create_engine, inspect
from sqlalchemy.dialects import oracle as oracle_dialect

from localdata_mcp import config as config_module
from localdata_mcp import loader as loader_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import _VARCHAR2_MAX, backend_for

# ---------------------------------------------------------------------------
# What the declaration says — no container needed
# ---------------------------------------------------------------------------


def _rendered(type_) -> str:
    """The column type as Oracle's own dialect would write it in DDL."""
    return type_.compile(dialect=oracle_dialect.dialect())


def test_a_loaded_text_column_is_sized_in_characters_not_bytes():
    """``VARCHAR2(n CHAR)`` counts characters, which is what was measured."""
    backend = backend_for("oracle")

    assert _rendered(backend.column_type("TEXT", longest=11)) == "VARCHAR2(11 CHAR)"
    assert _rendered(backend.column_type("TEXT", longest=_VARCHAR2_MAX)) == (
        f"VARCHAR2({_VARCHAR2_MAX} CHAR)"
    )


def test_text_past_the_widest_varchar2_is_still_a_clob():
    backend = backend_for("oracle")

    assert _rendered(backend.column_type("TEXT", longest=_VARCHAR2_MAX + 1)) == "CLOB"


#: Each value with its width in characters and in UTF-8 bytes.
_UTF8 = {
    "plain": (5, 5),
    "été": (3, 5),
    "日本": (2, 6),
    "😀": (1, 4),
    "aé日😀": (4, 10),
}


@pytest.mark.parametrize("value", sorted(_UTF8))
def test_a_whole_frame_measures_the_widest_value_in_utf8_bytes(value):
    (column,) = loader_module.measure_frame(pd.DataFrame({"word": ["x", value]}))

    assert (column.longest, column.longest_utf8) == _UTF8[value]


def test_a_chunked_scan_measures_utf8_bytes_across_chunks():
    """The widest value in bytes need not be the widest in characters."""
    scan = loader_module._ColumnScan("word")
    scan.observe(pd.Series(["abcd"], dtype=object))
    scan.observe(pd.Series(["日日"], dtype=object))

    column = scan.finish()
    assert (column.longest, column.longest_utf8) == (4, 6)


def test_a_column_holding_no_text_has_no_utf8_width():
    (column,) = loader_module.measure_frame(pd.DataFrame({"n": [1, 2]}))

    assert column.longest_utf8 is None


def _declared_for(values: list[str]) -> str:
    """The DDL a column of these values gets on Oracle, measured as the loader does."""
    backend = backend_for("oracle")
    (column,) = loader_module.measure_frame(pd.DataFrame({"body": values}))
    width = backend.text_width(
        characters=column.longest,
        utf16_units=column.longest_utf16,
        utf8_bytes=column.longest_utf8,
    )
    return _rendered(backend.column_type("TEXT", longest=width))


def test_text_inside_4000_bytes_is_sized_in_characters():
    """2,000 ``é`` are 4,000 bytes: the widest ``VARCHAR2``, and still one."""
    assert _declared_for(["é" * 2000, "short"]) == "VARCHAR2(2000 CHAR)"


def test_text_past_4000_bytes_is_a_clob_however_few_characters_it_is():
    """1,400 CJK ideographs are 4,200 bytes, past what any ``VARCHAR2`` here holds.

    Oracle refuses the insert with the column sized in characters, measured
    (#132): the 4,000 ceiling counts bytes whatever the length semantics.
    """
    assert _declared_for(["日" * 1400, "short"]) == "CLOB"


def test_one_byte_past_the_ceiling_is_a_clob():
    assert _declared_for(["é" * 2000 + "x"]) == "CLOB"


# ---------------------------------------------------------------------------
# Live, against the Oracle container
# ---------------------------------------------------------------------------


def _oracle_targets() -> list[Target]:
    targets = [target for target in TARGETS if target.endpoint.dialect == "oracle"]
    assert targets, "the endpoint matrix lost its Oracle target"
    return targets


def call(tool: str, **arguments):
    """Call a tool over the protocol and return its decoded payload.

    Duplicated rather than imported from another test module, which would
    collect that module's fixtures twice.
    """

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


@dataclass(frozen=True)
class Oracle:
    url: str
    root: Path
    prefix: str

    def table(self, name: str) -> str:
        return f"{self.prefix}_{name}"


@pytest.fixture(params=_oracle_targets(), ids=lambda target: target.name)
def oracle(request, tmp_path):
    """An Oracle target, a fresh session, and no tables left behind."""
    target: Target = request.param
    try:
        reached = reach(target)
    except Unavailable as exc:
        pytest.skip(str(exc))

    root = tmp_path / "root"
    root.mkdir()
    config_module.use(Config(roots=(root,), network_enabled=True))
    server_module._reset()
    prefix = f"t{uuid.uuid4().hex[:8]}"
    with applied(reached.environ):
        try:
            payload = call(
                "attach", database=reached.url, nickname="ora", writable=True
            )
            assert payload["ok"] is True, payload
            yield Oracle(url=reached.url, root=root, prefix=prefix)
        finally:
            server_module._reset()
            _drop_tables_named(reached.url, prefix)


def _drop_tables_named(url: str, prefix: str) -> None:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            names = [
                name
                for name in inspect(conn).get_table_names()
                if name.lower().startswith(prefix)
            ]
        if not names:
            return
        with engine.begin() as conn:
            preparer = conn.dialect.identifier_preparer
            for name in names:
                conn.execute(DDL(f"DROP TABLE {preparer.quote(name)}"))
    finally:
        engine.dispose()


def _land(oracle: Oracle, name: str, csv_text: str) -> tuple[str, dict]:
    """Put a CSV into the endpoint through ``create``, the server's own verb."""
    path = oracle.root / f"{name}.csv"
    path.write_text(csv_text, encoding="utf-8")
    table = oracle.table(name)
    made = call(
        "create",
        nickname="ora",
        type="table",
        source=str(path),
        table=table,
        delimiter=",",
    )
    assert made["ok"] is True, made
    return table, made


def _rows(sql: str) -> list[list]:
    answer = call("query", nickname="ora", sql=sql)
    assert answer["ok"] is True, answer
    return answer["rows"]


def _declared(table: str, column: str) -> tuple[str, int, str]:
    """The column's type, its width in characters, and what that width counts.

    ``CHAR_USED`` is ``C`` for a width in characters and ``B`` for one in bytes.
    """
    ((data_type, width, unit),) = _rows(
        "SELECT data_type, char_length, char_used FROM user_tab_columns "
        f"WHERE table_name = '{table.upper()}' AND column_name = '{column.upper()}'"
    )
    return data_type, width, unit


#: One value per kind of character: plain ASCII, a Latin-1 accent (two bytes in
#: UTF-8), two CJK ideographs (three bytes each), and an emoji (four bytes).
_WORDS = ("plain", "été", "日本", "😀")


@pytest.mark.endpoint
def test_the_database_counts_text_in_bytes_by_default(oracle):
    """The premise of #132, read from the engine: the session's length semantics."""
    assert _rows(
        "SELECT value FROM nls_session_parameters "
        "WHERE parameter = 'NLS_LENGTH_SEMANTICS'"
    ) == [["BYTE"]]
    assert _rows(
        "SELECT value FROM nls_database_parameters WHERE parameter = 'NLS_CHARACTERSET'"
    ) == [["AL32UTF8"]]


@pytest.mark.endpoint
def test_non_ascii_text_at_the_columns_width_reads_back_exactly(oracle):
    """Each value is as wide as the column; every one comes back byte for byte."""
    for index, word in enumerate(_WORDS):
        table, _ = _land(oracle, f"w{index}", f"n,word\n0,{word}\n")

        assert _rows(f"SELECT word FROM {table}") == [[word]], word


@pytest.mark.endpoint
def test_a_column_of_mixed_words_is_declared_in_characters(oracle):
    csv_text = "n,word\n" + "".join(f"{i},{w}\n" for i, w in enumerate(_WORDS))
    table, _ = _land(oracle, "words", csv_text)

    assert _declared(table, "word") == ("VARCHAR2", 5, "C")
    stored = _rows(f"SELECT word FROM {table} ORDER BY n")
    assert [row[0] for row in stored] == list(_WORDS)


@pytest.mark.endpoint
def test_accented_text_at_the_widest_varchar2_reads_back_whole(oracle):
    """4,000 characters is the widest ``VARCHAR2``; what the bytes allow is measured.

    1,000 ``é`` is 1,000 characters and 2,000 bytes: inside the 4,000-byte
    ceiling either way, and wider than the column in bytes it used to be given.
    """
    accented = "é" * 1000
    table, _ = _land(oracle, "accented", f"n,body\n0,{accented}\n1,short\n")

    assert _declared(table, "body") == ("VARCHAR2", 1000, "C")
    assert _rows(f"SELECT body FROM {table} WHERE n = 0") == [[accented]]


@pytest.mark.endpoint
def test_text_past_4000_bytes_but_under_4000_characters_lands_whole(oracle):
    """1,400 CJK ideographs are 1,400 characters and 4,200 bytes.

    The 4,000 ceiling of a database without extended string sizes counts bytes
    whatever the column's length semantics, so the value has to land somewhere
    that holds it, and come back whole.
    """
    wide = "日" * 1400
    table, _ = _land(oracle, "cjk", f"n,body\n0,{wide}\n1,short\n")

    assert _rows(f"SELECT body FROM {table} WHERE n = 0") == [[wide]]
