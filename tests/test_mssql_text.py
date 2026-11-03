"""How a loaded text column lands on SQL Server, and what it reads back (#128).

A text column used to land as ``VARCHAR(n)``, which is non-Unicode: what it
can hold is the column collation's code page, and its width counts bytes of
that code page. The width this server measured counted characters. Two things
decide whether a value survives the round trip:

* **What the column can store.** Under a code-page collation a character the
  code page lacks is stored as ``?``, and the load still succeeds.
* **What the width counts.** SQL Server's Unicode type, ``NVARCHAR(n)``, counts
  UTF-16 code units, so a character outside the Basic Multilingual Plane — an
  emoji — needs two. A width measured in characters is one short for every
  such character.

The live tests here run only against the SQL Server targets, every way the
suite reaches them; the unit tests at the top need no container at all.
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
from sqlalchemy.dialects import mssql as mssql_dialect

from localdata_mcp import config as config_module
from localdata_mcp import loader as loader_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import _NVARCHAR_MAX, backend_for

# ---------------------------------------------------------------------------
# What the declaration says — no container needed
# ---------------------------------------------------------------------------


def _rendered(type_) -> str:
    """The column type as SQL Server's own dialect would write it in DDL."""
    return type_.compile(dialect=mssql_dialect.dialect())


def test_a_loaded_text_column_is_declared_nvarchar_so_it_holds_any_character():
    """``VARCHAR`` holds the collation's code page; ``NVARCHAR`` holds Unicode."""
    backend = backend_for("mssql")

    assert _rendered(backend.column_type("TEXT", longest=11)) == "NVARCHAR(11)"
    assert _rendered(backend.column_type("TEXT", longest=_NVARCHAR_MAX)) == (
        f"NVARCHAR({_NVARCHAR_MAX})"
    )


def test_text_past_the_widest_nvarchar_is_nvarchar_max():
    """Past 4,000 code units the only spelling is ``NVARCHAR(max)``, never ``NTEXT``."""
    backend = backend_for("mssql")

    assert _rendered(backend.column_type("TEXT", longest=_NVARCHAR_MAX + 1)) == (
        "NVARCHAR(max)"
    )


def test_the_widest_nvarchar_is_the_measured_4000():
    assert _NVARCHAR_MAX == 4000


def test_sql_server_sizes_text_in_utf16_code_units_and_no_other_backend_does():
    """``NVARCHAR(n)`` counts UTF-16 code units; every other width counts characters.

    An emoji is one character and two code units, so a width counted in
    characters is one short for each one a value holds.
    """
    assert backend_for("mssql").text_width(characters=3, utf16_units=5) == 5
    for dialect in ("sqlite", "postgresql", "oracle", "firebirdsql", "mysql"):
        assert backend_for(dialect).text_width(characters=3, utf16_units=5) == 3


#: Each value with its width in characters and in UTF-16 code units.
_WIDTHS = {
    "plain": (5, 5),
    "été": (3, 3),
    "日本": (2, 2),
    "😀": (1, 2),
    "a😀b😀": (4, 6),
}


@pytest.mark.parametrize("value", sorted(_WIDTHS))
def test_a_whole_frame_measures_both_widths(value):
    measured = loader_module.measure_frame(pd.DataFrame({"word": ["x", value]}))

    (column,) = measured
    assert (column.longest, column.longest_utf16) == (
        max(1, _WIDTHS[value][0]),
        max(1, _WIDTHS[value][1]),
    )


def test_a_chunked_scan_measures_both_widths_across_chunks():
    """The widest value in code units need not be the widest in characters."""
    scan = loader_module._ColumnScan("word")
    scan.observe(pd.Series(["abc"], dtype=object))
    scan.observe(pd.Series(["😀😀"], dtype=object))

    column = scan.finish()
    assert (column.longest, column.longest_utf16) == (3, 4)


def test_a_column_holding_no_text_has_no_width_in_either_unit():
    (column,) = loader_module.measure_frame(pd.DataFrame({"n": [1, 2]}))

    assert (column.longest, column.longest_utf16) == (None, None)


# ---------------------------------------------------------------------------
# Live, against the SQL Server container
# ---------------------------------------------------------------------------


def _mssql_targets() -> list[Target]:
    targets = [target for target in TARGETS if target.endpoint.dialect == "mssql"]
    assert targets, "the endpoint matrix lost its SQL Server targets"
    return targets


def call(tool: str, **arguments):
    """Call a tool over the protocol and return its decoded payload.

    The same in-process client the endpoint suite uses, duplicated rather than
    imported for the reason it gives: importing a test module from another
    collects its fixtures twice.
    """

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


@dataclass(frozen=True)
class MSSQL:
    url: str
    root: Path
    prefix: str

    def table(self, name: str) -> str:
        return f"{self.prefix}_{name}"


@pytest.fixture(params=_mssql_targets(), ids=lambda target: target.name)
def mssql(request, tmp_path):
    """A SQL Server target, a fresh session, and no tables left behind."""
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
            payload = call("attach", database=reached.url, nickname="ms", writable=True)
            assert payload["ok"] is True, payload
            yield MSSQL(url=reached.url, root=root, prefix=prefix)
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


def _land(mssql: MSSQL, name: str, csv_text: str) -> tuple[str, dict]:
    """Put a CSV into the endpoint through ``create``, the server's own verb."""
    path = mssql.root / f"{name}.csv"
    path.write_text(csv_text, encoding="utf-8")
    table = mssql.table(name)
    made = call(
        "create",
        nickname="ms",
        type="table",
        source=str(path),
        table=table,
        delimiter=",",
    )
    assert made["ok"] is True, made
    return table, made


def _rows(sql: str) -> list[list]:
    answer = call("query", nickname="ms", sql=sql)
    assert answer["ok"] is True, answer
    return answer["rows"]


def _declared(mssql: MSSQL, table: str, column: str) -> tuple[str, int]:
    """The column's type and width as SQL Server's own catalogue states them.

    ``CHARACTER_MAXIMUM_LENGTH`` is ``-1`` for a ``(max)`` type.
    """
    ((data_type, width),) = _rows(
        "SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS "
        f"WHERE TABLE_NAME = '{table}' AND COLUMN_NAME = '{column}'"
    )
    return data_type, width


#: One value per kind of character a code page may lack: plain ASCII, a Latin-1
#: accent, two CJK ideographs, and an emoji, which is outside the Basic
#: Multilingual Plane and so two UTF-16 code units.
_WORDS = ("plain", "été", "日本", "😀")


@pytest.mark.endpoint
def test_non_ascii_text_reads_back_exactly_what_the_file_held(mssql):
    """The defect itself: every value comes back byte for byte, not as ``?``."""
    csv_text = "n,word\n" + "".join(f"{i},{w}\n" for i, w in enumerate(_WORDS))
    table, _ = _land(mssql, "words", csv_text)

    stored = _rows(f"SELECT n, word FROM {table} ORDER BY n")

    assert [row[1] for row in stored] == list(_WORDS)
    assert [row[1].encode("utf-8") for row in stored] == [
        word.encode("utf-8") for word in _WORDS
    ]


@pytest.mark.endpoint
def test_an_emoji_is_given_the_two_code_units_it_takes(mssql):
    """One emoji is one character and needs ``NVARCHAR(2)``, not ``NVARCHAR(1)``."""
    table, _ = _land(mssql, "emoji", "n,face\n0,😀\n")

    assert _declared(mssql, table, "face") == ("nvarchar", 2)
    assert _rows(f"SELECT face FROM {table}") == [["😀"]]


@pytest.mark.endpoint
def test_an_emoji_column_at_the_widest_nvarchar_is_sized_in_code_units(mssql):
    """2,000 emoji are 4,000 code units: the widest ``NVARCHAR``, and still one."""
    at_ceiling = "😀" * (_NVARCHAR_MAX // 2)
    table, made = _land(mssql, "ceiling", f"n,body\n0,{at_ceiling}\n1,short\n")

    assert _declared(mssql, table, "body") == ("nvarchar", _NVARCHAR_MAX)
    assert _rows(f"SELECT body FROM {table} WHERE n = 0") == [[at_ceiling]]
    assert not any("body" in warning for warning in made.get("warnings", [])), made


@pytest.mark.endpoint
def test_one_code_unit_past_the_widest_nvarchar_is_nvarchar_max_and_whole(mssql):
    """2,000 emoji and one letter are 4,001 code units, so ``NVARCHAR(max)``.

    Measured in characters the value is 2,001 wide and would have been given
    ``NVARCHAR(2001)``, which refuses it. It groups and compares as text: a
    large object on SQL Server answers set operations correctly.
    """
    beyond = "😀" * (_NVARCHAR_MAX // 2) + "x"
    rows = [beyond, "short"] * 3
    csv_text = "n,body\n" + "".join(f"{i},{v}\n" for i, v in enumerate(rows))
    table, _ = _land(mssql, "beyond", csv_text)

    assert _declared(mssql, table, "body") == ("nvarchar", -1)
    stored = _rows(f"SELECT body FROM {table} WHERE n = 0")
    assert stored == [[beyond]]
    grouped = _rows(f"SELECT count(*) FROM {table} GROUP BY body ORDER BY count(*)")
    assert grouped == [[3], [3]]


@pytest.mark.endpoint
def test_a_landed_text_column_can_be_indexed(mssql):
    table, _ = _land(
        mssql, "words", "n,word\n" + "".join(f"{i},{w}\n" for i, w in enumerate(_WORDS))
    )

    made = call("create", nickname="ms", type="index", table=table, columns=["word"])
    assert made["ok"] is True, made
    hit = _rows(f"SELECT n FROM {table} WHERE word = N'日本'")
    assert hit == [[2]]


@pytest.mark.endpoint
def test_an_index_over_a_value_past_the_key_limit_is_refused_out_loud(mssql):
    """A nonclustered key holds 1,700 bytes, which is 850 ``NVARCHAR`` code units.

    SQL Server refuses the index when a value it holds is wider, and the
    refusal reaches the caller with the limit in it — nothing is created.
    """
    wide = "x" * 851
    table, _ = _land(mssql, "keyed", f"n,body\n0,{wide}\n1,short\n")

    made = call("create", nickname="ms", type="index", table=table, columns=["body"])

    assert made["ok"] is False, made
    assert "1700" in made["error"], made


@pytest.mark.endpoint
def test_an_index_over_nvarchar_max_is_refused_out_loud(mssql):
    beyond = "y" * (_NVARCHAR_MAX + 1)
    table, _ = _land(mssql, "huge", f"n,body\n0,{beyond}\n")

    made = call("create", nickname="ms", type="index", table=table, columns=["body"])

    assert made["ok"] is False, made
    assert "body" in made["error"], made
