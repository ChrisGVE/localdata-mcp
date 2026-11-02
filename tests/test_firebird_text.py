"""How a loaded text column lands on Firebird, and what it answers afterwards (#56).

Core's ``Text`` renders ``BLOB`` on Firebird, and a BLOB is grouped, made
distinct and sorted by its storage handle rather than by its text: a
``GROUP BY`` answers one group per row, with no error. A text column therefore
lands as a ``VARCHAR`` sized from the data. Two things decide whether that
holds:

* **What the width counts.** A ``VARCHAR`` declared without a character set
  takes the database's default, and on a database created with charset
  ``NONE`` — the test container is one — its width counts *bytes*. The width
  this server measures counts *characters*, so ``été`` (three characters, five
  bytes in UTF-8) was refused by the ``VARCHAR(3)`` sized for it. The column
  now states ``CHARACTER SET UTF8``, where the width counts characters on any
  database.
* **What happens past the widest VARCHAR.** There the column stays a BLOB, which
  keeps every value whole and answers set operations wrongly — so the load has
  to say so, per column, rather than leave the table looking ordinary.

The live tests here run only against the Firebird container; the unit tests at
the top need no container at all.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from endpoints import TARGETS, Target, Unavailable, applied, reach
from fastmcp import Client
from sqlalchemy import DDL, create_engine, inspect
from sqlalchemy.dialects import registry as dialect_registry

from localdata_mcp import config as config_module
from localdata_mcp import loader as loader_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.dialects import _FIREBIRD_VARCHAR_MAX, backend_for

# ---------------------------------------------------------------------------
# What the declaration says — no container needed
# ---------------------------------------------------------------------------


def _rendered(type_) -> str:
    """The column type as Firebird's own dialect would write it in DDL."""
    dialect = dialect_registry.load("firebirdsql")()
    return type_.compile(dialect=dialect)


def test_a_loaded_text_column_is_declared_utf8_so_its_width_counts_characters():
    """Without a character set the width counts bytes on a charset-NONE database.

    ``longest`` is measured in characters, so the declaration has to be one
    whose width means characters wherever it lands — ``CHARACTER SET UTF8``.
    """
    backend = backend_for("firebirdsql")

    assert _rendered(backend.column_type("TEXT", longest=11)) == (
        "VARCHAR(11) CHARACTER SET UTF8"
    )
    assert _rendered(backend.column_type("TEXT", longest=_FIREBIRD_VARCHAR_MAX)) == (
        f"VARCHAR({_FIREBIRD_VARCHAR_MAX}) CHARACTER SET UTF8"
    )


def test_text_past_the_widest_varchar_is_still_a_blob():
    backend = backend_for("firebirdsql")

    assert _rendered(
        backend.column_type("TEXT", longest=_FIREBIRD_VARCHAR_MAX + 1)
    ).startswith("BLOB")


def test_only_a_column_that_stays_a_blob_carries_a_note():
    """At the ceiling the column is a VARCHAR and there is nothing to say."""
    backend = backend_for("firebirdsql")

    assert backend.column_note("notes", "TEXT", longest=_FIREBIRD_VARCHAR_MAX) is None
    assert backend.column_note("amount", "INTEGER", longest=None) is None

    note = backend.column_note("notes", "TEXT", longest=_FIREBIRD_VARCHAR_MAX + 1)
    assert note is not None
    assert "notes" in note
    assert str(_FIREBIRD_VARCHAR_MAX + 1) in note
    for consequence in ("GROUP BY", "DISTINCT", "ORDER BY", "index"):
        assert consequence in note


def test_no_other_backend_notes_a_column():
    """The note is Firebird's fact; every other backend says nothing extra."""
    for dialect in ("sqlite", "postgresql", "oracle", "mssql"):
        assert backend_for(dialect).column_note("notes", "TEXT", longest=10**6) is None


# ---------------------------------------------------------------------------
# Live, against the Firebird container
# ---------------------------------------------------------------------------


def _firebird_target() -> Target:
    (target,) = [target for target in TARGETS if target.name == "firebird"]
    return target


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
class Firebird:
    url: str
    root: Path
    prefix: str

    def table(self, name: str) -> str:
        return f"{self.prefix}_{name}"


@pytest.fixture
def firebird(tmp_path):
    """The Firebird container, a fresh session, and no tables left behind."""
    target = _firebird_target()
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
            payload = call("attach", database=reached.url, nickname="fb", writable=True)
            assert payload["ok"] is True, payload
            yield Firebird(url=reached.url, root=root, prefix=prefix)
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


def _land(firebird: Firebird, name: str, csv_text: str) -> tuple[str, dict]:
    path = firebird.root / f"{name}.csv"
    path.write_text(csv_text, encoding="utf-8")
    table = firebird.table(name)
    made = call(
        "create",
        nickname="fb",
        type="table",
        source=str(path),
        table=table,
        delimiter=",",
    )
    assert made["ok"] is True, made
    return table, made


def _rows(sql: str) -> list[list]:
    answer = call("query", nickname="fb", sql=sql)
    assert answer["ok"] is True, answer
    return answer["rows"]


#: Seven values, each repeated nine times and interleaved, so that a column
#: grouped by storage handle would answer sixty-three groups and one grouped by
#: value answers seven. The issue records a BLOB probe that once came out right
#: by luck on three rows; at sixty-three rows and a known count per group, luck
#: cannot produce the assertion below.
_CITIES = ("Zürich", "été", "東京", "Ålesund", "naïve", "😀 smile", "plain")
_REPEATS = 9


def _city_csv() -> str:
    lines = ["n,city"]
    lines += [
        f"{index},{_CITIES[index % len(_CITIES)]}"
        for index in range(len(_CITIES) * _REPEATS)
    ]
    return "\n".join(lines) + "\n"


@pytest.mark.endpoint
def test_non_ascii_text_lands_whole_on_a_charset_none_database(firebird):
    """``été`` is three characters and five bytes; the column must hold it.

    Before the column stated its character set, the load failed outright with
    ``string right truncation — expected length 3, actual 5``: the width was
    measured in characters and enforced in bytes.
    """
    table, _ = _land(firebird, "cities", _city_csv())

    stored = _rows(f"SELECT DISTINCT city FROM {table}")
    assert sorted(row[0] for row in stored) == sorted(_CITIES)


@pytest.mark.endpoint
def test_grouping_distinct_and_ordering_compare_text_not_handles(firebird):
    """The defect itself: one group per value, counted, and in order."""
    table, _ = _land(firebird, "cities", _city_csv())

    grouped = _rows(f"SELECT city, count(*) FROM {table} GROUP BY city ORDER BY city")
    assert len(grouped) == len(_CITIES)
    assert all(count == _REPEATS for _, count in grouped)
    # Ordered as UTF-8 bytes order them, which is code-point order, which is
    # Python's own order for these strings.
    assert [city for city, _ in grouped] == sorted(_CITIES)

    distinct = _rows(f"SELECT count(DISTINCT city) FROM {table}")
    assert distinct == [[len(_CITIES)]]


@pytest.mark.endpoint
def test_a_landed_text_column_can_be_indexed(firebird):
    """A BLOB cannot be indexed at all; the VARCHAR it became can."""
    table, _ = _land(firebird, "cities", _city_csv())

    made = call("create", nickname="fb", type="index", table=table, columns=["city"])
    assert made["ok"] is True, made


@pytest.mark.endpoint
def test_a_column_at_the_ceiling_is_a_varchar_that_groups(firebird):
    """The widest VARCHAR is still comparable, and carries no warning."""
    wide_a = "a" * _FIREBIRD_VARCHAR_MAX
    wide_b = "é" * _FIREBIRD_VARCHAR_MAX
    rows = [wide_a, wide_b] * 6
    csv_text = "n,body\n" + "".join(f"{i},{v}\n" for i, v in enumerate(rows))

    table, made = _land(firebird, "ceiling", csv_text)

    assert not any("body" in warning for warning in made.get("warnings", [])), made
    grouped = _rows(
        f"SELECT char_length(body), count(*) FROM {table} GROUP BY body ORDER BY body"
    )
    assert grouped == [[_FIREBIRD_VARCHAR_MAX, 6], [_FIREBIRD_VARCHAR_MAX, 6]]


@pytest.mark.endpoint
def test_a_column_past_the_ceiling_is_kept_whole_and_said_to_group_wrongly(firebird):
    """Past the widest VARCHAR the load keeps the BLOB and names what breaks."""
    too_wide = _FIREBIRD_VARCHAR_MAX + 1
    rows = ["x" * too_wide] + ["short"] * 5
    csv_text = "n,body\n" + "".join(f"{i},{v}\n" for i, v in enumerate(rows))

    table, made = _land(firebird, "beyond", csv_text)

    warnings = made.get("warnings", [])
    named = [warning for warning in warnings if "body" in warning]
    assert len(named) == 1, made
    assert "GROUP BY" in named[0]
    assert str(too_wide) in named[0]
    # The values themselves are whole.
    lengths = _rows(f"SELECT max(char_length(body)) FROM {table}")
    assert lengths == [[too_wide]]


@pytest.mark.endpoint
def test_the_width_holds_when_the_widest_value_arrives_in_a_later_chunk(
    firebird, monkeypatch
):
    """A delimited file is measured whole before the table is made.

    The chunk size is shrunk so a small file spans many chunks, and the measuring
    pass is counted so the test cannot pass by reading the file in one piece.
    The widest value, non-ASCII so a byte-width column would refuse it, is the
    very last row.
    """
    monkeypatch.setattr(loader_module, "_READ_CHUNK", 10)
    observed: list[int] = []
    original = loader_module._ColumnScan.observe

    def counting(self, raw):
        if self.name == "label":
            observed.append(len(raw))
        return original(self, raw)

    monkeypatch.setattr(loader_module._ColumnScan, "observe", counting)

    widest = "ü" * 40
    lines = ["n,label"] + [f"{i},ab" for i in range(95)] + [f"95,{widest}"]
    table, _ = _land(firebird, "chunks", "\n".join(lines) + "\n")

    assert len(observed) == 10, observed
    longest = _rows(f"SELECT label FROM {table} WHERE n = 95")
    assert longest == [[widest]]
