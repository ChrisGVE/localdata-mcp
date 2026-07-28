"""Loading, querying and exporting, exercised against the harvested corpus.

The fixtures in ``tests/assets`` are deliberately hostile: mixed-type columns,
embedded newlines and commas inside quoted fields, malformed quoting, latin-1
bytes, emoji, blank rows, and a truncated final line. They are the reason this
suite is short — each file covers several failure modes at once.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import foreign
import pytest
from sqlalchemy import Integer, Text, text
from sqlalchemy.exc import SQLAlchemyError

from localdata_mcp import config as config_module
from localdata_mcp import export as export_module
from localdata_mcp.config import Config
from localdata_mcp import loader as loader_module
from localdata_mcp.loader import LoadError, Workspace
from localdata_mcp.paths import PathNotAllowed

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(autouse=True)
def root(monkeypatch, tmp_path):
    """Allow both the asset corpus and a scratch directory."""
    shared = tmp_path / "root"
    shared.mkdir()
    for asset in ASSETS.iterdir():
        (shared / asset.name).write_bytes(asset.read_bytes())
    config_module.use(Config(roots=(shared,)))
    return shared


@pytest.fixture()
def workspace():
    """A workspace with one tag open, since a tag is now what holds tables.

    ``main`` here is an ordinary tag name and carries no special meaning — the
    shared ``main`` schema every datasource used to attach beside is exactly what
    the per-tag model removed.
    """
    ws = Workspace.in_memory()
    ws.attach_memory("main")
    yield ws
    ws.close()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_simple_csv_loads_and_answers_correctly(workspace, root):
    (info,) = workspace.load_file(str(root / "simple.csv"), "main")

    assert info.row_count == 5
    assert [c.name for c in info.columns] == [
        "name",
        "age",
        "department",
        "salary",
        "start_date",
        "active",
    ]

    _, rows = workspace.query("main", "SELECT sum(salary) FROM simple")
    assert rows[0][0] == 75000 + 65000 + 70000 + 80000 + 55000

    _, rows = workspace.query(
        "main", "SELECT count(*) FROM simple WHERE department = 'Engineering'"
    )
    assert rows[0][0] == 2


def test_numeric_column_is_declared_numeric(workspace, root):
    (info,) = workspace.load_file(str(root / "simple.csv"), "main")
    by_name = {c.name: c for c in info.columns}
    assert by_name["age"].declared_type == "INTEGER"
    assert by_name["salary"].declared_type == "INTEGER"
    assert by_name["department"].declared_type == "TEXT"


def test_messy_csv_loads_without_losing_rows(workspace, root):
    """Quoted newlines, embedded commas, emoji and a blank row all survive."""
    (info,) = workspace.load_file(str(root / "messy_mixed_types.csv"), "main")
    assert info.row_count > 0

    _, rows = workspace.query(
        "main", "SELECT count(*) FROM messy_mixed_types WHERE name LIKE '%Emoji%'"
    )
    assert rows[0][0] == 1

    # The value column mixes 123.45, '456abc', 'N/A' and '123,456.78'. It must
    # not have been silently coerced to a numeric type.
    by_name = {c.name: c for c in info.columns}
    assert by_name["value"].declared_type == "TEXT"


def test_mostly_numeric_text_column_is_flagged_as_mixed(workspace, root):
    """The case a storage-class histogram alone cannot see.

    pandas reads a column of 1, 2, 3a as ``object``; we declare it TEXT; every
    value stores as text. The histogram therefore reports a single storage class
    and says nothing — while ``avg()`` over the column silently returns a wrong
    answer, because SQLite coerces the text to 0 and keeps it in the divisor.
    """
    target = root / "partly_numeric.csv"
    # 'n/a' is one of pandas' default null markers and lands as SQL NULL;
    # 'unknown' is not, so it survives as text. Both behaviours are asserted
    # here because the difference decides what the counts below mean.
    target.write_text("v\n1\n2\n3\n4\n5\nn/a\nunknown\n")
    (info,) = workspace.load_file(str(target), "main")

    column = info.columns[0]
    assert column.declared_type == "TEXT"
    assert column.storage_classes == {"null": 1, "text": 6}
    assert len([c for c in column.storage_classes if c != "null"]) == 1
    assert column.numeric_values == 5
    assert column.non_numeric_values == 1
    assert column.is_mixed is True
    assert info.mixed_columns == ["v"]

    # Why the flag matters: the naive average is wrong, and the guarded one is not.
    # SQLite skips the NULL, coerces 'unknown' to 0, and keeps it in the divisor.
    _, naive = workspace.query("main", "SELECT avg(v) FROM partly_numeric")
    assert naive[0][0] == pytest.approx(15 / 6)
    _, guarded = workspace.query(
        "main",
        "SELECT avg(CAST(v AS REAL)) FROM partly_numeric WHERE typeof(v) = 'text' "
        "AND CAST(v AS REAL) != 0",
    )
    assert guarded[0][0] == pytest.approx(3.0)


def test_uniformly_numeric_text_column_is_not_flagged(workspace, root):
    """No false positive: a column that is entirely non-numeric is not mixed."""
    target = root / "all_text.csv"
    target.write_text("v\nalpha\nbeta\ngamma\n")
    (info,) = workspace.load_file(str(target), "main")
    assert info.columns[0].is_mixed is False
    assert info.mixed_columns == []


def test_unicode_survives_the_round_trip(workspace, root):
    workspace.load_file(str(root / "messy_mixed_types.csv"), "main")
    _, rows = workspace.query(
        "main", "SELECT name FROM messy_mixed_types WHERE name LIKE '%Garc%'"
    )
    assert rows and "í" in rows[0][0]


def test_truncated_file_loads(workspace, root):
    """A missing trailing value is a NULL, not a failure."""
    (info,) = workspace.load_file(str(root / "truncated.csv"), "main")
    assert info.row_count == 2
    _, rows = workspace.query(
        "main", "SELECT count(*) FROM truncated WHERE value IS NULL"
    )
    assert rows[0][0] == 1


def test_empty_file_is_refused_clearly(workspace, root):
    with pytest.raises(LoadError):
        workspace.load_file(str(root / "empty.csv"), "main")


def test_duplicate_and_blank_headers_become_usable_columns(workspace, root, tmp_path):
    target = root / "dupes.csv"
    target.write_text("a,a,,1x\n1,2,3,4\n")
    (info,) = workspace.load_file(str(target), "main")
    names = [c.name for c in info.columns]
    assert len(set(names)) == len(names), names
    assert all(names), names


def test_unsupported_extension_names_what_is_supported(workspace, root):
    target = root / "thing.wibble"
    target.write_bytes(b"not a format this server knows")
    with pytest.raises(LoadError, match="csv"):
        workspace.load_file(str(target), "main")


def test_table_name_can_be_overridden(workspace, root):
    (info,) = workspace.load_file(str(root / "simple.csv"), "main", table_name="staff")
    assert info.name == "staff"
    _, rows = workspace.query("main", "SELECT count(*) FROM staff")
    assert rows[0][0] == 5


# ---------------------------------------------------------------------------
# Path boundary
# ---------------------------------------------------------------------------


def test_path_outside_root_is_refused(workspace, tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n1\n")
    with pytest.raises(PathNotAllowed):
        workspace.load_file(str(outside), "main")


def test_traversal_out_of_root_is_refused(workspace, root, tmp_path):
    outside = tmp_path / "secret.csv"
    outside.write_text("a\n1\n")
    with pytest.raises(PathNotAllowed):
        workspace.load_file(str(root / ".." / "secret.csv"), "main")


def test_symlink_pointing_outside_root_is_refused(workspace, root, tmp_path):
    """Resolve-then-check: the link resolves to its real, disallowed location."""
    outside = tmp_path / "secret.csv"
    outside.write_text("a\n1\n")
    link = root / "innocent.csv"
    link.symlink_to(outside)
    with pytest.raises(PathNotAllowed):
        workspace.load_file(str(link), "main")


def test_missing_file_is_refused(workspace, root):
    with pytest.raises(PathNotAllowed, match="No such file"):
        workspace.load_file(str(root / "nope.csv"), "main")


# ---------------------------------------------------------------------------
# Attaching a database, and the flagship join across sources
# ---------------------------------------------------------------------------


def _build_database(path: Path) -> None:
    foreign.build_database(
        path,
        "departments",
        [("department", Text), ("floor", Integer)],
        [("Engineering", 3), ("Sales", 1), ("Marketing", 2)],
    )


# A read-only workspace opened directly from a database file no longer exists:
# under the slot model a database file is attached beside the others, so its
# read-only guarantee and its adopted tables are covered in test_slots.py by
# test_an_attached_database_cannot_be_written_through and
# test_a_database_file_arrives_with_its_existing_tables.


# ---------------------------------------------------------------------------
# Residency, and moving a database out of memory
# ---------------------------------------------------------------------------


def test_residency_is_measured_and_grows_with_the_data(workspace, root):
    """The number the spill decision reads must track what is actually held."""
    workspace.attach_memory("scratch")
    empty = workspace.resident_bytes("scratch")

    workspace.load_file(str(root / "simple.csv"), "scratch")
    loaded = workspace.resident_bytes("scratch")

    assert empty == 0
    assert loaded > empty


def test_residency_falls_back_after_a_drop_despite_the_freelist(workspace, root):
    """page_count alone ratchets: it never shrinks once pages have been used.

    Without the freelist correction a slot that had been loaded and emptied
    would keep measuring at its high-water mark, and would be spilled to disk
    for data it no longer holds.
    """
    workspace.attach_memory("scratch")
    workspace.load_file(str(root / "large_dataset.csv"), "scratch")
    full = workspace.resident_bytes("scratch")

    workspace.drop_table("scratch", "large_dataset")
    emptied = workspace.resident_bytes("scratch")

    assert full > 0
    assert emptied < full
    # The uncorrected reading is the one that would have ratcheted: it stays at
    # the high-water mark, which is exactly what the correction is subtracting.
    with workspace.engine("scratch").connect() as conn:
        page_size = conn.execute(text("PRAGMA page_size")).scalar_one()
        uncorrected = conn.execute(text("PRAGMA page_count")).scalar_one() * page_size
    assert uncorrected >= full
    assert uncorrected > emptied


def test_vacuum_into_produces_a_database_that_still_answers(workspace, root, tmp_path):
    """Assert on what the copy returns, never on the fact that a file appeared."""
    workspace.attach_memory("scratch")
    workspace.load_file(str(root / "simple.csv"), "scratch")
    original = workspace.query(
        "scratch", "SELECT name, salary FROM simple ORDER BY name"
    )
    target = tmp_path / "copy.sqlite"

    workspace.snapshot("scratch", target)

    copied = sqlite3.connect(target)
    try:
        assert copied.execute("SELECT count(*) FROM simple").fetchone()[0] == 5
        assert [
            tuple(r)
            for r in copied.execute("SELECT name, salary FROM simple ORDER BY name")
        ] == [tuple(r) for r in original[1]]
        assert (
            copied.execute("SELECT sum(salary) FROM simple").fetchone()[0]
            == (workspace.query("scratch", "SELECT sum(salary) FROM simple")[1][0][0])
        )
    finally:
        copied.close()


def test_vacuum_into_refuses_to_replace_an_existing_database(workspace, root, tmp_path):
    """The refusal holds only when the target *is* a database.

    Onto a zero-length file SQLite writes happily, and onto a non-database it
    fails with an unrelated complaint about the file's contents. So this is not
    a guard anything may lean on: the overwrite refusal lives at the path
    boundary, where it is unconditional.
    """
    workspace.attach_memory("scratch")
    workspace.load_file(str(root / "simple.csv"), "scratch")
    target = tmp_path / "copy.sqlite"
    workspace.snapshot("scratch", target)

    with pytest.raises(SQLAlchemyError, match="already exists"):
        workspace.snapshot("scratch", target)


def test_vacuum_into_does_not_refuse_a_zero_length_target(workspace, root, tmp_path):
    """Recorded because it is the gap that makes the path-boundary guard load-bearing."""
    workspace.attach_memory("scratch")
    workspace.load_file(str(root / "simple.csv"), "scratch")
    target = tmp_path / "copy.sqlite"
    target.write_bytes(b"")

    workspace.snapshot("scratch", target)

    assert target.stat().st_size > 0


def test_a_writable_attach_accepts_what_a_readonly_one_refuses(workspace, root):
    """The grant is a different URI, not a check this module is trusted to make."""
    _build_database(root / "hr.db")
    workspace.attach_file("locked", root / "hr.db")
    workspace.attach_file("open", root / "hr.db", readonly=False)

    with pytest.raises(SQLAlchemyError):
        with workspace.engine("locked").begin() as conn:
            conn.execute(text("INSERT INTO departments VALUES ('X', 9)"))

    with workspace.engine("open").begin() as conn:
        conn.execute(text("INSERT INTO departments VALUES ('X', 9)"))
    _, rows = workspace.query(
        "open", "SELECT floor FROM departments WHERE department='X'"
    )
    assert [tuple(r) for r in rows] == [(9,)]


def test_a_question_mark_in_a_filename_is_not_read_as_a_uri_query(workspace, root):
    """The connection runs in URI mode, so the path has to be percent-encoded."""
    awkward = root / "why? not.db"
    _build_database(awkward)

    workspace.attach_file("odd", awkward)

    _, rows = workspace.query("odd", "SELECT count(*) FROM departments")
    assert [tuple(r) for r in rows] == [(3,)]


# ---------------------------------------------------------------------------
# Composing a database from a declared schema
# ---------------------------------------------------------------------------


def test_dropping_a_table_removes_it_and_dropping_it_twice_is_an_error(workspace, root):
    workspace.attach_memory("scratch")
    workspace.load_file(str(root / "simple.csv"), "scratch")
    assert workspace.has_table("scratch", "simple")

    workspace.drop_table("scratch", "simple")

    assert not workspace.has_table("scratch", "simple")
    with pytest.raises(LoadError, match="simple"):
        workspace.drop_table("scratch", "simple")


# ---------------------------------------------------------------------------
# Reading without assembling — query_stream
# ---------------------------------------------------------------------------


def test_query_stream_returns_what_query_returns(workspace, root):
    """One read path, so the two must agree on rows and on column names."""
    workspace.load_file(str(root / "simple.csv"), "main")
    sql = "SELECT name, salary FROM simple ORDER BY name"

    columns, rows = workspace.query("main", sql)
    with workspace.query_stream("main", sql) as (streamed_columns, streamed_rows):
        streamed = list(streamed_rows)

    assert streamed_columns == columns
    assert streamed == rows


def test_query_stream_hands_back_rows_it_has_not_read_yet(workspace, root):
    """The point of the method: the first row arrives before the last is read.

    Asserted against the cursor rather than against memory — a lazy iterator is
    the mechanism, and a peak measurement is the consequence. The consequence is
    asserted separately, as a growth shape, in ``test_volume``.
    """
    workspace.load_file(str(root / "simple.csv"), "main")

    with workspace.query_stream("main", "SELECT name FROM simple") as (_, rows):
        assert not isinstance(rows, (list, tuple))
        first = next(iter(rows))
        assert isinstance(first, tuple)
        # Four of the five are still on the cursor at this point; draining them
        # here proves the iterator is a live view of it rather than a spent one.
        assert len(list(rows)) == 4


def test_query_stream_refuses_a_write_exactly_as_query_does(workspace, root):
    """The read-only posture is decided once, so it cannot differ between them."""
    workspace.load_file(str(root / "simple.csv"), "main")
    sql = "UPDATE simple SET salary = 0"

    with pytest.raises(LoadError) as materialised:
        workspace.query("main", sql)

    with pytest.raises(LoadError) as streamed:
        with workspace.query_stream("main", sql) as (_, rows):
            list(rows)

    assert str(streamed.value) == str(materialised.value)


def test_query_stream_explains_a_missing_table_the_same_way(workspace, root):
    """A statement that cannot run fails at the top of the block, explained."""
    workspace.load_file(str(root / "simple.csv"), "main")

    with pytest.raises(LoadError, match="No such table"):
        with workspace.query_stream("main", "SELECT * FROM absent") as (_, rows):
            list(rows)


# ---------------------------------------------------------------------------
# Export — the overwrite boundary
# ---------------------------------------------------------------------------


def test_export_writes_the_rows(workspace, root):
    workspace.load_file(str(root / "simple.csv"), "main")
    columns, rows = workspace.query(
        "main", "SELECT name, salary FROM simple ORDER BY name"
    )
    target = root / "out.csv"

    result = export_module.export_rows(columns, rows, str(target))

    assert result.row_count == 5
    written = target.read_text().splitlines()
    assert written[0] == "name,salary"
    assert len(written) == 6


def test_export_refuses_an_existing_file(workspace, root):
    target = root / "out.csv"
    target.write_text("do not lose me\n")
    with pytest.raises(PathNotAllowed, match="already exists"):
        export_module.export_rows(["a"], [(1,)], str(target))
    assert target.read_text() == "do not lose me\n"


def test_export_replaces_only_when_forced(workspace, root):
    target = root / "out.csv"
    target.write_text("stale\n")

    result = export_module.export_rows(["a"], [(1,), (2,)], str(target), force=True)

    assert result.row_count == 2
    assert target.read_text().splitlines() == ["a", "1", "2"]


def test_export_will_not_force_over_a_file_a_slot_is_sitting_on(workspace, root):
    """Even forced. The user consented to lose a spare file, not to cut a slot
    loose from the file it is reading."""
    source = root / "simple.csv"
    before = source.read_text()

    with pytest.raises(PathNotAllowed, match="attached to"):
        export_module.export_rows(
            ["a"],
            [(1,)],
            str(source),
            force=True,
            claimed={source.resolve(): "simple"},
        )

    assert source.read_text() == before


def test_export_is_owner_readable_only(workspace, root):
    target = root / "out.csv"
    export_module.export_rows(["a"], [(1,)], str(target))
    assert oct(os.stat(target).st_mode & 0o777) == "0o600"


def test_export_outside_root_is_refused(root, tmp_path):
    with pytest.raises(PathNotAllowed):
        export_module.export_rows(["a"], [(1,)], str(tmp_path / "escape.csv"))


def test_export_to_a_missing_directory_is_refused(root):
    with pytest.raises(PathNotAllowed, match="Directory does not exist"):
        export_module.export_rows(["a"], [(1,)], str(root / "nope" / "out.csv"))


def test_export_round_trips_back_into_the_workspace(workspace, root):
    """The strongest check that the export is really well-formed."""
    workspace.load_file(str(root / "messy_mixed_types.csv"), "main")
    columns, rows = workspace.query("main", "SELECT * FROM messy_mixed_types")
    target = root / "exported.csv"
    export_module.export_rows(columns, rows, str(target))

    (reloaded,) = workspace.load_file(str(target), "main", table_name="reloaded")
    assert reloaded.row_count == len(rows)


def test_a_tsv_export_is_tab_separated(workspace, root):
    """The suffix chooses the format. It used to choose nothing at all."""
    target = root / "out.tsv"

    export_module.export_rows(["a", "b"], [(1, 2)], str(target))

    assert target.read_text().splitlines() == ["a\tb", "1\t2"]


def test_an_unwritable_suffix_is_refused_by_name_and_leaves_no_file(workspace, root):
    """Silently writing CSV under another name is the worse answer.

    A caller who asks for Parquet and is told it worked has a file whose name
    lies about its contents, and nothing anywhere says so.
    """
    target = root / "out.wibble"

    with pytest.raises(export_module.ExportError) as raised:
        export_module.export_rows(["a"], [(1,)], str(target))

    assert ".wibble" in str(raised.value)
    assert ".csv" in str(raised.value)  # says what it can do instead
    assert not target.exists()


def test_a_target_with_no_suffix_is_refused_and_says_why(workspace, root):
    target = root / "out"

    with pytest.raises(export_module.ExportError, match="no suffix"):
        export_module.export_rows(["a"], [(1,)], str(target))

    assert not target.exists()


def test_the_suffix_is_checked_before_the_file_is_touched(workspace, root):
    """An unwritable suffix must not destroy what is already there, forced or not."""
    target = root / "keep.wibble"
    target.write_text("do not lose me\n")

    with pytest.raises(export_module.ExportError):
        export_module.export_rows(["a"], [(1,)], str(target), force=True)

    assert target.read_text() == "do not lose me\n"


def test_every_writer_produces_an_owner_readable_file(workspace, root):
    """Not just CSV: a format whose library opens the path itself would land 0o644."""
    for suffix in sorted(export_module.WRITERS):
        target = root / f"modes{suffix}"
        export_module.export_rows(["a"], [(1,)], str(target))
        assert oct(os.stat(target).st_mode & 0o777) == "0o600", suffix


def test_mixed_text_column_names_the_values_that_do_not_parse(workspace, root):
    """The counts alone cost a caller a round trip to learn what the junk *is*.

    Live-agent validation: every agent shown ``non_numeric_values: 41`` spent its
    next call running a GROUP BY to find out which value that was, because the
    answer decides the filter it has to write. The loader already sees those
    values while it counts them; carrying a bounded sample out is what turns the
    warning into something actionable.
    """
    target = root / "sentinels.csv"
    target.write_text("v\n1\n2\n3\npending\npending\nvoid\n")
    (loaded,) = workspace.load_file(str(target), "main")
    column = loaded.columns[0]

    assert column.numeric_values == 3
    assert column.non_numeric_values == 3
    # Distinct and bounded, not one entry per offending row.
    assert column.non_numeric_examples == ("pending", "void")


def test_non_numeric_examples_are_capped(workspace, root):
    """A column of unique junk must not return a copy of itself."""
    values = "\n".join(f"junk_{n}" for n in range(50))
    target = root / "many_sentinels.csv"
    target.write_text(f"v\n1\n2\n{values}\n")
    (loaded,) = workspace.load_file(str(target), "main")
    column = loaded.columns[0]

    assert column.non_numeric_values == 50
    assert len(column.non_numeric_examples) == loader_module.MAX_NON_NUMERIC_EXAMPLES


def test_mixed_kind_says_which_signal_fired(workspace, root):
    """Two signals, two different remedies — so the caller must be told which.

    ``typeof(col)='integer'`` separates the values only when the storage classes
    genuinely differ. On a TEXT-affinity column every value is stored as text,
    that filter matches nothing useful, and prescribing it sends the caller down
    a road with no end.
    """
    target = root / "affinity.csv"
    target.write_text("v\n1\n2\npending\n")
    (loaded,) = workspace.load_file(str(target), "main")
    assert loaded.columns[0].mixed_kind == "text"

    external = root / "heterogeneous.sqlite"
    with sqlite3.connect(external) as conn:
        conn.execute("CREATE TABLE t (v)")  # no declared type: dynamic storage
        conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), ("pending",)])

    workspace.attach_file("ext", external, readonly=True)
    described = workspace.describe("ext", "t").columns[0]
    assert described.storage_classes == {"integer": 2, "text": 1}
    assert described.mixed_kind == "storage"


# ---------------------------------------------------------------------------
# JSON and JSON Lines
# ---------------------------------------------------------------------------


def test_a_json_array_of_objects_is_a_table(workspace, root):
    (info,) = workspace.load_file(str(root / "records.json"), "main")

    assert info.row_count == 3
    assert [c.name for c in info.columns] == ["name", "role", "salary", "started"]
    assert info.notes == ()  # nothing was assumed, so nothing is said


def test_a_json_date_is_recognised_like_any_other(workspace, root):
    """The temporal pass runs on the frame, so it does not care which reader made it."""
    (info,) = workspace.load_file(str(root / "records.json"), "main")
    started = {c.name: c for c in info.columns}["started"]

    assert started.temporal_standard == "iso8601_utc"
    assert started.unparsed_temporal_examples == ()


def test_the_only_array_in_a_wrapped_object_is_the_table_and_it_says_so(
    workspace, root
):
    """The common API-dump shape. One candidate means nothing was chosen between.

    Refusing this would be a dead end: the agent has no way to lift the array out
    of the file, so a refusal it cannot act on is worse than a load it is told
    about.
    """
    (info,) = workspace.load_file(str(root / "wrapped.json"), "main")

    assert info.row_count == 2
    assert [c.name for c in info.columns] == ["name", "salary"]
    assert len(info.notes) == 1
    assert "employees" in info.notes[0]


def test_two_candidate_arrays_are_refused_and_both_are_named(workspace, root):
    """Now there IS a choice, so the server does not make it."""
    with pytest.raises(LoadError) as raised:
        workspace.load_file(str(root / "two_tables.json"), "main")

    assert "employees" in str(raised.value)
    assert "departments" in str(raised.value)


def test_a_nested_value_becomes_json_text_and_the_columns_are_named(workspace, root):
    """SQL has no nested type. Encoding is lossless; silence about it is not."""
    (info,) = workspace.load_file(str(root / "nested.json"), "main")

    assert info.row_count == 2
    columns, rows = workspace.query(
        "main", "SELECT address, tags FROM nested ORDER BY name"
    )
    assert rows[0][0] == '{"city": "London", "postcode": "NW1"}'
    assert rows[0][1] == '["math", "engines"]'

    note = " ".join(info.notes)
    assert "address" in note and "tags" in note
    assert "json_extract" in note  # the remedy, not just the diagnosis


def test_json_lines_is_one_object_per_line(workspace, root):
    (info,) = workspace.load_file(str(root / "records.jsonl"), "main")

    assert info.row_count == 3
    assert [c.name for c in info.columns] == ["name", "salary"]
    assert info.notes == ()


def test_a_json_scalar_array_is_refused_because_it_names_no_column(workspace, root):
    target = root / "scalars.json"
    target.write_text("[1, 2, 3]")

    with pytest.raises(LoadError, match="objects"):
        workspace.load_file(str(target), "main")


def test_a_json_object_with_no_array_at_all_is_refused(workspace, root):
    target = root / "single.json"
    target.write_text('{"name": "Ada", "salary": 120000}')

    with pytest.raises(LoadError) as raised:
        workspace.load_file(str(target), "main")
    assert "array of objects" in str(raised.value)


def test_malformed_json_names_the_file(workspace, root):
    target = root / "broken.json"
    target.write_text('[{"name": "Ada",}]')

    with pytest.raises(LoadError, match="broken.json"):
        workspace.load_file(str(target), "main")


def test_json_round_trips_through_the_export(workspace, root):
    """The strongest check that the writer and the reader agree."""
    workspace.load_file(str(root / "records.json"), "main")
    columns, rows = workspace.query("main", "SELECT * FROM records ORDER BY name")
    target = root / "again.json"

    export_module.export_rows(columns, rows, str(target))
    (reloaded,) = workspace.load_file(str(target), "main", table_name="reloaded")

    assert reloaded.row_count == len(rows)
    _, back = workspace.query("main", "SELECT * FROM reloaded ORDER BY name")
    assert back == rows


def test_jsonl_round_trips_through_the_export(workspace, root):
    workspace.load_file(str(root / "records.jsonl"), "main")
    columns, rows = workspace.query("main", "SELECT * FROM records ORDER BY name")
    target = root / "again.jsonl"

    export_module.export_rows(columns, rows, str(target))
    (reloaded,) = workspace.load_file(str(target), "main", table_name="reloaded")

    assert reloaded.row_count == 3
    _, back = workspace.query("main", "SELECT * FROM reloaded ORDER BY name")
    assert back == rows


def test_a_jsonl_export_is_one_object_per_line_with_no_wrapper(workspace, root):
    target = root / "lines.jsonl"

    export_module.export_rows(["a", "b"], [(1, "x"), (2, "y")], str(target))

    written = target.read_text().splitlines()
    assert written == ['{"a": 1, "b": "x"}', '{"a": 2, "b": "y"}']


def test_a_null_survives_the_json_round_trip_as_a_null(workspace, root):
    """It is the value most likely to come back as the string 'None'."""
    target = root / "nulls.json"
    export_module.export_rows(["a", "b"], [(1, None)], str(target))

    assert '"b": null' in target.read_text()
    workspace.load_file(str(target), "main", table_name="nulls")
    _, rows = workspace.query("main", "SELECT b FROM nulls")
    assert rows[0][0] is None


# ---------------------------------------------------------------------------
# XML
# ---------------------------------------------------------------------------


def test_repeated_elements_under_the_root_are_the_rows(workspace, root):
    (info,) = workspace.load_file(str(root / "employees.xml"), "main")

    assert info.row_count == 3
    assert [c.name for c in info.columns] == ["name", "role", "salary", "started"]
    assert info.notes == ()

    _, rows = workspace.query("main", "SELECT sum(salary) FROM employees")
    assert rows[0][0] == 353000


def test_an_xml_number_is_a_number_and_a_date_is_recognised(workspace, root):
    """Element text is all strings until something types it."""
    (info,) = workspace.load_file(str(root / "employees.xml"), "main")
    by_name = {c.name: c for c in info.columns}

    assert by_name["salary"].declared_type == "INTEGER"
    assert by_name["started"].temporal_standard == "iso8601_utc"


def test_attributes_are_columns_too(workspace, root):
    (info,) = workspace.load_file(str(root / "attributes.xml"), "main")

    assert info.row_count == 2
    assert [c.name for c in info.columns] == ["id", "name"]


def test_the_one_repeated_element_is_the_table_and_the_singletons_are_named(
    workspace, root
):
    """A metadata element beside the rows is the XML spelling of a wrapped JSON object."""
    (info,) = workspace.load_file(str(root / "wrapped.xml"), "main")

    assert info.row_count == 2
    assert [c.name for c in info.columns] == ["name", "salary"]
    assert len(info.notes) == 1
    assert "row" in info.notes[0] and "generated" in info.notes[0]


def test_two_repeated_elements_are_two_tables_and_are_refused(workspace, root):
    with pytest.raises(LoadError) as raised:
        workspace.load_file(str(root / "two_kinds.xml"), "main")

    assert "employee" in str(raised.value) and "department" in str(raised.value)


def test_a_nested_element_is_kept_as_xml_text_rather_than_dropped(workspace, root):
    """pandas.read_xml drops the subtree and leaves NaN. That is data loss with no signal."""
    (info,) = workspace.load_file(str(root / "nested.xml"), "main")

    _, rows = workspace.query("main", "SELECT address FROM nested ORDER BY name")
    assert "<city>London</city>" in rows[0][0]

    note = " ".join(info.notes)
    assert "address" in note


def test_a_repeated_child_is_a_list_and_is_refused_by_name(workspace, root):
    """pandas.read_xml keeps the last one and says nothing, losing the rest."""
    with pytest.raises(LoadError) as raised:
        workspace.load_file(str(root / "repeated.xml"), "main")

    assert "tag" in str(raised.value)


def test_an_xml_row_holding_only_text_names_no_column(workspace, root):
    target = root / "scalars.xml"
    target.write_text("<data><item>1</item><item>2</item></data>")

    with pytest.raises(LoadError):
        workspace.load_file(str(target), "main")


def test_malformed_xml_names_the_file(workspace, root):
    target = root / "broken.xml"
    target.write_text("<data><row><a>1</a></data>")

    with pytest.raises(LoadError, match="broken.xml"):
        workspace.load_file(str(target), "main")


def test_xml_round_trips_through_the_export(workspace, root):
    workspace.load_file(str(root / "employees.xml"), "main")
    columns, rows = workspace.query("main", "SELECT * FROM employees ORDER BY name")
    target = root / "again.xml"

    export_module.export_rows(columns, rows, str(target))
    (reloaded,) = workspace.load_file(str(target), "main", table_name="reloaded")

    assert reloaded.row_count == len(rows)
    _, back = workspace.query("main", "SELECT * FROM reloaded ORDER BY name")
    assert back == rows


def test_a_null_is_an_absent_element_so_it_comes_back_null(workspace, root):
    """An empty element would come back as text and stop being a NULL."""
    target = root / "nulls.xml"
    export_module.export_rows(["a", "b"], [(1, None), (2, "x")], str(target))

    assert "<b>" not in target.read_text().split("</row>")[0]
    workspace.load_file(str(target), "main", table_name="nulls")
    _, rows = workspace.query("main", "SELECT b FROM nulls ORDER BY a")
    assert rows[0][0] is None


def test_markup_in_a_value_is_escaped_and_survives(workspace, root):
    target = root / "markup.xml"
    export_module.export_rows(["a"], [("<b> & </b>",)], str(target))

    assert "&lt;b&gt;" in target.read_text()
    workspace.load_file(str(target), "main", table_name="markup")
    _, rows = workspace.query("main", "SELECT a FROM markup")
    assert rows[0][0] == "<b> & </b>"


def test_a_column_name_xml_cannot_spell_is_refused_not_mangled(workspace, root):
    """The remedy is one AS in the caller's SQL, so say that rather than guess a name."""
    target = root / "bad_name.xml"

    with pytest.raises(export_module.ExportError) as raised:
        export_module.export_rows(["first name"], [(1,)], str(target))

    assert "first name" in str(raised.value)
    assert not target.exists()


# ---------------------------------------------------------------------------
# YAML, Markdown, fixed width, and the columnar three
# ---------------------------------------------------------------------------


def test_yaml_is_read_on_the_same_rules_as_json(workspace, root):
    """It parses to the same structures, so it gets the same reader logic."""
    (info,) = workspace.load_file(str(root / "config_rows.yaml"), "main")

    assert info.row_count == 2
    assert [c.name for c in info.columns] == ["name", "role", "salary"]
    assert info.notes == ()


def test_a_wrapped_yaml_document_names_the_key_it_loaded(workspace, root):
    (info,) = workspace.load_file(str(root / "wrapped.yaml"), "main")

    assert info.row_count == 2
    assert len(info.notes) == 1
    assert "employees" in info.notes[0]


def test_yaml_round_trips_through_the_export(workspace, root):
    workspace.load_file(str(root / "config_rows.yaml"), "main")
    columns, rows = workspace.query("main", "SELECT * FROM config_rows ORDER BY name")
    target = root / "again.yaml"

    export_module.export_rows(columns, rows, str(target))
    (reloaded,) = workspace.load_file(str(target), "main", table_name="reloaded")

    _, back = workspace.query("main", "SELECT * FROM reloaded ORDER BY name")
    assert reloaded.row_count == 2
    assert back == rows


def test_markdown_is_written_as_a_table_and_is_write_only(workspace, root):
    """pandas has no Markdown reader, so offering one would be a promise we cannot keep."""
    target = root / "out.md"

    export_module.export_rows(["name", "salary"], [("Ada", 120000)], str(target))

    written = target.read_text()
    assert "| name" in written and "Ada" in written and "120000" in written
    assert ".md" not in loader_module.READERS


def test_fixed_width_is_read_and_says_the_boundaries_were_inferred(workspace, root):
    """Nothing in the file states the columns, so the caller is told they were guessed."""
    (info,) = workspace.load_file(str(root / "payroll.fwf"), "main")

    assert info.row_count == 3
    assert [c.name for c in info.columns] == ["name", "salary"]
    assert len(info.notes) == 1
    assert "inferred" in info.notes[0]


@pytest.mark.parametrize("suffix", [".parquet", ".feather", ".orc"])
def test_a_columnar_format_round_trips_every_type_exactly(workspace, root, suffix):
    """Typed formats: the types survive, so this is stricter than the CSV round trip."""
    workspace.load_file(str(root / "simple.csv"), "main")
    columns, rows = workspace.query(
        "main", "SELECT name, age, salary FROM simple ORDER BY name"
    )
    target = root / f"out{suffix}"

    result = export_module.export_rows(columns, rows, str(target))
    assert result.row_count == 5

    (reloaded,) = workspace.load_file(str(target), "main", table_name="back")
    assert reloaded.row_count == 5
    by_name = {c.name: c for c in reloaded.columns}
    assert by_name["age"].declared_type == "INTEGER"

    _, back = workspace.query(
        "main", "SELECT name, age, salary FROM back ORDER BY name"
    )
    assert back == rows


def test_a_columnar_file_that_is_not_one_is_refused_by_name(workspace, root):
    target = root / "lying.parquet"
    target.write_bytes(b"this is not parquet")

    with pytest.raises(LoadError, match="lying.parquet"):
        workspace.load_file(str(target), "main")


# ---------------------------------------------------------------------------
# Spreadsheets and HTML — the formats that hold more than one table
# ---------------------------------------------------------------------------


def test_a_workbook_becomes_one_table_per_sheet(workspace, root):
    """Reading only the first sheet would leave the rest unreachable."""
    landed = workspace.load_file(str(root / "workbook.xlsx"), "main")

    assert [info.name for info in landed] == ["staff", "departments"]
    assert [info.row_count for info in landed] == [3, 2]

    _, rows = workspace.query("main", "SELECT sum(salary) FROM staff")
    assert rows[0][0] == 353000
    _, rows = workspace.query("main", "SELECT floor FROM departments ORDER BY floor")
    assert [r[0] for r in rows] == [1, 3]


def test_a_sheet_keeps_its_own_name_not_the_filename(workspace, root):
    (info,) = workspace.load_file(str(root / "one_sheet.xlsx"), "main")
    assert info.name == "people"


def test_a_table_name_cannot_cover_several_sheets_and_says_so(workspace, root):
    with pytest.raises(LoadError, match="table_name"):
        workspace.load_file(str(root / "workbook.xlsx"), "main", table_name="all")


def test_spreadsheet_types_survive(workspace, root):
    (info,) = workspace.load_file(str(root / "one_sheet.xlsx"), "main")
    by_name = {c.name: c for c in info.columns}

    assert by_name["salary"].declared_type == "INTEGER"
    assert by_name["started"].temporal_standard == "iso8601_utc"


def test_a_legacy_xls_is_read(workspace, root):
    """xlrd reads it; nothing writes it, which is why .xls is read-only here."""
    (info,) = workspace.load_file(str(root / "legacy.xls"), "main")

    assert info.name == "staff"
    assert info.row_count == 3
    assert ".xls" not in export_module.WRITERS


def test_an_ods_workbook_reads_like_any_other(workspace, root):
    landed = workspace.load_file(str(root / "book.ods"), "main")
    assert [info.name for info in landed] == ["staff", "departments"]


def test_every_table_on_an_html_page_is_loaded(workspace, root):
    """A page's tables are numbered, because HTML gives them no names."""
    landed = workspace.load_file(str(root / "tables.html"), "main")

    assert len(landed) == 2
    assert [info.row_count for info in landed] == [3, 2]


def test_a_page_with_one_table_needs_no_number(workspace, root):
    (info,) = workspace.load_file(str(root / "one_table.html"), "main")
    assert info.name == "one_table"
    assert info.row_count == 3


def test_html_round_trips_through_the_export(workspace, root):
    workspace.load_file(str(root / "one_table.html"), "main")
    columns, rows = workspace.query("main", "SELECT * FROM one_table ORDER BY name")
    target = root / "again.html"

    export_module.export_rows(columns, rows, str(target))
    (reloaded,) = workspace.load_file(str(target), "main", table_name="reloaded")

    assert reloaded.row_count == 3


def test_xlsx_round_trips_through_the_export(workspace, root):
    workspace.load_file(str(root / "one_sheet.xlsx"), "main")
    columns, rows = workspace.query(
        "main", "SELECT name, salary FROM people ORDER BY name"
    )
    target = root / "again.xlsx"

    export_module.export_rows(columns, rows, str(target))
    (reloaded,) = workspace.load_file(str(target), "main", table_name="reloaded")

    _, back = workspace.query("main", "SELECT name, salary FROM reloaded ORDER BY name")
    assert back == rows


def test_a_page_with_no_table_is_refused(workspace, root):
    target = root / "prose.html"
    target.write_text("<html><body><p>No tables here at all.</p></body></html>")

    with pytest.raises(LoadError, match="no table"):
        workspace.load_file(str(target), "main")


@pytest.mark.slow
def test_an_html_table_too_large_to_parse_says_so_instead_of_unknown_error(
    workspace, root
):
    """lxml's own words for this are, in full, "unknown error".

    Measured: pandas locates tables with ``//table``, and libxml2 abandons an
    XPath evaluation past ``XPATH_MAX_NODES`` — 10,000,000, exactly. 1,230,000
    three-column rows parse; 1,250,000 do not. A row costs about
    ``2 * columns + 2`` nodes, which is why the wide benchmark corpus (1M x 11)
    fails at a quarter of that row count.

    Marked slow because the file has to be genuinely over the limit — there is
    no smaller input that produces this failure, and a mocked one would test the
    mock. The refusal is what matters: this server will *write* an HTML table of
    any size and cannot read that one back, so the far end of a round trip it
    offers has to say what happened and name a format that works.
    """
    target = root / "enormous.html"
    cells = "".join(f"<td>v{column}</td>" for column in range(3))
    with target.open("w") as handle:
        handle.write(
            "<table>\n<thead><tr><th>a</th><th>b</th><th>c</th></tr></thead>\n"
        )
        handle.write("<tbody>\n")
        for _ in range(1_300_000):
            handle.write(f"<tr>{cells}</tr>\n")
        handle.write("</tbody>\n</table>\n")

    with pytest.raises(LoadError) as raised:
        workspace.load_file(str(target), "main")

    message = str(raised.value)
    assert "10,000,000" in message
    assert "unknown error" not in message
    # It names a way out, rather than only refusing.
    assert ".parquet" in message


def test_apple_numbers_is_read_without_its_empty_grid(workspace, root):
    """A Numbers table is a fixed canvas, so the cells past the data come back null."""
    (info,) = workspace.load_file(str(root / "payroll.numbers"), "main")

    assert info.name == "staff"
    assert info.row_count == 2
    assert [c.name for c in info.columns] == ["name", "role", "salary"]

    _, rows = workspace.query("main", "SELECT sum(salary) FROM staff")
    assert rows[0][0] == 255000


# ---------------------------------------------------------------------------
# The delimiter: a fact about the source the caller often knows
# ---------------------------------------------------------------------------


def test_a_semicolon_file_loads_as_one_fat_column_and_says_so(workspace, root):
    """The silent failure the parameter exists for. Without the warning it is invisible."""
    (info,) = workspace.load_file(str(root / "semicolons.csv"), "main")

    assert len(info.columns) == 1
    assert len(info.notes) == 1
    assert "delimiter" in info.notes[0]
    assert "';'" in info.notes[0]  # names the one it found, not a list to guess from


def test_the_delimiter_parameter_reads_the_same_file_properly(workspace, root):
    (info,) = workspace.load_file(str(root / "semicolons.csv"), "main", delimiter=";")

    assert [c.name for c in info.columns] == ["name", "role", "salary"]
    assert info.row_count == 2
    assert info.notes == ()

    _, rows = workspace.query("main", "SELECT sum(salary) FROM semicolons")
    assert rows[0][0] == 255000


def test_a_tsv_still_defaults_to_tab(workspace, root):
    """The default is per extension. A flat comma default would regress this."""
    (info,) = workspace.load_file(str(root / "mixed_tabs.tsv"), "main")
    assert len(info.columns) > 1


def test_an_explicit_delimiter_overrides_what_the_extension_implied(workspace, root):
    (info,) = workspace.load_file(str(root / "pipes.txt"), "main", delimiter="|")
    assert [c.name for c in info.columns] == ["name", "role"]


def test_a_delimiter_on_a_format_that_has_none_is_refused(workspace, root):
    """Silently ignoring it would leave the caller believing it did something."""
    with pytest.raises(LoadError, match="delimiter"):
        workspace.load_file(str(root / "records.json"), "main", delimiter=";")


def test_nothing_sniffs_the_delimiter(workspace, root):
    """A sniffer that is right most of the time is the fail-open shape, not a fix.

    The semicolon file above must still load as one column by default — if some
    later change starts guessing, this fails.
    """
    (info,) = workspace.load_file(str(root / "semicolons.csv"), "main")
    assert len(info.columns) == 1


def test_a_delimiter_chooses_the_separator_on_the_way_out(workspace, root):
    target = root / "out.csv"

    export_module.export_rows(["a", "b"], [(1, 2)], str(target), delimiter=";")

    assert target.read_text().splitlines() == ["a;b", "1;2"]


def test_a_delimiter_on_a_non_delimited_target_is_ignored(workspace, root):
    """Ignored, not refused — unlike the read side.

    The file the caller asked for is correct Parquet either way, so a delimiter
    here is inert rather than misleading, and a caller carrying a default
    delimiter through a wrapper should not be blocked by it.
    """
    target = root / "out.parquet"

    result = export_module.export_rows(["a", "b"], [(1, 2)], str(target), delimiter=";")

    assert result.row_count == 1
    (reloaded,) = workspace.load_file(str(target), "main", table_name="back")
    assert [c.name for c in reloaded.columns] == ["a", "b"]


def test_the_written_delimiter_round_trips_when_read_back_with_the_same_one(
    workspace, root
):
    """The two sides are the same fact about the file, so they must agree."""
    target = root / "piped.csv"
    export_module.export_rows(
        ["name", "salary"], [("Ada", 120000)], str(target), delimiter="|"
    )

    (info,) = workspace.load_file(str(target), "main", delimiter="|")
    assert [c.name for c in info.columns] == ["name", "salary"]


def test_a_tsv_written_without_a_delimiter_is_still_tab_separated(workspace, root):
    """The suffix keeps deciding when nothing overrides it."""
    target = root / "plain.tsv"
    export_module.export_rows(["a", "b"], [(1, 2)], str(target))
    assert target.read_text().splitlines()[0] == "a\tb"
