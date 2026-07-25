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

import pytest

from localdata_mcp import config as config_module
from localdata_mcp import export as export_module
from localdata_mcp.config import Config
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
    ws = Workspace.in_memory()
    yield ws
    ws.close()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_simple_csv_loads_and_answers_correctly(workspace, root):
    info = workspace.load_file(str(root / "simple.csv"))

    assert info.row_count == 5
    assert [c.name for c in info.columns] == [
        "name",
        "age",
        "department",
        "salary",
        "start_date",
        "active",
    ]

    _, rows = workspace.query("SELECT sum(salary) FROM simple")
    assert rows[0][0] == 75000 + 65000 + 70000 + 80000 + 55000

    _, rows = workspace.query(
        "SELECT count(*) FROM simple WHERE department = 'Engineering'"
    )
    assert rows[0][0] == 2


def test_numeric_column_is_declared_numeric(workspace, root):
    info = workspace.load_file(str(root / "simple.csv"))
    by_name = {c.name: c for c in info.columns}
    assert by_name["age"].declared_type == "INTEGER"
    assert by_name["salary"].declared_type == "INTEGER"
    assert by_name["department"].declared_type == "TEXT"


def test_messy_csv_loads_without_losing_rows(workspace, root):
    """Quoted newlines, embedded commas, emoji and a blank row all survive."""
    info = workspace.load_file(str(root / "messy_mixed_types.csv"))
    assert info.row_count > 0

    _, rows = workspace.query(
        "SELECT count(*) FROM messy_mixed_types WHERE name LIKE '%Emoji%'"
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
    info = workspace.load_file(str(target))

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
    _, naive = workspace.query("SELECT avg(v) FROM partly_numeric")
    assert naive[0][0] == pytest.approx(15 / 6)
    _, guarded = workspace.query(
        "SELECT avg(CAST(v AS REAL)) FROM partly_numeric WHERE typeof(v) = 'text' "
        "AND CAST(v AS REAL) != 0"
    )
    assert guarded[0][0] == pytest.approx(3.0)


def test_uniformly_numeric_text_column_is_not_flagged(workspace, root):
    """No false positive: a column that is entirely non-numeric is not mixed."""
    target = root / "all_text.csv"
    target.write_text("v\nalpha\nbeta\ngamma\n")
    info = workspace.load_file(str(target))
    assert info.columns[0].is_mixed is False
    assert info.mixed_columns == []


def test_unicode_survives_the_round_trip(workspace, root):
    workspace.load_file(str(root / "messy_mixed_types.csv"))
    _, rows = workspace.query(
        "SELECT name FROM messy_mixed_types WHERE name LIKE '%Garc%'"
    )
    assert rows and "í" in rows[0][0]


def test_truncated_file_loads(workspace, root):
    """A missing trailing value is a NULL, not a failure."""
    info = workspace.load_file(str(root / "truncated.csv"))
    assert info.row_count == 2
    _, rows = workspace.query("SELECT count(*) FROM truncated WHERE value IS NULL")
    assert rows[0][0] == 1


def test_empty_file_is_refused_clearly(workspace, root):
    with pytest.raises(LoadError):
        workspace.load_file(str(root / "empty.csv"))


def test_duplicate_and_blank_headers_become_usable_columns(workspace, root, tmp_path):
    target = root / "dupes.csv"
    target.write_text("a,a,,1x\n1,2,3,4\n")
    info = workspace.load_file(str(target))
    names = [c.name for c in info.columns]
    assert len(set(names)) == len(names), names
    assert all(names), names


def test_unsupported_extension_names_what_is_supported(workspace, root):
    target = root / "thing.parquet"
    target.write_bytes(b"not really parquet")
    with pytest.raises(LoadError, match="csv"):
        workspace.load_file(str(target))


def test_table_name_can_be_overridden(workspace, root):
    info = workspace.load_file(str(root / "simple.csv"), table_name="staff")
    assert info.name == "staff"
    _, rows = workspace.query("SELECT count(*) FROM staff")
    assert rows[0][0] == 5


# ---------------------------------------------------------------------------
# Path boundary
# ---------------------------------------------------------------------------


def test_path_outside_root_is_refused(workspace, tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n1\n")
    with pytest.raises(PathNotAllowed):
        workspace.load_file(str(outside))


def test_traversal_out_of_root_is_refused(workspace, root, tmp_path):
    outside = tmp_path / "secret.csv"
    outside.write_text("a\n1\n")
    with pytest.raises(PathNotAllowed):
        workspace.load_file(str(root / ".." / "secret.csv"))


def test_symlink_pointing_outside_root_is_refused(workspace, root, tmp_path):
    """Resolve-then-check: the link resolves to its real, disallowed location."""
    outside = tmp_path / "secret.csv"
    outside.write_text("a\n1\n")
    link = root / "innocent.csv"
    link.symlink_to(outside)
    with pytest.raises(PathNotAllowed):
        workspace.load_file(str(link))


def test_missing_file_is_refused(workspace, root):
    with pytest.raises(PathNotAllowed, match="No such file"):
        workspace.load_file(str(root / "nope.csv"))


# ---------------------------------------------------------------------------
# Attaching a database, and the flagship join across sources
# ---------------------------------------------------------------------------


def _build_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE departments (department TEXT, floor INTEGER)")
    connection.executemany(
        "INSERT INTO departments VALUES (?, ?)",
        [("Engineering", 3), ("Sales", 1), ("Marketing", 2)],
    )
    connection.commit()
    connection.close()


def test_join_across_a_csv_and_a_database(workspace, root):
    _build_database(root / "hr.db")
    workspace.load_file(str(root / "simple.csv"))
    workspace._conn.execute(
        "ATTACH DATABASE ? AS hr", (f"file:{root / 'hr.db'}?mode=ro",)
    )

    columns, rows = workspace.query(
        "SELECT s.name, d.floor FROM simple s "
        "JOIN hr.departments d ON s.department = d.department "
        "ORDER BY s.name"
    )
    assert columns == ["name", "floor"]
    # Four, not five: the CSV's fifth employee is in HR, which the database has
    # no row for, so an inner join correctly drops her.
    assert len(rows) == 4
    assert ("Alice Johnson", 3) in [tuple(r) for r in rows]

    _, unmatched = workspace.query(
        "SELECT s.name FROM simple s "
        "LEFT JOIN hr.departments d ON s.department = d.department "
        "WHERE d.floor IS NULL"
    )
    assert [r[0] for r in unmatched] == ["Eve Davis"]


def test_attached_database_cannot_be_written(workspace, root):
    _build_database(root / "hr.db")
    workspace._conn.execute(
        "ATTACH DATABASE ? AS hr", (f"file:{root / 'hr.db'}?mode=ro",)
    )
    with pytest.raises(sqlite3.OperationalError):
        workspace._conn.execute("INSERT INTO hr.departments VALUES ('X', 9)")


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

    workspace.load_file(str(root / "simple.csv"), schema="scratch")
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
    workspace.load_file(str(root / "large_dataset.csv"), schema="scratch")
    full = workspace.resident_bytes("scratch")

    workspace.drop_table("scratch", "large_dataset")
    emptied = workspace.resident_bytes("scratch")

    assert full > 0
    assert emptied < full
    # The uncorrected reading is the one that would have ratcheted: it stays at
    # the high-water mark, which is exactly what the correction is subtracting.
    page_size = workspace._scalar('PRAGMA "scratch".page_size')
    uncorrected = workspace._scalar('PRAGMA "scratch".page_count') * page_size
    assert uncorrected >= full
    assert uncorrected > emptied


def test_vacuum_into_produces_a_database_that_still_answers(workspace, root, tmp_path):
    """Assert on what the copy returns, never on the fact that a file appeared."""
    workspace.attach_memory("scratch")
    workspace.load_file(str(root / "simple.csv"), schema="scratch")
    original = workspace.query("SELECT name, salary FROM scratch.simple ORDER BY name")
    target = tmp_path / "copy.sqlite"

    workspace.vacuum_into("scratch", target)

    copied = sqlite3.connect(target)
    try:
        assert copied.execute("SELECT count(*) FROM simple").fetchone()[0] == 5
        assert [
            tuple(r)
            for r in copied.execute("SELECT name, salary FROM simple ORDER BY name")
        ] == [tuple(r) for r in original[1]]
        assert (
            copied.execute("SELECT sum(salary) FROM simple").fetchone()[0]
            == (workspace.query("SELECT sum(salary) FROM scratch.simple")[1][0][0])
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
    workspace.load_file(str(root / "simple.csv"), schema="scratch")
    target = tmp_path / "copy.sqlite"
    workspace.vacuum_into("scratch", target)

    with pytest.raises(sqlite3.Error, match="already exists"):
        workspace.vacuum_into("scratch", target)


def test_vacuum_into_does_not_refuse_a_zero_length_target(workspace, root, tmp_path):
    """Recorded because it is the gap that makes the path-boundary guard load-bearing."""
    workspace.attach_memory("scratch")
    workspace.load_file(str(root / "simple.csv"), schema="scratch")
    target = tmp_path / "copy.sqlite"
    target.write_bytes(b"")

    workspace.vacuum_into("scratch", target)

    assert target.stat().st_size > 0


def test_a_writable_attach_accepts_what_a_readonly_one_refuses(workspace, root):
    """The grant is a different URI, not a check this module is trusted to make."""
    _build_database(root / "hr.db")
    workspace.attach_file("locked", root / "hr.db")
    workspace.attach_file("open", root / "hr.db", readonly=False)

    with pytest.raises(sqlite3.OperationalError):
        workspace._conn.execute("INSERT INTO locked.departments VALUES ('X', 9)")

    workspace._conn.execute("INSERT INTO open.departments VALUES ('X', 9)")
    workspace._conn.commit()
    _, rows = workspace.query("SELECT floor FROM open.departments WHERE department='X'")
    assert [tuple(r) for r in rows] == [(9,)]


def test_a_question_mark_in_a_filename_is_not_read_as_a_uri_query(workspace, root):
    """The connection runs in URI mode, so the path has to be percent-encoded."""
    awkward = root / "why? not.db"
    _build_database(awkward)

    workspace.attach_file("odd", awkward)

    _, rows = workspace.query("SELECT count(*) FROM odd.departments")
    assert [tuple(r) for r in rows] == [(3,)]


# ---------------------------------------------------------------------------
# Composing a database from a declared schema
# ---------------------------------------------------------------------------


def test_a_declared_table_is_created_and_writable(workspace):
    workspace.attach_memory("book")

    info = workspace.create_table(
        "book", "ledger", {"entry": "TEXT", "amount": "INTEGER"}
    )

    assert info.qualified == "book.ledger"
    assert info.row_count == 0
    assert [c.name for c in info.columns] == ["entry", "amount"]

    workspace._conn.execute("INSERT INTO book.ledger VALUES ('rent', 1200)")
    workspace._conn.commit()
    _, rows = workspace.query("SELECT sum(amount) FROM book.ledger")
    assert [tuple(r) for r in rows] == [(1200,)]


def test_a_declared_type_outside_sqlites_own_is_refused(workspace):
    """The type reaches DDL unquoted, so the vocabulary has to be closed."""
    workspace.attach_memory("book")
    with pytest.raises(LoadError, match="INTEGER"):
        workspace.create_table("book", "ledger", {"amount": "BIGINT); DROP TABLE x--"})
    assert not workspace.has_table("book", "ledger")


def test_a_declared_table_with_no_columns_is_refused(workspace):
    workspace.attach_memory("book")
    with pytest.raises(LoadError, match="at least one column"):
        workspace.create_table("book", "ledger", {})


def test_declared_column_names_are_sanitised_like_any_other(workspace):
    workspace.attach_memory("book")
    info = workspace.create_table("book", "ledger", {"Total Amount (£)": "REAL"})
    assert [c.name for c in info.columns] == ["total_amount"]


def test_dropping_a_table_removes_it_and_dropping_it_twice_is_an_error(workspace, root):
    workspace.attach_memory("scratch")
    workspace.load_file(str(root / "simple.csv"), schema="scratch")
    assert workspace.has_table("scratch", "simple")

    workspace.drop_table("scratch", "simple")

    assert not workspace.has_table("scratch", "simple")
    with pytest.raises(LoadError, match="simple"):
        workspace.drop_table("scratch", "simple")


# ---------------------------------------------------------------------------
# Export — the overwrite boundary
# ---------------------------------------------------------------------------


def test_export_writes_the_rows(workspace, root):
    workspace.load_file(str(root / "simple.csv"))
    columns, rows = workspace.query("SELECT name, salary FROM simple ORDER BY name")
    target = root / "out.csv"

    result = export_module.export_csv(columns, rows, str(target))

    assert result.row_count == 5
    assert result.replaced_existing is False
    written = target.read_text().splitlines()
    assert written[0] == "name,salary"
    assert len(written) == 6


def test_export_refuses_an_existing_file(workspace, root):
    target = root / "out.csv"
    target.write_text("do not lose me\n")
    with pytest.raises(PathNotAllowed, match="already exists"):
        export_module.export_csv(["a"], [(1,)], str(target))
    assert target.read_text() == "do not lose me\n"


def test_export_replaces_when_told_to(workspace, root):
    target = root / "out.csv"
    target.write_text("stale\n")
    result = export_module.export_csv(["a"], [(1,), (2,)], str(target), overwrite=True)
    assert result.replaced_existing is True
    assert target.read_text().splitlines() == ["a", "1", "2"]


def test_export_is_owner_readable_only(workspace, root):
    target = root / "out.csv"
    export_module.export_csv(["a"], [(1,)], str(target))
    assert oct(os.stat(target).st_mode & 0o777) == "0o600"


def test_export_outside_root_is_refused(root, tmp_path):
    with pytest.raises(PathNotAllowed):
        export_module.export_csv(["a"], [(1,)], str(tmp_path / "escape.csv"))


def test_export_to_a_missing_directory_is_refused(root):
    with pytest.raises(PathNotAllowed, match="Directory does not exist"):
        export_module.export_csv(["a"], [(1,)], str(root / "nope" / "out.csv"))


def test_export_round_trips_back_into_the_workspace(workspace, root):
    """The strongest check that the export is really well-formed."""
    workspace.load_file(str(root / "messy_mixed_types.csv"))
    columns, rows = workspace.query("SELECT * FROM messy_mixed_types")
    target = root / "exported.csv"
    export_module.export_csv(columns, rows, str(target))

    reloaded = workspace.load_file(str(target), table_name="reloaded")
    assert reloaded.row_count == len(rows)
