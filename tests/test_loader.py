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

from localdata_mcp import export as export_module
from localdata_mcp import paths as paths_module
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
    monkeypatch.setenv(paths_module.ROOT_ENV_VAR, str(shared))
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


def test_opening_a_sqlite_file_is_read_only(root):
    _build_database(root / "hr.db")
    ws = Workspace.open_sqlite_file(str(root / "hr.db"))
    try:
        assert "departments" in ws.tables
        assert ws.tables["departments"].row_count == 3
        with pytest.raises(LoadError):
            ws.load_file(str(root / "simple.csv"))
    finally:
        ws.close()


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
