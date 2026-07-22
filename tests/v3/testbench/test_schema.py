"""tests/v3/testbench/test_schema.py — schema.py owns DDL, migrations, PRAGMAs.

Covers: migration to the current version, version bookkeeping in meta,
WAL journaling, per-connection foreign-key enforcement, and the
additive-only refusal of a store from the future.
"""

import sqlite3
from pathlib import Path

import pytest

from localdata_mcp.testbench.results_store import schema

from .conftest import BUSY_TIMEOUT_MS, open_store


class TestConnectPragmas:
    def test_wal_mode_is_active(self, tmp_path: Path) -> None:
        connection = open_store(tmp_path / "s.db")
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode == "wal"
        connection.close()

    def test_foreign_keys_are_enforced_per_connection(self, tmp_path: Path) -> None:
        """SQLite defaults FKs off; every store connection must switch them on."""
        connection = open_store(tmp_path / "s.db")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO battery_results"
                " (run_id, test_id, pass, numeric_output, duration_ms)"
                " VALUES ('no-such-run', 't1', 1, NULL, 1.0)"
            )
        connection.close()

    def test_busy_timeout_is_applied(self, tmp_path: Path) -> None:
        connection = schema.connect(tmp_path / "s.db", busy_timeout_ms=BUSY_TIMEOUT_MS)
        timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == BUSY_TIMEOUT_MS
        connection.close()


class TestMigrations:
    def test_fresh_store_reaches_current_version(self, tmp_path: Path) -> None:
        connection = open_store(tmp_path / "s.db")
        assert schema.schema_version(connection) == schema.SCHEMA_VERSION
        connection.close()

    def test_tables_exist_after_migration(self, tmp_path: Path) -> None:
        connection = open_store(tmp_path / "s.db")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"meta", "battery_runs", "battery_results"} <= tables
        connection.close()

    def test_ensure_schema_is_idempotent(self, tmp_path: Path) -> None:
        connection = open_store(tmp_path / "s.db")
        schema.ensure_schema(connection)  # second run must be a no-op
        assert schema.schema_version(connection) == schema.SCHEMA_VERSION
        connection.close()

    def test_unmigrated_store_reports_version_zero(self, tmp_path: Path) -> None:
        connection = schema.connect(tmp_path / "s.db", busy_timeout_ms=BUSY_TIMEOUT_MS)
        assert schema.schema_version(connection) == 0
        connection.close()

    def test_future_store_is_refused(self, tmp_path: Path) -> None:
        """Additive-only: an older release must refuse a newer store."""
        connection = open_store(tmp_path / "s.db")
        connection.execute(
            "UPDATE meta SET schema_version = ?", (schema.SCHEMA_VERSION + 1,)
        )
        connection.commit()
        with pytest.raises(schema.SchemaVersionError):
            schema.ensure_schema(connection)
        connection.close()

    def test_run_mode_check_constraint(self, tmp_path: Path) -> None:
        connection = open_store(tmp_path / "s.db")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO battery_runs (run_id, battery_name, run_mode,"
                " dataset_hash, software_versions, started_at)"
                " VALUES ('r1', 'b', 'not-a-mode', 'h', '{}', 't')"
            )
        connection.close()
