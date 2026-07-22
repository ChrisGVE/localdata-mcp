"""localdata_mcp/testbench/results_store/schema.py — the schema's one owner.

Sole owner of the results-store DDL and of forward migrations
(ARCHITECTURE.md section 5, "Results provenance store"). The posture is
additive-only versioned migrations, run before any write, so historical
rows stay readable across releases. Neighbors: store.py writes through
this schema; merge.py asserts version equality across store files.

Every connection opens with foreign keys ON (SQLite defaults them OFF),
WAL journaling, and the caller-provided busy timeout. All tunables are
required parameters — their configuration home arrives with the config
nexus (E1); this module declares no defaults of its own.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Dict

SCHEMA_VERSION = 1


class SchemaVersionError(Exception):
    """The on-disk store is ahead of this code's schema version."""


def connect(db_path: Path | str, *, busy_timeout_ms: int) -> sqlite3.Connection:
    """Open a results-store connection with the required pragmas applied."""
    connection = sqlite3.connect(str(db_path))
    connection.isolation_level = None  # explicit transactions (store.py)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def schema_version(connection: sqlite3.Connection) -> int:
    """The store's current version; 0 for a store never migrated."""
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
    ).fetchone()
    if table_exists is None:
        return 0
    row = connection.execute("SELECT schema_version FROM meta").fetchone()
    return int(row[0]) if row else 0


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Migrate forward to SCHEMA_VERSION; refuse a store from the future."""
    current = schema_version(connection)
    if current > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"store is at schema_version {current}, this code knows only "
            f"{SCHEMA_VERSION} - refusing (migrations are forward-only)"
        )
    connection.execute("BEGIN IMMEDIATE")
    try:
        for version in range(current + 1, SCHEMA_VERSION + 1):
            _MIGRATIONS[version](connection)
            _record_version(connection, version)
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _record_version(connection: sqlite3.Connection, version: int) -> None:
    """Keep meta at exactly one row carrying the store's version."""
    if version == 1:
        connection.execute("INSERT INTO meta (schema_version) VALUES (?)", (version,))
    else:
        connection.execute("UPDATE meta SET schema_version = ?", (version,))


# Version-1 DDL: the three-table shape of ARCHITECTURE.md section 5.
_V1_DDL = (
    "CREATE TABLE meta (schema_version INTEGER NOT NULL)",
    """
    CREATE TABLE battery_runs (
        run_id            TEXT PRIMARY KEY,
        battery_name      TEXT NOT NULL,
        run_mode          TEXT NOT NULL
                          CHECK (run_mode IN ('deterministic', 'randomized')),
        seed              INTEGER,
        dataset_hash      TEXT NOT NULL,
        software_versions TEXT NOT NULL,  -- JSON object
        started_at        TEXT NOT NULL,  -- ISO-8601 UTC
        finished_at       TEXT,           -- NULL while the run is open
        git_sha           TEXT
    )
    """,
    """
    CREATE TABLE battery_results (
        run_id         TEXT NOT NULL
                       REFERENCES battery_runs (run_id) ON DELETE CASCADE,
        test_id        TEXT NOT NULL,  -- domain.dataset.operation naming
        pass           INTEGER NOT NULL,
        numeric_output TEXT,            -- JSON, nullable by design
        duration_ms    REAL,
        UNIQUE (run_id, test_id)
    )
    """,
    # Serves both the retention prune's per-group ordering and the
    # longitudinal comparison joins on (battery_name, run_mode).
    """
    CREATE INDEX battery_runs_group_recency
        ON battery_runs (battery_name, run_mode, started_at)
    """,
)


def _migrate_to_v1(connection: sqlite3.Connection) -> None:
    for statement in _V1_DDL:
        connection.execute(statement)


_MIGRATIONS: Dict[int, Callable[[sqlite3.Connection], None]] = {1: _migrate_to_v1}
