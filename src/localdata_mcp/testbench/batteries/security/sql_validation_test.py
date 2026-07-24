"""testbench/batteries/security/sql_validation_test.py — NFR-104 at the L3 seam.

The AST allow-list (E6.2, unit-tested at the `classify()` seam in
tests/v3/test_sql_validate.py) proven through the wire the agent uses: a
hostile SQL string handed to the `query` tool becomes a structured FR-403
refusal, never a backend round-trip. This is the security battery's
NFR-104 row (S7.1) — the dynamic, end-to-end complement to the static
construct-level unit suite.

Coverage runs unconditionally against the two file/embedded engines
(SQLite, DuckDB); the networked dialects (PostgreSQL, MySQL) add their
server-side deny-outright rows whenever the E14.2 fixture DSNs are present
and SKIP with a named reason otherwise. MSSQL/Oracle deny-outright stays
at the unit seam (extras-gated, nightly) — no per-PR fixture engine.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.endpoints import Posture

from . import _seam

_PG_DSN = os.environ.get("LOCALDATA_TEST_POSTGRES_DSN")
_MYSQL_DSN = os.environ.get("LOCALDATA_TEST_MYSQL_DSN")


def _seed(guard: Chokepoint, endpoint: str) -> None:
    """One tiny table so a legitimate read has something to return."""
    with guard._persistence.connection(endpoint) as connection:
        if hasattr(connection, "exec_driver_sql"):
            connection.exec_driver_sql("CREATE TABLE t (id INTEGER, label TEXT)")
            connection.exec_driver_sql("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
            connection.commit()
        else:  # native DuckDB connection
            connection.execute("CREATE TABLE t (id INTEGER, label TEXT)")
            connection.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")


@pytest.fixture()
def sql_bench(tmp_path: Path):
    declarations = _seam.declare(
        sqlite=(f"sqlite:///{tmp_path / 'sec.db'}", "read_write"),
        duckdb=(f"duckdb:///{tmp_path / 'sec.duckdb'}", "read_write"),
    )
    with _seam.booted(
        allowed_paths=(str(tmp_path),), declarations=declarations
    ) as guard:
        _seed(guard, "sqlite")
        _seed(guard, "duckdb")
        yield guard, tmp_path


# -- statement-type / structural refusals (dialect-independent) --------
#
# Each payload is a known regex-denylist bypass or an out-of-category
# construct that a statement-type-only gate would wave through; the AST
# allow-list refuses every one at the read path.
_STRUCTURAL_PAYLOADS: tuple[tuple[str, str], ...] = (
    ("drop_table", "DROP TABLE t"),
    ("create_table", "CREATE TABLE evil (x INTEGER)"),
    ("alter_table", "ALTER TABLE t ADD COLUMN y INTEGER"),
    ("delete_rows", "DELETE FROM t WHERE id = 1"),
    ("update_rows", "UPDATE t SET label = 'x' WHERE id = 1"),
    ("insert_rows", "INSERT INTO t (id) VALUES (99)"),
    ("grant", "GRANT SELECT ON t TO someone"),
    ("multi_statement", "SELECT 1; SELECT 2"),
    ("stacked_mutation", "SELECT id FROM t; DROP TABLE t"),
    ("parse_failure", "SELECT * FROM"),
    ("data_modifying_cte", "WITH d AS (DELETE FROM t RETURNING id) SELECT * FROM d"),
)


@pytest.mark.parametrize("engine", ["sqlite", "duckdb"])
@pytest.mark.parametrize(
    ("label", "sql"), _STRUCTURAL_PAYLOADS, ids=[p[0] for p in _STRUCTURAL_PAYLOADS]
)
def test_out_of_category_sql_is_refused(
    sql_bench, engine: str, label: str, sql: str
) -> None:
    _guard, _tmp = sql_bench
    envelope = _seam.call_envelope("query", {"endpoint": engine, "sql": sql})
    _seam.expect_refused(envelope)


# -- deny-outright network / capability constructs, per dialect --------
_SQLITE_DENY: tuple[tuple[str, str], ...] = (
    ("sqlite_load_extension", "SELECT load_extension('/tmp/e.so')"),
)
_DUCKDB_DENY: tuple[tuple[str, str], ...] = (
    ("duckdb_install_httpfs", "INSTALL httpfs"),
    ("duckdb_load_httpfs", "LOAD httpfs"),
)


@pytest.mark.parametrize(
    ("engine", "label", "sql"),
    [("sqlite", lbl, sql) for lbl, sql in _SQLITE_DENY]
    + [("duckdb", lbl, sql) for lbl, sql in _DUCKDB_DENY],
    ids=[lbl for lbl, _ in _SQLITE_DENY] + [lbl for lbl, _ in _DUCKDB_DENY],
)
def test_network_capability_construct_is_refused(
    sql_bench, engine: str, label: str, sql: str
) -> None:
    _guard, _tmp = sql_bench
    envelope = _seam.call_envelope("query", {"endpoint": engine, "sql": sql})
    _seam.expect_refused(envelope)


# -- in-process local-file constructs: NFR-108 containment at the SQL layer


def test_duckdb_file_read_outside_allowed_paths_is_refused(sql_bench) -> None:
    """A DuckDB read-side table function is the one deny-vs-contain
    exception — permitted only inside `allowed_paths`. A path outside the
    configured root is refused (the SQL layer's tie to NFR-108)."""
    _guard, tmp = sql_bench
    outside = tmp.parent / "outside_root.csv"
    pd.DataFrame({"id": [7]}).to_csv(outside, index=False)
    try:
        envelope = _seam.call_envelope(
            "query",
            {"endpoint": "duckdb", "sql": f"SELECT * FROM read_csv_auto('{outside}')"},
        )
        _seam.expect_refused(envelope)
    finally:
        outside.unlink(missing_ok=True)


def test_duckdb_copy_to_is_refused_on_the_read_path(sql_bench) -> None:
    """The write-side local-file construct (COPY … TO) never rides the
    read `query` tool — refused before it can spill an artifact."""
    _guard, tmp = sql_bench
    target = tmp / "exfil.csv"
    envelope = _seam.call_envelope(
        "query",
        {"endpoint": "duckdb", "sql": f"COPY (SELECT 1) TO '{target}'"},
    )
    _seam.expect_refused(envelope)
    assert not target.exists()


# -- positive control: a rich legitimate read must still classify + run


@pytest.mark.parametrize("engine", ["sqlite", "duckdb"])
def test_legitimate_select_still_succeeds(sql_bench, engine: str) -> None:
    """Guards against a blanket-refuse regression that would pass every
    negative row above vacuously."""
    _guard, _tmp = sql_bench
    data = _seam.expect_ok(
        _seam.call_envelope(
            "query",
            {
                "endpoint": engine,
                "sql": (
                    "SELECT label, COUNT(*) AS n FROM t "
                    "WHERE id >= 1 GROUP BY label ORDER BY label"
                ),
            },
        )
    )
    assert [row[0] for row in data["rows"]] == ["a", "b"]


# -- networked dialects: server-side deny-outright when the fixtures exist
_NETWORKED_DENY: tuple[tuple[str, str, str | None], ...] = (
    ("postgres_copy_server_path", "COPY t TO '/srv/x.csv'", _PG_DSN),
    ("postgres_read_file", "SELECT pg_read_file('/etc/passwd')", _PG_DSN),
    ("mysql_load_file", "SELECT LOAD_FILE('/etc/passwd')", _MYSQL_DSN),
    ("mysql_into_outfile", "SELECT id INTO OUTFILE '/tmp/x' FROM t", _MYSQL_DSN),
)


@pytest.mark.parametrize(
    ("label", "sql", "dsn"),
    _NETWORKED_DENY,
    ids=[row[0] for row in _NETWORKED_DENY],
)
def test_networked_server_side_construct_is_refused(
    tmp_path: Path, label: str, sql: str, dsn: str | None
) -> None:
    if dsn is None:
        pytest.skip(f"dockerized fixture DSN absent for {label} — provisioned by E14.2")
    engine = "pg" if label.startswith("postgres") else "my"
    fixture: dict[str, tuple[str, Posture]] = {engine: (dsn, "read_write")}
    declarations = _seam.declare(**fixture)
    with _seam.booted(
        allowed_paths=(str(tmp_path),), declarations=declarations
    ) as guard:
        _seed(guard, engine)
        envelope = _seam.call_envelope("query", {"endpoint": engine, "sql": sql})
        _seam.expect_refused(envelope)
