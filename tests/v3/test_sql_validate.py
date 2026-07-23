"""tests/v3/test_sql_validate.py — E6.2/NFR-104 allow-list battery.

The construct-level allow-list per dialect: legitimate reads classify,
mutations classify with their nodes recorded, local-file constructs
classify with paths extracted, and the whole regex-denylist-bypass +
file-reach/side-effect payload class is refused. The dialect mapping's
every-dialect parse_one assertion (S4.1) and the fail-safe dispositions
(parse failure, multi-statement, unknown construct) are covered here;
posture and containment are guard.py's per-call concern (tested there).
"""

from __future__ import annotations

import sqlglot
import pytest

from localdata_mcp.nexus.chokepoint.sql_validate import (
    BACKEND_TO_SQLGLOT_DIALECT,
    POLICIES,
    classify,
)
from localdata_mcp.nexus.chokepoint.sql_validate.walker import SqlRefusedError

CORE_DIALECTS = ["sqlite", "postgresql", "mysql", "duckdb"]
ALL_DIALECTS = CORE_DIALECTS + ["mssql", "oracle"]


class TestDialectMapping:
    """S4.1: the declared backend_kind → sqlglot mapping is real at the
    pinned sqlglot version — a bare parse_one succeeds for every key."""

    @pytest.mark.parametrize("backend_kind", ALL_DIALECTS)
    def test_every_mapped_dialect_parses(self, backend_kind: str) -> None:
        dialect = BACKEND_TO_SQLGLOT_DIALECT[backend_kind]
        parsed = sqlglot.parse_one("SELECT 1", dialect=dialect)
        assert parsed is not None

    def test_the_two_renames_are_declared(self) -> None:
        assert BACKEND_TO_SQLGLOT_DIALECT["postgresql"] == "postgres"
        assert BACKEND_TO_SQLGLOT_DIALECT["mssql"] == "tsql"

    def test_the_unmapped_names_would_have_failed(self) -> None:
        # The reason the mapping exists: the backend_kind strings raise.
        for bad in ("postgresql", "mssql"):
            with pytest.raises(Exception):
                sqlglot.parse_one("SELECT 1", dialect=bad)

    def test_a_policy_exists_for_every_mapped_dialect(self) -> None:
        assert set(POLICIES) == set(BACKEND_TO_SQLGLOT_DIALECT)


class TestLegitimateReads:
    @pytest.mark.parametrize("backend_kind", ALL_DIALECTS)
    def test_rich_select_classifies_as_query(self, backend_kind: str) -> None:
        sql = (
            "SELECT a, COUNT(*), SUM(b) FROM t "
            "WHERE a > 1 AND c IN (1, 2) GROUP BY a HAVING COUNT(*) > 1 "
            "ORDER BY a LIMIT 10"
        )
        result = classify(sql, backend_kind)
        assert result.category == "query"
        assert not result.contains_mutation_nodes

    def test_cte_and_set_ops_classify_as_query(self) -> None:
        result = classify(
            "WITH c AS (SELECT a FROM t) SELECT * FROM c UNION SELECT a FROM u",
            "sqlite",
        )
        assert result.category == "query"


class TestMutations:
    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO t (a) VALUES (1)",
            "UPDATE t SET a = 1 WHERE b = 2",
            "DELETE FROM t WHERE a = 1",
        ],
    )
    def test_dml_classifies_as_mutation(self, sql: str) -> None:
        result = classify(sql, "sqlite")
        assert result.category == "mutation"
        assert result.contains_mutation_nodes

    def test_data_modifying_cte_is_flagged_on_the_query_root(self) -> None:
        # NFR-104/113: parses as WITH/SELECT but carries a Delete node.
        result = classify(
            "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
            "postgresql",
        )
        assert result.category == "query"
        assert result.contains_mutation_nodes  # guard.py refuses on read path


class TestLocalFileConstructs:
    def test_duckdb_read_csv_auto_extracts_its_path(self) -> None:
        result = classify("SELECT * FROM read_csv_auto('/data/x.csv')", "duckdb")
        assert result.category == "local_file_read"
        assert result.path_literals == ("/data/x.csv",)

    def test_duckdb_copy_to_is_write_side_with_its_path(self) -> None:
        result = classify("COPY (SELECT 1) TO '/out/dump.csv'", "duckdb")
        assert result.category == "local_file_write"
        assert result.path_literals == ("/out/dump.csv",)

    def test_sqlite_attach_is_write_side_with_its_path(self) -> None:
        result = classify("ATTACH DATABASE '/data/side.db' AS s", "sqlite")
        assert result.category == "local_file_write"
        assert result.path_literals == ("/data/side.db",)


class TestDenyOutright:
    @pytest.mark.parametrize(
        "backend_kind,sql",
        [
            ("duckdb", "INSTALL httpfs"),
            ("duckdb", "LOAD httpfs"),
            ("sqlite", "SELECT load_extension('/e.so')"),
            ("postgresql", "COPY t TO '/srv/x'"),  # server-side path
            ("postgresql", "SELECT pg_read_file('x')"),
            ("mysql", "SELECT LOAD_FILE('/etc/passwd')"),
            ("mssql", "SELECT * FROM OPENROWSET('a', 'b', 'c')"),
            ("oracle", "SELECT BFILENAME('D', 'f') FROM DUAL"),
        ],
    )
    def test_network_capability_construct_is_refused(
        self, backend_kind: str, sql: str
    ) -> None:
        with pytest.raises(SqlRefusedError):
            classify(sql, backend_kind)


class TestFailSafeDispositions:
    def test_parse_failure_refuses(self) -> None:
        with pytest.raises(SqlRefusedError, match="parse"):
            classify("SELECT * FROM", "sqlite")

    def test_lenient_garbage_is_refused_by_category(self) -> None:
        # sqlglot parses some garbage leniently (e.g. as an Alias); the
        # allow-list still refuses it — never a vacuous pass.
        with pytest.raises(SqlRefusedError):
            classify("NOT VALID SQL", "sqlite")

    def test_multi_statement_refuses(self) -> None:
        with pytest.raises(SqlRefusedError, match="one statement"):
            classify("SELECT 1; SELECT 2", "sqlite")

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE t",
            "CREATE TABLE t (x INTEGER)",
            "ALTER TABLE t ADD COLUMN y INTEGER",
            "TRUNCATE TABLE t",
            "GRANT SELECT ON t TO u",
        ],
    )
    def test_ddl_and_control_are_refused_by_absence(self, sql: str) -> None:
        with pytest.raises(SqlRefusedError):
            classify(sql, "sqlite")

    def test_unknown_backend_is_refused(self) -> None:
        with pytest.raises(SqlRefusedError, match="no validation policy"):
            classify("SELECT 1", "cassandra")


class TestPolicyIsPureData:
    def test_no_dialect_fragment_carries_control_flow(self) -> None:
        # Each fragment exports exactly one POLICY dataclass instance.
        from localdata_mcp.nexus.chokepoint.sql_validate.dialects import (
            duckdb,
            mssql,
            mysql,
            oracle,
            postgresql,
            sqlite,
        )

        for fragment in (sqlite, postgresql, mysql, duckdb, mssql, oracle):
            assert fragment.POLICY.backend_kind in BACKEND_TO_SQLGLOT_DIALECT
