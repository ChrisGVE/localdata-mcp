"""Tests for the size estimation engine foundation."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from localdata_mcp.size_estimator import (
    BLOB_FLAG,
    COLUMN_TYPE_BYTES,
    ColumnSizeInfo,
    SizeEstimate,
    SizeEstimator,
    _estimators,
    extract_tables_from_query,
    get_column_size_info,
    get_size_estimator,
    get_type_byte_size,
    normalize_sqlalchemy_type,
    normalize_type_string,
)


class TestColumnSizeInfo:
    """Tests for the ColumnSizeInfo dataclass."""

    def test_column_size_info_creation(self):
        info = ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8)
        assert info.name == "id"
        assert info.type_name == "INTEGER"
        assert info.estimated_bytes == 8
        assert info.is_blob is False
        assert info.max_length is None

    def test_column_size_info_with_blob(self):
        info = ColumnSizeInfo(
            name="data",
            type_name="BLOB",
            estimated_bytes=0,
            is_blob=True,
            max_length=65535,
        )
        assert info.is_blob is True
        assert info.max_length == 65535


class TestSizeEstimate:
    """Tests for the SizeEstimate dataclass."""

    def test_size_estimate_defaults(self):
        estimate = SizeEstimate(
            estimated_rows=100,
            estimated_bytes_per_row=64,
            estimated_total_bytes=6400,
        )
        assert estimate.estimated_rows == 100
        assert estimate.estimated_bytes_per_row == 64
        assert estimate.estimated_total_bytes == 6400
        assert estimate.blob_columns == []
        assert estimate.confidence == "low"
        assert estimate.source == "heuristic"

    def test_size_estimate_custom_values(self):
        estimate = SizeEstimate(
            estimated_rows=1000,
            estimated_bytes_per_row=128,
            estimated_total_bytes=128000,
            blob_columns=["photo"],
            confidence="high",
            source="explain",
        )
        assert estimate.confidence == "high"
        assert estimate.source == "explain"
        assert estimate.blob_columns == ["photo"]


class TestColumnTypeBytes:
    """Tests for the COLUMN_TYPE_BYTES mapping."""

    @pytest.mark.parametrize(
        "type_name",
        ["INTEGER", "INT", "SMALLINT", "BIGINT", "TINYINT"],
    )
    def test_column_type_bytes_integers(self, type_name):
        assert COLUMN_TYPE_BYTES[type_name] == 8

    def test_column_type_bytes_boolean(self):
        assert COLUMN_TYPE_BYTES["BOOLEAN"] == 1
        assert COLUMN_TYPE_BYTES["BOOL"] == 1

    def test_column_type_bytes_varchar_with_length(self):
        func = COLUMN_TYPE_BYTES["VARCHAR"]
        assert func(100) == 100

    def test_column_type_bytes_varchar_no_length(self):
        func = COLUMN_TYPE_BYTES["VARCHAR"]
        assert func(None) == 256

    def test_column_type_bytes_varchar_large_length(self):
        func = COLUMN_TYPE_BYTES["VARCHAR"]
        assert func(1000) == 256

    def test_column_type_bytes_blob_flag(self):
        assert COLUMN_TYPE_BYTES["BLOB"] == BLOB_FLAG
        assert COLUMN_TYPE_BYTES["BINARY"] == BLOB_FLAG
        assert COLUMN_TYPE_BYTES["BYTEA"] == BLOB_FLAG
        assert COLUMN_TYPE_BYTES["IMAGE"] == BLOB_FLAG


class TestNormalizeTypeString:
    """Tests for normalize_type_string."""

    def test_normalize_type_postgres_aliases(self):
        assert normalize_type_string("int4") == "INTEGER"
        assert normalize_type_string("float8") == "DOUBLE"
        assert normalize_type_string("int2") == "SMALLINT"
        assert normalize_type_string("int8") == "BIGINT"
        assert normalize_type_string("bpchar") == "CHAR"

    def test_normalize_type_mysql_aliases(self):
        assert normalize_type_string("mediumtext") == "TEXT"
        assert normalize_type_string("longtext") == "TEXT"
        assert normalize_type_string("mediumint") == "INTEGER"

    def test_normalize_type_unknown(self):
        assert normalize_type_string("SOMETYPE") == "SOMETYPE"
        assert normalize_type_string("weird_type") == "WEIRD_TYPE"


class TestGetTypeByteSizeFunction:
    """Tests for get_type_byte_size."""

    def test_get_type_byte_size_unknown(self):
        assert get_type_byte_size("UNKNOWN_TYPE") == 256

    def test_get_type_byte_size_integer(self):
        assert get_type_byte_size("INTEGER") == 8

    def test_get_type_byte_size_varchar_with_length(self):
        assert get_type_byte_size("VARCHAR", max_length=100) == 100

    def test_get_type_byte_size_varchar_no_length(self):
        assert get_type_byte_size("VARCHAR") == 256

    def test_get_type_byte_size_blob(self):
        assert get_type_byte_size("BLOB") == BLOB_FLAG

    def test_get_type_byte_size_postgres_alias(self):
        assert get_type_byte_size("int4") == 8


class TestExtractTablesFromQuery:
    """Tests for extract_tables_from_query."""

    def test_extract_tables_simple(self):
        result = extract_tables_from_query("SELECT * FROM users")
        assert result == [("users", None)]

    def test_extract_tables_with_schema(self):
        result = extract_tables_from_query("SELECT * FROM public.users")
        assert result == [("users", "public")]

    def test_extract_tables_with_join(self):
        result = extract_tables_from_query(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        )
        assert ("users", None) in result
        assert ("orders", None) in result
        assert len(result) == 2

    def test_extract_tables_complex_returns_empty(self):
        result = extract_tables_from_query(
            "SELECT * FROM (SELECT id FROM users) AS sub"
        )
        assert result == []

    def test_extract_tables_cte_returns_empty(self):
        result = extract_tables_from_query("WITH cte AS (SELECT 1) SELECT * FROM cte")
        assert result == []


class TestNormalizeSQLAlchemyType:
    """Tests for normalize_sqlalchemy_type."""

    @staticmethod
    def _make_type(name: str):
        """Create a fake SQLAlchemy type object with the given class name."""
        cls = type(name, (), {})
        return cls()

    def test_integer_type(self):
        assert normalize_sqlalchemy_type(self._make_type("Integer")) == "INTEGER"

    def test_string_type(self):
        assert normalize_sqlalchemy_type(self._make_type("String")) == "VARCHAR"

    def test_text_type(self):
        assert normalize_sqlalchemy_type(self._make_type("Text")) == "TEXT"

    def test_boolean_type(self):
        assert normalize_sqlalchemy_type(self._make_type("Boolean")) == "BOOLEAN"

    def test_largebinary_type(self):
        assert normalize_sqlalchemy_type(self._make_type("LargeBinary")) == "BLOB"

    def test_unknown_type(self):
        assert normalize_sqlalchemy_type(self._make_type("CustomType")) == "CUSTOMTYPE"


class TestGetColumnSizeInfo:
    """Tests for get_column_size_info helper."""

    def test_varchar_column(self):
        info = get_column_size_info("name", "VARCHAR", max_length=100)
        assert info.name == "name"
        assert info.type_name == "VARCHAR"
        assert info.estimated_bytes == 100
        assert info.is_blob is False
        assert info.max_length == 100

    def test_integer_column(self):
        info = get_column_size_info("id", "INTEGER")
        assert info.estimated_bytes == 8
        assert info.is_blob is False

    def test_blob_column(self):
        info = get_column_size_info("data", "BLOB")
        assert info.estimated_bytes == 0
        assert info.is_blob is True

    def test_unknown_type_defaults_256(self):
        info = get_column_size_info("mystery", "UNKNOWN_TYPE")
        assert info.estimated_bytes == 256
        assert info.is_blob is False


class TestSizeEstimator:
    """Tests for the SizeEstimator class."""

    def _make_engine(self, dialect_name: str = "sqlite") -> MagicMock:
        engine = MagicMock()
        engine.dialect.name = dialect_name
        return engine

    def test_init_without_engine(self):
        est = SizeEstimator()
        assert est.dialect == "unknown"

    def test_init_with_engine(self):
        est = SizeEstimator(engine=self._make_engine("sqlite"))
        assert est.dialect == "sqlite"

    def test_estimate_no_columns_heuristic(self):
        est = SizeEstimator()
        result = est.estimate_result_size("SELECT * FROM t")
        assert result.estimated_rows == 1000
        assert result.estimated_bytes_per_row == 256
        assert result.estimated_total_bytes == 1000 * 256
        assert result.confidence == "low"
        assert result.source == "heuristic"

    def test_estimate_with_columns(self):
        est = SizeEstimator()
        cols = [
            ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
            ColumnSizeInfo(name="name", type_name="VARCHAR", estimated_bytes=100),
        ]
        result = est.estimate_result_size("SELECT * FROM t", columns=cols)
        assert result.estimated_bytes_per_row == 108
        assert result.estimated_total_bytes == 1000 * 108

    def test_estimate_with_blob_columns(self):
        est = SizeEstimator()
        cols = [
            ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
            ColumnSizeInfo(
                name="photo", type_name="BLOB", estimated_bytes=0, is_blob=True
            ),
        ]
        result = est.estimate_result_size("SELECT * FROM t", columns=cols)
        assert result.blob_columns == ["photo"]
        assert result.estimated_bytes_per_row == 8

    def test_estimate_with_row_count(self):
        est = SizeEstimator()
        cols = [
            ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
        ]
        result = est.estimate_result_size(
            "SELECT * FROM t", columns=cols, estimated_rows=500
        )
        assert result.estimated_rows == 500
        assert result.estimated_total_bytes == 500 * 8

    def test_cache_columns(self):
        est = SizeEstimator()
        cols = [
            ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
        ]
        query = "SELECT id FROM t"
        est.cache_columns(query, cols)
        result = est.estimate_result_size(query)
        assert result.estimated_bytes_per_row == 8
        assert result.source == "metadata_only"

    def test_confidence_with_rows(self):
        est = SizeEstimator()
        cols = [
            ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
        ]
        result = est.estimate_result_size(
            "SELECT * FROM t", columns=cols, estimated_rows=200
        )
        assert result.confidence == "medium"

    def test_confidence_without_rows(self):
        est = SizeEstimator()
        cols = [
            ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
        ]
        result = est.estimate_result_size("SELECT * FROM t", columns=cols)
        assert result.confidence == "low"


class TestExtractTableColumns:
    """Tests for SizeEstimator._extract_table_columns."""

    def _make_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.dialect.name = "sqlite"
        return engine

    def test_extract_columns_from_reflection(self):
        """Mock Inspector returning column defs produces ColumnSizeInfo list."""
        engine = self._make_engine()
        type_obj = MagicMock()
        type_obj.length = None

        mock_inspector = MagicMock()
        mock_inspector.get_columns.return_value = [
            {"name": "id", "type": type_obj},
            {"name": "name", "type": type_obj},
        ]

        est = SizeEstimator(engine=engine)
        with (
            patch("sqlalchemy.inspect", return_value=mock_inspector),
            patch(
                "localdata_mcp.size_estimator.normalize_sqlalchemy_type",
                return_value="INTEGER",
            ),
        ):
            result = est._extract_table_columns("users")

        assert len(result) == 2
        assert result[0].name == "id"
        assert result[1].name == "name"

    def test_extract_columns_no_engine(self):
        """Returns empty list when no engine is set."""
        est = SizeEstimator()
        result = est._extract_table_columns("users")
        assert result == []

    def test_extract_columns_reflection_failure(self):
        """Returns empty list when Inspector raises an exception."""
        engine = self._make_engine()
        est = SizeEstimator(engine=engine)

        with patch(
            "sqlalchemy.inspect",
            side_effect=RuntimeError("reflection failed"),
        ):
            result = est._extract_table_columns("users")
        assert result == []


class TestExtractColumnsFromResult:
    """Tests for SizeEstimator._extract_columns_from_result."""

    def _make_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.dialect.name = "sqlite"
        return engine

    def test_extract_from_result_proxy(self):
        """Mock engine + connection returning column keys."""
        engine = self._make_engine()
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name", "email"]

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        engine.connect.return_value = mock_conn

        est = SizeEstimator(engine=engine)
        result = est._extract_columns_from_result("SELECT * FROM users")

        assert len(result) == 3
        assert result[0].name == "id"
        assert result[1].name == "name"
        assert result[2].name == "email"
        for col in result:
            assert col.type_name == "VARCHAR"

    def test_extract_from_result_no_engine(self):
        """Returns empty list when no engine is set."""
        est = SizeEstimator()
        result = est._extract_columns_from_result("SELECT * FROM t")
        assert result == []


class TestExtractColumnInfo:
    """Tests for SizeEstimator._extract_column_info."""

    def _make_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.dialect.name = "sqlite"
        return engine

    def test_uses_table_reflection_first(self):
        """When table reflection succeeds, result proxy is not called."""
        engine = self._make_engine()
        est = SizeEstimator(engine=engine)

        reflection_cols = [
            ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
        ]
        est._extract_table_columns = MagicMock(return_value=reflection_cols)
        est._extract_columns_from_result = MagicMock(return_value=[])

        result = est._extract_column_info("SELECT * FROM users")

        est._extract_table_columns.assert_called_once_with("users", None)
        est._extract_columns_from_result.assert_not_called()
        assert result == reflection_cols

    def test_falls_back_to_result_proxy(self):
        """When reflection returns empty, falls back to result proxy."""
        engine = self._make_engine()
        est = SizeEstimator(engine=engine)

        proxy_cols = [
            ColumnSizeInfo(name="col1", type_name="VARCHAR", estimated_bytes=256),
        ]
        est._extract_table_columns = MagicMock(return_value=[])
        est._extract_columns_from_result = MagicMock(return_value=proxy_cols)

        result = est._extract_column_info("SELECT * FROM users")

        est._extract_table_columns.assert_called_once()
        est._extract_columns_from_result.assert_called_once()
        assert result == proxy_cols

    def test_caches_results(self):
        """Second call with same query uses cache, no re-extraction."""
        engine = self._make_engine()
        est = SizeEstimator(engine=engine)

        reflection_cols = [
            ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
        ]
        est._extract_table_columns = MagicMock(return_value=reflection_cols)
        est._extract_columns_from_result = MagicMock(return_value=[])

        # First call populates cache
        result1 = est._extract_column_info("SELECT * FROM users")
        # Second call should use cache
        result2 = est._extract_column_info("SELECT * FROM users")

        assert est._extract_table_columns.call_count == 1
        assert result1 == result2

    def test_empty_for_complex_query(self):
        """Subquery returns empty tables, falls through to result proxy."""
        est = SizeEstimator()  # no engine
        result = est._extract_column_info("SELECT * FROM (SELECT id FROM users) AS sub")
        assert result == []


class TestEstimateWithExtraction:
    """Tests for estimate_result_size using _extract_column_info."""

    def test_estimate_uses_extract_when_no_columns(self):
        """Verify _extract_column_info is called when columns not provided."""
        engine = MagicMock()
        engine.dialect.name = "sqlite"
        est = SizeEstimator(engine=engine)

        extracted_cols = [
            ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
            ColumnSizeInfo(name="val", type_name="FLOAT", estimated_bytes=8),
        ]
        est._extract_column_info = MagicMock(return_value=extracted_cols)

        result = est.estimate_result_size("SELECT * FROM t")

        est._extract_column_info.assert_called_once_with("SELECT * FROM t")
        assert result.estimated_bytes_per_row == 16
        assert result.source == "metadata_only"


class TestExplainParsers:
    """Tests for EXPLAIN parsers and the _run_explain dispatcher."""

    @staticmethod
    def _mock_engine(dialect: str = "sqlite") -> MagicMock:
        engine = MagicMock()
        engine.dialect.name = dialect
        return engine

    @staticmethod
    def _mock_conn_with_rows(rows, engine):
        """Wire up engine.connect() context manager to return rows."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = rows
        mock_result.scalar.return_value = rows[0] if rows else None
        mock_result.keys.return_value = ["id", "select_type", "table", "type", "rows"]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = mock_conn
        return mock_result

    def test_sqlite_explain_with_index(self):
        from localdata_mcp._explain_parsers import parse_explain_sqlite

        engine = self._mock_engine()
        rows = [("0", "0", "0", "SEARCH users USING INDEX idx_name")]
        self._mock_conn_with_rows(rows, engine)
        result = parse_explain_sqlite(engine, "SELECT * FROM users WHERE name='a'")
        assert result is not None
        assert result["scan_type"] == "index"
        assert result["confidence"] == 0.5

    def test_sqlite_explain_scan(self):
        from localdata_mcp._explain_parsers import parse_explain_sqlite

        engine = self._mock_engine()
        rows = [("0", "0", "0", "SCAN users")]
        self._mock_conn_with_rows(rows, engine)
        result = parse_explain_sqlite(engine, "SELECT * FROM users")
        assert result is not None
        assert result["scan_type"] == "scan"
        assert result["confidence"] == 0.3

    def test_sqlite_explain_failure(self):
        from localdata_mcp._explain_parsers import parse_explain_sqlite

        engine = self._mock_engine()
        engine.connect.side_effect = RuntimeError("fail")
        result = parse_explain_sqlite(engine, "SELECT 1")
        assert result is None

    def test_postgresql_explain_json(self):
        from localdata_mcp._explain_parsers import parse_explain_postgresql

        engine = self._mock_engine("postgresql")
        import json

        plan_json = json.dumps(
            [{"Plan": {"Node Type": "Seq Scan", "Plan Rows": 500, "Total Cost": 10.5}}]
        )
        mock_result = MagicMock()
        mock_result.scalar.return_value = plan_json
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = mock_conn
        result = parse_explain_postgresql(engine, "SELECT * FROM users")
        assert result is not None
        assert result["estimated_rows"] == 500
        assert result["confidence"] == 0.7
        assert result["scan_type"] == "Seq Scan"

    def test_mysql_explain_rows(self):
        from localdata_mcp._explain_parsers import parse_explain_mysql

        engine = self._mock_engine("mysql")
        mock_row = MagicMock()
        mock_row._mapping = {"rows": 100, "type": "ref"}
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_result.keys.return_value = ["type", "rows"]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = mock_conn
        result = parse_explain_mysql(engine, "SELECT * FROM users")
        assert result is not None
        assert result["estimated_rows"] == 100
        assert result["confidence"] == 0.7
        assert result["scan_type"] == "ref"

    def test_run_explain_sqlite_dialect(self):
        engine = self._mock_engine("sqlite")
        rows = [("0", "0", "0", "SCAN users")]
        self._mock_conn_with_rows(rows, engine)
        est = SizeEstimator(engine=engine)
        result = est._run_explain("SELECT * FROM users")
        assert result is not None
        assert result["scan_type"] == "scan"

    def test_run_explain_unknown_dialect(self):
        engine = self._mock_engine("duckdb")
        est = SizeEstimator(engine=engine)
        result = est._run_explain("SELECT * FROM users")
        assert result is None


class TestHeuristics:
    """Tests for _estimate_rows_heuristic."""

    @staticmethod
    def _mock_engine(dialect: str = "sqlite") -> MagicMock:
        engine = MagicMock()
        engine.dialect.name = dialect
        return engine

    def test_heuristic_with_count(self):
        engine = self._mock_engine()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5000
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = mock_conn
        est = SizeEstimator(engine=engine)
        rows, conf = est._estimate_rows_heuristic("SELECT * FROM users")
        assert rows == 5000
        assert conf == "low"

    def test_heuristic_with_where(self):
        engine = self._mock_engine()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5000
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = mock_conn
        est = SizeEstimator(engine=engine)
        rows, conf = est._estimate_rows_heuristic("SELECT * FROM users WHERE active=1")
        assert rows == 500  # 10% of 5000
        assert conf == "low"

    def test_heuristic_default(self):
        est = SizeEstimator()  # no engine
        rows, conf = est._estimate_rows_heuristic("SELECT * FROM users")
        assert rows == 1000
        assert conf == "low"


class TestSampleEstimation:
    """Tests for estimate_from_sample."""

    def test_estimate_from_sample(self):
        est = SizeEstimator()
        sample = [[1, "hello", 3.14], [2, "world", 2.72]]
        cols = ["id", "name", "val"]
        result = est.estimate_from_sample(sample, cols, total_rows=1000)
        assert result.estimated_rows == 1000
        assert result.confidence == "high"
        assert result.source == "metadata_only"
        expected_bytes = sum(sys.getsizeof(c) for row in sample for c in row)
        expected_bpr = expected_bytes // 2
        assert result.estimated_bytes_per_row == expected_bpr
        assert result.estimated_total_bytes == 1000 * expected_bpr

    def test_estimate_from_empty_sample(self):
        est = SizeEstimator()
        result = est.estimate_from_sample([], ["id"], total_rows=500)
        assert result.estimated_rows == 500
        assert result.estimated_bytes_per_row == 256
        assert result.estimated_total_bytes == 500 * 256
        assert result.confidence == "low"
        assert result.source == "heuristic"


class TestGetSizeEstimatorFactory:
    """Tests for the get_size_estimator singleton factory."""

    def setup_method(self):
        """Clear the global cache before each test."""
        _estimators.clear()

    def teardown_method(self):
        _estimators.clear()

    @staticmethod
    def _make_engine(url: str = "sqlite:///test.db", dialect: str = "sqlite"):
        engine = MagicMock()
        engine.dialect.name = dialect
        engine.url = url
        return engine

    def test_get_size_estimator_caching(self):
        """Same engine URL returns the same SizeEstimator instance."""
        engine = self._make_engine("sqlite:///a.db")
        est1 = get_size_estimator(engine)
        est2 = get_size_estimator(engine)
        assert est1 is est2

    def test_get_size_estimator_different_engines(self):
        """Different engine URLs produce different instances."""
        eng_a = self._make_engine("sqlite:///a.db")
        eng_b = self._make_engine("sqlite:///b.db")
        est_a = get_size_estimator(eng_a)
        est_b = get_size_estimator(eng_b)
        assert est_a is not est_b

    def test_get_size_estimator_no_engine(self):
        """No engine returns a fresh SizeEstimator with unknown dialect."""
        est = get_size_estimator()
        assert est.dialect == "unknown"
        # Each call without engine should return a new instance
        est2 = get_size_estimator()
        assert est is not est2


class TestFullEstimateWithExplain:
    """Test estimate_result_size with mocked EXPLAIN support."""

    @staticmethod
    def _make_engine():
        engine = MagicMock()
        engine.dialect.name = "sqlite"
        engine.url = "sqlite:///test.db"
        return engine

    def test_full_estimate_with_explain(self):
        """When EXPLAIN succeeds, confidence should be high or medium."""
        engine = self._make_engine()
        est = SizeEstimator(engine=engine)

        explain_result = {
            "estimated_rows": 200,
            "confidence": 0.8,
            "scan_type": "index",
        }
        est._run_explain = MagicMock(return_value=explain_result)
        est._extract_column_info = MagicMock(
            return_value=[
                ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
            ]
        )

        result = est.estimate_result_size("SELECT * FROM t", use_explain=True)
        assert result.estimated_rows == 200
        assert result.confidence == "high"
        assert result.estimated_bytes_per_row == 8
        est._run_explain.assert_called_once()

    def test_full_estimate_explain_fails_falls_back(self):
        """When EXPLAIN returns None, heuristic is used instead."""
        engine = self._make_engine()
        est = SizeEstimator(engine=engine)

        est._run_explain = MagicMock(return_value=None)
        est._extract_column_info = MagicMock(
            return_value=[
                ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
            ]
        )
        est._estimate_rows_heuristic = MagicMock(return_value=(3000, "low"))

        result = est.estimate_result_size("SELECT * FROM t", use_explain=True)
        assert result.estimated_rows == 3000
        assert result.confidence == "low"
        est._estimate_rows_heuristic.assert_called_once()

    def test_estimate_result_size_with_blobs(self):
        """Blob columns are excluded from byte calculation."""
        est = SizeEstimator()
        cols = [
            ColumnSizeInfo(name="id", type_name="INTEGER", estimated_bytes=8),
            ColumnSizeInfo(
                name="avatar", type_name="BLOB", estimated_bytes=0, is_blob=True
            ),
            ColumnSizeInfo(name="name", type_name="VARCHAR", estimated_bytes=100),
        ]
        result = est.estimate_result_size(
            "SELECT * FROM t", columns=cols, estimated_rows=10
        )
        assert result.blob_columns == ["avatar"]
        assert result.estimated_bytes_per_row == 108  # 8 + 100, blob excluded
        assert result.estimated_total_bytes == 10 * 108


class TestStreamingIntegrationMock:
    """Verify SizeEstimator is wired into StreamingQueryExecutor."""

    def test_streaming_executor_calls_size_estimator(self):
        """Verify size estimation runs when a SQL source is used."""
        from localdata_mcp.streaming_executor import StreamingSQLSource

        engine = MagicMock()
        engine.dialect.name = "sqlite"
        engine.url = "sqlite:///test.db"
        source = StreamingSQLSource(engine, "SELECT * FROM t")

        fake_estimate = SizeEstimate(
            estimated_rows=500,
            estimated_bytes_per_row=64,
            estimated_total_bytes=32000,
            confidence="medium",
            source="heuristic",
        )
        with patch("localdata_mcp.streaming_executor.get_size_estimator") as mock_get:
            mock_estimator = MagicMock()
            mock_estimator.estimate_result_size.return_value = fake_estimate
            mock_get.return_value = mock_estimator

            # Verify the factory and estimator are properly wired
            est = mock_get(engine)
            result = est.estimate_result_size("SELECT * FROM t")
            assert result.estimated_rows == 500
            mock_get.assert_called_once_with(engine)
