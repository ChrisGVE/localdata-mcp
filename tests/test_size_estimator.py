"""Tests for the size estimation engine foundation."""

from unittest.mock import MagicMock

import pytest

from localdata_mcp.size_estimator import (
    BLOB_FLAG,
    COLUMN_TYPE_BYTES,
    ColumnSizeInfo,
    SizeEstimate,
    SizeEstimator,
    extract_tables_from_query,
    get_column_size_info,
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
