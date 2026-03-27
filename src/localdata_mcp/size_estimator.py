"""Result size estimation engine for LocalData MCP.

Combines column type metadata with estimated row counts to predict
memory requirements before query execution.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from localdata_mcp._size_types import (  # noqa: F401 — re-exported
    BLOB_FLAG,
    COLUMN_TYPE_BYTES,
    normalize_sqlalchemy_type,
    normalize_type_string,
)

logger = logging.getLogger(__name__)


@dataclass
class ColumnSizeInfo:
    """Metadata about a single column's estimated memory footprint."""

    name: str
    type_name: str
    estimated_bytes: int
    is_blob: bool = False
    max_length: Optional[int] = None


@dataclass
class SizeEstimate:
    """Complete result size estimation."""

    estimated_rows: int
    estimated_bytes_per_row: int
    estimated_total_bytes: int
    blob_columns: List[str] = field(default_factory=list)
    confidence: str = "low"  # "high", "medium", "low"
    source: str = "heuristic"  # "explain", "heuristic", "metadata_only"


def get_type_byte_size(
    type_name: str, max_length: Optional[int] = None
) -> Union[int, str]:
    """Get estimated byte size for a column type."""
    canonical = normalize_type_string(type_name)
    size = COLUMN_TYPE_BYTES.get(canonical, 256)
    if callable(size):
        return size(max_length)
    return size


def get_column_size_info(
    name: str, type_name: str, max_length: Optional[int] = None
) -> ColumnSizeInfo:
    """Create a ColumnSizeInfo from column name and type."""
    size = get_type_byte_size(type_name, max_length)
    is_blob = size == BLOB_FLAG
    return ColumnSizeInfo(
        name=name,
        type_name=type_name,
        estimated_bytes=0 if is_blob else size,
        is_blob=is_blob,
        max_length=max_length,
    )


_FROM_PATTERN = re.compile(
    r'\bFROM\s+(["`\[]?\w+["`\]]?(?:\s*\.\s*["`\[]?\w+["`\]]?)?)',
    re.IGNORECASE,
)
_JOIN_PATTERN = re.compile(
    r'\bJOIN\s+(["`\[]?\w+["`\]]?(?:\s*\.\s*["`\[]?\w+["`\]]?)?)',
    re.IGNORECASE,
)


def extract_tables_from_query(
    query: str,
) -> List[Tuple[str, Optional[str]]]:
    """Extract table names from simple SELECT queries.

    Returns list of (table_name, schema) tuples.
    Returns empty list for complex queries (subqueries, CTEs).
    """
    upper = query.upper().strip()
    if "WITH " in upper[:20] or upper.count("SELECT") > 1:
        return []

    tables = []
    for pattern in [_FROM_PATTERN, _JOIN_PATTERN]:
        for match in pattern.finditer(query):
            raw = match.group(1).strip('"`[]')
            if "." in raw:
                schema, table = raw.rsplit(".", 1)
                tables.append((table.strip(), schema.strip()))
            else:
                tables.append((raw, None))
    return tables


class SizeEstimator:
    """Estimates query result memory footprint."""

    def __init__(self, engine: Any = None):
        self._engine = engine
        self._dialect = engine.dialect.name if engine else "unknown"
        self._column_cache: Dict[str, List[ColumnSizeInfo]] = {}

    @property
    def dialect(self) -> str:
        return self._dialect

    def _extract_table_columns(
        self, table_name: str, schema: Optional[str] = None
    ) -> List[ColumnSizeInfo]:
        """Extract column metadata via SQLAlchemy Inspector."""
        if not self._engine:
            return []
        try:
            from sqlalchemy import inspect as sa_inspect

            inspector = sa_inspect(self._engine)
            columns = inspector.get_columns(table_name, schema=schema)
            result = []
            for col in columns:
                type_obj = col.get("type")
                if type_obj is not None:
                    type_name = normalize_sqlalchemy_type(type_obj)
                    max_length = getattr(type_obj, "length", None)
                else:
                    type_name = str(col.get("type", "VARCHAR"))
                    max_length = None
                result.append(get_column_size_info(col["name"], type_name, max_length))
            return result
        except Exception as e:
            logger.debug("Table reflection failed for %s: %s", table_name, e)
            return []

    def _extract_columns_from_result(self, query: str) -> List[ColumnSizeInfo]:
        """Extract column info by executing a LIMIT 0 query."""
        if not self._engine:
            return []
        try:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                limited = f"SELECT * FROM ({query}) _sq LIMIT 0"
                result = conn.execute(text(limited))
                columns = []
                for col_name in result.keys():
                    columns.append(get_column_size_info(col_name, "VARCHAR"))
                return columns
        except Exception as e:
            logger.debug("Result proxy extraction failed: %s", e)
            return []

    def _extract_column_info(self, query: str) -> List[ColumnSizeInfo]:
        """Extract column info using best available method."""
        cache_key = query.strip()[:200]
        if cache_key in self._column_cache:
            return self._column_cache[cache_key]

        columns: List[ColumnSizeInfo] = []

        # Try table reflection first (most accurate)
        tables = extract_tables_from_query(query)
        if tables:
            for table_name, schema in tables:
                table_cols = self._extract_table_columns(table_name, schema)
                if table_cols:
                    columns.extend(table_cols)
                    break  # Use first successful table

        # Fall back to result proxy
        if not columns:
            columns = self._extract_columns_from_result(query)

        if columns:
            self._column_cache[cache_key] = columns

        return columns

    def estimate_result_size(
        self,
        query: str,
        columns: Optional[List[ColumnSizeInfo]] = None,
        estimated_rows: Optional[int] = None,
    ) -> SizeEstimate:
        """Estimate result size from column info and row estimate.

        If columns not provided, extracts via reflection or result proxy.
        If estimated_rows not provided, defaults to 1000.
        """
        if columns is None:
            columns = self._extract_column_info(query)

        if not columns:
            return SizeEstimate(
                estimated_rows=estimated_rows or 1000,
                estimated_bytes_per_row=256,
                estimated_total_bytes=(estimated_rows or 1000) * 256,
                confidence="low",
                source="heuristic",
            )

        blob_columns = [c.name for c in columns if c.is_blob]
        bytes_per_row = sum(c.estimated_bytes for c in columns if not c.is_blob)
        rows = estimated_rows or 1000

        return SizeEstimate(
            estimated_rows=rows,
            estimated_bytes_per_row=bytes_per_row,
            estimated_total_bytes=rows * bytes_per_row,
            blob_columns=blob_columns,
            confidence="medium" if estimated_rows else "low",
            source="metadata_only",
        )

    def cache_columns(self, query: str, columns: List[ColumnSizeInfo]) -> None:
        """Cache column info for a query."""
        self._column_cache[query] = columns
