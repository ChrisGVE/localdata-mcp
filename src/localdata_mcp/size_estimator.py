"""Result size estimation engine for LocalData MCP.

Combines column type metadata with estimated row counts to predict
memory requirements before query execution.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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


BLOB_FLAG = "BLOB_FLAG"

COLUMN_TYPE_BYTES: Dict[str, Union[int, str, Callable]] = {
    # Numeric
    "INTEGER": 8,
    "INT": 8,
    "SMALLINT": 8,
    "BIGINT": 8,
    "TINYINT": 8,
    "FLOAT": 8,
    "DOUBLE": 8,
    "DECIMAL": 8,
    "NUMERIC": 8,
    "REAL": 8,
    "BOOLEAN": 1,
    "BOOL": 1,
    # Date/time
    "DATE": 8,
    "TIME": 8,
    "TIMESTAMP": 8,
    "DATETIME": 8,
    "TIMESTAMPTZ": 8,
    "INTERVAL": 8,
    # String (variable length uses lambda)
    "VARCHAR": lambda n: min(n, 256) if n else 256,
    "CHAR": lambda n: min(n, 256) if n else 256,
    "NVARCHAR": lambda n: min(n * 2, 512) if n else 256,
    "TEXT": 256,
    "CLOB": 256,
    "NCLOB": 256,
    "NTEXT": 256,
    "STRING": 256,
    # Binary (BLOB sentinel)
    "BLOB": BLOB_FLAG,
    "BINARY": BLOB_FLAG,
    "VARBINARY": BLOB_FLAG,
    "BYTEA": BLOB_FLAG,
    "IMAGE": BLOB_FLAG,
    "RAW": BLOB_FLAG,
    "LONGBLOB": BLOB_FLAG,
    "MEDIUMBLOB": BLOB_FLAG,
}

# Database-specific type aliases
_TYPE_ALIASES: Dict[str, str] = {
    # PostgreSQL
    "int2": "SMALLINT",
    "int4": "INTEGER",
    "int8": "BIGINT",
    "float4": "REAL",
    "float8": "DOUBLE",
    "serial": "INTEGER",
    "bigserial": "BIGINT",
    "varchar": "VARCHAR",
    "bpchar": "CHAR",
    "timestamptz": "TIMESTAMP",
    "timetz": "TIME",
    # MySQL
    "mediumtext": "TEXT",
    "longtext": "TEXT",
    "mediumint": "INTEGER",
    "tinytext": "TEXT",
    "enum": "VARCHAR",
    "set": "VARCHAR",
    # SQLite
    "integer": "INTEGER",
    "real": "REAL",
    # DuckDB
    "hugeint": "BIGINT",
    "utinyint": "TINYINT",
    "usmallint": "SMALLINT",
    "uinteger": "INTEGER",
    "ubigint": "BIGINT",
}


def normalize_type_string(type_str: str) -> str:
    """Normalize a database type string to canonical form."""
    base = type_str.split("(")[0].strip().upper()
    return _TYPE_ALIASES.get(type_str.lower(), _TYPE_ALIASES.get(base.lower(), base))


def get_type_byte_size(
    type_name: str, max_length: Optional[int] = None
) -> Union[int, str]:
    """Get estimated byte size for a column type."""
    canonical = normalize_type_string(type_name)
    size = COLUMN_TYPE_BYTES.get(canonical, 256)
    if callable(size):
        return size(max_length)
    return size


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
    # Skip complex queries
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
