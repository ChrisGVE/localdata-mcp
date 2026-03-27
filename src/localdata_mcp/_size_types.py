"""Column type mappings and normalization for size estimation.

Maps database column types to estimated byte sizes for memory
footprint prediction.
"""

from typing import Any, Callable, Dict, Union

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

_SA_TYPE_MAP: Dict[str, str] = {
    "INTEGER": "INTEGER",
    "SMALLINTEGER": "SMALLINT",
    "BIGINTEGER": "BIGINT",
    "FLOAT": "FLOAT",
    "NUMERIC": "NUMERIC",
    "BOOLEAN": "BOOLEAN",
    "STRING": "VARCHAR",
    "TEXT": "TEXT",
    "UNICODE": "VARCHAR",
    "UNICODETEXT": "TEXT",
    "DATE": "DATE",
    "DATETIME": "DATETIME",
    "TIME": "TIME",
    "LARGEBINARY": "BLOB",
    "BLOB": "BLOB",
    "NVARCHAR": "NVARCHAR",
    "NSTRING": "NVARCHAR",
}


def normalize_type_string(type_str: str) -> str:
    """Normalize a database type string to canonical form."""
    base = type_str.split("(")[0].strip().upper()
    return _TYPE_ALIASES.get(type_str.lower(), _TYPE_ALIASES.get(base.lower(), base))


def normalize_sqlalchemy_type(type_obj: Any) -> str:
    """Normalize a SQLAlchemy TypeEngine to canonical type name."""
    type_name = type(type_obj).__name__.upper()
    return _SA_TYPE_MAP.get(type_name, type_name)
