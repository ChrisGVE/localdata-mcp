"""BLOB column detection and placeholder handling for LocalData MCP."""

import base64
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BLOB_TYPE_NAMES = {
    "BLOB",
    "BINARY",
    "VARBINARY",
    "LARGEBINARY",
    "BYTEA",  # PostgreSQL
    "RAW",
    "LONG RAW",  # Oracle
    "IMAGE",  # MS SQL (legacy)
    "MEDIUMBLOB",
    "LONGBLOB",
    "TINYBLOB",  # MySQL
}

# MIME detection without python-magic dependency
_MAGIC_NUMBERS = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/zip",
    b"\x1f\x8b": "application/gzip",
}


def is_blob_type(column_type) -> bool:
    """Check if a column type is a BLOB/binary type."""
    type_name = str(column_type).upper()
    for blob_name in BLOB_TYPE_NAMES:
        if blob_name in type_name:
            return True
    try:
        from sqlalchemy.types import LargeBinary

        if isinstance(column_type, LargeBinary):
            return True
    except ImportError:
        pass
    return False


@dataclass
class BlobPlaceholder:
    """Informative placeholder for BLOB data."""

    size_bytes: int
    mime_type: Optional[str] = None

    def __str__(self) -> str:
        size_str = self._format_size(self.size_bytes)
        mime_str = self.mime_type or "unknown"
        return f"[BLOB: {size_str}, {mime_str}]"

    def _format_size(self, size: int) -> str:
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f}KB"
        return f"{size / (1024 * 1024):.1f}MB"


def detect_mime_type(data: bytes) -> Optional[str]:
    """Detect MIME type from binary header bytes."""
    if len(data) < 4:
        return None
    for magic, mime in _MAGIC_NUMBERS.items():
        if data[: len(magic)] == magic:
            return mime
    return None


def mark_blob_columns(table_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add 'is_blob' flag to column metadata."""
    for col in table_schema.get("columns", []):
        col["is_blob"] = is_blob_type(col.get("type", ""))
    return table_schema


def process_blob_columns(
    rows: List[Dict[str, Any]],
    blob_columns: List[str],
    include_blobs: bool = False,
    max_blob_size_mb: int = 5,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Process BLOB columns in query results.

    Replaces BLOB data with placeholders or base64 encodes small BLOBs.
    Returns (processed_rows, warnings).
    """
    max_bytes = max_blob_size_mb * 1024 * 1024
    warnings: List[str] = []
    processed = []

    for row in rows:
        new_row = dict(row)
        for col in blob_columns:
            if col not in new_row:
                continue
            value = new_row[col]
            if value is None:
                continue
            if isinstance(value, (bytes, bytearray)):
                size = len(value)
                mime = detect_mime_type(value)
                if include_blobs and size <= max_bytes:
                    new_row[col] = base64.b64encode(value).decode("ascii")
                else:
                    new_row[col] = str(BlobPlaceholder(size, mime))
        processed.append(new_row)

    if include_blobs and blob_columns:
        warnings.append("BLOB columns included as base64. Response size may be large.")

    return processed, warnings
