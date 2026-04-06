"""Tests for BLOB column detection and placeholder handling."""

import base64

import pytest

from localdata_mcp.blob_handler import (
    BlobPlaceholder,
    detect_mime_type,
    is_blob_type,
    mark_blob_columns,
    process_blob_columns,
)

# --- is_blob_type tests ---


def test_is_blob_type_blob():
    assert is_blob_type("BLOB") is True


def test_is_blob_type_bytea():
    assert is_blob_type("BYTEA") is True


def test_is_blob_type_image():
    assert is_blob_type("IMAGE") is True


def test_is_blob_type_varchar():
    assert is_blob_type("VARCHAR") is False


def test_is_blob_type_integer():
    assert is_blob_type("INTEGER") is False


# --- BlobPlaceholder tests ---


def test_blob_placeholder_bytes():
    p = BlobPlaceholder(size_bytes=500)
    assert str(p) == "[BLOB: 500B, unknown]"


def test_blob_placeholder_kb():
    p = BlobPlaceholder(size_bytes=int(1.5 * 1024), mime_type="image/png")
    assert str(p) == "[BLOB: 1.5KB, image/png]"


def test_blob_placeholder_mb():
    p = BlobPlaceholder(size_bytes=int(3.2 * 1024 * 1024), mime_type="application/pdf")
    assert str(p) == "[BLOB: 3.2MB, application/pdf]"


# --- detect_mime_type tests ---


def test_detect_mime_png():
    data = b"\x89PNG" + b"\x00" * 100
    assert detect_mime_type(data) == "image/png"


def test_detect_mime_pdf():
    data = b"%PDF-1.4" + b"\x00" * 100
    assert detect_mime_type(data) == "application/pdf"


def test_detect_mime_unknown():
    data = b"\x00\x01\x02\x03\x04\x05\x06\x07"
    assert detect_mime_type(data) is None


def test_detect_mime_too_short():
    data = b"\x89P"
    assert detect_mime_type(data) is None


# --- mark_blob_columns tests ---


def test_mark_blob_columns():
    schema = {
        "columns": [
            {"name": "id", "type": "INTEGER"},
            {"name": "data", "type": "BLOB"},
            {"name": "name", "type": "VARCHAR"},
        ]
    }
    result = mark_blob_columns(schema)
    assert result["columns"][0]["is_blob"] is False
    assert result["columns"][1]["is_blob"] is True
    assert result["columns"][2]["is_blob"] is False


# --- process_blob_columns tests ---


def test_process_blob_columns_placeholder():
    png_header = b"\x89PNG" + b"\x00" * 96
    rows = [{"id": 1, "photo": png_header}]
    processed, warnings = process_blob_columns(rows, ["photo"])
    assert processed[0]["photo"] == "[BLOB: 100B, image/png]"


def test_process_blob_columns_base64():
    small_blob = b"\x89PNG" + b"\x00" * 10
    rows = [{"id": 1, "photo": small_blob}]
    processed, warnings = process_blob_columns(rows, ["photo"], include_blobs=True)
    expected = base64.b64encode(small_blob).decode("ascii")
    assert processed[0]["photo"] == expected


def test_process_blob_columns_large_blob():
    large_blob = b"\x00" * (6 * 1024 * 1024)  # 6MB, exceeds 5MB default
    rows = [{"id": 1, "data": large_blob}]
    processed, warnings = process_blob_columns(rows, ["data"], include_blobs=True)
    assert processed[0]["data"].startswith("[BLOB:")


def test_process_blob_columns_none_value():
    rows = [{"id": 1, "photo": None}]
    processed, warnings = process_blob_columns(rows, ["photo"])
    assert processed[0]["photo"] is None


def test_process_blob_columns_warnings():
    rows = [{"id": 1, "photo": b"\x00" * 10}]
    _, warnings = process_blob_columns(rows, ["photo"], include_blobs=True)
    assert len(warnings) == 1
    assert "base64" in warnings[0]
