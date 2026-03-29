"""Database setup helpers for integration tests."""

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from .data_generator import TestDataGenerator


def create_sqlite_test_db(rows: int = 1000, path: str = None) -> str:
    """Create a SQLite database with test data. Returns file path."""
    from sqlalchemy import create_engine, text

    if path is None:
        fd, path = tempfile.mkstemp(suffix=".db", prefix="localdata_test_")
        os.close(fd)
    engine = create_engine(f"sqlite:///{path}")
    gen = TestDataGenerator()
    data = gen.generate_standard_rows(rows)

    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS test_data (
                id INTEGER PRIMARY KEY, name TEXT, email TEXT,
                amount REAL, created_at TEXT, is_active INTEGER,
                category TEXT, score REAL, notes TEXT
            )
        """)
        )
        for row in data:
            conn.execute(
                text(
                    "INSERT INTO test_data VALUES "
                    "(:id, :name, :email, :amount, :created_at, "
                    ":is_active, :category, :score, :notes)"
                ),
                {**row, "is_active": int(row["is_active"])},
            )
        conn.commit()
    engine.dispose()
    return path


def create_csv_test_file(rows: int = 1000, path: str = None) -> str:
    """Create a CSV file with test data."""
    gen = TestDataGenerator()
    data = gen.generate_standard_rows(rows)
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".csv", prefix="localdata_test_")
        os.close(fd)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    return path


def create_duckdb_test_db(rows: int = 1000, path: str = None) -> str:
    """Create a DuckDB database with test data. Returns file path."""
    import duckdb

    if path is None:
        fd, path = tempfile.mkstemp(suffix=".duckdb", prefix="localdata_test_")
        os.close(fd)
    conn = duckdb.connect(path)
    gen = TestDataGenerator()
    data = gen.generate_standard_rows(rows)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_data (
            id INTEGER PRIMARY KEY, name VARCHAR, email VARCHAR,
            amount DOUBLE, created_at VARCHAR, is_active BOOLEAN,
            category VARCHAR, score DOUBLE, notes VARCHAR
        )
    """)
    for row in data:
        conn.execute(
            "INSERT INTO test_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                row["id"],
                row["name"],
                row["email"],
                row["amount"],
                row["created_at"],
                row["is_active"],
                row["category"],
                row["score"],
                row["notes"],
            ],
        )
    conn.close()
    return path


def create_json_test_file(rows: int = 100, path: str = None) -> str:
    """Create a JSON file with test data."""
    gen = TestDataGenerator()
    data = gen.generate_standard_rows(rows)
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="localdata_test_")
        os.close(fd)
    with open(path, "w") as f:
        json.dump(data, f)
    return path
