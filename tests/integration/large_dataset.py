"""NYC Yellow Taxi dataset downloader and cache for enterprise-scale integration testing.

Downloads 3 months (Jan-Mar 2024) of NYC Yellow Taxi trip data (~10M rows)
from the TLC public data portal, concatenates them, and caches as a single
parquet file for reuse across test runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS = ["2024-01", "2024-02", "2024-03"]
CACHE_PATH = Path("tests/datasets/nyc_taxi_enterprise.parquet")

_COLUMNS = [
    ("vendorid", "INT"),
    ("tpep_pickup_datetime", "TIMESTAMP"),
    ("tpep_dropoff_datetime", "TIMESTAMP"),
    ("passenger_count", "FLOAT"),
    ("trip_distance", "FLOAT"),
    ("ratecodeid", "FLOAT"),
    ("store_and_fwd_flag", "VARCHAR(1)"),
    ("pulocationid", "INT"),
    ("dolocationid", "INT"),
    ("payment_type", "INT"),
    ("fare_amount", "FLOAT"),
    ("extra", "FLOAT"),
    ("mta_tax", "FLOAT"),
    ("tip_amount", "FLOAT"),
    ("tolls_amount", "FLOAT"),
    ("improvement_surcharge", "FLOAT"),
    ("total_amount", "FLOAT"),
    ("congestion_surcharge", "FLOAT"),
    ("airport_fee", "FLOAT"),
]

_DIALECT_TYPE_MAP: dict[str, dict[str, str]] = {
    "postgresql": {
        "INT": "INTEGER",
        "FLOAT": "DOUBLE PRECISION",
        "TIMESTAMP": "TIMESTAMP",
        "VARCHAR(1)": "VARCHAR(1)",
    },
    "mysql": {
        "INT": "INT",
        "FLOAT": "DOUBLE",
        "TIMESTAMP": "DATETIME",
        "VARCHAR(1)": "VARCHAR(1)",
    },
    "sqlite": {
        "INT": "INTEGER",
        "FLOAT": "REAL",
        "TIMESTAMP": "TEXT",
        "VARCHAR(1)": "TEXT",
    },
    "mssql": {
        "INT": "INT",
        "FLOAT": "FLOAT",
        "TIMESTAMP": "DATETIME2",
        "VARCHAR(1)": "VARCHAR(1)",
    },
    "oracle": {
        "INT": "NUMBER(10)",
        "FLOAT": "NUMBER",
        "TIMESTAMP": "TIMESTAMP",
        "VARCHAR(1)": "VARCHAR2(1)",
    },
}

_NULLABLE_COLUMNS = {
    "passenger_count",
    "ratecodeid",
    "store_and_fwd_flag",
    "congestion_surcharge",
    "airport_fee",
}


def _download_month(month: str, dest: Path) -> Path:
    """Download a single month of taxi data to dest directory."""
    url = f"{BASE_URL}/yellow_tripdata_{month}.parquet"
    out = dest / f"yellow_tripdata_{month}.parquet"
    if out.exists():
        return out

    print(f"Downloading NYC Taxi data for {month}...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download {url}: {exc}\n"
            "Check your internet connection and try again."
        ) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
    print(f"  Saved {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def ensure_dataset() -> Path:
    """Ensure dataset is downloaded and cached. Returns cache path."""
    if CACHE_PATH.exists():
        return CACHE_PATH

    tmp_dir = CACHE_PATH.parent / "_taxi_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    part_files = [_download_month(m, tmp_dir) for m in MONTHS]

    print("Concatenating monthly files into single cache...")
    frames = [pd.read_parquet(p) for p in part_files]
    combined = pd.concat(frames, ignore_index=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(CACHE_PATH, index=False)
    print(f"Cached {len(combined):,} rows to {CACHE_PATH}")

    # Clean up temporary per-month files
    for p in part_files:
        p.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    return CACHE_PATH


def get_dataset(max_rows: int | None = None) -> pd.DataFrame:
    """Return the cached dataset. Downloads if not cached.

    Args:
        max_rows: Optional cap on rows (for faster dev cycles). None = all ~10M rows.
    """
    ensure_dataset()
    df = pd.read_parquet(CACHE_PATH)
    df.columns = [c.lower() for c in df.columns]
    if max_rows is not None:
        df = df.head(max_rows)
    return df


def get_create_table_sql(dialect: str, table_name: str = "taxi_trips") -> str:
    """Return CREATE TABLE statement for the given SQL dialect.

    Supported dialects: postgresql, mysql, sqlite, mssql, oracle
    """
    dialect = dialect.lower()
    if dialect not in _DIALECT_TYPE_MAP:
        raise ValueError(
            f"Unsupported dialect '{dialect}'. "
            f"Supported: {', '.join(sorted(_DIALECT_TYPE_MAP))}"
        )
    type_map = _DIALECT_TYPE_MAP[dialect]
    lines = []
    for col_name, base_type in _COLUMNS:
        sql_type = type_map[base_type]
        nullable = "" if col_name in _NULLABLE_COLUMNS else " NOT NULL"
        lines.append(f"    {col_name} {sql_type}{nullable}")

    cols_sql = ",\n".join(lines)
    return f"CREATE TABLE {table_name} (\n{cols_sql}\n);"


def get_dataset_info() -> dict:
    """Return metadata: row count, columns, size on disk, etc."""
    if not CACHE_PATH.exists():
        return {
            "cached": False,
            "cache_path": str(CACHE_PATH),
            "message": "Dataset not yet downloaded. Call ensure_dataset() first.",
        }

    size_bytes = CACHE_PATH.stat().st_size
    import pyarrow.parquet as pq

    meta = pq.read_metadata(CACHE_PATH)
    schema = pq.read_schema(CACHE_PATH)

    return {
        "cached": True,
        "cache_path": str(CACHE_PATH),
        "row_count": meta.num_rows,
        "column_count": meta.num_columns,
        "columns": schema.names,
        "dtypes": {f.name: str(f.type) for f in schema},
        "size_mb": round(size_bytes / 1e6, 1),
        "months": MONTHS,
        "source": "NYC TLC Yellow Taxi Trip Data",
    }
