#!/usr/bin/env python3
"""Download public test datasets for integration testing.

Usage:
    python scripts/download_test_datasets.py --all
    python scripts/download_test_datasets.py --taxi --housing
    python scripts/download_test_datasets.py --worldbank
"""

import argparse
import sys
from pathlib import Path

import requests

DATASETS_DIR = Path(__file__).resolve().parent.parent / "tests" / "datasets"

TAXI_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
)
HOUSING_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
WORLDBANK_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD"
    "?format=json&per_page=500&date=2020:2023"
)


def _sizeof_fmt(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


def download_taxi() -> None:
    """Download NYC Yellow Taxi trip data (sampled to 100K rows)."""
    import pandas as pd

    dest = DATASETS_DIR / "nyc_taxi_2023_01_100k.parquet"
    print(f"Downloading NYC Yellow Taxi data from TLC...")

    try:
        resp = requests.get(TAXI_URL, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  WARNING: failed to download taxi data: {exc}")
        return

    # Write raw parquet to a temp file, read + sample, then save
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(resp.content)

    df = pd.read_parquet(tmp)
    sample = df.sample(n=min(100_000, len(df)), random_state=42)
    sample.to_parquet(dest, index=False)
    tmp.unlink()

    size = dest.stat().st_size
    print(f"  Saved {dest.name} ({len(sample)} rows, {_sizeof_fmt(size)})")


def download_housing() -> None:
    """Download Ames Housing dataset (CSV)."""
    dest = DATASETS_DIR / "housing.csv"
    print("Downloading Ames Housing data...")

    try:
        resp = requests.get(HOUSING_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  WARNING: failed to download housing data: {exc}")
        return

    dest.write_text(resp.text, encoding="utf-8")
    size = dest.stat().st_size
    lines = resp.text.count("\n")
    print(f"  Saved {dest.name} ({lines} rows, {_sizeof_fmt(size)})")


def download_worldbank() -> None:
    """Download World Bank GDP indicator data (JSON)."""
    import json

    dest = DATASETS_DIR / "worldbank_gdp_2020_2023.json"
    print("Downloading World Bank GDP data...")

    try:
        resp = requests.get(WORLDBANK_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  WARNING: failed to download World Bank data: {exc}")
        return

    data = resp.json()
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    size = dest.stat().st_size
    records = len(data[1]) if isinstance(data, list) and len(data) > 1 else "?"
    print(f"  Saved {dest.name} ({records} records, {_sizeof_fmt(size)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public test datasets for integration testing."
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument(
        "--taxi", action="store_true", help="NYC Yellow Taxi (parquet, 100K rows)"
    )
    parser.add_argument("--housing", action="store_true", help="Ames Housing (CSV)")
    parser.add_argument(
        "--worldbank", action="store_true", help="World Bank GDP (JSON)"
    )
    args = parser.parse_args()

    if not any([args.all, args.taxi, args.housing, args.worldbank]):
        parser.print_help()
        sys.exit(1)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directory: {DATASETS_DIR}\n")

    if args.all or args.taxi:
        download_taxi()
    if args.all or args.housing:
        download_housing()
    if args.all or args.worldbank:
        download_worldbank()

    print("\nDone.")


if __name__ == "__main__":
    main()
