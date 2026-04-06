#!/usr/bin/env python3
"""Download public test datasets for integration testing.

Usage:
    python scripts/download_test_datasets.py --all
    python scripts/download_test_datasets.py --taxi --housing
    python scripts/download_test_datasets.py --worldbank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import requests

DATASETS_DIR = Path("tests/datasets")

_TAXI_URL_TEMPLATE = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_{year}-{month:02d}.parquet"
)

_HOUSING_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"

_WORLDBANK_URL_TEMPLATE = (
    "https://api.worldbank.org/v2/country/all/indicator/"
    "{indicator}?format=json&per_page=1000"
)

_WORLDBANK_INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "SP.POP.TOTL": "Population, total",
}


def _sizeof_fmt(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


def _safe_get(
    url: str,
    *,
    timeout: int = 30,
    label: str = "data",
) -> requests.Response | None:
    """Perform a GET request with error handling."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  SKIP: failed to download {label}: {exc}")
        return None
    return resp


def _save_parquet_sample(
    content: bytes,
    dest: Path,
    max_rows: int,
) -> int:
    """Write parquet bytes, sample, and save to dest."""
    import pandas as pd

    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(content)
    df = pd.read_parquet(tmp)
    sample = df.sample(n=min(max_rows, len(df)), random_state=42)
    sample.to_parquet(dest, index=False)
    tmp.unlink()
    return len(sample)


def download_nyc_taxi(
    year: int = 2023,
    month: int = 1,
) -> bool:
    """Download NYC Yellow Taxi trip data, sampled to 1M rows."""
    dest = DATASETS_DIR / "nyc_taxi.parquet"
    url = _TAXI_URL_TEMPLATE.format(year=year, month=month)
    print(f"Downloading NYC Yellow Taxi data ({year}-{month:02d})...")

    resp = _safe_get(url, timeout=120, label="taxi data")
    if resp is None:
        return False

    nrows = _save_parquet_sample(resp.content, dest, 1_000_000)
    size = dest.stat().st_size
    print(f"  Saved {dest.name} ({nrows} rows, {_sizeof_fmt(size)})")
    return True


def download_ames_housing() -> bool:
    """Download Ames Housing dataset as CSV."""
    dest = DATASETS_DIR / "ames_housing.csv"
    print("Downloading Ames Housing data...")

    resp = _safe_get(_HOUSING_URL, label="housing data")
    if resp is None:
        return False

    dest.write_text(resp.text, encoding="utf-8")
    size = dest.stat().st_size
    lines = resp.text.count("\n")
    print(f"  Saved {dest.name} ({lines} rows, {_sizeof_fmt(size)})")
    return True


def _fetch_indicator(indicator: str) -> list[dict[str, Any]]:
    """Fetch a single World Bank indicator, return records."""
    url = _WORLDBANK_URL_TEMPLATE.format(indicator=indicator)
    resp = _safe_get(url, label=f"indicator {indicator}")
    if resp is None:
        return []
    data = resp.json()
    if not isinstance(data, list) or len(data) < 2:
        print(f"  SKIP: unexpected response for {indicator}")
        return []
    return [
        {
            "country": r.get("country", {}).get("value", ""),
            "country_code": r.get("countryiso3code", ""),
            "indicator": indicator,
            "year": r.get("date", ""),
            "value": r.get("value"),
        }
        for r in data[1]
        if r.get("value") is not None
    ]


def download_world_bank() -> bool:
    """Download World Bank indicators and save as CSV."""
    import pandas as pd

    dest = DATASETS_DIR / "world_bank.csv"
    print("Downloading World Bank indicator data...")

    all_records: list[dict[str, Any]] = []
    for code, label in _WORLDBANK_INDICATORS.items():
        print(f"  Fetching {label} ({code})...")
        records = _fetch_indicator(code)
        all_records.extend(records)

    if not all_records:
        print("  SKIP: no World Bank records retrieved")
        return False

    df = pd.DataFrame(all_records)
    df.to_csv(dest, index=False)
    size = dest.stat().st_size
    print(f"  Saved {dest.name} ({len(df)} rows, {_sizeof_fmt(size)})")
    return True


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Download public test datasets.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets",
    )
    parser.add_argument(
        "--taxi",
        action="store_true",
        help="NYC Yellow Taxi (parquet, 1M rows)",
    )
    parser.add_argument(
        "--housing",
        action="store_true",
        help="Ames Housing (CSV)",
    )
    parser.add_argument(
        "--worldbank",
        action="store_true",
        help="World Bank indicators (CSV)",
    )
    return parser


def _print_summary(results: dict[str, bool]) -> None:
    """Print a summary of download results."""
    print("\n--- Summary ---")
    for name, ok in results.items():
        status = "OK" if ok else "SKIPPED"
        print(f"  {name}: {status}")
    total = len(results)
    passed = sum(results.values())
    print(f"\n{passed}/{total} datasets downloaded successfully.")


def main() -> None:
    """Entry point: parse args, download requested datasets."""
    parser = _build_parser()
    args = parser.parse_args()

    if not any([args.all, args.taxi, args.housing, args.worldbank]):
        parser.print_help()
        sys.exit(1)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directory: {DATASETS_DIR.resolve()}\n")

    results: dict[str, bool] = {}
    if args.all or args.taxi:
        results["nyc_taxi"] = download_nyc_taxi()
    if args.all or args.housing:
        results["ames_housing"] = download_ames_housing()
    if args.all or args.worldbank:
        results["world_bank"] = download_world_bank()

    _print_summary(results)


if __name__ == "__main__":
    main()
