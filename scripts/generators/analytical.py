"""Generate analytical test fixtures (Parquet, Feather)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from generators._common import RowCounts, sub_dir


def _build_dataframe(n_rows: int) -> pd.DataFrame:
    """Build a synthetic analytical DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": range(1, n_rows + 1),
            "name": [f"item_{i}" for i in range(1, n_rows + 1)],
            "value": rng.normal(100.0, 25.0, n_rows).round(2),
            "category": rng.choice(["A", "B", "C", "D"], n_rows),
            "flag": rng.choice([True, False], n_rows),
        }
    )


def generate_analytical(output_dir: str, row_counts: RowCounts) -> list[str]:
    """Generate Parquet and Feather files in analytical/ subdirectory.

    Requires pyarrow; silently skips if the library is unavailable.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("    [skip] pyarrow not installed, skipping Parquet/Feather")
        return []

    d = sub_dir(output_dir, "analytical")
    files: list[str] = []

    for label, n_rows in row_counts.items():
        df = _build_dataframe(n_rows)

        parquet_path = os.path.join(d, f"{label}.parquet")
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        files.append(parquet_path)

        feather_path = os.path.join(d, f"{label}.feather")
        df.to_feather(feather_path)
        files.append(feather_path)

    return files
