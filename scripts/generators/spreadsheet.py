"""Generate spreadsheet test fixtures (Excel .xlsx)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from generators._common import RowCounts, sub_dir


def _build_dataframe(n_rows: int) -> pd.DataFrame:
    """Build a synthetic spreadsheet DataFrame."""
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


def generate_spreadsheets(output_dir: str, row_counts: RowCounts) -> list[str]:
    """Generate Excel files in spreadsheet/ subdirectory.

    Requires openpyxl; silently skips if the library is unavailable.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("    [skip] openpyxl not installed, skipping Excel generation")
        return []

    d = sub_dir(output_dir, "spreadsheet")
    files: list[str] = []

    for label, n_rows in row_counts.items():
        df = _build_dataframe(n_rows)
        path = os.path.join(d, f"{label}.xlsx")
        df.to_excel(path, index=False, engine="openpyxl")
        files.append(path)

    return files
