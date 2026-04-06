"""Generate tabular test fixtures (CSV, TSV, JSON, JSONL)."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from generators._common import RowCounts, sub_dir


def _build_dataframe(n_rows: int) -> pd.DataFrame:
    """Build a synthetic tabular DataFrame with mixed column types."""
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


def _write_jsonl(path: str, df: pd.DataFrame) -> str:
    """Write a DataFrame as newline-delimited JSON (JSONL)."""
    records = df.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str) + "\n")
    return path


def generate_tabular(output_dir: str, row_counts: RowCounts) -> list[str]:
    """Generate CSV, TSV, JSON, and JSONL files in tabular/ subdirectory."""
    d = sub_dir(output_dir, "tabular")
    files: list[str] = []

    for label, n_rows in row_counts.items():
        df = _build_dataframe(n_rows)
        prefix = f"{label}"

        csv_path = os.path.join(d, f"{prefix}.csv")
        df.to_csv(csv_path, index=False)
        files.append(csv_path)

        tsv_path = os.path.join(d, f"{prefix}.tsv")
        df.to_csv(tsv_path, index=False, sep="\t")
        files.append(tsv_path)

        json_path = os.path.join(d, f"{prefix}.json")
        df.to_json(json_path, orient="records", indent=2)
        files.append(json_path)

        jsonl_path = os.path.join(d, f"{prefix}.jsonl")
        _write_jsonl(jsonl_path, df)
        files.append(jsonl_path)

    return files
