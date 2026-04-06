#!/usr/bin/env python3
"""Pre-generate test fixture files for integration tests.

Generates test data in multiple formats across subdirectories:
  tabular/     - CSV, TSV, JSON, JSONL
  tree/        - YAML, TOML, INI, XML
  graph/       - DOT, GML, GraphML, Mermaid
  spreadsheet/ - Excel (.xlsx, requires openpyxl)
  analytical/  - Parquet, Feather (requires pyarrow)
  rdf/         - Turtle, N-Triples
  edge_cases/  - empty, BOM, wide, unicode CSV files

Usage:
  python scripts/generate_test_files.py
  python scripts/generate_test_files.py --size large --output-dir tests/fixtures
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

# Add scripts/ to sys.path so generators package is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generators import (  # noqa: E402
    generate_analytical,
    generate_edge_cases,
    generate_graph,
    generate_rdf,
    generate_spreadsheets,
    generate_tabular,
    generate_tree,
)

ROW_COUNTS: dict[str, dict[str, int]] = {
    "small": {"standard": 1_000, "large": 10_000},
    "medium": {"standard": 10_000, "large": 100_000},
    "large": {"standard": 100_000, "large": 500_000},
}


def _format_size(size_bytes: int) -> str:
    """Return a human-readable file size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.1f} KB"
    return f"{size_kb / 1024:.1f} MB"


def _print_results(files: list[str]) -> None:
    """Print a summary of all generated files with their sizes."""
    print(f"\nDone. Generated {len(files)} files:")
    for path in files:
        print(f"  {path} ({_format_size(os.path.getsize(path))})")


def _run_generator(
    label: str,
    step: int,
    total: int,
    func: Callable[..., list[str]],
    *args: object,
) -> list[str]:
    """Run a single generator with progress output and error handling."""
    print(f"  [{step}/{total}] {label}...")
    try:
        return func(*args)
    except Exception as exc:
        print(f"    [error] {label}: {exc}")
        return []


def generate_all(output_dir: str, size: str) -> list[str]:
    """Orchestrate all generators and return the list of created files."""
    row_counts = ROW_COUNTS[size]
    total = 7

    steps: list[tuple[str, Callable[..., list[str]], tuple[object, ...]]] = [
        ("Tabular (CSV, TSV, JSON, JSONL)", generate_tabular, (output_dir, row_counts)),
        ("Tree (YAML, TOML, INI, XML)", generate_tree, (output_dir,)),
        ("Graph (DOT, GML, GraphML, Mermaid)", generate_graph, (output_dir,)),
        ("Spreadsheet (Excel)", generate_spreadsheets, (output_dir, row_counts)),
        (
            "Analytical (Parquet, Feather)",
            generate_analytical,
            (output_dir, row_counts),
        ),
        ("RDF (Turtle, N-Triples)", generate_rdf, (output_dir,)),
        ("Edge cases (empty, BOM, wide, unicode)", generate_edge_cases, (output_dir,)),
    ]

    all_files: list[str] = []
    for idx, (label, func, args) in enumerate(steps, start=1):
        all_files.extend(_run_generator(label, idx, total, func, *args))

    return all_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate test fixture files for integration tests.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/fixtures",
        help="Output directory (default: tests/fixtures)",
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Dataset size preset (default: medium)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the test file generation script."""
    args = parse_args(argv)
    row_counts = ROW_COUNTS[args.size]
    standard = row_counts["standard"]
    large = row_counts["large"]

    print(
        f"Generating fixtures: size={args.size} "
        f"(standard={standard}, large={large}), "
        f"output={args.output_dir}"
    )

    files = generate_all(args.output_dir, args.size)
    _print_results(files)


if __name__ == "__main__":
    main()
