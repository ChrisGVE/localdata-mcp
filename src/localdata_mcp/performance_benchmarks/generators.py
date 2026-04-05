"""Test dataset generation for benchmarks.

This module provides utilities to create synthetic test DataFrames and
SQLite databases of configurable size and shape for benchmark scenarios.
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from ..logging_manager import get_logger

logger = get_logger(__name__)


class DatasetGenerator:
    """Generate test datasets of various sizes and characteristics."""

    @staticmethod
    def create_test_dataframe(
        rows: int, text_heavy: bool = False, wide_table: bool = False
    ) -> pd.DataFrame:
        """Create test DataFrame with specified characteristics.

        Args:
            rows: Number of rows to generate
            text_heavy: Whether to include text-heavy columns
            wide_table: Whether to create many columns

        Returns:
            Generated test DataFrame
        """
        np.random.seed(42)  # For reproducible results

        # Base columns
        data = {
            "id": range(rows),
            "value": np.random.randint(0, 1000, rows),
            "score": np.random.uniform(0, 100, rows),
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="1min"),
        }

        if text_heavy:
            # Add text columns with varying lengths
            data["short_text"] = [f"Short text {i % 100}" for i in range(rows)]
            data["medium_text"] = [
                f"Medium length text content for testing purposes {i}" * 2
                for i in range(rows)
            ]
            data["long_text"] = [
                f"Very long text content that simulates real-world data " * 10 + f" {i}"
                for i in range(rows)
            ]

        if wide_table:
            # Add many numeric columns
            for i in range(20):
                data[f"metric_{i}"] = np.random.uniform(0, 1000, rows)

        return pd.DataFrame(data)

    @staticmethod
    def create_test_database(
        db_path: str,
        rows: int,
        table_name: str = "test_data",
        text_heavy: bool = False,
        wide_table: bool = False,
    ) -> str:
        """Create SQLite test database with specified characteristics.

        Args:
            db_path: Path for the database file
            rows: Number of rows to generate
            table_name: Name of the table to create
            text_heavy: Whether to include text-heavy columns
            wide_table: Whether to create many columns

        Returns:
            SQLAlchemy connection string
        """
        # Generate test data
        df = DatasetGenerator.create_test_dataframe(rows, text_heavy, wide_table)

        # Create database
        connection_string = f"sqlite:///{db_path}"
        engine = create_engine(connection_string)

        with engine.connect() as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        logger.info(f"Created test database with {rows} rows at {db_path}")
        return connection_string
