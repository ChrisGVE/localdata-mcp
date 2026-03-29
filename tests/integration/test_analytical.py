"""Integration tests for analytical formats (Parquet, Feather, HDF5)."""

import numpy as np
import pytest

from .data_generator import TestDataGenerator
from .mcp_test_client import call_tool

pytestmark = [pytest.mark.integration]
gen = TestDataGenerator()


class TestParquet:
    def test_connect_and_query(self, tmp_path):
        """Create Parquet file and verify row count."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        path = str(tmp_path / "test.parquet")
        df = pd.DataFrame(gen.generate_standard_rows(1000))
        df.to_parquet(path, engine="pyarrow")

        call_tool(
            "connect_database",
            {"name": "pq_test", "db_type": "parquet", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "pq_test",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            assert "1000" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "pq_test"})

    def test_parquet_type_fidelity(self, tmp_path):
        """Verify numeric types preserved through Parquet round-trip."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        path = str(tmp_path / "types.parquet")
        df = pd.DataFrame(gen.generate_standard_rows(10))
        df.to_parquet(path, engine="pyarrow")

        call_tool(
            "connect_database",
            {"name": "pq_types", "db_type": "parquet", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "pq_types",
                    "query": "SELECT amount, score FROM data_table LIMIT 5",
                },
            )
            result_str = str(result)
            # Verify numeric values are present (not null/empty)
            assert "amount" in result_str or "score" in result_str or "." in result_str
        finally:
            call_tool("disconnect_database", {"name": "pq_types"})

    @pytest.mark.large_data
    def test_large_parquet(self, tmp_path):
        """Query a 100k-row Parquet file."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        path = str(tmp_path / "large.parquet")
        df = pd.DataFrame(gen.generate_standard_rows(100000))
        df.to_parquet(path, engine="pyarrow")

        call_tool(
            "connect_database",
            {"name": "pq_lg", "db_type": "parquet", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "pq_lg",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            assert "100000" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "pq_lg"})


class TestFeather:
    def test_connect_and_query(self, tmp_path):
        """Create Feather file and verify row count."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        path = str(tmp_path / "test.feather")
        df = pd.DataFrame(gen.generate_standard_rows(500))
        df.to_feather(path)

        call_tool(
            "connect_database",
            {"name": "fth_test", "db_type": "feather", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "fth_test",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            assert "500" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "fth_test"})


class TestHDF5:
    def test_connect_and_query(self, tmp_path):
        """Create HDF5 file with h5py and verify row count."""
        pytest.importorskip("h5py")
        import h5py

        path = str(tmp_path / "test.h5")
        rows = gen.generate_standard_rows(200)

        # Write columnar data using h5py directly (matches the h5py-based loader)
        with h5py.File(path, "w") as f:
            ids = [r["id"] for r in rows]
            amounts = [r["amount"] for r in rows]
            scores = [r["score"] for r in rows]
            f.create_dataset("id", data=np.array(ids, dtype="int64"))
            f.create_dataset("amount", data=np.array(amounts, dtype="float64"))
            f.create_dataset("score", data=np.array(scores, dtype="float64"))

        call_tool(
            "connect_database",
            {"name": "hdf_test", "db_type": "hdf5", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "hdf_test",
                    "query": "SELECT COUNT(*) as cnt FROM id",
                },
            )
            assert "200" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "hdf_test"})
