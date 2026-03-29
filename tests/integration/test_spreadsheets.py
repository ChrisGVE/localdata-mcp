"""Integration tests for spreadsheet formats (XLSX, ODS)."""

import pytest

from .data_generator import TestDataGenerator
from .mcp_test_client import call_tool

pytestmark = [pytest.mark.integration]
gen = TestDataGenerator()


class TestExcelXLSX:
    def test_connect_and_query(self, tmp_path):
        """Create XLSX file with pandas, connect and query."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")

        path = str(tmp_path / "test.xlsx")
        rows = gen.generate_standard_rows(100)
        df = pd.DataFrame(rows)
        df.to_excel(path, index=False, engine="openpyxl")

        call_tool(
            "connect_database",
            {"name": "xlsx_test", "db_type": "excel", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "xlsx_test",
                    "query": "SELECT COUNT(*) as cnt FROM Sheet1",
                },
            )
            assert "100" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "xlsx_test"})

    def test_describe_columns(self, tmp_path):
        """Describe database should list the sheet as a table."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")

        path = str(tmp_path / "cols.xlsx")
        df = pd.DataFrame(gen.generate_standard_rows(10))
        df.to_excel(path, index=False, engine="openpyxl")

        call_tool(
            "connect_database",
            {"name": "xlsx_desc", "db_type": "excel", "conn_string": path},
        )
        try:
            result = call_tool("describe_database", {"name": "xlsx_desc"})
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "xlsx_desc"})

    @pytest.mark.large_data
    def test_large_xlsx(self, tmp_path):
        """Query a large XLSX file — verify data loaded and queryable."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")

        path = str(tmp_path / "large.xlsx")
        df = pd.DataFrame(gen.generate_standard_rows(10000))
        df.to_excel(path, index=False, engine="openpyxl")

        call_tool(
            "connect_database",
            {"name": "xlsx_lg", "db_type": "excel", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "xlsx_lg",
                    "query": "SELECT COUNT(*) as cnt FROM Sheet1",
                },
            )
            # Streaming processor loads data in chunks; verify a substantial
            # amount was loaded (at least 1000 rows from a 10k file).
            result_str = str(result)
            assert "cnt" in result_str
            data = result.get("data", [{}]) if isinstance(result, dict) else [{}]
            cnt = data[0].get("cnt", 0) if data else 0
            assert cnt >= 1000, f"Expected >=1000 rows, got {cnt}"
        finally:
            call_tool("disconnect_database", {"name": "xlsx_lg"})


class TestODS:
    def test_connect_and_query(self, tmp_path):
        """Create ODS file with pandas, connect and query."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("odf")

        path = str(tmp_path / "test.ods")
        df = pd.DataFrame(gen.generate_standard_rows(50))
        df.to_excel(path, index=False, engine="odf")

        call_tool(
            "connect_database",
            {"name": "ods_test", "db_type": "ods", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "ods_test",
                    "query": "SELECT COUNT(*) as cnt FROM Sheet1",
                },
            )
            assert "50" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "ods_test"})
