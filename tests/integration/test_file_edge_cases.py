"""Integration tests for file format edge cases.

Verifies that the server handles unusual or boundary-condition inputs
gracefully — either succeeding or returning a clear error, never crashing.
"""

import csv
import json

import pytest

from .mcp_test_client import call_tool

pytestmark = [pytest.mark.integration]


def _assert_graceful(result):
    """Assert that a result is either successful or a clear error, not a crash."""
    assert result is not None
    if isinstance(result, dict) and result.get("error"):
        err = result["error"]
        # Error should be a readable string, not a raw traceback
        assert (
            isinstance(err, str) or "message" in str(result) or "Error" in str(result)
        )


class TestCSVEdgeCases:
    """Edge cases for CSV file handling."""

    def test_empty_csv_file(self, tmp_path):
        """0-byte CSV — connect should handle gracefully."""
        path = str(tmp_path / "empty.csv")
        with open(path, "w") as f:
            pass  # 0 bytes

        result = call_tool(
            "connect_database",
            {"name": "edge_empty_csv", "db_type": "csv", "conn_string": path},
        )
        try:
            _assert_graceful(result)
        finally:
            call_tool("disconnect_database", {"name": "edge_empty_csv"})

    def test_csv_headers_only(self, tmp_path):
        """CSV with header row but no data rows — should connect, 0 rows."""
        path = str(tmp_path / "headers_only.csv")
        with open(path, "w", newline="") as f:
            f.write("id,name,value\n")

        call_tool(
            "connect_database",
            {"name": "edge_hdr_csv", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "edge_hdr_csv",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            _assert_graceful(result)
            # Should report 0 rows
            assert "0" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "edge_hdr_csv"})

    def test_csv_single_row(self, tmp_path):
        """CSV with exactly one data row — should work normally."""
        path = str(tmp_path / "single.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "name"])
            w.writeheader()
            w.writerow({"id": "1", "name": "Alice"})

        call_tool(
            "connect_database",
            {"name": "edge_single_csv", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "edge_single_csv",
                    "query": "SELECT * FROM data_table",
                },
            )
            _assert_graceful(result)
            assert "Alice" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "edge_single_csv"})

    def test_csv_utf8_bom(self, tmp_path):
        """CSV with UTF-8 BOM — BOM should be stripped, columns named correctly."""
        path = str(tmp_path / "bom.csv")
        with open(path, "wb") as f:
            f.write(b"\xef\xbb\xbf")  # UTF-8 BOM
            f.write("id,name,value\n1,test,100\n".encode("utf-8"))

        call_tool(
            "connect_database",
            {"name": "edge_bom_csv", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "edge_bom_csv",
                    "query": "SELECT * FROM data_table",
                },
            )
            _assert_graceful(result)
            assert "test" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "edge_bom_csv"})

    def test_csv_wide_200_columns(self, tmp_path):
        """CSV with 200+ columns — should load all columns without error."""
        path = str(tmp_path / "wide.csv")
        num_cols = 200
        headers = [f"col_{i}" for i in range(num_cols)]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for row_idx in range(5):
                w.writerow({h: f"v{row_idx}_{i}" for i, h in enumerate(headers)})

        call_tool(
            "connect_database",
            {"name": "edge_wide_csv", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "edge_wide_csv",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            _assert_graceful(result)
            assert "5" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "edge_wide_csv"})

    def test_csv_malformed_inconsistent_columns(self, tmp_path):
        """CSV with inconsistent column counts — should handle gracefully."""
        path = str(tmp_path / "malformed.csv")
        with open(path, "w") as f:
            f.write("a,b,c\n")
            f.write("1,2,3\n")
            f.write("4,5\n")  # missing column
            f.write("6,7,8,9\n")  # extra column

        result = call_tool(
            "connect_database",
            {"name": "edge_malform_csv", "db_type": "csv", "conn_string": path},
        )
        try:
            _assert_graceful(result)
            # If connect succeeded, try a query too
            if not (isinstance(result, dict) and result.get("error")):
                qresult = call_tool(
                    "execute_query",
                    {
                        "name": "edge_malform_csv",
                        "query": "SELECT COUNT(*) as cnt FROM data_table",
                    },
                )
                _assert_graceful(qresult)
        finally:
            call_tool("disconnect_database", {"name": "edge_malform_csv"})

    def test_csv_special_characters_in_headers(self, tmp_path):
        """CSV with spaces, dots, brackets in column names."""
        path = str(tmp_path / "special_headers.csv")
        with open(path, "w", newline="") as f:
            f.write("First Name,last.name,value[usd],count (total)\n")
            f.write("Alice,Smith,100,5\n")
            f.write("Bob,Jones,200,10\n")

        call_tool(
            "connect_database",
            {"name": "edge_spechdr_csv", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "edge_spechdr_csv",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            _assert_graceful(result)
            assert "2" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "edge_spechdr_csv"})


class TestJSONEdgeCases:
    """Edge cases for JSON (tree) file handling."""

    def test_empty_json_object(self, tmp_path):
        """Empty JSON object {} — should connect, tree has no children."""
        path = str(tmp_path / "empty_obj.json")
        with open(path, "w") as f:
            json.dump({}, f)

        call_tool(
            "connect_database",
            {"name": "edge_empty_json", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool("get_children", {"name": "edge_empty_json"})
            _assert_graceful(result)
        finally:
            call_tool("disconnect_database", {"name": "edge_empty_json"})

    def test_empty_json_array(self, tmp_path):
        """Empty JSON array [] — should handle gracefully."""
        path = str(tmp_path / "empty_arr.json")
        with open(path, "w") as f:
            json.dump([], f)

        result = call_tool(
            "connect_database",
            {"name": "edge_arr_json", "db_type": "json", "conn_string": path},
        )
        try:
            _assert_graceful(result)
        finally:
            call_tool("disconnect_database", {"name": "edge_arr_json"})

    def test_deeply_nested_json(self, tmp_path):
        """JSON with 10+ levels of nesting — should load without stack overflow."""
        path = str(tmp_path / "deep.json")
        data = {"level": 0}
        current = data
        for i in range(1, 15):
            current["nested"] = {"level": i}
            current = current["nested"]
        current["leaf_value"] = "deep_leaf"

        with open(path, "w") as f:
            json.dump(data, f)

        call_tool(
            "connect_database",
            {"name": "edge_deep_json", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool("get_children", {"name": "edge_deep_json"})
            _assert_graceful(result)
            assert "level" in str(result) or "nested" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "edge_deep_json"})

    def test_large_single_value_json(self, tmp_path):
        """JSON with a ~1MB string value — should handle without issues."""
        path = str(tmp_path / "large_val.json")
        large_string = "x" * (1024 * 1024)  # 1MB
        data = {"big_field": large_string, "small_field": "ok"}

        with open(path, "w") as f:
            json.dump(data, f)

        call_tool(
            "connect_database",
            {"name": "edge_large_json", "db_type": "json", "conn_string": path},
        )
        try:
            # Just verify we can navigate the tree, don't fetch the huge value
            result = call_tool("get_children", {"name": "edge_large_json"})
            _assert_graceful(result)
            # Root should have a node with property_count reflecting the two keys
            assert "root" in str(result) or "children" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "edge_large_json"})
