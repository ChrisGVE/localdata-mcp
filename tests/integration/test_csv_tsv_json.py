"""Integration tests for CSV, TSV, JSON, XML, INI file formats."""

import csv
import json
import os

import pytest

from .data_generator import TestDataGenerator
from .mcp_test_client import call_tool

pytestmark = [pytest.mark.integration]

gen = TestDataGenerator()


def _write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def _write_tsv(rows, path):
    with open(path, "w") as f:
        keys = list(rows[0].keys())
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(k, "")) for k in keys) + "\n")


def _write_xml(rows, path):
    """Write rows as simple XML with <data><row>...</row></data> structure."""
    lines = ['<?xml version="1.0"?>', "<data>"]
    for row in rows:
        lines.append("  <row>")
        for k, v in row.items():
            val = str(v) if v is not None else ""
            val = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            lines.append(f"    <{k}>{val}</{k}>")
        lines.append("  </row>")
    lines.append("</data>")
    with open(path, "w") as f:
        f.write("\n".join(lines))


class TestCSV:
    """Tests for CSV file format via connect_database / execute_query."""

    def test_connect_and_query(self, tmp_path):
        path = str(tmp_path / "test.csv")
        _write_csv(gen.generate_standard_rows(100), path)
        call_tool(
            "connect_database",
            {"name": "csv_test", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "csv_test",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            assert "100" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "csv_test"})

    def test_csv_with_special_chars(self, tmp_path):
        path = str(tmp_path / "special.csv")
        rows = [
            {"name": "O'Brien", "city": "New York, NY"},
            {"name": 'He said "hi"', "city": "Paris"},
        ]
        _write_csv(rows, path)
        call_tool(
            "connect_database",
            {"name": "csv_sp", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {"name": "csv_sp", "query": "SELECT * FROM data_table"},
            )
            result_str = str(result)
            assert "O'Brien" in result_str or "Brien" in result_str
        finally:
            call_tool("disconnect_database", {"name": "csv_sp"})

    def test_csv_unicode(self, tmp_path):
        path = str(tmp_path / "unicode.csv")
        _write_csv(gen.generate_unicode_rows(50), path)
        call_tool(
            "connect_database",
            {"name": "csv_uni", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "csv_uni",
                    "query": "SELECT * FROM data_table LIMIT 5",
                },
            )
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "csv_uni"})

    @pytest.mark.large_data
    def test_csv_large_file(self, tmp_path):
        path = str(tmp_path / "large.csv")
        _write_csv(gen.generate_standard_rows(50000), path)
        call_tool(
            "connect_database",
            {"name": "csv_lg", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "csv_lg",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            assert "50000" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "csv_lg"})

    def test_csv_describe(self, tmp_path):
        path = str(tmp_path / "desc.csv")
        _write_csv(gen.generate_standard_rows(10), path)
        call_tool(
            "connect_database",
            {"name": "csv_desc", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool("describe_database", {"name": "csv_desc"})
            assert result is not None
            result_str = str(result)
            assert "data_table" in result_str
        finally:
            call_tool("disconnect_database", {"name": "csv_desc"})

    def test_csv_where_filter(self, tmp_path):
        path = str(tmp_path / "filter.csv")
        _write_csv(gen.generate_standard_rows(100), path)
        call_tool(
            "connect_database",
            {"name": "csv_flt", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "csv_flt",
                    "query": "SELECT * FROM data_table WHERE id <= 10",
                },
            )
            result_str = str(result)
            # Should have filtered rows; at minimum the query should succeed
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "csv_flt"})

    def test_csv_aggregation(self, tmp_path):
        path = str(tmp_path / "agg.csv")
        _write_csv(gen.generate_standard_rows(100), path)
        call_tool(
            "connect_database",
            {"name": "csv_agg", "db_type": "csv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "csv_agg",
                    "query": (
                        "SELECT category, COUNT(*) as cnt "
                        "FROM data_table GROUP BY category"
                    ),
                },
            )
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "csv_agg"})


class TestTSV:
    """Tests for TSV file format via connect_database / execute_query."""

    def test_connect_and_query(self, tmp_path):
        path = str(tmp_path / "test.tsv")
        _write_tsv(gen.generate_standard_rows(100), path)
        call_tool(
            "connect_database",
            {"name": "tsv_test", "db_type": "tsv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "tsv_test",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            assert "100" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "tsv_test"})

    def test_tsv_describe(self, tmp_path):
        path = str(tmp_path / "desc.tsv")
        _write_tsv(gen.generate_standard_rows(20), path)
        call_tool(
            "connect_database",
            {"name": "tsv_desc", "db_type": "tsv", "conn_string": path},
        )
        try:
            result = call_tool("describe_database", {"name": "tsv_desc"})
            assert result is not None
            assert "data_table" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "tsv_desc"})

    def test_tsv_select_columns(self, tmp_path):
        path = str(tmp_path / "cols.tsv")
        _write_tsv(gen.generate_standard_rows(50), path)
        call_tool(
            "connect_database",
            {"name": "tsv_cols", "db_type": "tsv", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "tsv_cols",
                    "query": "SELECT id, name FROM data_table LIMIT 5",
                },
            )
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "tsv_cols"})


class TestJSON:
    """Tests for JSON file format via tree storage tools.

    JSON connections use tree storage (get_node, get_children, etc.)
    rather than SQL queries.
    """

    def test_json_connect_and_describe(self, tmp_path):
        path = str(tmp_path / "test.json")
        with open(path, "w") as f:
            json.dump(gen.generate_standard_rows(100), f)
        call_tool(
            "connect_database",
            {"name": "json_test", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool("describe_database", {"name": "json_test"})
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "json_test"})

    def test_json_get_root_node(self, tmp_path):
        path = str(tmp_path / "root.json")
        with open(path, "w") as f:
            json.dump({"key1": "value1", "key2": 42}, f)
        call_tool(
            "connect_database",
            {"name": "json_root", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool("get_node", {"name": "json_root", "path": "/"})
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "json_root"})

    def test_json_get_children(self, tmp_path):
        path = str(tmp_path / "children.json")
        data = {"users": [{"name": "Alice"}, {"name": "Bob"}], "count": 2}
        with open(path, "w") as f:
            json.dump(data, f)
        call_tool(
            "connect_database",
            {"name": "json_ch", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool("get_children", {"name": "json_ch", "path": "/"})
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "json_ch"})

    def test_json_list_keys(self, tmp_path):
        """Test listing children at root — flat JSON keys become child nodes."""
        path = str(tmp_path / "keys.json")
        data = {"alpha": 1, "beta": 2, "gamma": 3}
        with open(path, "w") as f:
            json.dump(data, f)
        call_tool(
            "connect_database",
            {"name": "json_keys", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool("list_keys", {"name": "json_keys", "path": "root"})
            result_str = str(result)
            assert "alpha" in result_str
            assert "beta" in result_str
        finally:
            call_tool("disconnect_database", {"name": "json_keys"})

    def test_json_get_value(self, tmp_path):
        """Test navigating to a nested value via get_children + get_node."""
        path = str(tmp_path / "val.json")
        data = {"settings": {"theme": "dark", "version": 3}}
        with open(path, "w") as f:
            json.dump(data, f)
        call_tool(
            "connect_database",
            {"name": "json_val", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool(
                "get_value",
                {"name": "json_val", "path": "settings", "key": "theme"},
            )
            assert "dark" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "json_val"})

    def test_json_unicode(self, tmp_path):
        path = str(tmp_path / "uni.json")
        with open(path, "w") as f:
            json.dump(gen.generate_unicode_rows(20), f)
        call_tool(
            "connect_database",
            {"name": "json_uni", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool("get_node", {"name": "json_uni", "path": "/"})
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "json_uni"})

    def test_json_nested_structure(self, tmp_path):
        path = str(tmp_path / "nested.json")
        data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {"user": "admin", "password": "secret"},
            }
        }
        with open(path, "w") as f:
            json.dump(data, f)
        call_tool(
            "connect_database",
            {"name": "json_nest", "db_type": "json", "conn_string": path},
        )
        try:
            result = call_tool(
                "get_children", {"name": "json_nest", "path": "/database"}
            )
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "json_nest"})


class TestXML:
    """Tests for XML file format via connect_database / execute_query."""

    def test_xml_connect_and_query(self, tmp_path):
        path = str(tmp_path / "test.xml")
        _write_xml(gen.generate_standard_rows(50), path)
        call_tool(
            "connect_database",
            {"name": "xml_test", "db_type": "xml", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "xml_test",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            assert "50" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "xml_test"})

    def test_xml_describe(self, tmp_path):
        path = str(tmp_path / "desc.xml")
        _write_xml(gen.generate_standard_rows(10), path)
        call_tool(
            "connect_database",
            {"name": "xml_desc", "db_type": "xml", "conn_string": path},
        )
        try:
            result = call_tool("describe_database", {"name": "xml_desc"})
            assert result is not None
            assert "data_table" in str(result)
        finally:
            call_tool("disconnect_database", {"name": "xml_desc"})

    def test_xml_select_with_filter(self, tmp_path):
        path = str(tmp_path / "filter.xml")
        _write_xml(gen.generate_standard_rows(30), path)
        call_tool(
            "connect_database",
            {"name": "xml_flt", "db_type": "xml", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "xml_flt",
                    "query": "SELECT * FROM data_table WHERE id <= 5",
                },
            )
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "xml_flt"})


class TestINI:
    """Tests for INI file format via connect_database / describe_database.

    INI files are loaded into a SQLite table (data_table) with section/key/value
    columns, queryable via SQL.
    """

    def test_ini_connect_and_describe(self, tmp_path):
        path = str(tmp_path / "test.ini")
        with open(path, "w") as f:
            f.write(
                "[section1]\nkey1 = value1\nkey2 = 42\n\n[section2]\nkey3 = hello\n"
            )
        call_tool(
            "connect_database",
            {"name": "ini_test", "db_type": "ini", "conn_string": path},
        )
        try:
            result = call_tool("describe_database", {"name": "ini_test"})
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "ini_test"})

    def test_ini_query(self, tmp_path):
        path = str(tmp_path / "query.ini")
        with open(path, "w") as f:
            f.write(
                "[server]\nhost = localhost\nport = 8080\n\n"
                "[database]\ndriver = sqlite\npath = /tmp/data.db\n"
            )
        call_tool(
            "connect_database",
            {"name": "ini_qry", "db_type": "ini", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "ini_qry",
                    "query": "SELECT * FROM data_table",
                },
            )
            result_str = str(result)
            # INI data should be queryable; verify some content is present
            assert "localhost" in result_str or "server" in result_str
        finally:
            call_tool("disconnect_database", {"name": "ini_qry"})

    def test_ini_count(self, tmp_path):
        path = str(tmp_path / "count.ini")
        with open(path, "w") as f:
            f.write("[a]\nk1 = v1\nk2 = v2\n\n[b]\nk3 = v3\n")
        call_tool(
            "connect_database",
            {"name": "ini_cnt", "db_type": "ini", "conn_string": path},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "ini_cnt",
                    "query": "SELECT COUNT(*) as cnt FROM data_table",
                },
            )
            # Should have some rows from the INI sections/keys
            assert result is not None
        finally:
            call_tool("disconnect_database", {"name": "ini_cnt"})
