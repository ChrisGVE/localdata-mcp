"""tests/v3/test_export.py — E7.3 NX-8 interface, guards, renderers.

The exit-gate assertions for nexus/export/: the ordered write guards
(containment injected from NX-6 and consulted first, the NFR-115
explicit-overwrite refusal, NFR-111 temp-file-rename atomicity with a
mid-write fault leaving the target untouched), the CWE-1236
formula-injection neutralization on exactly the CSV/Excel/Markdown
renderers, and one round-trip per FR-902 format.
"""

from __future__ import annotations

import io
import json as jsonlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from localdata_mcp.nexus.chokepoint.guard import Result
from localdata_mcp.nexus.chokepoint.path_contain import PathRefusedError
from localdata_mcp.nexus.export.interface import (
    ExportError,
    OverwriteRefusedError,
    UnknownFormatError,
    as_dataframe,
    export_to_file,
    neutralize_formulas,
    render,
    write_atomic,
)


def contain_under(root: Path) -> Any:
    """A stand-in for NX-6's contain_path service, rooted at tmp."""

    def contain(candidate: Any, *, mode: str) -> Path:
        real = Path(candidate).resolve()
        if not (real == root or real.is_relative_to(root)):
            raise PathRefusedError(
                f"{real} outside {root}",
                mode="write",  # type: ignore[arg-type]
            )
        return real

    return contain


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2], "label": ["a", "b"]})


class TestWriteGuards:
    def test_containment_is_consulted_before_anything(self, tmp_path: Path) -> None:
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        with pytest.raises(PathRefusedError):
            export_to_file(
                pd.DataFrame({"x": [1]}),
                "csv",
                tmp_path / "outside.csv",
                contain=contain_under(allowed),
            )
        assert not (tmp_path / "outside.csv").exists()

    def test_nfr115_existing_target_refused_without_overwrite(
        self, tmp_path: Path, frame: pd.DataFrame
    ) -> None:
        target = tmp_path / "out.csv"
        target.write_text("previous content")
        with pytest.raises(OverwriteRefusedError) as refusal:
            export_to_file(frame, "csv", target, contain=contain_under(tmp_path))
        assert "overwrite=True" in str(refusal.value)
        assert target.read_text() == "previous content"

    def test_explicit_overwrite_replaces(
        self, tmp_path: Path, frame: pd.DataFrame
    ) -> None:
        target = tmp_path / "out.csv"
        target.write_text("previous content")
        written = export_to_file(
            frame, "csv", target, contain=contain_under(tmp_path), overwrite=True
        )
        assert written == target.resolve()
        assert "id,label" in target.read_text()

    def test_nfr111_failure_leaves_target_unmodified(self, tmp_path: Path) -> None:
        """Fault-injection mid-write: os.replace never ran, the temp
        file is cleaned up, the prior artifact is intact."""
        target = tmp_path / "out.bin"
        target.write_bytes(b"prior")
        import localdata_mcp.nexus.export.interface as interface

        real_fdopen = interface.os.fdopen

        def exploding_fdopen(*args: Any, **kwargs: Any) -> Any:
            raise OSError("disk failure mid-write")

        interface.os.fdopen = exploding_fdopen  # type: ignore[assignment]
        try:
            with pytest.raises(OSError):
                write_atomic(b"new content", target)
        finally:
            interface.os.fdopen = real_fdopen  # type: ignore[assignment]
        assert target.read_bytes() == b"prior"
        assert list(tmp_path.glob(".out.bin.*")) == []

    def test_atomic_write_lands_complete(self, tmp_path: Path) -> None:
        target = tmp_path / "fresh.bin"
        write_atomic(b"complete artifact", target)
        assert target.read_bytes() == b"complete artifact"
        assert list(tmp_path.glob(".fresh.bin.*")) == []

    def test_unknown_format_is_refused_by_name(self, frame: pd.DataFrame) -> None:
        with pytest.raises(UnknownFormatError) as refusal:
            render(frame, "pdf")
        assert "csv" in str(refusal.value)  # declared roster named


class TestFormulaInjectionDefense:
    HOSTILE = pd.DataFrame(
        {"cell": ["=cmd|' /C calc'!A0", "+SUM(A1)", "-2+3", "@evil", "safe"]}
    )

    def test_neutralizer_prefixes_formula_strings_only(self) -> None:
        frame = pd.DataFrame({"s": ["=x", "ok"], "n": [-5, 2]})
        out = neutralize_formulas(frame)
        assert list(out["s"]) == ["'=x", "ok"]
        assert list(out["n"]) == [-5, 2]  # numbers untouched

    @pytest.mark.parametrize("format_name", ["csv", "markdown"])
    def test_text_renderers_neutralize(self, format_name: str) -> None:
        text = render(self.HOSTILE, format_name).decode("utf-8")
        assert "'=cmd" in text
        assert "\n=cmd" not in text

    def test_excel_renderer_neutralizes(self) -> None:
        payload = render(self.HOSTILE, "excel")
        frame = pd.read_excel(io.BytesIO(payload))
        assert frame["cell"].iloc[0].startswith("'=")

    def test_binary_formats_do_not_mangle_data(self, frame: pd.DataFrame) -> None:
        """Parquet/Arrow carry no formula semantics — a leading '-' in
        DATA must round-trip untouched."""
        hostile = pd.DataFrame({"v": ["-real-value"]})
        back = pd.read_parquet(io.BytesIO(render(hostile, "parquet")))
        assert back["v"].iloc[0] == "-real-value"


class TestRenderers:
    def test_csv_round_trip(self, frame: pd.DataFrame) -> None:
        text = render(frame, "csv").decode("utf-8")
        assert text.splitlines()[0] == "id,label"

    def test_guard_result_is_a_first_class_payload(self) -> None:
        result = Result(columns=("a", "b"), rows=((1, "x"),), category="query")
        text = render(result, "csv").decode("utf-8")
        assert "a,b" in text and "1,x" in text

    def test_records_payload_accepted(self) -> None:
        text = render([{"k": 1}, {"k": 2}], "csv").decode("utf-8")
        assert text.splitlines()[0] == "k"

    def test_non_tabular_payload_refused_with_shape_error(self) -> None:
        with pytest.raises(ExportError):
            as_dataframe(42)

    def test_parquet_and_arrow_round_trip(self, frame: pd.DataFrame) -> None:
        import pyarrow.ipc

        back = pd.read_parquet(io.BytesIO(render(frame, "parquet")))
        assert back.equals(frame)
        table = pyarrow.ipc.open_file(
            io.BytesIO(render(frame, "arrow"))
        ).read_all()
        assert table.to_pandas().equals(frame)

    def test_json_tabular_is_records(self, frame: pd.DataFrame) -> None:
        records = jsonlib.loads(render(frame, "json"))
        assert records == [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}]

    def test_json_mapping_passes_through(self) -> None:
        payload = {"meta": {"rows": 2}}
        assert jsonlib.loads(render(payload, "json")) == payload

    def test_markdown_table_document(self, frame: pd.DataFrame) -> None:
        text = render(frame, "markdown").decode("utf-8")
        assert text.startswith("| id | label |")
        assert "| 1 | a |" in text

    def test_schema_document(self) -> None:
        schema = {
            "users": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "email", "type": "TEXT", "nullable": False},
                ]
            }
        }
        text = render(schema, "schema").decode("utf-8")
        assert "## users" in text
        assert "| id | INTEGER | True | True |" in text

    def test_schema_malformed_refused(self) -> None:
        with pytest.raises(ExportError):
            render({"users": {"no_columns": []}}, "schema")

    def test_graph_document_with_mermaid(self) -> None:
        graph = {
            "nodes": [{"id": "a", "label": "Alpha"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b", "label": "links"}],
        }
        text = render(graph, "graph").decode("utf-8")
        assert "2 nodes, 1 edges" in text
        assert "```mermaid" in text
        assert 'a["Alpha"]' in text
        assert "a -->|links| b" in text

    def test_graph_mermaid_labels_cannot_inject_syntax(self) -> None:
        graph = {
            "nodes": [{"id": "a", "label": 'x"]-->evil["y'}],
            "edges": [],
        }
        text = render(graph, "graph").decode("utf-8")
        assert '"]-->' not in text.split("```mermaid")[1]

    def test_graph_malformed_refused(self) -> None:
        with pytest.raises(ExportError):
            render({"nodes": [{"no_id": 1}], "edges": []}, "graph")

    def test_tree_outline(self) -> None:
        tree = {"root": {"leaf": 1, "list": ["x", {"deep": 2}]}}
        text = render(tree, "tree").decode("utf-8")
        assert "- **root**" in text
        assert "  - leaf: 1" in text
        assert "- [1]" in text and "deep: 2" in text

    def test_tree_non_mapping_refused(self) -> None:
        with pytest.raises(ExportError):
            render([1, 2, 3], "tree")

    def test_every_declared_format_renders_something(self, frame: pd.DataFrame) -> None:
        """FR-902 coverage: one render per roster entry."""
        from localdata_mcp.nexus.export.renderers import RENDERERS

        tabular_ok = {"csv", "parquet", "arrow", "json", "excel", "markdown"}
        assert set(RENDERERS) == tabular_ok | {"schema", "graph", "tree"}
        for name in tabular_ok:
            assert render(frame, name)
