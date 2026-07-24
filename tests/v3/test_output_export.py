"""tests/v3/test_output_export.py — E13 export_result tool + FR-901/902/903.

The exit-gate assertions for the Output capability (output/tools.py):
the three-way exactly-one source contract (inline `source=`, a
`stream_id=` drained to the file, the composition-leaf frame the engine
injects), per-format export coverage over the whole FR-902 roster, the
three ordered write guards surfaced through the tool (NX-6 containment,
the NFR-115 explicit-overwrite refusal, NFR-111 atomic rename), the
CWE-1236 neutralization reaching the file, FR-903 round-trip fidelity
per applicable format, and the FR-901 single-export-surface static gate
(`export_to_file` invoked from exactly one tool-package module).
"""

from __future__ import annotations

import ast
import base64
import os
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.explore.addressing import pipeline_input
from localdata_mcp.ingest.connectors.file.readers import read_path, resolve_format
from localdata_mcp.nexus.chokepoint.guard import (
    Chokepoint,
    GuardedExecutionError,
    Result,
)
from localdata_mcp.nexus.config.models import (
    ConfigModel,
    QueryConfig,
    ResponseConfig,
    SecurityConfig,
)
from localdata_mcp.nexus.export.interface import (
    OverwriteRefusedError,
    supported_formats,
)
from localdata_mcp.nexus.gated_tree import SRC_ROOT, iter_v3_sources
from localdata_mcp.output.tools import export_result


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Path]:
    """A booted guard contained to tmp, with the inline budget tiny so a
    handful of rows becomes a genuine stream (the stream_id path)."""
    config = ConfigModel(
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
        response=ResponseConfig(inline_max_rows=2),
        query=QueryConfig(default_chunk_size=2),
    )
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    runtime.configure_ingest(guard)
    yield tmp_path
    runtime._CHOKEPOINT = None
    guard.shutdown()


# -- source resolution: the exactly-one contract ----------------------


class TestSourceContract:
    def test_inline_records_export(self, bench: Path) -> None:
        target = bench / "a.csv"
        answer = export_result(
            format="csv",
            path=str(target),
            source=[{"x": 1}, {"x": 2}],
        )
        assert answer["source"] == "inline"
        assert answer["format"] == "csv"
        assert Path(answer["path"]) == target.resolve()
        assert answer["bytes_written"] == target.stat().st_size
        assert target.read_text().startswith("x\n1\n2")

    def test_composition_leaf_uses_injected_frame(self, bench: Path) -> None:
        target = bench / "leaf.json"
        with pipeline_input(pd.DataFrame({"a": [1, 2]}), "clean:step"):
            answer = export_result(format="json", path=str(target))
        assert answer["source"] == "clean:step"
        assert target.exists()

    def test_stream_id_drains_to_file(self, bench: Path) -> None:
        # Five rows past the 2-row inline budget becomes a real stream.
        result = Result(
            columns=("n",),
            rows=tuple((value,) for value in range(5)),
            category="read",
        )
        opened = runtime.chokepoint().serve_result(result, "probe")
        stream_id = opened.stream_id  # type: ignore[union-attr]
        target = bench / "drained.csv"
        answer = export_result(format="csv", path=str(target), stream_id=stream_id)
        assert answer["source"] == f"stream:{stream_id}"
        reloaded = pd.read_csv(target)
        assert list(reloaded["n"]) == [0, 1, 2, 3, 4]

    def test_no_source_is_refused(self, bench: Path) -> None:
        with pytest.raises(GuardedExecutionError) as caught:
            export_result(format="csv", path=str(bench / "z.csv"))
        assert "No export source" in str(caught.value.structured.message)

    def test_two_sources_are_refused(self, bench: Path) -> None:
        with pytest.raises(GuardedExecutionError) as caught:
            export_result(
                format="csv",
                path=str(bench / "z.csv"),
                source=[{"x": 1}],
                stream_id="file:probe:deadbeef",
            )
        assert "Both source= and stream_id=" in str(caught.value.structured.message)


# -- FR-902: full-format coverage -------------------------------------

_TABULAR_FORMATS = ("csv", "parquet", "arrow", "json", "excel", "markdown")


class TestFormatCoverage:
    @pytest.mark.parametrize("fmt", _TABULAR_FORMATS)
    def test_tabular_format_writes_a_file(self, bench: Path, fmt: str) -> None:
        target = bench / f"out.{fmt}"
        answer = export_result(
            format=fmt,
            path=str(target),
            source=[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
        )
        assert target.stat().st_size > 0
        assert answer["bytes_written"] == target.stat().st_size

    def test_schema_mapping(self, bench: Path) -> None:
        target = bench / "schema.md"
        export_result(
            format="schema",
            path=str(target),
            source={
                "t": {
                    "columns": [
                        {
                            "name": "id",
                            "type": "int",
                            "nullable": False,
                            "primary_key": True,
                        }
                    ]
                }
            },
        )
        assert "## t" in target.read_text()

    def test_graph_mapping(self, bench: Path) -> None:
        target = bench / "graph.md"
        export_result(
            format="graph",
            path=str(target),
            source={
                "nodes": [{"id": "a"}, {"id": "b"}],
                "edges": [{"source": "a", "target": "b"}],
            },
        )
        assert "2 nodes, 1 edges" in target.read_text()

    def test_tree_mapping(self, bench: Path) -> None:
        target = bench / "tree.md"
        export_result(
            format="tree",
            path=str(target),
            source={"root": {"leaf": 1}},
        )
        assert "root" in target.read_text()

    def test_svg_from_chart_artifact_envelope(self, bench: Path) -> None:
        target = bench / "chart.svg"
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><g/></svg>'
        envelope = {
            "artifact": {
                "media_type": "image/svg+xml",
                "encoding": "utf-8",
                "content": svg,
            }
        }
        export_result(format="svg", path=str(target), source=envelope)
        assert target.read_text().startswith("<svg")

    def test_png_from_base64_artifact_envelope(self, bench: Path) -> None:
        target = bench / "chart.png"
        raw = b"\x89PNG\r\n\x1a\n" + b"payload-bytes"
        envelope = {
            "media_type": "image/png",
            "encoding": "base64",
            "content": base64.b64encode(raw).decode("ascii"),
        }
        export_result(format="png", path=str(target), source=envelope)
        assert target.read_bytes() == raw

    def test_unknown_format_is_refused(self, bench: Path) -> None:
        with pytest.raises(GuardedExecutionError) as caught:
            export_result(format="xlsxx", path=str(bench / "o.x"), source=[{"a": 1}])
        assert "Unknown export format" in str(caught.value.structured.message)

    def test_shape_mismatch_is_refused(self, bench: Path) -> None:
        # A tabular frame handed to svg — the renderer refuses; the tool
        # surfaces it as a structured shape refusal, not a raw crash.
        with pytest.raises(GuardedExecutionError):
            export_result(format="svg", path=str(bench / "bad.svg"), source=[{"a": 1}])

    def test_every_roster_format_is_reachable(self) -> None:
        # The tool advertises exactly NX-8's registered roster (FR-902).
        assert set(supported_formats()) == {
            "csv",
            "parquet",
            "arrow",
            "json",
            "excel",
            "markdown",
            "schema",
            "graph",
            "tree",
            "svg",
            "png",
        }


# -- the three ordered write guards, through the tool -----------------


class TestWriteGuards:
    def test_overwrite_is_refused_without_the_flag(self, bench: Path) -> None:
        target = bench / "once.csv"
        export_result(format="csv", path=str(target), source=[{"a": 1}])
        with pytest.raises(OverwriteRefusedError):
            export_result(format="csv", path=str(target), source=[{"a": 2}])

    def test_overwrite_true_replaces(self, bench: Path) -> None:
        target = bench / "twice.csv"
        export_result(format="csv", path=str(target), source=[{"a": 1}])
        export_result(format="csv", path=str(target), source=[{"a": 2}], overwrite=True)
        assert "2" in target.read_text()

    def test_containment_refuses_outside_allowed_paths(
        self, bench: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside = tmp_path_factory.mktemp("outside") / "escape.csv"
        with pytest.raises(Exception):  # PathRefusedError family
            export_result(format="csv", path=str(outside), source=[{"a": 1}])

    def test_formula_injection_neutralized_in_file(self, bench: Path) -> None:
        target = bench / "inj.csv"
        export_result(
            format="csv",
            path=str(target),
            source=[{"c": "=1+2"}, {"c": "safe"}],
        )
        # CWE-1236: the leading '=' cell is quote-prefixed on disk.
        assert "'=1+2" in target.read_text()


# -- FR-903: round-trip fidelity --------------------------------------

_FRAME = pd.DataFrame({"i": [1, 2, 3], "s": ["a", "b", "c"], "f": [1.5, 2.5, 3.5]})


def _admit_all(estimated_bytes: int) -> None:
    """A no-op memory-admission seam for the round-trip reloads: these
    reload tiny exported fixtures and only assert fidelity, not the
    CR-005 gate (which test_ingest_file covers)."""


class TestRoundTrip:
    @pytest.mark.parametrize("fmt", ["parquet", "arrow"])
    def test_type_preserving_round_trip(self, bench: Path, fmt: str) -> None:
        target = bench / f"rt.{fmt}"
        export_result(
            format=fmt,
            path=str(target),
            source=_FRAME.to_dict(orient="records"),
        )
        reloaded = read_path(target, resolve_format(target, fmt), _admit_all)
        pd.testing.assert_frame_equal(reloaded, _FRAME)

    def test_csv_round_trip_values_documented_lossy(self, bench: Path) -> None:
        # CSV is text: values round-trip, dtypes are re-inferred on read
        # (the documented-lossy class alongside markdown, FR-903).
        target = bench / "rt.csv"
        export_result(
            format="csv",
            path=str(target),
            source=_FRAME.to_dict(orient="records"),
        )
        reloaded = read_path(target, resolve_format(target, "csv"), _admit_all)
        assert reloaded.to_dict(orient="records") == _FRAME.to_dict(orient="records")

    def test_json_round_trip(self, bench: Path) -> None:
        target = bench / "rt.json"
        export_result(
            format="json",
            path=str(target),
            source=_FRAME.to_dict(orient="records"),
        )
        reloaded = read_path(target, resolve_format(target, "json"), _admit_all)
        assert reloaded.to_dict(orient="records") == _FRAME.to_dict(orient="records")


# -- FR-901: the single export surface (static gate) ------------------


def _invokes(name: str, path: Path) -> bool:
    """True iff `path` references the identifier `name` (a call target
    or an imported symbol) — the AST-level 'this module writes files'
    signal for the export seam."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
        if isinstance(node, ast.alias) and node.name == name:
            return True
    return False


class TestSingleExportSurface:
    """FR-901: every file an v3 tool writes goes through NX-8's one
    `export_to_file` seam, and exactly one tool-package module invokes
    it — no second export surface, no tool with its own file output."""

    _OUTPUT_TOOL = SRC_ROOT / "output" / "tools.py"
    _TOOL_PACKAGES = ("ingest", "explore", "process", "visualize", "output")

    def _tool_sources(self) -> Iterator[Path]:
        for path in iter_v3_sources():
            if any(
                path.is_relative_to(SRC_ROOT / package)
                for package in self._TOOL_PACKAGES
            ):
                yield path

    def test_export_to_file_invoked_only_by_the_output_tool(self) -> None:
        callers = [
            str(path)
            for path in self._tool_sources()
            if _invokes("export_to_file", path)
        ]
        assert callers == [str(self._OUTPUT_TOOL)], callers

    def test_output_tool_is_the_single_write_home(self) -> None:
        # The tool that owns the write seam is registered and present.
        assert self._OUTPUT_TOOL.exists()
        assert _invokes("export_to_file", self._OUTPUT_TOOL)
