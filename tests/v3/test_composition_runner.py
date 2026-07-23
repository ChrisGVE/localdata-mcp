"""tests/v3/test_composition_runner.py — E11.1 execution runtime.

The stage_runner subpackage end-to-end over a private registry of
controlled tools: upstream handoff through the injection channel,
fan-out multi-leaf assembly, per-stage sentinel enforcement, named
stage failures with strict no-partial semantics, the handoff contract,
and NFR-105's one-ledger-entry accounting released on every exit
path."""

from __future__ import annotations

import os
from typing import Any, Iterator

import pytest

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.explore.addressing import resolve_frame
from localdata_mcp.nexus.chokepoint.guard import Chokepoint, GuardedExecutionError
from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.contract.registry import ToolRegistry, default_registry
from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.process.composition.stage_runner.sequence import run_pipeline

_RAN: list[str] = []


def _registry() -> ToolRegistry:
    target = ToolRegistry()

    @tool_spec(
        name="fake_source",
        summary="Emit a fixed relation.",
        params=(),
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.TABULAR,
        registry=target,
    )
    def fake_source() -> dict[str, Any]:
        _RAN.append("fake_source")
        return {
            "columns": ["value", "group"],
            "rows": [[1.0, "a"], [2.0, "b"], [3.0, "a"]],
        }

    @tool_spec(
        name="fake_clean",
        summary="Pass the addressed relation through, tagging it.",
        params=(Param("path", str, "source path", required=False),),
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.TABULAR,
        registry=target,
    )
    def fake_clean(path: "str | None" = None) -> dict[str, Any]:
        _RAN.append("fake_clean")
        frame, source = resolve_frame(None, path, None, None)
        return {
            "source": source,
            "columns": [str(c) for c in frame.columns],
            "rows": frame.to_numpy().tolist(),
        }

    @tool_spec(
        name="fake_stat",
        summary="Mean of the addressed relation's first column.",
        params=(),
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        registry=target,
    )
    def fake_stat() -> dict[str, Any]:
        _RAN.append("fake_stat")
        frame, source = resolve_frame(None, None, None, None)
        first = frame.columns[0]
        return {
            "source": source,
            "column": str(first),
            "mean": float(frame[first].mean()),
            "n": int(len(frame)),
        }

    @tool_spec(
        name="fake_vector",
        summary="Emit a forecast vector.",
        params=(),
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.VECTOR,
        registry=target,
    )
    def fake_vector() -> dict[str, Any]:
        _RAN.append("fake_vector")
        frame, _ = resolve_frame(None, None, None, None)
        return {"forecast": [float(v) for v in frame[frame.columns[0]]]}

    @tool_spec(
        name="fake_broken",
        summary="Raise mid-chain.",
        params=(),
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.TABULAR,
        registry=target,
    )
    def fake_broken() -> dict[str, Any]:
        _RAN.append("fake_broken")
        raise ValueError("deliberate stage explosion")

    @tool_spec(
        name="fake_degenerate",
        summary="Emit a non-converged result.",
        params=(),
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.TABULAR,
        registry=target,
    )
    def fake_degenerate() -> dict[str, Any]:
        _RAN.append("fake_degenerate")
        return {"converged": False, "columns": ["x"], "rows": [[1.0]]}

    @tool_spec(
        name="fake_junk_tabular",
        summary="Declare TABULAR, emit no relation.",
        params=(),
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.TABULAR,
        registry=target,
    )
    def fake_junk_tabular() -> dict[str, Any]:
        _RAN.append("fake_junk_tabular")
        return {"note": "no columns or rows here"}

    return target


@pytest.fixture()
def bench() -> Iterator[ToolRegistry]:
    registry = _registry()
    guard = Chokepoint.boot(ConfigModel(), environ=dict(os.environ))
    configure_shaping(ConfigModel(), registry)
    runtime.configure_ingest(guard)
    _RAN.clear()
    yield registry
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _chain(*stages: dict[str, Any]) -> list[dict[str, Any]]:
    return list(stages)


class TestLinearChain:
    def test_end_to_end_result_shape(self, bench: ToolRegistry) -> None:
        result = run_pipeline(
            _chain(
                {"stage": "load", "tool": "fake_source"},
                {"stage": "clean", "tool": "fake_clean", "depends_on": ["load"]},
                {"stage": "stat", "tool": "fake_stat", "depends_on": ["clean"]},
            ),
            registry=bench,
        )
        assert "3 stage(s) executed" in result["summary"]
        assert set(result["results"]) == {"stat"}
        envelope = result["results"]["stat"]
        assert envelope["error"] is None
        assert envelope["inline"] is not None
        assert envelope["data"]["mean"] == pytest.approx(2.0)
        assert envelope["data"]["source"] == "pipeline:clean"
        chain = result["provenance"]
        assert chain["execution_order"] == ["load", "clean", "stat"]
        assert chain["terminal_stages"] == ["stat"]
        assert [entry["tool"] for entry in chain["stages"]] == [
            "fake_source",
            "fake_clean",
            "fake_stat",
        ]

    def test_upstream_data_actually_flows(self, bench: ToolRegistry) -> None:
        result = run_pipeline(
            _chain(
                {"stage": "load", "tool": "fake_source"},
                {"stage": "stat", "tool": "fake_stat", "depends_on": ["load"]},
            ),
            registry=bench,
        )
        data = result["results"]["stat"]["data"]
        assert data["n"] == 3
        assert data["column"] == "value"
        assert data["source"] == "pipeline:load"

    def test_vector_handoff_becomes_value_column(self, bench: ToolRegistry) -> None:
        result = run_pipeline(
            _chain(
                {"stage": "load", "tool": "fake_source"},
                {"stage": "vec", "tool": "fake_vector", "depends_on": ["load"]},
                {"stage": "stat", "tool": "fake_stat", "depends_on": ["vec"]},
            ),
            registry=bench,
        )
        data = result["results"]["stat"]["data"]
        assert data["column"] == "value"
        assert data["mean"] == pytest.approx(2.0)


class TestFanOut:
    def test_two_leaves_one_provenance(self, bench: ToolRegistry) -> None:
        result = run_pipeline(
            _chain(
                {"stage": "load", "tool": "fake_source"},
                {"stage": "left", "tool": "fake_stat", "depends_on": ["load"]},
                {"stage": "right", "tool": "fake_vector", "depends_on": ["load"]},
            ),
            registry=bench,
        )
        assert set(result["results"]) == {"left", "right"}
        for envelope in result["results"].values():
            assert envelope["error"] is None
        assert result["provenance"]["terminal_stages"] == ["left", "right"]

    def test_siblings_each_get_the_upstream_relation(self, bench: ToolRegistry) -> None:
        result = run_pipeline(
            _chain(
                {"stage": "load", "tool": "fake_source"},
                {"stage": "left", "tool": "fake_stat", "depends_on": ["load"]},
                {"stage": "right", "tool": "fake_stat", "depends_on": ["load"]},
            ),
            registry=bench,
        )
        left = result["results"]["left"]["data"]
        right = result["results"]["right"]["data"]
        assert left["n"] == right["n"] == 3


class TestFailureSemantics:
    def test_stage_failure_names_stage_and_stops_the_chain(
        self, bench: ToolRegistry
    ) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            run_pipeline(
                _chain(
                    {"stage": "load", "tool": "fake_source"},
                    {"stage": "boom", "tool": "fake_broken", "depends_on": ["load"]},
                    {"stage": "stat", "tool": "fake_stat", "depends_on": ["boom"]},
                ),
                registry=bench,
            )
        message = failure.value.structured.message
        assert "'boom'" in message
        assert "fake_broken" in message
        assert "fake_stat" not in _RAN  # strict: nothing after the failure

    def test_degenerate_output_is_a_named_stage_failure(
        self, bench: ToolRegistry
    ) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            run_pipeline(
                _chain(
                    {"stage": "load", "tool": "fake_source"},
                    {
                        "stage": "sick",
                        "tool": "fake_degenerate",
                        "depends_on": ["load"],
                    },
                    {"stage": "stat", "tool": "fake_stat", "depends_on": ["sick"]},
                ),
                registry=bench,
            )
        assert "'sick'" in failure.value.structured.message
        assert "fake_stat" not in _RAN

    def test_handoff_contract_violation_is_engine_level(
        self, bench: ToolRegistry
    ) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            run_pipeline(
                _chain(
                    {"stage": "load", "tool": "fake_source"},
                    {
                        "stage": "junk",
                        "tool": "fake_junk_tabular",
                        "depends_on": ["load"],
                    },
                    {"stage": "stat", "tool": "fake_stat", "depends_on": ["junk"]},
                ),
                registry=bench,
            )
        message = failure.value.structured.message
        assert "'junk'" in message
        assert "columns" in message

    def test_rejected_spec_runs_nothing(self, bench: ToolRegistry) -> None:
        with pytest.raises(GuardedExecutionError):
            run_pipeline(
                _chain(
                    {"stage": "load", "tool": "fake_source"},
                    {"stage": "stat", "tool": "no_such_tool", "depends_on": ["load"]},
                ),
                registry=bench,
            )
        assert _RAN == []  # FR-606: no partial pipeline run


class TestMemoryAccounting:
    def test_ledger_is_released_after_success(self, bench: ToolRegistry) -> None:
        run_pipeline(
            _chain(
                {"stage": "load", "tool": "fake_source"},
                {"stage": "stat", "tool": "fake_stat", "depends_on": ["load"]},
            ),
            registry=bench,
        )
        assert runtime.chokepoint()._bounds.live_residency() == 0

    def test_ledger_is_released_after_failure(self, bench: ToolRegistry) -> None:
        with pytest.raises(GuardedExecutionError):
            run_pipeline(
                _chain(
                    {"stage": "load", "tool": "fake_source"},
                    {"stage": "boom", "tool": "fake_broken", "depends_on": ["load"]},
                ),
                registry=bench,
            )
        assert runtime.chokepoint()._bounds.live_residency() == 0
