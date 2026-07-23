"""tests/v3/test_composition_dag_spec.py — E11.1 FR-606 validation.

Every rejection is pre-execution, structured (NX-3), and NAMES the
offending stage or depends_on pair — the acceptance FR-606 spells out.
Specs register into a private ToolRegistry so the checks run against
controlled shapes, not the live tool population."""

from __future__ import annotations

from typing import Any

import pytest

from localdata_mcp.nexus.chokepoint.guard import GuardedExecutionError
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import Param, ToolSpec, TypeShape
from localdata_mcp.process.composition.dag_spec import (
    ValidatedDag,
    validate_dag_spec,
)

_MAX_LENGTH = 3 + 1  # mirrors the S8 row-14 default without restating it


def _spec(
    name: str,
    input_shape: TypeShape,
    output_shape: TypeShape,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        summary=f"test tool {name}",
        params=(Param("column", str, "a column", required=False),),
        input_shape=input_shape,
        output_shape=output_shape,
    )


@pytest.fixture()
def registry() -> ToolRegistry:
    target = ToolRegistry()
    target.register(_spec("source_tool", TypeShape.NONE, TypeShape.TABULAR))
    target.register(_spec("clean_tool", TypeShape.TABULAR, TypeShape.TABULAR))
    target.register(_spec("stat_tool", TypeShape.TABULAR, TypeShape.SCALAR))
    target.register(_spec("vector_tool", TypeShape.TABULAR, TypeShape.VECTOR))
    target.register(_spec("sink_tool", TypeShape.TABULAR, TypeShape.NONE))
    target.register(_spec("dynamic_tool", TypeShape.DYNAMIC, TypeShape.DYNAMIC))
    return target


def _message(failure: pytest.ExceptionInfo[GuardedExecutionError]) -> str:
    return failure.value.structured.message


class TestValidChains:
    def test_linear_chain_validates(self, registry: ToolRegistry) -> None:
        dag = validate_dag_spec(
            [
                {"stage": "clean", "tool": "clean_tool", "params": {"path": "x.csv"}},
                {"stage": "stat", "tool": "stat_tool", "depends_on": ["clean"]},
            ],
            registry,
            _MAX_LENGTH,
        )
        assert isinstance(dag, ValidatedDag)
        assert dag.order == ("clean", "stat")
        assert dag.leaves == ("stat",)
        assert dag.stage_named("stat").spec.name == "stat_tool"

    def test_fan_out_multi_leaf(self, registry: ToolRegistry) -> None:
        dag = validate_dag_spec(
            [
                {"stage": "clean", "tool": "clean_tool", "params": {"path": "x.csv"}},
                {"stage": "left", "tool": "stat_tool", "depends_on": ["clean"]},
                {"stage": "right", "tool": "vector_tool", "depends_on": ["clean"]},
            ],
            registry,
            _MAX_LENGTH,
        )
        assert dag.leaves == ("left", "right")
        assert ("left", "right") in dag.groups

    def test_source_tool_chain_initial(self, registry: ToolRegistry) -> None:
        dag = validate_dag_spec(
            [
                {"stage": "load", "tool": "source_tool"},
                {"stage": "stat", "tool": "stat_tool", "depends_on": ["load"]},
            ],
            registry,
            _MAX_LENGTH,
        )
        assert dag.order == ("load", "stat")


class TestStructuralRejections:
    @pytest.mark.parametrize("bad", [None, [], "chain", 7])
    def test_non_list_or_empty(self, registry: ToolRegistry, bad: Any) -> None:
        with pytest.raises(GuardedExecutionError):
            validate_dag_spec(bad, registry, _MAX_LENGTH)

    def test_length_guard_names_the_bound(self, registry: ToolRegistry) -> None:
        chain = [{"stage": "clean0", "tool": "clean_tool", "params": {"path": "x"}}]
        chain += [
            {
                "stage": f"clean{i}",
                "tool": "clean_tool",
                "depends_on": [f"clean{i - 1}"],
            }
            for i in range(1, _MAX_LENGTH + 1)
        ]
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(chain, registry, _MAX_LENGTH)
        assert "max_pipeline_length" in _message(failure)
        assert str(_MAX_LENGTH) in _message(failure)

    def test_duplicate_stage_name(self, registry: ToolRegistry) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(
                [
                    {"stage": "twin", "tool": "clean_tool"},
                    {"stage": "twin", "tool": "stat_tool"},
                ],
                registry,
                _MAX_LENGTH,
            )
        assert "twin" in _message(failure)

    def test_unknown_tool_names_the_stage(self, registry: ToolRegistry) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(
                [{"stage": "ghost", "tool": "no_such_tool"}], registry, _MAX_LENGTH
            )
        assert "ghost" in _message(failure)
        assert "no_such_tool" in _message(failure)

    def test_dynamic_tool_is_barred_as_stage(self, registry: ToolRegistry) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(
                [{"stage": "nest", "tool": "dynamic_tool"}], registry, _MAX_LENGTH
            )
        assert "DYNAMIC" in _message(failure)

    def test_undeclared_dependency_named(self, registry: ToolRegistry) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(
                [{"stage": "stat", "tool": "stat_tool", "depends_on": ["phantom"]}],
                registry,
                _MAX_LENGTH,
            )
        assert "phantom" in _message(failure)

    def test_cycle_is_rejected(self, registry: ToolRegistry) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(
                [
                    {"stage": "a", "tool": "clean_tool", "depends_on": ["b"]},
                    {"stage": "b", "tool": "clean_tool", "depends_on": ["a"]},
                ],
                registry,
                _MAX_LENGTH,
            )
        assert "circular" in _message(failure)


class TestTopologyRejections:
    def test_fan_in_is_rejected_naming_the_pair(self, registry: ToolRegistry) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(
                [
                    {"stage": "one", "tool": "clean_tool", "params": {"path": "x"}},
                    {"stage": "two", "tool": "clean_tool", "params": {"path": "y"}},
                    {
                        "stage": "merge",
                        "tool": "stat_tool",
                        "depends_on": ["one", "two"],
                    },
                ],
                registry,
                _MAX_LENGTH,
            )
        assert "merge" in _message(failure)
        assert "fan-out" in _message(failure)

    def test_source_accepts_no_inbound_edge(self, registry: ToolRegistry) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(
                [
                    {"stage": "clean", "tool": "clean_tool", "params": {"path": "x"}},
                    {"stage": "load", "tool": "source_tool", "depends_on": ["clean"]},
                ],
                registry,
                _MAX_LENGTH,
            )
        assert "load" in _message(failure)
        assert "source" in _message(failure)

    def test_sink_accepts_no_downstream_edge(self, registry: ToolRegistry) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(
                [
                    {"stage": "out", "tool": "sink_tool", "params": {"path": "x"}},
                    {"stage": "stat", "tool": "stat_tool", "depends_on": ["out"]},
                ],
                registry,
                _MAX_LENGTH,
            )
        assert "out" in _message(failure)
        assert "sink" in _message(failure)

    def test_dependent_stage_addressing_its_own_source(
        self, registry: ToolRegistry
    ) -> None:
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(
                [
                    {"stage": "clean", "tool": "clean_tool", "params": {"path": "x"}},
                    {
                        "stage": "stat",
                        "tool": "stat_tool",
                        "params": {"endpoint": "db"},
                        "depends_on": ["clean"],
                    },
                ],
                registry,
                _MAX_LENGTH,
            )
        assert "stat" in _message(failure)
        assert "endpoint" in _message(failure)


class TestEdgeCompatibility:
    def test_illegal_edge_names_pair_and_shapes(self, registry: ToolRegistry) -> None:
        # SCALAR feeds only SCALAR consumers; stat -> clean is illegal.
        with pytest.raises(GuardedExecutionError) as failure:
            validate_dag_spec(
                [
                    {"stage": "stat", "tool": "stat_tool", "params": {"path": "x"}},
                    {"stage": "clean", "tool": "clean_tool", "depends_on": ["stat"]},
                ],
                registry,
                _MAX_LENGTH,
            )
        message = _message(failure)
        assert "'stat' -> 'clean'" in message
        assert "scalar" in message
        assert "tabular" in message

    def test_vector_feeds_tabular(self, registry: ToolRegistry) -> None:
        dag = validate_dag_spec(
            [
                {"stage": "vec", "tool": "vector_tool", "params": {"path": "x"}},
                {"stage": "stat", "tool": "stat_tool", "depends_on": ["vec"]},
            ],
            registry,
            _MAX_LENGTH,
        )
        assert dag.order == ("vec", "stat")

    def test_no_partial_run_semantics(self, registry: ToolRegistry) -> None:
        """Validation is pure: a rejected spec ran nothing (the specs
        here have no side effects to observe, so the assertion is that
        rejection comes from validate alone — before any executor is
        even constructed)."""
        with pytest.raises(GuardedExecutionError):
            validate_dag_spec(
                [
                    {"stage": "clean", "tool": "clean_tool", "params": {"path": "x"}},
                    {"stage": "bad", "tool": "clean_tool", "depends_on": ["missing"]},
                ],
                registry,
                _MAX_LENGTH,
            )
