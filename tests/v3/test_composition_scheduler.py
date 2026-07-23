"""tests/v3/test_composition_scheduler.py — E11.1 scheduler (FR-602).

The harvested Kahn ordering, fan-out grouping, and leaf identification
over declared `stage -> upstream` mappings — including FR-602's own
acceptance shape (>= 3 stages, non-trivial dependencies, order
consistent with a topological sort)."""

from __future__ import annotations

import pytest

from localdata_mcp.process.composition.scheduler import (
    CompositionCycleError,
    execution_order,
    fanout_groups,
    terminal_stages,
)


def _is_topological(order: tuple[str, ...], deps: dict[str, tuple[str, ...]]) -> bool:
    position = {stage: index for index, stage in enumerate(order)}
    return all(
        position[upstream] < position[stage]
        for stage, upstream_set in deps.items()
        for upstream in upstream_set
    )


class TestExecutionOrder:
    def test_linear_chain_keeps_declared_order(self) -> None:
        deps = {"clean": (), "profile": ("clean",), "chart": ("profile",)}
        assert execution_order(deps) == ("clean", "profile", "chart")

    def test_fan_out_order_is_topological(self) -> None:
        # FR-602 acceptance: >= 3 stages, non-trivial shape.
        deps = {
            "load": (),
            "left": ("load",),
            "right": ("load",),
            "deep": ("right",),
        }
        order = execution_order(deps)
        assert set(order) == set(deps)
        assert _is_topological(order, deps)

    def test_declaration_order_breaks_ties(self) -> None:
        deps = {"b": (), "a": ()}
        assert execution_order(deps) == ("b", "a")

    def test_cycle_is_refused_naming_stages(self) -> None:
        deps = {"x": ("y",), "y": ("x",)}
        with pytest.raises(CompositionCycleError, match="'x'.*'y'|circular"):
            execution_order(deps)

    def test_self_loop_is_a_cycle(self) -> None:
        with pytest.raises(CompositionCycleError):
            execution_order({"solo": ("solo",)})

    def test_empty_dag(self) -> None:
        assert execution_order({}) == ()


class TestFanoutGroups:
    def test_same_upstream_set_groups_together(self) -> None:
        deps = {
            "load": (),
            "left": ("load",),
            "right": ("load",),
            "tail": ("left",),
        }
        order = execution_order(deps)
        groups = fanout_groups(order, deps)
        assert ("left", "right") in groups
        assert ("load",) in groups
        assert ("tail",) in groups

    def test_linear_chain_is_singleton_groups(self) -> None:
        deps = {"a": (), "b": ("a",), "c": ("b",)}
        order = execution_order(deps)
        assert fanout_groups(order, deps) == (("a",), ("b",), ("c",))

    def test_groups_partition_the_stages(self) -> None:
        deps = {
            "src": (),
            "one": ("src",),
            "two": ("src",),
            "three": ("src",),
        }
        order = execution_order(deps)
        groups = fanout_groups(order, deps)
        flattened = [stage for group in groups for stage in group]
        assert sorted(flattened) == sorted(deps)
        assert len(flattened) == len(set(flattened))


class TestTerminalStages:
    def test_linear_chain_has_one_leaf(self) -> None:
        deps = {"a": (), "b": ("a",), "c": ("b",)}
        assert terminal_stages(deps) == ("c",)

    def test_fan_out_has_multiple_leaves(self) -> None:
        deps = {"load": (), "left": ("load",), "right": ("load",)}
        assert terminal_stages(deps) == ("left", "right")

    def test_single_stage_is_its_own_leaf(self) -> None:
        assert terminal_stages({"only": ()}) == ("only",)
