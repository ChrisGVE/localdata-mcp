"""localdata_mcp/process/composition/dag_spec.py — the dag_spec model + FR-606 validation.

The `{stage, tool, params, depends_on}` spec model (harvesting
`PipelineComposer.add_pipeline`/`resolve_dependencies`'s shape) and the
whole pre-execution validation FR-606 demands: every check runs before
any stage executes, and an incompatible chain is a structured NX-3
rejection that NAMES the offending stage or depends_on pair — never a
mid-pipeline crash, never a partial run. The compatibility verdict is
NX-1's: tool shapes come from the live ToolSpec registry and edge
legality from the declared adjacency table (`contract/compatibility.py`
— the import-graph gate grants this package, alone among tool packages,
that pair of NX-1 seams, §6.3). Topology at launch: linear chains with
fan-out, no fan-in — at most ONE upstream per stage (§6.3). Neighbors:
scheduler.py orders the validated DAG; stage_runner/ executes it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from localdata_mcp.nexus.chokepoint.guard import GuardedExecutionError
from localdata_mcp.nexus.contract.compatibility import may_feed
from localdata_mcp.nexus.contract.errors import UnknownToolError
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import ToolSpec, TypeShape
from localdata_mcp.nexus.error.model import ErrorType, StructuredError

from .scheduler import (
    CompositionCycleError,
    execution_order,
    fanout_groups,
    terminal_stages,
)

# The X-2 source-selector parameters a DEPENDENT stage must not carry:
# its input is its upstream stage's output, not a freshly addressed
# source. Only the UNAMBIGUOUS selectors are listed — `endpoint`/`path`
# are the two primary selectors the injection channel keys on, and
# `table`/`target` are the secondary-slot selectors. `query` is
# deliberately EXCLUDED: it is overloaded — a SQL source for the raw
# addressing tools but the search PATTERN for search_data (whose own
# source slot is `target`) — so name alone cannot tell a re-addressing
# mistake from a legitimate search term. Guarding the four unambiguous
# names catches the real "I addressed my own source" error without the
# false positive.
_SOURCE_PARAMS = ("endpoint", "path", "table", "target")

_SUGGESTION = (
    "Fix the dag_spec and resubmit: each entry is {stage, tool, params, "
    "depends_on}; tools compose only where the declared type-shapes are "
    "adjacency-legal (a tool's composition_metadata names its applicable "
    "next steps). No stage ran."
)


def rejection(detail: str) -> GuardedExecutionError:
    """One FR-606 structured rejection — pre-execution, no partial run."""
    return GuardedExecutionError(
        StructuredError(
            error_type=ErrorType.DATA_VALIDATION,
            message=detail,
            suggestion=_SUGGESTION,
            retryable=False,
        )
    )


@dataclass(frozen=True)
class StageSpec:
    """One validated stage: its name, resolved ToolSpec, caller params,
    and its (at most one) upstream stage."""

    stage: str
    spec: ToolSpec
    params: Mapping[str, Any]
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedDag:
    """The whole validated chain, scheduling facts included."""

    stages: tuple[StageSpec, ...]
    order: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    leaves: tuple[str, ...]

    def stage_named(self, name: str) -> StageSpec:
        for entry in self.stages:
            if entry.stage == name:
                return entry
        raise KeyError(name)


def validate_dag_spec(
    dag_spec: Any, registry: ToolRegistry, max_length: int
) -> ValidatedDag:
    """FR-606's whole pre-execution validation, in declaration order."""
    entries = _well_formed_entries(dag_spec, max_length)
    stages = _resolved_stages(entries, registry)
    by_name = {entry.stage: entry for entry in stages}
    _check_endpoint_rules(stages, by_name)
    _check_edges(stages, by_name)
    dependencies = {entry.stage: entry.depends_on for entry in stages}
    try:
        order = execution_order(dependencies)
    except CompositionCycleError as failure:
        raise rejection(str(failure)) from None
    return ValidatedDag(
        stages=tuple(stages),
        order=order,
        groups=fanout_groups(order, dependencies),
        leaves=terminal_stages(dependencies),
    )


def _well_formed_entries(dag_spec: Any, max_length: int) -> "list[dict[str, Any]]":
    """Structural screen: a non-empty ordered list of stage mappings
    within the composition length bound."""
    if not isinstance(dag_spec, (list, tuple)) or not dag_spec:
        raise rejection(
            "dag_spec must be a non-empty ordered list of "
            "{stage, tool, params, depends_on} entries"
        )
    if len(dag_spec) > max_length:
        raise rejection(
            f"dag_spec declares {len(dag_spec)} stages, over the "
            f"composition.max_pipeline_length bound of {max_length}"
        )
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, raw in enumerate(dag_spec):
        if not isinstance(raw, Mapping):
            raise rejection(f"dag_spec entry {position} is not a mapping")
        stage = raw.get("stage")
        if not isinstance(stage, str) or not stage:
            raise rejection(f"dag_spec entry {position} lacks a non-empty 'stage' name")
        if stage in seen:
            raise rejection(f"stage {stage!r} is declared twice")
        seen.add(stage)
        tool = raw.get("tool")
        if not isinstance(tool, str) or not tool:
            raise rejection(f"stage {stage!r} lacks a non-empty 'tool' name")
        entries.append(dict(raw))
    return entries


def _resolved_stages(
    entries: "list[dict[str, Any]]", registry: ToolRegistry
) -> "list[StageSpec]":
    """Registry resolution: every tool exists, DYNAMIC is barred, and
    depends_on names declared stages (at most one — no fan-in)."""
    declared = {entry["stage"] for entry in entries}
    stages: list[StageSpec] = []
    for entry in entries:
        stage, tool = entry["stage"], entry["tool"]
        try:
            spec = registry.lookup(tool)
        except UnknownToolError:
            raise rejection(
                f"stage {stage!r} names unregistered tool {tool!r}"
            ) from None
        if TypeShape.DYNAMIC in (spec.input_shape, spec.output_shape):
            raise rejection(
                f"stage {stage!r}: tool {tool!r} declares DYNAMIC shapes and "
                "cannot appear inside a dag_spec (no nesting)"
            )
        depends_on = _upstream_names(stage, entry.get("depends_on"), declared)
        params = entry.get("params") or {}
        if not isinstance(params, Mapping):
            raise rejection(f"stage {stage!r}: 'params' must be a mapping")
        stages.append(
            StageSpec(stage=stage, spec=spec, params=params, depends_on=depends_on)
        )
    return stages


def _upstream_names(stage: str, raw: Any, declared: "set[str]") -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise rejection(f"stage {stage!r}: 'depends_on' must be a list of stage names")
    upstream = tuple(str(name) for name in raw)
    for name in upstream:
        if name == stage:
            raise rejection(f"stage {stage!r} depends on itself")
        if name not in declared:
            raise rejection(f"stage {stage!r} depends on undeclared stage {name!r}")
    if len(upstream) > 1:
        raise rejection(
            f"stage {stage!r} declares {len(upstream)} upstream stages "
            f"({list(upstream)}) — the launch topology is linear chains "
            "with fan-out, no fan-in (one upstream at most, §6.3)"
        )
    return upstream


def _check_endpoint_rules(
    stages: "list[StageSpec]", by_name: "Mapping[str, StageSpec]"
) -> None:
    """NONE-endpoint and addressing rules: sources are chain-initial,
    sinks are leaves, dependent stages read their upstream edge."""
    dependents: dict[str, list[str]] = {entry.stage: [] for entry in stages}
    for entry in stages:
        for upstream in entry.depends_on:
            dependents[upstream].append(entry.stage)
    for entry in stages:
        if entry.depends_on and entry.spec.input_shape is TypeShape.NONE:
            raise rejection(
                f"stage {entry.stage!r}: tool {entry.spec.name!r} is a source "
                "(input_shape NONE) and accepts no depends_on edge — it may "
                "appear only chain-initial"
            )
        if dependents[entry.stage] and entry.spec.output_shape is TypeShape.NONE:
            raise rejection(
                f"stage {entry.stage!r}: tool {entry.spec.name!r} is a "
                "terminal sink (output_shape NONE) and may appear only as a "
                f"leaf — remove the edge to {dependents[entry.stage]!r}"
            )
        if entry.depends_on:
            addressed = [name for name in _SOURCE_PARAMS if name in entry.params]
            if addressed:
                raise rejection(
                    f"stage {entry.stage!r} declares depends_on="
                    f"{list(entry.depends_on)!r} but also addresses its own "
                    f"source via {addressed!r} — a dependent stage consumes "
                    "its upstream stage's output"
                )


def _check_edges(stages: "list[StageSpec]", by_name: "Mapping[str, StageSpec]") -> None:
    """Adjacency legality per declared edge, naming the offending pair."""
    for entry in stages:
        for upstream_name in entry.depends_on:
            upstream = by_name[upstream_name]
            if TypeShape.NONE in (
                upstream.spec.output_shape,
                entry.spec.input_shape,
            ) or not may_feed(upstream.spec.output_shape, entry.spec.input_shape):
                raise rejection(
                    f"incompatible edge {upstream_name!r} -> {entry.stage!r}: "
                    f"{upstream.spec.name!r} emits "
                    f"{upstream.spec.output_shape.value} which may not feed "
                    f"{entry.spec.name!r} consuming "
                    f"{entry.spec.input_shape.value} (FR-606 adjacency table)"
                )
