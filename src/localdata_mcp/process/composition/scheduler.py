"""localdata_mcp/process/composition/scheduler.py — topo sort + fan-out groups (E11.1).

The FR-602 harvest: Kahn's-algorithm topological ordering and the
same-dependency-set fan-out grouping, re-authored from the dead
`PipelineComposer` (`pipeline/core/composer.py` — `_topological_sort`,
`_identify_parallel_groups`, `resolve_dependencies`'s DFS cycle check
collapses into Kahn's own completeness test). Operates over the
declared dependency mapping alone — no execution, no registry — so
the ordering logic stays independently provable. The structural model
admits arbitrary DAGs (fan-in later is additive, §6.3); the launch
no-fan-in topology is dag_spec.py's validation concern, not enforced
here. Neighbors: dag_spec.py validates before scheduling;
stage_runner/sequence.py executes the resulting order.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping


class CompositionCycleError(ValueError):
    """The declared dependencies contain a cycle — named so dag_spec's
    validation can reject it as a structured NX-3 refusal."""


def execution_order(dependencies: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Kahn's algorithm over `stage -> upstream stages`: a
    deterministic topological order (declaration order breaks ties via
    the mapping's own ordering). Raises CompositionCycleError when any
    stage never reaches in-degree zero."""
    in_degree = {stage: len(upstream) for stage, upstream in dependencies.items()}
    dependents: dict[str, list[str]] = {stage: [] for stage in dependencies}
    for stage, upstream in dependencies.items():
        for name in upstream:
            dependents[name].append(stage)
    ready = deque(stage for stage, degree in in_degree.items() if degree == 0)
    ordered: list[str] = []
    while ready:
        current = ready.popleft()
        ordered.append(current)
        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                ready.append(dependent)
    if len(ordered) != len(dependencies):
        stuck = sorted(set(dependencies) - set(ordered))
        raise CompositionCycleError(
            f"circular dependency among stage(s) {stuck} — a dag_spec must be acyclic"
        )
    return tuple(ordered)


def fanout_groups(
    order: tuple[str, ...], dependencies: Mapping[str, tuple[str, ...]]
) -> tuple[tuple[str, ...], ...]:
    """Consecutive grouping of stages sharing the identical upstream
    set (the harvested same-level rule): members of a group have no
    edges between each other, so a group is one fan-out level."""
    groups: list[tuple[str, ...]] = []
    assigned: set[str] = set()
    for stage in order:
        if stage in assigned:
            continue
        upstream = frozenset(dependencies[stage])
        group = [stage]
        for other in order:
            if other == stage or other in assigned:
                continue
            if frozenset(dependencies[other]) == upstream:
                group.append(other)
        groups.append(tuple(group))
        assigned.update(group)
    return tuple(groups)


def terminal_stages(dependencies: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """The DAG's leaves — stages no other stage depends on — in
    declaration order; these key the multi-leaf response map (§6.3)."""
    depended_on = {name for upstream in dependencies.values() for name in upstream}
    return tuple(stage for stage in dependencies if stage not in depended_on)
