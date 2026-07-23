"""localdata_mcp/process/composition/stage_runner/sequence.py — chain orchestration (E11.1).

The harvest-rewrite of `PipelineComposer._execute_sequential_workflow`
/`_execute_parallel_group` onto v3's nexuses: validate first (FR-606,
dag_spec.py — no partial run), then execute the topological order one
stage at a time (fan-out groups run sequentially within one process —
the grouping is scheduling structure, not thread parallelism), each
stage's output handed downstream via the injection channel, each raw
result sentinel-checked (S3.3) before anything consumes it. NFR-105's
aggregate accounting: the frames held for downstream stages charge ONE
guard ledger entry for the whole pipeline, released on every exit
path. v2's adaptive-workflow mode and continue/partial error-recovery
modes are dead mass, not harvested — a failed stage fails the chain
with a structured NX-3 refusal naming it. Neighbors: tools.py (E11.2)
exposes this as compose_pipeline.
"""

from __future__ import annotations

import uuid
from typing import Any

from localdata_mcp.ingest.runtime import chokepoint
from localdata_mcp.nexus.contract.registry import ToolRegistry, default_registry
from localdata_mcp.nexus.observability.manager import get_logger
from localdata_mcp.nexus.response.shaping import stage_sentinel

from ..dag_spec import ValidatedDag, validate_dag_spec
from .errors import degenerate_stage, stage_failure
from .results import StageOutput, leaf_map, provenance_chain, stage_output
from .runner import run_stage

_logger = get_logger(__name__)


def run_pipeline(
    dag_spec: Any, *, registry: "ToolRegistry | None" = None
) -> dict[str, Any]:
    """The whole composed run: FR-606 validation, ordered execution,
    and the §6.3 multi-leaf response under one provenance chain."""
    target = registry if registry is not None else default_registry()
    guard = chokepoint()
    limits = guard.composition_limits()
    dag = validate_dag_spec(dag_spec, target, limits.max_pipeline_length)
    pipeline_id = uuid.uuid4().hex
    raws = _execute(dag, pipeline_id)
    results = leaf_map(dag, raws)
    return {
        "summary": (
            f"{len(dag.order)} stage(s) executed "
            f"({' -> '.join(dag.order)}); terminal result(s): "
            f"{', '.join(dag.leaves)}."
        ),
        "results": results,
        "provenance": provenance_chain(dag),
    }


def _execute(dag: ValidatedDag, pipeline_id: str) -> dict[str, Any]:
    """Run every stage in topological order, holding only the outputs
    downstream stages still need — charged as one aggregate ledger
    entry (NFR-105) and released on every exit path."""
    guard = chokepoint()
    remaining = _dependent_counts(dag)
    outputs: dict[str, StageOutput] = {}
    raws: dict[str, Any] = {}
    try:
        for stage_name in dag.order:
            entry = dag.stage_named(stage_name)
            upstream = outputs.get(entry.depends_on[0]) if entry.depends_on else None
            try:
                raw = run_stage(entry, upstream)
            except Exception as failure:
                raise stage_failure(stage_name, entry.spec.name, failure) from failure
            tripped = stage_sentinel(raw)
            if tripped is not None:
                raise degenerate_stage(stage_name, entry.spec.name, tripped)
            raws[stage_name] = raw
            for consumed in entry.depends_on:
                # Streaming-first residency: drop an upstream frame the
                # moment its last dependent has consumed it.
                remaining[consumed] -= 1
                if remaining[consumed] == 0:
                    outputs.pop(consumed, None)
            if remaining[stage_name]:
                outputs[stage_name] = stage_output(entry, raw)
            _charge(guard, pipeline_id, outputs, stage_name, entry.spec.name)
            _logger.debug(
                "pipeline stage completed", stage=stage_name, tool=entry.spec.name
            )
        return raws
    finally:
        guard.release_composition(pipeline_id)


def _dependent_counts(dag: ValidatedDag) -> dict[str, int]:
    counts = {entry.stage: 0 for entry in dag.stages}
    for entry in dag.stages:
        for upstream in entry.depends_on:
            counts[upstream] += 1
    return counts


def _charge(
    guard: Any,
    pipeline_id: str,
    outputs: "dict[str, StageOutput]",
    stage: str,
    tool: str,
) -> None:
    """Re-charge the pipeline's one ledger entry with the current
    aggregate of held inter-stage frames; a refusal is a stage failure
    (the ledger rolls back, nothing ran partially downstream)."""
    total = sum(
        int(output.frame.memory_usage(deep=True).sum()) for output in outputs.values()
    )
    try:
        guard.charge_composition(pipeline_id, total)
    except Exception as failure:
        raise stage_failure(stage, tool, failure) from failure
