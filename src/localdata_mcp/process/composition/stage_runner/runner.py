"""localdata_mcp/process/composition/stage_runner/runner.py — one stage's execution (E11.1).

The per-stage invocation loop, harvest-rewritten from
`PipelineComposer._execute_single_pipeline`: a stage IS its registered
tool's implementation function, called with the caller's params — a
chain-initial stage addresses its own source through X-2 exactly as a
standalone call would, a dependent stage receives its upstream
sibling's output through the addressing home's injection channel
(`explore.addressing.pipeline_input`). §4d's NX-6 re-crossing needs no
runner code BECAUSE of this design: every data touch a stage performs
goes through the tool's own `addressed_frame`/guard path with full
posture/paths/bounds checks — the runner adds no bypass to close.
Neighbors: sequence.py orders the calls; errors.py names the failures.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.explore.addressing import pipeline_input
from localdata_mcp.nexus.observability.manager import get_logger

from ..dag_spec import StageSpec
from .results import StageOutput

_logger = get_logger(__name__)


def run_stage(entry: StageSpec, upstream: "StageOutput | None") -> Any:
    """Execute one stage, returning the implementation's raw result
    (exception classification is the caller's, errors.py)."""
    arguments = dict(entry.params)
    _logger.debug(
        "pipeline stage starting",
        stage=entry.stage,
        tool=entry.spec.name,
        dependent=upstream is not None,
    )
    if upstream is None:
        return entry.spec.func(**arguments)
    with pipeline_input(upstream.frame, upstream.label):
        return entry.spec.func(**arguments)
