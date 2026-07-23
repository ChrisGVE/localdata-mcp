"""localdata_mcp/process/composition/stage_runner/errors.py — stage-failure classification (E11.1).

The harvest-rewrite of `PipelineComposer`'s two `_handle_*_error`
paths, rewired onto NX-3: every stage failure becomes ONE structured
refusal that names the failing stage and tool, whatever the underlying
exception class — a guard refusal keeps its structured shape, a
resource refusal becomes the over-budget wording, anything else
crosses `NX3.wrap`'s redaction. A failed stage fails the WHOLE
pipeline (strict semantics — v2's continue/partial recovery modes are
dead mass, not harvested): partial results never leak. Neighbors:
sequence.py raises these; the outer shaped_call turns them into the
error envelope.
"""

from __future__ import annotations

from localdata_mcp.nexus.chokepoint.guard import (
    GuardedExecutionError,
    ResourceRefusedError,
)
from localdata_mcp.nexus.error.model import ErrorType, StructuredError
from localdata_mcp.nexus.error.wire import wrap


def stage_failure(stage: str, tool: str, failure: Exception) -> GuardedExecutionError:
    """`failure` as the pipeline's one structured stage-failure shape,
    the stage and tool named in front of the underlying message."""
    if isinstance(failure, GuardedExecutionError):
        inner = failure.structured
    elif isinstance(failure, ResourceRefusedError):
        inner = StructuredError(
            error_type=ErrorType.RESOURCE_ERROR,
            message=str(failure),
            suggestion=(
                "The pipeline's aggregate inter-stage data exceeded the "
                "process-wide memory ceiling (NFR-105). Reduce the input's "
                "size (a tighter query, fewer columns) or shorten the chain."
            ),
            retryable=True,
        )
    else:
        inner = wrap(failure, "generic")
    return GuardedExecutionError(_named(stage, tool, inner))


def degenerate_stage(
    stage: str, tool: str, tripped: StructuredError
) -> GuardedExecutionError:
    """A mid-chain sentinel trip (S3.3): the degenerate output is a
    named stage failure, never silent input to the next stage."""
    return GuardedExecutionError(_named(stage, tool, tripped))


def handoff_failure(stage: str, tool: str, detail: str) -> GuardedExecutionError:
    """The engine could not extract composable data from a stage's
    result for its downstream edge — an engine-level failure named as
    such (distinct from a domain-level refusal, FR-302)."""
    return GuardedExecutionError(
        _named(
            stage,
            tool,
            StructuredError(
                error_type=ErrorType.DATA_VALIDATION,
                message=detail,
                suggestion=(
                    "The stage ran, but its result carries no composable "
                    "payload for the declared edge — report this chain: the "
                    "declared output shape and the result disagree."
                ),
                retryable=False,
            ),
        )
    )


def _named(stage: str, tool: str, inner: StructuredError) -> StructuredError:
    return StructuredError(
        error_type=inner.error_type,
        message=f"pipeline stage {stage!r} (tool {tool!r}) failed: {inner.message}",
        suggestion=inner.suggestion,
        retryable=inner.retryable,
    )
