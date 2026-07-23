"""localdata_mcp/nexus/response/sentinel.py — the degenerate-output sentinel (E7.2).

S3.3 / FR-308 extended to the SILENT half of the failure space: raised
exceptions are only half of statistical failure — zero-variance groups
return `nan` without raising, rank-deficient design matrices return
garbage coefficients, optimizers report non-convergence in flags while
returning fitted-looking parameters. This is the ONE shared
post-computation check (applied by NX-1's generated wrapper inside
envelope shaping — never per-domain ad-hoc code) over the four
declared signal classes:

1. **Non-finite values** — any `NaN`/`Inf` anywhere in the numeric
   payload (dicts, sequences, DataFrames, arrays walked recursively).
2. **Non-convergence flags** — the declared result-key conventions
   domain tools emit: `converged` False, or `optimizer_status`
   non-zero.
3. **Degenerate shapes** — a `groups`/`clusters` entry that is empty
   or contains an empty member (an empty top-level result is NOT this:
   that is O-1's legitimate zero statement, handled by envelope.py).
4. **Rank deficiency** — `rank` below `design_columns`, or
   `condition_number` above `process.sentinel_max_condition_number`
   (S8 row 32; κ computed by the tool on the matrix as solved) — the
   necessary fourth class: a rank-deficient direct solve returns a
   full-shaped, NaN-free minimum-norm solution that trips none of the
   first three.

Every trip converts to a structured NX-3 error — never a silent
successful envelope. Neighbors: shaping.py runs this before
shape_envelope; the E10 domain batteries pin one fixture per class.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from localdata_mcp.nexus.error.model import ErrorType, StructuredError

# The declared result-key conventions the sentinel reads (E10 domain
# tools emit these; the battery asserts each class end to end).
_CONVERGENCE_FLAG = "converged"
_OPTIMIZER_STATUS = "optimizer_status"
_GROUP_KEYS = ("groups", "clusters")
_RANK_KEY = "rank"
_DESIGN_COLUMNS_KEY = "design_columns"
_CONDITION_NUMBER_KEY = "condition_number"


def inspect(result: Any, max_condition_number: float) -> StructuredError | None:
    """The four-class verdict: a structured error on the first tripped
    signal, None for a clean result."""
    finding = (
        _non_finite_signal(result)
        or _convergence_signal(result)
        or _degenerate_shape_signal(result)
        or _rank_signal(result, max_condition_number)
    )
    if finding is None:
        return None
    return StructuredError(
        error_type=ErrorType.DATA_VALIDATION,
        message=f"degenerate numeric output: {finding}",
        suggestion=(
            "The computation completed but its output is not trustworthy — "
            "inspect the input data (empty or constant groups, collinear "
            "columns) before re-running."
        ),
        retryable=False,
    )


def _non_finite_signal(value: Any, path: str = "result") -> str | None:
    """Class 1: NaN/Inf anywhere in the numeric payload."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return f"non-finite value at {path}"
        return None
    if isinstance(value, Mapping):
        for key, entry in value.items():
            found = _non_finite_signal(entry, f"{path}[{key!r}]")
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, entry in enumerate(value):
            found = _non_finite_signal(entry, f"{path}[{index}]")
            if found:
                return found
        return None
    # DataFrames / arrays: anything exposing a to-list-able flat view.
    values = getattr(value, "values", None)
    if values is not None and hasattr(values, "flatten"):
        for entry in values.flatten().tolist():
            if isinstance(entry, float) and (math.isnan(entry) or math.isinf(entry)):
                return f"non-finite value inside {path}"
    return None


def _convergence_signal(result: Any) -> str | None:
    """Class 2: the declared non-convergence flag conventions."""
    if not isinstance(result, Mapping):
        return None
    if result.get(_CONVERGENCE_FLAG) is False:
        return "the solver reported non-convergence (converged=False)"
    status = result.get(_OPTIMIZER_STATUS)
    if isinstance(status, int) and status != 0:
        return f"the optimizer reported failure status {status}"
    return None


def _degenerate_shape_signal(result: Any) -> str | None:
    """Class 3: empty groups/clusters INSIDE a result (a wholly empty
    top-level result is O-1's legitimate case, not this)."""
    if not isinstance(result, Mapping):
        return None
    for key in _GROUP_KEYS:
        entries = result.get(key)
        if entries is None:
            continue
        if isinstance(entries, (list, tuple, Mapping)) and len(entries) == 0:
            return f"{key} is empty — a degenerate grouping"
        if isinstance(entries, Mapping):
            members: Sequence[Any] = tuple(entries.values())
        elif isinstance(entries, (list, tuple)):
            members = entries
        else:
            continue
        for member in members:
            if hasattr(member, "__len__") and len(member) == 0:
                return f"{key} contains an empty member group"
    return None


def _rank_signal(result: Any, max_condition_number: float) -> str | None:
    """Class 4: solver-reported rank below the design-matrix column
    count, or condition number over the S8 row-32 threshold."""
    if not isinstance(result, Mapping):
        return None
    rank = result.get(_RANK_KEY)
    columns = result.get(_DESIGN_COLUMNS_KEY)
    if isinstance(rank, int) and isinstance(columns, int) and rank < columns:
        return (
            f"rank deficiency: solver rank {rank} below the design "
            f"matrix's {columns} columns"
        )
    kappa = result.get(_CONDITION_NUMBER_KEY)
    if isinstance(kappa, (int, float)) and kappa > max_condition_number:
        return (
            f"ill-conditioned system: condition number {kappa:.3g} exceeds "
            f"the {max_condition_number:.3g} threshold (S8 row 32)"
        )
    return None
