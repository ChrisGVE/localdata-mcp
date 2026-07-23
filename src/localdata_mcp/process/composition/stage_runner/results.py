"""localdata_mcp/process/composition/stage_runner/results.py — handoff + multi-leaf assembly (E11.1).

Two determinate contracts. (1) The HANDOFF contract: how composable
data leaves one stage's result dict for its downstream edge —
TABULAR results carry `columns`+`rows` (the shape every TABULAR-output
tool emits), VECTOR results carry one of the declared vector keys,
rebuilt as a one-column `value` frame. (2) The RESPONSE contract
(§6.3): the composed result is `{terminal_stage_name: envelope}` — one
standard NX-7 envelope per leaf via the shaping seam — under a single
top-level provenance chain recording the full DAG execution. v2's
metadata enrich/propagate layers are superseded by NX-7's envelope
channel and not harvested. Neighbors: sequence.py drives both;
errors.py names a failed handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from localdata_mcp.nexus.contract.spec import TypeShape
from localdata_mcp.nexus.response.shaping import shaped_stage_envelope

from ..dag_spec import StageSpec, ValidatedDag
from .errors import handoff_failure

# The declared vector-carrier keys a VECTOR-output result may use, in
# lookup order (forecast_time_series emits `forecast`; `values` and
# `labels` are the generic and label-series carriers).
_VECTOR_KEYS = ("forecast", "values", "labels")

# The column name a VECTOR handoff lands under downstream.
VECTOR_COLUMN = "value"


@dataclass(frozen=True)
class StageOutput:
    """One stage's composable output, ready for injection downstream."""

    frame: pd.DataFrame
    label: str


def stage_output(entry: StageSpec, raw: Any) -> StageOutput:
    """`raw` per the handoff contract for `entry`'s declared output
    shape; raises the named engine-level handoff failure otherwise."""
    frame = _extracted_frame(entry, raw)
    return StageOutput(frame=frame, label=f"pipeline:{entry.stage}")


def _extracted_frame(entry: StageSpec, raw: Any) -> pd.DataFrame:
    shape = entry.spec.output_shape
    if not isinstance(raw, Mapping):
        raise handoff_failure(
            entry.stage,
            entry.spec.name,
            f"result is {type(raw).__name__}, not a mapping — no "
            f"{shape.value} payload to hand downstream",
        )
    if shape is TypeShape.TABULAR:
        columns, rows = raw.get("columns"), raw.get("rows")
        if columns is None or rows is None:
            raise handoff_failure(
                entry.stage,
                entry.spec.name,
                "a TABULAR result must carry 'columns' and 'rows'",
            )
        return pd.DataFrame(list(rows), columns=list(columns))
    if shape is TypeShape.VECTOR:
        for key in _VECTOR_KEYS:
            if key in raw:
                return pd.DataFrame({VECTOR_COLUMN: list(raw[key])})
        raise handoff_failure(
            entry.stage,
            entry.spec.name,
            f"a VECTOR result must carry one of {list(_VECTOR_KEYS)}",
        )
    raise handoff_failure(
        entry.stage,
        entry.spec.name,
        f"no handoff contract for output shape {shape.value} — no "
        "registered tool consumes it as a pipeline edge at launch",
    )


def leaf_map(dag: ValidatedDag, raws: Mapping[str, Any]) -> dict[str, Any]:
    """§6.3's response map: one standard NX-7 envelope per terminal
    stage, shaped by the same rules as a standalone call."""
    return {
        leaf: shaped_stage_envelope(dag.stage_named(leaf).spec.name, raws[leaf])
        for leaf in dag.leaves
    }


def provenance_chain(dag: ValidatedDag) -> dict[str, Any]:
    """The single top-level provenance chain: every stage's tool and
    edge, the executed order, and the terminal set."""
    return {
        "stages": [
            {
                "stage": entry.stage,
                "tool": entry.spec.name,
                "depends_on": list(entry.depends_on),
            }
            for entry in dag.stages
        ],
        "execution_order": list(dag.order),
        "terminal_stages": list(dag.leaves),
    }
