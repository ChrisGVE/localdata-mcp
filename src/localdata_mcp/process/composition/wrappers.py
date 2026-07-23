"""localdata_mcp/process/composition/wrappers.py — C-2 convenience wrappers (E11.3).

The curated zero-logic wrappers (S3.6 C-2): each is a `ToolSpec` whose
body is a FIXED `dag_spec` into the same `run_pipeline` engine, so a
wrapper's result equals the equivalent explicit `compose_pipeline`
call bit-for-bit by construction (NFR-402-safe — one engine, no
second execution path). Two ship here — the two highest-traffic
prep->analyze couplings; `cluster_then_chart` names an E12 tool
(`render_chart`) so it registers in E12.5 with its equality test in the
same change-set. `missing_strategy` inherits P-2's enum and `drop`
default, so `clean_then_profile(path=...)` is genuinely callable with
just a source (the wrapper's whole point). Neighbors: sequence.py is
the engine; spec_modules.py rosters this module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..domains.support import source_params
from .stage_runner.sequence import run_pipeline


def _clean_stage(params: dict[str, Any]) -> dict[str, Any]:
    """The shared prep-first stage: only the caller-supplied source and
    (optional) strategy reach prepare_missing_values, so its own P-2
    defaults govern — never a wrapper-invented value (NFR-403)."""
    clean_params = {key: value for key, value in params.items() if value is not None}
    return {"stage": "clean", "tool": "prepare_missing_values", "params": clean_params}


@tool_spec(
    name="clean_then_profile",
    summary=(
        "Convenience pipeline: prepare_missing_values -> profile_data on "
        "one addressed source. missing_strategy defaults to drop "
        "(fabricates nothing); callable with a source alone. Equivalent "
        "to the explicit two-stage compose_pipeline."
    ),
    params=(
        *source_params(),
        Param(
            "missing_strategy",
            str,
            "drop (default), mean, median, mode, forward_fill, constant.",
            required=False,
        ),
    ),
    input_shape=TypeShape.DYNAMIC,
    output_shape=TypeShape.DYNAMIC,
    domain="composition",
)
def clean_then_profile_tool(
    endpoint: "str | None" = None,
    path: "str | None" = None,
    table: "str | None" = None,
    query: "str | None" = None,
    missing_strategy: "str | None" = None,
) -> dict[str, Any]:
    clean = _clean_stage(
        {
            "endpoint": endpoint,
            "path": path,
            "table": table,
            "query": query,
            "missing_strategy": missing_strategy,
        }
    )
    return run_pipeline(
        [
            clean,
            {"stage": "profile", "tool": "profile_data", "depends_on": ["clean"]},
        ]
    )


@tool_spec(
    name="clean_then_regress",
    summary=(
        "Convenience pipeline: prepare_missing_values -> analyze_regression "
        "on one addressed source, predicting target. missing_strategy "
        "defaults to drop. Equivalent to the explicit two-stage "
        "compose_pipeline."
    ),
    params=(
        *source_params(),
        Param("target", str, "The regression target column."),
        Param(
            "missing_strategy",
            str,
            "drop (default), mean, median, mode, forward_fill, constant.",
            required=False,
        ),
    ),
    input_shape=TypeShape.DYNAMIC,
    output_shape=TypeShape.DYNAMIC,
    domain="composition",
)
def clean_then_regress_tool(
    target: str,
    endpoint: "str | None" = None,
    path: "str | None" = None,
    table: "str | None" = None,
    query: "str | None" = None,
    missing_strategy: "str | None" = None,
) -> dict[str, Any]:
    clean = _clean_stage(
        {
            "endpoint": endpoint,
            "path": path,
            "table": table,
            "query": query,
            "missing_strategy": missing_strategy,
        }
    )
    return run_pipeline(
        [
            clean,
            {
                "stage": "regress",
                "tool": "analyze_regression",
                "params": {"target_column": target},
                "depends_on": ["clean"],
            },
        ]
    )
