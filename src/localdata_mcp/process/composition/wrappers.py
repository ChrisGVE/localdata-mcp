"""localdata_mcp/process/composition/wrappers.py — C-2 convenience wrappers (E11.3).

The curated zero-logic wrappers (S3.6 C-2): each is a `ToolSpec` whose
body is a FIXED `dag_spec` into the same `run_pipeline` engine, so a
wrapper's result equals the equivalent explicit `compose_pipeline`
call bit-for-bit by construction (NFR-402-safe — one engine, no
second execution path). Three ship here (S3.6 C-2): the two
highest-traffic prep->analyze couplings, and `cluster_then_chart`
(assign_clusters -> render_chart, the Process->Visualize archetype) —
it chains `assign_clusters` (the TABULAR composable counterpart to the
analyze_clusters verdict) into a scatter coloured by the `cluster`
column, so the whole chain is type-legal (TABULAR -> TABULAR). `missing_strategy` inherits P-2's enum and `drop`
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


@tool_spec(
    name="cluster_then_chart",
    summary=(
        "Convenience pipeline: assign_clusters -> render_chart(scatter_fit). "
        "Clusters an addressed source, then scatters its first two "
        "numeric feature columns coloured by cluster. k pins the cluster "
        "count (default: silhouette sweep); format is svg (default) or "
        "png. Equivalent to the explicit two-stage compose_pipeline."
    ),
    params=(
        *source_params(),
        Param(
            "k",
            int,
            "Cluster count (default: silhouette sweep over 2..8).",
            required=False,
        ),
        Param(
            "seed",
            int,
            "Random seed pinning the clustering (default: library "
            "behavior) — set it for a reproducible chart.",
            required=False,
        ),
        Param(
            "format",
            str,
            "Chart image format: svg (default) or png.",
            required=False,
        ),
    ),
    input_shape=TypeShape.DYNAMIC,
    output_shape=TypeShape.DYNAMIC,
    domain="composition",
)
def cluster_then_chart_tool(
    endpoint: "str | None" = None,
    path: "str | None" = None,
    table: "str | None" = None,
    query: "str | None" = None,
    k: "int | None" = None,
    seed: "int | None" = None,
    format: "str | None" = None,
) -> dict[str, Any]:
    cluster_params = {
        key: value
        for key, value in {
            "endpoint": endpoint,
            "path": path,
            "table": table,
            "query": query,
            "n_clusters": k,
            "seed": seed,
        }.items()
        if value is not None
    }
    chart_params = {"kind": "scatter_fit", "encoding": {"color": "cluster"}}
    if format is not None:
        chart_params["format"] = format
    return run_pipeline(
        [
            {"stage": "cluster", "tool": "assign_clusters", "params": cluster_params},
            {
                "stage": "chart",
                "tool": "render_chart",
                "params": chart_params,
                "depends_on": ["cluster"],
            },
        ]
    )
