"""localdata_mcp/process/preprocessing/tools.py — E10.x8 ToolSpecs (FR-303).

The two data-prep stages' ToolSpecs: `prepare_missing_values` and
`convert_types`, both TABULAR→TABULAR so they compose as `dag_spec`
stages (E11) or run standalone. Thin over stages.py with the X-2
addressing contract. Neighbors: spec_modules.py rosters this module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..domains.support import addressed_frame, source_params
from .stages import convert_types, prepare_missing_values


@tool_spec(
    name="prepare_missing_values",
    summary=(
        "Handle missing values on an addressed tabular source: "
        "missing_strategy drop (default — fabricates nothing), mean, "
        "median, mode, forward_fill, or constant (needs fill_value). "
        "Returns the cleaned relation; composes as a pipeline stage."
    ),
    params=(
        *source_params(),
        Param(
            "columns",
            list,
            "Columns to clean (default: every column).",
            required=False,
        ),
        Param(
            "missing_strategy",
            str,
            "drop (default), mean, median, mode, forward_fill, constant.",
            required=False,
        ),
        Param(
            "fill_value",
            str,
            "Fill value for missing_strategy='constant'.",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.TABULAR,
    domain="preprocessing",
)
def prepare_missing_values_tool(
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = prepare_missing_values(frame, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="convert_types",
    summary=(
        "Coerce named columns of an addressed tabular source to a "
        "target type (numeric, integer, string, datetime, boolean) via "
        "the conversions map. Reports failed casts; composes as a "
        "pipeline stage."
    ),
    params=(
        *source_params(),
        Param(
            "conversions",
            dict,
            "Map of column name to target type.",
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.TABULAR,
    domain="preprocessing",
)
def convert_types_tool(
    conversions: dict[str, str],
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = convert_types(frame, conversions)
    result["source"] = source
    return result
