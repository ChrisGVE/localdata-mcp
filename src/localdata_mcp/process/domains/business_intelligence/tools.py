"""localdata_mcp/process/domains/business_intelligence/tools.py — E10.h ToolSpecs.

The BI family's two tools: `analyze_rfm` (carried by name from
`main`, FR-307's reordered cascade) and `calculate_clv` (S9.2
category 6 — first registration in v3; FR-309's configurable
customer column). Thin over rfm.py / clv.py with the X-2 addressing
contract. Neighbors: spec_modules.py rosters this module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..support import addressed_frame, source_params

_CUSTOMER = Param("customer_column", str, "The customer identifier column.")
_DATE = Param("date_column", str, "The transaction date column.")
_VALUE = Param("value_column", str, "The transaction amount column.")


@tool_spec(
    name="analyze_rfm",
    summary=(
        "RFM customer segmentation on an addressed tabular source: "
        "quintile recency/frequency/monetary scores and the named "
        "segment cascade (Champions ... Lost — every segment "
        "reachable). Returns per-customer scores and per-segment "
        "summaries."
    ),
    params=(*source_params(), _CUSTOMER, _DATE, _VALUE),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="process",
)
def analyze_rfm(
    customer_column: str,
    date_column: str,
    value_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
) -> Any:
    from .rfm import analyze_rfm_segments

    frame, source = addressed_frame(endpoint, path, table, query)
    result = analyze_rfm_segments(frame, customer_column, date_column, value_column)
    result["source"] = source
    return result


@tool_spec(
    name="calculate_clv",
    summary=(
        "Historical customer lifetime value on an addressed tabular "
        "source: per-customer average order value x purchase frequency "
        "x gross_margin, annualized. The customer identifier column is "
        "whatever customer_column names."
    ),
    params=(
        *source_params(),
        _CUSTOMER,
        _DATE,
        _VALUE,
        Param(
            "gross_margin",
            float,
            "Gross margin share applied to revenue (implementation default 0.2).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="process",
)
def calculate_clv(
    customer_column: str,
    date_column: str,
    value_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    from .clv import calculate_lifetime_value

    frame, source = addressed_frame(endpoint, path, table, query)
    result = calculate_lifetime_value(
        frame, customer_column, date_column, value_column, **knobs
    )
    result["source"] = source
    return result
