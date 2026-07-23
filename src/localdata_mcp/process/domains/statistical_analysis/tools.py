"""localdata_mcp/process/domains/statistical_analysis/tools.py — E10.a ToolSpecs.

The statistical family's four tools, carried by name from `main`
(DR GP2): `analyze_hypothesis_test`, `analyze_anova`,
`analyze_effect_sizes` (FR-310 closure), `analyze_ab_test`. Each is a
thin declaration over its computation module: resolve the addressed
frame (the X-2 exactly-one-source contract via support.py), run the
computation, attach the source label. Caller-omittable knobs are
`required=False` Params — the implementation default governs (one
default site, NFR-403). input_shape=TABULAR: these consume a table
(chain-initial via their own addressing, or fed by an upstream E11
stage); output SCALAR — a verdict dict, not a relation. Neighbors:
hypothesis/anova/effects/ab_test.py compute; spec_modules.py rosters
this module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..support import addressed_frame, source_params
from .ab_test import perform_ab_test
from .anova import perform_anova
from .effects import calculate_effect_sizes
from .hypothesis import run_hypothesis_test

_ALPHA = Param(
    "alpha",
    float,
    "Significance level for the verdict (implementation default 0.05).",
    required=False,
)
_ALTERNATIVE = Param(
    "alternative",
    str,
    "Alternative hypothesis: two-sided (default), greater, or less.",
    required=False,
)


@tool_spec(
    name="analyze_hypothesis_test",
    summary=(
        "Run a hypothesis test on an addressed tabular source. "
        "test_type auto (default) selects from the supplied columns: "
        "group_column= compares two groups, second_column= correlates "
        "two columns, column= alone tests normality. Explicit types: "
        "ttest_1samp, ttest_ind, ttest_rel, mann_whitney, wilcoxon, "
        "chi2, normality, correlation."
    ),
    params=(
        *source_params(),
        Param(
            "test_type",
            str,
            "Test to run (default auto — selected from the supplied columns).",
            required=False,
        ),
        Param("column", str, "The primary value column.", required=False),
        Param(
            "second_column",
            str,
            "Second column for paired/correlation/chi2 tests.",
            required=False,
        ),
        Param(
            "group_column",
            str,
            "Column defining the two groups for two-sample tests.",
            required=False,
        ),
        Param(
            "popmean",
            float,
            "Population mean for ttest_1samp (implementation default 0.0).",
            required=False,
        ),
        _ALPHA,
        _ALTERNATIVE,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="statistical_analysis",
)
def analyze_hypothesis_test(
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = run_hypothesis_test(frame, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="analyze_anova",
    summary=(
        "One-way ANOVA across every group of group_var on an addressed "
        "tabular source: F statistic, p-value, eta squared, per-group "
        "summary, and Tukey HSD post-hoc when significant."
    ),
    params=(
        *source_params(),
        Param("dependent_var", str, "The numeric outcome column."),
        Param("group_var", str, "The column defining the groups."),
        _ALPHA,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="statistical_analysis",
)
def analyze_anova(
    dependent_var: str,
    group_var: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = perform_anova(frame, dependent_var, group_var, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="analyze_effect_sizes",
    summary=(
        "Effect sizes for a grouping on an addressed tabular source: "
        "Cohen's d, Hedges' g, Glass's delta, and Cliff's delta for two "
        "groups; eta and omega squared for three or more."
    ),
    params=(
        *source_params(),
        Param("column", str, "The numeric value column."),
        Param("group_column", str, "The column defining the groups."),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="statistical_analysis",
)
def analyze_effect_sizes(
    column: str,
    group_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = calculate_effect_sizes(frame, column, group_column)
    result["source"] = source
    return result


@tool_spec(
    name="analyze_ab_test",
    summary=(
        "A/B test between the exactly-two variants of variant_column on "
        "an addressed tabular source. test_type auto (default) picks the "
        "two-proportion z-test for a binary metric, Welch's t-test "
        "otherwise; mann_whitney on request. Names the winner and lift."
    ),
    params=(
        *source_params(),
        Param("metric_column", str, "The outcome metric column."),
        Param("variant_column", str, "The column assigning the two variants."),
        Param(
            "test_type",
            str,
            "auto (default), proportion, t_test, or mann_whitney.",
            required=False,
        ),
        _ALPHA,
        _ALTERNATIVE,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="statistical_analysis",
)
def analyze_ab_test(
    metric_column: str,
    variant_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = perform_ab_test(frame, metric_column, variant_column, **knobs)
    result["source"] = source
    return result
