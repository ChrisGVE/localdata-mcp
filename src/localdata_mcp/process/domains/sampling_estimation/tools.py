"""localdata_mcp/process/domains/sampling_estimation/tools.py — E10.g ToolSpecs.

The sampling family's four tools, carried by name from `main`
(DR GP2): `generate_sample`, `bootstrap_statistic`,
`monte_carlo_simulate`, `bayesian_estimate`. The stochastic counts
default to the S8 rows 30/31 values fetched through the guard's
`process_defaults()` seam at call time — the tool layer never reads
NX-2 (section 6.2), and a caller-supplied count overrides. Every
stochastic tool carries `seed` (S3.3). Neighbors:
sampling/estimation/monte_carlo.py compute; spec_modules.py rosters
this module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from localdata_mcp.ingest.runtime import chokepoint
from ..support import addressed_frame, source_params
from .estimation import bayesian_posterior, bootstrap
from .monte_carlo import simulate
from .sampling import draw_sample

_SEED = Param(
    "seed",
    int,
    "Random seed pinning every draw (default: fresh entropy).",
    required=False,
)


@tool_spec(
    name="generate_sample",
    summary=(
        "Draw a sample from an addressed tabular source: "
        "sampling_method simple_random (default), stratified, "
        "systematic, cluster, or weighted. sample_size: integer = row "
        "count, fraction = share (default 0.1). Returns the drawn "
        "relation plus the design summary."
    ),
    params=(
        *source_params(),
        Param(
            "sampling_method",
            str,
            "simple_random (default), stratified, systematic, cluster, weighted.",
            required=False,
        ),
        Param(
            "sample_size",
            float,
            "Integer row count, or fractional share (implementation default 0.1).",
            required=False,
        ),
        Param("stratify_column", str, "Stratum column (stratified).", required=False),
        Param("cluster_column", str, "Cluster column (cluster).", required=False),
        Param("weight_column", str, "Weight column (weighted).", required=False),
        _SEED,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.TABULAR,
    domain="sampling_estimation",
)
def generate_sample(
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = draw_sample(frame, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="bootstrap_statistic",
    summary=(
        "Percentile-bootstrap a statistic (mean, median, std, var) of "
        "one column on an addressed tabular source: estimate, "
        "confidence interval, and standard error. resamples defaults "
        "to the operator-configured count (S8 row 30)."
    ),
    params=(
        *source_params(),
        Param("column", str, "The numeric column to bootstrap."),
        Param(
            "statistic",
            str,
            "mean (default), median, std, or var.",
            required=False,
        ),
        Param(
            "resamples",
            int,
            "Bootstrap resamples (default: the configured S8 row-30 count).",
            required=False,
        ),
        Param(
            "confidence_level",
            float,
            "Interval coverage (implementation default 0.95).",
            required=False,
        ),
        _SEED,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="sampling_estimation",
)
def bootstrap_statistic(
    column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    defaults = chokepoint().process_defaults()
    result = bootstrap(
        frame, column, default_resamples=defaults.bootstrap_resamples, **knobs
    )
    result["source"] = source
    return result


@tool_spec(
    name="monte_carlo_simulate",
    summary=(
        "Monte Carlo over one column of an addressed tabular source: "
        "simulation_type uncertainty (default — resampled distribution "
        "of the mean) or integration (probability mass inside "
        "bounds=[lower, upper] under the fitted normal). iterations "
        "defaults to the operator-configured count (S8 row 31)."
    ),
    params=(
        *source_params(),
        Param("column", str, "The numeric column to simulate over."),
        Param(
            "simulation_type",
            str,
            "uncertainty (default) or integration.",
            required=False,
        ),
        Param(
            "iterations",
            int,
            "Simulation draws (default: the configured S8 row-31 count).",
            required=False,
        ),
        Param(
            "bounds",
            list,
            "[lower, upper] integration bounds (integration only).",
            required=False,
        ),
        _SEED,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="sampling_estimation",
)
def monte_carlo_simulate(
    column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    defaults = chokepoint().process_defaults()
    result = simulate(
        frame,
        column,
        default_iterations=defaults.monte_carlo_iterations,
        **knobs,
    )
    result["source"] = source
    return result


@tool_spec(
    name="bayesian_estimate",
    summary=(
        "Conjugate-normal Bayesian posterior of one column's mean on "
        "an addressed tabular source (noninformative prior): posterior "
        "mean, scale, and the Student-t credible interval."
    ),
    params=(
        *source_params(),
        Param("column", str, "The numeric column to estimate."),
        Param(
            "prior_distribution",
            str,
            "Conjugate prior family (normal — the launch set).",
            required=False,
        ),
        Param(
            "credible_level",
            float,
            "Interval coverage (implementation default 0.95).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="sampling_estimation",
)
def bayesian_estimate(
    column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = bayesian_posterior(frame, column, **knobs)
    result["source"] = source
    return result
