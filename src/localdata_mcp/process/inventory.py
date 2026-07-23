"""localdata_mcp/process/inventory.py — FR-703's orphan disposition (E10.x9).

FR-703: every domain function intended for agent use is either
tool-registered or explicitly listed here with a rationale — v3 ships
no implemented-but-unexposed domain logic silently (#27). v3 was
authored tool-first, so every compute function backs a registered
ToolSpec by construction; what this module dispositions is `main`'s
harvested domain capabilities that v3 deliberately did NOT expose at
launch. Each deferred entry names the capability and why it waits
(S10). The test_process_inventory gate reads both halves: the live
registry (must equal LAUNCH_PROCESS_TOOLS) and this deferred list
(every entry carries a non-empty rationale). Neighbors:
nexus/contract/registry.py holds the live surface; the PRD S10 is the
narrative home of the deferrals.
"""

from __future__ import annotations

from typing import Final

# The process-inventory domain vocabulary (E11 made ToolSpec.domain
# carry the FAMILY name, so the battery's launch domain set derives
# from registry declarations, §6.3): the nine P-1 analysis families
# plus the FR-303 preprocessing stage family. The composition surface
# (FR-6xx: compose_pipeline and its wrappers, domain="composition") is
# deliberately NOT inventoried here — it composes domain tools, it is
# not a domain function FR-703 could orphan.
PROCESS_INVENTORY_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "statistical_analysis",
        "regression_modeling",
        "pattern_recognition",
        "time_series",
        "geospatial_analysis",
        "optimization",
        "sampling_estimation",
        "business_intelligence",
        "network_graph",
        "preprocessing",
    }
)

# The 34 process-domain tools that ship at launch (PRD S3.3's P-1
# families plus the FR-303 preprocessing stages). The inventory gate
# asserts the live registry's process surface equals this set exactly,
# so a new tool or a dropped one forces a reviewed edit here.
LAUNCH_PROCESS_TOOLS: Final[frozenset[str]] = frozenset(
    {
        # statistical_analysis
        "analyze_hypothesis_test",
        "analyze_anova",
        "analyze_effect_sizes",
        "analyze_ab_test",
        # regression_modeling
        "analyze_regression",
        "evaluate_model_performance",
        # pattern_recognition
        "analyze_clusters",
        "detect_anomalies",
        "reduce_dimensions",
        "transform_data",
        # time_series
        "analyze_time_series",
        "forecast_time_series",
        # sampling_estimation
        "generate_sample",
        "bootstrap_statistic",
        "monte_carlo_simulate",
        "bayesian_estimate",
        # optimization
        "solve_linear_program",
        "optimize_constrained",
        "solve_assignment_problem",
        # network_graph
        "analyze_network",
        # business_intelligence
        "analyze_rfm",
        "calculate_clv",
        # geospatial_analysis (extras tier)
        "check_geospatial_capabilities",
        "analyze_spatial_autocorrelation",
        "find_spatial_hotspots",
        "calculate_spatial_distances",
        "perform_spatial_join",
        "perform_spatial_overlay",
        "aggregate_points_in_polygons",
        "optimize_route",
        "analyze_accessibility",
        "generate_service_isochrones",
        # preprocessing (FR-303 stages)
        "prepare_missing_values",
        "convert_types",
    }
)

# `main`'s harvested domain capabilities deliberately NOT exposed at
# launch — each with the rationale that keeps it off the surface (S10).
# This is FR-703's "explicitly listed in Deferred Items with rationale"
# half: the disposition of every orphan main domain function that v3
# did not register.
DEFERRED_DOMAIN_CAPABILITIES: Final[dict[str, str]] = {
    "time_series.var": (
        "Vector autoregression is a multivariate-forecasting engine "
        "whose value surfaces only alongside cointegration/Granger; "
        "the launch forecast family is univariate (§6f). Deferred to "
        "S10 as one multivariate-time-series batch."
    ),
    "time_series.cointegration": (
        "Cointegration testing pairs with VAR and Granger causality; "
        "deferred together as the multivariate-time-series batch (S10)."
    ),
    "time_series.granger_causality": (
        "Granger causality is a multivariate relationship test with no "
        "univariate launch consumer; deferred with VAR (S10)."
    ),
    "time_series.changepoint": (
        "Structural-break / changepoint detection is a distinct "
        "analytical surface from forecasting; deferred to S10 pending "
        "its own tool design rather than folded into forecast_time_series."
    ),
    "sampling.importance_sampling": (
        "Monte Carlo importance sampling needs a caller-supplied "
        "target/proposal density pair — an expression-input surface "
        "beyond the launch simulation_type set; deferred to S10."
    ),
    "sampling.mcmc": (
        "Full MCMC posterior sampling exceeds the conjugate-normal "
        "launch estimator; deferred to S10 pending a sampler design."
    ),
    "geospatial.remote_sparql_endpoints": (
        "Remote SPARQL/OSM routing backends are an additive engine "
        "kind outside the extras-tier acceptance; deferred (I-3 note)."
    ),
    "geospatial.raster_interpolation": (
        "Kriging / raster interpolation (rasterio/skgstat) is a heavy "
        "surface beyond the launch vector-analysis ten; deferred to S10."
    ),
}


def deferred_without_rationale() -> list[str]:
    """Deferred entries whose rationale is empty — a documentation
    defect the gate fails on (FR-703 requires a rationale each)."""
    return [
        name
        for name, rationale in DEFERRED_DOMAIN_CAPABILITIES.items()
        if not rationale.strip()
    ]
