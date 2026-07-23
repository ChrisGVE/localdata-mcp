"""localdata_mcp/process/domains/pattern_recognition/tools.py — E10.c ToolSpecs.

The pattern family's tools, carried by name from `main`
(DR GP2): `analyze_clusters` (SCALAR verdict), `assign_clusters` (its
TABULAR composable counterpart — labeled rows a downstream stage
consumes, E12.5), `detect_anomalies`, `reduce_dimensions`
(FR-308's explained-variance closure), `transform_data` (regex column
rewrite, TABULAR out — a composable stage). Thin over
clustering/anomalies/reduction/transform.py with the X-2 addressing
contract; stochastic tools carry `seed` (S3.3 determinism).
Neighbors: matrices.py preps input; spec_modules.py rosters this
module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..support import addressed_frame, source_params
from .anomalies import find_anomalies
from .clustering import assign_cluster_frame, perform_clustering
from .reduction import reduce_to_components
from .transform import transform_column

_COLUMNS = Param(
    "columns",
    list,
    "Columns to analyze (default: every numeric column).",
    required=False,
)
_SEED = Param(
    "seed",
    int,
    "Random seed pinning stochastic steps (default: library behavior).",
    required=False,
)
_ALGORITHM_PARAMS = Param(
    "algorithm_params",
    dict,
    "Estimator constructor parameters, passed through verbatim.",
    required=False,
)


@tool_spec(
    name="analyze_clusters",
    summary=(
        "Cluster an addressed tabular source: method kmeans (default), "
        "hierarchical, dbscan, gmm, or spectral. Without n_clusters= a "
        "silhouette sweep picks k. Reports labels, cluster sizes, and "
        "silhouette score; seed= pins stochastic initialization."
    ),
    params=(
        *source_params(),
        _COLUMNS,
        Param(
            "method",
            str,
            "kmeans (default), hierarchical, dbscan, gmm, spectral.",
            required=False,
        ),
        Param(
            "n_clusters",
            int,
            "Cluster count (default: silhouette sweep over 2..8).",
            required=False,
        ),
        _SEED,
        _ALGORITHM_PARAMS,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="pattern_recognition",
)
def analyze_clusters(
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = perform_clustering(frame, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="assign_clusters",
    summary=(
        "Cluster an addressed tabular source and return the clustered "
        "rows tagged with an integer cluster label — the composable "
        "(TABULAR) counterpart to analyze_clusters' verdict, so a "
        "clustering result feeds a downstream stage (e.g. a chart "
        "coloured by cluster). method kmeans (default), hierarchical, "
        "dbscan, gmm, spectral; without n_clusters a silhouette sweep "
        "picks k; seed pins stochastic initialization."
    ),
    params=(
        *source_params(),
        _COLUMNS,
        Param(
            "method",
            str,
            "kmeans (default), hierarchical, dbscan, gmm, spectral.",
            required=False,
        ),
        Param(
            "n_clusters",
            int,
            "Cluster count (default: silhouette sweep over 2..8).",
            required=False,
        ),
        _SEED,
        _ALGORITHM_PARAMS,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.TABULAR,
    domain="pattern_recognition",
)
def assign_clusters(
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = assign_cluster_frame(frame, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="detect_anomalies",
    summary=(
        "Find anomalous rows in an addressed tabular source: method "
        "isolation_forest (default), lof, or zscore (three-sigma rule). "
        "Reports anomaly indices, share, and a score summary."
    ),
    params=(
        *source_params(),
        _COLUMNS,
        Param(
            "method",
            str,
            "isolation_forest (default), lof, or zscore.",
            required=False,
        ),
        Param(
            "contamination",
            float,
            "Expected anomaly share (implementation default 0.1).",
            required=False,
        ),
        _SEED,
        _ALGORITHM_PARAMS,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="pattern_recognition",
)
def detect_anomalies(
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = find_anomalies(frame, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="reduce_dimensions",
    summary=(
        "Embed an addressed tabular source into fewer dimensions: "
        "method pca (default, always reports explained_variance_ratio) "
        "or tsne (reports trustworthiness against the original data)."
    ),
    params=(
        *source_params(),
        _COLUMNS,
        Param("method", str, "pca (default) or tsne.", required=False),
        Param(
            "n_components",
            int,
            "Target dimensionality (implementation default 2).",
            required=False,
        ),
        _SEED,
        _ALGORITHM_PARAMS,
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.MATRIX,
    domain="pattern_recognition",
)
def reduce_dimensions(
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = reduce_to_components(frame, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="transform_data",
    summary=(
        "Regex find/replace over one column of an addressed tabular "
        "source (pattern crosses the hardened safety screen). Returns "
        "the rewritten relation plus a change summary — composable "
        "into downstream stages."
    ),
    params=(
        *source_params(),
        Param("column", str, "The column to rewrite."),
        Param("find", str, "The regex pattern to find (safety-screened)."),
        Param("replace", str, "The replacement text (backrefs allowed)."),
        Param(
            "case_sensitive",
            bool,
            "Match case-sensitively (implementation default true).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.TABULAR,
    domain="pattern_recognition",
)
def transform_data(
    column: str,
    find: str,
    replace: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = transform_column(frame, column, find, replace, **knobs)
    result["source"] = source
    return result
