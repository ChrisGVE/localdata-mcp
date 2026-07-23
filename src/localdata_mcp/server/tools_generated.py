"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.wrapper — DO NOT EDIT.

FastMCP registration wrappers for every registered ToolSpec
(ARCHITECTURE.md 6.1 artifacts 1+2). Regenerate via
`python -m localdata_mcp.nexus.contract.generate`; hand edits fail CI
through nexus/contract/check_drift.py.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.nexus.response.shaping import shaped_call


def register_tools(app: FastMCP) -> None:
    """Register every generated tool wrapper on `app`."""
    load_spec_modules()
    registry = default_registry()

    _impl_ping = registry.lookup("ping").func

    def ping() -> Any:
        """
        Report server liveness with a constant probe response.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        """
        return shaped_call("ping", _impl_ping, {})

    app.tool(ping)

    _impl_probe_table = registry.lookup("probe_table").func

    def probe_table(rows: int) -> Any:
        """
        Produce a small numbered table of squares for pipeline probing.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: no.

        Args:
            rows: How many rows the probe table carries.
        """
        return shaped_call("probe_table", _impl_probe_table, {"rows": rows})

    app.tool(probe_table)

    _impl_probe_vector = registry.lookup("probe_vector").func

    def probe_vector(length: int) -> Any:
        """
        Produce an ordered series of triangular numbers for probing.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: VECTOR.
        Streaming-capable: no.

        Args:
            length: How many entries the series carries.
        """
        return shaped_call("probe_vector", _impl_probe_vector, {"length": length})

    app.tool(probe_vector)

    _impl_probe_matrix = registry.lookup("probe_matrix").func

    def probe_matrix(size: int) -> Any:
        """
        Produce an identity matrix of the requested size for probing.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: MATRIX.
        Streaming-capable: no.

        Args:
            size: Row and column count of the matrix.
        """
        return shaped_call("probe_matrix", _impl_probe_matrix, {"size": size})

    app.tool(probe_matrix)

    _impl_probe_model = registry.lookup("probe_model").func

    def probe_model(points: int) -> Any:
        """
        Fit a line to a tiny generated sample and report coefficients.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: FITTED_MODEL.
        Streaming-capable: no.

        Args:
            points: Sample size drawn from y = 2n + 1.
        """
        return shaped_call("probe_model", _impl_probe_model, {"points": points})

    app.tool(probe_model)

    _impl_probe_graph = registry.lookup("probe_graph").func

    def probe_graph(nodes: int) -> Any:
        """
        Produce a path graph with the requested node count for probing.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: GRAPH.
        Streaming-capable: no.

        Args:
            nodes: How many nodes the path graph carries.
        """
        return shaped_call("probe_graph", _impl_probe_graph, {"nodes": nodes})

    app.tool(probe_graph)

    _impl_probe_geo = registry.lookup("probe_geo").func

    def probe_geo(points: int) -> Any:
        """
        Produce evenly spaced points along the equator for probing.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: GEO.
        Streaming-capable: no.

        Args:
            points: How many geometry-bearing rows to emit.
        """
        return shaped_call("probe_geo", _impl_probe_geo, {"points": points})

    app.tool(probe_geo)

    _impl_probe_chart = registry.lookup("probe_chart").func

    def probe_chart(points: int) -> Any:
        """
        Build a line-chart specification over computed square values.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: CHART_SPEC.
        Streaming-capable: no.

        Args:
            points: How many x/y pairs the chart spec plots.
        """
        return shaped_call("probe_chart", _impl_probe_chart, {"points": points})

    app.tool(probe_chart)

    _impl_probe_sink = registry.lookup("probe_sink").func

    def probe_sink(text: str) -> Any:
        """
        Measure a text payload and report its size as a terminal result.

        Input shape: TABULAR.
        Output shape: NONE (chain endpoint — composes with nothing).
        Streaming-capable: no.

        Args:
            text: Payload whose size the sink reports.
        """
        return shaped_call("probe_sink", _impl_probe_sink, {"text": text})

    app.tool(probe_sink)
