"""localdata_mcp/server/skeleton_tools.py — walking-skeleton ToolSpecs.

Trivial but REAL pure tools (each computes its declared output shape,
no stubs) proving the NX-1 one-declaration pipeline end to end: spec
here -> generated wrapper/docs/test-stub/shape-registry artifacts ->
FastMCP registration in mcp_app.py. E3.6 keeps one tool per
non-DYNAMIC TypeShape so FR-701/704 acceptance runs against a
populated registry; later epics land the real tool surface. Registered
via nexus/contract/spec_modules.py's roster — never imported directly
by the server.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec


@tool_spec(
    name="ping",
    summary="Report server liveness with a constant probe response.",
    params=[],
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.SCALAR,
)
def ping() -> str:
    return "pong"


@tool_spec(
    name="probe_table",
    summary="Produce a small numbered table of squares for pipeline probing.",
    params=[Param("rows", int, "How many rows the probe table carries.")],
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.TABULAR,
)
def probe_table(rows: int) -> dict[str, Any]:
    return {
        "columns": ["n", "square"],
        "rows": [[n, n * n] for n in range(rows)],
    }


@tool_spec(
    name="probe_vector",
    summary="Produce an ordered series of triangular numbers for probing.",
    params=[Param("length", int, "How many entries the series carries.")],
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.VECTOR,
)
def probe_vector(length: int) -> list[int]:
    return [n * (n + 1) // 2 for n in range(length)]


@tool_spec(
    name="probe_matrix",
    summary="Produce an identity matrix of the requested size for probing.",
    params=[Param("size", int, "Row and column count of the matrix.")],
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.MATRIX,
)
def probe_matrix(size: int) -> list[list[int]]:
    return [[1 if row == col else 0 for col in range(size)] for row in range(size)]


@tool_spec(
    name="probe_model",
    summary="Fit a line to a tiny generated sample and report coefficients.",
    params=[Param("points", int, "Sample size drawn from y = 2n + 1.")],
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.FITTED_MODEL,
)
def probe_model(points: int) -> dict[str, float]:
    xs = list(range(points))
    ys = [2 * x + 1 for x in xs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
        if var_x
        else 0.0
    )
    return {"slope": slope, "intercept": mean_y - slope * mean_x}


@tool_spec(
    name="probe_graph",
    summary="Produce a path graph with the requested node count for probing.",
    params=[Param("nodes", int, "How many nodes the path graph carries.")],
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.GRAPH,
)
def probe_graph(nodes: int) -> dict[str, Any]:
    return {
        "nodes": list(range(nodes)),
        "edges": [[n, n + 1] for n in range(nodes - 1)],
    }


@tool_spec(
    name="probe_geo",
    summary="Produce evenly spaced points along the equator for probing.",
    params=[Param("points", int, "How many geometry-bearing rows to emit.")],
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.GEO,
)
def probe_geo(points: int) -> dict[str, Any]:
    return {
        "columns": ["name", "geometry"],
        "rows": [
            [f"p{n}", {"type": "Point", "coordinates": [float(n), 0.0]}]
            for n in range(points)
        ],
    }


@tool_spec(
    name="probe_chart",
    summary="Build a line-chart specification over computed square values.",
    params=[Param("points", int, "How many x/y pairs the chart spec plots.")],
    input_shape=TypeShape.NONE,
    output_shape=TypeShape.CHART_SPEC,
)
def probe_chart(points: int) -> dict[str, Any]:
    xs = list(range(points))
    return {"kind": "line", "x": xs, "y": [x * x for x in xs]}


@tool_spec(
    name="probe_sink",
    summary="Measure a text payload and report its size as a terminal result.",
    params=[Param("text", str, "Payload whose size the sink reports.")],
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.NONE,
)
def probe_sink(text: str) -> dict[str, int]:
    return {"characters": len(text), "words": len(text.split())}
