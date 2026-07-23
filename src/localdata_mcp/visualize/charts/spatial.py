"""localdata_mcp/visualize/charts/spatial.py — spatial kinds (E12.1).

The two topology-bearing charts (FR-503's geospatial and network
families): `geo_map` (lon/lat point scatter, optionally value-weighted)
and `network_layout` (an edge list drawn on a DETERMINISTIC circular
layout — no random seed, so the golden PNG is reproducible across CI
runs, NFR-502/S8 SSIM). Pure extractors — the layout geometry is
computed here, the renderer only draws. Neighbors: columns.py resolves
channels; spec.py registers these; render/ draws the points/edges.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from localdata_mcp.ingest.refusals import invalid_source_refusal

from .base import ChartKind
from .columns import channel_column, require_column


def _geo_map_data(frame: pd.DataFrame, encoding: Mapping[str, Any]) -> "dict[str, Any]":
    """Lon/lat points, optionally carrying a value channel for size or
    colour — the renderer maps them."""
    lon_col = channel_column(encoding, "lon", "geo_map")
    lat_col = channel_column(encoding, "lat", "geo_map")
    columns = [lon_col, lat_col]
    value_col = encoding.get("value")
    if value_col is not None:
        columns.append(str(value_col))
    for column in columns:
        require_column(frame, column)
    points = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if points.empty:
        raise invalid_source_refusal(
            f"Columns {columns} share no rows with numeric lon/lat "
            "values — a geo map needs numeric coordinates."
        )
    data: dict[str, Any] = {
        "lon": [float(v) for v in points[lon_col]],
        "lat": [float(v) for v in points[lat_col]],
    }
    if value_col is not None:
        data["value"] = [float(v) for v in points[str(value_col)]]
    return data


def _network_layout_data(
    frame: pd.DataFrame, encoding: Mapping[str, Any]
) -> "dict[str, Any]":
    """Nodes, edges, and a deterministic circular layout over the
    source/target edge list."""
    source_col = channel_column(encoding, "source", "network_layout")
    target_col = channel_column(encoding, "target", "network_layout")
    require_column(frame, source_col)
    require_column(frame, target_col)
    edge_frame = frame[[source_col, target_col]].dropna()
    if edge_frame.empty:
        raise invalid_source_refusal(
            f"Columns {[source_col, target_col]} yield no complete "
            "edges — a network layout needs source/target pairs."
        )
    nodes = sorted(
        {str(v) for v in edge_frame[source_col]}
        | {str(v) for v in edge_frame[target_col]}
    )
    index = {node: position for position, node in enumerate(nodes)}
    edges = [
        [index[str(s)], index[str(t)]]
        for s, t in zip(edge_frame[source_col], edge_frame[target_col])
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "positions": _circular_layout(len(nodes)),
    }


def _circular_layout(count: int) -> "list[list[float]]":
    """`count` points evenly spaced on the unit circle — deterministic
    (no random seed), so the network golden is reproducible."""
    if count == 0:
        return []
    if count == 1:
        return [[0.0, 0.0]]
    return [
        [
            round(math.cos(2 * math.pi * n / count), 12),
            round(math.sin(2 * math.pi * n / count), 12),
        ]
        for n in range(count)
    ]


GEO_MAP = ChartKind(name="geo_map", mark="points", extract=_geo_map_data)
NETWORK_LAYOUT = ChartKind(
    name="network_layout", mark="edges", extract=_network_layout_data
)
