"""Parsers for loading DOT, GML, and GraphML files into graph storage.

Each parser reads a graph file via NetworkX, then populates a
:class:`GraphStorageManager` using :func:`_networkx_to_storage` as the
shared conversion engine.
"""

import json
import os
from typing import Any, Dict

import networkx as nx
import pydot

from localdata_mcp.graph_storage import GraphStorageManager


# Fields handled specially during node/edge import (not stored as properties)
_NODE_SKIP_KEYS = {"label"}
_EDGE_SKIP_KEYS = {"label", "weight"}


def _coerce_value(value: Any) -> Any:
    """Convert non-primitive values to a JSON string for storage.

    Dicts and lists-of-dicts are serialized so that
    :func:`infer_value_type` never encounters an unsupported type.
    """
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def _networkx_to_storage(
    G: nx.DiGraph,
    manager: GraphStorageManager,
) -> Dict[str, Any]:
    """Convert a NetworkX DiGraph into graph storage rows.

    Nodes are created via :meth:`manager.create_node` with the ``label``
    attribute (if present).  Remaining node attributes are stored as
    properties.  Edges are added with ``label`` and ``weight`` mapped to
    the corresponding storage fields; other attributes become edge
    properties.

    Args:
        G: A directed NetworkX graph.
        manager: The graph storage manager to populate.

    Returns:
        Summary statistics dict from :meth:`manager.get_graph_stats`.
    """
    for node_id, attrs in G.nodes(data=True):
        nid = str(node_id)
        label = attrs.get("label")
        if isinstance(label, str):
            # pydot wraps labels in quotes — strip them
            label = label.strip('"')
        manager.create_node(nid, label=label)

        for key, value in attrs.items():
            if key in _NODE_SKIP_KEYS:
                continue
            val = value.strip('"') if isinstance(value, str) else _coerce_value(value)
            manager.set_property("node", nid, key, val)

    for source, target, attrs in G.edges(data=True):
        src = str(source)
        tgt = str(target)

        raw_label = attrs.get("label")
        label = raw_label.strip('"') if isinstance(raw_label, str) else raw_label

        raw_weight = attrs.get("weight")
        weight: float | None = None
        if raw_weight is not None:
            try:
                weight = float(str(raw_weight).strip('"'))
            except (ValueError, TypeError):
                weight = None

        edge = manager.add_edge(src, tgt, label=label, weight=weight)

        for key, value in attrs.items():
            if key in _EDGE_SKIP_KEYS:
                continue
            val = value.strip('"') if isinstance(value, str) else _coerce_value(value)
            manager.set_property("edge", str(edge.id), key, val)

    return manager.get_graph_stats()


def _validate_file(file_path: str) -> None:
    """Raise ValueError if *file_path* does not exist or is not a file."""
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")


def _normalize_gml_attrs(G: nx.Graph) -> None:
    """Flatten GML-specific nested attributes in-place.

    GML stores display labels in ``LabelGraphics.text`` and colors in
    ``graphics.fill``.  This function promotes those to top-level
    ``label`` and ``color`` attributes, then removes the nested dicts.
    """
    for _node_id, attrs in G.nodes(data=True):
        lg = attrs.pop("LabelGraphics", None)
        if isinstance(lg, dict) and "text" in lg:
            attrs["label"] = lg["text"]
        gfx = attrs.pop("graphics", None)
        if isinstance(gfx, dict) and "fill" in gfx:
            attrs["color"] = gfx["fill"]


def parse_dot_to_graph(
    file_path: str,
    manager: GraphStorageManager,
) -> Dict[str, Any]:
    """Parse a DOT (Graphviz) file and store its graph.

    Args:
        file_path: Path to the ``.dot`` file.
        manager: The graph storage manager to populate.

    Returns:
        Summary statistics dict.

    Raises:
        ValueError: If the file is missing or cannot be parsed.
    """
    _validate_file(file_path)
    try:
        graphs = pydot.graph_from_dot_file(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse DOT file {file_path}: {exc}") from exc

    if not graphs:
        raise ValueError(f"No graph found in DOT file: {file_path}")

    G: nx.DiGraph = nx.drawing.nx_pydot.from_pydot(graphs[0])
    if not G.is_directed():
        G = G.to_directed()
    return _networkx_to_storage(G, manager)


def parse_gml_to_graph(
    file_path: str,
    manager: GraphStorageManager,
) -> Dict[str, Any]:
    """Parse a GML file and store its graph.

    Args:
        file_path: Path to the ``.gml`` file.
        manager: The graph storage manager to populate.

    Returns:
        Summary statistics dict.

    Raises:
        ValueError: If the file is missing or cannot be parsed.
    """
    _validate_file(file_path)
    try:
        G = nx.read_gml(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse GML file {file_path}: {exc}") from exc

    if not G.is_directed():
        G = G.to_directed()
    _normalize_gml_attrs(G)
    return _networkx_to_storage(G, manager)


def parse_graphml_to_graph(
    file_path: str,
    manager: GraphStorageManager,
) -> Dict[str, Any]:
    """Parse a GraphML file and store its graph.

    Args:
        file_path: Path to the ``.graphml`` file.
        manager: The graph storage manager to populate.

    Returns:
        Summary statistics dict.

    Raises:
        ValueError: If the file is missing or cannot be parsed.
    """
    _validate_file(file_path)
    try:
        G = nx.read_graphml(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse GraphML file {file_path}: {exc}") from exc

    if not G.is_directed():
        G = G.to_directed()
    return _networkx_to_storage(G, manager)
