"""Export tree storage back to structured formats (TOML, JSON, YAML).

Reconstructs nested Python dicts from the node/property representation
and serialises them to the target format.
"""

import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import toml
import yaml

from .tree_storage import (
    TreeStorageManager,
    ValueType,
    build_path,
    deserialize_value,
    parse_path,
)


# ---------------------------------------------------------------------------
# Maximum export payload size (bytes)
# ---------------------------------------------------------------------------

MAX_EXPORT_BYTES = 100 * 1024  # 100 KB


# ---------------------------------------------------------------------------
# Tree reconstruction
# ---------------------------------------------------------------------------


def _collect_properties(manager: TreeStorageManager, path: str) -> Dict[str, Any]:
    """Return all properties on a node as a plain dict with native values."""
    result: Dict[str, Any] = {}
    offset = 0
    limit = 200
    while True:
        props = manager.list_properties(path, offset=offset, limit=limit)
        for prop in props:
            result[prop.key] = deserialize_value(
                prop.value, prop.value_type, prop.original_repr
            )
        if len(props) < limit:
            break
        offset += limit
    return result


def _children_are_array_items(manager: TreeStorageManager, parent_path: str) -> bool:
    """Return True when *all* direct children are array items."""
    children = manager.get_children(parent_path=parent_path, offset=0, limit=1)
    if not children:
        return False
    # Check first child; if it is an array item we assume the rest are too
    # (parsers always mark entire arrays consistently).
    return children[0].is_array_item


def _reconstruct_node(
    manager: TreeStorageManager,
    path: str,
) -> Any:
    """Recursively reconstruct the subtree rooted at *path*.

    Returns a dict for regular nodes and a list for array-of-tables nodes.
    """
    props = _collect_properties(manager, path)
    children_count = manager.get_children_count(path)

    if children_count == 0:
        return props if props else {}

    if _children_are_array_items(manager, path):
        # Array of tables: collect each child as a list element
        items: List[Any] = []
        offset = 0
        limit = 200
        while True:
            children = manager.get_children(
                parent_path=path, offset=offset, limit=limit
            )
            for child in children:
                items.append(_reconstruct_node(manager, child.path))
            if len(children) < limit:
                break
            offset += limit
        # Merge any direct properties with the array under a special key?
        # No -- parsers store array-of-tables parent as a pure container.
        return items

    # Regular children: merge properties and sub-dicts
    result: Dict[str, Any] = dict(props)
    offset = 0
    limit = 200
    while True:
        children = manager.get_children(parent_path=path, offset=offset, limit=limit)
        for child in children:
            result[child.name] = _reconstruct_node(manager, child.path)
        if len(children) < limit:
            break
        offset += limit

    return result


def reconstruct_tree(
    manager: TreeStorageManager,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Walk the tree and return a nested Python dict.

    Args:
        manager: The tree storage backend.
        path: If given, reconstruct only the subtree at this path.
              If ``None``, reconstruct the full tree from all root nodes.

    Returns:
        A nested dict suitable for serialisation to TOML/JSON/YAML.
    """
    if path is not None:
        node = manager.get_node(path)
        if node is None:
            raise ValueError(f"Node not found: {path}")
        return {node.name: _reconstruct_node(manager, path)}

    # Reconstruct from all root nodes
    result: Dict[str, Any] = {}
    offset = 0
    limit = 200
    while True:
        roots = manager.get_children(parent_path=None, offset=offset, limit=limit)
        for root in roots:
            result[root.name] = _reconstruct_node(manager, root.path)
        if len(roots) < limit:
            break
        offset += limit
    return result


# ---------------------------------------------------------------------------
# Format exporters
# ---------------------------------------------------------------------------


def export_toml(
    manager: TreeStorageManager,
    path: Optional[str] = None,
) -> str:
    """Export the tree (or subtree) as a TOML string."""
    tree = reconstruct_tree(manager, path)
    return toml.dumps(tree)


def export_json(
    manager: TreeStorageManager,
    path: Optional[str] = None,
) -> str:
    """Export the tree (or subtree) as a JSON string."""
    tree = reconstruct_tree(manager, path)
    return json.dumps(tree, indent=2, default=str)


def export_yaml(
    manager: TreeStorageManager,
    path: Optional[str] = None,
) -> str:
    """Export the tree (or subtree) as a YAML string."""
    tree = reconstruct_tree(manager, path)
    return yaml.dump(tree, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# MCP tool entry point
# ---------------------------------------------------------------------------

_EXPORTERS = {
    "toml": export_toml,
    "json": export_json,
    "yaml": export_yaml,
}


def tool_export_structured(
    manager: TreeStorageManager,
    name: str,
    format: str,
    path: Optional[str] = None,
) -> str:
    """Export tree data in the requested format.

    Args:
        manager: The tree storage backend.
        name: Connection name (for error messages).
        format: One of ``"toml"``, ``"json"``, or ``"yaml"``.
        path: Optional subtree path to export.

    Returns:
        JSON-encoded string containing the exported content or an error.
    """
    fmt = format.lower()
    if fmt not in _EXPORTERS:
        return json.dumps(
            {"error": f"Unsupported format '{format}'. Use toml, json, or yaml."}
        )

    try:
        output = _EXPORTERS[fmt](manager, path)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    if len(output.encode("utf-8")) > MAX_EXPORT_BYTES:
        truncated = output[: MAX_EXPORT_BYTES // 2]
        return json.dumps(
            {
                "format": fmt,
                "truncated": True,
                "content": truncated,
                "notice": (
                    f"Output exceeded {MAX_EXPORT_BYTES // 1024}KB limit. "
                    "Use a subtree path to export a smaller section."
                ),
            }
        )

    return json.dumps({"format": fmt, "content": output})
