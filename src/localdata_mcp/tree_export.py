"""Export tree storage back to structured formats (TOML, JSON, YAML).

Reconstructs nested Python dicts from the node/property representation
and serialises them to the target format.
"""

import json
from typing import Any, Dict, List, Optional

import toml
import yaml

from .tree_storage import (
    TreeStorageManager,
    deserialize_value,
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
    return children[0].is_array_item


def _reconstruct_node(
    manager: TreeStorageManager,
    path: str,
) -> Any:
    """Recursively reconstruct the subtree rooted at *path*."""
    props = _collect_properties(manager, path)
    children_count = manager.get_children_count(path)

    if children_count == 0:
        return props if props else {}

    if _children_are_array_items(manager, path):
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
        return items

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
    """Walk the tree and return a nested Python dict."""
    if path is not None:
        node = manager.get_node(path)
        if node is None:
            raise ValueError(f"Node not found: {path}")
        return {node.name: _reconstruct_node(manager, path)}

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


def _export_markdown(
    manager: TreeStorageManager,
    path: Optional[str] = None,
) -> str:
    """Export the tree (or subtree) as markdown via markdown_export."""
    from .markdown_export import export_tree_markdown

    tree = reconstruct_tree(manager, path)
    result = export_tree_markdown(tree)
    return result["content"]


_EXPORTERS = {
    "toml": export_toml,
    "json": export_json,
    "yaml": export_yaml,
    "markdown": _export_markdown,
    "md": _export_markdown,
}


def tool_export_structured(
    manager: TreeStorageManager,
    name: str,
    format: str,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Export tree data in the requested format.

    Returns a dict with format + content (or error).
    """
    fmt = format.lower()
    if fmt not in _EXPORTERS:
        return {
            "error": f"Unsupported format '{format}'. Use toml, json, yaml, or markdown."
        }

    try:
        output = _EXPORTERS[fmt](manager, path)
    except ValueError as exc:
        return {"error": str(exc)}

    if len(output.encode("utf-8")) > MAX_EXPORT_BYTES:
        truncated = output[: MAX_EXPORT_BYTES // 2]
        return {
            "format": fmt,
            "truncated": True,
            "content": truncated,
            "notice": (
                f"Output exceeded {MAX_EXPORT_BYTES // 1024}KB limit. "
                "Use a subtree path to export a smaller section."
            ),
        }

    return {"format": fmt, "content": output}
