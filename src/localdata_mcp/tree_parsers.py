"""Parsers for loading TOML, JSON, and YAML files into tree storage.

Each parser reads a file, converts the hierarchical data into nodes and
properties via :class:`TreeStorageManager`, using :func:`parse_dict_to_tree`
as the shared recursive engine.
"""

import json
from typing import Any, List, Optional

import toml
import yaml

from localdata_mcp.tree_storage import (
    TreeStorageManager,
    ValueType,
    build_path,
    infer_value_type,
)


def _is_list_of_dicts(value: Any) -> bool:
    """Return True if *value* is a non-empty list where every item is a dict."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, dict) for item in value)
    )


def parse_dict_to_tree(
    data: dict,
    manager: TreeStorageManager,
    parent_segments: Optional[List[str]] = None,
) -> None:
    """Recursively store a nested dict as tree nodes and properties.

    Args:
        data: The dictionary to store.
        manager: The tree storage manager to write into.
        parent_segments: Path segments of the current node (None for root).
    """
    if parent_segments is None:
        parent_segments = []

    for key, value in data.items():
        child_segments = [*parent_segments, key]

        if isinstance(value, dict):
            child_path = build_path(child_segments)
            manager.create_node(child_path)
            parse_dict_to_tree(value, manager, child_segments)

        elif _is_list_of_dicts(value):
            # Array of tables: create numbered children (key.0, key.1, ...)
            child_path = build_path(child_segments)
            manager.create_node(child_path)
            for idx, item in enumerate(value):
                item_segments = [*child_segments, str(idx)]
                item_path = build_path(item_segments)
                manager.create_node(item_path, is_array_item=True)
                parse_dict_to_tree(item, manager, item_segments)

        else:
            # Scalar or list of scalars: store as property on parent node
            if parent_segments:
                node_path = build_path(parent_segments)
                # Ensure parent node exists
                if not manager.node_exists(node_path):
                    manager.create_node(node_path)
            else:
                # Root-level scalar: store on a virtual "root" node
                node_path = build_path(["root"])
                if not manager.node_exists(node_path):
                    manager.create_node(node_path)

            vtype = infer_value_type(value)
            manager.set_property(node_path, key, value, vtype)


def parse_toml_to_tree(
    file_path: str,
    manager: TreeStorageManager,
) -> None:
    """Parse a TOML file and store its contents as a tree.

    Args:
        file_path: Path to the TOML file.
        manager: The tree storage manager to write into.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        data = toml.load(fh)
    parse_dict_to_tree(data, manager)


def parse_json_to_tree(
    file_path: str,
    manager: TreeStorageManager,
) -> None:
    """Parse a JSON file and store its contents as a tree.

    If the root is a list of dicts, numbered root nodes (root.0, root.1, ...)
    are created. If the root is a dict, :func:`parse_dict_to_tree` handles it.

    Args:
        file_path: Path to the JSON file.
        manager: The tree storage manager to write into.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        root_path = "root"
        manager.create_node(root_path)
        for idx, item in enumerate(data):
            item_segments = ["root", str(idx)]
            item_path = build_path(item_segments)
            manager.create_node(item_path, is_array_item=True)
            if isinstance(item, dict):
                parse_dict_to_tree(item, manager, item_segments)
            else:
                vtype = infer_value_type(item)
                manager.set_property(item_path, "value", item, vtype)
    elif isinstance(data, dict):
        parse_dict_to_tree(data, manager)
    else:
        raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")


def parse_yaml_to_tree(
    file_path: str,
    manager: TreeStorageManager,
) -> None:
    """Parse a YAML file (multi-document aware) and store as a tree.

    Single documents are stored directly via :func:`parse_dict_to_tree`.
    Multiple documents are stored under ``doc_0``, ``doc_1``, etc.

    Args:
        file_path: Path to the YAML file.
        manager: The tree storage manager to write into.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        documents = list(yaml.safe_load_all(fh))

    # Filter out None documents (empty YAML docs)
    documents = [d for d in documents if d is not None]

    if not documents:
        return

    if len(documents) == 1:
        doc = documents[0]
        if isinstance(doc, dict):
            parse_dict_to_tree(doc, manager)
        else:
            raise ValueError(f"Unsupported YAML document type: {type(doc).__name__}")
    else:
        for idx, doc in enumerate(documents):
            doc_segments = [f"doc_{idx}"]
            doc_path = build_path(doc_segments)
            manager.create_node(doc_path)
            if isinstance(doc, dict):
                parse_dict_to_tree(doc, manager, doc_segments)
