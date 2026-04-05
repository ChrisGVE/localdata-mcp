"""Tree storage model for structured data files (TOML, JSON, YAML).

Stores hierarchical data as a tree of nodes with attached key-value
properties, using two SQL tables: ``nodes`` (tree structure) and
``properties`` (typed key-value data on each node).
"""

from localdata_mcp.tree_storage.manager import TreeStorageManager
from localdata_mcp.tree_storage.models import NodeProperty, TreeNode
from localdata_mcp.tree_storage.paths import (
    build_path,
    escape_path_segment,
    parse_path,
    unescape_path_segment,
)
from localdata_mcp.tree_storage.schema import (
    INDEX_STATEMENTS,
    NODES_TABLE_SQL,
    PROPERTIES_TABLE_SQL,
    create_tree_schema,
)
from localdata_mcp.tree_storage.serialization import (
    deserialize_value,
    infer_value_type,
    infer_value_type_from_string,
    serialize_value,
)
from localdata_mcp.tree_storage.types import ValueType

__all__ = [
    "ValueType",
    "infer_value_type",
    "serialize_value",
    "deserialize_value",
    "infer_value_type_from_string",
    "escape_path_segment",
    "unescape_path_segment",
    "build_path",
    "parse_path",
    "TreeNode",
    "NodeProperty",
    "NODES_TABLE_SQL",
    "PROPERTIES_TABLE_SQL",
    "INDEX_STATEMENTS",
    "create_tree_schema",
    "TreeStorageManager",
]
