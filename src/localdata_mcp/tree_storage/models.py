"""Tree node and property dataclasses."""

from dataclasses import dataclass
from typing import Any, Optional

from localdata_mcp.tree_storage.serialization import deserialize_value
from localdata_mcp.tree_storage.types import ValueType


@dataclass
class TreeNode:
    """A node in the tree."""

    id: int
    parent_id: Optional[int]
    name: str
    path: str
    depth: int
    is_array_item: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row) -> "TreeNode":
        """Create from a SQLAlchemy Row or dict-like mapping."""
        mapping = row._mapping if hasattr(row, "_mapping") else row
        return cls(
            id=mapping["id"],
            parent_id=mapping["parent_id"],
            name=mapping["name"],
            path=mapping["path"],
            depth=mapping["depth"],
            is_array_item=bool(mapping["is_array_item"]),
            created_at=mapping["created_at"],
            updated_at=mapping["updated_at"],
        )


@dataclass
class NodeProperty:
    """A typed key-value pair attached to a node."""

    id: int
    node_id: int
    key: str
    value: Optional[str]
    value_type: ValueType
    original_repr: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row) -> "NodeProperty":
        """Create from a SQLAlchemy Row or dict-like mapping."""
        mapping = row._mapping if hasattr(row, "_mapping") else row
        return cls(
            id=mapping["id"],
            node_id=mapping["node_id"],
            key=mapping["key"],
            value=mapping["value"],
            value_type=ValueType(mapping["value_type"]),
            original_repr=mapping.get("original_repr"),
            created_at=mapping["created_at"],
            updated_at=mapping["updated_at"],
        )

    def to_python_value(self) -> Any:
        """Deserialize to the native Python value."""
        return deserialize_value(self.value, self.value_type, self.original_repr)
