"""Tree storage model for structured data files (TOML, JSON, YAML).

Stores hierarchical data as a tree of nodes with attached key-value
properties, using two SQL tables: ``nodes`` (tree structure) and
``properties`` (typed key-value data on each node).
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

NODES_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER REFERENCES nodes(id),
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    depth INTEGER NOT NULL,
    is_array_item BOOLEAN DEFAULT FALSE,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)"""

PROPERTIES_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT,
    value_type TEXT NOT NULL,
    original_repr TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(node_id, key)
)"""

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_path ON nodes(path)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_depth ON nodes(depth)",
    "CREATE INDEX IF NOT EXISTS idx_properties_node ON properties(node_id)",
]


def create_tree_schema(engine: Engine) -> None:
    """Create the nodes and properties tables (idempotent)."""
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.execute(text(NODES_TABLE_SQL))
        conn.execute(text(PROPERTIES_TABLE_SQL))
        for stmt in INDEX_STATEMENTS:
            conn.execute(text(stmt))
        conn.commit()


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class ValueType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    NULL = "null"
    DATETIME = "datetime"


def infer_value_type(value: Any) -> ValueType:
    """Infer the ValueType for a Python value."""
    if value is None:
        return ValueType.NULL
    # bool must be checked before int (bool is a subclass of int)
    if isinstance(value, bool):
        return ValueType.BOOLEAN
    if isinstance(value, int):
        return ValueType.INTEGER
    if isinstance(value, float):
        return ValueType.FLOAT
    if isinstance(value, str):
        return ValueType.STRING
    if isinstance(value, list):
        return ValueType.ARRAY
    if isinstance(value, datetime):
        return ValueType.DATETIME
    raise TypeError(f"Unsupported value type: {type(value).__name__}")


def serialize_value(
    value: Any, value_type: ValueType
) -> Tuple[Optional[str], Optional[str]]:
    """Serialize a Python value to (stored_string, original_repr).

    Returns ``(None, None)`` for NULL values.
    """
    if value_type == ValueType.NULL:
        return (None, None)
    if value_type == ValueType.BOOLEAN:
        return ("true" if value else "false", None)
    if value_type == ValueType.INTEGER:
        return (str(value), None)
    if value_type == ValueType.FLOAT:
        return (str(value), repr(value))
    if value_type == ValueType.STRING:
        return (value, None)
    if value_type == ValueType.ARRAY:
        return (json.dumps(value), None)
    if value_type == ValueType.DATETIME:
        return (value.isoformat(), str(value))
    raise ValueError(f"Unknown ValueType: {value_type}")


def deserialize_value(
    value: Optional[str],
    value_type: ValueType,
    original_repr: Optional[str] = None,
) -> Any:
    """Reconstruct a Python value from its stored string representation."""
    if value_type == ValueType.NULL or value is None:
        return None
    if value_type == ValueType.BOOLEAN:
        return value.lower() == "true"
    if value_type == ValueType.INTEGER:
        return int(value)
    if value_type == ValueType.FLOAT:
        return float(value)
    if value_type == ValueType.STRING:
        return value
    if value_type == ValueType.ARRAY:
        return json.loads(value)
    if value_type == ValueType.DATETIME:
        return datetime.fromisoformat(value)
    raise ValueError(f"Unknown ValueType: {value_type}")


def infer_value_type_from_string(text_value: str) -> Tuple[ValueType, Any]:
    """Infer type from a raw string input (for set_value tool).

    Returns ``(inferred_type, converted_value)``.
    """
    if text_value.lower() in ("true", "false"):
        return (ValueType.BOOLEAN, text_value.lower() == "true")

    # Try integer
    try:
        return (ValueType.INTEGER, int(text_value))
    except ValueError:
        pass

    # Try float
    try:
        return (ValueType.FLOAT, float(text_value))
    except ValueError:
        pass

    # Try JSON array
    if text_value.startswith("["):
        try:
            parsed = json.loads(text_value)
            if isinstance(parsed, list):
                return (ValueType.ARRAY, parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try null
    if text_value.lower() == "null":
        return (ValueType.NULL, None)

    return (ValueType.STRING, text_value)


# ---------------------------------------------------------------------------
# Path escaping
# ---------------------------------------------------------------------------


def escape_path_segment(name: str) -> str:
    r"""Escape a single path segment so dots and backslashes are literal.

    ``\`` → ``\\``  then  ``.`` → ``\.``
    """
    return name.replace("\\", "\\\\").replace(".", "\\.")


def unescape_path_segment(segment: str) -> str:
    r"""Reverse :func:`escape_path_segment`.

    Raises ``ValueError`` on invalid escape sequences.
    """
    result: list[str] = []
    i = 0
    while i < len(segment):
        ch = segment[i]
        if ch == "\\":
            if i + 1 >= len(segment):
                raise ValueError("Trailing backslash in path segment")
            nxt = segment[i + 1]
            if nxt == ".":
                result.append(".")
            elif nxt == "\\":
                result.append("\\")
            else:
                raise ValueError(f"Invalid escape sequence '\\{nxt}' at position {i}")
            i += 2
        else:
            result.append(ch)
            i += 1
    return "".join(result)


def build_path(segments: List[str]) -> str:
    """Join raw key names into an escaped dot-path."""
    if not segments:
        return ""
    return ".".join(escape_path_segment(s) for s in segments)


def parse_path(path: str) -> List[str]:
    """Split an escaped dot-path into raw key names."""
    if not path:
        return []
    segments: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == "\\":
            if i + 1 >= len(path):
                raise ValueError("Trailing backslash in path")
            nxt = path[i + 1]
            if nxt == ".":
                current.append(".")
            elif nxt == "\\":
                current.append("\\")
            else:
                raise ValueError(f"Invalid escape sequence '\\{nxt}' at position {i}")
            i += 2
        elif ch == ".":
            segments.append("".join(current))
            current = []
            i += 1
        else:
            current.append(ch)
            i += 1
    segments.append("".join(current))
    return segments


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tree Storage Manager
# ---------------------------------------------------------------------------


class TreeStorageManager:
    """Manage a tree of nodes with properties backed by SQLite."""

    def __init__(self, engine: Engine):
        self.engine = engine
        create_tree_schema(engine)

    # -- node operations ----------------------------------------------------

    def create_node(
        self,
        path: str,
        is_array_item: bool = False,
    ) -> TreeNode:
        """Create a node and any missing ancestors. Idempotent."""
        segments = parse_path(path)
        if not segments:
            raise ValueError("Path must not be empty")

        now = time.time()
        created_node: Optional[TreeNode] = None

        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))

            for depth, _ in enumerate(segments):
                partial_segments = segments[: depth + 1]
                node_path = build_path(partial_segments)
                node_name = partial_segments[-1]

                row = conn.execute(
                    text("SELECT * FROM nodes WHERE path = :path"),
                    {"path": node_path},
                ).fetchone()

                if row is not None:
                    created_node = TreeNode.from_row(row)
                    continue

                parent_id = None
                if depth > 0:
                    parent_path = build_path(partial_segments[:-1])
                    parent_row = conn.execute(
                        text("SELECT id FROM nodes WHERE path = :path"),
                        {"path": parent_path},
                    ).fetchone()
                    if parent_row:
                        parent_id = parent_row[0]

                is_arr = is_array_item if (depth == len(segments) - 1) else False
                conn.execute(
                    text(
                        "INSERT INTO nodes "
                        "(parent_id, name, path, depth, is_array_item, created_at, updated_at) "
                        "VALUES (:pid, :name, :path, :depth, :arr, :ca, :ua)"
                    ),
                    {
                        "pid": parent_id,
                        "name": node_name,
                        "path": node_path,
                        "depth": depth,
                        "arr": is_arr,
                        "ca": now,
                        "ua": now,
                    },
                )
                row = conn.execute(
                    text("SELECT * FROM nodes WHERE path = :path"),
                    {"path": node_path},
                ).fetchone()
                created_node = TreeNode.from_row(row)

            conn.commit()

        assert created_node is not None
        return created_node

    def get_node(self, path: str) -> Optional[TreeNode]:
        """Get a node by its full path, or None if it doesn't exist."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM nodes WHERE path = :path"),
                {"path": path},
            ).fetchone()
            return TreeNode.from_row(row) if row else None

    def node_exists(self, path: str) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM nodes WHERE path = :path"),
                {"path": path},
            ).fetchone()
            return row is not None

    def get_children(
        self, parent_path: Optional[str] = None, offset: int = 0, limit: int = 50
    ) -> List[TreeNode]:
        """Get direct children of a node (or root nodes if parent_path is None)."""
        with self.engine.connect() as conn:
            if parent_path is None:
                rows = conn.execute(
                    text(
                        "SELECT * FROM nodes WHERE parent_id IS NULL "
                        "ORDER BY name LIMIT :lim OFFSET :off"
                    ),
                    {"lim": limit, "off": offset},
                ).fetchall()
            else:
                parent = conn.execute(
                    text("SELECT id FROM nodes WHERE path = :path"),
                    {"path": parent_path},
                ).fetchone()
                if parent is None:
                    return []
                rows = conn.execute(
                    text(
                        "SELECT * FROM nodes WHERE parent_id = :pid "
                        "ORDER BY name LIMIT :lim OFFSET :off"
                    ),
                    {"pid": parent[0], "lim": limit, "off": offset},
                ).fetchall()
            return [TreeNode.from_row(r) for r in rows]

    def get_children_count(self, parent_path: Optional[str] = None) -> int:
        with self.engine.connect() as conn:
            if parent_path is None:
                row = conn.execute(
                    text("SELECT COUNT(*) FROM nodes WHERE parent_id IS NULL")
                ).fetchone()
            else:
                parent = conn.execute(
                    text("SELECT id FROM nodes WHERE path = :path"),
                    {"path": parent_path},
                ).fetchone()
                if parent is None:
                    return 0
                row = conn.execute(
                    text("SELECT COUNT(*) FROM nodes WHERE parent_id = :pid"),
                    {"pid": parent[0]},
                ).fetchone()
            return row[0] if row else 0

    def delete_node(self, path: str) -> Tuple[int, int]:
        """Delete a node and all descendants. Returns (nodes_deleted, properties_deleted)."""
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))

            # Count properties that will be deleted
            prop_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM properties WHERE node_id IN "
                    "(SELECT id FROM nodes WHERE path = :p OR path LIKE :prefix)"
                ),
                {"p": path, "prefix": path + ".%"},
            ).fetchone()[0]

            # Delete nodes (CASCADE removes properties)
            result = conn.execute(
                text("DELETE FROM nodes WHERE path = :p OR path LIKE :prefix"),
                {"p": path, "prefix": path + ".%"},
            )
            node_count = result.rowcount

            conn.commit()
            return (node_count, prop_count)

    # -- property operations ------------------------------------------------

    def set_property(
        self,
        node_path: str,
        key: str,
        value: Any,
        value_type: Optional[ValueType] = None,
    ) -> NodeProperty:
        """Set a property on a node (upsert). Auto-creates the node if needed."""
        if value_type is None:
            value_type = infer_value_type(value)
        serialized, orig = serialize_value(value, value_type)
        now = time.time()

        # Ensure node exists
        node = self.get_node(node_path)
        if node is None:
            node = self.create_node(node_path)

        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))

            existing = conn.execute(
                text("SELECT id FROM properties WHERE node_id = :nid AND key = :key"),
                {"nid": node.id, "key": key},
            ).fetchone()

            if existing:
                conn.execute(
                    text(
                        "UPDATE properties SET value = :val, value_type = :vt, "
                        "original_repr = :orig, updated_at = :ua "
                        "WHERE id = :pid"
                    ),
                    {
                        "val": serialized,
                        "vt": value_type.value,
                        "orig": orig,
                        "ua": now,
                        "pid": existing[0],
                    },
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO properties "
                        "(node_id, key, value, value_type, original_repr, created_at, updated_at) "
                        "VALUES (:nid, :key, :val, :vt, :orig, :ca, :ua)"
                    ),
                    {
                        "nid": node.id,
                        "key": key,
                        "val": serialized,
                        "vt": value_type.value,
                        "orig": orig,
                        "ca": now,
                        "ua": now,
                    },
                )

            conn.commit()

            row = conn.execute(
                text("SELECT * FROM properties WHERE node_id = :nid AND key = :key"),
                {"nid": node.id, "key": key},
            ).fetchone()
            return NodeProperty.from_row(row)

    def get_property(self, node_path: str, key: str) -> Optional[NodeProperty]:
        """Get a single property by node path and key."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT p.* FROM properties p "
                    "JOIN nodes n ON p.node_id = n.id "
                    "WHERE n.path = :path AND p.key = :key"
                ),
                {"path": node_path, "key": key},
            ).fetchone()
            return NodeProperty.from_row(row) if row else None

    def list_properties(
        self, node_path: str, offset: int = 0, limit: int = 50
    ) -> List[NodeProperty]:
        """List properties on a node with pagination."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT p.* FROM properties p "
                    "JOIN nodes n ON p.node_id = n.id "
                    "WHERE n.path = :path "
                    "ORDER BY p.key LIMIT :lim OFFSET :off"
                ),
                {"path": node_path, "lim": limit, "off": offset},
            ).fetchall()
            return [NodeProperty.from_row(r) for r in rows]

    def get_property_count(self, node_path: str) -> int:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM properties p "
                    "JOIN nodes n ON p.node_id = n.id "
                    "WHERE n.path = :path"
                ),
                {"path": node_path},
            ).fetchone()
            return row[0] if row else 0

    def delete_property(self, node_path: str, key: str) -> bool:
        """Delete a property. Returns True if it existed."""
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            result = conn.execute(
                text(
                    "DELETE FROM properties WHERE id IN ("
                    "  SELECT p.id FROM properties p "
                    "  JOIN nodes n ON p.node_id = n.id "
                    "  WHERE n.path = :path AND p.key = :key"
                    ")"
                ),
                {"path": node_path, "key": key},
            )
            conn.commit()
            return result.rowcount > 0

    # -- tree statistics ----------------------------------------------------

    def get_tree_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the tree."""
        with self.engine.connect() as conn:
            total_nodes = conn.execute(text("SELECT COUNT(*) FROM nodes")).fetchone()[0]
            total_props = conn.execute(
                text("SELECT COUNT(*) FROM properties")
            ).fetchone()[0]
            max_depth = conn.execute(
                text("SELECT COALESCE(MAX(depth), -1) FROM nodes")
            ).fetchone()[0]
            root_count = conn.execute(
                text("SELECT COUNT(*) FROM nodes WHERE parent_id IS NULL")
            ).fetchone()[0]

            root_nodes = conn.execute(
                text(
                    "SELECT name FROM nodes WHERE parent_id IS NULL "
                    "ORDER BY name LIMIT 20"
                )
            ).fetchall()

            return {
                "total_nodes": total_nodes,
                "total_properties": total_props,
                "max_depth": max_depth,
                "root_count": root_count,
                "root_nodes": [r[0] for r in root_nodes],
            }
