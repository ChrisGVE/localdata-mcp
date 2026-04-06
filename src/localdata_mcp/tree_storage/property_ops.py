"""Property CRUD and tree statistics operations mixin."""

import time
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from localdata_mcp.tree_storage.models import NodeProperty
from localdata_mcp.tree_storage.serialization import (
    infer_value_type,
    serialize_value,
)
from localdata_mcp.tree_storage.types import ValueType


class PropertyOperationsMixin:
    """Mixin providing property CRUD and stats. Requires ``self.engine``."""

    engine: Engine

    def set_property(
        self,
        node_path: str,
        key: str,
        value: Any,
        value_type: Optional[ValueType] = None,
    ) -> NodeProperty:
        """Set a property on a node (upsert). Auto-creates the node."""
        if value_type is None:
            value_type = infer_value_type(value)
        serialized, orig = serialize_value(value, value_type)
        now = time.time()

        node = self.get_node(node_path)  # type: ignore[attr-defined]
        if node is None:
            node = self.create_node(node_path)  # type: ignore[attr-defined]

        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))

            existing = conn.execute(
                text("SELECT id FROM properties WHERE node_id = :nid AND key = :key"),
                {"nid": node.id, "key": key},
            ).fetchone()

            if existing:
                conn.execute(
                    text(
                        "UPDATE properties SET value = :val, "
                        "value_type = :vt, original_repr = :orig, "
                        "updated_at = :ua WHERE id = :pid"
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
                        "(node_id, key, value, value_type, "
                        "original_repr, created_at, updated_at) "
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
        """Count properties on a node."""
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
                    "SELECT name FROM nodes "
                    "WHERE parent_id IS NULL "
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
