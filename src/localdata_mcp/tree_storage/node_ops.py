"""Node CRUD operations mixin for TreeStorageManager."""

import time
from typing import List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from localdata_mcp.tree_storage.models import TreeNode
from localdata_mcp.tree_storage.paths import build_path, parse_path


class NodeOperationsMixin:
    """Mixin providing node CRUD methods. Requires ``self.engine``."""

    engine: Engine

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
                        "(parent_id, name, path, depth, is_array_item, "
                        "created_at, updated_at) "
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
        """Check whether a node exists at the given path."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM nodes WHERE path = :path"),
                {"path": path},
            ).fetchone()
            return row is not None

    def get_children(
        self,
        parent_path: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> List[TreeNode]:
        """Get direct children of a node (root nodes if parent is None)."""
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
        """Count direct children of a node (root nodes if parent is None)."""
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
        """Delete a node and all descendants.

        Returns (nodes_deleted, properties_deleted).
        """
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))

            prop_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM properties WHERE node_id IN "
                    "(SELECT id FROM nodes "
                    "WHERE path = :p OR path LIKE :prefix)"
                ),
                {"p": path, "prefix": path + ".%"},
            ).fetchone()[0]

            result = conn.execute(
                text("DELETE FROM nodes WHERE path = :p OR path LIKE :prefix"),
                {"p": path, "prefix": path + ".%"},
            )
            node_count = result.rowcount

            conn.commit()
            return (node_count, prop_count)

    def move_node(self, path: str, new_parent: Optional[str] = None) -> Tuple[int, str]:
        """Move a node (and its subtree) under a new parent.

        Args:
            path: The node to move.
            new_parent: Target parent path, or ``None`` for root.

        Returns:
            ``(nodes_moved, new_path)``.

        Raises:
            ValueError: If the node doesn't exist, the target parent
                doesn't exist, or the target is inside the subtree.
        """
        node = self.get_node(path)
        if node is None:
            raise ValueError(f"Node not found: {path}")

        if new_parent is not None:
            if new_parent == path or new_parent.startswith(path + "."):
                raise ValueError(
                    f"Cannot move '{path}' under its own subtree '{new_parent}'."
                )
            if not self.node_exists(new_parent):
                raise ValueError(f"Target parent not found: {new_parent}")

        segments = parse_path(path)
        node_name = segments[-1]

        if new_parent is not None:
            new_path = build_path(parse_path(new_parent) + [node_name])
            new_parent_node = self.get_node(new_parent)
            new_parent_id = new_parent_node.id if new_parent_node else None
            new_depth_base = (new_parent_node.depth + 1) if new_parent_node else 0
        else:
            new_path = build_path([node_name])
            new_parent_id = None
            new_depth_base = 0

        if self.node_exists(new_path) and new_path != path:
            raise ValueError(f"A node already exists at '{new_path}'.")

        old_prefix = path
        new_prefix = new_path
        old_depth = node.depth
        now = time.time()

        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))

            rows = conn.execute(
                text(
                    "SELECT id, path, depth FROM nodes "
                    "WHERE path = :p OR path LIKE :prefix "
                    "ORDER BY depth"
                ),
                {"p": old_prefix, "prefix": old_prefix + ".%"},
            ).fetchall()

            for row in rows:
                nid, old_p, old_d = row[0], row[1], row[2]
                if old_p == old_prefix:
                    updated_path = new_prefix
                    updated_parent_id = new_parent_id
                else:
                    suffix = old_p[len(old_prefix) :]
                    updated_path = new_prefix + suffix
                    parent_segs = parse_path(updated_path)[:-1]
                    parent_path = build_path(parent_segs)
                    parent_row = conn.execute(
                        text("SELECT id FROM nodes WHERE path = :p"),
                        {"p": parent_path},
                    ).fetchone()
                    updated_parent_id = parent_row[0] if parent_row else None

                updated_depth = old_d - old_depth + new_depth_base

                conn.execute(
                    text(
                        "UPDATE nodes SET path = :new_path, "
                        "parent_id = :pid, depth = :depth, "
                        "updated_at = :ua WHERE id = :nid"
                    ),
                    {
                        "new_path": updated_path,
                        "pid": updated_parent_id,
                        "depth": updated_depth,
                        "ua": now,
                        "nid": nid,
                    },
                )

            conn.commit()
            return (len(rows), new_path)
