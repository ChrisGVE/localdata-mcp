"""Database schema creation for tree storage tables."""

from typing import List

from sqlalchemy import text
from sqlalchemy.engine import Engine

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

INDEX_STATEMENTS: List[str] = [
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
