"""localdata_mcp/nexus/persistence/store_schemas.py — store DDL, one home (E8.3).

The harvested successors of `tree_storage/schema.py` and
`graph_storage.py`'s DDL: the SQLite table shapes behind the declared
kv/tree and graph endpoint kinds (FR-103). Schema creation is an
ENGINE-CREATION concern, not a guarded operation — DDL sits outside
every NX-6 category by design, so the tables exist before the first
tool call the same way SQLite's `query_only` pragma does: applied by
engines.py when the handle is built (read-write posture only; a
read-only store endpoint expects a pre-seeded file). The tree schema
keeps the FK `ON DELETE CASCADE` from `main` — every store connection
opens with `PRAGMA foreign_keys = ON` (engines.py's connect listener),
so a node delete cascades its properties in one statement. The graph
schema keeps `main`'s manual-cascade design (TEXT references and a
polymorphic owner pattern SQLite cannot cascade). Neighbors:
engines.py calls `ensure_store_schema` at handle creation;
ingest/connectors/{kv,graph_tree} speak these table shapes through
the guard.
"""

from __future__ import annotations

from typing import Mapping

from sqlalchemy import text
from sqlalchemy.engine import Engine

# -- the kv/tree shape (harvested: tree_storage/schema.py) ------------

TREE_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """\
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER REFERENCES nodes(id),
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    depth INTEGER NOT NULL,
    is_array_item BOOLEAN DEFAULT FALSE,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)""",
    """\
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
)""",
    "CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_path ON nodes(path)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_depth ON nodes(depth)",
    "CREATE INDEX IF NOT EXISTS idx_properties_node ON properties(node_id)",
)

# -- the graph shape (harvested: graph_storage.py) --------------------

GRAPH_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """\
CREATE TABLE IF NOT EXISTS graph_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL UNIQUE,
    label TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)""",
    """\
CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES graph_nodes(node_id),
    target_id TEXT NOT NULL REFERENCES graph_nodes(node_id),
    label TEXT,
    weight REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(source_id, target_id, label)
)""",
    """\
CREATE TABLE IF NOT EXISTS graph_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_type TEXT NOT NULL CHECK(owner_type IN ('node', 'edge')),
    owner_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    value_type TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(owner_type, owner_id, key)
)""",
    "CREATE INDEX IF NOT EXISTS idx_graph_nodes_node_id ON graph_nodes(node_id)",
    "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id)",
    (
        "CREATE INDEX IF NOT EXISTS idx_graph_properties_owner "
        "ON graph_properties(owner_type, owner_id)"
    ),
)

# The declared store kinds and the schema each carries: kv and tree
# share the node/property shape (a kv store IS a flat tree — `main`'s
# kv tool family served tree-backed connections), graph carries the
# node/edge/property triple.
STORE_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "kv": TREE_SCHEMA_STATEMENTS,
    "tree": TREE_SCHEMA_STATEMENTS,
    "graph": GRAPH_SCHEMA_STATEMENTS,
}


def ensure_store_schema(engine: Engine, store_kind: str) -> None:
    """Create the kind's tables and indexes (idempotent) — called by
    engines.py at read-write handle creation, never from a tool path."""
    with engine.connect() as connection:
        for statement in STORE_SCHEMAS[store_kind]:
            connection.execute(text(statement))
        connection.commit()
