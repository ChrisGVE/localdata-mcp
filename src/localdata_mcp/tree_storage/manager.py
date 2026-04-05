"""TreeStorageManager: manages a tree of nodes with properties in SQLite."""

from sqlalchemy.engine import Engine

from localdata_mcp.tree_storage.node_ops import NodeOperationsMixin
from localdata_mcp.tree_storage.property_ops import PropertyOperationsMixin
from localdata_mcp.tree_storage.schema import create_tree_schema


class TreeStorageManager(NodeOperationsMixin, PropertyOperationsMixin):
    """Manage a tree of nodes with properties backed by SQLite.

    Composes node operations (:class:`NodeOperationsMixin`) and property
    operations (:class:`PropertyOperationsMixin`) into a single manager
    backed by a SQLAlchemy :class:`Engine`.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        create_tree_schema(engine)
