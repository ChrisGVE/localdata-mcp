"""localdata_mcp/nexus/chokepoint/sql_validate — the SQL AST allow-list.

E6.2 (§7's SQL-AST detail, NFR-104): a genuine allow-list gate over
`sqlglot`, packaged as declarative per-dialect policy data
(`policy.py` + `dialects/*.py`), one shared entrypoint-independent walk
(`walker.py`), and a bounded classification LRU (`cache.py`, E6.3). The
package boundary is the §9 pre-split: adding a dialect is a one-file
data edit under `dialects/`.
"""

from .cache import ValidationCache
from .policy import BACKEND_TO_SQLGLOT_DIALECT, POLICIES, DialectPolicy
from .walker import Category, SqlClassification, SqlRefusedError, classify

__all__ = [
    "BACKEND_TO_SQLGLOT_DIALECT",
    "POLICIES",
    "Category",
    "DialectPolicy",
    "SqlClassification",
    "SqlRefusedError",
    "ValidationCache",
    "classify",
]
