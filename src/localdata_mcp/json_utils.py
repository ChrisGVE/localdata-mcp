"""Safe JSON serialization for database values.

Handles types that the stdlib json module cannot serialize:
decimal.Decimal, numpy scalars, datetime objects, bytes, etc.
"""

import datetime
import json
from decimal import Decimal
from typing import Any


def _safe_default(obj: Any) -> Any:
    """Convert non-serializable objects to JSON-safe types.

    - Decimal → float (preserves numeric type in JSON)
    - numpy integers/floats → Python int/float
    - numpy bool → Python bool
    - datetime/date/time → ISO format string
    - bytes → hex string
    - Everything else → str
    """
    if isinstance(obj, Decimal):
        return float(obj)

    # numpy types (check without importing numpy at module level)
    type_name = type(obj).__module__
    if type_name == "numpy":
        cls_name = type(obj).__name__
        if "bool" in cls_name:
            return bool(obj)
        if "int" in cls_name:
            return int(obj)
        if "float" in cls_name:
            return float(obj)
        if cls_name == "ndarray":
            return obj.tolist()

    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.hex()

    return str(obj)


def safe_dumps(obj: Any, **kwargs: Any) -> str:
    """json.dumps with safe default handler for database types."""
    kwargs.setdefault("default", _safe_default)
    return json.dumps(obj, **kwargs)
