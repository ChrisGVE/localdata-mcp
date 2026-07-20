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


def _safe_key(key: Any) -> Any:
    """Convert a mapping key into one the json module accepts.

    ``json.dumps`` only accepts str/int/float/bool/None keys, and the ``default``
    handler is never consulted for keys — so a dict keyed by numpy scalars (for
    example cluster labels used as keys) raises ``TypeError`` no matter what
    ``default`` does. Numeric keys are stringified by json anyway, so mapping a
    numpy integer to a Python int preserves the existing output.
    """
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key

    converted = _safe_default(key)
    if isinstance(converted, (str, int, float, bool)) or converted is None:
        return converted
    return str(converted)


def _normalize_keys(obj: Any) -> Any:
    """Recursively rebuild mappings so every key is JSON-encodable."""
    if isinstance(obj, dict):
        return {_safe_key(k): _normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_keys(item) for item in obj]
    return obj


def safe_dumps(obj: Any, **kwargs: Any) -> str:
    """json.dumps with safe handling for database and analysis result types."""
    kwargs.setdefault("default", _safe_default)
    return json.dumps(_normalize_keys(obj), **kwargs)
