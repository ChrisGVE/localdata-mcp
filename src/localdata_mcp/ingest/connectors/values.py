"""localdata_mcp/ingest/connectors/values.py — typed property values (E8.3).

The harvested successor of `tree_storage/serialization.py` +
`tree_storage/types.py`: the one typed-value vocabulary the kv and
graph/tree store families share. Store properties persist as
`(value TEXT, value_type TEXT, original_repr TEXT)` rows
(store_schemas.py); this module owns the round-trip — type inference
from Python values and from raw tool-call strings, serialization to
the stored text, and deserialization back to the native value. Shared
at the connectors level because the vocabulary IS shared state: both
families read and write the same property row shape, and two copies
would drift (NFR-402's one-declaration discipline). Neighbors:
kv/tools.py and graph_tree/ speak it; store_schemas.py declares the
tables it fills.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Tuple


class ValueType(Enum):
    """Supported property value types (carried from `main`)."""

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
    """Infer type from a raw string input (for the set_value tool).

    Returns ``(inferred_type, converted_value)``.
    """
    if text_value.lower() in ("true", "false"):
        return (ValueType.BOOLEAN, text_value.lower() == "true")

    try:
        return (ValueType.INTEGER, int(text_value))
    except ValueError:
        pass

    try:
        return (ValueType.FLOAT, float(text_value))
    except ValueError:
        pass

    if text_value.startswith("["):
        try:
            parsed = json.loads(text_value)
            if isinstance(parsed, list):
                return (ValueType.ARRAY, parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    if text_value.lower() == "null":
        return (ValueType.NULL, None)

    return (ValueType.STRING, text_value)
