"""Serialization and deserialization of typed property values."""

import json
from datetime import datetime
from typing import Any, Optional, Tuple

from localdata_mcp.tree_storage.types import ValueType


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
