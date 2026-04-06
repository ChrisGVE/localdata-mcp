"""Value type enumeration for tree storage properties."""

from enum import Enum


class ValueType(Enum):
    """Supported property value types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    NULL = "null"
    DATETIME = "datetime"
