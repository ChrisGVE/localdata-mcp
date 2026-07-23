"""tests/v3/test_connector_values.py — E8.3: the shared store substrate.

The typed-value round-trip (values.py) and the escaped dot-path
grammar (treepaths.py) both store families speak — harvested from
`main`'s tree_storage with semantics intact (DR GP2), asserted here so
the connectors-level copy cannot silently diverge from what the
fixtures were written with.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from localdata_mcp.ingest.connectors.treepaths import (
    build_path,
    escape_path_segment,
    parse_path,
    unescape_path_segment,
)
from localdata_mcp.ingest.connectors.values import (
    ValueType,
    deserialize_value,
    infer_value_type,
    infer_value_type_from_string,
    serialize_value,
)


class TestValueRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [None, True, False, 0, -17, 3.5, "plain", ["a", 1], datetime(2026, 7, 23)],
    )
    def test_serialize_then_deserialize_is_identity(self, value: object) -> None:
        value_type = infer_value_type(value)
        stored, original = serialize_value(value, value_type)
        assert deserialize_value(stored, value_type, original) == value

    def test_bool_is_not_mistaken_for_int(self) -> None:
        assert infer_value_type(True) is ValueType.BOOLEAN

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError):
            infer_value_type({"a": 1})


class TestStringInference:
    @pytest.mark.parametrize(
        ("text", "expected_type", "expected_value"),
        [
            ("true", ValueType.BOOLEAN, True),
            ("42", ValueType.INTEGER, 42),
            ("2.5", ValueType.FLOAT, 2.5),
            ("[1, 2]", ValueType.ARRAY, [1, 2]),
            ("null", ValueType.NULL, None),
            ("hello", ValueType.STRING, "hello"),
        ],
    )
    def test_inference_matches_main_semantics(
        self, text: str, expected_type: ValueType, expected_value: object
    ) -> None:
        inferred, converted = infer_value_type_from_string(text)
        assert inferred is expected_type
        assert converted == expected_value


class TestDotPaths:
    def test_build_then_parse_is_identity(self) -> None:
        segments = ["plain", "with.dot", "with\\slash"]
        assert parse_path(build_path(segments)) == segments

    def test_segment_escape_round_trip(self) -> None:
        assert unescape_path_segment(escape_path_segment("a.b\\c")) == "a.b\\c"

    def test_invalid_escape_refused(self) -> None:
        with pytest.raises(ValueError):
            parse_path("bad\\x")

    def test_empty_path_is_no_segments(self) -> None:
        assert parse_path("") == []
        assert build_path([]) == ""
