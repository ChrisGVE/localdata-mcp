"""tests/v3/test_error_model.py — E4.1 the one model/taxonomy/wire shape."""

from __future__ import annotations

import json

from localdata_mcp.nexus.error.model import ErrorType, StructuredError


class TestWireShape:
    def test_wire_form_carries_exactly_four_fields(self) -> None:
        wire = StructuredError(
            error_type=ErrorType.SYNTAX_ERROR,
            message="near SELECTT",
            suggestion="Check the statement syntax.",
            retryable=False,
        ).to_wire()
        assert set(wire) == {"error_type", "message", "suggestion", "retryable"}
        assert wire["error_type"] == "syntax_error"

    def test_wire_form_is_json_serializable(self) -> None:
        wire = StructuredError(
            error_type=ErrorType.TRANSIENT_ERROR,
            message="deadlock detected",
            suggestion="Re-issue the call.",
            retryable=True,
        ).to_wire()
        assert json.loads(json.dumps(wire)) == wire


class TestTaxonomy:
    def test_taxonomy_is_closed_and_value_stable(self) -> None:
        assert {e.value for e in ErrorType} == {
            "connection_error",
            "auth_error",
            "permission",
            "schema_error",
            "syntax_error",
            "constraint_error",
            "resource_error",
            "transient_error",
            "timeout",
            "security_violation",
            "data_validation",
            "configuration",
            "query_execution",
        }

    def test_connection_fault_classes_are_exactly_two(self) -> None:
        faulting = {e for e in ErrorType if e.signals_connection_fault}
        assert faulting == {ErrorType.CONNECTION_ERROR, ErrorType.TIMEOUT}
