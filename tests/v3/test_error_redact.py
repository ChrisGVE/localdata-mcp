"""tests/v3/test_error_redact.py — E4.3 fault-injection on the wire edge.

Exceptions carrying credentialed DSNs in their text must produce wire
shapes with zero credential material — asserted by grepping the
serialized wire form for the injected secrets.
"""

from __future__ import annotations

import json

from localdata_mcp.nexus.error.model import ErrorType, StructuredError
from localdata_mcp.nexus.error.redact import redact_structured
from localdata_mcp.nexus.error.translate import translate

_SECRET = "s3cr3t-hunter2"


class TestRedaction:
    def test_url_password_is_scrubbed_from_both_fields(self) -> None:
        error = StructuredError(
            error_type=ErrorType.CONNECTION_ERROR,
            message=f"cannot reach postgresql://app:{_SECRET}@db.internal/w",
            suggestion=f"try postgresql://app:{_SECRET}@db.internal/w later",
            retryable=True,
        )
        wire = json.dumps(redact_structured(error).to_wire())
        assert _SECRET not in wire
        assert "db.internal" in wire  # diagnosability survives

    def test_keyword_credential_is_scrubbed(self) -> None:
        error = StructuredError(
            error_type=ErrorType.AUTH_ERROR,
            message=f"login failed for DSN 'Server=db;Password={_SECRET};'",
            suggestion="check the declared credentials_ref",
            retryable=False,
        )
        assert _SECRET not in json.dumps(redact_structured(error).to_wire())

    def test_translation_output_survives_injection_end_to_end(self) -> None:
        exc = Exception(f"connection refused: mysql://root:{_SECRET}@10.0.0.5/prod")
        wire = json.dumps(redact_structured(translate(exc, "mysql")).to_wire())
        assert _SECRET not in wire

    def test_clean_text_passes_unchanged(self) -> None:
        error = StructuredError(
            error_type=ErrorType.SYNTAX_ERROR,
            message="near SELECTT",
            suggestion="Check the statement syntax.",
            retryable=False,
        )
        assert redact_structured(error) == error
