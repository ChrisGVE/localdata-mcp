"""tests/v3/test_error_wire.py — E4.4 the wrap seam + fault signaling.

The §4b ordered sequence end to end against E4.0's conformant double:
connection-class classification fires mark_faulted synchronously;
non-connection classes never touch the lifecycle; the returned shape
is redacted regardless.
"""

from __future__ import annotations

import json

from localdata_mcp.nexus.error.fault_signal import RecordingFaultSink
from localdata_mcp.nexus.error.model import ErrorType
from localdata_mcp.nexus.error.wire import wrap

_SECRET = "hunter2-pw"


class TestFaultSignaling:
    def test_connection_class_fault_fires_synchronously(self) -> None:
        sink = RecordingFaultSink()
        result = wrap(
            TimeoutError("connection timed out"),
            "generic",
            fault_sink=sink,
            record_id="ep1-rec3",
        )
        assert result.error_type.signals_connection_fault
        assert ("ep1-rec3", "faulted") in sink.transitions
        assert ("ep1-rec3", "resetting") in sink.transitions

    def test_non_connection_class_never_touches_the_lifecycle(self) -> None:
        sink = RecordingFaultSink()
        result = wrap(
            Exception("syntax error near SELECTT"),
            "sqlite",
            fault_sink=sink,
            record_id="ep1-rec3",
        )
        assert not result.error_type.signals_connection_fault
        assert sink.transitions == []

    def test_no_record_means_no_signal_even_for_connection_class(self) -> None:
        sink = RecordingFaultSink()
        wrap(TimeoutError("connection timed out"), "generic", fault_sink=sink)
        assert sink.transitions == []


class TestWireOutput:
    def test_wrap_output_is_redacted(self) -> None:
        exc = Exception(f"refused: postgresql://app:{_SECRET}@db/prod")
        wire = json.dumps(wrap(exc, "postgresql").to_wire())
        assert _SECRET not in wire

    def test_wrap_returns_the_one_shape(self) -> None:
        result = wrap(Exception("boom"), "generic")
        assert set(result.to_wire()) == {
            "error_type",
            "message",
            "suggestion",
            "retryable",
        }
        assert isinstance(result.error_type, ErrorType)
