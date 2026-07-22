"""tests/v3/test_fault_signal.py — E4.0 contract test for the fault seam.

Both NX-3 (caller) and NX-5 (implementer) build against this contract:
the double is a conformant implementation whose state walk is
observable, and any implementation of the protocol must satisfy the
same assertions (E6's gate re-runs them against the real
ConnectionRecord lifecycle).
"""

from __future__ import annotations

from localdata_mcp.nexus.error.fault_signal import (
    DisposeAndReissueEntry,
    FaultSignal,
    RecordingFaultSink,
)


class TestContract:
    def test_double_satisfies_the_protocol(self) -> None:
        assert isinstance(RecordingFaultSink(), FaultSignal)

    def test_mark_faulted_is_synchronous_and_reports_disposal(self) -> None:
        sink = RecordingFaultSink()
        entry = sink.mark_faulted("ep1-rec7")
        assert entry == DisposeAndReissueEntry(
            record_id="ep1-rec7", disposed=True, reissued=True
        )

    def test_state_walk_follows_the_lifecycle(self) -> None:
        sink = RecordingFaultSink()
        sink.mark_faulted("r1")
        assert sink.transitions == [
            ("r1", "faulted"),
            ("r1", "resetting"),
            ("r1", "active"),
        ]

    def test_failed_reissue_is_reported_not_raised(self) -> None:
        sink = RecordingFaultSink(reissue=False)
        entry = sink.mark_faulted("r2")
        assert entry.disposed and not entry.reissued
        assert ("r2", "active") not in sink.transitions
