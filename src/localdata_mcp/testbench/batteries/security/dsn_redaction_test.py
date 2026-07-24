"""testbench/batteries/security/dsn_redaction_test.py — NFR-110 at the L3 seam.

Connection credentials have one declared home (the Config nexus) and never
leave it: no log record, error response, or tool result may carry
password material. The structural redaction at the Error and Logging
nexuses (NX-3/NX-4, ARCHITECTURE.md section 8) proven through the wire the
agent uses.

A networked endpoint is declared with a bogus credential-bearing DSN
pointing at a closed loopback port, so the connection always refuses with
a real backend connection error (no live server needed). The battery then
greps every outbound surface the fault touches — the FR-403 error envelope
and the whole NFR-303 stdout/stderr log stream produced during boot and
the failing call — for the fixture secret and finds zero hits, while the
non-secret parts of the DSN (host) stay diagnosable and the backend
surface is demonstrably populated (so zero hits is redaction, not
silence).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import _seam

# A distinctive fixture password + username: userinfo travels as one unit,
# so neither half may appear on any outbound surface. The DSN targets a
# closed loopback port — the connection refuses unconditionally.
_SECRET = "pl4intext-DONOTLEAK-cred-abc123"
_USER = "leakuser"
_DSN = f"postgresql://{_USER}:{_SECRET}@127.0.0.1:1/nodb"


def _run_failing_query(tmp_path: Path, capfd) -> tuple[dict, str]:
    """Boot the networked endpoint on the bogus credential-bearing DSN,
    force the connection error through the `query` wire, and return the
    FR-403 envelope together with every byte written to stdout/stderr
    during boot and call (the NFR-303 log surface)."""
    declarations = _seam.declare(pg=(_DSN, "read_only"))
    with _seam.booted(allowed_paths=(str(tmp_path),), declarations=declarations):
        envelope = _seam.call_envelope("query", {"endpoint": "pg", "sql": "SELECT 1"})
    captured = capfd.readouterr()
    return envelope, captured.out + captured.err


def test_connection_error_response_carries_no_secret(tmp_path: Path, capfd) -> None:
    envelope, _logs = _run_failing_query(tmp_path, capfd)
    error = _seam.expect_refused(envelope)
    blob = json.dumps(envelope)
    # Neither half of the userinfo reaches the error the LLM sees.
    assert _SECRET not in blob, "fixture password leaked into error response"
    assert _USER not in blob, "fixture username leaked into error response"
    # Redaction keeps the error diagnosable: the non-secret host survives.
    assert "127.0.0.1" in error["message"], error["message"]


def test_log_surface_carries_no_secret(tmp_path: Path, capfd) -> None:
    envelope, logs = _run_failing_query(tmp_path, capfd)
    _seam.expect_refused(envelope)
    assert _SECRET not in logs, "fixture password leaked into the log stream"
    assert f"{_USER}:{_SECRET}" not in logs, (
        "raw DSN userinfo leaked into the log stream"
    )
    # Non-vacuous: the connection WAS attempted and the backend logged its
    # failure, so a clean grep is structural redaction, not an empty surface.
    assert "postgresql" in logs
