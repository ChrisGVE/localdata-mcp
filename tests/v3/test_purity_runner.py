"""tests/v3/test_purity_runner.py — E2.5 subprocess stdout-purity battery.

The reusable NFR-303 mechanism, exercised for real: the runner
launches the v3 server as a child process, captures TRUE fd 1, drives
a genuine initialize handshake plus a tool call over stdio, and
asserts every byte on stdout parses as JSON-RPC frames. This is the
epic's exit-gate proof — a positive L3 round-trip through the guarded
descriptor (PRD S5.3), not only the no-stray-bytes assertion — plus
unit cases proving the frame parser detects each impurity shape.
Neighbors: testbench/purity_runner.py under test; server/mcp_app.py is
the child it launches.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from localdata_mcp.testbench.purity_runner import (
    PurityViolation,
    SessionResult,
    initialize_request,
    initialized_notification,
    parse_frames,
    run_session,
    tool_call_request,
)


@pytest.fixture(scope="module")
def l3_session(tmp_path_factory: pytest.TempPathFactory) -> SessionResult:
    """One real child-process session shared by the round-trip
    assertions below: initialize, initialized, call the skeleton ping
    tool, close stdin. Hermetic: HOME points into a temp dir and no
    LOCALDATA_* env override leaks in, so only model defaults apply."""
    home = tmp_path_factory.mktemp("child-home")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LOCALDATA_")
    }
    env["HOME"] = str(home)
    return run_session(
        [
            initialize_request(request_id=1),
            initialized_notification(),
            tool_call_request(request_id=2, name="ping"),
        ],
        cwd=home,
        env=env,
    )


class TestL3RoundTrip:
    def test_child_exits_cleanly(self, l3_session: SessionResult) -> None:
        assert l3_session.returncode == 0

    def test_every_stdout_byte_is_a_frame(self, l3_session: SessionResult) -> None:
        # parse_frames already ran inside run_session; a violation
        # would have raised. Assert the positive: frames arrived.
        assert l3_session.frames
        assert all(frame["jsonrpc"] == "2.0" for frame in l3_session.frames)

    def test_initialize_response_traverses_guarded_fd(
        self, l3_session: SessionResult
    ) -> None:
        response = l3_session.responses_by_id()[1]
        assert response["result"]["serverInfo"]["name"] == "localdata"

    def test_tool_call_response_traverses_guarded_fd(
        self, l3_session: SessionResult
    ) -> None:
        response = l3_session.responses_by_id()[2]
        content = response["result"]["content"]
        # E7.2: the wrapper ships the FR-403 envelope; `inline` carries
        # the scalar the tool produced.
        envelope = json.loads(content[0]["text"])
        assert envelope["inline"] == "pong"
        assert envelope["error"] is None

    def test_logs_went_to_stderr_not_stdout(self, l3_session: SessionResult) -> None:
        assert b"starting stdio transport" in l3_session.stderr


class TestImpurityDetection:
    def test_runner_catches_an_impure_server(self, tmp_path: Path) -> None:
        # A fake server that pollutes stdout must fail the assertion —
        # proving the mechanism has teeth, not just green paths.
        impure = (
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('progress: 42%\\n')",
        )
        with pytest.raises(PurityViolation):
            run_session([initialize_request()], command=impure, cwd=tmp_path)


class TestParseFrames:
    def test_empty_stream_is_pure(self) -> None:
        assert parse_frames(b"") == ()

    def test_clean_frames_parse(self) -> None:
        raw = b'{"jsonrpc": "2.0", "id": 1, "result": {}}\n'
        (frame,) = parse_frames(raw)
        assert frame["id"] == 1

    def test_non_json_line_raises(self) -> None:
        with pytest.raises(PurityViolation):
            parse_frames(b"Loading model weights...\n")

    def test_json_without_jsonrpc_key_raises(self) -> None:
        with pytest.raises(PurityViolation):
            parse_frames(b'{"progress": 0.4}\n')

    def test_non_object_json_raises(self) -> None:
        with pytest.raises(PurityViolation):
            parse_frames(b"42\n")

    def test_unterminated_trailing_bytes_raise(self) -> None:
        with pytest.raises(PurityViolation):
            parse_frames(b'{"jsonrpc": "2.0", "id": 1}\n{"jsonrpc": "2.0"')
