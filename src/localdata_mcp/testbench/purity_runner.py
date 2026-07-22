"""localdata_mcp/testbench/purity_runner.py — the NFR-303 primitive.

The reusable stdout-purity mechanism every battery wraps (PRD S6 E2.5):
launch the server as a CHILD PROCESS, drive a real JSON-RPC session
over its stdio, capture TRUE fd 1, and refuse any byte that is not a
newline-delimited JSON-RPC frame. OS-level by construction — the pipe
handed to the child IS its fd 1, so a stray print() or C-extension
write is caught here regardless of what the in-process guard missed.
Neighbors: server/mcp_app.py is the default child; schema.py/store.py
hold the results store the batteries write to; tests/v3 and every
later battery import this module rather than re-implementing capture.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mcp.types import LATEST_PROTOCOL_VERSION

# The v3 server entrypoint (ARCHITECTURE.md section 4e) as a module
# invocation of the running interpreter — the .venv python always.
DEFAULT_SERVER_COMMAND: tuple[str, ...] = (
    sys.executable,
    "-m",
    "localdata_mcp.server.mcp_app",
)

# Batteries run long sessions; a generous ceiling that still fails a
# hung child. Callers override per session, never globally.
DEFAULT_TIMEOUT_SECONDS = 120.0


class PurityViolation(AssertionError):
    """Bytes reached fd 1 that are not JSON-RPC frames (NFR-303)."""


@dataclass(frozen=True)
class SessionResult:
    """Everything one child session produced, frames pre-verified."""

    raw_stdout: bytes
    stderr: bytes
    frames: tuple[Mapping[str, Any], ...]
    returncode: int

    def responses_by_id(self) -> dict[Any, Mapping[str, Any]]:
        """The id-carrying frames (responses), keyed by request id."""
        return {frame["id"]: frame for frame in self.frames if "id" in frame}


def initialize_request(*, request_id: int = 1) -> dict[str, Any]:
    """A client `initialize` request opening the session."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "purity-runner", "version": "0"},
        },
    }


def initialized_notification() -> dict[str, Any]:
    """The client's `initialized` notification completing the handshake."""
    return {"jsonrpc": "2.0", "method": "notifications/initialized"}


def tool_call_request(
    *, request_id: int, name: str, arguments: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """A `tools/call` request for one named tool."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": dict(arguments or {})},
    }


def run_session(
    messages: Sequence[Mapping[str, Any]],
    *,
    command: Sequence[str] = DEFAULT_SERVER_COMMAND,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    cwd: "str | None" = None,
    env: Mapping[str, str] | None = None,
) -> SessionResult:
    """Run one full stdio session against a child server.

    Writes `messages` as newline-delimited JSON to the child's stdin,
    HOLDS stdin open until every id-carrying request has been answered
    (closing immediately races the EOF-driven shutdown against in-
    flight calls), then closes it, collects both output streams, and
    verifies stdout purity before returning.
    """
    payload = "".join(json.dumps(message) + "\n" for message in messages)
    expected_ids = {message["id"] for message in messages if "id" in message}
    child = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=None if env is None else dict(env),
    )
    stdout_collector = _StreamCollector(child.stdout, expected_ids)
    stderr_collector = _StreamCollector(child.stderr, None)
    _send_requests(child, payload)
    stdout_collector.settled.wait(timeout_seconds)
    _close_stdin(child)
    returncode = _wait_or_kill(child, timeout_seconds)
    raw_stdout = stdout_collector.join_data()
    stderr = stderr_collector.join_data()
    return SessionResult(raw_stdout, stderr, parse_frames(raw_stdout), returncode)


class _StreamCollector:
    """Drains one child pipe on a thread; `settled` fires once every
    expected response id has appeared (or on EOF), so the caller knows
    when closing stdin can no longer cut off an in-flight response."""

    def __init__(self, pipe: Any, expected_ids: "set[Any] | None") -> None:
        self._pipe = pipe
        self._pending = set(expected_ids or ())
        self._chunks: list[bytes] = []
        self.settled = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        for line in iter(self._pipe.readline, b""):
            self._chunks.append(line)
            self._note_response(line)
        self.settled.set()  # EOF: nothing more will arrive

    def _note_response(self, line: bytes) -> None:
        try:
            frame = json.loads(line)
        except ValueError:
            return  # impurity — parse_frames reports it on the whole stream
        if isinstance(frame, dict):
            self._pending.discard(frame.get("id"))
        if not self._pending:
            self.settled.set()

    def join_data(self) -> bytes:
        self._thread.join()
        return b"".join(self._chunks)


def _send_requests(child: subprocess.Popen[bytes], payload: str) -> None:
    """Write the session's requests, tolerating a child that exited
    before reading them (its output still gets purity-checked)."""
    assert child.stdin is not None
    try:
        child.stdin.write(payload.encode("utf-8"))
        child.stdin.flush()
    except BrokenPipeError:
        pass


def _close_stdin(child: subprocess.Popen[bytes]) -> None:
    """EOF the child's stdin — the stdio server's shutdown signal."""
    assert child.stdin is not None
    try:
        child.stdin.close()
    except BrokenPipeError:
        pass


def _wait_or_kill(child: subprocess.Popen[bytes], timeout_seconds: float) -> int:
    """The child's exit code, killing it if it outlives the timeout."""
    try:
        return child.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait()


def parse_frames(raw_stdout: bytes) -> tuple[Mapping[str, Any], ...]:
    """Every byte of `raw_stdout` as JSON-RPC frames, or PurityViolation.

    The stream must be newline-delimited JSON objects each carrying
    `"jsonrpc"` — anything else (a partial trailing line included) is
    exactly the corruption NFR-303 exists to catch.
    """
    if raw_stdout == b"":
        return ()
    if not raw_stdout.endswith(b"\n"):
        tail = raw_stdout.rsplit(b"\n", 1)[-1]
        raise PurityViolation(f"unterminated bytes on fd 1: {tail[:200]!r}")
    return tuple(_parse_frame_line(line) for line in raw_stdout.split(b"\n")[:-1])


def _parse_frame_line(line: bytes) -> Mapping[str, Any]:
    """One line as a JSON-RPC frame, or PurityViolation."""
    try:
        frame = json.loads(line)
    except (ValueError, UnicodeDecodeError) as error:
        raise PurityViolation(f"non-JSON bytes on fd 1: {line[:200]!r}") from error
    if not isinstance(frame, dict) or "jsonrpc" not in frame:
        raise PurityViolation(f"JSON on fd 1 that is not a frame: {line[:200]!r}")
    return frame
