"""tests/v3/test_fd_guard.py — E2.3 fd-1 guard mechanics (PRD S5.3).

Unit-level proof of the guard's three moves: the real stdout survives
on a dup'ed TEXT-mode descriptor for the transport, fd 1 itself is
re-pointed at stderr so stray C-extension writes cannot corrupt the
protocol stream, and `sys.stdout` is rebound to `sys.stderr` for stray
Python writes. Each test suspends pytest's own fd capture
(`capfd.disabled()` — the capture manager re-points fds 1/2 between
fixture and test body otherwise), redirects fds 1/2 onto temp files,
installs the guard, and restores everything afterwards. The end-to-end
subprocess proof lives in test_purity_runner.py (E2.5). Neighbors:
server/fd_guard.py under test; server/mcp_app.py wires it.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import anyio
import pytest

from localdata_mcp.server.fd_guard import StdoutGuard, install_stdout_guard


@contextmanager
def guarded_fds(tmp_path: Path) -> Iterator[tuple[StdoutGuard, Path, Path]]:
    """Point fd 1 / fd 2 at temp files, install the guard, and restore
    the process's real descriptors and sys.stdout afterwards."""
    stdout_target = tmp_path / "real_stdout"
    stderr_target = tmp_path / "real_stderr"
    with stdout_target.open("wb") as out_file, stderr_target.open("wb") as err_file:
        saved_1, saved_2, prev_stdout = os.dup(1), os.dup(2), sys.stdout
        os.dup2(out_file.fileno(), 1)
        os.dup2(err_file.fileno(), 2)
        try:
            guard = install_stdout_guard()
            yield guard, stdout_target, stderr_target
            anyio.run(guard.protocol_out.aclose)
        finally:
            os.dup2(saved_1, 1)
            os.dup2(saved_2, 2)
            os.close(saved_1)
            os.close(saved_2)
            sys.stdout = prev_stdout


async def _write_frame(guard: StdoutGuard, text: str) -> None:
    await guard.protocol_out.write(text)
    await guard.protocol_out.flush()


class TestStdoutGuard:
    def test_saved_descriptor_is_text_mode_utf8(
        self, capfd: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        with capfd.disabled(), guarded_fds(tmp_path) as (guard, _, _):
            # The SDK types stdout as anyio.AsyncFile[str] and writes
            # str frames: a binary-mode file would TypeError on frame 1.
            assert guard.protocol_out.wrapped.encoding == "utf-8"
            assert "b" not in guard.protocol_out.wrapped.mode

    def test_protocol_channel_reaches_original_stdout(
        self, capfd: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        with capfd.disabled(), guarded_fds(tmp_path) as (guard, stdout_target, _):
            anyio.run(_write_frame, guard, '{"jsonrpc": "2.0"}\n')
            assert stdout_target.read_bytes() == b'{"jsonrpc": "2.0"}\n'

    def test_fd1_writes_land_on_stderr_target(
        self, capfd: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        with (
            capfd.disabled(),
            guarded_fds(tmp_path) as (_, stdout_target, stderr_target),
        ):
            os.write(1, b"stray C-extension bytes\n")
            assert stderr_target.read_bytes() == b"stray C-extension bytes\n"
            assert stdout_target.read_bytes() == b""

    def test_sys_stdout_rebound_to_stderr(
        self, capfd: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        with capfd.disabled(), guarded_fds(tmp_path):
            assert sys.stdout is sys.stderr
