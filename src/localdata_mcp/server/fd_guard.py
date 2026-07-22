"""localdata_mcp/server/fd_guard.py — the NFR-303 fd-1 guard.

PRD S5.3's wiring, verified against the installed mcp==1.27.0: fd 1
belongs exclusively to JSON-RPC frames, so before the transport starts
the real stdout is saved onto a dup'ed TEXT-mode descriptor (the SDK
types its injectable `stdout` as `anyio.AsyncFile[str]` and writes str
frames), fd 1 is re-pointed at stderr so stray C-extension writes land
there, and `sys.stdout` is rebound to `sys.stderr` for stray Python
writes. Neighbors: mcp_app.py installs the guard at boot and hands
`protocol_out` to `mcp.server.stdio.stdio_server(stdout=...)`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TextIO

import anyio
from anyio import AsyncFile


@dataclass(frozen=True)
class StdoutGuard:
    """The one legitimate route to the real stdout after guarding."""

    # TEXT-mode async file over the saved real-stdout descriptor — the
    # exact shape stdio_server's frame writer expects (S5.3).
    protocol_out: AsyncFile[str]


def install_stdout_guard() -> StdoutGuard:
    """Guard fd 1 before the transport starts (S5.3, order-dependent).

    (1) dup the real stdout to a saved descriptor and wrap it in a
    text-mode file; (3) dup2 stderr onto fd 1 and rebind `sys.stdout`
    to `sys.stderr`. Step (2) — starting the transport over the saved
    descriptor — is the caller's move: the dup MUST precede the dup2,
    which is what this function guarantees.
    """
    saved_fd = os.dup(1)
    protocol_file: TextIO = os.fdopen(saved_fd, "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return StdoutGuard(protocol_out=anyio.wrap_file(protocol_file))
