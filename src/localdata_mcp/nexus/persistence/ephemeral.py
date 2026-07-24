"""localdata_mcp/nexus/persistence/ephemeral.py — ad-hoc file opens (E5.3).

Ephemeral local-file connections are a distinct lightweight type, NOT a
`ConnectionRecord` variant (§5): opened per-call over a SQLite/DuckDB
file, never pooled, identity is the canonicalized path — no `name`, no
`credentials_ref`, no health probe, and NFR-112 is vacuous because the
connection never outlives the call that opened it. Posture is read-only,
period, unless an operator-trust-layer NX-2 entry
(`security.ephemeral_write_paths`, introduction-gated) grants
read-write for the canonical path or a containing prefix. Path
CONTAINMENT (`allowed_paths`, NFR-108) is NX-6's check, not this
module's — the chokepoint refuses an out-of-tree path before an open is
ever attempted here. The ATOMIC contain-and-open guard IS this module's
step (`_reject_symlinked_target`, CR-024): the canonical path is
re-opened O_NOFOLLOW and its identity re-validated before the engine
re-opens it by string, closing the resolve-then-reopen symlink-swap
window. Resource limits are the global NX-2 defaults,
enforced at the same NX-6/NX-5 point that enforces record limits (§5).
Neighbors: engines.py supplies the posture-applying engine builders;
manager.py exposes the open to NX-6.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from localdata_mcp.nexus.config.endpoints import Posture
from localdata_mcp.nexus.persistence.engines import DuckDbHandle, _sqlite_handle

EphemeralEngineKind = Literal["sqlite", "duckdb"]


class EphemeralOpenRefusedError(PermissionError):
    """CR-024: the canonical path changed identity between NX-6's
    containment verdict and the ephemeral open — a resolve-then-reopen
    TOCTOU, refused fail-safe rather than opened."""


def rw_granted(canonical_path: Path, write_grants: Iterable[str]) -> bool:
    """Whether an operator grant keys this canonical path read-write —
    exact match or containment under a granted prefix, both sides
    canonicalized so a symlinked or relative grant cannot diverge."""
    for grant in write_grants:
        granted = Path(grant).resolve()
        if canonical_path == granted or canonical_path.is_relative_to(granted):
            return True
    return False


@dataclass(frozen=True)
class EphemeralFileConnection:
    """One ad-hoc file source: canonical identity + computed posture.

    `open()` builds a fresh engine, yields the live connection for the
    duration of the `with` block, and disposes it on exit — the §5
    never-outlives-the-call guarantee is structural, not a convention.
    """

    canonical_path: Path
    engine_kind: EphemeralEngineKind
    posture: Posture

    @contextmanager
    def open(self) -> Iterator[Any]:
        _reject_symlinked_target(self.canonical_path)
        read_only = self.posture == "read_only"
        if self.engine_kind == "duckdb":
            handle: Any = DuckDbHandle(
                path=str(self.canonical_path), read_only=read_only
            )
        else:
            handle = _sqlite_handle(f"sqlite:///{self.canonical_path}", self.posture)
        try:
            with handle.connect() as connection:
                yield connection
        finally:
            handle.dispose()


def ephemeral_for(
    path: str | Path,
    engine_kind: EphemeralEngineKind,
    write_grants: Iterable[str],
) -> EphemeralFileConnection:
    """The one constructor: canonicalize the identity, compute the
    posture from the operator grants — read-only unless granted."""
    canonical = Path(path).resolve()
    posture: Posture = (
        "read_write" if rw_granted(canonical, write_grants) else "read_only"
    )
    return EphemeralFileConnection(
        canonical_path=canonical, engine_kind=engine_kind, posture=posture
    )


def _reject_symlinked_target(canonical_path: Path) -> None:
    """Atomic contain-and-open guard (CR-024): re-open the canonical path
    with O_NOFOLLOW so a symlink swapped into the final component after
    NX-6's containment cannot redirect the engine outside the tree, and
    confirm the descriptor's identity still matches the path. SQLite and
    DuckDB need a real path (journal/WAL/lock files derive from it), so —
    unlike the read-only file parsers — a descriptor cannot be threaded
    through; the engine still re-opens by string, so the residual is the
    narrow window between this check and that open, bounded in the
    single-user deployment. A path that does not exist yet (a to-be-created
    rw file) has nothing to swap against and is left to the engine."""
    try:
        fd = os.open(canonical_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return
    try:
        opened = os.fstat(fd)
        current = os.stat(canonical_path)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise EphemeralOpenRefusedError(
                f"ephemeral open refused: {str(canonical_path)!r} changed "
                "identity between containment and open (CR-024 TOCTOU)"
            )
    finally:
        os.close(fd)
