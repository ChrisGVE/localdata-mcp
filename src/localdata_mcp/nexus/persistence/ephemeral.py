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
ever attempted here; resource limits are the global NX-2 defaults,
enforced at the same NX-6/NX-5 point that enforces record limits (§5).
Neighbors: engines.py supplies the posture-applying engine builders;
manager.py exposes the open to NX-6.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from localdata_mcp.nexus.config.endpoints import Posture
from localdata_mcp.nexus.persistence.engines import DuckDbHandle, _sqlite_handle

EphemeralEngineKind = Literal["sqlite", "duckdb"]


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
