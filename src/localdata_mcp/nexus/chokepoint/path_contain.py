"""localdata_mcp/nexus/chokepoint/path_contain.py — NFR-108 containment.

The one canonical-real-path check every filesystem-touching operation
crosses, read side AND write side (E6.4): the candidate is resolved to
its real path (symlinks followed — a link pointing out of the tree is
out of the tree) and must sit at or under one of the operator's
`allowed_paths` entries, themselves canonicalized at comparison time.
Empty or unset `allowed_paths` is FAIL-CLOSED: no filesystem I/O at
all (S8 row 19) — the refusal, not the pass, is the default. A
write-side target that does not exist yet is contained by its resolved
parent chain (`Path.resolve(strict=False)` follows every existing
symlink prefix), so a to-be-created file cannot escape through a
linked directory. Neighbors: guard.py applies this to every extracted
SQL path literal and to NX-8's writes; ephemeral opens cross it before
nexus/persistence ever sees the path.

RESIDUAL TOCTOU (CR-024): this seam returns a resolved path STRING, and
its callers re-open by that string later (the file readers'
`pd.read_*`, nexus/persistence's ephemeral open). Between this
resolution and that open a path component can be swapped for a symlink
pointing out of the tree — the classic time-of-check/time-of-use
window. Closing it needs an ATOMIC contain-and-open at the open site
(open the final component `O_NOFOLLOW`, then re-validate the opened
descriptor's real path, or pass an fd through instead of a string);
that lives in the reader/persistence call sites (outside NX-6), so this
containment service cannot close it alone. Bounded in the single-user
deployment, but not eliminated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

AccessMode = Literal["read", "write"]


class PathRefusedError(PermissionError):
    """The candidate path failed containment — structured refusal
    naming the mode; the caller shapes it through NX-3."""

    def __init__(self, message: str, *, mode: AccessMode) -> None:
        super().__init__(message)
        self.mode: AccessMode = mode


def contain(
    candidate: str | Path,
    allowed_paths: Iterable[str],
    *,
    mode: AccessMode,
) -> Path:
    """The canonical real path of `candidate`, iff contained.

    Raises PathRefusedError otherwise — including always when the
    allow-list is empty (fail-closed, S8 row 19).
    """
    roots = [Path(entry).resolve() for entry in allowed_paths]
    if not roots:
        raise PathRefusedError(
            f"filesystem {mode} refused: allowed_paths is empty — "
            "no filesystem I/O is permitted at all (fail-closed default)",
            mode=mode,
        )
    real = Path(candidate).resolve()
    for root in roots:
        if real == root or real.is_relative_to(root):
            return real
    raise PathRefusedError(
        f"filesystem {mode} refused: {str(real)!r} is outside every "
        "allowed_paths entry",
        mode=mode,
    )
