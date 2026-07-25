"""Path handling at the trust boundary.

Every path in this server arrives from an LLM, which means it arrives from
whatever the LLM read most recently. Treat all of them as untrusted input.

Three rules do the work:

* **Resolve first, then check.** ``Path.resolve()`` collapses ``..`` segments
  *and* follows symlinks, so a link pointing outside the allowed area resolves
  to its real location and fails the containment test. Checking before
  resolving would pass a link whose name looks innocent.
* **The working directory is always in scope**, together with everything below
  it. An MCP client launches this server somewhere deliberate, so the common
  case needs no configuration at all; ``paths.roots`` adds to that rather than
  replacing it.
* **Never silently overwrite.** SQLite's own ``VACUUM INTO`` refuses an existing
  target rather than replacing it, and that is the right default to copy: an
  export that quietly replaces a file the user still needed is unrecoverable,
  while one that refuses costs a single retry with an explicit flag.

``paths.path_limited = false`` switches containment off, and nothing else. It
does not relax the overwrite refusal, which guards against a different accident
and stays in force regardless.
"""

from __future__ import annotations

from pathlib import Path

from . import config

__all__ = [
    "PathNotAllowed",
    "allowed_paths",
    "resolve_read_path",
    "resolve_write_path",
]


class PathNotAllowed(ValueError):
    """A path that this server will not read from or write to."""


def allowed_paths() -> tuple[Path, ...]:
    """Everywhere the server may reach: the working directory, then the roots.

    Recomputed per call rather than cached, because the configuration can be
    replaced at startup and the working directory is read at the moment of use.
    """
    return (Path.cwd().resolve(), *config.active().roots)


def _contained(path: Path) -> Path:
    if not config.active().path_limited:
        return path

    permitted = allowed_paths()
    for root in permitted:
        if path == root or path.is_relative_to(root):
            return path

    listed = ", ".join(str(root) for root in permitted)
    raise PathNotAllowed(
        f"{path} is outside the allowed paths ({listed}). Add its directory to "
        f"paths.roots in the configuration file, or set paths.path_limited = "
        f"false to remove the restriction."
    )


def resolve_read_path(raw: str) -> Path:
    """Resolve a path to read, or explain why it is refused."""
    try:
        path = Path(raw).expanduser().resolve()
    except (OSError, ValueError) as exc:
        raise PathNotAllowed(f"Invalid path {raw!r}: {exc}") from exc

    _contained(path)

    if not path.exists():
        raise PathNotAllowed(f"No such file: {path}")
    if not path.is_file():
        raise PathNotAllowed(f"Not a regular file: {path}")
    return path


def resolve_write_path(raw: str, *, overwrite: bool = False) -> Path:
    """Resolve a path to write, refusing an existing target unless asked.

    ``strict=False`` on the resolve is deliberate: the target usually does not
    exist yet, which is not an error. Containment is still checked against the
    fully-resolved path, so a ``..`` escape or a symlinked parent is caught.
    """
    try:
        path = Path(raw).expanduser().resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise PathNotAllowed(f"Invalid path {raw!r}: {exc}") from exc

    _contained(path)

    parent = path.parent
    if not parent.is_dir():
        raise PathNotAllowed(f"Directory does not exist: {parent}")

    if path.exists():
        if not overwrite:
            raise PathNotAllowed(
                f"{path} already exists. Pass overwrite=true to replace it."
            )
        if not path.is_file():
            # A directory, socket or device would not be replaced by a write;
            # refusing is clearer than whatever the OS would do instead.
            raise PathNotAllowed(f"Refusing to overwrite a non-file: {path}")

    return path
