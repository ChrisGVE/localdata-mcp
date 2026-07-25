"""Path handling at the trust boundary.

Every path in this server arrives from an LLM, which means it arrives from
whatever the LLM read most recently. Treat all of them as untrusted input.

Two rules do most of the work:

* **Resolve first, then check.** ``Path.resolve()`` collapses ``..`` segments
  *and* follows symlinks, so a link pointing outside the allowed root resolves
  to its real location and fails the containment test. Checking before
  resolving would pass a link whose name looks innocent.
* **Never silently overwrite.** SQLite's own ``VACUUM INTO`` refuses an existing
  target rather than replacing it, and that is the right default to copy: an
  export that quietly replaces a file the user still needed is unrecoverable,
  while one that refuses costs a single retry with an explicit flag.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PathNotAllowed", "allowed_root", "resolve_read_path", "resolve_write_path"]

#: Environment variable naming the directory this server may read and write.
ROOT_ENV_VAR = "LOCALDATA_ROOT"


class PathNotAllowed(ValueError):
    """A path that this server will not read from or write to."""


def allowed_root() -> Path:
    """The single directory tree this server may touch.

    Defaults to the process working directory, which for an MCP server is
    whatever the client launched it in. Override with ``LOCALDATA_ROOT`` to point
    at a data directory instead.
    """
    return Path(os.environ.get(ROOT_ENV_VAR, os.getcwd())).resolve()


def _contained(path: Path, root: Path) -> Path:
    try:
        path.relative_to(root)
    except ValueError:
        raise PathNotAllowed(
            f"{path} is outside the allowed root {root}. "
            f"Set {ROOT_ENV_VAR} to widen the allowed area."
        ) from None
    return path


def resolve_read_path(raw: str) -> Path:
    """Resolve a path to read, or explain why it is refused."""
    root = allowed_root()
    try:
        path = Path(raw).expanduser().resolve()
    except (OSError, ValueError) as exc:
        raise PathNotAllowed(f"Invalid path {raw!r}: {exc}") from exc

    _contained(path, root)

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
    root = allowed_root()
    try:
        path = Path(raw).expanduser().resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise PathNotAllowed(f"Invalid path {raw!r}: {exc}") from exc

    _contained(path, root)

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
