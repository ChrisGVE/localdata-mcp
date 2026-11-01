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
* **Never overwrite by default; replace only when told to.** A destination path
  is the *user's* choice, relayed through an agent, so the decision to destroy
  what is already there is the user's as well. ``force`` carries that decision
  in — it is the user's answer, not the agent's judgement, which is why the
  refusal names the file and says who has to be asked rather than reading as an
  instruction to retry.
* **A file some slot is living on is never a target, force or not.** Deleting
  it would pull the ground out from under a datasource that is currently
  attached, and on POSIX the unlink would *succeed* while that slot went on
  answering from an inode with no name — a divergence nothing would report.
  ``force`` is authority over the user's own spare files; it is not authority
  to break this server's open state.

``paths.path_limited = false`` switches containment off, and nothing else. It
relaxes neither of the two rules above, which guard against different accidents
and stay in force regardless.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from . import config
from .credentials import shown_url

__all__ = [
    "PathNotAllowed",
    "allowed_paths",
    "refuse_url_destination",
    "resolve_read_path",
    "resolve_write_path",
    "url_scheme",
]

#: A scheme must be at least two characters, so a Windows drive letter is never
#: mistaken for one, and the ``://`` must be present so a bare path never is.
_URL = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]+)://")


class PathNotAllowed(ValueError):
    """A path that this server will not read from or write to."""


def url_scheme(database: str) -> str | None:
    """The scheme of a datasource URL, or ``None`` for a filesystem path.

    The one test this server applies to tell the two apart, wherever a string
    could be either: ``attach`` opens a URL as an engine, ``create`` refuses one
    as its source (#88), and a destination refuses one too (#108). It lives here
    because the last of those is the path boundary's own question.
    """
    match = _URL.match(database)
    return match.group("scheme") if match else None


def refuse_url_destination(raw: str) -> None:
    """Refuse a URL as a place to write, by name, before it is taken as a path.

    ``query(path=…)`` and ``save`` write files. Taken as a path, a database URL
    is normalised into ``<cwd>/postgresql:/u:***@host`` and refused as a
    directory that does not exist — a message about a directory nobody named
    (issue #108). Recognised first, the refusal says what a destination is,
    which verb puts rows into a database, and shows the URL the way ``attach``
    would, password masked.
    """
    if url_scheme(raw) is None:
        return
    raise PathNotAllowed(
        f"A destination is a file, and {shown_url(raw)} is a URL: query(path=…) "
        f"and save write files. Writing rows into a database is create's job — "
        f"attach the database with writable=true, then create(nickname, "
        f"type='table', source=…) on it from a file."
    )


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


def resolve_write_path(
    raw: str,
    *,
    force: bool = False,
    claimed: Mapping[Path, str] | None = None,
) -> Path:
    """Resolve a path to write, clearing an existing target only if forced.

    ``strict=False`` on the resolve is deliberate: the target usually does not
    exist yet, which is not an error. Containment is still checked against the
    fully-resolved path, so a ``..`` escape or a symlinked parent is caught.

    ``claimed`` maps the files live slots are sitting on to the nickname of the
    slot holding each. Those are refused as targets whatever ``force`` says.
    Returns a path with nothing at it, so every caller may simply write.

    A URL is refused before anything else, so it is never read as a path.
    """
    refuse_url_destination(raw)
    try:
        path = Path(raw).expanduser().resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise PathNotAllowed(f"Invalid path {raw!r}: {exc}") from exc

    _contained(path)

    parent = path.parent
    if not parent.is_dir():
        raise PathNotAllowed(f"Directory does not exist: {parent}")

    holder = (claimed or {}).get(path)
    if holder is not None:
        # Checked before `force`, deliberately: force is the user's consent to
        # lose a file of their own, never consent to cut a live slot loose.
        raise PathNotAllowed(
            f"{path} is the file {holder!r} is attached to, so writing over it "
            f"would break that datasource while it carried on answering. "
            f"Detach {holder!r} first, or choose another name."
        )

    if path.exists():
        if not force:
            # The wording asks for a decision rather than announcing a retry:
            # the agent is relaying a name the user chose, so replacing what is
            # there is the user's call and has to be put to them.
            raise PathNotAllowed(
                f"{path} already exists. Ask the user whether to replace it — "
                f"if they say yes, call again with force=true; otherwise write "
                f"to a different name."
            )
        if not path.is_file():
            # A directory, socket or device would not be replaced by a write.
            raise PathNotAllowed(f"Refusing to replace a non-file: {path}")
        path.unlink()

    return path
