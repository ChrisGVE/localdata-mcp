"""What the path trust boundary must do once it has more than one root.

Two properties carry most of the weight. **The working directory and everything
below it is always in scope**, so the common case needs no configuration at all.
And containment is checked *after* resolution, so a traversal or a symlink is
judged by where it really lands rather than by how it is spelled.

``path_limited = false`` switches containment off entirely. The tests below pin
that it does exactly that and no more — in particular that it does not also
disarm the refusal to overwrite an existing file, which is a separate guard
against a different accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localdata_mcp import config as config_module
from localdata_mcp.config import Config
from localdata_mcp.paths import (
    PathNotAllowed,
    allowed_paths,
    resolve_read_path,
    resolve_write_path,
)


@pytest.fixture(autouse=True)
def clean_configuration(monkeypatch, tmp_path):
    """Run in a scratch working directory under a known configuration."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    config_module.use(Config())
    yield
    config_module.reset()


def make(path: Path, text: str = "a\n1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# The working directory is always in scope
# ---------------------------------------------------------------------------


def test_the_working_directory_needs_no_configuration():
    target = make(Path("data.csv"))
    assert resolve_read_path("data.csv") == target.resolve()


def test_subdirectories_of_the_working_directory_are_in_scope():
    target = make(Path("nested/deeper/data.csv"))
    assert resolve_read_path(str(target)) == target.resolve()


def test_the_working_directory_is_listed_first_among_allowed_paths(tmp_path):
    config_module.use(Config(roots=(tmp_path,)))
    assert allowed_paths()[0] == Path.cwd().resolve()


# ---------------------------------------------------------------------------
# Configured roots
# ---------------------------------------------------------------------------


def test_a_configured_root_is_in_scope(tmp_path):
    elsewhere = make(tmp_path / "elsewhere" / "data.csv")
    config_module.use(Config(roots=(tmp_path / "elsewhere",)))
    assert resolve_read_path(str(elsewhere)) == elsewhere.resolve()


def test_every_configured_root_is_in_scope(tmp_path):
    first = make(tmp_path / "one" / "a.csv")
    second = make(tmp_path / "two" / "b.csv")
    config_module.use(Config(roots=(tmp_path / "one", tmp_path / "two")))
    assert resolve_read_path(str(first)) == first.resolve()
    assert resolve_read_path(str(second)) == second.resolve()


def test_a_path_outside_every_root_is_refused(tmp_path):
    outside = make(tmp_path / "outside" / "secret.csv")
    config_module.use(Config(roots=(tmp_path / "allowed",)))
    with pytest.raises(PathNotAllowed):
        resolve_read_path(str(outside))


def test_the_refusal_names_every_allowed_path_and_the_way_to_widen(tmp_path):
    """A refusal that does not say where it would have looked is unactionable."""
    outside = make(tmp_path / "outside" / "secret.csv")
    root = tmp_path / "allowed"
    root.mkdir()
    config_module.use(Config(roots=(root,)))

    with pytest.raises(PathNotAllowed) as raised:
        resolve_read_path(str(outside))

    message = str(raised.value)
    assert str(root) in message
    assert str(Path.cwd().resolve()) in message
    assert "path_limited" in message or "roots" in message


# ---------------------------------------------------------------------------
# Resolve, then check
# ---------------------------------------------------------------------------


def test_traversal_out_of_scope_is_refused(tmp_path):
    make(tmp_path / "secret.csv")
    with pytest.raises(PathNotAllowed):
        resolve_read_path("../secret.csv")


def test_a_symlink_pointing_out_of_scope_is_refused(tmp_path):
    outside = make(tmp_path / "secret.csv")
    link = Path("innocent.csv")
    link.symlink_to(outside)
    with pytest.raises(PathNotAllowed):
        resolve_read_path(str(link))


def test_a_symlink_inside_scope_is_allowed():
    real = make(Path("real.csv"))
    link = Path("link.csv")
    link.symlink_to(real)
    assert resolve_read_path("link.csv") == real.resolve()


def test_a_missing_file_is_refused():
    with pytest.raises(PathNotAllowed, match="No such file"):
        resolve_read_path("absent.csv")


def test_a_directory_is_not_a_file():
    Path("adirectory").mkdir()
    with pytest.raises(PathNotAllowed, match="Not a regular file"):
        resolve_read_path("adirectory")


# ---------------------------------------------------------------------------
# path_limited = false
# ---------------------------------------------------------------------------


def test_unlimited_reaches_outside_every_root(tmp_path):
    outside = make(tmp_path / "far" / "away.csv")
    config_module.use(Config(path_limited=False))
    assert resolve_read_path(str(outside)) == outside.resolve()


def test_unlimited_still_refuses_a_file_that_is_not_there(tmp_path):
    config_module.use(Config(path_limited=False))
    with pytest.raises(PathNotAllowed, match="No such file"):
        resolve_read_path(str(tmp_path / "absent.csv"))


def test_unlimited_does_not_disarm_the_overwrite_guard():
    """Containment and not-clobbering-a-file are separate guards."""
    existing = make(Path("report.csv"))
    config_module.use(Config(path_limited=False))
    with pytest.raises(PathNotAllowed, match="already exists"):
        resolve_write_path(str(existing))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_a_write_inside_scope_is_allowed():
    assert resolve_write_path("out.csv") == (Path.cwd() / "out.csv").resolve()


def test_a_write_outside_scope_is_refused(tmp_path):
    with pytest.raises(PathNotAllowed):
        resolve_write_path(str(tmp_path / "elsewhere" / "out.csv"))


def test_a_write_into_a_configured_root_is_allowed(tmp_path):
    root = tmp_path / "outbox"
    root.mkdir()
    config_module.use(Config(roots=(root,)))
    assert resolve_write_path(str(root / "out.csv")) == (root / "out.csv").resolve()


def test_an_existing_file_is_refused_until_forced(tmp_path):
    """The refusal asks a question; ``force`` carries back the user's answer.

    A destination is a name the user chose and an agent relayed, so replacing
    what sits there is the user's call. The message has to read as a decision to
    put to them rather than a retry to make — otherwise an agent simply retries.
    """
    existing = make(Path("report.csv"))

    with pytest.raises(PathNotAllowed) as refusal:
        resolve_write_path("report.csv")
    assert "Ask the user" in str(refusal.value)

    assert resolve_write_path("report.csv", force=True) == existing.resolve()
    assert not existing.exists(), "force clears the way, so callers may just write"


def test_a_file_a_slot_is_sitting_on_is_refused_even_with_force(tmp_path):
    """``force`` is authority over the user's spare files, not over open state.

    Deleting a file an attached datasource is reading would succeed on POSIX
    while that slot went on answering from an unnamed inode — a divergence
    nothing would report.
    """
    held = make(Path("sales.csv"))
    claimed = {held.resolve(): "shop"}

    with pytest.raises(PathNotAllowed) as refusal:
        resolve_write_path("sales.csv", force=True, claimed=claimed)

    assert "'shop'" in str(refusal.value)
    assert held.exists()


def test_a_write_through_a_symlink_leaving_scope_is_refused(tmp_path):
    """The parent directory is a link out; resolution catches it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    Path("escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathNotAllowed):
        resolve_write_path("escape/out.csv")
