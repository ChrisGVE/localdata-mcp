"""Where configuration comes from, and what it is allowed to say.

Discovery is a **cascade, first found wins** — not a merge. The order is:

1. ``LOCALDATA_CONFIG_PATH`` — an explicit pointer. An environment variable is
   used to *locate* the file, never to carry a setting.
2. ``$XDG_CONFIG_HOME/localdata/config.toml`` (``$XDG_CONFIG_HOME`` defaults to
   ``~/.config``) — the primary convention on every platform, macOS included.
3. ``./localdata.toml`` — the next-most-specific statement of intent.
4. ``~/Library/Application Support/localdata/config.toml`` — macOS only.
5. ``%APPDATA%\\localdata\\config.toml`` — Windows only.

The last two are mutually exclusive by platform, so in practice any given
machine walks four candidates. Defaults live in :class:`Config` rather than in a
shipped file, so a user's configuration only ever states its overrides.

**Unknown keys are refused rather than ignored.** The settings here decide what
the server may reach on the filesystem and the network, and a mistyped
``path_limitted = false`` that silently kept the safe default would be a switch
the user believes they have thrown. Failing to start is the louder, better
outcome — the message reaches the client's server log.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = [
    "Config",
    "ConfigError",
    "MAX_SLOTS",
    "active",
    "config_search_path",
    "load",
    "reset",
    "use",
]

#: Directory name used under every configuration root.
APP_NAME = "localdata"

#: Names the configuration file. Never carries a setting itself.
PATH_ENV_VAR = "LOCALDATA_CONFIG_PATH"

#: The most slots there can be, because SQLite refuses the eleventh ``ATTACH``
#: on a connection: ``sqlite3.OperationalError: too many attached databases -
#: max 10``. Measured, not read from a document. Since every slot is an attached
#: database — a file-born slot attaches ``:memory:`` just as a database file
#: attaches itself — this ceiling is the slot ceiling. ``main`` is not counted
#: against it, so it stays free.
MAX_SLOTS = 10


class ConfigError(RuntimeError):
    """A configuration that will not be run under."""


@dataclass(frozen=True)
class Config:
    """The whole of what is configurable, with its defaults."""

    #: How many datasource slots may be held at once.
    slots: int = MAX_SLOTS
    #: Directories the server may reach, in addition to the working directory
    #: and everything below it, which is always in scope.
    roots: tuple[Path, ...] = ()
    #: When true, paths must be contained in the working directory or a root.
    #: Defaults closed: every path this server receives arrives from an LLM.
    path_limited: bool = True
    #: When true, a datasource URL naming a network service may be opened.
    network_enabled: bool = False
    #: The file this came from, or ``None`` when nothing was found.
    source: Path | None = field(default=None, compare=True)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def config_search_path() -> list[Path]:
    """The candidates to consult, in order. First one that exists is used."""
    explicit = os.environ.get(PATH_ENV_VAR)
    if explicit:
        return [Path(explicit).expanduser()]

    xdg_home = os.environ.get("XDG_CONFIG_HOME")
    xdg_root = Path(xdg_home).expanduser() if xdg_home else Path.home() / ".config"

    candidates = [
        xdg_root / APP_NAME / "config.toml",
        Path("localdata.toml"),
    ]

    if sys.platform == "darwin":
        candidates.append(
            Path.home() / "Library" / "Application Support" / APP_NAME / "config.toml"
        )
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / APP_NAME / "config.toml")

    return candidates


def load() -> Config:
    """Walk the cascade and build the configuration."""
    explicit = os.environ.get(PATH_ENV_VAR)
    candidates = config_search_path()

    for candidate in candidates:
        if candidate.is_file():
            return _parse(candidate)

    if explicit:
        raise ConfigError(
            f"{PATH_ENV_VAR} names {explicit}, which is not a file. Point it at a "
            f"configuration file or unset it to use the search path."
        )
    return Config()


# ---------------------------------------------------------------------------
# Parsing, and refusing
# ---------------------------------------------------------------------------

#: Every section, and every key each may hold. The gate against typos.
_SCHEMA: dict[str, set[str]] = {
    "workspace": {"slots"},
    "paths": {"roots", "path_limited"},
    "network": {"enabled"},
}


def _parse(path: Path) -> Config:
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc

    _reject_unknown(raw, path)

    workspace = raw.get("workspace", {})
    paths = raw.get("paths", {})
    network = raw.get("network", {})

    return Config(
        slots=_slots(workspace.get("slots", MAX_SLOTS), path),
        roots=_roots(paths.get("roots", []), path),
        path_limited=_flag(paths.get("path_limited", True), "paths.path_limited", path),
        network_enabled=_flag(network.get("enabled", False), "network.enabled", path),
        source=path.resolve(),
    )


def _reject_unknown(raw: dict[str, Any], path: Path) -> None:
    for section, contents in raw.items():
        if section not in _SCHEMA:
            known = ", ".join(sorted(_SCHEMA))
            raise ConfigError(
                f"{path}: unknown section [{section}]. Known sections: {known}."
            )
        if not isinstance(contents, dict):
            raise ConfigError(f"{path}: [{section}] must be a table of settings.")
        for key in contents:
            if key not in _SCHEMA[section]:
                known = ", ".join(sorted(_SCHEMA[section]))
                raise ConfigError(
                    f"{path}: unknown setting {key!r} in [{section}]. "
                    f"Known settings: {known}."
                )


def _slots(value: Any, path: Path) -> int:
    # bool is an int in Python, and `slots = true` meaning one slot would be a
    # silent misreading of an obvious mistake.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{path}: workspace.slots must be a whole number, got {value!r}."
        )
    if not 1 <= value <= MAX_SLOTS:
        raise ConfigError(
            f"{path}: workspace.slots must be between 1 and {MAX_SLOTS}, got {value}. "
            f"SQLite refuses the {MAX_SLOTS + 1}th attached database, and every slot "
            f"is an attached database."
        )
    return value


def _roots(value: Any, path: Path) -> tuple[Path, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(
            f"{path}: paths.roots must be a list of directory strings, got {value!r}."
        )
    return tuple(Path(item).expanduser().resolve() for item in value)


def _flag(value: Any, name: str, path: Path) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: {name} must be true or false, got {value!r}.")
    return value


# ---------------------------------------------------------------------------
# The process-wide configuration
# ---------------------------------------------------------------------------

_active: Config | None = None


def active() -> Config:
    """The configuration this process is running under, loaded on first use."""
    global _active
    if _active is None:
        _active = load()
    return _active


def use(config: Config) -> None:
    """Install a configuration directly. For tests and for startup override."""
    global _active
    _active = config


def reset() -> None:
    """Forget the loaded configuration so the next call re-reads the cascade."""
    global _active
    _active = None
