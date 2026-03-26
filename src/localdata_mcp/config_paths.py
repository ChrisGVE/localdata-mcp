"""OS-aware configuration path resolution for LocalData MCP.

Discovers configuration file locations following platform conventions:
XDG_CONFIG_HOME on Linux, ~/Library/Application Support on macOS,
%APPDATA% on Windows. Provides backward compatibility with legacy
~/.localdata.yaml via deprecation warnings.
"""

import logging
import os
import platform
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ConfigLocationType(str, Enum):
    """Where a config file was found."""

    EXPLICIT = "explicit"
    PROJECT_LOCAL = "project_local"
    USER_XDG = "user_xdg"
    USER_MACOS = "user_macos"
    USER_WINDOWS = "user_windows"
    SYSTEM = "system"
    LEGACY = "legacy"


@dataclass
class ConfigPathInfo:
    """Metadata about a discovered config file path."""

    path: Path
    priority: int
    location_type: ConfigLocationType
    is_legacy: bool = False


def _user_config_paths(system: str) -> List[ConfigPathInfo]:
    """Return user-level config paths for the current OS."""
    paths: List[ConfigPathInfo] = []
    if system in ("Linux", "Darwin"):
        xdg = os.environ.get("XDG_CONFIG_HOME", "~/.config")
        paths.append(
            ConfigPathInfo(
                path=Path(xdg).expanduser() / "localdata" / "config.yaml",
                priority=2,
                location_type=ConfigLocationType.USER_XDG,
            )
        )
    if system == "Darwin":
        paths.append(
            ConfigPathInfo(
                path=Path(
                    "~/Library/Application Support/localdata/config.yaml"
                ).expanduser(),
                priority=3,
                location_type=ConfigLocationType.USER_MACOS,
            )
        )
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            paths.append(
                ConfigPathInfo(
                    path=Path(appdata) / "localdata" / "config.yaml",
                    priority=2,
                    location_type=ConfigLocationType.USER_WINDOWS,
                )
            )
    return paths


def get_config_paths() -> List[ConfigPathInfo]:
    """Return config file paths in priority order (highest first).

    Priority:
      0. LOCALDATA_CONFIG environment variable
      1. Project-local ./.localdata.yaml
      2. User config (XDG / AppData)
      3. macOS ~/Library/Application Support fallback
      4. System /etc/localdata/config.yaml
      5. Legacy ~/.localdata.yaml (emits deprecation warning if used)
    """
    paths: List[ConfigPathInfo] = []

    explicit = os.environ.get("LOCALDATA_CONFIG")
    if explicit:
        paths.append(
            ConfigPathInfo(
                path=Path(explicit),
                priority=0,
                location_type=ConfigLocationType.EXPLICIT,
            )
        )

    paths.append(
        ConfigPathInfo(
            path=Path("./.localdata.yaml"),
            priority=1,
            location_type=ConfigLocationType.PROJECT_LOCAL,
        )
    )

    system = platform.system()
    paths.extend(_user_config_paths(system))

    if system in ("Linux", "Darwin"):
        paths.append(
            ConfigPathInfo(
                path=Path("/etc/localdata/config.yaml"),
                priority=4,
                location_type=ConfigLocationType.SYSTEM,
            )
        )

    paths.append(
        ConfigPathInfo(
            path=Path("~/.localdata.yaml").expanduser(),
            priority=5,
            location_type=ConfigLocationType.LEGACY,
            is_legacy=True,
        )
    )

    return paths


def get_recommended_path() -> Path:
    """Return the preferred user config path for the current OS."""
    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "localdata" / "config.yaml"
    xdg = os.environ.get("XDG_CONFIG_HOME", "~/.config")
    return Path(xdg).expanduser() / "localdata" / "config.yaml"


def emit_deprecation_warning(legacy_path: Path) -> None:
    """Warn when a legacy config location is in use."""
    recommended = get_recommended_path()
    msg = f"Config at '{legacy_path}' is deprecated. Migrate to '{recommended}'."
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    logger.warning(
        "Legacy config location: %s — recommended: %s",
        legacy_path,
        recommended,
    )
