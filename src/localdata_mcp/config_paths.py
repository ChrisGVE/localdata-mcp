"""OS-aware configuration path resolution for LocalData MCP.

Discovers configuration file locations following platform conventions:
XDG_CONFIG_HOME on Linux, ~/Library/Application Support on macOS,
%APPDATA% on Windows. Provides backward compatibility with legacy
~/.localdata.yaml via deprecation warnings.
"""

import logging
import os
import platform
import shutil
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


DEFAULT_CONFIG_TEMPLATE = """\
# LocalData MCP Configuration
# See https://localdata-mcp.readthedocs.io/en/latest/configuration.html

# databases:
#   my_postgres:
#     type: postgresql
#     connection_string: postgresql://user:pass@host:5432/db

# staging:
#   max_concurrent: 10
#   max_size_mb: 2048
#   max_total_mb: 10240
#   timeout_minutes: 30
#   eviction_policy: lru

# memory:
#   max_budget_mb: 512
#   budget_percent: 10
#   low_memory_threshold_gb: 1.0

# query:
#   default_chunk_size: 100
#   buffer_timeout_seconds: 600
#   blob_handling: exclude
#   blob_max_size_mb: 5
#   preflight_default: false

# connections:
#   max_concurrent: 10
#   timeout_seconds: 30

# security:
#   allowed_paths: ["."]
#   restrict_paths: true
#   max_query_length: 10000
#   blocked_keywords: []
#   readonly: false

# logging:
#   level: info
#   console_output: true

# performance:
#   memory_limit_mb: 2048
#   chunk_size: 100
"""


def create_default_config(path: Optional[Path] = None) -> Path:
    """Create a default configuration file with all sections commented out.

    Args:
        path: Where to write the file. Defaults to ``get_recommended_path()``.

    Returns:
        The path of the created file.

    Raises:
        FileExistsError: If a file already exists at *path*.
    """
    if path is None:
        path = get_recommended_path()
    if path.exists():
        raise FileExistsError(f"Config file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    logger.info("Created default config at %s", path)
    return path


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


def migrate_config(
    source: Optional[Path] = None,
    dest: Optional[Path] = None,
    backup: bool = True,
) -> bool:
    """Migrate a configuration file from a legacy location to the recommended path.

    Args:
        source: Path to the existing config file.  Defaults to
            ``~/.localdata.yaml``.
        dest: Where to place the migrated file.  Defaults to
            :func:`get_recommended_path`.
        backup: When *True* (the default), copy *source* to
            ``<source>.yaml.bak`` before migrating.

    Returns:
        ``True`` on successful migration.

    Raises:
        FileNotFoundError: If *source* does not exist.
        FileExistsError: If *dest* already exists.
    """
    if source is None:
        source = Path("~/.localdata.yaml").expanduser()
    if dest is None:
        dest = get_recommended_path()

    if not source.exists():
        raise FileNotFoundError(f"Source config not found: {source}")
    if dest.exists():
        raise FileExistsError(f"Destination config already exists: {dest}")

    if backup:
        shutil.copy2(source, source.with_suffix(".yaml.bak"))

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)

    logger.info("Migrated config from %s to %s", source, dest)
    return True
