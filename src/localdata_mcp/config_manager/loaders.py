"""YAML and validation helpers for LocalData MCP configuration."""

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml
from pydantic import ValidationError as PydanticValidationError

from .models import LocalDataConfig


def substitute_env_vars(content: str) -> str:
    """Substitute environment variables in YAML content using ${VAR} syntax."""
    pattern = re.compile(r"\$\{([^}]+)\}")

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        # Support default values: ${VAR:default_value}
        if ":" in var_name:
            var_name, default = var_name.split(":", 1)
            return os.getenv(var_name, default)
        else:
            return os.getenv(var_name, match.group(0))  # Return original if not found

    return pattern.sub(replace_var, content)


def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Deep merge source dictionary into target dictionary."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_merge(target[key], value)
        else:
            target[key] = value


def load_yaml_config(
    config_file: Optional[str],
    file_mtimes: Dict[str, float],
    merge_callback: Callable[[Dict[str, Any]], None],
) -> Optional[Dict[str, Any]]:
    """Load configuration from YAML files with OS-aware discovery.

    Args:
        config_file: Explicit config file path, or None for auto-discovery.
        file_mtimes: Mutable dict tracking file modification times.
        merge_callback: Callback to merge global config before returning project-local.
    """
    from ..config_paths import emit_deprecation_warning, get_config_paths

    if config_file:
        return _load_explicit_config(config_file, file_mtimes)

    infos = get_config_paths()

    project_local = None
    global_config = None

    for info in infos:
        expanded = info.path.expanduser()
        if not expanded.exists():
            continue
        try:
            with open(expanded, "r") as f:
                content = substitute_env_vars(f.read())
            data = yaml.safe_load(content)
            file_mtimes[str(expanded)] = os.path.getmtime(expanded)
            if info.is_legacy:
                emit_deprecation_warning(expanded)
            if info.location_type.value == "project_local":
                project_local = data
            elif global_config is None:
                global_config = data
        except Exception as e:
            print(f"Warning: Could not load config file {info.path}: {e}")
            continue

    if global_config and project_local:
        merge_callback(global_config)
        return project_local
    return project_local or global_config


def _load_explicit_config(
    config_file: str, file_mtimes: Dict[str, float]
) -> Optional[Dict[str, Any]]:
    """Load a single explicit config file."""
    expanded = Path(config_file).expanduser()
    if not expanded.exists():
        return None
    try:
        with open(expanded, "r") as f:
            content = substitute_env_vars(f.read())
        file_mtimes[str(expanded)] = os.path.getmtime(expanded)
        return yaml.safe_load(content)
    except Exception as e:
        print(f"Warning: Could not load config file {expanded}: {e}")
        return None


def validate_config(config_data: Dict[str, Any]) -> None:
    """Validate the final merged configuration."""
    try:
        LocalDataConfig(**config_data)
    except PydanticValidationError as e:
        print(f"Configuration validation errors: {e}")

    from ..config_schemas import (
        ConnectionsConfig,
        DiskBudgetConfig,
        MemoryConfig,
        QueryConfig,
        SecurityConfig,
        StagingConfig,
    )

    for section, cls in [
        ("staging", StagingConfig),
        ("memory", MemoryConfig),
        ("query", QueryConfig),
        ("connections", ConnectionsConfig),
        ("security", SecurityConfig),
        ("disk_budget", DiskBudgetConfig),
    ]:
        data = config_data.get(section, {})
        if data:
            try:
                cls(**data)
            except (ValueError, TypeError) as e:
                print(f"Configuration validation error in '{section}': {e}")
