"""YAML and validation helpers for LocalData MCP configuration."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class ConfigLayer:
    """One configuration file that was found and parsed.

    Keeping the layers separate rather than merging them on sight is what lets
    the security resolution tell an operator floor from a project-local
    request, and name the offending file when the two disagree.
    """

    path: str
    location_type: str
    data: Dict[str, Any]

    @property
    def is_project_local(self) -> bool:
        """Was this file found in the working directory, beside the data?"""
        return self.location_type == "project_local"


def load_config_layers(
    config_file: Optional[str],
    file_mtimes: Dict[str, float],
) -> List[ConfigLayer]:
    """Load every discoverable config file, highest priority first.

    An explicit ``config_file`` (or ``LOCALDATA_CONFIG``) replaces discovery
    entirely: the operator named the file to use, so no other file is consulted.

    Args:
        config_file: Explicit config file path, or None for auto-discovery.
        file_mtimes: Mutable dict tracking file modification times.
    """
    from ..config_paths import emit_deprecation_warning, get_config_paths

    if config_file:
        data = _load_explicit_config(config_file, file_mtimes)
        if data is None:
            return []
        return [
            ConfigLayer(
                path=str(Path(config_file).expanduser()),
                location_type="explicit",
                data=data,
            )
        ]

    layers: List[ConfigLayer] = []
    for info in get_config_paths():
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
        except Exception as e:
            print(f"Warning: Could not load config file {info.path}: {e}")
            continue
        if data:
            layers.append(
                ConfigLayer(
                    path=str(expanded),
                    location_type=info.location_type.value,
                    data=data,
                )
            )

    return layers


def select_effective_layers(
    layers: List[ConfigLayer],
) -> Tuple[Optional[ConfigLayer], Optional[ConfigLayer]]:
    """Pick the layers whose ordinary settings apply: one operator, one local.

    Settings come from the highest-priority operator layer and, on top of it,
    the project-local file. Lower-priority operator files are not consulted for
    settings -- longstanding behaviour, preserved here deliberately.

    Note that security *floors* are handled differently: they are collected from
    every operator layer, so a floor set in ``/etc`` cannot be dropped merely by
    the existence of a higher-priority user config.

    Returns:
        ``(operator_layer, project_local_layer)``, either of which may be None.
    """
    operator = next((l for l in layers if not l.is_project_local), None)
    project_local = next((l for l in layers if l.is_project_local), None)
    return operator, project_local


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
