"""localdata_mcp/nexus/config/env_derive.py — derived env-var overrides.

Derives every override name mechanically from the field path declared in
models.py (`LOCALDATA_<SECTION>_<FIELD>` — ARCHITECTURE.md section 5:
"not independently declared") and parses env values to the declared
field types. loaders.py feeds the result in as the env source, ranked at
the user layer. Unknown LOCALDATA_* names are refused fail-closed: a
typo'd bound silently ignored is a limit the operator wrongly believes
is set. Endpoint declarations have no env encoding — they are entities,
not scalar fields, and enter through operator-layer files only.
"""

from __future__ import annotations

import os
from dataclasses import Field
from typing import Any, Mapping

from .errors import TypeMismatchError, UnknownFieldError
from .models import field_type, iter_config_fields

ENV_PREFIX = "LOCALDATA"


def env_var_name(section_name: str, field_name: str) -> str:
    """The one derivation: LOCALDATA_<SECTION>_<FIELD>, upper-cased."""
    return f"{ENV_PREFIX}_{section_name.upper()}_{field_name.upper()}"


def env_field_map() -> dict[str, tuple[str, str]]:
    """Derived env name -> (section, field) for every model field."""
    return {
        env_var_name(section, fld.name): (section, fld.name)
        for section, fld in iter_config_fields()
    }


def env_overrides(environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Read LOCALDATA_* variables into a nested {section: {field: value}}
    layer dict, typed per the field declarations."""
    mapping = env_field_map()
    overrides: dict[str, dict[str, Any]] = {}
    for name, raw in environ.items():
        if not name.startswith(f"{ENV_PREFIX}_"):
            continue
        if name not in mapping:
            raise UnknownFieldError(
                f"{name} matches no ConfigModel field "
                f"(names derive as {ENV_PREFIX}_<SECTION>_<FIELD>)",
                source="env",
                attempted_value=raw,
            )
        section, field_name = mapping[name]
        value = _parse_env_value(raw, section, field_name)
        overrides.setdefault(section, {})[field_name] = value
    return overrides


def _parse_env_value(raw: str, section: str, field_name: str) -> Any:
    """Parse one env string to the field's declared type."""
    declared = field_type(section, field_name)
    try:
        if declared is int:
            return int(raw)
        if declared is float:
            return float(raw)
        if declared is str:
            return raw
        # The only collection type in the model: tuple[str, ...] path
        # lists, encoded like PATH (os.pathsep-separated).
        return tuple(part for part in raw.split(os.pathsep) if part)
    except ValueError as error:
        raise TypeMismatchError(
            f"cannot read {raw!r} as {declared} for {section}.{field_name}",
            field_path=f"{section}.{field_name}",
            source="env",
            attempted_value=raw,
        ) from error


def field_by_name(section: str, field_name: str) -> Field[Any]:
    """Convenience lookup used by merge.py: the Field declaration."""
    for declared_section, fld in iter_config_fields():
        if declared_section == section and fld.name == field_name:
            return fld
    raise KeyError(f"{section}.{field_name}")
