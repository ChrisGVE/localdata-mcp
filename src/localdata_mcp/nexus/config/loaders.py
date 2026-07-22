"""localdata_mcp/nexus/config/loaders.py — layered config loading.

The one config-file search-path list (ARCHITECTURE.md section 5: one
list, one home) and the assembly of LayerSources from TOML files plus
the derived env overrides (user rank, after the user file). TOML is the
chosen format: pyproject-adjacent, stdlib tomllib (tomli backport below
3.11), no code execution on parse. Neighbors: env_derive.py supplies
the env layer; merge.py performs the two-tier merge this module feeds.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 — tomllib landed in 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from .env_derive import env_overrides
from .errors import ConfigurationError
from .merge import ConfigLoadResult, merge_sources
from .provenance import Layer, LayerSource


class SearchPath(NamedTuple):
    """One entry of the search-path list: source name, trust, location."""

    name: str
    layer: Layer
    path: str


# THE search-path list — the single declaration of where config lives,
# carrying main's /etc + home + cwd convention over to TOML.
CONFIG_SEARCH_PATHS: tuple[SearchPath, ...] = (
    SearchPath("system-file", Layer.SYSTEM, "/etc/localdata.toml"),
    SearchPath("user-file", Layer.USER, "~/.localdata.toml"),
    SearchPath("project-file", Layer.PROJECT, "./.localdata.toml"),
)

# Env applies within the user layer AFTER the user file, so it wins
# ordinary last-wins fields there while the file wins pinned ones.
_ENV_ORDER_WITHIN_USER_LAYER = 1

ReadText = Callable[[Path], "str | None"]


def _read_optional(path: Path) -> str | None:
    """Default reader: the file's text, or None when it does not exist."""
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return None


def load_config(
    *,
    environ: Mapping[str, str] | None = None,
    search_paths: Sequence[SearchPath] = CONFIG_SEARCH_PATHS,
    read_text: ReadText = _read_optional,
) -> ConfigLoadResult:
    """Load and merge the layered configuration (section 4e's step 2).

    Missing files simply contribute nothing; a present-but-malformed
    file is a fatal typed ConfigurationError.
    """
    environ = os.environ if environ is None else environ
    sources = [
        LayerSource(spec.name, spec.layer, 0, values)
        for spec, values in _read_layers(search_paths, read_text)
    ]
    overrides = env_overrides(environ)
    if overrides:
        sources.append(
            LayerSource("env", Layer.USER, _ENV_ORDER_WITHIN_USER_LAYER, overrides)
        )
    return merge_sources(sources)


def _read_layers(
    search_paths: Sequence[SearchPath], read_text: ReadText
) -> list[tuple[SearchPath, Mapping[str, Any]]]:
    """Parse every present file on the search-path list."""
    layers: list[tuple[SearchPath, Mapping[str, Any]]] = []
    for spec in search_paths:
        text = read_text(Path(spec.path).expanduser())
        if text is not None:
            layers.append((spec, _parse_toml(text, spec.name)))
    return layers


def _parse_toml(text: str, source_name: str) -> Mapping[str, Any]:
    """Parse one layer's TOML text; malformed input is a typed error."""
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(
            f"{source_name} is not valid TOML: {error}",
            source=source_name,
        ) from error
